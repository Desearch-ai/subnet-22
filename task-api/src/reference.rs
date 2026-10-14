//! Reference vectors for validators without a model server of their own: the texts a validator sampled from an embed upload's input, embedded once on Desearch's model server and kept for the other validators.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use bytes::Bytes;
use desearch::r2;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use redis::AsyncCommands;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::http::ApiError;
use crate::state::State;

/// More than a validator's sample needs; a validator asks at most this many texts of one upload.
pub const MAX_TEXTS: usize = 64;
/// The model reads this many tokens of a text, as miners' servers are told to cut them.
const MAX_TOKENS: u32 = 2048;
const LOCK_S: i64 = 120;
const TIMEOUT: Duration = Duration::from_secs(120);

/// Desearch's model server, behind an OpenAI-compatible `/v1/embeddings` with a key.
pub struct Reference {
    http: reqwest::Client,
    url: String,
    key: String,
    calls: Semaphore,
}

impl Reference {
    pub fn new(url: &str, key: &str, concurrency: usize) -> Result<Reference> {
        let http = reqwest::Client::builder().timeout(TIMEOUT).build()?;
        Ok(Reference { http, url: url.to_string(), key: key.to_string(), calls: Semaphore::new(concurrency.max(1)) })
    }

    /// Each asked text's vector, as base64 little-endian float32, from the kept ones or the model server.
    pub async fn vectors(&self, state: &State, task_id: &str, job: &Value, validator: &str, text_ids: &[String]) -> Result<Vec<String>, ApiError> {
        let mut redis = state.redis.clone();
        let asked = format!("eref:asked:{task_id}:{validator}");
        let before: HashSet<String> = redis.smembers(&asked).await?;
        if before.union(&text_ids.iter().cloned().collect()).count() > MAX_TEXTS {
            return Err(ApiError::status(429, format!("a validator may ask for at most {MAX_TEXTS} texts of one upload")));
        }
        let keep_s = (state.settings.validation_ttl as i64) * 2;
        let _: i64 = redis.sadd(&asked, text_ids).await?;
        let _: bool = redis.expire(&asked, keep_s).await?;

        let kept_key = format!("eref:{task_id}");
        let mut kept: Vec<Option<String>> = redis::cmd("HMGET").arg(&kept_key).arg(text_ids).query_async(&mut redis).await?;
        if kept.iter().all(Option::is_some) {
            return Ok(kept.into_iter().flatten().collect());
        }
        let lock = format!("eref:lock:{task_id}");
        if !state.set_once(&lock, validator, LOCK_S).await? {
            return Err(ApiError::retry(503, "the reference for this upload is being computed", "2"));
        }
        let computed = self.compute(state, job, text_ids, &kept).await;
        state.delete(&lock).await?;
        let computed = computed.map_err(|failure| match failure {
            Failure::Unknown(text_id) => ApiError::status(422, format!("text {text_id} is not in this upload's input")),
            Failure::Ours(error) => {
                eprintln!("task={task_id} reference vectors failed: {error:#}");
                ApiError::retry(503, "the reference model server is unavailable", "30")
            }
        })?;
        let fresh: Vec<(String, String)> = computed.into_iter().collect();
        if !fresh.is_empty() {
            let _: () = redis.hset_multiple(&kept_key, &fresh).await?;
            let _: bool = redis.expire(&kept_key, keep_s).await?;
        }
        let fresh: HashMap<String, String> = fresh.into_iter().collect();
        for (slot, text_id) in kept.iter_mut().zip(text_ids) {
            if slot.is_none() {
                *slot = fresh.get(text_id).cloned();
            }
        }
        kept.into_iter().map(|vector| vector.ok_or_else(|| ApiError::status(500, "a reference vector went missing"))).collect()
    }

    /// The texts not kept yet, read from the upload's input and embedded in one request.
    async fn compute(&self, state: &State, job: &Value, text_ids: &[String], kept: &[Option<String>]) -> Result<Vec<(String, String)>, Failure> {
        let missing: Vec<&String> = text_ids.iter().zip(kept).filter(|(_, kept)| kept.is_none()).map(|(id, _)| id).collect();
        let input_key = job["input_key"].as_str().ok_or_else(|| anyhow!("the upload has no input"))?;
        let input = match state.storage.get(input_key).await {
            Ok(body) => body,
            Err(r2::Error::Missing) => return Err(anyhow!("its input {input_key} is gone").into()),
            Err(error) => return Err(anyhow!("reading {input_key}: {error}").into()),
        };
        if job["input_sha256"].as_str() != Some(hex(&Sha256::digest(&input)).as_str()) {
            return Err(anyhow!("its input does not match the hash it was issued with").into());
        }
        let wanted: HashSet<String> = missing.iter().map(|id| id.to_string()).collect();
        let texts = tokio::task::spawn_blocking(move || input_texts(input, &wanted)).await.map_err(anyhow::Error::from)??;
        let mut inputs = Vec::with_capacity(missing.len());
        for id in &missing {
            inputs.push(texts.get(id.as_str()).ok_or_else(|| Failure::Unknown(id.to_string()))?.as_str());
        }
        let _permit = self.calls.acquire().await.map_err(anyhow::Error::from)?;
        let body = json!({"model": job["model"], "input": inputs, "encoding_format": "base64", "truncate_prompt_tokens": MAX_TOKENS});
        let sent = self
            .http
            .post(&self.url)
            .bearer_auth(&self.key)
            .json(&body)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(anyhow::Error::from)?;
        let answer: Value = sent.json().await.context("the model server's answer")?;
        let mut rows = answer["data"].as_array().cloned().ok_or_else(|| anyhow!("the model server answered without data"))?;
        rows.sort_by_key(|row| row["index"].as_u64().unwrap_or(u64::MAX));
        if rows.len() != missing.len() {
            return Err(anyhow!("the model server answered {} vectors for {} texts", rows.len(), missing.len()).into());
        }
        let mut vectors = Vec::with_capacity(rows.len());
        for (row, id) in rows.iter().zip(missing) {
            let vector = row["embedding"].as_str().ok_or_else(|| anyhow!("a vector that is not base64"))?;
            vectors.push((id.clone(), vector.to_string()));
        }
        Ok(vectors)
    }
}

/// Why reference vectors could not be made: a text the input does not hold, or a problem on our side.
enum Failure {
    Unknown(String),
    Ours(anyhow::Error),
}

impl From<anyhow::Error> for Failure {
    fn from(error: anyhow::Error) -> Self {
        Failure::Ours(error)
    }
}

/// The text of each wanted text id in an embed input file.
fn input_texts(file: Bytes, wanted: &HashSet<String>) -> Result<HashMap<String, String>> {
    let mut found = HashMap::new();
    for batch in ParquetRecordBatchReaderBuilder::try_new(file)?.build()? {
        let batch = batch?;
        let (ids, texts) = (column(batch.column_by_name("text_id"))?, column(batch.column_by_name("text"))?);
        for row in 0..batch.num_rows() {
            if let (Some(id), Some(text)) = (ids(row), texts(row)) {
                if wanted.contains(id) {
                    found.insert(id.to_string(), text.to_string());
                }
            }
        }
    }
    Ok(found)
}

/// A string column read by row, whichever string type its writer chose.
fn column<'a>(array: Option<&'a ArrayRef>) -> Result<Box<dyn Fn(usize) -> Option<&'a str> + 'a>> {
    let array = array.ok_or_else(|| anyhow!("the input lacks a column"))?;
    Ok(match array.data_type() {
        DataType::Utf8 => {
            let strings = array.as_string::<i32>();
            Box::new(move |row| strings.is_valid(row).then(|| strings.value(row)))
        }
        DataType::LargeUtf8 => {
            let strings = array.as_string::<i64>();
            Box::new(move |row| strings.is_valid(row).then(|| strings.value(row)))
        }
        other => bail!("a text column of type {other}"),
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

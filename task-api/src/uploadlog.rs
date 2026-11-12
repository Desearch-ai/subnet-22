//! Every completed upload of either kind, written in numbered signed files to the uploads bucket, so validators count each miner's work themselves.
//! Each entry names the version its completion pinned, so every validator can look at it and sign what it saw.
//!
//! The files form a chain: each names the SHA-256 of the previous file's bytes (`prev`) and when that file was written
//! (`prev_written_at`), and `latest.json`, signed too, names the newest file's hash. A file that is gone while a later one
//! names it, younger than the bucket keeps files, was withheld; validators record that as fraud rather than read past it.

use anyhow::Result;
use bytes::Bytes;
use desearch::r2::{Bucket, Error as StorageError};
use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};

use crate::budgets::{CRAWL, EMBED};
use crate::journal::GENESIS;
use crate::lifecycle::kind_of;
use crate::manifest::{log_payload, upload_log_key, UPLOAD_LOG_LATEST};
use crate::proofs::sha256;
use crate::state::{Redis, State};

pub const PENDING: &str = "uploadlog:pending";
pub const NEXT: &str = "uploadlog:next";
/// How many pending entries the file being written holds, so a retry writes the same ones.
pub const WRITING: &str = "uploadlog:writing";
/// The bytes of the file being written: a retry puts them again unchanged, since a reader may already hold their hash.
pub const FILE: &str = "uploadlog:file";
/// `{"seq", "hash", "written_at"}` of the last file written, the link the next one names.
pub const HEAD: &str = "uploadlog:head";
pub const BATCH: isize = 1000;
const JSON: &str = "application/json";

/// An embed upload is counted in the texts and characters its input holds; a crawl upload in the URLs it was given and the miner's counts.
pub fn entry(job: &Value) -> Value {
    let mut entry = counted(job);
    for name in ["etag", "version"] {
        if let Some(pinned) = job.get(name).filter(|v| v.as_str().is_some_and(|v| !v.is_empty())) {
            entry[name] = pinned.clone();
        }
    }
    entry
}

fn counted(job: &Value) -> Value {
    if kind_of(job) == EMBED {
        return json!({
            "kind": EMBED,
            "key": job["key"],
            "task_id": job["task_id"],
            "miner": job["miner"],
            "completed_at": job["completed_at"],
            "texts": job["texts"],
            "chars": job["chars"],
        });
    }
    let reported = &job["reported"];
    let number = |name: &str| reported.get(name).and_then(Value::as_f64).map_or(0, |n| n as i64);
    json!({
        "kind": CRAWL,
        "task_id": job["task_id"],
        "round_id": job["round_id"],
        "miner": job["miner"],
        "key": job["key"],
        "completed_at": job["completed_at"],
        "assigned": crate::lifecycle::assigned(job),
        "rows": number("rows"),
        "ok": number("ok"),
        "errors": number("errors"),
    })
}

pub async fn note(redis: &Redis, job: &Value) -> Result<()> {
    let _: i64 = redis.clone().rpush(PENDING, entry(job).to_string()).await?;
    Ok(())
}

/// Writes the pending entries as the next numbered file; on a failed write the same file goes up again on the next pass.
pub async fn flush(state: &State, storage: &Bucket) -> Result<Option<i64>> {
    let mut redis = state.redis.clone();
    let kept: Option<Vec<u8>> = redis.get(FILE).await?;
    let writing: Option<isize> = redis.get(WRITING).await?;
    let (bytes, count) = match (kept, writing) {
        (Some(bytes), Some(count)) => (bytes, count),
        _ => {
            let pending: Vec<String> = redis.lrange(PENDING, 0, BATCH - 1).await?;
            if pending.is_empty() {
                return Ok(None);
            }
            let (seq, prev, prev_written_at) = link(state, storage).await?;
            let entries: Vec<Value> = pending.iter().map(|raw| serde_json::from_str(raw).unwrap_or(Value::Null)).collect();
            let mut body = Map::new();
            body.insert("seq".into(), seq.into());
            body.insert("written_at".into(), now().into());
            body.insert("prev".into(), prev);
            body.insert("prev_written_at".into(), prev_written_at);
            body.insert("entries".into(), entries.into());
            body.insert("signer".into(), state.signer().into());
            let signature = state.sign(&log_payload(&body));
            body.insert("signature".into(), signature.into());
            let bytes = serde_json::to_vec(&Value::Object(body))?;
            let _: () = redis::pipe().atomic().set(FILE, bytes.clone()).set(WRITING, pending.len()).query_async(&mut redis).await?;
            (bytes, pending.len() as isize)
        }
    };
    let file: Value = serde_json::from_slice(&bytes)?;
    let seq = file["seq"].as_i64().unwrap_or(1);
    let hash = sha256(&bytes);
    storage.put(&upload_log_key(seq as u64), Bytes::from(bytes), JSON, None).await?;
    let mut latest = Map::new();
    latest.insert("seq".into(), seq.into());
    latest.insert("head".into(), hash.clone().into());
    latest.insert("written_at".into(), file["written_at"].clone());
    latest.insert("signer".into(), state.signer().into());
    let signature = state.sign(&log_payload(&latest));
    latest.insert("signature".into(), signature.into());
    storage.put(UPLOAD_LOG_LATEST, Bytes::from(Value::Object(latest).to_string()), JSON, Some("no-store")).await?;
    let head = json!({"seq": seq, "hash": hash, "written_at": file["written_at"]}).to_string();
    let _: () = redis::pipe().atomic().ltrim(PENDING, count, -1).set(NEXT, seq + 1).set(HEAD, head).del(WRITING).del(FILE).query_async(&mut redis).await?;
    Ok(Some(seq))
}

/// The next file's number and the link it names: from Redis while it holds the last file written, else from the bucket, so a
/// copy that lost its Redis or took the log over goes on from the newest published file rather than writing over it.
async fn link(state: &State, storage: &Bucket) -> Result<(i64, Value, Value)> {
    let mut redis = state.redis.clone();
    let next: Option<i64> = redis.get(NEXT).await?;
    let head: Option<String> = redis.get(HEAD).await?;
    let head: Option<Value> = head.and_then(|raw| serde_json::from_str(&raw).ok());
    if let (Some(seq), Some(head)) = (next, &head) {
        if head["seq"].as_i64() == Some(seq - 1) {
            return Ok((seq, head["hash"].clone(), head["written_at"].clone()));
        }
    }
    let published = match storage.get(UPLOAD_LOG_LATEST).await {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes).ok().and_then(|latest| latest["seq"].as_i64()).unwrap_or(0),
        Err(StorageError::Missing) => 0,
        Err(error) => return Err(error.into()),
    };
    let seq = next.unwrap_or(1).max(published + 1);
    if seq == 1 {
        return Ok((1, GENESIS.into(), Value::Null));
    }
    match storage.get(&upload_log_key((seq - 1) as u64)).await {
        Ok(bytes) => {
            let written_at = serde_json::from_slice::<Value>(&bytes).ok().map(|before| before["written_at"].clone()).unwrap_or(Value::Null);
            Ok((seq, sha256(&bytes).into(), written_at))
        }
        // The file before is gone from the bucket: the chain starts again here, which readers that hold that file can see.
        Err(StorageError::Missing) => Ok((seq, Value::Null, Value::Null)),
        Err(error) => Err(error.into()),
    }
}

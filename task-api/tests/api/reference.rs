//! Reference vectors for validators without a model server: texts read from the upload's own input, embedded once, kept for the next validator.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, Int32Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::routing::post;
use axum::{Json, Router};
use base64::Engine;
use bytes::Bytes;
use redis::AsyncCommands;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use task_api::lifecycle;
use task_api::queues::EMBED_INPUTS;

use crate::harness::{task_id, vectors, Harness};

/// A model server answering each text with a vector holding its length, counting the texts it was sent.
async fn model_server(texts: Arc<AtomicUsize>) -> String {
    slow_model_server(texts, Duration::ZERO).await
}

async fn slow_model_server(texts: Arc<AtomicUsize>, delay: Duration) -> String {
    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<Value>| {
            let texts = texts.clone();
            async move {
                tokio::time::sleep(delay).await;
                let inputs = body["input"].as_array().cloned().unwrap_or_default();
                texts.fetch_add(inputs.len(), Ordering::SeqCst);
                assert_eq!((body["model"].clone(), body["truncate_prompt_tokens"].clone()), (json!("desearch-4b-v1"), json!(2048)));
                let data: Vec<Value> = inputs
                    .iter()
                    .enumerate()
                    .map(|(index, text)| {
                        let length = text.as_str().unwrap().len() as f32;
                        let raw: Vec<u8> = [length, 0.0, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
                        json!({"index": index, "embedding": base64::engine::general_purpose::STANDARD.encode(raw)})
                    })
                    .rev()
                    .collect();
                Json(json!({"data": data}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{addr}/v1/embeddings")
}

fn input_file(texts: &[(&str, &str)]) -> Vec<u8> {
    let text = |name: &str| Field::new(name, DataType::Utf8, false);
    let schema = Arc::new(Schema::new(vec![
        text("text_id"),
        text("page_key"),
        text("url"),
        text("content_sha1"),
        text("kind"),
        Field::new("index", DataType::Int32, false),
        Field::new("text", DataType::LargeUtf8, false),
    ]));
    let strings = |value: &dyn Fn(&(&str, &str)) -> String| -> ArrayRef { Arc::new(StringArray::from(texts.iter().map(value).collect::<Vec<_>>())) };
    let columns = vec![
        strings(&|t| t.0.to_string()),
        strings(&|_| "pages/a.example/1".to_string()),
        strings(&|_| "https://a.example/1".to_string()),
        strings(&|_| "c".repeat(40)),
        strings(&|_| "page".to_string()),
        Arc::new(Int32Array::from(vec![0; texts.len()])) as ArrayRef,
        Arc::new(LargeStringArray::from(texts.iter().map(|t| t.1).collect::<Vec<_>>())) as ArrayRef,
    ];
    let mut out = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut out, schema.clone(), None).unwrap();
    writer.write(&RecordBatch::try_new(schema, columns).unwrap()).unwrap();
    writer.close().unwrap();
    out
}

fn first_number(embedding: &Value) -> f32 {
    let raw = base64::engine::general_purpose::STANDARD.decode(embedding.as_str().unwrap()).unwrap();
    f32::from_le_bytes(raw[..4].try_into().unwrap())
}

#[tokio::test]
async fn a_validator_without_a_model_server_gets_reference_vectors_from_the_uploads_own_input() {
    let sent = Arc::new(AtomicUsize::new(0));
    let url = model_server(sent.clone()).await;
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_reference_url = url.clone();
        s.embed_reference_key = "key".into();
    })
    .await;
    let input = input_file(&[("t1", "one"), ("t2", "three"), ("t3", "seventeen letters")]);
    let input_key = "embed/inputs/dt=2026-10-14/a.parquet";
    h.state.storage.put(input_key, Bytes::from(input.clone()), "application/vnd.apache.parquet", None).await.unwrap();
    let entry = json!({"input_key": input_key, "input_sha256": format!("{:x}", Sha256::digest(&input)), "texts": 3, "chars": 25,
                       "pages": [{"page_key": "pages/a.example/1", "content_sha1": "c".repeat(40), "url": "https://a.example/1", "host": "a.example"}]});
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    let task = h.miner.post("/v1/tasks/claim", json!({"kind": "embed", "models": ["desearch-4b-v1"]})).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), vectors(3)).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    h.opened(Some(&h.validator), Some(&task_id(&task))).await;
    let path = format!("/v1/embed/{}/reference", task_id(&task));

    let answer = h.validator.post(&path, json!({"text_ids": ["t3", "t1"]})).await.unwrap();
    let vectors = answer["vectors"].as_array().unwrap();
    assert_eq!((vectors[0]["text_id"].clone(), vectors[1]["text_id"].clone()), (json!("t3"), json!("t1")));
    assert_eq!((first_number(&vectors[0]["embedding"]), first_number(&vectors[1]["embedding"])), (17.0, 3.0), "each vector is its own text's");
    assert_eq!(answer["model"], "desearch-4b-v1");

    let again = h.other_validator.post(&path, json!({"text_ids": ["t1", "t2"]})).await.unwrap();
    assert_eq!(first_number(&again["vectors"][1]["embedding"]), 5.0);
    assert_eq!(sent.load(Ordering::SeqCst), 3, "a text is embedded once, whichever validator asks first");

    assert_eq!(h.validator.post(&path, json!({"text_ids": ["nope"]})).await.unwrap_err().status, 422);
    assert_eq!(h.validator.post(&path, json!({"text_ids": (0..65).map(|i| format!("t{i}")).collect::<Vec<_>>()})).await.unwrap_err().status, 422);
    assert_eq!(h.miner.post(&path, json!({"text_ids": ["t1"]})).await.unwrap_err().status, 403, "only validators");
    assert_eq!(h.validator.post("/v1/embed/0000000000000000/reference", json!({"text_ids": ["t1"]})).await.unwrap_err().status, 409);
}

#[tokio::test]
async fn a_validator_that_gives_up_mid_call_still_leaves_the_vectors_for_its_retry() {
    let sent = Arc::new(AtomicUsize::new(0));
    let url = slow_model_server(sent.clone(), Duration::from_secs(1)).await;
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_reference_url = url.clone();
        s.embed_reference_key = "key".into();
    })
    .await;
    let input = input_file(&[("t1", "one"), ("t2", "three")]);
    let input_key = "embed/inputs/dt=2026-10-19/slow.parquet";
    h.state.storage.put(input_key, Bytes::from(input.clone()), "application/vnd.apache.parquet", None).await.unwrap();
    let entry = json!({"input_key": input_key, "input_sha256": format!("{:x}", Sha256::digest(&input)), "texts": 2, "chars": 8,
                       "pages": [{"page_key": "pages/a.example/1", "content_sha1": "c".repeat(40), "url": "https://a.example/1", "host": "a.example"}]});
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    let task = h.miner.post("/v1/tasks/claim", json!({"kind": "embed", "models": ["desearch-4b-v1"]})).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), vectors(2)).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    h.opened(Some(&h.validator), Some(&task_id(&task))).await;
    let path = format!("/v1/embed/{}/reference", task_id(&task));

    assert!(tokio::time::timeout(Duration::from_millis(200), h.validator.post(&path, json!({"text_ids": ["t1", "t2"]}))).await.is_err());
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let answer = h.validator.post(&path, json!({"text_ids": ["t1", "t2"]})).await.expect("the retry gets the vectors the first call computed");
    assert_eq!(first_number(&answer["vectors"][1]["embedding"]), 5.0);
    assert_eq!(sent.load(Ordering::SeqCst), 2, "the model server was asked once");
}

#[tokio::test]
async fn without_a_reference_model_server_the_endpoint_says_so() {
    let h = Harness::with(|s| s.embed_tasks = true).await;
    assert_eq!(h.validator.post("/v1/embed/0000000000000000/reference", json!({"text_ids": ["t1"]})).await.unwrap_err().status, 503);
}

#[tokio::test]
async fn a_task_handed_to_another_miner_is_sampled_afresh_from_the_same_input() {
    let sent = Arc::new(AtomicUsize::new(0));
    let url = model_server(sent.clone()).await;
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_reference_url = url.clone();
        s.embed_reference_key = "key".into();
    })
    .await;
    let names: Vec<(String, String)> = (0..70).map(|i| (format!("t{i}"), "x".repeat(i + 1))).collect();
    let input = input_file(&names.iter().map(|(id, text)| (id.as_str(), text.as_str())).collect::<Vec<_>>());
    let input_key = "embed/inputs/dt=2026-10-14/b.parquet";
    h.state.storage.put(input_key, Bytes::from(input.clone()), "application/vnd.apache.parquet", None).await.unwrap();
    let entry = json!({"input_key": input_key, "input_sha256": format!("{:x}", Sha256::digest(&input)), "texts": 70, "chars": 2485,
                       "pages": [{"page_key": "pages/a.example/1", "content_sha1": "c".repeat(40), "url": "https://a.example/1", "host": "a.example"}]});
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    let claim = json!({"kind": "embed", "models": ["desearch-4b-v1"]});
    let ids = |range: std::ops::Range<usize>| json!({"text_ids": range.map(|i| format!("t{i}")).collect::<Vec<_>>()});

    let task = h.miner.post("/v1/tasks/claim", claim.clone()).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), vectors(70)).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    h.opened(Some(&h.validator), Some(&task_id(&task))).await;
    let path = format!("/v1/embed/{}/reference", task_id(&task));
    h.validator.post(&path, ids(0..40)).await.unwrap();
    let failed = json!({"sampled": 2, "matched": 0, "mismatched": 2, "min_similarity": 0.12, "verdict": "fail", "reason": "vectors_mismatch",
                        "samples": [{"text_id": "t0", "index": 0, "outcome": "mismatched", "similarity": 0.12}, {"text_id": "t1", "index": 1, "outcome": "mismatched", "similarity": 0.12}]});
    h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), failed).await.unwrap();

    let again = h.rival.post("/v1/tasks/claim", claim).await.unwrap()["tasks"][0].clone();
    assert_eq!(task_id(&again), task_id(&task));
    h.put(again["upload"]["url"].as_str().unwrap(), vectors(70)).await;
    h.rival.post(&format!("/v1/tasks/{}/complete", task_id(&again)), json!({"key": again["upload"]["key"], "bytes": 7})).await.unwrap();
    h.opened(None, Some(&task_id(&again))).await;
    h.validator.post(&path, ids(30..70)).await.expect("a new upload gets its own sample");
    assert_eq!(sent.load(Ordering::SeqCst), 70, "texts asked before are not embedded again");
    assert_eq!(h.validator.post(&path, ids(0..30)).await.unwrap_err().status, 429, "still at most 64 texts of one upload");
}

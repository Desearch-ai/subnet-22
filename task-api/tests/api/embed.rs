//! Embed tasks: rounds from the publisher's inputs, their own pool, and closed until switched on.

use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::lifecycle;
use task_api::queues::EMBED_INPUTS;

use crate::harness::{task_id, Client, Harness};

fn pages() -> Value {
    json!((0..2).map(|i| json!({"url": format!("https://site{i}.example/story/{i}"), "host": format!("site{i}.example"), "page_key": format!("pages/{i:040x}.zst"), "content_sha1": format!("{i:040x}")})).collect::<Vec<_>>())
}

fn input() -> Value {
    json!({"input_key": "embed-inputs/dt=2026-09-24/task=crawl-1.parquet", "input_sha256": "ab".repeat(32), "texts": 6, "chars": 5400, "pages": pages()})
}

fn embed_score(verdict: &str, outcome: &str, reason: &str) -> Value {
    json!({
        "returned": 6,
        "sampled": 2,
        "matched": if outcome == "matched" { 2 } else { 0 },
        "mismatched": if outcome == "mismatched" { 2 } else { 0 },
        "min_similarity": if outcome == "matched" { 0.999 } else { 0.12 },
        "verdict": verdict,
        "reason": reason,
        "samples": (0..2).map(|i| json!({"text_id": format!("t{i}"), "outcome": outcome, "similarity": 0.999})).collect::<Vec<_>>(),
    })
}

async fn embedding() -> Harness {
    Harness::with(|s| s.embed_tasks = true).await
}

async fn embed_round(h: &Harness, entry: Value) -> String {
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    let round = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    round.round_id
}

async fn embed_task(h: &Harness, miner: &Client) -> Value {
    let task = miner.post("/v1/tasks/claim", json!({"kind": "embed"})).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), b"PAR1vectorsPAR1".to_vec()).await;
    miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    task
}

#[tokio::test]
async fn a_published_page_is_embedded_credited_and_recorded() {
    let h = embedding().await;
    let round_id = embed_round(&h, input()).await;
    assert_eq!(h.miner.claim().await["refusal"]["code"], "QUEUE_EMPTY", "crawl sees no embed work");

    let task = embed_task(&h, &h.miner).await;
    assert_eq!((task["kind"].clone(), task["round_id"].clone()), (json!("embed"), json!(round_id)));
    assert_eq!((task["model"].clone(), task["texts"].clone()), (json!(h.state.settings.embed_model), json!(6)));
    assert_eq!(task["input"]["sha256"], input()["input_sha256"]);
    assert!(task["input"]["url"].as_str().unwrap().contains("X-Amz-Signature="));
    assert_eq!(task["urls"], json!(pages().as_array().unwrap().iter().map(|p| p["url"].clone()).collect::<Vec<_>>()));

    let job = h.opened(Some(&h.validator), None).await;
    let listed = h.open_list().await;
    assert_eq!(listed["uploads"].as_array().unwrap().iter().map(|m| m["kind"].clone()).collect::<Vec<_>>(), [json!("embed")]);
    assert_eq!(
        (job["kind"].clone(), job["model"].clone(), job["input_key"].clone()),
        (json!("embed"), json!(h.state.settings.embed_model), input()["input_key"].clone())
    );

    let scored = h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("pass", "matched", "ok")).await.unwrap();
    assert_eq!((scored["verdict"].clone(), scored["credited"].clone()), (json!("pass"), json!(5400)));
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!((miner["pools"]["embed"]["budget"].clone(), miner["pools"]["crawl"]["budget"].clone()), (json!(2), json!(1)));
    assert_eq!(h.public("/v1/shares").await.1["pools"], json!({"embed": {h.miner.ss58(): 1.0}}));

    let page = pages()[0]["page_key"].as_str().unwrap().to_string();
    let catalog: Vec<(String, String, Option<String>)> = h
        .state
        .db
        .run(move |conn| {
            Ok(conn
                .prepare("SELECT model, state, vectors_key FROM embeddings WHERE page_key = ?")?
                .query_map([page], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap();
    assert_eq!((catalog[0].0.as_str(), catalog[0].1.as_str()), (h.state.settings.embed_model.as_str(), "done"));
    let vectors_key = catalog[0].2.clone().unwrap();
    assert!(vectors_key.ends_with(&format!("task={}.parquet", task_id(&task))));

    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let published = h.public(&format!("/v1/rounds/{round_id}")).await.1;
    assert_eq!(published["kind"], "embed");
    assert_eq!(published["manifest"][0]["input_sha256"], input()["input_sha256"]);
    let manifest: Vec<Value> = published["manifest"].as_array().unwrap().clone();
    assert_eq!(task_api::proofs::manifest_hash(&manifest, published["seed_block"].as_i64().unwrap()), published["manifest_hash"].as_str().unwrap());
}

#[tokio::test]
async fn the_same_page_version_is_not_embedded_twice() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, input().to_string()).await.unwrap();
    assert!(lifecycle::open_embed_rounds(&h.state).await.unwrap().is_none());
    let left: i64 = h.redis.clone().llen(EMBED_INPUTS).await.unwrap();
    assert_eq!(left, 0);
    let mut changed = input();
    let mut page = pages()[0].clone();
    page["content_sha1"] = "f".repeat(40).into();
    changed["pages"] = json!([page]);
    assert!(lifecycle::open_embed_rounds(&h.state).await.unwrap().is_none());
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, changed.to_string()).await.unwrap();
    assert!(lifecycle::open_embed_rounds(&h.state).await.unwrap().is_some());
}

#[tokio::test]
async fn a_failed_embed_task_strikes_only_the_embed_pool() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    h.opened(Some(&h.validator), None).await;
    let scored = h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("fail", "mismatched", "vectors_mismatch")).await.unwrap();
    assert_eq!(scored["verdict"], "fail");
    let hotkey = h.miner.ss58();
    let strikes: Vec<(String, String)> = h
        .state
        .db
        .run(move |conn| {
            Ok(conn
                .prepare("SELECT pool, reason FROM strikes WHERE hotkey = ?")?
                .query_map([hotkey], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap();
    assert_eq!(strikes, [("embed".to_string(), "vectors_mismatch".to_string())]);
    assert_eq!(h.miner.post("/v1/tasks/claim", json!({"kind": "embed"})).await.unwrap()["refusal"]["code"], "ALREADY_HELD");
    assert_eq!(h.rival.post("/v1/tasks/claim", json!({"kind": "embed"})).await.unwrap()["tasks"][0]["task_id"], task["task_id"]);
}

#[tokio::test]
async fn embed_tasks_stay_closed_until_switched_on() {
    let h = Harness::start().await;
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, input().to_string()).await.unwrap();
    assert!(lifecycle::open_embed_rounds(&h.state).await.unwrap().is_none());
    let left: i64 = h.redis.clone().llen(EMBED_INPUTS).await.unwrap();
    assert_eq!(left, 1, "nothing is thrown away");
    let answer = h.miner.post("/v1/tasks/claim", json!({"kind": "embed"})).await.unwrap();
    assert_eq!(answer["refusal"], json!({"code": "KIND_CLOSED", "inputs": {"kind": "embed", "retry_after": 3600.0}}));
    assert!(!answer["receipt"].is_null());
    let health = h.public("/v1/health").await.1;
    assert_eq!((health["embed_tasks"].clone(), health["embed_model"].clone()), (json!(false), json!("qwen3-embedding-8b")));
}

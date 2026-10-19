//! Round URL lists wait in Redis between commit and reveal; SQLite keeps each batch's count and hash.

use redis::AsyncCommands;
use serde_json::json;
use task_api::janitor::LEASE_KEY;
use task_api::queues::staged_key;
use task_api::rounds::{self, Url};
use task_api::{lifecycle, roundstore};

use crate::harness::{urls, Harness};

async fn stored_batches(h: &Harness, round_id: &str) -> String {
    let asked = round_id.to_string();
    h.state.db.run(move |conn| Ok(conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [&asked], |row| row.get(0))?)).await.unwrap()
}

#[tokio::test]
async fn a_round_waits_in_redis_and_sqlite_keeps_only_its_counts() {
    let h = Harness::start().await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": urls(6)})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    assert!(!stored_batches(&h, &round_id).await.contains("https://"));
    assert!(h.redis.clone().exists::<_, bool>(staged_key(&round_id)).await.unwrap());
    assert_eq!(h.revealed().await, 2);
    assert!(!h.redis.clone().exists::<_, bool>(staged_key(&round_id)).await.unwrap(), "dropped once its tasks are queued");
    let task = h.miner.claim().await["tasks"][0].clone();
    assert_eq!(task["urls"].as_array().unwrap().len(), 3);
    assert_eq!(task["round_id"], round_id.as_str());
}

#[tokio::test]
async fn a_round_kept_with_its_urls_before_staging_still_fills() {
    let h = Harness::start().await;
    let urls: Vec<Url> = (0..3).map(|i| Url { host: "old.example".into(), url: format!("https://old.example/{i}") }).collect();
    let target = h.state.seeds.target_block().await.unwrap();
    let round = rounds::open_batches(rounds::pack(urls, 3), target, "crawl", desearch::time::now());
    let saved = round.clone();
    h.state.db.run(move |conn| roundstore::save(conn, &saved)).await.unwrap();
    assert_eq!(h.revealed().await, 1);
    assert_eq!(h.miner.claim().await["tasks"][0]["urls"], json!(["https://old.example/0", "https://old.example/1", "https://old.example/2"]));
    assert!(!stored_batches(&h, &round.round_id).await.contains("https://"), "sealed once queued");
}

#[tokio::test]
async fn a_round_whose_lists_were_lost_closes_without_tasks() {
    let h = Harness::start().await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": urls(6)})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    let _: i64 = h.redis.clone().del(staged_key(&round_id)).await.unwrap();
    assert_eq!(h.revealed().await, 0, "no task is queued without its URLs");
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
}

#[tokio::test]
async fn enqueues_wait_while_a_leader_from_before_staging_runs() {
    let h = Harness::start().await;
    let _: () = h.redis.clone().set(LEASE_KEY, "0123456789abcdef0123456789abcdef").await.unwrap();
    let refused = h.admin.post("/v1/admin/enqueue", json!({"urls": urls(3)})).await.unwrap_err();
    assert_eq!(refused.status, 503);
    let _: () = h.redis.clone().set(LEASE_KEY, "staging:0123456789abcdef").await.unwrap();
    assert!(h.admin.post("/v1/admin/enqueue", json!({"urls": urls(3)})).await.is_ok());
    let _: i64 = h.redis.clone().del(LEASE_KEY).await.unwrap();
    assert!(h.admin.post("/v1/admin/enqueue", json!({"urls": urls(4)})).await.is_ok());
}

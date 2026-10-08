//! When a completion counts: on arrival, once, and never for a file written after it.

use std::time::{Duration, SystemTime};

use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::lifecycle;
use task_api::manifest::OPEN_LIST_KEY;
use task_api::queues::CLAIMS;
use task_api::rounds::UPLOAD_GRACE_S;

use crate::harness::{parquet, task_id, urls, Harness, BUCKET};

/// A claimed task with its file uploaded, and the completion report for it.
async fn uploaded(h: &Harness) -> (Value, Value) {
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let body = parquet(&task);
    h.put(task["upload"]["url"].as_str().unwrap(), body.clone()).await;
    let rows = task["urls"].as_array().unwrap().len();
    let report = json!({"key": task["upload"]["key"], "rows": rows, "ok": rows, "errors": 0, "bytes": body.len()});
    (task, report)
}

/// The next upload copies take this long, as when storage is slow.
fn slow_copies(h: &Harness, delay: Duration) {
    h.stub.state.slow("PUT", &format!("{BUCKET}/submitted/"), 10, delay);
}

async fn strikes(h: &Harness) -> i64 {
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap()
}

async fn expire_at(h: &Harness, task: &str, at: f64) {
    let _: i64 = redis::cmd("ZADD").arg(CLAIMS).arg("XX").arg(at).arg(task).query_async(&mut h.redis.clone()).await.unwrap();
}

#[tokio::test]
async fn a_completion_that_arrived_in_time_counts_however_long_the_api_takes() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    let id = task_id(&task);
    slow_copies(&h, Duration::from_secs(1));
    let sent = desearch::time::now();
    expire_at(&h, &id, sent + 0.3).await;
    let path = format!("/v1/tasks/{id}/complete");
    let completing = h.miner.post(&path, report);
    let reclaiming = async {
        tokio::time::sleep(Duration::from_millis(600)).await;
        lifecycle::reclaim_expired(&h.state, desearch::time::now()).await.unwrap()
    };
    let (done, reclaimed) = tokio::join!(completing, reclaiming);
    assert!(reclaimed.is_empty(), "a claim being completed is not taken back");
    assert_eq!(done.unwrap()["status"], "open_for_validation");
    let job = h.state.validation.job(&h.redis, &id).await.unwrap().unwrap();
    let completed = job["completed_at"].as_f64().unwrap();
    assert!(sent - 0.05 <= completed && completed < sent + 0.3, "stamped on arrival");
    assert_eq!(strikes(&h).await, 0);
}

#[tokio::test]
async fn a_completion_that_arrived_after_the_claim_ended_is_refused() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    expire_at(&h, &task_id(&task), desearch::time::now() - 1.0).await;
    assert_eq!(h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), report).await.unwrap_err().status, 409);
}

#[tokio::test]
async fn a_file_written_after_the_completion_arrived_is_refused() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    let stored = h.stub.state.root.join(BUCKET).join(report["key"].as_str().unwrap());
    let file = std::fs::File::options().write(true).open(stored).unwrap();
    file.set_modified(SystemTime::now() + Duration::from_secs(60)).unwrap();
    assert_eq!(h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), report).await.unwrap_err().status, 409);
    assert!(h.state.validation.job(&h.redis, &task_id(&task)).await.unwrap().is_none());
}

#[tokio::test]
async fn a_repeated_completion_gets_the_first_ones_answer() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    let path = format!("/v1/tasks/{}/complete", task_id(&task));
    slow_copies(&h, Duration::from_millis(500));
    let (first, again) = tokio::join!(h.miner.post(&path, report.clone()), h.miner.post(&path, report.clone()));
    let later = h.miner.post(&path, report.clone()).await.unwrap();
    assert_eq!(first.unwrap(), again.unwrap());
    assert_eq!(h.miner.post(&path, report.clone()).await.unwrap(), later);
    let frozen: Vec<String> = h.stub.state.keys(BUCKET, "submitted/").into_iter().filter(|k| k.ends_with(".parquet")).collect();
    assert_eq!(frozen.len(), 1, "the upload was taken in once");
    assert_eq!(h.rival.post(&path, report).await.unwrap_err().status, 409);
}

#[tokio::test]
async fn an_abandon_during_a_completion_waits_and_costs_nothing() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    let id = task_id(&task);
    slow_copies(&h, Duration::from_millis(500));
    let path = format!("/v1/tasks/{id}/complete");
    let completing = h.miner.post(&path, report);
    let abandoning = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        h.miner.post(&format!("/v1/tasks/{id}/abandon"), json!({})).await
    };
    let (done, abandoned) = tokio::join!(completing, abandoning);
    assert_eq!(abandoned.unwrap_err().status, 409);
    assert_eq!(done.unwrap()["status"], "open_for_validation");
    assert_eq!(strikes(&h).await, 0);
}

#[tokio::test]
async fn a_claims_clock_starts_when_the_claim_is_handed_over() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let handed = desearch::time::now();
    let expiry: f64 = h.redis.clone().zscore(CLAIMS, task_id(&task)).await.unwrap();
    assert!(expiry >= handed + h.state.settings.claim_ttl - 0.2);
    assert_eq!(task["expires_at"].as_f64().unwrap(), expiry - UPLOAD_GRACE_S);
}

#[tokio::test]
async fn an_upload_too_close_to_its_deadline_is_not_listed_for_validators() {
    let h = Harness::start().await;
    let (task, report) = uploaded(&h).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), report).await.unwrap();
    assert_eq!(h.opened(None, None).await["task_id"], task_id(&task).as_str());
    let job = h.state.validation.job(&h.redis, &task_id(&task)).await.unwrap().unwrap();
    let late = job["deadline"].as_f64().unwrap() - lifecycle::CHECKABLE_LEFT_S + 30.0;
    lifecycle::publish_open(&h.state, late).await.unwrap();
    assert_eq!(h.json_object(OPEN_LIST_KEY).unwrap()["uploads"], json!([]));
}

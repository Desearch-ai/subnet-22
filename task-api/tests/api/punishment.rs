//! What a miner loses for lapsed claims, failed and hostile uploads, and how refusals are logged.

use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::budgets::{self, HOSTILE_LOCKOUT_H, HOUR, LOCKOUT_STEPS_H};
use task_api::lifecycle;
use task_api::queues::{CLAIMS, PPENDING};

use crate::harness::{judged, parquet, task_id, urls, Harness};

async fn started(task_urls: usize) -> Harness {
    let h = Harness::with(|s| s.task_urls = task_urls).await;
    h.enqueue(urls(6)).await;
    h
}

async fn lapse_all(h: &Harness) {
    let claimed: Vec<String> = h.redis.clone().zrange(CLAIMS, 0, -1).await.unwrap();
    for task in claimed {
        let _: i64 = h.redis.clone().zadd(CLAIMS, task, 0).await.unwrap();
    }
    lifecycle::reclaim_expired(&h.state, desearch::time::now()).await.unwrap();
}

async fn sql<T: rusqlite::types::FromSql + Send + 'static>(h: &Harness, query: &'static str, hotkey: String) -> T {
    h.state.db.run(move |conn| Ok(conn.query_row(query, [hotkey], |row| row.get(0))?)).await.unwrap()
}

async fn credit_of(h: &Harness, hotkey: String) -> i64 {
    sql(h, "SELECT COALESCE(SUM(amount), 0) FROM credits WHERE hotkey = ?", hotkey).await
}

async fn with_budget(h: &Harness, budget: i64) {
    let hotkey = h.miner.ss58();
    h.state
        .db
        .run(move |conn| {
            budgets::get_or_create(conn, &hotkey, "crawl")?;
            conn.execute("UPDATE miners SET budget = ?", [budget])?;
            Ok(())
        })
        .await
        .unwrap();
}

async fn locked_until(h: &Harness) -> Option<f64> {
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| budgets::locked_until(conn, &hotkey, "crawl", desearch::time::now())).await.unwrap()
}

#[tokio::test]
async fn a_hotkey_that_only_hoards_is_locked_out_on_its_second_lapse() {
    let h = started(1).await;
    h.miner.post("/v1/tasks/claim", json!({"count": 1})).await.unwrap();
    lapse_all(&h).await;
    assert!(locked_until(&h).await.is_none(), "one lapse is a warning");
    let _: () = redis::cmd("FLUSHDB").query_async(&mut h.redis.clone()).await.unwrap();
    h.state.db.run(|conn| Ok(conn.execute("UPDATE strikes SET at = at - 600", [])?)).await.unwrap();
    h.enqueue(urls(6)).await;
    h.miner.claim().await;
    lapse_all(&h).await;
    let second = locked_until(&h).await.expect("the second lapse locks");
    assert!(second - LOCKOUT_STEPS_H[0] * HOUR > 0.0);
    assert_eq!(credit_of(&h, h.miner.ss58()).await, -2, "each lapse takes its URLs back");
}

#[tokio::test]
async fn claims_that_lapse_together_cost_one_strike() {
    let h = started(1).await;
    with_budget(&h, 5).await;
    let claimed = h.miner.post("/v1/tasks/claim", json!({"count": 5})).await.unwrap();
    lapse_all(&h).await;
    let strikes: i64 = sql(&h, "SELECT COUNT(*) FROM strikes WHERE hotkey = ?", h.miner.ss58()).await;
    assert!(claimed["tasks"].as_array().unwrap().len() > 1);
    assert_eq!(strikes, 1);
}

#[tokio::test]
async fn an_abandoned_task_is_a_lapse_too() {
    let h = started(1).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let answer = h.miner.post(&format!("/v1/tasks/{}/abandon", task_id(&task)), json!({})).await.unwrap();
    let reason: String = sql(&h, "SELECT reason FROM strikes WHERE hotkey = ?", h.miner.ss58()).await;
    assert_eq!((answer["budget"].clone(), reason.as_str()), (json!(1), "abandoned"));
    assert_eq!(credit_of(&h, h.miner.ss58()).await, -1);
}

#[tokio::test]
async fn a_failed_task_takes_its_urls_back_from_the_day() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    judged(&h, &h.validator, "fail", None, json!({})).await.1.unwrap();
    assert_eq!(credit_of(&h, h.miner.ss58()).await, -3);
}

#[tokio::test]
async fn an_upload_that_crashes_most_checks_locks_the_miner_out() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    judged(&h, &h.validator, "fail", None, json!({"reason": "unscorable", "crashed": true})).await.1.unwrap();
    let left = locked_until(&h).await.expect("locked out") - desearch::time::now();
    assert!(HOSTILE_LOCKOUT_H * HOUR - 60.0 < left && left <= HOSTILE_LOCKOUT_H * HOUR);
    assert_eq!(h.miner.claim().await["refusal"]["code"], "LOCKED_OUT");
}

#[tokio::test]
async fn a_crash_the_validator_could_not_reproduce_costs_nothing() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    judged(&h, &h.validator, "fail", None, json!({"reason": "unscorable"})).await.1.unwrap();
    assert!(locked_until(&h).await.is_none());
}

#[tokio::test]
async fn an_upload_that_is_not_parquet_is_refused_at_completion() {
    let h = started(1).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let url = task["upload"]["url"].as_str().unwrap();
    let complete = format!("/v1/tasks/{}/complete", task_id(&task));
    h.put(url, b"<html>not a file of rows</html>".to_vec()).await;
    assert_eq!(h.miner.post(&complete, json!({"key": task["upload"]["key"]})).await.unwrap_err().status, 422);
    h.put(url, parquet(&task)).await;
    let done = h.miner.post(&complete, json!({"key": task["upload"]["key"]})).await.unwrap();
    assert_eq!(done["status"], "open_for_validation", "a fixed upload is still accepted");
}

#[tokio::test]
async fn one_claim_returns_several_tasks_and_a_receipt_for_each() {
    let h = started(1).await;
    with_budget(&h, 3).await;
    let answer = h.miner.post("/v1/tasks/claim", json!({"count": 5})).await.unwrap();
    let tasks: Vec<Value> = answer["tasks"].as_array().unwrap().iter().map(|t| t["task_id"].clone()).collect();
    let receipts: Vec<Value> = answer["receipts"].as_array().unwrap().iter().map(|r| r["body"]["task_id"].clone()).collect();
    assert_eq!(tasks.len(), 3, "as many as asked, up to the budget");
    assert_eq!(receipts, tasks);
}

#[tokio::test]
async fn a_repeated_refusal_is_answered_but_signed_only_once_a_minute() {
    let h = started(1).await;
    let _: () = redis::cmd("FLUSHDB").query_async(&mut h.redis.clone()).await.unwrap();
    let first = h.miner.claim().await;
    let again = h.miner.claim().await;
    assert_eq!((first["refusal"]["code"].clone(), again["refusal"]["code"].clone()), (json!("QUEUE_EMPTY"), json!("QUEUE_EMPTY")));
    assert!(!first["receipt"].is_null() && again["receipt"].is_null());
}

#[tokio::test]
async fn the_bot_gets_no_room_and_miners_no_tasks_while_publishing_is_far_behind() {
    let h = Harness::with(|s| {
        s.publish_lag = 60.0;
        s.publish_lag_limit = 120.0;
    })
    .await;
    h.enqueue(urls(6)).await;
    assert!(h.public("/v1/room").await.1["room_tasks"].as_i64().unwrap() > 0);
    let now = desearch::time::now();
    for n in 0..250 {
        let _: i64 = h.redis.clone().zadd(PPENDING, format!("waiting-{n}"), now).await.unwrap();
    }
    let room = h.public("/v1/room").await.1;
    assert_eq!(room["room_tasks"], 0);
    assert!(room["in_system"].as_i64().unwrap() >= 250);
    let claim = h.miner.claim().await;
    assert_eq!((claim["tasks"].clone(), claim["refusal"]["code"].clone()), (json!([]), json!("VALIDATION_BACKLOG")));
    assert_eq!(claim["refusal"]["inputs"]["publish_waiting"], 250);
}

//! Which uploads are checked, what an unchecked one is paid, and what a failed check takes back.

use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::manifest::{log_payload, upload_log_key, UPLOAD_LOG_LATEST};
use task_api::{budgets, checks, lifecycle, sampling, uploadlog};

use crate::harness::{judged, parquet, score, task_id, urls, Harness, BUCKET};

async fn started(task_urls: usize, check_share: f64) -> Harness {
    let h = Harness::with(|s| {
        s.task_urls = task_urls;
        s.check_share = check_share;
    })
    .await;
    h.enqueue(urls(6)).await;
    h
}

/// A hotkey past its first checks, busy enough that the draw never picks it.
async fn established(h: &Harness, hotkey: String) {
    let passed_at = desearch::time::now() - 3600.0;
    let asked = hotkey.clone();
    h.state
        .db
        .run(move |conn| {
            (0..sampling::NEW_HOTKEY_PASSES).try_for_each(|n| checks::record(conn, &asked, &format!("earlier-{n}"), true, passed_at, 0, 0, passed_at))
        })
        .await
        .unwrap();
    let hour = (desearch::time::now() / 3600.0).floor() as i64;
    let _: () = h.redis.clone().set(format!("uploads:{hotkey}:{hour}"), 1_000_000_000i64).await.unwrap();
}

/// Why the upload was picked for a check, or "finalized" when it was not.
async fn settled(h: &Harness, task: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        lifecycle::settle_seeded(&h.state).await.unwrap();
        match h.state.validation.job(&h.redis, task).await.unwrap() {
            None => return "finalized".into(),
            Some(job) if job.get("picked").is_some_and(|p| p.is_string()) => return job["picked"].as_str().unwrap().into(),
            Some(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
    panic!("the upload's seed block never came");
}

async fn credits(h: &Harness, hotkey: String) -> i64 {
    h.state.db.run(move |conn| Ok(conn.query_row("SELECT COALESCE(SUM(amount), 0) FROM credits WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap()
}

async fn start_recheck(h: &Harness, hotkey: String) {
    h.state.db.run(move |conn| checks::start_recheck(conn, &hotkey)).await.unwrap();
}

#[tokio::test]
async fn a_new_hotkey_has_its_upload_checked() {
    let h = started(3, 1.0).await;
    let task = h.mine(&h.miner, 0).await;
    assert_eq!(settled(&h, &task_id(&task)).await, sampling::NEW);
}

#[tokio::test]
async fn an_upload_no_check_drew_is_paid_on_its_report_and_published() {
    let h = started(3, 0.0).await;
    established(&h, h.miner.ss58()).await;
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| checks::record(conn, &hotkey, "with-errors", true, 0.0, 8, 0, desearch::time::now())).await.unwrap();
    let task = task_id(&h.mine(&h.miner, 1).await);
    assert_eq!(settled(&h, &task).await, "finalized");
    let view = h.view(&task).await;
    assert_eq!((view["status"].clone(), view["score"]["reason"].clone()), (json!("pass"), json!("unchecked")));
    assert_eq!(view["score"]["credited"], 3, "two pages, and one error at 9/10");
    assert_eq!(h.state.publish.depth(&h.redis).await.unwrap(), 1);
    assert_eq!(credits(&h, h.miner.ss58()).await, 3);
}

#[tokio::test]
async fn a_report_short_of_coverage_fails_without_a_check() {
    let h = started(3, 0.0).await;
    established(&h, h.miner.ss58()).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), parquet(&task)).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "rows": 1, "ok": 1, "errors": 0})).await.unwrap();
    settled(&h, &task_id(&task)).await;
    let view = h.view(&task_id(&task)).await;
    assert_eq!((view["status"].clone(), view["score"]["reason"].clone()), (json!("queued"), json!("coverage")));
}

#[tokio::test]
async fn a_failed_check_takes_back_what_passed_since_the_last_pass() {
    let h = started(2, 0.0).await;
    let hotkey = h.miner.ss58();
    established(&h, hotkey.clone()).await;
    let unchecked = task_id(&h.mine(&h.miner, 0).await);
    settled(&h, &unchecked).await;
    start_recheck(&h, hotkey.clone()).await;
    let checked = task_id(&h.mine(&h.miner, 0).await);
    assert_eq!(settled(&h, &checked).await, sampling::RECHECK);
    let verdict = judged(&h, &h.validator, "fail", Some(&checked), json!({})).await.1.unwrap();
    assert_eq!(verdict["verdict"], "fail");
    assert_eq!(h.view(&unchecked).await["score"]["verdict"], "withdrawn");
    assert_eq!(credits(&h, hotkey.clone()).await, -2, "its credit is taken back, and the failed upload costs its URLs");
    assert_eq!(h.state.publish.depth(&h.redis).await.unwrap(), 2, "its publish job and the withdrawal");
    let asked = hotkey.clone();
    let (recheck, locked) = h
        .state
        .db
        .run(move |conn| Ok((checks::recheck_left(conn, &asked)?, budgets::locked_until(conn, &asked, "crawl", desearch::time::now())?)))
        .await
        .unwrap();
    assert_eq!(recheck, sampling::RECHECK_UPLOADS, "its next uploads are checked");
    assert!(locked.is_none(), "one fail is not a penalty");
}

#[tokio::test]
async fn two_fails_among_the_last_ten_checks_lock_out_and_wipe_the_day() {
    let h = started(2, 0.0).await;
    let hotkey = h.miner.ss58();
    established(&h, hotkey.clone()).await;
    for _ in 0..2 {
        start_recheck(&h, hotkey.clone()).await;
        let task = task_id(&h.mine(&h.miner, 0).await);
        settled(&h, &task).await;
        judged(&h, &h.validator, "fail", Some(&task), json!({})).await.1.unwrap();
    }
    let asked = hotkey.clone();
    let until = h.state.db.run(move |conn| budgets::locked_until(conn, &asked, "crawl", desearch::time::now())).await.unwrap().expect("locked out");
    assert!(until - desearch::time::now() > 47.0 * 3600.0);
    assert_eq!(credits(&h, hotkey).await, 0);
}

#[tokio::test]
async fn a_checked_upload_that_overstates_its_report_fails() {
    let h = started(3, 1.0).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    let verdict = judged(&h, &h.validator, "pass", None, score("pass", 3, "ok", Some("errors_confirmed"), "")).await.1.unwrap();
    assert_eq!(verdict["verdict"], "fail");
    assert_eq!(h.view(&task).await["score"]["reason"], lifecycle::REPORTED_ROWS);
}

#[tokio::test]
async fn every_upload_of_a_locked_out_hotkey_is_checked() {
    let h = started(3, 0.0).await;
    let hotkey = h.miner.ss58();
    established(&h, hotkey.clone()).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), parquet(&task)).await;
    h.state.db.run(move |conn| budgets::lock_out(conn, &hotkey, "crawl", 1.0, "test", desearch::time::now())).await.unwrap();
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0})).await.unwrap();
    assert_eq!(settled(&h, &task_id(&task)).await, lifecycle::LOCKED);
}

#[tokio::test]
async fn room_counts_the_queue_and_a_batch_sent_twice_is_queued_once() {
    let h = Harness::with(|s| s.queue_target = 10).await;
    let body = json!({"urls": urls(2), "batch_id": "b-1"});
    let first = h.admin.post("/v1/admin/enqueue", body.clone()).await.unwrap();
    let again = h.admin.post("/v1/admin/enqueue", body).await.unwrap();
    assert_eq!(first, again);
    assert_eq!(h.state.db.run(|conn| task_api::roundstore::unrevealed(conn, i64::MAX, 1000)).await.unwrap().len(), 1);
    let room = h.public("/v1/room").await.1;
    assert_eq!(room["room_tasks"].as_i64().unwrap(), 10 - room["queue"].as_i64().unwrap() - room["unrevealed"].as_i64().unwrap());
    assert_eq!((room["unrevealed"].clone(), room["refusing"].clone()), (json!(1), json!(false)));
}

fn outcome_rows(file: &[u8]) -> Vec<(String, String)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(file)).unwrap().build().unwrap();
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let (urls, outcomes) = (batch.column(0).as_string::<i32>(), batch.column(2).as_string::<i32>());
        rows.extend((0..batch.num_rows()).map(|i| (urls.value(i).to_string(), outcomes.value(i).to_string())));
    }
    rows
}

#[tokio::test]
async fn a_task_dropped_after_its_last_attempt_reaches_the_outcome_feed() {
    let h = started(3, 1.0).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let job =
        json!({"kind": "crawl", "attempts": h.state.settings.max_attempts - 1, "round_id": task["round_id"], "miner": h.miner.ss58(), "urls": task["urls"]});
    lifecycle::requeue(&h.state, &task_id(&task), &job, "fail").await.unwrap();
    assert_eq!(h.json_object("outcomes/latest.json"), Some(json!({"seq": 1})));
    let index = h.json_object(&desearch::outcomes::seq_key("outcomes", 1)).unwrap();
    assert_eq!(index["rows"], 3);
    let rows = outcome_rows(&h.object(index["key"].as_str().unwrap()).unwrap());
    let mut sent: Vec<String> = task["urls"].as_array().unwrap().iter().map(|u| u.as_str().unwrap().to_string()).collect();
    let mut dropped: Vec<String> = rows.iter().map(|(url, _)| url.clone()).collect();
    sent.sort();
    dropped.sort();
    assert_eq!(dropped, sent);
    assert!(rows.iter().all(|(_, outcome)| outcome == "dropped"));
}

#[tokio::test]
async fn every_completed_upload_is_logged_signed_for_validators() {
    let h = started(2, 1.0).await;
    let first = task_id(&h.mine(&h.miner, 0).await);
    let second = task_id(&h.mine(&h.miner, 1).await);
    h.stub.state.fail("PUT", &format!("{BUCKET}/log/uploads/seq/"), 5);
    assert!(uploadlog::flush(&h.state, &h.state.storage).await.is_err());
    h.mine(&h.rival, 0).await;
    assert_eq!(uploadlog::flush(&h.state, &h.state.storage).await.unwrap(), Some(1));
    let again = uploadlog::flush(&h.state, &h.state.storage).await.unwrap();
    let written = h.json_object(&upload_log_key(1)).unwrap();
    let later = uploadlog::flush(&h.state, &h.state.storage).await.unwrap();
    assert_eq!(h.json_object(UPLOAD_LOG_LATEST), Some(json!({"seq": 2})));
    let signature: Vec<u8> = (0..128).step_by(2).map(|i| u8::from_str_radix(&written["signature"].as_str().unwrap()[i..i + 2], 16).unwrap()).collect();
    assert!(desearch::hotkey::verify(&desearch::hotkey::public_of(&h.state.signer()).unwrap(), &log_payload(written.as_object().unwrap()), &signature));
    let entries: Vec<Value> = written["entries"].as_array().unwrap().iter().map(|e| e["task_id"].clone()).collect();
    assert_eq!(entries, [json!(first), json!(second)], "a failed write is retried with the same entries");
    assert_eq!((written["entries"][1]["ok"].clone(), written["entries"][1]["errors"].clone()), (json!(1), json!(1)));
    assert_eq!(again, Some(2), "the upload completed during the retry gets the next file");
    assert_eq!(later, None);
}

#[tokio::test]
async fn a_single_fail_takes_back_only_what_no_validator_checked() {
    let h = started(2, 0.0).await;
    let hotkey = h.miner.ss58();
    established(&h, hotkey.clone()).await;
    let unchecked = task_id(&h.mine(&h.miner, 0).await);
    settled(&h, &unchecked).await;
    start_recheck(&h, hotkey.clone()).await;
    let checked = task_id(&h.mine(&h.miner, 0).await);
    settled(&h, &checked).await;
    let mut job = h.state.validation.job(&h.redis, &checked).await.unwrap().unwrap();
    job.as_object_mut().unwrap().remove("picked");
    let _: () = h.redis.clone().set(format!("vjob:{checked}"), job.to_string()).await.unwrap();
    judged(&h, &h.validator, "pass", Some(&checked), json!({})).await.1.unwrap();
    assert_eq!(lifecycle::take_back(&h.state, &hotkey, 0.0, "test", false).await.unwrap(), [unchecked]);
    assert_eq!(lifecycle::take_back(&h.state, &hotkey, 0.0, "test", true).await.unwrap(), [checked], "a full penalty takes back the checked ones as well");
}

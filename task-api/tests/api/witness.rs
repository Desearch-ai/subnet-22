//! What block C of the trust design changes: upload keys nobody can tie to a miner before its completion, receipts
//! that name the signed request they answer, uploads pinned by version so a later write can't fail their miner,
//! and a storage canary whose outages can cost no miner anything.

use desearch::r2::Pin;
use desearch::time::now;
use reqwest::StatusCode;
use serde_json::{json, Value};
use task_api::queues::CLAIMS;
use task_api::{canary, lifecycle, manifest, uploadlog};

use crate::harness::{is_random_upload_key, parquet, task_id, urls, Harness, BUCKET};

async fn strikes(h: &Harness) -> i64 {
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap()
}

/// Uploads `body` through the claim's link and returns the upload's answer: its entity tag and version.
async fn put(h: &Harness, url: &str, body: Vec<u8>) -> (String, Option<String>) {
    let response = h.http.put(url).header("content-type", "application/vnd.apache.parquet").body(body).send().await.expect("R2 answers");
    assert!(response.status().is_success(), "upload: {}", response.status());
    let header = |name: &str| response.headers().get(name).and_then(|v| v.to_str().ok()).map(String::from);
    (header("etag").unwrap_or_default(), header("x-amz-version-id"))
}

fn report(task: &Value, extra: Value) -> Value {
    let rows = task["urls"].as_array().map_or(0, Vec::len);
    let mut report = json!({"key": task["upload"]["key"], "rows": rows, "ok": rows, "errors": 0, "bytes": 10});
    for (name, value) in extra.as_object().cloned().unwrap_or_default() {
        report[name] = value;
    }
    report
}

#[tokio::test]
async fn an_upload_key_names_no_miner_or_task_and_is_public_only_once_completed() {
    let h = Harness::start().await;
    let enqueued = h.enqueue(urls(6)).await;
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    let first = h.miner.claim().await["tasks"][0].clone();
    let second = h.rival.claim().await["tasks"][0].clone();
    let (key, other) = (first["upload"]["key"].as_str().unwrap().to_string(), second["upload"]["key"].as_str().unwrap().to_string());
    assert!(is_random_upload_key(&key, ".parquet") && is_random_upload_key(&other, ".parquet") && key != other);

    let public = [
        "/v1/live".to_string(),
        "/v1/events".to_string(),
        format!("/v1/rounds/{round_id}/log"),
        format!("/v1/tasks/{}", task_id(&first)),
        format!("/v1/miners/{}", h.miner.ss58()),
    ];
    for path in &public {
        let (status, shown) = h.public(path).await;
        assert!(status.is_success(), "{path}: {status}");
        assert!(!shown.to_string().contains(&key), "{path} shows the upload key before its completion");
    }
    let body = parquet(&first);
    put(&h, first["upload"]["url"].as_str().unwrap(), body).await;
    h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&first)), report(&first, json!({}))).await.unwrap();
    uploadlog::flush(&h.state, &h.state.storage).await.unwrap();
    let logged = h.json_object(&manifest::upload_log_key(1)).unwrap();
    assert_eq!(logged["entries"][0]["key"], key.as_str(), "the upload log names it once completed");
    assert!(!logged.to_string().contains(&other), "and still not the upload that was not");
}

#[tokio::test]
async fn every_receipt_names_the_signed_request_it_answers() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let (claimed, claim) = h.miner.post_hashed("/v1/tasks/claim", None).await;
    let claimed = claimed.unwrap();
    assert_eq!(claimed["receipts"][0]["body"]["request"], claim.as_str());
    let task = claimed["tasks"][0].clone();
    put(&h, task["upload"]["url"].as_str().unwrap(), parquet(&task)).await;
    let path = format!("/v1/tasks/{}/complete", task_id(&task));
    let (done, complete) = h.miner.post_hashed(&path, Some(report(&task, json!({})))).await;
    done.unwrap();
    assert_ne!(claim, complete);

    let (_, found) = h.public(&format!("/v1/events?request={complete}")).await;
    let events = found["events"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!((events[0]["outcome"].as_str(), events[0]["task_id"].as_str()), (Some("completed"), Some(task_id(&task).as_str())));
    let round = events[0]["round_id"].as_str().unwrap();
    let (_, log) = h.public(&format!("/v1/rounds/{round}/log")).await;
    let requests: Vec<&str> = log["entries"].as_array().unwrap().iter().filter_map(|e| e["request"].as_str()).collect();
    assert!(requests.contains(&claim.as_str()) && requests.contains(&complete.as_str()), "the signed round log holds both");

    // A refusal names its request too: the rival takes the last task, then asks for more than there is.
    h.rival.claim().await["tasks"][0].as_object().expect("the last task");
    let (refused, asked) = h.rival.post_hashed("/v1/tasks/claim", None).await;
    let refused = refused.unwrap();
    assert!(refused["refusal"]["code"].is_string(), "{refused}");
    assert_eq!(refused["receipt"]["body"]["request"], asked.as_str());
}

#[tokio::test]
async fn on_a_versioned_bucket_the_completion_pins_the_miners_own_version_and_a_later_write_changes_nothing() {
    let h = Harness::start().await;
    h.stub.state.set_versioned(true);
    h.stub.state.set_locked(true);
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let (id, key, url) = (task_id(&task), task["upload"]["key"].as_str().unwrap().to_string(), task["upload"]["url"].as_str().unwrap().to_string());
    let (_, mine) = put(&h, &url, parquet(&task)).await;
    let mine = mine.expect("a versioned bucket names the version");

    // Anyone else holding the link (a relay, the bucket's owner) writes over the key before the completion.
    let (_, theirs) = put(&h, &url, b"PAR1 junk that fails every check PAR1".to_vec()).await;
    assert_ne!(theirs.as_deref(), Some(mine.as_str()));
    h.miner.post(&format!("/v1/tasks/{id}/complete"), report(&task, json!({"version": mine}))).await.unwrap();

    let job = h.state.validation.job(&h.redis, &id).await.unwrap().unwrap();
    assert_eq!(job["version"], mine.as_str(), "the miner's version is pinned, not the newer one");
    assert_eq!(job["manifest"]["version"], mine.as_str(), "and the signed note names it");
    let signer = h.state.signer();
    let note = job["manifest"].as_object().unwrap();
    let signature: Vec<u8> = (0..note["signature"].as_str().unwrap().len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&note["signature"].as_str().unwrap()[i..i + 2], 16).unwrap())
        .collect();
    assert!(desearch::hotkey::verify(&desearch::hotkey::public_of(&signer).unwrap(), &manifest::payload(note), &signature));

    // After the completion, the owner writes over it again: readers by version still get the miner's file.
    put(&h, &url, b"PAR1 more junk PAR1".to_vec()).await;
    let size = parquet(&task).len() as u64;
    let read = h.state.storage.get_range_at(&key, 0..size, Pin::Version(&mine)).await.unwrap();
    assert_eq!(read.as_ref(), parquet(&task).as_slice());
    assert_ne!(h.object(&key), Some(parquet(&task)), "the key's current object is someone else's");
    let deleted = h.http.delete(format!("http://{}/{BUCKET}/{key}?versionId={mine}", h.stub.addr)).send().await.unwrap();
    assert_eq!(deleted.status(), StatusCode::FORBIDDEN, "Object Lock keeps the pinned version");

    // A validator that says the upload is missing is answered by version: it is there.
    h.opened(None, Some(&id)).await;
    let released = h.validator.post(&format!("/v1/validation/{id}/release"), json!({"reason": "missing"})).await.unwrap();
    assert_eq!(released["status"], "open");

    uploadlog::flush(&h.state, &h.state.storage).await.unwrap();
    let logged = h.json_object(&manifest::upload_log_key(1)).unwrap();
    assert_eq!(logged["entries"][0]["version"], mine.as_str(), "validators see which version to look at");
}

#[tokio::test]
async fn without_versions_the_miners_entity_tag_refuses_an_upload_someone_else_swapped_in() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let (id, url) = (task_id(&task), task["upload"]["url"].as_str().unwrap().to_string());
    let (mine, _) = put(&h, &url, parquet(&task)).await;
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    put(&h, &url, b"PAR1 swapped PAR1".to_vec()).await;
    let path = format!("/v1/tasks/{id}/complete");
    let refused = h.miner.post(&path, report(&task, json!({"etag": mine}))).await.unwrap_err();
    assert_eq!(refused.status, 409, "{:?}", refused.detail);

    // The miner still holds its claim and link: it uploads again and completes.
    let (again, _) = put(&h, &url, parquet(&task)).await;
    h.miner.post(&path, report(&task, json!({"etag": again}))).await.unwrap();
    let job = h.state.validation.job(&h.redis, &id).await.unwrap().unwrap();
    assert_eq!(job["etag"], again.as_str());
    assert!(job.get("version").is_none_or(Value::is_null), "an unversioned bucket names no version");
    assert!(job["manifest"].get("version").is_none(), "so its note signs the same bytes as before");
}

async fn claimed_then_storage_fails(h: &Harness) -> String {
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    assert!(canary::run(&h.state).await.unwrap().failed.is_none(), "storage works at first");
    h.stub.state.fail("PUT", &format!("{BUCKET}/uploads/"), 1);
    let probe = canary::run(&h.state).await.unwrap();
    assert_eq!(probe.failed.as_ref().map(|(step, _)| *step), Some("put"));
    let id = task_id(&task);
    let _: i64 = redis::cmd("ZADD").arg(CLAIMS).arg("XX").arg(now()).arg(&id).query_async(&mut h.redis.clone()).await.unwrap();
    lifecycle::reclaim_expired(&h.state, now() + 1.0).await.unwrap();
    id
}

async fn reclaim_cause(h: &Harness, id: &str) -> String {
    let (_, found) = h.public(&format!("/v1/events?task_id={id}&outcome=reclaimed")).await;
    found["events"][0]["cause"].as_str().unwrap_or_default().to_string()
}

#[tokio::test]
async fn a_claim_that_lapsed_while_the_canary_found_storage_down_costs_nothing_when_the_rule_is_on() {
    let h = Harness::with(|s| s.storage_outage_forgives = true).await;
    let id = claimed_then_storage_fails(&h).await;
    assert_eq!(reclaim_cause(&h, &id).await, canary::OUTAGE_CAUSE);
    assert_eq!(strikes(&h).await, 0);

    let (status, probes) = h.public("/v1/storage/probes").await;
    assert!(status.is_success());
    assert_eq!(probes["probes"].as_array().unwrap().len(), 2);
    let outages = probes["outages"].as_array().unwrap();
    assert_eq!(outages.len(), 1, "public: {probes}");
    assert!(outages[0]["to_at"].is_null(), "still down as far as the canary knows");
}

#[tokio::test]
async fn without_the_rule_the_canary_only_records_the_outage() {
    let h = Harness::start().await;
    let id = claimed_then_storage_fails(&h).await;
    assert_eq!(reclaim_cause(&h, &id).await, "expired");
    assert_eq!(strikes(&h).await, 1);
}

#[tokio::test]
async fn a_probe_whose_read_back_fails_marks_an_outage_until_the_next_one_passes() {
    let h = Harness::start().await;
    h.stub.state.fail("GET", &format!("{BUCKET}/uploads/"), 1);
    let probe = canary::run(&h.state).await.unwrap();
    assert_eq!(probe.failed.as_ref().map(|(step, _)| *step), Some("get"));
    assert!(h.stub.state.keys(BUCKET, "uploads/").is_empty(), "a probe deletes what it wrote");
    let ok = canary::run(&h.state).await.unwrap();
    assert!(ok.failed.is_none());
    let (_, probes) = h.public("/v1/storage/probes").await;
    assert_eq!(probes["outages"][0]["to_at"].as_f64(), Some(ok.at), "the outage ends at the next probe that passes");
}

#[tokio::test]
async fn with_the_upload_lock_on_the_pinned_version_is_locked_for_the_check_window() {
    let h = Harness::with(|s| s.upload_lock_s = 86_400.0).await;
    h.stub.state.set_versioned(true);
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let (id, key, url) = (task_id(&task), task["upload"]["key"].as_str().unwrap().to_string(), task["upload"]["url"].as_str().unwrap().to_string());
    let (_, mine) = put(&h, &url, parquet(&task)).await;
    let mine = mine.unwrap();
    h.miner.post(&format!("/v1/tasks/{id}/complete"), report(&task, json!({"version": mine}))).await.unwrap();

    let seen = h.http.head(format!("http://{}/{BUCKET}/{key}?versionId={mine}", h.stub.addr)).send().await.unwrap();
    let header = |name: &str| seen.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
    assert_eq!(header("x-amz-object-lock-mode"), "COMPLIANCE");
    let until = header("x-amz-object-lock-retain-until-date");
    assert!(until.as_str() > desearch::time::utc_day(now() + 86_000.0).as_str(), "locked about a day: {until}");
    let deleted = h.http.delete(format!("http://{}/{BUCKET}/{key}?versionId={mine}", h.stub.addr)).send().await.unwrap();
    assert_eq!(deleted.status(), StatusCode::FORBIDDEN);
}

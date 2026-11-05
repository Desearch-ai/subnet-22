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
async fn a_claim_that_lapsed_while_the_canary_found_storage_down_costs_nothing() {
    let h = Harness::start().await;
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

/// Publishes the log and replays every file with the API's rules; the divergences it found.
async fn replay_divergences(h: &Harness) -> Vec<String> {
    use task_api::journal;
    journal::log_requests(&h.state).await.unwrap();
    while journal::publish(&h.state).await.unwrap().is_some() {}
    let latest = h.json_object(journal::LATEST_KEY).unwrap();
    let mut replay = task_api::replay::Replay::open(None).unwrap();
    let blobs = |sha: &str| h.object(&journal::blob_key(sha));
    for n in 1..=latest["segment"].as_i64().unwrap() {
        let file = h.json_object(&journal::segment_key(n)).unwrap();
        replay.apply_segment(&file, &[h.state.signer()], false, &blobs).unwrap().unwrap();
    }
    replay.divergences.iter().map(|d| format!("{d:?}")).collect()
}

fn sha256(bytes: &[u8]) -> String {
    task_api::proofs::sha256(bytes)
}

fn signed_by_api(h: &Harness, body: &Value) -> bool {
    let map = body.as_object().unwrap();
    let hex = map["signature"].as_str().unwrap();
    let signature: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
    desearch::hotkey::verify(&desearch::hotkey::public_of(&h.state.signer()).unwrap(), &manifest::log_payload(map), &signature)
}

#[tokio::test]
async fn the_upload_log_is_a_chain_each_file_naming_the_hash_of_the_one_before() {
    let h = Harness::start().await;
    h.enqueue(urls(30)).await;
    h.mine(&h.miner, 0).await;
    assert_eq!(uploadlog::flush(&h.state, &h.state.storage).await.unwrap(), Some(1));
    let first = h.object(&manifest::upload_log_key(1)).unwrap();
    let parsed: Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(parsed["prev"], task_api::journal::GENESIS);

    // A write that fails goes up again byte for byte: a reader may already hold the hash of what it wrote.
    h.mine(&h.miner, 0).await;
    // Every attempt the bucket client makes fails.
    h.stub.state.fail("PUT", &format!("{BUCKET}/{}", manifest::UPLOAD_LOG_LATEST), 5);
    assert!(uploadlog::flush(&h.state, &h.state.storage).await.is_err());
    let tried = h.object(&manifest::upload_log_key(2)).unwrap();
    h.mine(&h.rival, 0).await;
    assert_eq!(uploadlog::flush(&h.state, &h.state.storage).await.unwrap(), Some(2));
    let second = h.object(&manifest::upload_log_key(2)).unwrap();
    assert_eq!(second, tried, "the retry wrote the same bytes");
    let parsed: Value = serde_json::from_slice(&second).unwrap();
    assert_eq!(parsed["prev"], sha256(&first).as_str());
    assert_eq!(parsed["prev_written_at"], serde_json::from_slice::<Value>(&first).unwrap()["written_at"]);
    assert!(signed_by_api(&h, &parsed), "the link is signed with the file");
    let latest = h.json_object(manifest::UPLOAD_LOG_LATEST).unwrap();
    assert_eq!((latest["seq"].clone(), latest["head"].clone()), (json!(2), json!(sha256(&second))));
    assert!(signed_by_api(&h, &latest), "and so is the newest file's hash");

    // Redis loses the chain's place: the next file goes on from the bucket instead of writing over file 1.
    let _: () = redis::cmd("DEL").arg(uploadlog::NEXT).arg(uploadlog::HEAD).query_async(&mut h.redis.clone()).await.unwrap();
    assert_eq!(uploadlog::flush(&h.state, &h.state.storage).await.unwrap(), Some(3), "the upload the retry left waiting");
    let third: Value = serde_json::from_slice(&h.object(&manifest::upload_log_key(3)).unwrap()).unwrap();
    assert_eq!(third["prev"], sha256(&second).as_str());
    assert_eq!(h.object(&manifest::upload_log_key(1)).unwrap(), first, "file 1 is untouched");
}

async fn journal_rows(h: &Harness, kind: &str) -> Vec<Value> {
    let kind = kind.to_string();
    h.state
        .db
        .run(move |conn| {
            let mut statement = conn.prepare("SELECT entry FROM journal WHERE kind = ? ORDER BY seq")?;
            let rows = statement.query_map([kind], |row| row.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows.into_iter().map(|text| serde_json::from_str(&text).unwrap()).collect())
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn a_lock_storage_refuses_fails_the_completion_closed_and_opens_an_outage_until_one_holds() {
    let h = Harness::with(|s| s.upload_lock_s = 86_400.0).await;
    h.stub.state.set_versioned(true);
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let (id, key, url) = (task_id(&task), task["upload"]["key"].as_str().unwrap().to_string(), task["upload"]["url"].as_str().unwrap().to_string());
    let (_, mine) = put(&h, &url, parquet(&task)).await;
    let path = format!("/v1/tasks/{id}/complete");

    // Every attempt the bucket client makes to lock the version is refused.
    h.stub.state.fail("PUT", &format!("{BUCKET}/{key}"), 5);
    let refused = h.miner.post(&path, report(&task, json!({"version": mine}))).await.unwrap_err();
    assert_eq!(refused.status, 503, "{:?}", refused.detail);
    assert!(h.state.validation.job(&h.redis, &id).await.unwrap().is_none(), "nothing unlocked was pinned");
    let (_, outages) = h.public("/v1/outages").await;
    let open: Vec<&Value> = outages["outages"].as_array().unwrap().iter().filter(|o| o["cause"] == "upload_lock").collect();
    assert_eq!(open.len(), 1, "{outages}");
    assert!(open[0]["end"].is_null(), "open until a lock holds");
    let logged = journal_rows(&h, "outage").await;
    assert_eq!((logged[0]["input"]["event"].clone(), logged[0]["input"]["cause"].clone()), (json!("begin"), json!("upload_lock")));

    // The miner completes again; the lock holds, the completion is recorded locked and the outage ends.
    h.miner.post(&path, report(&task, json!({"version": mine}))).await.unwrap();
    let complete = journal_rows(&h, "complete").await.into_iter().find(|e| e["output"].get("refused").is_none()).unwrap();
    assert_eq!(complete["input"]["lock"]["locked"], true);
    let (_, outages) = h.public("/v1/outages").await;
    assert!(outages["outages"].as_array().unwrap().iter().filter(|o| o["cause"] == "upload_lock").all(|o| !o["end"].is_null()));

    // A claim that lapsed while locks failed costs its miner nothing.
    let other = h.rival.claim().await["tasks"][0].clone();
    let (_, theirs) = put(&h, other["upload"]["url"].as_str().unwrap(), parquet(&other)).await;
    h.stub.state.fail("PUT", &format!("{BUCKET}/{}", other["upload"]["key"].as_str().unwrap()), 5);
    let other_path = format!("/v1/tasks/{}/complete", task_id(&other));
    assert_eq!(h.rival.post(&other_path, report(&other, json!({"version": theirs}))).await.unwrap_err().status, 503);
    let _: i64 = redis::cmd("ZADD").arg(CLAIMS).arg("XX").arg(now()).arg(task_id(&other)).query_async(&mut h.redis.clone()).await.unwrap();
    lifecycle::reclaim_expired(&h.state, now() + 1.0).await.unwrap();
    assert_eq!(reclaim_cause(&h, &task_id(&other)).await, "upload_lock_outage");
}

#[tokio::test]
async fn every_answer_to_a_miners_write_carries_its_receipt_or_its_refusal_is_logged() {
    let h = Harness::start().await;
    h.enqueue(urls(30)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    put(&h, task["upload"]["url"].as_str().unwrap(), parquet(&task)).await;
    let path = format!("/v1/tasks/{}/complete", task_id(&task));
    let (done, hash) = h.miner.post_hashed(&path, Some(report(&task, json!({})))).await;
    let receipt = done.unwrap()["receipt"].clone();
    assert_eq!((receipt["body"]["request"].as_str(), receipt["body"]["outcome"].as_str()), (Some(hash.as_str()), Some("completed")));
    assert!(receipt["body"]["block"].is_i64(), "with the block it was judged at");
    let signature: Vec<u8> = (0..128).step_by(2).map(|i| u8::from_str_radix(&receipt["signature"].as_str().unwrap()[i..i + 2], 16).unwrap()).collect();
    let body = task_api::py::canonical(&receipt["body"]);
    assert!(desearch::hotkey::verify(&desearch::hotkey::public_of(&h.state.signer()).unwrap(), &body, &signature));
    // Asked again, the API answers with the first completion's receipt, which names the first request.
    let (again, _) = h.miner.post_hashed(&path, Some(report(&task, json!({})))).await;
    assert_eq!(again.unwrap()["receipt"], receipt);

    let held = h.miner.claim().await["tasks"][0].clone();
    let (abandoned, asked) = h.miner.post_hashed(&format!("/v1/tasks/{}/abandon", task_id(&held)), Some(json!({}))).await;
    assert_eq!(abandoned.unwrap()["receipts"][0]["body"]["request"], asked.as_str());

    // Refusals before the API looks at the claim are logged too, with the block they were judged at.
    let (refused, not_held) = h.rival.post_hashed(&format!("/v1/tasks/{}/complete", task_id(&held)), Some(report(&held, json!({})))).await;
    assert_eq!(refused.unwrap_err().status, 409);
    let (malformed, bad_body) = h.rival.post_hashed(&format!("/v1/tasks/{}/complete", task_id(&held)), Some(json!({"rows": "many"}))).await;
    assert_eq!(malformed.unwrap_err().status, 422);
    let (refused_abandon, not_theirs) = h.rival.post_hashed(&format!("/v1/tasks/{}/abandon", task_id(&task)), Some(json!({}))).await;
    assert_eq!(refused_abandon.unwrap_err().status, 409);
    let refusals: Vec<Value> = [journal_rows(&h, "complete").await, journal_rows(&h, "abandon").await].concat();
    for (hash, status) in [(&not_held, 409), (&bad_body, 422), (&not_theirs, 409)] {
        let entry = refusals.iter().find(|e| e["input"]["request"]["request"] == hash.as_str()).unwrap_or_else(|| panic!("{hash} is logged"));
        assert_eq!(entry["output"]["refused"]["status"], status);
        assert!(entry["input"]["block"].is_i64());
    }

    // Past the cap a hotkey's refusals are answered 429 and not logged, so nobody grows the log at will.
    let before = journal_rows(&h, "complete").await.len();
    let mut statuses = Vec::new();
    for _ in 0..task_api::state::REFUSALS_LOGGED_PER_MINUTE {
        statuses.push(h.rival.post(&format!("/v1/tasks/{}/complete", task_id(&held)), report(&held, json!({}))).await.unwrap_err().status);
    }
    assert_eq!(statuses.last(), Some(&429));
    assert!(journal_rows(&h, "complete").await.len() - before < statuses.len());
    assert_eq!(replay_divergences(&h).await, Vec::<String>::new(), "refusals replay as changing nothing");
}

async fn witnessed(validator: &crate::harness::Client, record: Value) -> Result<Value, crate::harness::Refused> {
    validator.post("/v1/storage/witness", json!({"record": record})).await
}

fn failed_probe(validator: &crate::harness::Client) -> Value {
    json!({"kind": "probe", "version": 2, "witness": validator.ss58(), "ok": false, "step": "put", "detail": "HTTP 503", "block": 1, "at": now()})
}

async fn lapsed(h: &Harness, miner: &crate::harness::Client) -> String {
    let task = miner.claim().await["tasks"][0].clone();
    let id = task_id(&task);
    let _: i64 = redis::cmd("ZADD").arg(CLAIMS).arg("XX").arg(now()).arg(&id).query_async(&mut h.redis.clone()).await.unwrap();
    lifecycle::reclaim_expired(&h.state, now() + 1.0).await.unwrap();
    id
}

#[tokio::test]
async fn a_quorum_of_validators_storage_witnesses_forgives_a_lapse_the_apis_own_probes_did_not_see() {
    let h = Harness::start().await;
    h.enqueue(urls(30)).await;
    assert_eq!(witnessed(&h.miner, failed_probe(&h.miner)).await.unwrap_err().status, 403, "validators only");
    let passing = json!({"kind": "probe", "witness": h.validator.ss58(), "ok": true});
    assert_eq!(witnessed(&h.validator, passing).await.unwrap_err().status, 422, "only failures are reported");
    let someone_elses = failed_probe(&h.other_validator);
    assert_eq!(witnessed(&h.validator, someone_elses).await.unwrap_err().status, 422, "a validator reports what it saw itself");

    // One validator's word is not enough.
    witnessed(&h.validator, failed_probe(&h.validator)).await.unwrap();
    let id = lapsed(&h, &h.miner).await;
    assert_eq!(reclaim_cause(&h, &id).await, "expired");
    assert_eq!(strikes(&h).await, 1);

    // Two validators' are, though the API's own canary and probes saw storage working.
    let head = json!({"kind": "storage", "witness": h.other_validator.ss58(), "key": "uploads/x.parquet", "status": null, "error": "TimeoutError"});
    let answer = witnessed(&h.other_validator, head).await.unwrap();
    assert!(answer["seq"].is_i64() && answer["block"].is_i64());
    let id = lapsed(&h, &h.miner).await;
    assert_eq!(reclaim_cause(&h, &id).await, "storage_witnessed_outage");
    assert_eq!(strikes(&h).await, 1, "no strike for it");
    let (_, outages) = h.public("/v1/outages").await;
    assert_eq!(outages["witnessed"][0]["validators"], 2, "{outages}");
    assert_eq!(outages["outages"].as_array().unwrap().len(), 0, "the API itself recorded none");

    let logged = journal_rows(&h, "witness").await;
    let accepted: Vec<&Value> = logged.iter().filter(|e| e["output"].get("refused").is_none()).collect();
    assert_eq!(accepted.len(), 2);
    assert_eq!(logged.len(), 4, "the refused reports are logged too");
    assert_eq!(replay_divergences(&h).await, Vec::<String>::new(), "a replay forgives the same lapse on the logged witnesses");
}

#[tokio::test]
async fn a_validator_gets_probe_links_storage_cannot_tell_from_an_upload() {
    let h = Harness::start().await;
    let refused = h.miner.post("/v1/storage/probe-link", json!({})).await.unwrap_err();
    assert_eq!(refused.status, 403);
    let link = h.validator.post("/v1/storage/probe-link", json!({})).await.unwrap();
    let key = link["key"].as_str().unwrap();
    assert!(is_random_upload_key(key, ".parquet"), "{key}");
    let body = b"PAR1 probe PAR1".to_vec();
    let sent = h.http.put(link["put_url"].as_str().unwrap()).header("content-type", link["content_type"].as_str().unwrap()).body(body.clone()).send().await.unwrap();
    assert!(sent.status().is_success());
    let read = h.http.get(link["get_url"].as_str().unwrap()).send().await.unwrap().bytes().await.unwrap();
    assert_eq!(read.as_ref(), body.as_slice());
    let deleted = h.http.delete(link["delete_url"].as_str().unwrap()).send().await.unwrap();
    assert!(deleted.status().is_success(), "{}", deleted.status());
    assert_eq!(h.object(key), None);
}

//! The task API end to end over HTTP, as miners, validators and the bot use it.

mod checks;
mod embed;
mod harness;
mod limits;
mod logs;
mod punishment;
mod queueing;
mod timing;
mod trust;

use desearch::hotkey::{public_of, verify};
use harness::{first_url, parquet, score, task_id, urls, Harness};
use serde_json::{json, Value};
use task_api::budgets::{self, LOCKOUT_STEPS_H};
use task_api::lifecycle;
use task_api::manifest;
use task_api::py::{canonical, round_to};
use task_api::queues;

fn receipt_verifies(receipt: &Value, signer: &str) -> bool {
    let signature: Vec<u8> = (0..receipt["signature"].as_str().unwrap_or_default().len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&receipt["signature"].as_str().unwrap()[i..i + 2], 16).unwrap())
        .collect();
    verify(&public_of(signer).unwrap(), &canonical(&receipt["body"]), &signature)
}

fn manifest_verifies(note: &Value, signer: &str) -> bool {
    let signature: Vec<u8> = hex(note["signature"].as_str().unwrap_or_default());
    verify(&public_of(signer).unwrap(), &manifest::payload(note.as_object().unwrap()), &signature)
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap()).collect()
}

#[tokio::test]
async fn a_crawl_task_goes_round_trip() {
    let h = Harness::start().await;
    let enqueue = json!({"urls": urls(6)});
    let anonymous = h.http.post(format!("{}/v1/admin/enqueue", h.url)).json(&enqueue).send().await.unwrap();
    assert_eq!(anonymous.status(), 401);
    assert_eq!(h.miner.post("/v1/admin/enqueue", enqueue.clone()).await.unwrap_err().status, 403);
    let enqueued = h.admin.post("/v1/admin/enqueue", enqueue).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    assert_eq!(enqueued["batches"], 2);
    let listed = h.public("/v1/rounds").await.1["rounds"].clone();
    assert_eq!(listed.as_array().unwrap().len(), 1);
    assert_eq!(
        (listed[0]["round_id"].as_str(), listed[0]["manifest_hash"].clone(), listed[0]["revealed"].clone()),
        (Some(round_id.as_str()), enqueued["manifest_hash"].clone(), json!(false))
    );
    assert_eq!(h.miner.claim().await["refusal"]["code"], "QUEUE_EMPTY");
    assert_eq!(h.revealed().await, 2);

    assert_eq!(h.validator.call(reqwest::Method::POST, "/v1/tasks/claim", None).await.unwrap_err().status, 403);
    let claimed = h.miner.claim().await;
    let task = claimed["tasks"][0].clone();
    let (id, upload) = (task_id(&task), task["upload"].clone());
    assert_eq!(task["round_id"], round_id.as_str());
    assert_eq!(task["urls"].as_array().unwrap().len(), 3);
    let key = upload["key"].as_str().unwrap();
    assert!(key.starts_with("uploads/dt=") && key.ends_with(".parquet") && key.contains(&format!("/task={id}/{}-", h.miner.ss58())), "{key}");
    assert_eq!(upload["content_type"], "application/vnd.apache.parquet");
    assert_eq!(upload["expires_at"], task["expires_at"]);
    let signer = h.public("/v1/key").await.1["signer"].as_str().unwrap().to_string();
    assert!(receipt_verifies(&claimed["receipts"][0], &signer));
    assert_eq!(h.status(&id).await, "claimed");

    let complete = format!("/v1/tasks/{id}/complete");
    let report = json!({"key": key, "rows": 3, "ok": 3, "errors": 0, "bytes": 0});
    assert_eq!(h.rival.post(&complete, report.clone()).await.unwrap_err().status, 409);
    let forged = key.replace(&h.miner.ss58(), &h.rival.ss58());
    assert_eq!(h.miner.post(&complete, json!({"key": forged, "rows": 3, "ok": 3})).await.unwrap_err().status, 400);
    assert_eq!(h.miner.post(&complete, report.clone()).await.unwrap_err().status, 422, "nothing uploaded yet");

    let body = parquet(&task);
    h.put(upload["url"].as_str().unwrap(), body.clone()).await;
    let done = h.miner.post(&complete, report.clone()).await.unwrap();
    assert_eq!(done, json!({"task_id": id, "status": "open_for_validation"}));
    assert_eq!(h.status(&id).await, "open");
    assert!(h.object(key).is_none(), "the miner's upload is frozen into a copy");
    assert_eq!(h.state.crawl.in_flight(&h.redis, &h.miner.ss58()).await.unwrap(), 0, "uploaded, so not crawling");
    assert_eq!(h.state.crawl.waiting(&h.redis, &h.miner.ss58()).await.unwrap(), 1, "but waiting for its verdict");

    let score_path = format!("/v1/validation/{id}/score");
    assert_eq!(h.miner.post(&score_path, score("pass", 3, "ok", None, &first_url(&task))).await.unwrap_err().status, 403);

    let job = h.opened(Some(&h.validator), None).await;
    assert_eq!(job["task_id"], id.as_str());
    assert_eq!(job["miner"], h.miner.ss58().as_str());
    assert!(job["deadline"].as_f64() > job["completed_at"].as_f64());
    let frozen = job["key"].as_str().unwrap();
    assert!(frozen.starts_with("submitted/dt=") && frozen.contains(&format!("/task={id}/{}-", h.miner.ss58())), "{frozen}");
    assert_eq!(job["urls"], task["urls"]);
    assert_eq!(h.object(frozen), Some(body.clone()));
    let listing = h.open_list().await;
    assert_eq!(listing["uploads"].as_array().unwrap().iter().map(|m| m["task_id"].clone()).collect::<Vec<_>>(), [json!(id)]);
    assert_eq!(listing["signer"], signer.as_str());
    assert!(manifest_verifies(&listing["uploads"][0], &signer));
    assert_eq!(h.json_object(&lifecycle::manifest_key(frozen)), Some(listing["uploads"][0].clone()), "the same note sits next to the upload");

    let mut wrong = score("pass", 3, "ok", None, &first_url(&task));
    wrong["matched"] = 0.into();
    assert_eq!(h.validator.post(&score_path, wrong).await.unwrap_err().status, 422);
    let mut too_long = score("pass", 3, "ok", None, &first_url(&task));
    too_long["urls"] = json!([{"url": "https://site0.example/page/0", "miner_snippet": "x".repeat(501)}]);
    assert_eq!(h.validator.post(&score_path, too_long).await.unwrap_err().status, 422);
    let mut detailed = score("pass", 3, "ok", None, &first_url(&task));
    detailed["credited"] = 999.into();
    detailed["urls"] = json!([
        {"url": "https://site0.example/page/0", "status": 200, "sampled": true, "miner_snippet": "Story"},
        {"url": "https://site1.example/page/1", "status": 404, "error": "http_4xx"},
    ]);
    let scored = h.validator.post(&score_path, detailed).await.unwrap();
    assert_eq!(scored, json!({"task_id": id, "verdict": "pass", "credited": 3, "miner_budget": 2}));
    assert_eq!(h.validator.post(&score_path, score("pass", 3, "ok", None, &first_url(&task))).await.unwrap_err().status, 409);
    let view = h.view(&id).await;
    assert_eq!(view["status"], "pass");
    let details: Vec<(Value, Value, Value)> =
        view["urls"].as_array().unwrap().iter().map(|u| (u["url"].clone(), u["error"].clone(), u["miner_snippet"].clone())).collect();
    assert_eq!(
        details,
        [(json!("https://site0.example/page/0"), Value::Null, json!("Story")), (json!("https://site1.example/page/1"), json!("http_4xx"), Value::Null)]
    );
    for query in [format!("miner={}", h.miner.ss58()), format!("validator={}", h.validator.ss58())] {
        let listed = h.public(&format!("/v1/tasks?{query}")).await.1["tasks"].clone();
        assert_eq!(listed.as_array().unwrap().iter().map(|t| t["task_id"].clone()).collect::<Vec<_>>(), [json!(id)]);
    }
    let summary = view["score"].clone();
    assert_eq!((summary["returned"].clone(), summary["credited"].clone(), summary["validator"].clone()), (json!(3), json!(3), json!(h.validator.ss58())));
    let written = h.page(summary["report_key"].as_str().unwrap()).expect("the report is in the pages bucket");
    assert!(written.get("urls").is_none() && written["verdict"] == "pass");
    assert!(h.object(summary["report_key"].as_str().unwrap()).is_none(), "and not in the uploads bucket");
    assert_eq!(h.state.publish.depth(&h.redis).await.unwrap(), 1);
    let packed: String = redis::AsyncCommands::get(&mut h.redis.clone(), format!("pjob:{id}")).await.unwrap();
    let publish = unpack(&packed);
    assert_eq!(
        (publish["task_id"].clone(), publish["validator"].clone(), publish["validators"].clone()),
        (json!(id), json!(h.validator.ss58()), json!([h.validator.ss58()]))
    );
    assert_eq!((publish["claim_ttl"].clone(), publish["skip"].clone(), publish["key"].clone()), (json!(180), json!([]), json!(frozen)));

    let second = h.miner.claim().await["tasks"][0].clone();
    let second_id = task_id(&second);
    let second_body = parquet(&second);
    let second_key = second["upload"]["key"].as_str().unwrap().to_string();
    h.put(second["upload"]["url"].as_str().unwrap(), second_body.clone()).await;
    let second_report = json!({"key": second_key, "rows": 3, "ok": 3, "errors": 0, "bytes": second_body.len()});
    h.put(second["upload"]["url"].as_str().unwrap(), b"PAR1".repeat(3).into_iter().chain(vec![0; 64]).collect()).await;
    assert_eq!(h.miner.post(&format!("/v1/tasks/{second_id}/complete"), second_report.clone()).await.unwrap_err().status, 422, "not framed as Parquet");
    h.put(second["upload"]["url"].as_str().unwrap(), second_body).await;
    h.miner.post(&format!("/v1/tasks/{second_id}/complete"), second_report).await.unwrap();
    let opened = h.opened(Some(&h.validator), None).await;
    assert_eq!(opened["task_id"], second_id.as_str());
    let second_score = format!("/v1/validation/{second_id}/score");
    let failed = h.validator.post(&second_score, score("fail", 3, "content_mismatch", None, &first_url(&second))).await.unwrap();
    assert_eq!(failed, json!({"task_id": second_id, "verdict": "fail", "credited": 0, "miner_budget": 1}));
    let view = h.view(&second_id).await;
    assert_eq!(view["status"], "queued", "a failed task goes back out for someone else");
    assert_eq!((view["score"]["verdict"].clone(), view["score"]["reason"].clone()), (json!("fail"), json!("content_mismatch")));
    assert!(h.object(view["score"]["upload_key"].as_str().unwrap()).is_none(), "a failed upload is deleted, so the next miner cannot resubmit it");

    let again = h.mine(&h.rival, 0).await;
    assert_eq!((task_id(&again), again["urls"].clone()), (second_id.clone(), second["urls"].clone()));
    h.opened(Some(&h.validator), None).await;
    h.validator.post(&second_score, score("pass", 3, "ok", None, &first_url(&second))).await.unwrap();
    let drained = h.miner.claim().await;
    assert_eq!((drained["tasks"].clone(), drained["refusal"]["code"].clone()), (json!([]), json!("QUEUE_EMPTY")));

    let health = h.public("/v1/health").await.1;
    assert_eq!(health["verdicts"], json!({"pass": 2, "fail": 1}));
    assert_eq!((health["queue_depth"]["crawl"].clone(), health["validation_depth"].clone()), (json!(0), json!(0)));
    assert_eq!(health["active_validators"], json!([h.validator.ss58()]));
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!(miner["verdicts"], json!({"pass": 1, "fail": 1}));
    assert_eq!((miner["pools"]["crawl"]["budget"].clone(), miner["pools"]["crawl"]["in_flight"].clone()), (json!(1), json!(0)));
    assert_eq!(miner["coverage"]["returned"], 6);

    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let published = h.public(&format!("/v1/rounds/{round_id}")).await.1;
    let log = h.public(&format!("/v1/rounds/{round_id}/log")).await.1;
    let leaves: Vec<Vec<u8>> = log["entries"].as_array().unwrap().iter().map(canonical).collect();
    assert_eq!(task_api::proofs::merkle_root(&leaves), log["anchor_root"].as_str().unwrap());
    assert_eq!(published["signer"], signer.as_str());
    assert_eq!(published["serve_order"].as_array().unwrap().len(), 2);
    let outcomes: Vec<Value> = log["entries"].as_array().unwrap().iter().map(|e| e["outcome"].clone()).collect();
    assert_eq!(
        outcomes,
        ["issued", "completed", "issued", "completed", "reclaimed", "issued", "completed"].map(Value::from),
        "a second QUEUE_EMPTY in the same minute is answered but not logged"
    );
    let reclaimed: Vec<Value> = log["entries"].as_array().unwrap().iter().filter(|e| e["outcome"] == "reclaimed").map(|e| e["cause"].clone()).collect();
    assert_eq!(reclaimed, [json!("fail")]);
    assert!(log["entries"].as_array().unwrap().iter().all(|e| e["seq"].as_i64() > Some(0)));
}

fn unpack(packed: &str) -> Value {
    use base64::Engine;
    use std::io::Read;
    let compressed = base64::engine::general_purpose::STANDARD.decode(packed.strip_prefix(queues::PACKED).unwrap()).unwrap();
    let mut json = String::new();
    flate2::read::ZlibDecoder::new(compressed.as_slice()).read_to_string(&mut json).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[tokio::test]
async fn an_excess_poll_is_refused_with_a_retry_after_and_not_logged() {
    let h = Harness::with(|s| s.poll_rate = 0.1).await;
    let mut answers = Vec::new();
    loop {
        let answer = h.miner.claim().await;
        let limited = answer["refusal"]["code"] == "RATE_LIMITED";
        answers.push(answer);
        if limited || answers.len() >= 5 {
            break;
        }
    }
    let limited = answers.pop().unwrap();
    assert_eq!(limited["refusal"]["code"], "RATE_LIMITED");
    let retry = limited["refusal"]["inputs"]["retry_after"].as_f64().unwrap();
    assert!(retry > 0.0 && retry <= 10.0);
    assert!(limited["receipt"].is_null(), "an excess poll is not signed into the log");
    assert!(answers.iter().all(|a| a["refusal"]["inputs"]["retry_after"] == 10.0));
    let entries = h.state.db.run(|conn| task_api::roundlog::entries(conn, "")).await.unwrap();
    assert_eq!(entries.len(), answers.len());
    let signer = h.state.signer();
    for (answer, entry) in answers.iter().zip(&entries) {
        assert!(receipt_verifies(&answer["receipt"], &signer));
        assert_eq!(answer["receipt"]["signature"], entry["receipt_sig"]);
    }
    let mut forged = answers[0]["receipt"].clone();
    forged["body"]["outcome"] = "issued".into();
    assert!(!receipt_verifies(&forged, &signer));
}

#[tokio::test]
async fn a_locked_out_miner_is_refused_until_the_lockout_ends() {
    let h = Harness::start().await;
    let hotkey = h.miner.ss58();
    let now = desearch::time::now();
    let until = h
        .state
        .db
        .run(move |conn| {
            budgets::strike(conn, &hotkey, "content_mismatch", "t1", 1, "crawl", now)?;
            budgets::strike(conn, &hotkey, "coverage", "t2", 2, "crawl", now)
        })
        .await
        .unwrap()
        .expect("two strikes lock out");
    let answer = h.miner.claim().await;
    assert_eq!(answer["refusal"]["code"], "LOCKED_OUT");
    assert!(!answer["receipt"].is_null());
    assert_eq!(answer["refusal"]["inputs"]["until"], round_to(until, 3));
    let first = LOCKOUT_STEPS_H[0] * 3600.0;
    let retry = answer["refusal"]["inputs"]["retry_after"].as_f64().unwrap();
    assert!(first - 60.0 < retry && retry <= first);
    let view = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!(view["pools"]["crawl"]["locked_until"], until);
    assert!(view["pools"]["embed"]["locked_until"].is_null());
    assert_eq!(h.rival.claim().await["refusal"]["code"], "QUEUE_EMPTY");
}

#[tokio::test]
async fn races_backlogs_storage_faults_and_voids() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let id = task_id(&task);
    let body = parquet(&task);
    h.put(task["upload"]["url"].as_str().unwrap(), body.clone()).await;
    let complete = format!("/v1/tasks/{id}/complete");
    let report = json!({"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0, "bytes": 0});
    let (one, two) = tokio::join!(h.miner.post(&complete, report.clone()), h.miner.post(&complete, report));
    assert_eq!(one.unwrap(), two.unwrap(), "the second completion gets the first one's answer");
    let job = h.state.validation.job(&h.redis, &id).await.unwrap().unwrap();
    assert_eq!(h.object(job["key"].as_str().unwrap()), Some(body));

    let backlogged = desearch::time::now() - 50_000.0;
    let _: i64 = redis::AsyncCommands::zadd(&mut h.redis.clone(), queues::VOPEN, "backlogged", backlogged).await.unwrap();
    assert_eq!(h.miner.claim().await["refusal"]["code"], "VALIDATION_BACKLOG");
    let _: i64 = redis::AsyncCommands::zrem(&mut h.redis.clone(), queues::VOPEN, "backlogged").await.unwrap();

    assert_eq!(h.opened(Some(&h.validator), None).await["task_id"], id.as_str());
    h.stub.state.fail("PUT", &format!("{}/{}reports/", harness::BUCKET, harness::PAGES_PREFIX), 5);
    let pending = h.validator.post(&format!("/v1/validation/{id}/score"), score("pass", 3, "ok", None, &first_url(&task))).await.unwrap();
    assert_eq!(pending["verdict"], "pending", "the vote is kept, the final verdict waits");
    assert_eq!(h.status(&id).await, "voting");
    assert_eq!(lifecycle::finalize_due(&h.state, desearch::time::now()).await.unwrap(), [id.as_str()]);
    assert_eq!(h.status(&id).await, "pass");
    assert_eq!(h.public("/v1/health").await.1["verdicts"]["pass"], 1);

    let lost = h.miner.claim().await["tasks"][0].clone();
    let lost_id = task_id(&lost);
    h.put(lost["upload"]["url"].as_str().unwrap(), parquet(&lost)).await;
    h.miner.post(&format!("/v1/tasks/{lost_id}/complete"), json!({"key": lost["upload"]["key"], "bytes": 1})).await.unwrap();
    h.opened(None, Some(&lost_id)).await;
    let job = h.state.validation.job(&h.redis, &lost_id).await.unwrap().unwrap();
    assert_eq!(job["picked"], lifecycle::UNREPORTED, "a report without counts is checked");
    h.state.storage.delete(job["key"].as_str().unwrap()).await.unwrap();
    let released = h.validator.post(&format!("/v1/validation/{lost_id}/release"), json!({"reason": "missing"})).await.unwrap();
    assert_eq!(released["status"], "void");
    let view = h.view(&lost_id).await;
    assert_eq!((view["status"].clone(), view["score"]["reason"].clone()), (json!("queued"), json!("upload_missing")));
    assert_eq!(h.page(view["score"]["report_key"].as_str().unwrap()).unwrap()["verdict"], "void");
    assert_eq!(h.public("/v1/health").await.1["verdicts"]["void"], 1);

    assert_eq!(h.miner.claim().await["tasks"], json!([]), "a miner never gets a task twice");
    let again = h.rival.claim().await["tasks"][0].clone();
    assert_eq!((task_id(&again), again["urls"].clone()), (lost_id, lost["urls"].clone()));
    let view = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!(view["pools"]["crawl"]["budget"], 2, "one pass grew it, the void did not");
}

#[tokio::test]
async fn public_reads_are_limited_per_address_and_listed_a_page_at_a_time() {
    let h = Harness::with(|s| {
        s.log_reads_per_minute = 4;
        s.reads_per_minute = 1;
    })
    .await;
    h.state
        .db
        .run(|conn| {
            for n in 0..7 {
                let job = json!({"round_id": "r", "miner": if n % 2 == 1 { "m1" } else { "m2" }, "key": "k", "urls": []});
                let result = json!({"verdict": "pass", "reason": "ok"});
                let mut report = task_api::validations::build_report(&format!("t{n}"), &job, "v", result.as_object().unwrap(), &[], 0.0);
                report.insert("scored_at".into(), (n as f64).into());
                task_api::validations::record(conn, &report, None)?;
            }
            Ok(())
        })
        .await
        .unwrap();
    let ids = |page: &Value| page["tasks"].as_array().unwrap().iter().map(|t| t["task_id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    let get = |path: String| {
        let http = h.http.clone();
        let url = h.url.clone();
        async move {
            let response = http.get(format!("{url}{path}")).header("CF-Connecting-IP", "1.1.1.1").send().await.unwrap();
            (response.status().as_u16(), response.json::<Value>().await.unwrap_or(Value::Null))
        }
    };
    let first = get("/v1/tasks?limit=3".into()).await.1;
    assert_eq!(ids(&first), ["t6", "t5", "t4"]);
    let second = get(format!("/v1/tasks?limit=3&before={}", first["next"])).await.1;
    assert_eq!(ids(&second), ["t3", "t2", "t1"]);
    let mine = get("/v1/tasks?miner=m1".into()).await.1;
    assert_eq!((ids(&mine), mine["next"].clone()), (vec!["t5".to_string(), "t3".into(), "t1".into()], Value::Null));
    assert_eq!(get("/v1/tasks?limit=101".into()).await.0, 422);
    let (status, headers) = h.public_from("/v1/tasks", "1.1.1.1").await;
    assert_eq!(status, 429);
    let wait: i64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!(wait > 0 && wait <= 60);
    assert_eq!(h.public_from("/v1/tasks", "2.2.2.2").await.0, 200);
    assert_eq!(h.public_from("/v1/shares", "1.1.1.1").await.0, 200, "its own budget");
    assert_eq!(h.public_from("/v1/shares", "1.1.1.1").await.0, 429);
}

#[tokio::test]
async fn an_oversized_upload_is_refused_and_removed() {
    let h = Harness::with(|s| s.max_upload = 10).await;
    h.enqueue(urls(3)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let key = task["upload"]["key"].as_str().unwrap().to_string();
    h.put(task["upload"]["url"].as_str().unwrap(), parquet(&task)).await;
    let refused = h.miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": key, "rows": 3, "ok": 3})).await.unwrap_err();
    assert_eq!(refused.status, 413);
    assert!(h.object(&key).is_none(), "an oversized upload is removed");
}

#[tokio::test]
async fn a_miner_reads_its_own_verdicts_and_no_one_elses() {
    let h = Harness::start().await;
    h.enqueue(urls(3)).await;
    let task = h.mine(&h.miner, 0).await;
    harness::judged(&h, &h.validator, "pass", None, json!({})).await.1.unwrap();
    let own = h.miner.get(&format!("/v1/miners/{}/verdicts?limit=5", h.miner.ss58())).await.unwrap();
    assert_eq!(own["tasks"].as_array().unwrap().iter().map(|t| t["task_id"].clone()).collect::<Vec<_>>(), [json!(task_id(&task))]);
    let refused = h.rival.get(&format!("/v1/miners/{}/verdicts", h.miner.ss58())).await.unwrap_err();
    assert_eq!((refused.status, refused.detail), (403, json!("only the miner itself may read this")));
}

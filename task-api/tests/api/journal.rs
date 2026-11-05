//! The log: everything the API decided goes into it chained and signed, and a replay of the published files reaches the same accounts;
//! an altered, dropped, forked or made-up entry shows.

use serde_json::{json, Value};
use task_api::journal::{self, Break};
use task_api::replay::{check_anchor, replay_dir, Replay, FRAUD};
use task_api::settings::AnchorSettings;
use task_api::{lifecycle, snapshot};

use crate::harness::{judged, objective_fail, parquet, task_id, urls, Client, Harness};

/// Publishes what the log holds and reads every file back, oldest first.
async fn published(h: &Harness) -> Vec<Value> {
    journal::log_requests(&h.state).await.unwrap();
    while journal::publish(&h.state).await.unwrap().is_some() {}
    let latest = h.json_object(journal::LATEST_KEY).expect("the newest file is named");
    assert!(journal::signed_by(&latest, &h.state.signer()));
    (1..=latest["segment"].as_i64().unwrap()).map(|n| h.json_object(&journal::segment_key(n)).expect("every file is there")).collect()
}

fn replayed(h: &Harness, files: &[Value]) -> (Replay, Result<(), Break>) {
    let mut replay = Replay::open(None).unwrap();
    let blobs = |sha: &str| h.object(&journal::blob_key(sha));
    for file in files {
        if let Err(broken) = replay.apply_segment(file, &[h.state.signer()], false, &blobs).unwrap() {
            return (replay, Err(broken));
        }
    }
    (replay, Ok(()))
}

fn entries(files: &[Value]) -> Vec<Value> {
    files.iter().flat_map(|f| f["entries"].as_array().cloned().unwrap_or_default()).collect()
}

/// The files as one would publish them after changing entries in place: links, heads and signatures all made again with the API's key.
fn rechained(h: &Harness, files: &[Value], edit: impl Fn(&mut Value)) -> Vec<Value> {
    let (mut last, mut prev) = (0, journal::GENESIS.to_string());
    let mut out = Vec::new();
    for file in files {
        let mut file = file.clone();
        file["prev"] = prev.clone().into();
        file["first"] = (last + 1).into();
        for entry in file["entries"].as_array_mut().unwrap() {
            edit(entry);
            last += 1;
            entry["seq"] = last.into();
            entry["prev"] = prev.clone().into();
            prev = journal::hash_of(entry);
        }
        file["last"] = last.into();
        file["head"] = prev.clone().into();
        let signature = h.state.sign(&journal::signed_payload(&file));
        file["signature"] = signature.into();
        out.push(file);
    }
    out
}

/// Mines until the upload is settled: checked by `validator` with `verdict` when drawn (a fail on a fact of the file, which the
/// rules charge), finalized on the miner's counts otherwise.
async fn settled(h: &Harness, miner: &Client, validator: &Client, verdict: &str) -> String {
    let task = h.mine(miner, 0).await;
    let id = task_id(&task);
    for _ in 0..100 {
        lifecycle::settle_seeded(&h.state).await.unwrap();
        let picked = h.state.validation.job(&h.redis, &id).await.unwrap().is_some_and(|job| !job["picked"].is_null());
        if picked {
            let facts = if verdict == "fail" { objective_fail() } else { json!({}) };
            judged(h, validator, verdict, Some(&id), facts).await.1.unwrap();
            if verdict == "fail" {
                // A fail is decided at the validation deadline.
                h.past_deadline().await;
            }
            return id;
        }
        if ["pass", "fail", "queued"].contains(&h.status(&id).await.as_str()) {
            return id;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }
    panic!("task {id} was never settled");
}

async fn busy_day(h: &Harness) {
    h.enqueue(urls(60)).await;
    for _ in 0..10 {
        settled(h, &h.miner, &h.validator, "pass").await;
    }
    // Ten checked passes: from here an upload is checked only when drawn.
    settled(h, &h.miner, &h.validator, "pass").await;
    settled(h, &h.miner, &h.validator, "fail").await;
    settled(h, &h.miner, &h.validator, "fail").await;

    let held = h.rival.claim().await["tasks"][0].clone();
    h.rival.post(&format!("/v1/tasks/{}/abandon", task_id(&held)), json!({})).await.unwrap();
    let lapsing = h.rival.claim().await["tasks"][0].clone();
    lifecycle::reclaim_expired(&h.state, desearch::time::now() + 1_000.0).await.unwrap();
    assert_eq!(h.status(&task_id(&lapsing)).await, "queued");

    // Two lapses more than five minutes apart are two strikes, so the rival is locked out now.
    assert_eq!(h.rival.claim().await["refusal"]["code"], "LOCKED_OUT");
    let third = h.client("//third-miner-test", &h.url);
    let late = third.claim().await["tasks"][0].clone();
    h.put(late["upload"]["url"].as_str().unwrap(), parquet(&late)).await;
    let wrong_key = json!({"key": "uploads/elsewhere.parquet", "rows": 3, "ok": 3});
    assert_eq!(third.post(&format!("/v1/tasks/{}/complete", task_id(&late)), wrong_key).await.unwrap_err().status, 400);
    journal::log_shares(&h.state).await.unwrap();
    h.state.db.run(|conn| snapshot::record(conn, desearch::time::now())).await.unwrap();
}

#[tokio::test]
async fn a_replay_of_the_published_log_reaches_the_api_accounts() {
    // A short validation window, so the fails of the busy day are decided at their real deadline and the log shows it.
    let h = Harness::with(|s| s.validation_ttl = 2.0).await;
    busy_day(&h).await;
    let files = published(&h).await;
    let all = entries(&files);
    let kinds: std::collections::BTreeSet<&str> = all.iter().filter_map(|e| e["kind"].as_str()).collect();
    for kind in
        ["start", "snapshot", "enqueue", "reveal", "claim", "complete", "settle", "report", "finalize", "close", "abandon", "lapse", "requests", "shares"]
    {
        assert!(kinds.contains(kind), "no {kind} entry in {kinds:?}");
    }
    // A crawl upload's accounts settle when its round closes.
    let settled: Vec<&Value> = all.iter().filter(|e| e["kind"] == "close").flat_map(|e| e["output"]["settled"].as_array().into_iter().flatten()).collect();
    assert!(settled.iter().any(|upload| upload["penalty"] == true), "two fails among the last ten bring the full penalty");
    assert!(all.iter().any(|e| e["kind"] == "complete" && e["output"]["refused"]["status"] == 400), "a refused completion is logged");

    let (replay, chained) = replayed(&h, &files);
    assert_eq!(chained, Ok(()));
    assert!(replay.started().unwrap());
    assert_eq!(replay.divergences, [], "the rules give every output the API logged");
    let snapshots = all.iter().filter(|e| e["kind"] == "snapshot").count();
    assert!(snapshots >= 2, "the last snapshot was checked against the replayed accounts");
    let (last, head) = replay.position().unwrap();
    assert_eq!((last, head.clone()), h.state.db.run(journal::head).await.unwrap());
    assert!(replay.blocks.iter().all(|b| b.hash == task_api::manifest::local_block_hash(b.block)), "every block hash used is the chain's");

    // A validator mirrors the files into a folder and replays them as they come, going on from its saved state.
    let dir = std::env::temp_dir().join(format!("journal-replay-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(dir.join("seq")).unwrap();
    std::fs::create_dir_all(dir.join("blobs")).unwrap();
    for sha in journal::blobs_named(&all) {
        std::fs::write(dir.join("blobs").join(&sha), h.object(&journal::blob_key(&sha)).unwrap()).unwrap();
    }
    let state = dir.join("replay.db");
    let half = files.len().div_ceil(2);
    for (at, file) in files.iter().enumerate() {
        std::fs::write(dir.join("seq").join(format!("{:012}.json", file["segment"].as_i64().unwrap())), file.to_string()).unwrap();
        if at + 1 == half || at + 1 == files.len() {
            let mut replay = Replay::open(Some(&state)).unwrap();
            assert!(replay_dir(&mut replay, &dir, &[h.state.signer()], false).unwrap().is_ok());
            assert_eq!(replay.divergences, []);
        }
    }
    let replay = Replay::open(Some(&state)).unwrap();
    assert_eq!(replay.position().unwrap(), (last, head.clone()));

    // The tool a validator runs, from scratch over the same folder, with the anchors its own node shows.
    let first_snapshot = all.iter().find(|e| e["kind"] == "snapshot").unwrap()["seq"].as_i64().unwrap();
    std::fs::write(dir.join("anchors.json"), json!([{"seq": last, "head": head, "block": 10}]).to_string()).unwrap();
    let ran = std::process::Command::new(env!("CARGO_BIN_EXE_task-api-replay"))
        .args(["--dir", dir.to_str().unwrap(), "--state", dir.join("fresh.db").to_str().unwrap(), "--signer", &h.state.signer()])
        .args(["--anchors", dir.join("anchors.json").to_str().unwrap(), "--join"])
        .output()
        .unwrap();
    assert!(ran.status.success(), "{}", String::from_utf8_lossy(&ran.stderr));
    let lines: Vec<Value> = String::from_utf8_lossy(&ran.stdout).lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(lines.len(), 1, "no divergence, only the summary: {lines:?}");
    assert_eq!((lines[0]["last"].as_i64(), lines[0]["started"].as_bool(), lines[0]["since"].as_i64()), (Some(last), Some(true), Some(first_snapshot)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn altered_dropped_forked_and_made_up_entries_are_caught() {
    // A short validation window, so the fails of the busy day are decided at their real deadline and the log shows it.
    let h = Harness::with(|s| s.validation_ttl = 2.0).await;
    busy_day(&h).await;
    let files = published(&h).await;
    let signer = h.state.signer();

    // An entry changed in a published file: the file's signature no longer holds.
    let mut altered = files.clone();
    let at = altered[0]["entries"].as_array().unwrap().iter().position(|e| e["kind"] == "finalize").unwrap();
    altered[0]["entries"][at]["output"]["credited"] = 999_999.into();
    assert_eq!(replayed(&h, &altered).1, Err(Break::Signature { segment: 1 }));

    // Changed and signed again with the API's own key: the links after it break.
    let resigned = {
        let mut file = altered[0].clone();
        let signature = h.state.sign(&journal::signed_payload(&file));
        file["signature"] = signature.into();
        vec![file]
    };
    assert!(matches!(replayed(&h, &resigned).1, Err(Break::Link { .. }) | Err(Break::Head { .. })));

    // History rewritten whole, links and all: the replay shows the verdict is not what the rules give.
    let rewritten = rechained(&h, &files, |entry| {
        if entry["kind"] == "close" {
            for upload in entry["output"]["settled"].as_array_mut().into_iter().flatten() {
                if upload["verdict"] == "fail" {
                    upload["budget"] = 100.into();
                }
            }
        }
    });
    let (replay, chained) = replayed(&h, &rewritten);
    assert_eq!(chained, Ok(()));
    let fraud: Vec<_> = replay.fraud().collect();
    assert!(fraud.iter().any(|d| d.kind == "close" && d.what.contains("close settled")), "{fraud:?}");
    // and the anchor the chain holds names the head the API committed to before, not this one.
    let honest = entries(&files);
    let anchored = journal::hash_of(honest.last().unwrap());
    let seq = honest.last().unwrap()["seq"].as_i64().unwrap();
    assert!(check_anchor(&honest, seq, &anchored).is_ok());
    assert!(matches!(check_anchor(&entries(&rewritten), seq, &anchored), Err(Break::Anchor { .. })));

    // A dropped entry, links rewritten around it: a replay sees the accounts part from the next snapshot.
    let dropped_at = honest.iter().position(|e| e["kind"] == "abandon").unwrap();
    let mut without = files.clone();
    let mut skip = 0;
    for file in &mut without {
        let held = file["entries"].as_array().unwrap().len();
        if dropped_at >= skip && dropped_at < skip + held {
            file["entries"].as_array_mut().unwrap().remove(dropped_at - skip);
        }
        skip += held;
    }
    let mut unlinked = without.clone();
    for file in &mut unlinked {
        let signature = h.state.sign(&journal::signed_payload(file));
        file["signature"] = signature.into();
    }
    assert!(matches!(replayed(&h, &unlinked).1, Err(Break::Gap { .. }) | Err(Break::Head { .. })), "a plain gap is caught by the numbers");
    let relinked = rechained(&h, &without, |_| {});
    let (replay, chained) = replayed(&h, &relinked);
    assert_eq!(chained, Ok(()));
    assert!(replay.fraud().any(|d| d.kind == "snapshot" || d.kind == "lapse"), "{:?}", replay.divergences);

    // A request the miner never signed: the log cannot show the miner asked for it.
    let made_up = rechained(&h, &files, |entry| {
        if entry["kind"] == "complete" && entry["output"].get("refused").is_none() {
            entry["input"]["request"]["nonce"] = "f".repeat(32).into();
        }
    });
    let (replay, _) = replayed(&h, &made_up);
    assert!(replay.fraud().any(|d| d.what.contains("not signed by the hotkey")), "{:?}", replay.divergences);

    // A fork: another file with the same number cannot follow on from the head others already read.
    let (_, seen) = journal::verify_segment(&files[0], &signer, 0, journal::GENESIS).unwrap();
    let mut forked = rechained(&h, &files, |entry| {
        if entry["kind"] == "lapse" {
            entry["output"]["budget"] = 50.into();
        }
    });
    if files.len() > 1 {
        assert!(journal::verify_segment(&forked[1], &signer, files[0]["last"].as_i64().unwrap(), &seen).is_err());
    }
    forked[0]["entries"][0]["input"]["started_at"] = 0.into();
    assert_ne!(journal::verify_segment(&forked[0], &signer, 0, journal::GENESIS).map(|(_, h)| h), Ok(seen));

    // A blob swapped behind an entry: what it names no longer hashes to its name.
    let snapshot = honest.iter().find(|e| e["kind"] == "snapshot").unwrap();
    let sha = snapshot["output"]["state"]["blob"].as_str().unwrap().to_string();
    let mut replay = Replay::open(None).unwrap();
    let swapped = |asked: &str| if asked == sha { Some(b"{}".to_vec()) } else { h.object(&journal::blob_key(asked)) };
    replay.apply_segment(&files[0], std::slice::from_ref(&signer), false, &swapped).unwrap().unwrap();
    assert!(replay.fraud().any(|d| d.kind == "snapshot" && d.what.contains("not published")));
    assert!(replay.divergences.iter().all(|d| d.severity == FRAUD));
}

#[tokio::test]
async fn every_read_is_signed_over_its_path_time_and_body() {
    use task_api::http::{signed_read, SIGNATURE_HEADER, SIGNED_AT_HEADER, SIGNER_HEADER};
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    for path in ["/v1/overview", "/v1/tasks?limit=5", "/v1/miners", "/v1/shares", "/v1/key"] {
        let response = h.http.get(format!("{}{path}", h.url)).send().await.unwrap();
        let headers = response.headers().clone();
        let body = response.bytes().await.unwrap();
        let header = |name: &str| headers[name].to_str().unwrap().to_string();
        assert_eq!(header(SIGNER_HEADER), h.state.signer(), "{path}");
        let public = desearch::hotkey::public_of(&h.state.signer()).unwrap();
        let signature: Vec<u8> = (0..128).step_by(2).map(|i| u8::from_str_radix(&header(SIGNATURE_HEADER)[i..i + 2], 16).unwrap()).collect();
        let at = header(SIGNED_AT_HEADER);
        assert!(desearch::hotkey::verify(&public, &signed_read(path, &at, &body), &signature), "{path}");
        assert!(!desearch::hotkey::verify(&public, &signed_read(path, &at, b"{}"), &signature), "another body does not verify");
        assert!(!desearch::hotkey::verify(&public, &signed_read("/v1/health", &at, &body), &signature), "nor another path");
    }
    let missing = h.http.get(format!("{}/v1/tasks/nope", h.url)).send().await.unwrap();
    assert_eq!(missing.status(), 404);
    assert!(missing.headers().get(SIGNATURE_HEADER).is_none(), "only answered reads are signed");
}

async fn expired_now(h: &Harness, task_id: &str) {
    expired_at(h, task_id, desearch::time::now()).await;
}

/// Ends a claim at `expiry` and reclaims it just after.
async fn expired_at(h: &Harness, task_id: &str, expiry: f64) {
    let _: i64 = redis::cmd("ZADD").arg(task_api::queues::CLAIMS).arg("XX").arg(expiry).arg(task_id).query_async(&mut h.redis.clone()).await.unwrap();
    lifecycle::reclaim_expired(&h.state, expiry + 1.0).await.unwrap();
}

#[tokio::test]
async fn a_final_verdict_edited_in_place_shows_at_the_next_snapshot() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    settled(&h, &h.miner, &h.validator, "pass").await;
    h.state.db.run(|conn| snapshot::record(conn, desearch::time::now())).await.unwrap();
    let (replay, _) = replayed(&h, &published(&h).await);
    assert_eq!(replay.divergences, [], "the replay finalizes the upload as the API did");

    // The operator turns the stored verdict, which `/v1/tasks` and a late vote read back, into a fail.
    h.state.db.run(|conn| Ok(conn.execute("UPDATE final_verdicts SET verdict = 'fail'", [])?)).await.unwrap();
    h.state.db.run(|conn| snapshot::record(conn, desearch::time::now())).await.unwrap();
    let (replay, _) = replayed(&h, &published(&h).await);
    assert!(replay.fraud().any(|d| d.kind == "snapshot" && d.what.contains("final_verdicts")), "{:?}", replay.divergences);
}

#[tokio::test]
async fn a_lapse_forgiven_for_a_storage_outage_is_decided_again_from_the_logged_outages() {
    use task_api::canary;
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let held = h.miner.claim().await["tasks"][0].clone();
    assert!(canary::run(&h.state).await.unwrap().failed.is_none());
    h.stub.state.fail("PUT", &format!("{}/uploads/", crate::harness::BUCKET), 1);
    assert!(canary::run(&h.state).await.unwrap().failed.is_some(), "the bucket refuses a write while the miner holds its claim");
    expired_now(&h, &task_id(&held)).await;
    assert!(canary::run(&h.state).await.unwrap().failed.is_none(), "storage is back");
    // A claim whose whole time came after storage was back: it lapses a claim's time from now.
    let later = h.rival.claim().await["tasks"][0].clone();
    expired_at(&h, &task_id(&later), desearch::time::now() + 2.0 * h.state.crawl.claim_ttl).await;
    h.state.db.run(|conn| snapshot::record(conn, desearch::time::now())).await.unwrap();

    let files = published(&h).await;
    let all = entries(&files);
    let forgiven: Vec<Value> = all.iter().filter(|e| e["kind"] == "lapse").map(|e| e["input"]["forgiven"].clone()).collect();
    assert_eq!(forgiven, [json!(canary::OUTAGE_CAUSE), Value::Null], "only the claim held through the outage costs nothing");
    assert_eq!(all.iter().filter(|e| e["kind"] == "probe").count(), 3);
    let outages: Vec<Value> = all.iter().filter(|e| e["kind"] == "outage").map(|e| e["input"]["event"].clone()).collect();
    assert_eq!(outages, [json!("begin"), json!("end")], "the failed probe opened an outage and the next closed it");
    let claims: Vec<&Value> = all.iter().filter(|e| e["kind"] == "claim").collect();
    assert!(claims.iter().all(|c| c["output"]["receipts"][0]["body"]["request"] == c["input"]["request"]["request"]), "receipts name their request");
    let (replay, chained) = replayed(&h, &files);
    assert_eq!(chained, Ok(()));
    assert_eq!(replay.divergences, []);

    // Forgiving the second lapse too, with its budget left alone: the outages the log holds show none then.
    let forged = rechained(&h, &files, |entry| {
        if entry["kind"] == "lapse" && entry["input"]["forgiven"].is_null() {
            entry["input"]["forgiven"] = canary::OUTAGE_CAUSE.into();
        }
    });
    let (replay, _) = replayed(&h, &forged);
    assert!(replay.fraud().any(|d| d.kind == "lapse" && d.what.contains("costs the miner nothing")), "{:?}", replay.divergences);

    // Dropping the outage from the log takes away the ground for the first one.
    let without = rechained(
        &h,
        &files
            .iter()
            .map(|file| {
                let mut file = file.clone();
                file["entries"].as_array_mut().unwrap().retain(|e| e["kind"] != "outage");
                file
            })
            .collect::<Vec<_>>(),
        |_| {},
    );
    let (replay, _) = replayed(&h, &without);
    assert!(replay.fraud().any(|d| d.kind == "lapse"), "{:?}", replay.divergences);

    // A receipt that names another request than the one it answers.
    let renamed = rechained(&h, &files, |entry| {
        if entry["kind"] == "claim" {
            entry["output"]["receipts"][0]["body"]["request"] = "0".repeat(64).into();
        }
    });
    let (replay, _) = replayed(&h, &renamed);
    assert!(replay.fraud().any(|d| d.what.contains("does not name the request")), "{:?}", replay.divergences);
}

/// What `anchor::pass` writes when the chain confirms an anchor of the head: the row, with the block it was registered in, and its entry.
async fn confirm_anchor(h: &Harness, block: i64) -> i64 {
    h.state
        .db
        .run(move |conn| {
            let (seq, head) = journal::head(conn)?;
            conn.execute(
                "INSERT OR REPLACE INTO log_anchors (seq, head, block, sent_at, extrinsic) VALUES (?, ?, ?, ?, '')",
                rusqlite::params![seq, head, block, block - 1],
            )?;
            let input = json!({"netuid": 22, "hotkey": "", "seq": seq, "head": head, "block": block});
            journal::append(conn, journal::Record::new("anchor", desearch::time::now(), input, Value::Null))?;
            Ok(seq)
        })
        .await
        .unwrap()
}

async fn past_block(h: &Harness, block: i64) {
    while h.state.seeds.current_block().await.unwrap() <= block {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

async fn kept_round(h: &Harness, round_id: &str) -> task_api::rounds::Round {
    let asked = round_id.to_string();
    h.state.db.run(move |conn| task_api::roundstore::get(conn, &asked)).await.unwrap().expect("the round is kept")
}

#[tokio::test]
async fn with_anchoring_on_a_round_is_revealed_only_after_an_anchor_before_its_seed_block() {
    let anchoring = AnchorSettings { network: "ws://127.0.0.1:1".into(), netuid: 22, key_uri: String::new(), every_blocks: 12, min_blocks: 3, per_epoch: 30 };
    let h = Harness::with(|s| s.anchor = Some(anchoring)).await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": urls(6)})).await.unwrap();
    let (round_id, first_seed) = (enqueued["round_id"].as_str().unwrap().to_string(), enqueued["seed_block"].as_i64().unwrap());

    // An anchor of the commitment was sent before the seed block and may still land: the round waits, keeping its seed block.
    h.state
        .db
        .run(move |conn| {
            let (seq, head) = journal::head(conn)?;
            conn.execute(
                "INSERT INTO log_anchors (seq, head, block, sent_at, extrinsic) VALUES (?, ?, NULL, ?, '')",
                rusqlite::params![seq, head, first_seed - 5],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    past_block(&h, first_seed).await;
    lifecycle::reveal_pending(&h.state).await.unwrap();
    let waiting = kept_round(&h, &round_id).await;
    assert_eq!((waiting.seed.is_none(), waiting.seed_block), (true, first_seed), "not revealed, not moved");

    // It never landed: the API could have read the seed before the chain held the commitment, so it commits again to a later block.
    h.state.db.run(move |conn| Ok(conn.execute("UPDATE log_anchors SET sent_at = ?", [first_seed - 40])?)).await.unwrap();
    lifecycle::reveal_pending(&h.state).await.unwrap();
    let moved = kept_round(&h, &round_id).await;
    assert!(moved.seed.is_none() && moved.seed_block > first_seed + lifecycle::reveal_lead(1), "a miss leaves the next anchor more room: {moved:?}");
    assert_eq!(
        (lifecycle::reveal_lead(0), lifecycle::reveal_lead(1), lifecycle::reveal_lead(2), lifecycle::reveal_lead(3), lifecycle::reveal_lead(60)),
        (10, 30, 70, 150, 361),
        "each miss doubles the room past the inclusion window, up to an epoch"
    );
    assert_ne!(moved.manifest_hash, enqueued["manifest_hash"].as_str().unwrap(), "the new seed block is part of the commitment");

    // An anchor of the new commitment registered before the new seed block lets the seed cut it.
    confirm_anchor(&h, moved.seed_block - 1).await;
    h.revealed().await;
    assert!(kept_round(&h, &round_id).await.seed.is_some());

    let files = published(&h).await;
    let all = entries(&files);
    assert_eq!(all.iter().filter(|e| e["kind"] == "recommit").count(), 1);
    assert_eq!(all.iter().filter(|e| e["kind"] == "start").map(|e| e["input"]["settings"]["anchored"].clone()).next_back(), Some(json!(true)));
    let (replay, chained) = replayed(&h, &files);
    assert_eq!(chained, Ok(()));
    assert_eq!(replay.divergences, []);

    // Without the anchor in the log, the reveal came before the chain held the commitment.
    let unanchored = rechained(
        &h,
        &files
            .iter()
            .map(|file| {
                let mut file = file.clone();
                file["entries"].as_array_mut().unwrap().retain(|e| e["kind"] != "anchor");
                file
            })
            .collect::<Vec<_>>(),
        |_| {},
    );
    let (replay, _) = replayed(&h, &unanchored);
    assert!(replay.fraud().any(|d| d.kind == "reveal" && d.what.contains("anchored")), "{:?}", replay.divergences);

    // The chain held the first commitment before its seed block, but the log left that anchor out: a validator's own node shows
    // it, and committing the round again was a re-roll of its seed.
    let committed = all.iter().find(|e| e["kind"] == "enqueue").unwrap()["seq"].as_i64().unwrap();
    let mut replay = Replay::open(None).unwrap();
    replay.note_anchor(committed, first_seed - 1).unwrap();
    let blobs = |sha: &str| h.object(&journal::blob_key(sha));
    for file in &files {
        replay.apply_segment(file, &[h.state.signer()], false, &blobs).unwrap().unwrap();
    }
    assert!(replay.fraud().any(|d| d.kind == "recommit" && d.what.contains("anchored before its seed block")), "{:?}", replay.divergences);

    // A commitment the request's URLs do not make.
    let other = rechained(&h, &files, |entry| {
        if entry["kind"] == "enqueue" {
            entry["output"]["commitment"]["urls_hash"] = "0".repeat(64).into();
        }
    });
    let (replay, _) = replayed(&h, &other);
    assert!(replay.fraud().any(|d| d.kind == "enqueue" && d.what.contains("URL set")), "{:?}", replay.divergences);

    // A new commitment that is not the old one moved to its new seed block.
    let rerolled = rechained(&h, &files, |entry| {
        if entry["kind"] == "recommit" {
            entry["output"]["manifest_hash"] = "0".repeat(64).into();
        }
    });
    let (replay, _) = replayed(&h, &rerolled);
    assert!(replay.fraud().any(|d| d.kind == "recommit"), "{:?}", replay.divergences);
}

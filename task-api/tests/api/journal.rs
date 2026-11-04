//! The log: everything the API decided goes into it chained and signed, and a replay of the published files reaches the same accounts;
//! an altered, dropped, forked or made-up entry shows.

use serde_json::{json, Value};
use task_api::journal::{self, Break};
use task_api::replay::{check_anchor, replay_dir, Replay, FRAUD};
use task_api::{lifecycle, snapshot};

use crate::harness::{judged, parquet, task_id, urls, Client, Harness};

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
        if let Err(broken) = replay.apply_segment(file, &h.state.signer(), false, &blobs).unwrap() {
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

/// Mines until the upload is settled: checked by `validator` with `verdict` when drawn, finalized on the miner's counts otherwise.
async fn settled(h: &Harness, miner: &Client, validator: &Client, verdict: &str) -> String {
    let task = h.mine(miner, 0).await;
    let id = task_id(&task);
    for _ in 0..100 {
        lifecycle::settle_seeded(&h.state).await.unwrap();
        let picked = h.state.validation.job(&h.redis, &id).await.unwrap().is_some_and(|job| !job["picked"].is_null());
        if picked {
            judged(h, validator, verdict, Some(&id), json!({})).await.1.unwrap();
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
    let h = Harness::start().await;
    busy_day(&h).await;
    let files = published(&h).await;
    let all = entries(&files);
    let kinds: std::collections::BTreeSet<&str> = all.iter().filter_map(|e| e["kind"].as_str()).collect();
    for kind in ["start", "snapshot", "enqueue", "reveal", "claim", "complete", "settle", "report", "finalize", "abandon", "lapse", "requests", "shares"] {
        assert!(kinds.contains(kind), "no {kind} entry in {kinds:?}");
    }
    let finalized: Vec<&Value> = all.iter().filter(|e| e["kind"] == "finalize").collect();
    assert!(finalized.iter().any(|e| e["output"]["penalty"] == true), "two fails among the last ten bring the full penalty");
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
            assert!(replay_dir(&mut replay, &dir, &h.state.signer(), false).unwrap().is_ok());
            assert_eq!(replay.divergences, []);
        }
    }
    let replay = Replay::open(Some(&state)).unwrap();
    assert_eq!(replay.position().unwrap(), (last, head));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn altered_dropped_forked_and_made_up_entries_are_caught() {
    let h = Harness::start().await;
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
        if entry["kind"] == "finalize" && entry["output"]["verdict"] == "fail" {
            entry["output"]["budget"] = 100.into();
        }
    });
    let (replay, chained) = replayed(&h, &rewritten);
    assert_eq!(chained, Ok(()));
    let fraud: Vec<_> = replay.fraud().collect();
    assert!(fraud.iter().any(|d| d.kind == "finalize" && d.what.contains("verdict")), "{fraud:?}");
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
    replay.apply_segment(&files[0], &signer, false, &swapped).unwrap().unwrap();
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

//! Custody of the log: a CVM appointed by validators' stake, a handover to an approved successor, and the pause.

use std::sync::Arc;

use futures::FutureExt;
use serde_json::{json, Value};
use task_api::custody::{self, link_problem, signature_holds, Status};
use task_api::governance::Snapshot;
use task_api::settings::{KeySource, Settings};
use task_api::state::State;

use crate::fakedstack::{self, FakeDstack};
use crate::harness::{self, urls, Harness, TempDir, ADMIN, MINER, RIVAL};

fn short_dir() -> TempDir {
    TempDir(std::path::PathBuf::from("/tmp").join(format!("sn22-{}", &uuid::Uuid::new_v4().simple().to_string()[..12])))
}

fn in_custody(socket: String) -> impl FnOnce(&mut Settings) {
    move |s| {
        s.tee.key_source = KeySource::Cvm;
        s.tee.dstack_socket = socket;
        s.tee.custody = true;
    }
}

/// One independent validator holding all the stake, committed to `text` long ago; the owner's stake never counts.
fn approving(text: String) -> custody::SnapshotSource {
    Arc::new(move || {
        let text = text.clone();
        async move {
            let validator = desearch::hotkey::Hotkey::from_uri(harness::VALIDATOR)?.ss58();
            let neuron = |hotkey: String, coldkey: &str, stake| task_api::chain::Neuron {
                uid: 0,
                hotkey,
                coldkey: coldkey.into(),
                validator_permit: true,
                total_stake_rao: stake,
                alpha_stake_rao: 0,
            };
            let owner = desearch::hotkey::Hotkey::from_uri("//owner-api-test")?.ss58();
            Ok(Snapshot {
                block: 100_000,
                neurons: vec![neuron(validator.clone(), "independent", 100), neuron(owner.clone(), "owner", 10_000)],
                owner_coldkey: "owner".into(),
                owner_hotkey: owner,
                commitments: [(validator, (0, text))].into_iter().collect(),
            })
        }
        .boxed()
    })
}

fn genuine_quotes() -> custody::QuoteCheck {
    Arc::new(|_quote| async { Ok(None) }.boxed())
}

fn trust(state: &State, snapshots: custody::SnapshotSource) {
    *state.custody.snapshots.lock().unwrap() = Some(snapshots);
    *state.custody.quotes.lock().unwrap() = Some(genuine_quotes());
}

async fn latest(h: &Harness) -> Value {
    serde_json::from_slice(&h.state.storage.get(custody::LATEST).await.expect("a published record")).unwrap()
}

#[tokio::test]
async fn a_standby_cvm_signs_nothing_until_appointed_then_hands_the_log_to_an_approved_successor() {
    let (dir_a, dir_b) = (short_dir(), short_dir());
    let agent_a = FakeDstack::start(&dir_a.0).await;
    let h = Harness::with(in_custody(agent_a.socket.display().to_string())).await;
    let version = fakedstack::compose_hash();
    let a = h.state.signer();

    assert_eq!(h.state.custody.status(), Status::Standby);
    assert_eq!(h.miner.claim().await["refusal"]["code"], "UNAVAILABLE", "a CVM in standby hands out nothing");
    assert_eq!(h.admin.post("/v1/admin/enqueue", json!({"urls": urls(3)})).await.unwrap_err().status, 503);
    trust(&h.state, approving(format!("sn22tee:{version}")));
    custody::poll(&h.state).await.unwrap();
    assert_eq!(h.state.custody.status(), Status::Standby, "approving the version does not appoint a CVM of it");

    trust(&h.state, approving(format!("sn22tee:{version}@{a}")));
    custody::poll(&h.state).await.unwrap();
    assert_eq!(h.state.custody.status(), Status::Active);
    let appointed = latest(&h).await;
    assert_eq!((appointed["kind"].as_str(), appointed["seq"].as_i64(), appointed["signer"].as_str()), (Some("appoint"), Some(0), Some(a.as_str())));
    assert!(signature_holds(&appointed) && link_problem(&appointed, None, "").is_none());
    let attested: Value = serde_json::from_slice(&h.state.storage.get(&custody::attestation_key(&a)).await.unwrap()).unwrap();
    assert_eq!(appointed["attestation_sha256"].as_str().unwrap(), task_api::attestation::sha256_hex(&task_api::py::canonical(&attested["document"])));
    h.enqueue(urls(6)).await;
    let claimed = h.miner.claim().await;
    assert_eq!(claimed["tasks"].as_array().unwrap().len(), 1, "appointed, it hands out work");

    let agent_b = FakeDstack::start(&dir_b.0).await;
    let (b, b_url, _b_dir) = h.sibling(12, in_custody(agent_b.socket.display().to_string())).await;
    trust(&b, approving(format!("sn22tee:{version}")));
    trust(&h.state, approving(format!("sn22tee:{}", "ab".repeat(32))));
    let refused = h.admin.post("/v1/admin/handover", json!({"successor": b_url})).await.unwrap_err();
    assert_eq!(refused.status, 409);
    assert!(refused.detail.as_str().unwrap().contains("not approved"), "{}", refused.detail);
    assert_eq!(h.state.custody.status(), Status::Active, "a refused handover leaves the log where it was");
    assert_eq!(h.miner.post("/v1/admin/handover", json!({"successor": b_url})).await.unwrap_err().status, 403);

    trust(&h.state, approving(format!("sn22tee:{version}")));
    let handed = h.admin.post("/v1/admin/handover", json!({"successor": b_url})).await.unwrap();
    let record = handed["record"].clone();
    assert_eq!((record["kind"].as_str(), record["from"].as_str(), record["signer"].as_str()), (Some("handover"), Some(a.as_str()), Some(b.signer().as_str())));
    assert!(link_problem(&record, Some(&appointed), "").is_none());
    assert_eq!(latest(&h).await, record);
    assert_eq!(h.state.custody.status(), Status::Retired);
    assert_eq!(h.miner.claim().await["refusal"]["code"], "UNAVAILABLE", "the old CVM signs no more");

    assert_eq!(b.custody.status(), Status::Standby, "the successor waits for the published record");
    custody::poll(&b).await.unwrap();
    assert_eq!(b.custody.status(), Status::Active);
    let chain = b.db.run(custody::records).await.unwrap();
    assert_eq!(chain.iter().map(|r| r["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["appoint", "handover"]);
    let rounds = b.lookups.read(|conn| task_api::roundstore::recent(conn, 10)).await.unwrap();
    assert_eq!(rounds.len(), 1, "the round came with the state");
    let held = h.client_of(&b_url, MINER).claim().await;
    assert_eq!(held["refusal"]["code"], "NO_CAPACITY", "the miner's claim from before the handover came with the state");
    assert!(crate::receipt_verifies(&held["receipt"], &b.signer()), "signed by the successor");
    let next = h.client_of(&b_url, RIVAL).claim().await;
    assert_eq!(next["tasks"].as_array().unwrap().len(), 1, "{next}");
    let second = h.client_of(&b_url, ADMIN).post("/v1/admin/handover", json!({"successor": h.url})).await.unwrap_err();
    assert_eq!(second.status, 409, "a retired CVM takes no handover back");
    let _ = (dir_a, dir_b);
}

#[tokio::test]
async fn a_pause_stops_handing_out_tasks_and_nothing_else() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    assert_eq!(h.miner.post("/v1/admin/pause", json!({"paused": true, "reason": "x"})).await.unwrap_err().status, 403);
    assert_eq!(h.admin.post("/v1/admin/pause", json!({"paused": true})).await.unwrap_err().status, 422, "a pause needs a reason");
    h.admin.post("/v1/admin/pause", json!({"paused": true, "reason": "an upgrade"})).await.unwrap();
    let refused = h.miner.claim().await;
    assert_eq!((refused["refusal"]["code"].as_str(), refused["refusal"]["inputs"]["reason"].as_str()), (Some("PAUSED"), Some("an upgrade")));
    assert!(crate::receipt_verifies(&refused["receipt"], &h.state.signer()), "the refusal is signed into the log");
    assert_eq!(h.public("/v1/custody").await.1["paused"]["reason"], "an upgrade");
    h.admin.post("/v1/admin/pause", json!({"paused": false})).await.unwrap();
    assert_eq!(h.miner.claim().await["tasks"].as_array().unwrap().len(), 1);
}

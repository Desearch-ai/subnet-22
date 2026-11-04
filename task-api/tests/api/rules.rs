//! Block A's rules end to end: rounds cut by their seed, outages that forgive, loadability and neutral hosts, the incentive cap.

use std::sync::Arc;

use serde_json::{json, Value};
use task_api::budgets::{self, CEILING, CRAWL, START};
use task_api::registry::Entry;
use task_api::rounds::{self, Url};
use task_api::settings::RegistryMode;
use task_api::state::State;
use task_api::{lifecycle, outages, proofs, roundstore};

use crate::checks::{established_in, settled};
use crate::fakechain::FakeChain;
use crate::harness::{judged, task_id, Harness};

fn spread_urls(hosts: usize, per_host: usize) -> Vec<Value> {
    (0..hosts)
        .flat_map(|h| (0..per_host).map(move |n| json!({"host": format!("site{h}.example"), "url": format!("https://site{h}.example/page/{n}")})))
        .collect()
}

#[tokio::test]
async fn a_round_commits_to_its_urls_and_its_seed_alone_cuts_them_into_tasks() {
    let h = Harness::with(|s| s.task_urls = 4).await;
    let sent = spread_urls(3, 4);
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": sent})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    assert_eq!(enqueued["batches"], 3, "twelve URLs make three tasks of four");
    let committed = h.public(&format!("/v1/rounds/{round_id}")).await.1;
    let set: Vec<Url> = sent.iter().map(|u| Url { host: u["host"].as_str().unwrap().into(), url: u["url"].as_str().unwrap().into() }).collect();
    assert_eq!(
        (committed["algorithm"].clone(), committed["url_count"].clone(), committed["urls_hash"].clone()),
        (json!(proofs::SET_ALGORITHM), json!(12), json!(rounds::set_hash(&set)))
    );
    assert_eq!(committed["manifest_hash"], proofs::set_manifest_hash(&round_id, committed["seed_block"].as_i64().unwrap(), 4, 12, &rounds::set_hash(&set)));
    let asked = round_id.clone();
    let before = h.state.db.run(move |conn| roundstore::get(conn, &asked)).await.unwrap().unwrap();
    assert!(before.batches.is_empty(), "no batch exists before the seed");

    h.revealed().await;
    let asked = round_id.clone();
    let revealed = h.state.db.run(move |conn| roundstore::get(conn, &asked)).await.unwrap().unwrap();
    let seed = revealed.seed.clone().unwrap();
    let expected = rounds::cut(&set, &seed, &round_id, 4);
    let mut handed: Vec<Value> = Vec::new();
    for miner in [&h.miner, &h.rival] {
        handed.extend(miner.claim().await["tasks"].as_array().cloned().unwrap_or_default());
    }
    for task in &handed {
        let cut = expected.iter().find(|b| b.batch_id == task_id(task)).expect("a task the seed cut");
        let urls: Vec<&str> = cut.urls.iter().map(|u| u.url.as_str()).collect();
        assert_eq!(task["urls"], json!(urls));
        let hosts: std::collections::HashSet<&str> = cut.urls.iter().map(|u| u.host.as_str()).collect();
        assert_eq!(hosts.len(), 3, "every task has every host of the round");
    }
    let mut sealed: Vec<(String, Value)> = revealed.batches.iter().map(|b| (b.batch_id.clone(), b.seal())).collect();
    sealed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut cut: Vec<(String, Value)> = expected.iter().map(|b| (b.batch_id.clone(), b.seal())).collect();
    cut.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(sealed, cut, "SQLite keeps the batches the seed cut");
}

#[tokio::test]
async fn a_claim_alive_during_a_storage_outage_lapses_at_no_cost() {
    let h = Harness::start().await;
    h.enqueue(spread_urls(6, 1)).await;
    let task = h.miner.claim().await["tasks"][0].clone();
    let at = h.state.tick().await.unwrap();
    h.state.db.run(move |conn| outages::begin(conn, outages::STORAGE, at.at, at.block, "PUT probe/storage.json: timed out")).await.unwrap();
    let later = h.state.clock().await.unwrap().at(at.block + 100_000);
    h.state.db.run(move |conn| outages::end(conn, outages::STORAGE, later, at.block + 100_000)).await.unwrap();
    let reclaimed = lifecycle::reclaim_expired(&h.state, later + 3_600.0).await.unwrap();
    assert_eq!(reclaimed, [(task_id(&task), h.miner.ss58())]);
    let hotkey = h.miner.ss58();
    let (strikes, budget) = h
        .state
        .db
        .run(move |conn| {
            Ok((
                conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [&hotkey], |row| row.get::<_, i64>(0))?,
                budgets::get(conn, &hotkey, CRAWL)?.budget,
            ))
        })
        .await
        .unwrap();
    assert_eq!((strikes, budget), (0, START));
    let events = h.public(&format!("/v1/events?task_id={}", task_id(&task))).await.1;
    let causes: Vec<Value> = events["events"].as_array().unwrap().iter().map(|e| e["cause"].clone()).collect();
    assert!(causes.contains(&json!("storage_outage")), "{causes:?}");
    let listed = h.public("/v1/outages").await.1;
    assert_eq!(listed["outages"][0]["cause"], outages::STORAGE);
}

#[tokio::test]
async fn a_claim_made_after_storage_came_back_lapses_as_usual() {
    let h = Harness::start().await;
    h.enqueue(spread_urls(6, 1)).await;
    let at = h.state.tick().await.unwrap();
    let before = at.at - 10_000.0;
    h.state
        .db
        .run(move |conn| {
            outages::begin(conn, outages::STORAGE, before, at.block - 1_000, "HEAD probe/storage.json: 503")?;
            outages::end(conn, outages::STORAGE, before + 60.0, at.block - 995)
        })
        .await
        .unwrap();
    h.miner.claim().await;
    lifecycle::reclaim_expired(&h.state, at.at + 3_600.0).await.unwrap();
    let hotkey = h.miner.ss58();
    let struck: String =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT reason FROM strikes WHERE hotkey = ?", [&hotkey], |row| row.get(0))?)).await.unwrap();
    assert_eq!(struck, "claim_expired");
}

#[tokio::test]
async fn the_storage_probe_writes_and_reads_back_its_object() {
    let h = Harness::start().await;
    outages::probe_storage(&h.state, 1.0).await.expect("the stand-in storage answers");
    assert!(h.state.storage.head(outages::PROBE_KEY).await.unwrap().is_some());
}

#[tokio::test]
async fn a_round_validators_could_mostly_not_load_lowers_the_gate_and_is_published() {
    let h = Harness::with(|s| {
        s.task_urls = 3;
        s.check_share = 0.0;
        s.loadability_samples = 1;
    })
    .await;
    h.enqueue(spread_urls(6, 1)).await;
    // The first upload is checked: the validator could load none of what it sampled.
    let checked = task_id(&h.mine(&h.miner, 3).await);
    judged(
        &h,
        &h.validator,
        "pass",
        Some(&checked),
        json!({"matched": 0, "errors_confirmed": 1, "error_rows": 3, "samples": [{"url": "", "outcome": "errors_confirmed"}]}),
    )
    .await
    .1
    .unwrap();
    let round_id = h.view(&checked).await["round_id"].as_str().unwrap().to_string();
    let loads = h.public(&format!("/v1/rounds/{round_id}")).await.1["loadability"].clone();
    assert_eq!((loads["sampled"].clone(), loads["loaded"].clone(), loads["loadability"].clone()), (json!(1), json!(0), json!(0.0)));

    // An established hotkey's unchecked upload of the same round, one row of three: under the absolute gate a fail.
    established_in(&h, h.rival.ss58(), CRAWL).await;
    let task = h.rival.claim().await["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), crate::harness::parquet(&task)).await;
    h.rival.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "rows": 1, "ok": 1, "errors": 0})).await.unwrap();
    assert_eq!(settled(&h, &task_id(&task)).await, "finalized");
    let view = h.view(&task_id(&task)).await;
    assert_eq!((view["score"]["verdict"].clone(), view["score"]["reason"].clone()), (json!("pass"), json!("unchecked")), "the gate followed the round");
}

#[tokio::test]
async fn a_host_most_miners_could_not_load_is_neutral_for_the_round() {
    let h = Harness::with(|s| {
        s.task_urls = 2;
        s.neutral_uploads = 1;
    })
    .await;
    h.enqueue(vec![json!({"host": "owner.example", "url": "https://owner.example/1"}), json!({"host": "fine.example", "url": "https://fine.example/1"})]).await;
    let task = task_id(&h.mine(&h.miner, 1).await);
    let hosts = json!({"owner.example": [1, 1], "fine.example": [1, 0]});
    let verdict = judged(
        &h,
        &h.validator,
        "pass",
        Some(&task),
        json!({"hosts": hosts, "error_rows": 1, "sampled": 2, "matched": 1, "errors_confirmed": 1, "samples": [{"url": "", "outcome": "matched"}, {"url": "", "outcome": "errors_confirmed"}]}),
    )
    .await
    .1
    .unwrap();
    assert_eq!((verdict["verdict"].clone(), verdict["credited"].clone()), (json!("pass"), json!(1)), "only the host it could be paid for");
    let round_id = h.view(&task).await["round_id"].as_str().unwrap().to_string();
    let loads = h.public(&format!("/v1/rounds/{round_id}")).await.1["loadability"].clone();
    assert_eq!(loads["neutral"], json!(["owner.example"]));
    assert_eq!(h.view(&task).await["score"]["accounts"]["assigned"], 1);
}

#[tokio::test]
async fn validators_that_will_not_pay_a_miner_cap_its_budget() {
    let fake = FakeChain::start(1_000, 12.0).await;
    fake.refuse(true);
    let h = Harness::start().await;
    let mut settings = h.state.settings.clone();
    settings.registry = RegistryMode::Chain { netuid: 22, network: fake.url.clone() };
    let state = Arc::new(State::new(settings, h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.unwrap());
    let (paid, unpaid, newcomer) = (h.miner.ss58(), h.rival.ss58(), "5Newcomer".to_string());
    let entry = |hotkey: &str, incentive: f64| Entry { hotkey: hotkey.into(), uid: Some(1), is_validator: false, coldkey: None, incentive: Some(incentive) };
    state.registry.restore(vec![entry(&paid, 0.1), entry(&unpaid, 0.0), entry(&newcomer, 0.0)]);
    let at = state.rule_now().await.unwrap();
    let (a, b, c) = (paid.clone(), unpaid.clone(), newcomer.clone());
    state
        .db
        .run(move |conn| {
            // Each did a third of the counted crawl work over the last day; two of them started two days ago.
            budgets::credit(conn, &a, 500, CRAWL, at - 48.0 * 3600.0)?;
            budgets::credit(conn, &a, 500, CRAWL, at)?;
            budgets::credit(conn, &b, 500, CRAWL, at - 48.0 * 3600.0)?;
            budgets::credit(conn, &b, 500, CRAWL, at)?;
            budgets::credit(conn, &c, 500, CRAWL, at)
        })
        .await
        .unwrap();
    assert_eq!(state.budget_cap(&paid, at).await.unwrap(), CEILING, "validators pay it what its work implies");
    assert_eq!(state.budget_cap(&unpaid, at).await.unwrap(), START, "validators pay it nothing");
    assert_eq!(state.budget_cap(&newcomer, at).await.unwrap(), CEILING, "new work is not judged against an incentive yet to follow it");
}

//! The public log pages: votes, validators, miners, live work, series, events and tasks.

use serde_json::{json, Value};
use task_api::chain::Neuron;
use task_api::settings::RegistryMode;

use crate::harness::{judged, task_id, urls, Client, Harness};

async fn read(h: &Harness, path: &str) -> Value {
    let (status, body) = h.public(path).await;
    assert_eq!(status, 200, "{path}: {body}");
    body
}

async fn started(task_urls: usize) -> Harness {
    let h = Harness::with(|s| s.task_urls = task_urls).await;
    h.enqueue(urls(6)).await;
    h
}

async fn vote(h: &Harness, validator: &Client, verdict: &str, task: Option<&str>) {
    judged(h, validator, verdict, task, json!({})).await.1.expect("the vote is taken");
}

async fn three_active(h: &Harness) -> String {
    let first = task_id(&h.mine(&h.miner, 0).await);
    vote(h, &h.validator, "pass", Some(&first)).await;
    let second = task_id(&h.mine(&h.rival, 0).await);
    vote(h, &h.other_validator, "pass", Some(&second)).await;
    vote(h, &h.validator, "pass", Some(&second)).await;
    let third = task_id(&h.mine(&h.miner, 0).await);
    vote(h, &h.third_validator, "pass", Some(&third)).await;
    third
}

fn field(rows: &Value, name: &str) -> Vec<Value> {
    rows.as_array().unwrap().iter().map(|row| row[name].clone()).collect()
}

#[tokio::test]
async fn every_validators_vote_on_a_task_is_kept_and_marked() {
    let h = started(2).await;
    let task = three_active(&h).await;
    vote(&h, &h.validator, "fail", Some(&task)).await;
    vote(&h, &h.other_validator, "pass", Some(&task)).await;
    let on_task = h.view(&task).await["votes"].clone();
    assert_eq!(field(&on_task, "verdict"), [json!("pass"), json!("fail"), json!("pass")]);
    assert_eq!(field(&on_task, "agreed"), [json!(true), json!(false), json!(true)]);
    assert_eq!(field(&on_task, "decided").iter().filter(|d| **d == true).count(), 1);
    assert!(field(&on_task, "final_verdict").iter().all(|v| v == "pass"));
    let listed = read(&h, &format!("/v1/votes?task_id={task}")).await["votes"].clone();
    let mut newest_first = on_task.as_array().unwrap().clone();
    newest_first.reverse();
    assert_eq!(listed, Value::Array(newest_first), "the list is newest first");
    assert_eq!(field(&read(&h, "/v1/votes?agreed=false").await["votes"], "validator"), [json!(h.validator.ss58())]);
    let validator = read(&h, &format!("/v1/validators/{}", h.validator.ss58())).await;
    assert_eq!((validator["votes"].clone(), validator["disagreed"].clone()), (json!(3), json!(1)));
    assert_eq!((validator["agreement"].clone(), validator["active"].clone()), (json!(0.6667), json!(true)));
    assert_eq!((validator["audits"].clone(), validator["excluded"].clone()), (json!(2), json!(false)));
    let validators = read(&h, "/v1/validators").await["validators"].clone();
    assert_eq!(validators.as_array().unwrap().len(), 3);
    assert_eq!(field(&validators, "decided").iter().map(|d| d.as_i64().unwrap()).sum::<i64>(), 3);
}

#[tokio::test]
async fn votes_on_a_task_without_a_majority_count_for_nobody() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    vote(&h, &h.validator, "pass", None).await;
    let task = task_id(&h.mine(&h.rival, 0).await);
    vote(&h, &h.other_validator, "fail", Some(&task)).await;
    vote(&h, &h.validator, "pass", Some(&task)).await;
    let votes = read(&h, &format!("/v1/votes?task_id={task}")).await["votes"].clone();
    let mut verdicts: Vec<String> = field(&votes, "verdict").iter().map(|v| v.as_str().unwrap().to_string()).collect();
    verdicts.sort();
    assert_eq!(verdicts, ["fail", "pass"]);
    assert!(field(&votes, "final_verdict").iter().all(|v| v == "void"));
    assert!(votes.as_array().unwrap().iter().all(|v| v["agreed"].is_null() && v["decided"] == false));
}

#[tokio::test]
async fn the_overview_and_the_miner_list_follow_the_work() {
    let h = started(2).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    vote(&h, &h.validator, "pass", None).await;
    let held = task_id(&h.rival.claim().await["tasks"][0]);
    let waiting = task_id(&h.mine(&h.miner, 0).await);

    let overview = read(&h, "/v1/overview").await;
    assert_eq!((overview["claimed"].clone(), overview["validating"].clone()), (json!(1), json!(1)));
    let window = &overview["window"];
    assert_eq!(
        (window["tasks"].clone(), window["pass"].clone(), window["credited"].clone(), window["votes"].clone()),
        (json!(1), json!(1), json!(2), json!(1))
    );
    assert_eq!(overview["validators"], json!({"active": 1, "known": 1, "agreement": 1.0}));
    assert_eq!((overview["miners"].clone(), overview["total"]["pass"].clone()), (json!(1), json!(1)));

    let miners = read(&h, "/v1/miners").await["miners"].clone();
    let first = &miners[0];
    assert_eq!(
        (first["share"].clone(), first["credited"].clone(), first["budget"].clone(), first["coverage"].clone()),
        (json!(1.0), json!(2), json!(2), json!(1.0))
    );
    assert_eq!((first["in_flight"].clone(), first["waiting"].clone(), first["locked_until"].clone()), (json!(0), json!(1), Value::Null));
    let mut hotkeys: Vec<String> = field(&miners, "hotkey").iter().map(|h| h.as_str().unwrap().to_string()).collect();
    hotkeys.sort();
    let mut expected = vec![h.miner.ss58(), h.rival.ss58()];
    expected.sort();
    assert_eq!(hotkeys, expected);
    let miner = read(&h, &format!("/v1/miners/{}", h.miner.ss58())).await;
    assert_eq!((miner["share"].clone(), miner["window"]["tasks"].clone()), (json!(1.0), json!(1)));

    let live = read(&h, "/v1/live").await;
    assert_eq!(field(&live["claims"], "task_id"), [json!(held)]);
    assert_eq!(field(&live["uploads"], "task_id"), [json!(waiting)]);
    assert_eq!((live["uploads"][0]["voters"].clone(), live["claims"][0]["urls"].clone()), (json!([]), json!(2)));

    let points = read(&h, "/v1/stats/series?bucket_minutes=5&buckets=3").await["points"].clone();
    assert_eq!(points.as_array().unwrap().len(), 3);
    assert_eq!((points[2]["tasks"].clone(), points[2]["pass"].clone()), (json!(1), json!(1)));
    assert_eq!((points[0]["tasks"].clone(), points[1]["tasks"].clone()), (json!(0), json!(0)));
    let events = read(&h, &format!("/v1/events?miner={}", h.miner.ss58())).await["events"].clone();
    assert_eq!(field(&events, "outcome"), ["completed", "issued", "completed", "issued"].map(Value::from));
    assert_eq!(events[3]["task_id"], task.as_str());
    assert_eq!(read(&h, "/v1/tasks?verdict=fail").await["tasks"], json!([]));
}

#[tokio::test]
async fn log_reads_are_bounded() {
    let h = Harness::start().await;
    assert_eq!(h.public("/v1/stats/series?bucket_minutes=7").await.0, 422);
    assert_eq!(h.public("/v1/stats/series?miner=a&validator=b").await.0, 422);
    assert_eq!(h.public(&format!("/v1/votes?miner={}", "m".repeat(65))).await.0, 422);
    assert_eq!(h.public("/v1/tasks?verdict=maybe").await.0, 422);
    assert_eq!(h.public("/v1/overview?hours=169").await.0, 422);
}

#[tokio::test]
async fn a_finalized_task_says_when_it_was_claimed_and_uploaded() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    vote(&h, &h.validator, "pass", None).await;
    let task = read(&h, "/v1/tasks").await["tasks"][0].clone();
    let (claimed, completed, scored) = (task["claimed_at"].as_f64().unwrap(), task["completed_at"].as_f64().unwrap(), task["scored_at"].as_f64().unwrap());
    assert!(claimed <= completed && completed <= scored);
    assert_eq!((task["miner_uid"].clone(), task["validator_uid"].clone()), (Value::Null, Value::Null), "the local registry knows no uids");
}

fn neuron(client: &Client, uid: i64, coldkey: &str, validator: bool) -> Neuron {
    let stake = if validator { 50_000_000_000_000 } else { 0 };
    Neuron { uid, hotkey: client.ss58(), coldkey: coldkey.into(), validator_permit: validator, total_stake_rao: stake, alpha_stake_rao: stake }
}

#[tokio::test]
async fn hotkeys_come_with_their_uids_and_coldkeys() {
    let h = Harness::with(|s| s.registry = RegistryMode::Chain { netuid: 22, network: "finney".into() }).await;
    h.state.registry.load(vec![neuron(&h.miner, 7, "ck", false), neuron(&h.validator, 2, "cv", true)]);
    h.enqueue(urls(6)).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    vote(&h, &h.validator, "pass", None).await;
    let listed = read(&h, "/v1/tasks").await["tasks"][0].clone();
    assert_eq!((listed["miner_uid"].clone(), listed["validator_uid"].clone()), (json!(7), json!(2)));
    assert_eq!((listed["miner_coldkey"].clone(), listed["validator_coldkey"].clone()), (json!("ck"), json!("cv")));
    let miners = read(&h, "/v1/miners").await["miners"].clone();
    assert_eq!((miners[0]["uid"].clone(), miners[0]["coldkey"].clone()), (json!(7), json!("ck")));
    let validators = read(&h, "/v1/validators").await["validators"].clone();
    assert_eq!((validators[0]["uid"].clone(), validators[0]["coldkey"].clone()), (json!(2), json!("cv")));
    let view = h.view(&task).await;
    assert_eq!((view["miner_uid"].clone(), view["score"]["validator_uid"].clone(), view["votes"][0]["validator_uid"].clone()), (json!(7), json!(2), json!(2)));
    let miner = read(&h, &format!("/v1/miners/{}", h.miner.ss58())).await;
    assert_eq!((miner["uid"].clone(), miner["coldkey"].clone()), (json!(7), json!("ck")));
    let validator = read(&h, &format!("/v1/validators/{}", h.validator.ss58())).await;
    assert_eq!((validator["uid"].clone(), validator["coldkey"].clone()), (json!(2), json!("cv")));
    let first = read(&h, "/v1/votes").await["votes"][0].clone();
    assert_eq!((first["miner_uid"].clone(), first["validator_uid"].clone()), (json!(7), json!(2)));
    assert_eq!(first["miner_coldkey"], "ck");
}

#[tokio::test]
async fn the_miners_page_lists_only_hotkeys_on_the_subnet_once_the_metagraph_is_known() {
    let fake = crate::fakechain::FakeChain::start(1, 12.0).await;
    fake.refuse(true);
    let h = Harness::with(|s| s.registry = task_api::settings::RegistryMode::Chain { netuid: 22, network: fake.url.clone() }).await;
    h.state
        .db
        .run(|conn| {
            for hotkey in ["5Deregistered", "5OnTheSubnet"] {
                conn.execute("INSERT INTO miners (hotkey, pool, budget, verified) VALUES (?, 'crawl', 1, 0)", [hotkey])?;
            }
            Ok(())
        })
        .await
        .unwrap();
    h.state.registry.restore(vec![task_api::registry::Entry { hotkey: "5OnTheSubnet".into(), uid: Some(4), is_validator: false, coldkey: None }]);
    let listed = h.public("/v1/miners").await.1;
    let hotkeys: Vec<&str> = listed["miners"].as_array().unwrap().iter().filter_map(|m| m["hotkey"].as_str()).collect();
    assert_eq!(hotkeys, ["5OnTheSubnet"]);
    assert_eq!(listed["miners"][0]["uid"], 4);
}

async fn tallied(h: &Harness) {
    h.state
        .db
        .run(|conn| {
            let at = desearch::time::now();
            task_api::hours::roll_up_claims(conn, 1000, at)?;
            task_api::hours::roll_up_checks(conn, 1000, at)
        })
        .await
        .expect("the tallies are counted");
}

#[tokio::test]
async fn each_hour_shows_the_work_handed_out_the_claims_refused_and_how_checks_went() {
    let h = started(2).await;
    let passed = task_id(&h.mine(&h.miner, 0).await);
    vote(&h, &h.validator, "pass", Some(&passed)).await;
    let failed = task_id(&h.mine(&h.miner, 0).await);
    vote(&h, &h.validator, "fail", Some(&failed)).await;
    h.mine(&h.rival, 0).await;
    assert_eq!(h.miner.claim().await["tasks"].as_array().unwrap().len(), 1);
    assert_eq!(h.rival.claim().await["refusal"]["code"], "QUEUE_EMPTY");
    tallied(&h).await;

    let hours = read(&h, "/v1/stats/hours?hours=3").await;
    let points = hours["points"].as_array().unwrap();
    assert_eq!((points.len(), hours["kind"].clone(), hours["miner"].clone()), (3, json!("crawl"), Value::Null));
    let now = &points[2];
    assert_eq!(
        (now["issued"].clone(), now["completed"].clone(), now["requeued"].clone(), now["refused"].clone(), now["miners"].clone()),
        (json!(4), json!(3), json!(1), json!({"QUEUE_EMPTY": 1}), json!(2))
    );
    assert_eq!((now["tasks"].clone(), now["pass"].clone(), now["fail"].clone(), now["checked"].clone()), (json!(2), json!(1), json!(1), json!(2)));
    assert_eq!((now["verdict_s"]["checked"]["tasks"].clone(), now["verdict_s"]["unchecked"].clone()), (json!(2), Value::Null));
    assert_eq!((points[0]["issued"].clone(), points[0]["refused"].clone(), points[0]["verdict_s"]["checked"].clone()), (json!(0), json!({}), Value::Null));

    let rival = read(&h, &format!("/v1/stats/hours?hours=1&miner={}", h.rival.ss58())).await["points"][0].clone();
    assert_eq!((rival["issued"].clone(), rival["refused"].clone(), rival["tasks"].clone()), (json!(1), json!({"QUEUE_EMPTY": 1}), json!(0)));
    assert!(rival.get("checked").is_none() && rival.get("miners").is_none(), "the subnet's checks are not one miner's");

    let overview = read(&h, "/v1/overview").await;
    assert_eq!((overview["claims"]["issued"].clone(), overview["claims"]["refused"].clone()), (json!(4), json!({"QUEUE_EMPTY": 1})));
    let checks = &overview["checks"];
    assert_eq!((checks["checked"].clone(), checks["verdict_s"]["checked"]["tasks"].clone()), (json!(2), json!(2)));
    assert!(checks["reasons"].as_array().unwrap().contains(&json!({"verdict": "fail", "reason": "content_mismatch", "checked": true, "tasks": 1})));
    assert_eq!((overview["validators"]["agreement"].clone(), overview["locked_out"].clone()), (json!(1.0), json!(0)));
    assert_eq!(overview["taken_back"], json!({"take_backs": 0, "tasks": 0, "miners": 0, "credited": 0}));

    let miner = read(&h, &format!("/v1/miners/{}", h.miner.ss58())).await;
    let claims = &miner["kinds"]["crawl"]["claims"];
    assert_eq!((claims["issued"].clone(), claims["completed"].clone(), claims["refused"].clone()), (json!(3), json!(2), json!({})));
    assert_eq!((miner["kinds"]["crawl"]["window"]["fail"].clone(), miner["take_backs"].clone()), (json!(1), json!([])));
    let fails = read(&h, "/v1/tasks?verdict=fail").await["tasks"].clone();
    assert_eq!(field(&fails, "task_id"), [json!(failed)]);
    assert_eq!(read(&h, &format!("/v1/tasks?verdict=fail&miner={}", h.rival.ss58())).await["tasks"], json!([]));
}

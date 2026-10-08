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
    assert_eq!(overview["validators"], json!({"active": 1, "known": 1}));
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
async fn hotkeys_come_with_their_uids() {
    let h = Harness::with(|s| s.registry = RegistryMode::Chain { netuid: 22, network: "finney".into() }).await;
    h.state.registry.load(vec![neuron(&h.miner, 7, "ck", false), neuron(&h.validator, 2, "cv", true)]);
    h.enqueue(urls(6)).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    vote(&h, &h.validator, "pass", None).await;
    let listed = read(&h, "/v1/tasks").await["tasks"][0].clone();
    assert_eq!((listed["miner_uid"].clone(), listed["validator_uid"].clone()), (json!(7), json!(2)));
    let view = h.view(&task).await;
    assert_eq!((view["miner_uid"].clone(), view["score"]["validator_uid"].clone(), view["votes"][0]["validator_uid"].clone()), (json!(7), json!(2), json!(2)));
    let miner = read(&h, &format!("/v1/miners/{}", h.miner.ss58())).await;
    assert_eq!((miner["uid"].clone(), miner["coldkey"].clone()), (json!(7), json!("ck")));
    let validator = read(&h, &format!("/v1/validators/{}", h.validator.ss58())).await;
    assert_eq!((validator["uid"].clone(), validator["coldkey"].clone()), (json!(2), json!("cv")));
    let first = read(&h, "/v1/votes").await["votes"][0].clone();
    assert_eq!((first["miner_uid"].clone(), first["validator_uid"].clone()), (json!(7), json!(2)));
}

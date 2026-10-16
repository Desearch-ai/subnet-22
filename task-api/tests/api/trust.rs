//! Validators' votes: who must vote, majorities, audits, deadlines and what is paid once.

use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::manifest::REVEAL_AFTER_BLOCKS;
use task_api::queues::VOPEN;
use task_api::state::State;
use task_api::{lifecycle, roundlog, rounds, roundstore, validations};

use crate::harness::{judged, score, task_id, urls, Client, Harness};

async fn started(task_urls: usize) -> Harness {
    let h = Harness::with(|s| s.task_urls = task_urls).await;
    h.enqueue(urls(6)).await;
    h
}

async fn standing(h: &Harness) -> Value {
    json!(h.state.db.run(validations::audit_standing).await.unwrap())
}

async fn verdict(h: &Harness, validator: &Client, verdict: &str, task: Option<&str>, overrides: Value) -> Value {
    judged(h, validator, verdict, task, overrides).await.1.expect("the vote is taken")
}

#[tokio::test]
async fn one_active_validator_finalizes_alone() {
    let h = started(3).await;
    let task = h.mine(&h.miner, 0).await;
    let answer = verdict(&h, &h.validator, "pass", None, json!({})).await;
    assert_eq!((answer["verdict"].clone(), answer["credited"].clone()), (json!("pass"), json!(3)));
    let view = h.view(&task_id(&task)).await;
    assert_eq!(view["status"], "pass");
    assert!(h.page(view["score"]["report_key"].as_str().unwrap()).unwrap().get("votes").is_none());
}

#[tokio::test]
async fn a_second_active_validator_must_vote_before_a_task_finalizes() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    let second = task_id(&h.mine(&h.rival, 0).await);
    let pending = verdict(&h, &h.other_validator, "pass", Some(&second), json!({})).await;
    assert_eq!((pending["verdict"].clone(), h.status(&second).await), (json!("pending"), "voting".to_string()));
    let finalized = verdict(&h, &h.validator, "pass", Some(&second), json!({})).await;
    assert_eq!((finalized["verdict"].clone(), finalized["credited"].clone()), (json!("pass"), json!(3)));
    let report = h.page(h.view(&second).await["score"]["report_key"].as_str().unwrap()).unwrap();
    assert_eq!(report["votes"].as_array().unwrap().iter().map(|v| v["verdict"].clone()).collect::<Vec<_>>(), [json!("pass"), json!("pass")]);
    for (_, standing) in standing(&h).await.as_object().unwrap() {
        assert_eq!(standing, &json!({"audits": 1, "disagreements": 0, "excluded": false}));
    }
}

#[tokio::test]
async fn a_report_on_a_finalized_upload_still_makes_the_validator_present() {
    let h = started(3).await;
    let first = task_id(&h.mine(&h.miner, 0).await);
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    let late = h.other_validator.post(&format!("/v1/validation/{first}/score"), score("pass", 3, "ok", None, "x")).await;
    assert_eq!(late.unwrap_err().status, 409);
    let second = task_id(&h.mine(&h.rival, 0).await);
    let pending = verdict(&h, &h.validator, "pass", Some(&second), json!({})).await;
    let last = verdict(&h, &h.other_validator, "pass", Some(&second), json!({})).await;
    assert_eq!((pending["verdict"].clone(), last["verdict"].clone()), (json!("pending"), json!("pass")));
}

#[tokio::test]
async fn the_open_list_drops_an_upload_once_it_is_finalized() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    h.opened(None, None).await;
    assert_eq!(h.open_list().await["uploads"].as_array().unwrap().len(), 1);
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    assert_eq!(h.open_list().await["uploads"], json!([]));
}

/// Three tasks, so that every validator has voted once and the third is open.
async fn three_active(h: &Harness) -> String {
    let first = task_id(&h.mine(&h.miner, 0).await);
    verdict(h, &h.validator, "pass", Some(&first), json!({})).await;
    let second = task_id(&h.mine(&h.rival, 0).await);
    verdict(h, &h.other_validator, "pass", Some(&second), json!({})).await;
    verdict(h, &h.validator, "pass", Some(&second), json!({})).await;
    let third = task_id(&h.mine(&h.miner, 0).await);
    verdict(h, &h.third_validator, "pass", Some(&third), json!({})).await;
    third
}

#[tokio::test]
async fn a_disputed_task_is_decided_by_the_majority_and_the_minority_is_marked() {
    let h = started(2).await;
    let task = three_active(&h).await;
    let still = verdict(&h, &h.validator, "fail", Some(&task), json!({})).await;
    assert_eq!(still["verdict"], "pending", "three validators are active, so three votes");
    let last = verdict(&h, &h.other_validator, "pass", Some(&task), json!({})).await;
    assert_eq!((last["verdict"].clone(), last["credited"].clone(), h.status(&task).await), (json!("pass"), json!(2), "pass".to_string()));
    let standing = standing(&h).await;
    assert_eq!(standing[h.validator.ss58()]["disagreements"], 1);
    let mut disagreements: Vec<i64> = standing.as_object().unwrap().values().map(|s| s["disagreements"].as_i64().unwrap()).collect();
    disagreements.sort();
    assert_eq!(disagreements, [0, 0, 1]);
}

#[tokio::test]
async fn two_validators_that_disagree_void_the_task() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    let task = task_id(&h.mine(&h.rival, 0).await);
    verdict(&h, &h.other_validator, "fail", Some(&task), json!({})).await;
    let void = verdict(&h, &h.validator, "pass", Some(&task), json!({})).await;
    let view = h.view(&task).await;
    assert_eq!(
        (void["verdict"].clone(), view["status"].clone(), view["score"]["reason"].clone()),
        (json!("void"), json!("queued"), json!("validators_disagree"))
    );
}

#[tokio::test]
async fn a_task_past_its_deadline_is_decided_on_the_votes_it_has() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    let task = task_id(&h.mine(&h.rival, 0).await);
    let pending = verdict(&h, &h.other_validator, "pass", Some(&task), json!({})).await;
    assert_eq!(pending["verdict"], "pending");
    assert!(lifecycle::finalize_due(&h.state, desearch::time::now()).await.unwrap().is_empty(), "a missing vote is waited for");
    let mut job = h.state.validation.job(&h.redis, &task).await.unwrap().unwrap();
    job["deadline"] = (desearch::time::now() - 1.0).into();
    let _: () = h.redis.clone().set(format!("vjob:{task}"), job.to_string()).await.unwrap();
    assert_eq!(lifecycle::finalize_due(&h.state, desearch::time::now()).await.unwrap().len(), 1);
    assert_eq!(h.status(&task).await, "pass", "never voided for a slow validator");
}

#[tokio::test]
async fn an_upload_no_validator_voted_on_by_its_deadline_passes_on_its_report() {
    let h = started(3).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    h.opened(None, Some(&task)).await;
    let mut job = h.state.validation.job(&h.redis, &task).await.unwrap().unwrap();
    job["deadline"] = 0.into();
    let _: () = h.redis.clone().set(format!("vjob:{task}"), job.to_string()).await.unwrap();
    assert_eq!(lifecycle::finalize_due(&h.state, desearch::time::now()).await.unwrap(), [task.as_str()]);
    let view = h.view(&task).await;
    assert_eq!((view["status"].clone(), view["score"]["reason"].clone(), view["score"]["credited"].clone()), (json!("pass"), json!("unchecked"), json!(3)));
    assert_eq!(h.state.publish.depth(&h.redis).await.unwrap(), 1);
}

#[tokio::test]
async fn a_lowballed_pass_among_three_is_the_odd_one_out() {
    let h = started(2).await;
    let task = three_active(&h).await;
    verdict(&h, &h.validator, "pass", Some(&task), json!({})).await;
    let lowball = json!({"sampled": 2, "matched": 1, "mismatched": 1, "samples": [{"url": "", "outcome": "matched"}, {"url": "", "outcome": "mismatched"}]});
    let last = verdict(&h, &h.other_validator, "pass", Some(&task), lowball).await;
    assert_eq!((last["verdict"].clone(), last["credited"].clone()), (json!("pass"), json!(2)));
    assert_eq!(standing(&h).await[h.other_validator.ss58()]["disagreements"], 1);
}

#[tokio::test]
async fn a_report_the_task_could_not_have_produced_is_refused() {
    let h = started(3).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    assert_eq!(judged(&h, &h.validator, "pass", None, json!({"returned": 0})).await.1.unwrap_err().status, 422);
    assert_eq!(h.status(&task).await, "open", "the refused report changed nothing");
}

async fn exclude(h: &Harness, validator: &Client) {
    let hotkey = validator.ss58();
    h.state.db.run(move |conn| (0..10).try_for_each(|_| validations::record_audit(conn, &[], std::slice::from_ref(&hotkey)))).await.unwrap();
}

#[tokio::test]
async fn an_excluded_validator_cannot_score() {
    let h = started(3).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    exclude(&h, &h.validator).await;
    let job = h.opened(Some(&h.other_validator), None).await;
    let refused = h.validator.post(&format!("/v1/validation/{task}/score"), score("pass", 3, "ok", None, job["urls"][0].as_str().unwrap())).await;
    assert_eq!(refused.unwrap_err().status, 403);
    assert_eq!(h.status(&task).await, "open");
}

#[tokio::test]
async fn an_excluded_validator_no_longer_holds_up_the_others() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    exclude(&h, &h.validator).await;
    let task = task_id(&h.mine(&h.rival, 0).await);
    assert_eq!(h.validator.post(&format!("/v1/validation/{task}/score"), score("pass", 3, "ok", None, "x")).await.unwrap_err().status, 403);
    let last = verdict(&h, &h.other_validator, "pass", Some(&task), json!({})).await;
    assert_eq!(last["verdict"], "pass");
    assert_eq!(h.state.validation.active(&h.redis, desearch::time::now()).await.unwrap(), [h.other_validator.ss58()]);
}

#[tokio::test]
async fn a_task_that_keeps_failing_is_dropped_after_its_last_attempt() {
    let h = Harness::with(|s| s.max_attempts = 2).await;
    h.enqueue(urls(6)).await;
    let task = h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "fail", None, json!({})).await;
    let retry = h.mine(&h.rival, 0).await;
    verdict(&h, &h.validator, "fail", None, json!({})).await;
    assert_eq!(task_id(&retry), task_id(&task));
    assert!(h.state.payload(&task_id(&task)).await.unwrap().is_none());
    let round_id = task["round_id"].as_str().unwrap().to_string();
    let asked = round_id.clone();
    let entries = h.state.db.run(move |conn| roundlog::entries(conn, &asked)).await.unwrap();
    assert!(entries.iter().any(|e| e["outcome"] == "dropped"));
    let open: i64 = h.redis.clone().scard(lifecycle::open_round_key(&round_id)).await.unwrap();
    assert_eq!(open, 1, "the other task of the round is still out");
}

#[tokio::test]
async fn nothing_compared_is_void_and_confirmed_errors_are_paid_but_not_published() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    let void = verdict(&h, &h.validator, "pass", None, score("pass", 3, "ok", Some("unverifiable"), "")).await;
    h.mine(&h.rival, 3).await;
    let errors = verdict(&h, &h.validator, "pass", None, score("pass", 3, "ok", Some("errors_confirmed"), "")).await;
    assert_eq!((void["verdict"].clone(), void["credited"].clone()), (json!("void"), json!(0)));
    assert_eq!((errors["verdict"].clone(), errors["credited"].clone()), (json!("pass"), json!(3)));
    assert_eq!(h.state.publish.depth(&h.redis).await.unwrap(), 0);
}

#[tokio::test]
async fn a_validators_own_timeout_costs_the_miner_nothing() {
    let h = started(3).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    let overrides = json!({"reason": "unscorable", "returned": 0, "sampled": 0, "matched": 0, "mismatched": 0, "samples": []});
    let answer = verdict(&h, &h.validator, "fail", None, overrides).await;
    assert_eq!((answer["verdict"].clone(), answer["credited"].clone()), (json!("void"), json!(0)));
    let view = h.view(&task).await;
    assert_eq!((view["status"].clone(), view["score"]["reason"].clone()), (json!("queued"), json!("unscorable")));
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!((miner["coverage"].clone(), miner["transitions"].clone()), (json!({}), json!([])));
}

#[tokio::test]
async fn two_reports_for_one_task_at_once_leave_one_verdict_and_its_report() {
    let h = started(3).await;
    let task = task_id(&h.mine(&h.miner, 0).await);
    let job = h.opened(Some(&h.validator), None).await;
    let path = format!("/v1/validation/{task}/score");
    let body = score("pass", 3, "ok", None, job["urls"][0].as_str().unwrap());
    let (one, two) = tokio::join!(h.validator.post(&path, body.clone()), h.validator.post(&path, body));
    let outcomes: Vec<u16> = [one, two].into_iter().map(|r| r.map_or_else(|refused| refused.status, |_| 200)).collect();
    assert!(outcomes.contains(&200) && outcomes.iter().all(|s| [200, 409, 503].contains(s)), "{outcomes:?}");
    assert_eq!(h.page(h.view(&task).await["score"]["report_key"].as_str().unwrap()).unwrap()["verdict"], "pass");
    let health = h.public("/v1/health").await.1;
    assert_eq!((health["verdicts"]["pass"].clone(), health["verdicts"]["fail"].clone()), (json!(1), json!(0)));
}

#[tokio::test]
async fn a_final_verdict_recorded_before_redis_closed_the_upload_is_paid_once() {
    let h = started(3).await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.other_validator, "pass", None, json!({})).await;
    let task = task_id(&h.mine(&h.rival, 0).await);
    let pending = verdict(&h, &h.validator, "pass", Some(&task), json!({})).await;
    assert_eq!(pending["verdict"], "pending", "the other validator is still to vote");
    let job = h.state.validation.job(&h.redis, &task).await.unwrap().unwrap();
    let completed: f64 = h.redis.clone().zscore(VOPEN, &task).await.unwrap();
    let _: i64 = h.redis.clone().zrem(VOPEN, &task).await.unwrap();
    assert!(lifecycle::finalize_task(&h.state, &task, &job, f64::MAX).await.unwrap().is_none(), "closed under the verdict, so not finished");
    let once = h.public(&format!("/v1/miners/{}", h.rival.ss58())).await.1;
    assert_eq!(once["pools"]["crawl"]["verified"], 3, "finalized before Redis lost it");
    let _: i64 = h.redis.clone().zadd(VOPEN, &task, completed).await.unwrap();
    let mut due = job.clone();
    due["deadline"] = 0.into();
    let _: () = h.redis.clone().set(format!("vjob:{task}"), due.to_string()).await.unwrap();
    assert_eq!(lifecycle::finalize_due(&h.state, desearch::time::now()).await.unwrap(), [task.as_str()]);
    assert_eq!(h.status(&task).await, "pass");
    let miner = h.public(&format!("/v1/miners/{}", h.rival.ss58())).await.1;
    assert_eq!(miner["pools"]["crawl"]["verified"], 3, "closed later, not paid again");
    let packed: String = h.redis.clone().get(format!("pjob:{task}")).await.unwrap();
    assert_eq!(crate::unpack(&packed)["urls"], job["urls"], "published with the URLs its kept copy left out");
}

#[tokio::test]
async fn a_completion_receipt_names_the_block_the_upload_was_frozen_at() {
    let h = started(3).await;
    let task = h.mine(&h.miner, 0).await;
    let job = h.opened(Some(&h.validator), None).await;
    let round_id = task["round_id"].as_str().unwrap().to_string();
    let entries = h.state.db.run(move |conn| roundlog::entries(conn, &round_id)).await.unwrap();
    let completed = entries.iter().find(|e| e["outcome"] == "completed").unwrap();
    assert_eq!(job["seed_block"].as_i64().unwrap(), completed["block"].as_i64().unwrap() + REVEAL_AFTER_BLOCKS);
}

#[tokio::test]
async fn a_round_revealed_before_redis_took_its_tasks_is_filled_afterwards() {
    let h = Harness::start().await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": urls(6)})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    let asked = round_id.clone();
    let mut round = h.state.db.run(move |conn| roundstore::get(conn, &asked)).await.unwrap().unwrap();
    while h.state.seeds.current_block().await.unwrap() < round.seed_block {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let seed = h.state.seeds.seed_for(round.seed_block).await.unwrap().unwrap();
    rounds::reveal(&mut round, &seed);
    let saved = round.clone();
    h.state.db.run(move |conn| roundstore::save(conn, &saved)).await.unwrap();
    let unfilled: Vec<String> = h.state.db.run(roundstore::unfilled).await.unwrap().into_iter().map(|r| r.round_id).collect();
    assert_eq!(unfilled, [round_id.as_str()]);
    assert!(lifecycle::close_finished(&h.state).await.unwrap().is_empty(), "not closed as served");
    assert_eq!(lifecycle::fill_missing(&h.state).await.unwrap(), [round_id.as_str()]);
    assert_eq!(h.state.crawl.depth(&h.redis).await.unwrap(), 2);
    assert_eq!(h.miner.claim().await["tasks"][0]["round_id"], round_id.as_str());
}

#[tokio::test]
async fn polling_after_a_round_closes_leaves_its_proof_intact() {
    let h = Harness::start().await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": [{"host": "a.example", "url": "https://a.example/1"}]})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    h.revealed().await;
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let asked = round_id.clone();
    let anchored = h.state.db.run(move |conn| roundlog::anchored_root(conn, &asked)).await.unwrap();
    let idle = h.miner.claim().await;
    assert_eq!(idle["refusal"]["code"], "QUEUE_EMPTY");
    assert_eq!(idle["receipt"]["body"]["round_id"], "", "refused outside any round");
    let log = h.public(&format!("/v1/rounds/{round_id}/log")).await.1;
    let leaves: Vec<Vec<u8>> = log["entries"].as_array().unwrap().iter().map(task_api::py::canonical).collect();
    assert_eq!(Some(task_api::proofs::merkle_root(&leaves)), anchored);
    assert_eq!(log["anchor_root"], json!(anchored));
}

#[tokio::test]
async fn rounds_survive_a_restart_and_close_once_every_task_is_decided() {
    let h = Harness::start().await;
    let enqueued = h.admin.post("/v1/admin/enqueue", json!({"urls": [{"host": "a.example", "url": "https://a.example/1"}]})).await.unwrap();
    let round_id = enqueued["round_id"].as_str().unwrap().to_string();
    let restarted = State::new(h.state.settings.clone(), h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.unwrap();
    let pending: Vec<String> = restarted.db.run(|conn| roundstore::unrevealed(conn, i64::MAX, 1000)).await.unwrap().into_iter().map(|r| r.round_id).collect();
    assert_eq!(pending, [round_id.as_str()]);
    while restarted.db.run(|conn| roundstore::unrevealed(conn, i64::MAX, 1000)).await.unwrap().len() == 1 {
        lifecycle::reveal_pending(&restarted).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(restarted.crawl.depth(&h.redis).await.unwrap(), 1);
    assert!(lifecycle::close_finished(&h.state).await.unwrap().is_empty());
    h.mine(&h.miner, 0).await;
    verdict(&h, &h.validator, "pass", None, json!({})).await;
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    assert!(!h.public(&format!("/v1/rounds/{round_id}")).await.1["closed_at"].is_null());
}

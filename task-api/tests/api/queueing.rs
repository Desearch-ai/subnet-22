//! The Redis queues on their own: each step one script, so concurrent callers never see half of it.

use std::io::Read;

use base64::Engine;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};
use task_api::queues::{pack_job, waiting_key, Claim, PublishQueue, Refusal, TaskQueue, ValidationQueue, PACKED, PCLAIMS, PPENDING, PUBLISH, QUEUE, VOPEN};
use task_api::state::Redis;

use crate::harness::Harness;

const FAR: f64 = 1e12;

async fn filled(redis: &Redis, kind: &'static str, count: usize, round_id: &str) -> (TaskQueue, Vec<String>) {
    let queue = TaskQueue::new(kind, 60.0);
    let order: Vec<String> = (0..count).map(|n| if kind == "crawl" { format!("{round_id}-t{n}") } else { format!("{kind}-{round_id}-t{n}") }).collect();
    let payloads: Map<String, Value> = order.iter().map(|t| (t.clone(), json!({"url_count": 1, "urls": [format!("https://x.example/{t}")]}))).collect();
    queue.fill(redis, round_id, &order, &payloads).await.unwrap();
    (queue, order)
}

async fn one(queue: &TaskQueue, redis: &Redis, hotkey: &str, budget: i64) -> Result<Claim, Refusal> {
    queue.claim(redis, hotkey, budget, 2 * budget, 1, desearch::time::now()).await.unwrap().map(|mut claims| claims.remove(0))
}

/// A claim with its upload key issued, so it can be completed.
async fn claimed(queue: &TaskQueue, redis: &Redis, hotkey: &str, budget: i64) -> Claim {
    let got = one(queue, redis, hotkey, budget).await.unwrap();
    let _: () = redis.clone().set(format!("issued:{}", got.task_id), json!({"key": format!("k-{}", got.seq)}).to_string()).await.unwrap();
    got
}

async fn complete(queue: &TaskQueue, redis: &Redis, got: &Claim, hotkey: &str, key: &str) -> Option<i64> {
    queue.complete(redis, &got.task_id, hotkey, &json!({"task_id": got.task_id, "miner": hotkey}), key, desearch::time::now()).await.unwrap()
}

#[tokio::test]
async fn concurrent_claims_cannot_exceed_the_budget() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 10, "r1").await;
    let claims = futures::future::join_all((0..10).map(|_| one(&queue, &h.redis, "m", 1))).await;
    let refused: Vec<&str> = claims.iter().filter_map(|c| c.as_ref().err().map(|r| r.code)).collect();
    assert_eq!((claims.len() - refused.len(), refused.iter().filter(|c| **c == "NO_CAPACITY").count()), (1, 9));
}

#[tokio::test]
async fn a_completed_task_can_never_return_to_the_queue() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 3, "r1").await;
    let got = claimed(&queue, &h.redis, "m", 5).await;
    assert!(complete(&queue, &h.redis, &got, "m", &format!("k-{}", got.seq)).await.is_some());
    assert!(queue.reclaim(&h.redis, &got.task_id, FAR).await.unwrap().is_none());
    assert!(queue.abandon(&h.redis, &got.task_id, "m").await.unwrap().is_none());
    let payload: Option<String> = h.redis.clone().get(format!("task:{}", got.task_id)).await.unwrap();
    let queued: Option<f64> = h.redis.clone().zscore(QUEUE, &got.task_id).await.unwrap();
    assert!(payload.is_none() && queued.is_none());
}

#[tokio::test]
async fn only_an_expired_claim_is_reclaimed_and_only_by_its_holder_abandoned() {
    let h = Harness::start().await;
    let (queue, order) = filled(&h.redis, "crawl", 3, "r1").await;
    let got = claimed(&queue, &h.redis, "m", 5).await;
    assert!(queue.reclaim(&h.redis, &got.task_id, desearch::time::now()).await.unwrap().is_none());
    assert_eq!(queue.reclaim(&h.redis, &got.task_id, FAR).await.unwrap().map(|(holder, _)| holder), Some("m".to_string()));
    let head: Vec<String> = h.redis.clone().zrange(QUEUE, 0, 0).await.unwrap();
    assert_eq!(head, [order[0].clone()]);
    assert_eq!(queue.in_flight(&h.redis, "m").await.unwrap(), 0);
    let again = claimed(&queue, &h.redis, "n", 5).await;
    assert!(queue.abandon(&h.redis, &again.task_id, "rival").await.unwrap().is_none());
    assert!(queue.abandon(&h.redis, &again.task_id, "n").await.unwrap().is_some());
}

#[tokio::test]
async fn completion_needs_the_issued_key_and_a_live_claim() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 3, "r1").await;
    let got = claimed(&queue, &h.redis, "m", 5).await;
    assert!(complete(&queue, &h.redis, &got, "m", "someone-else").await.is_none());
    let _: i64 = h.redis.clone().zadd("claims:expiry", &got.task_id, 1).await.unwrap();
    assert!(complete(&queue, &h.redis, &got, "m", &format!("k-{}", got.seq)).await.is_none());
}

#[tokio::test]
async fn older_rounds_are_served_first_and_a_reclaimed_task_goes_back_ahead() {
    let h = Harness::start().await;
    let (queue, first) = filled(&h.redis, "crawl", 3, "r1").await;
    let (_, second) = filled(&h.redis, "crawl", 3, "r2").await;
    let mut served = Vec::new();
    for _ in 0..6 {
        served.push(one(&queue, &h.redis, "m", 10).await.unwrap().task_id);
    }
    assert_eq!(served, [first.clone(), second].concat());
    let reclaimed = served[0].clone();
    queue.reclaim(&h.redis, &reclaimed, FAR).await.unwrap();
    filled(&h.redis, "crawl", 2, "r3").await;
    assert_eq!(one(&queue, &h.redis, "n", 10).await.unwrap().task_id, reclaimed);
}

#[tokio::test]
async fn a_miner_is_never_given_back_a_task_it_held() {
    let h = Harness::start().await;
    let (queue, order) = filled(&h.redis, "crawl", 2, "r1").await;
    let got = claimed(&queue, &h.redis, "m", 5).await;
    queue.reclaim(&h.redis, &got.task_id, FAR).await.unwrap();
    assert_eq!(got.task_id, order[0]);
    assert_eq!(one(&queue, &h.redis, "m", 5).await.unwrap().task_id, order[1]);
    assert_eq!(one(&queue, &h.redis, "n", 5).await.unwrap().task_id, order[0]);
}

#[tokio::test]
async fn a_miner_that_held_every_waiting_task_is_told_so() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 1, "r1").await;
    let got = claimed(&queue, &h.redis, "m", 5).await;
    queue.abandon(&h.redis, &got.task_id, "m").await.unwrap();
    let refusal = one(&queue, &h.redis, "m", 5).await.unwrap_err();
    assert_eq!((refusal.code, Value::Object(refusal.inputs)), ("ALREADY_HELD", json!({"depth": 1, "held": 1})));
}

async fn validation_job(redis: &Redis, task_id: &str, at: f64) {
    let _: () = redis.clone().set(format!("vjob:{task_id}"), json!({"task_id": task_id, "miner": "m", "kind": "crawl"}).to_string()).await.unwrap();
    let _: i64 = redis.clone().zadd(VOPEN, task_id, at).await.unwrap();
    let _: i64 = redis.clone().sadd(waiting_key("crawl", "m"), task_id).await.unwrap();
}

#[tokio::test]
async fn one_vote_per_validator_per_open_upload() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 3600.0 };
    validation_job(&h.redis, "a", 1.0).await;
    validation_job(&h.redis, "b", 2.0).await;
    assert_eq!(validation.open_ids(&h.redis, 500).await.unwrap(), ["a", "b"]);
    let now = desearch::time::now();
    assert_eq!(validation.vote(&h.redis, "a", "v", &json!({"validator": "v"}), now).await.unwrap(), 1);
    assert_eq!(validation.vote(&h.redis, "a", "v", &json!({"validator": "v"}), now).await.unwrap(), 0, "one vote per validator per upload");
    assert_eq!(validation.voters(&h.redis, "a").await.unwrap(), ["v"]);
    assert!(validation.voters(&h.redis, "b").await.unwrap().is_empty());
}

#[tokio::test]
async fn reporting_marks_the_validator_active_for_the_window() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 100.0 };
    validation_job(&h.redis, "t", 1.0).await;
    validation.present(&h.redis, "u", 990.0).await.unwrap();
    validation.vote(&h.redis, "t", "v", &json!({"validator": "v"}), 1000.0).await.unwrap();
    validation.vote(&h.redis, "t", "w", &json!({"validator": "w"}), 1050.0).await.unwrap();
    let mut active = validation.active(&h.redis, 1080.0).await.unwrap();
    active.sort();
    assert_eq!(active, ["u", "v", "w"]);
    assert_eq!(validation.active(&h.redis, 1120.0).await.unwrap(), ["w"]);
    assert!(validation.active(&h.redis, 1200.0).await.unwrap().is_empty());
}

#[tokio::test]
async fn finalizing_closes_the_upload_and_hands_back_the_votes() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 3600.0 };
    validation_job(&h.redis, "t", 1.0).await;
    let now = desearch::time::now();
    validation.vote(&h.redis, "t", "v", &json!({"validator": "v", "verdict": "pass"}), now).await.unwrap();
    assert_eq!(validation.vote(&h.redis, "t", "w", &json!({"validator": "w", "verdict": "fail"}), now).await.unwrap(), 2);
    let finalized = validation.finalize(&h.redis, "t", None, false, now).await.unwrap().unwrap();
    assert_eq!(finalized.job["task_id"], "t");
    assert_eq!(finalized.votes.iter().map(|v| v["validator"].clone()).collect::<Vec<_>>(), [json!("v"), json!("w")]);
    assert!(validation.finalize(&h.redis, "t", None, false, now).await.unwrap().is_none());
    assert!(validation.job(&h.redis, "t").await.unwrap().is_none());
    assert_eq!(validation.depth(&h.redis).await.unwrap(), 0);
}

#[tokio::test]
async fn a_finalize_without_votes_hands_back_none() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 3600.0 };
    validation_job(&h.redis, "t", 1.0).await;
    let finalized = validation.finalize(&h.redis, "t", None, false, desearch::time::now()).await.unwrap().unwrap();
    assert!(finalized.votes.is_empty(), "Lua encodes an empty list as an object");
}

#[tokio::test]
async fn the_oldest_open_upload_sets_the_backlog_age() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 2, "r1").await;
    let validation = ValidationQueue { active_s: 3600.0 };
    let first = claimed(&queue, &h.redis, "m", 5).await;
    let second = claimed(&queue, &h.redis, "m", 5).await;
    complete(&queue, &h.redis, &first, "m", &format!("k-{}", first.seq)).await.unwrap();
    let _: i64 = h.redis.clone().zadd(VOPEN, &first.task_id, 1).await.unwrap();
    complete(&queue, &h.redis, &second, "m", &format!("k-{}", second.seq)).await.unwrap();
    let now = desearch::time::now();
    assert!(validation.oldest_age(&h.redis, now).await.unwrap() > 1e9);
    validation.finalize(&h.redis, &first.task_id, None, false, now).await.unwrap();
    assert!(validation.oldest_age(&h.redis, now).await.unwrap() < 10.0);
}

fn unpack(packed: &str) -> Value {
    let compressed = base64::engine::general_purpose::STANDARD.decode(packed.strip_prefix(PACKED).unwrap()).unwrap();
    let mut json = String::new();
    flate2::read::ZlibDecoder::new(compressed.as_slice()).read_to_string(&mut json).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[tokio::test]
async fn a_pass_ends_validation_and_queues_publishing_in_one_step() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 3600.0 };
    validation_job(&h.redis, "t", 1.0).await;
    let publish = json!({"task_id": "t", "key": "k"});
    let passed = validation.finalize(&h.redis, "t", Some(&publish), false, desearch::time::now()).await.unwrap().unwrap();
    assert_eq!(passed.job["task_id"], "t");
    assert!(validation.job(&h.redis, "t").await.unwrap().is_none());
    let ready: Vec<String> = h.redis.clone().lrange(PUBLISH, 0, -1).await.unwrap();
    let packed: String = h.redis.clone().get("pjob:t").await.unwrap();
    assert_eq!((ready, unpack(&packed)), (vec!["t".to_string()], publish));
    let waiting: bool = h.redis.clone().sismember("waiting:m", "t").await.unwrap();
    assert!(!waiting, "a decided upload no longer counts as waiting");
}

#[tokio::test]
async fn a_publish_that_keeps_failing_is_set_aside_and_stops_counting_as_backlog() {
    let h = Harness::start().await;
    let validation = ValidationQueue { active_s: 3600.0 };
    let publish = PublishQueue { claim_ttl: 60.0, max_tries: 2 };
    validation_job(&h.redis, "t", 1.0).await;
    validation.finalize(&h.redis, "t", Some(&json!({"task_id": "t", "completed_at": 1000.0})), false, desearch::time::now()).await.unwrap();
    assert!(publish.oldest_age(&h.redis, desearch::time::now()).await.unwrap() > 1e9 - 1e5);
    let mut outcomes = Vec::new();
    for _ in 0..2 {
        let _: Option<String> = h.redis.clone().lpop(PUBLISH, None).await.unwrap();
        let _: i64 = h.redis.clone().zadd(PCLAIMS, "t", 0).await.unwrap();
        assert_eq!(publish.expired(&h.redis, desearch::time::now()).await.unwrap(), ["t"]);
        outcomes.push(publish.give_back(&h.redis, "t").await.unwrap());
    }
    assert_eq!(outcomes, [1, -1]);
    assert_eq!(publish.dead_count(&h.redis).await.unwrap(), 1);
    assert_eq!(publish.oldest_age(&h.redis, desearch::time::now()).await.unwrap(), 0.0);
    let pending: Option<f64> = h.redis.clone().zscore(PPENDING, "t").await.unwrap();
    assert!(pending.is_none());
}

#[tokio::test]
async fn an_upload_waiting_for_its_verdict_leaves_room_to_crawl_but_counts_as_waiting() {
    let h = Harness::start().await;
    let (queue, _) = filled(&h.redis, "crawl", 4, "r1").await;
    let validation = ValidationQueue { active_s: 3600.0 };
    let got = claimed(&queue, &h.redis, "m", 1).await;
    complete(&queue, &h.redis, &got, "m", &format!("k-{}", got.seq)).await.unwrap();
    let crawling = queue.claim(&h.redis, "m", 1, 2, 1, desearch::time::now()).await.unwrap().unwrap().remove(0);
    let _: () = h.redis.clone().set(format!("issued:{}", crawling.task_id), json!({"key": "k-x"}).to_string()).await.unwrap();
    complete(&queue, &h.redis, &crawling, "m", "k-x").await.unwrap();
    let blocked = queue.claim(&h.redis, "m", 1, 2, 1, desearch::time::now()).await.unwrap().unwrap_err();
    assert_eq!(blocked.code, "WAITING_FOR_VERDICTS");
    validation.finalize(&h.redis, &got.task_id, None, true, desearch::time::now()).await.unwrap();
    assert!(queue.claim(&h.redis, "m", 1, 2, 1, desearch::time::now()).await.unwrap().is_ok());
    assert_eq!(queue.waiting(&h.redis, "m").await.unwrap(), 1);
}

#[tokio::test]
async fn one_claim_takes_as_many_tasks_as_asked_and_the_budget_allows() {
    let h = Harness::start().await;
    let (queue, order) = filled(&h.redis, "crawl", 10, "r1").await;
    let ids = |claims: Vec<Claim>| claims.into_iter().map(|c| c.task_id).collect::<Vec<_>>();
    let three = ids(queue.claim(&h.redis, "m", 5, 10, 3, desearch::time::now()).await.unwrap().unwrap());
    let rest = ids(queue.claim(&h.redis, "m", 5, 10, 50, desearch::time::now()).await.unwrap().unwrap());
    assert_eq!((three, rest), (order[..3].to_vec(), order[3..5].to_vec()));
    assert_eq!(queue.claim(&h.redis, "m", 5, 10, 1, desearch::time::now()).await.unwrap().unwrap_err().code, "NO_CAPACITY");
}

#[tokio::test]
async fn each_kind_serves_only_its_own_tasks_and_counts_its_own_budget() {
    let h = Harness::start().await;
    let (crawl, crawl_tasks) = filled(&h.redis, "crawl", 2, "r1").await;
    let (embed, embed_tasks) = filled(&h.redis, "embed", 2, "r1").await;
    let got_crawl = one(&crawl, &h.redis, "m", 1).await.unwrap();
    let got_embed = one(&embed, &h.redis, "m", 1).await.unwrap();
    assert_eq!((got_crawl.task_id.clone(), got_crawl.payload["kind"].clone()), (crawl_tasks[0].clone(), json!("crawl")));
    assert_eq!((got_embed.task_id.clone(), got_embed.payload["kind"].clone()), (embed_tasks[0].clone(), json!("embed")));
    assert_eq!(one(&crawl, &h.redis, "m", 1).await.unwrap_err().code, "NO_CAPACITY", "an embed task does not use up the crawl budget");
    let (holder, _) = embed.reclaim(&h.redis, &got_embed.task_id, FAR).await.unwrap().unwrap();
    let held: i64 = h.redis.clone().scard("inflight:embed:m").await.unwrap();
    assert_eq!((holder.as_str(), held, embed.depth(&h.redis).await.unwrap(), crawl.depth(&h.redis).await.unwrap()), ("m", 0, 2, 1));
}

#[tokio::test]
async fn a_miner_that_held_the_whole_front_of_the_queue_is_served_from_behind() {
    let h = Harness::start().await;
    let (queue, order) = filled(&h.redis, "crawl", 60, "r1").await;
    for task_id in &order[..55] {
        let _: i64 = h.redis.clone().sadd(format!("holders:{task_id}"), "m").await.unwrap();
    }
    assert_eq!(one(&queue, &h.redis, "m", 5).await.unwrap().task_id, order[55]);
}

#[test]
fn a_waiting_publish_job_is_kept_compressed() {
    let job = json!({"task_id": "t", "urls": (0..1000).map(|n| format!("https://site{n}.example/page/{n}")).collect::<Vec<_>>()});
    let packed = pack_job(&job);
    assert_eq!(unpack(&packed), job);
    assert!(packed.len() * 3 < job.to_string().len());
}

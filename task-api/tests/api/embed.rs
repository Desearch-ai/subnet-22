//! Embed tasks: rounds from the planner's inputs, their own pool, room, claim time and electorate, only for miners running the model, and closed until switched on.

use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::lifecycle;
use task_api::queues::{self, EMBED_DROPPED, EMBED_INPUTS};

use crate::harness::{judged, task_id, urls, Client, Harness};

fn pages() -> Value {
    json!((0..2).map(|i| json!({"url": format!("https://site{i}.example/story/{i}"), "host": format!("site{i}.example"), "page_key": format!("pages/{i:040x}.zst"), "content_sha1": format!("{i:040x}")})).collect::<Vec<_>>())
}

fn input() -> Value {
    json!({"input_key": "embed-inputs/dt=2026-09-24/task=crawl-1.parquet", "input_sha256": "ab".repeat(32), "texts": 6, "chars": 5400, "pages": pages()})
}

fn embed_score(verdict: &str, outcome: &str, reason: &str) -> Value {
    json!({
        "returned": 6,
        "sampled": 2,
        "matched": if outcome == "matched" { 2 } else { 0 },
        "mismatched": if outcome == "mismatched" { 2 } else { 0 },
        "min_similarity": if outcome == "matched" { 0.999 } else { 0.12 },
        "verdict": verdict,
        "reason": reason,
        "samples": (0..2).map(|i| json!({"text_id": format!("t{i}"), "outcome": outcome, "similarity": 0.999})).collect::<Vec<_>>(),
    })
}

async fn embedding() -> Harness {
    Harness::with(|s| s.embed_tasks = true).await
}

async fn embed_round(h: &Harness, entry: Value) -> String {
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    let round = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    round.round_id
}

fn embed_claim() -> Value {
    json!({"kind": "embed", "models": ["desearch-4b-v1"]})
}

async fn embed_task(h: &Harness, miner: &Client) -> Value {
    let task = miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), b"PAR1vectorsPAR1".to_vec()).await;
    miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    task
}

#[tokio::test]
async fn a_published_page_is_embedded_credited_and_recorded() {
    let h = embedding().await;
    let round_id = embed_round(&h, input()).await;
    assert_eq!(h.miner.claim().await["refusal"]["code"], "QUEUE_EMPTY", "crawl sees no embed work");

    let task = embed_task(&h, &h.miner).await;
    assert_eq!((task["kind"].clone(), task["round_id"].clone()), (json!("embed"), json!(round_id)));
    assert_eq!((task["model"].clone(), task["texts"].clone()), (json!(h.state.settings.embed_model), json!(6)));
    assert_eq!(task["input"]["sha256"], input()["input_sha256"]);
    assert!(task["input"]["url"].as_str().unwrap().contains("X-Amz-Signature="));
    assert_eq!(task["urls"], json!(pages().as_array().unwrap().iter().map(|p| p["url"].clone()).collect::<Vec<_>>()));

    let job = h.opened(Some(&h.validator), None).await;
    let listed = h.open_list().await;
    assert_eq!(listed["uploads"].as_array().unwrap().iter().map(|m| m["kind"].clone()).collect::<Vec<_>>(), [json!("embed")]);
    assert_eq!(
        (job["kind"].clone(), job["model"].clone(), job["input_key"].clone()),
        (json!("embed"), json!(h.state.settings.embed_model), input()["input_key"].clone())
    );

    let scored = h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("pass", "matched", "ok")).await.unwrap();
    assert_eq!((scored["verdict"].clone(), scored["credited"].clone()), (json!("pass"), json!(5400)));
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!((miner["pools"]["embed"]["budget"].clone(), miner["pools"]["crawl"]["budget"].clone()), (json!(2), json!(1)));
    assert_eq!(h.public("/v1/shares").await.1["pools"], json!({"embed": {h.miner.ss58(): 1.0}}));

    let packed: String = h.redis.clone().get(format!("pjob:{}", task_id(&task))).await.unwrap();
    let publish = queues::unpack_job(&packed).expect("a publish job");
    assert_eq!(
        (publish["kind"].clone(), publish["model"].clone(), publish["input_key"].clone(), publish["texts"].clone()),
        (json!("embed"), json!(h.state.settings.embed_model), input()["input_key"].clone(), json!(6))
    );
    assert!(publish.get("pages").is_none() && publish.get("vectors_key").is_none(), "the publisher reads pages from the input file");

    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let published = h.public(&format!("/v1/rounds/{round_id}")).await.1;
    assert_eq!(published["kind"], "embed");
    assert_eq!(published["manifest"][0]["input_sha256"], input()["input_sha256"]);
    let manifest: Vec<Value> = published["manifest"].as_array().unwrap().clone();
    assert_eq!(task_api::proofs::manifest_hash(&manifest, published["seed_block"].as_i64().unwrap()), published["manifest_hash"].as_str().unwrap());
}

#[tokio::test]
async fn the_planner_gets_room_for_embed_inputs_apart_from_the_bots_crawl_room() {
    {
        let closed = Harness::start().await;
        assert_eq!(closed.public("/v1/room?kind=embed").await.1["room_tasks"], 0, "no room while embed tasks are closed");
    }
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_queue_target = 5;
    })
    .await;
    assert_eq!(h.public("/v1/room?kind=embed").await.1["room_tasks"], 5);
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, input().to_string()).await.unwrap();
    assert_eq!(h.public("/v1/room?kind=embed").await.1["room_tasks"], 4, "a waiting input takes room");
    lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    let unrevealed = h.public("/v1/room?kind=embed").await.1;
    assert_eq!((unrevealed["room_tasks"].clone(), unrevealed["unrevealed"].clone()), (json!(4), json!(1)));
    assert_eq!(h.public("/v1/room").await.1["unrevealed"], 0, "embed rounds take none of the crawl room");
    h.revealed().await;
    let queued = h.public("/v1/room?kind=embed").await.1;
    assert_eq!((queued["room_tasks"].clone(), queued["queue"].clone()), (json!(4), json!(1)));
}

#[tokio::test]
async fn an_embed_upload_waits_only_for_validators_that_check_embed_uploads() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.task_urls = 3;
    })
    .await;
    h.enqueue(urls(3)).await;
    let crawl = task_id(&h.mine(&h.miner, 0).await);
    let (_, scored) = judged(&h, &h.other_validator, "pass", Some(&crawl), json!({})).await;
    assert!(scored.is_ok(), "the other validator is active, on crawl uploads only");

    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.rival).await;
    h.opened(Some(&h.validator), Some(&task_id(&task))).await;
    let scored = h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("pass", "matched", "ok")).await.unwrap();
    assert_eq!(scored["verdict"], "pass", "decided on the first embed checker's vote, without waiting for the crawl-only validator");
}

#[tokio::test]
async fn a_failed_embed_task_strikes_only_the_embed_pool() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    h.opened(Some(&h.validator), None).await;
    let scored = h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("fail", "mismatched", "vectors_mismatch")).await.unwrap();
    assert_eq!(scored["verdict"], "fail");
    let hotkey = h.miner.ss58();
    let strikes: Vec<(String, String)> = h
        .state
        .db
        .run(move |conn| {
            Ok(conn
                .prepare("SELECT pool, reason FROM strikes WHERE hotkey = ?")?
                .query_map([hotkey], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap();
    assert_eq!(strikes, [("embed".to_string(), "vectors_mismatch".to_string())]);
    assert_eq!(h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["refusal"]["code"], "ALREADY_HELD");
    assert_eq!(h.rival.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0]["task_id"], task["task_id"]);
}

#[tokio::test]
async fn embed_tasks_stay_closed_until_switched_on() {
    let h = Harness::start().await;
    let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, input().to_string()).await.unwrap();
    assert!(lifecycle::open_embed_rounds(&h.state).await.unwrap().is_none());
    let left: i64 = h.redis.clone().llen(EMBED_INPUTS).await.unwrap();
    assert_eq!(left, 1, "nothing is thrown away");
    let answer = h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap();
    assert_eq!(answer["refusal"], json!({"code": "KIND_CLOSED", "inputs": {"kind": "embed", "retry_after": 3600.0}}));
    assert!(!answer["receipt"].is_null());
    let health = h.public("/v1/health").await.1;
    assert_eq!((health["embed_tasks"].clone(), health["embed_model"].clone()), (json!(false), json!("desearch-4b-v1")));
}

#[tokio::test]
async fn a_miner_that_does_not_run_the_model_is_refused_at_no_cost() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    for claim in [json!({"kind": "embed"}), json!({"kind": "embed", "models": ["qwen3-embedding-8b"]})] {
        let answer = h.miner.post("/v1/tasks/claim", claim).await.unwrap();
        assert_eq!(answer["refusal"], json!({"code": "MODEL_MISMATCH", "inputs": {"model": "desearch-4b-v1", "retry_after": 3600.0}}));
    }
    let bad = h.miner.post("/v1/tasks/claim", json!({"kind": "embed", "models": [7]})).await.unwrap_err();
    assert_eq!(bad.status, 422);
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!(miner["pools"]["embed"]["budget"], json!(1), "a refusal is not a strike");
    assert_eq!(h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn an_embed_task_has_its_own_claim_time() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let claimed_at = now();
    let task = h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0].clone();
    let left = task["expires_at"].as_f64().unwrap() - claimed_at;
    assert!((h.state.settings.embed_claim_ttl - 1.0..=h.state.settings.embed_claim_ttl + 1.0).contains(&left), "{left}");
    assert!(h.state.settings.embed_claim_ttl > h.state.settings.claim_ttl);
}

#[tokio::test]
async fn an_embed_upload_no_validator_scored_by_its_deadline_is_void_and_goes_to_another_miner() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    let task_id = task_id(&task);
    h.opened(None, Some(&task_id)).await;
    let mut job = h.state.validation.job(&h.redis, &task_id).await.unwrap().unwrap();
    let key = job["key"].as_str().unwrap().to_string();
    assert!(h.object(&key).is_some());
    job["deadline"] = 0.into();
    let _: () = h.redis.clone().set(format!("vjob:{task_id}"), job.to_string()).await.unwrap();

    assert_eq!(lifecycle::finalize_due(&h.state, now()).await.unwrap(), [task_id.as_str()]);

    let view = h.view(&task_id).await;
    assert_eq!(
        (view["status"].clone(), view["score"]["verdict"].clone(), view["score"]["reason"].clone()),
        (json!("queued"), json!("void"), json!("unscored"))
    );
    assert!(h.object(&key).is_none(), "the upload is deleted, so the next miner cannot copy it");
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!(miner["pools"]["embed"]["budget"], json!(1));
    let hotkey = h.miner.ss58();
    let strikes: i64 =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap();
    assert_eq!(strikes, 0);
    h.revealed().await;
    assert_eq!(h.rival.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0]["task_id"], json!(task_id));
}

#[tokio::test]
async fn an_embed_task_dropped_after_its_last_attempt_hands_its_pages_back_to_the_planner() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.max_attempts = 1;
    })
    .await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    h.opened(Some(&h.validator), None).await;
    h.validator.post(&format!("/v1/validation/{}/score", task_id(&task)), embed_score("fail", "mismatched", "vectors_mismatch")).await.unwrap();

    let dropped: Vec<String> = h.redis.clone().lrange(EMBED_DROPPED, 0, -1).await.unwrap();
    let dropped: Vec<Value> = dropped.iter().map(|raw| serde_json::from_str(raw).unwrap()).collect();
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0]["input_key"], input()["input_key"]);
    let given: Vec<(Value, Value)> = dropped[0]["pages"].as_array().unwrap().iter().map(|p| (p["page_key"].clone(), p["content_sha1"].clone())).collect();
    let planned: Vec<(Value, Value)> = pages().as_array().unwrap().iter().map(|p| (p["page_key"].clone(), p["content_sha1"].clone())).collect();
    assert_eq!(given, planned);
    assert_eq!(h.rival.post("/v1/tasks/claim", embed_claim()).await.unwrap()["refusal"]["code"], "QUEUE_EMPTY");
}

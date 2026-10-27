//! Embed tasks: rounds from the planner's inputs, their own pool, room, claim time and electorate, only for miners running the model, and closed until switched on.

use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::budgets::{self, CRAWL, EMBED};
use task_api::manifest::upload_log_key;
use task_api::queues::{self, EMBED_DROPPED, EMBED_INPUTS, EMBED_MALFORMED, PWITHDRAWN};
use task_api::verdicts::UPLOAD_CHANGED;
use task_api::{checks, lifecycle, sampling, uploadlog};

use crate::checks::{established_in, settled};
use crate::harness::{judged, task_id, urls, vectors, Client, Harness};

fn pages() -> Value {
    json!((0..2).map(|i| json!({"url": format!("https://site{i}.example/story/{i}"), "host": format!("site{i}.example"), "page_key": format!("pages/{i:040x}.zst"), "content_sha1": format!("{i:040x}")})).collect::<Vec<_>>())
}

fn input() -> Value {
    json!({"input_key": "embed-inputs/dt=2026-09-24/task=crawl-1.parquet", "input_sha256": "ab".repeat(32), "texts": 6, "chars": 5400, "pages": pages()})
}

/// A check of every one of the input's six texts, each with the same outcome.
fn embed_score(verdict: &str, outcome: &str, reason: &str) -> Value {
    let texts = input()["texts"].as_i64().unwrap();
    let counted = |name: &str| if outcome == name { texts } else { 0 };
    json!({
        "sampled": texts,
        "matched": counted("matched"),
        "mismatched": counted("mismatched"),
        "malformed": counted("malformed"),
        "min_similarity": if outcome == "matched" { 0.999 } else { 0.12 },
        "verdict": verdict,
        "reason": reason,
        "samples": (0..texts).map(|i| json!({"text_id": format!("t{i}"), "index": i, "outcome": outcome, "similarity": 0.999})).collect::<Vec<_>>(),
    })
}

async fn embedding() -> Harness {
    Harness::with(|s| s.embed_tasks = true).await
}

/// Embed tasks are drawn for a check as rarely as the share allows.
async fn embedding_unchecked() -> Harness {
    Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_check_share = 0.0;
    })
    .await
}

async fn embed_round(h: &Harness, entry: Value) -> String {
    embed_rounds(h, vec![entry]).await
}

async fn embed_rounds(h: &Harness, entries: Vec<Value>) -> String {
    for entry in entries {
        let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    }
    let round = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("an embed round");
    h.revealed().await;
    round.round_id
}

/// Inputs of the same pages under keys of their own, one task each.
fn inputs(count: usize) -> Vec<Value> {
    (0..count)
        .map(|i| {
            let mut entry = input();
            entry["input_key"] = format!("embed-inputs/dt=2026-09-24/task=crawl-{i}.parquet").into();
            entry
        })
        .collect()
}

fn embed_claim() -> Value {
    json!({"kind": "embed", "models": ["desearch-4b-v1"]})
}

async fn embed_task(h: &Harness, miner: &Client) -> Value {
    let task = miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0].clone();
    h.put(task["upload"]["url"].as_str().unwrap(), vectors(6)).await;
    miner.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
    task
}

async fn score(h: &Harness, task: &str, body: Value) -> Value {
    h.validator.post(&format!("/v1/validation/{task}/score"), body).await.unwrap()
}

async fn credits(h: &Harness, pool: &'static str) -> i64 {
    let hotkey = h.miner.ss58();
    h.state
        .db
        .run(move |conn| {
            Ok(conn.query_row("SELECT COALESCE(SUM(amount), 0) FROM credits WHERE hotkey = ? AND pool = ?", [hotkey.as_str(), pool], |row| row.get(0))?)
        })
        .await
        .unwrap()
}

async fn rechecks(h: &Harness) -> (i64, i64) {
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| Ok((checks::recheck_left(conn, &hotkey, CRAWL)?, checks::recheck_left(conn, &hotkey, EMBED)?))).await.unwrap()
}

/// The miner's strikes, each with its pool and reason.
async fn strikes(h: &Harness) -> Vec<(String, String)> {
    let hotkey = h.miner.ss58();
    h.state
        .db
        .run(move |conn| {
            Ok(conn
                .prepare("SELECT pool, reason FROM strikes WHERE hotkey = ?")?
                .query_map([hotkey], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap()
}

async fn locked(h: &Harness) -> (Option<f64>, Option<f64>) {
    let hotkey = h.miner.ss58();
    let at = now();
    h.state.db.run(move |conn| Ok((budgets::locked_until(conn, &hotkey, CRAWL, at)?, budgets::locked_until(conn, &hotkey, EMBED, at)?))).await.unwrap()
}

#[tokio::test]
async fn a_published_page_is_embedded_credited_and_recorded() {
    let h = embedding().await;
    let round_id = embed_round(&h, input()).await;
    assert_eq!(h.miner.claim().await["refusal"]["code"], "QUEUE_EMPTY", "crawl sees no embed work");

    let asked = round_id.clone();
    let stored: String =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [asked], |row| row.get(0))?)).await.unwrap();
    assert!(!stored.contains("page_key") && stored.contains("input_sha256"), "a queued round keeps its input but no page list");

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
    assert_eq!(publish["pages"], pages(), "the planner's pages, each with its host");
    let etag = h.state.storage.head(task["upload"]["key"].as_str().unwrap()).await.unwrap().unwrap().etag;
    assert_eq!((publish["key"].clone(), publish["etag"].clone()), (task["upload"]["key"].clone(), json!(etag)));
    assert_eq!(job["etag"], json!(etag));

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
    assert_eq!(strikes(&h).await, [("embed".to_string(), "vectors_mismatch".to_string())]);
    assert_eq!(h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["refusal"]["code"], "ALREADY_HELD");
    assert_eq!(h.rival.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0]["task_id"], task["task_id"]);
}

#[tokio::test]
async fn an_embed_upload_written_again_after_completion_fails_its_check_and_strikes_the_miner() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = task_id(&embed_task(&h, &h.miner).await);
    h.opened(Some(&h.validator), None).await;
    let changed = json!({"verdict": "fail", "reason": UPLOAD_CHANGED, "sampled": 0, "matched": 0, "mismatched": 0, "malformed": 0, "min_similarity": null, "samples": []});
    assert_eq!(score(&h, &task, changed).await["verdict"], "fail");
    assert_eq!(h.view(&task).await["score"]["reason"], UPLOAD_CHANGED);
    assert_eq!(strikes(&h).await, [("embed".to_string(), UPLOAD_CHANGED.to_string())]);
    assert_eq!(rechecks(&h).await, (0, sampling::RECHECK_UPLOADS), "its next embed uploads are checked");
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

#[tokio::test]
async fn the_explorer_shows_an_embed_task_s_checked_texts_and_counts_each_kind_apart() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    assert_eq!(h.view(&task_id(&task)).await["kind"], "embed", "the kind shows before any verdict");
    let live = h.public("/v1/live").await.1;
    assert_eq!((live["uploads"][0]["kind"].clone(), live["uploads"][0]["texts"].clone()), (json!("embed"), json!(6)));

    h.opened(Some(&h.validator), None).await;
    let keys: Vec<String> = pages().as_array().unwrap().iter().map(|p| p["page_key"].as_str().unwrap().to_string()).collect();
    let mut checked = embed_score("pass", "matched", "ok");
    checked["samples"][0] = json!({"text_id": format!("{}#page", keys[0]), "index": 0, "outcome": "matched", "similarity": 0.9999});
    checked["samples"][1] = json!({"text_id": format!("{}#p2", keys[1]), "index": 3, "outcome": "matched", "similarity": 0.99995});
    checked["samples"][3]["index"] = 1.into();
    score(&h, &task_id(&task), checked).await;

    let view = h.view(&task_id(&task)).await;
    assert_eq!(view["urls"], json!([]));
    assert_eq!(view["texts"].as_array().unwrap().len(), 6);
    assert_eq!(
        view["texts"].as_array().unwrap()[..2],
        [
            json!({"text_id": format!("{}#page", keys[0]), "kind": "page", "index": 0, "row": 0, "url": "https://site0.example/story/0", "outcome": "matched", "similarity": 0.9999}),
            json!({"text_id": format!("{}#p2", keys[1]), "kind": "passage", "index": 2, "row": 3, "url": "https://site1.example/story/1", "outcome": "matched", "similarity": 0.99995}),
        ]
    );

    let hotkey = h.miner.ss58();
    let row = |list: &Value| list["miners"].as_array().unwrap().iter().find(|m| m["hotkey"] == hotkey.as_str()).cloned();
    let embed_miners = h.public("/v1/miners?kind=embed").await.1;
    let embedded = row(&embed_miners).expect("the embed miner is listed");
    assert_eq!((embed_miners["kind"].clone(), embedded["credited"].clone(), embedded["share"].clone()), (json!("embed"), json!(5400), json!(1.0)));
    assert_eq!(row(&h.public("/v1/miners").await.1).map_or(json!(0), |m| m["tasks"].clone()), json!(0), "crawl lists no embed work");

    let (crawl, embed) = (h.public("/v1/overview").await.1, h.public("/v1/overview?kind=embed").await.1);
    assert_eq!((crawl["kind"].clone(), crawl["window"]["tasks"].clone()), (json!("crawl"), json!(0)));
    assert_eq!((embed["kind"].clone(), embed["window"]["tasks"].clone(), embed["window"]["credited"].clone()), (json!("embed"), json!(1), json!(5400)));
    assert_eq!(h.public("/v1/overview?kind=search").await.0, 422);

    let miner = h.public(&format!("/v1/miners/{hotkey}")).await.1;
    assert_eq!((miner["kinds"]["embed"]["window"]["credited"].clone(), miner["kinds"]["embed"]["share"].clone()), (json!(5400), json!(1.0)));
    assert_eq!((miner["kinds"]["crawl"]["window"]["tasks"].clone(), miner["window"]["tasks"].clone()), (json!(0), json!(0)));
    let series = h.public("/v1/stats/series?kind=embed&bucket_minutes=5&buckets=2").await.1;
    assert_eq!(series["points"].as_array().unwrap().iter().map(|p| p["tasks"].as_i64().unwrap()).sum::<i64>(), 1);
}

#[tokio::test]
async fn an_embed_upload_is_one_float16_vector_per_text_and_its_size_is_checked_not_its_report() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0].clone();
    let key = task["upload"]["key"].as_str().unwrap();
    assert!(key.starts_with("uploads/task=") && key.ends_with(".f16"), "{key}");
    assert_eq!(task["upload"]["content_type"], "application/octet-stream");
    let url = task["upload"]["url"].as_str().unwrap();
    let complete = format!("/v1/tasks/{}/complete", task_id(&task));
    let report = json!({"key": key, "rows": 999, "ok": 999, "errors": 9, "bytes": 1});

    h.put(url, vectors(5)).await;
    assert_eq!(h.miner.post(&complete, report.clone()).await.unwrap_err().status, 422, "a vector short");
    h.put(url, [vectors(6), vec![0; 2]].concat()).await;
    assert_eq!(h.miner.post(&complete, report.clone()).await.unwrap_err().status, 422, "a value over");
    h.put(url, vectors(6)).await;
    assert_eq!(h.miner.post(&complete, report).await.unwrap()["status"], "open_for_validation", "the right size, whatever the report says");

    let job = h.state.validation.job(&h.redis, &task_id(&task)).await.unwrap().unwrap();
    assert_eq!(job["key"], key, "the upload is pinned where the miner put it");
    assert_eq!(h.object(key), Some(vectors(6)));
    assert!(job.get("reported").is_none(), "the miner's counts are not kept");
    let hotkey = h.miner.ss58();
    let strikes: i64 =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap();
    assert_eq!(strikes, 0, "a refused size is not a strike");
    let hour = budgets::hour_of(now());
    let counted: (Option<i64>, Option<i64>) = redis::pipe()
        .get(sampling::counter_key(EMBED, &h.miner.ss58(), hour))
        .get(sampling::counter_key(CRAWL, &h.miner.ss58(), hour))
        .query_async(&mut h.redis.clone())
        .await
        .unwrap();
    assert_eq!(counted, (Some(1), None), "counted among embed uploads only");
}

#[tokio::test]
async fn every_upload_of_a_hotkey_new_to_embed_is_checked_whatever_its_crawl_record() {
    let h = embedding_unchecked().await;
    established_in(&h, h.miner.ss58(), CRAWL).await;
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    assert_eq!(settled(&h, &task_id(&task)).await, sampling::NEW);
}

#[tokio::test]
async fn an_embed_upload_no_check_drew_is_paid_its_characters_and_published() {
    let h = embedding_unchecked().await;
    established_in(&h, h.miner.ss58(), EMBED).await;
    embed_round(&h, input()).await;
    let task = task_id(&embed_task(&h, &h.miner).await);
    assert_eq!(settled(&h, &task).await, "finalized");
    let view = h.view(&task).await;
    assert_eq!(
        (view["status"].clone(), view["score"]["reason"].clone(), view["score"]["credited"].clone(), view["score"]["validator"].clone()),
        (json!("pass"), json!("unchecked"), json!(5400), json!(""))
    );
    assert_eq!((credits(&h, EMBED).await, credits(&h, CRAWL).await), (5400, 0));
    let packed: String = h.redis.clone().get(format!("pjob:{task}")).await.unwrap();
    let publish = queues::unpack_job(&packed).expect("a publish job");
    assert_eq!(
        (publish["kind"].clone(), publish["model"].clone(), publish["input_key"].clone(), publish["texts"].clone()),
        (json!("embed"), json!("desearch-4b-v1"), input()["input_key"].clone(), json!(6))
    );
    assert_eq!(publish["pages"], pages());
    let miner = h.public(&format!("/v1/miners/{}", h.miner.ss58())).await.1;
    assert_eq!((miner["pools"]["embed"]["budget"].clone(), miner["pools"]["embed"]["waiting"].clone()), (json!(2), json!(0)));
}

#[tokio::test]
async fn a_share_of_one_checks_every_embed_upload_whatever_the_volume() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_check_share = 1.0;
        s.embed_checks_per_hour = 1.0;
    })
    .await;
    established_in(&h, h.miner.ss58(), EMBED).await;
    let _: () = h.redis.clone().set(sampling::counter_key(EMBED, sampling::ALL, budgets::hour_of(now())), 1_000_000_000i64).await.unwrap();
    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    assert_eq!(settled(&h, &task_id(&task)).await, sampling::DRAW);
}

#[tokio::test]
async fn a_failed_embed_check_takes_back_the_unchecked_embed_uploads_since_the_last_pass_and_checks_the_next_ones() {
    let h = embedding_unchecked().await;
    established_in(&h, h.miner.ss58(), EMBED).await;
    embed_rounds(&h, inputs(2)).await;
    let unchecked = task_id(&embed_task(&h, &h.miner).await);
    assert_eq!(settled(&h, &unchecked).await, "finalized");
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| checks::start_recheck(conn, &hotkey, EMBED)).await.unwrap();
    let checked = task_id(&embed_task(&h, &h.miner).await);
    assert_eq!(settled(&h, &checked).await, sampling::RECHECK);
    assert_eq!(rechecks(&h).await, (0, sampling::RECHECK_UPLOADS - 1));

    let verdict = score(&h, &checked, embed_score("fail", "mismatched", "vectors_mismatch")).await;
    assert_eq!(verdict["verdict"], "fail");
    assert_eq!(h.view(&unchecked).await["score"]["verdict"], "withdrawn");
    assert_eq!(credits(&h, EMBED).await, 0, "its characters are taken back");
    let refused: Vec<String> = h.redis.clone().zrange(PWITHDRAWN, 0, -1).await.unwrap();
    assert_eq!(refused, [unchecked], "publishers take its vectors back");
    assert_eq!(rechecks(&h).await, (0, sampling::RECHECK_UPLOADS), "its next embed uploads are checked, its crawl ones as before");
    assert_eq!(locked(&h).await, (None, None), "one fail is not a penalty");
}

#[tokio::test]
async fn a_second_embed_fail_among_the_last_checks_costs_the_embed_day_and_locks_out_of_embed_only() {
    let h = embedding_unchecked().await;
    let hotkey = h.miner.ss58();
    established_in(&h, hotkey.clone(), EMBED).await;
    h.state.db.run(move |conn| budgets::credit(conn, &hotkey, 100, CRAWL, now())).await.unwrap();
    embed_rounds(&h, inputs(3)).await;
    let unchecked = task_id(&embed_task(&h, &h.miner).await);
    settled(&h, &unchecked).await;
    for _ in 0..2 {
        let hotkey = h.miner.ss58();
        h.state.db.run(move |conn| checks::start_recheck(conn, &hotkey, EMBED)).await.unwrap();
        let task = task_id(&embed_task(&h, &h.miner).await);
        assert_eq!(settled(&h, &task).await, sampling::RECHECK);
        score(&h, &task, embed_score("fail", "malformed", "vectors_malformed")).await;
    }
    let (crawl, embed) = locked(&h).await;
    assert!(embed.expect("locked out of embed") - now() > 47.0 * 3600.0);
    assert_eq!(crawl, None);
    assert_eq!((credits(&h, EMBED).await, credits(&h, CRAWL).await), (0, 100));
    assert_eq!(h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["refusal"]["code"], "LOCKED_OUT");
}

#[tokio::test]
async fn every_completed_embed_upload_is_logged_with_its_texts_and_characters() {
    let h = embedding().await;
    embed_round(&h, input()).await;
    let task = task_id(&embed_task(&h, &h.miner).await);
    let job = h.state.validation.job(&h.redis, &task).await.unwrap().unwrap();
    assert_eq!(uploadlog::flush(&h.state, &h.state.storage).await.unwrap(), Some(1));
    let written = h.json_object(&upload_log_key(1)).unwrap();
    assert_eq!(
        written["entries"],
        json!([{"kind": "embed", "key": job["key"], "task_id": task, "miner": h.miner.ss58(), "completed_at": job["completed_at"], "texts": 6, "chars": 5400}])
    );
}

#[tokio::test]
async fn malformed_vectors_the_publisher_finds_get_the_miner_s_next_embed_uploads_checked() {
    let h = embedding_unchecked().await;
    established_in(&h, h.miner.ss58(), EMBED).await;
    let reports = [json!({"task_id": "published-1", "miner": h.miner.ss58(), "malformed": 3}).to_string(), "not a report".into()];
    let _: i64 = h.redis.clone().rpush(EMBED_MALFORMED, &reports).await.unwrap();
    assert_eq!(lifecycle::recheck_malformed(&h.state).await.unwrap(), 1);
    let left: i64 = h.redis.clone().llen(EMBED_MALFORMED).await.unwrap();
    assert_eq!(left, 0);
    assert_eq!(rechecks(&h).await, (0, sampling::RECHECK_UPLOADS));
    let hotkey = h.miner.ss58();
    let (strikes, verdicts): (i64, i64) = h
        .state
        .db
        .run(move |conn| {
            let strikes = conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [&hotkey], |row| row.get(0))?;
            Ok((strikes, conn.query_row("SELECT COUNT(*) FROM validations WHERE miner = ?", [&hotkey], |row| row.get(0))?))
        })
        .await
        .unwrap();
    assert_eq!((strikes, verdicts), (0, 0), "no strike and no verdict");

    embed_round(&h, input()).await;
    let task = embed_task(&h, &h.miner).await;
    assert_eq!(settled(&h, &task_id(&task)).await, sampling::RECHECK);
}

#[tokio::test]
async fn the_planner_s_room_leaves_out_embed_uploads_awaiting_a_verdict() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_queue_target = 5;
    })
    .await;
    embed_round(&h, input()).await;
    let task = task_id(&embed_task(&h, &h.miner).await);
    h.opened(Some(&h.validator), Some(&task)).await;
    let waiting = h.public("/v1/room?kind=embed").await.1;
    assert_eq!((waiting["awaiting_verdict"].clone(), waiting["queue"].clone(), waiting["room_tasks"].clone()), (json!(1), json!(0), json!(4)));
    score(&h, &task, embed_score("pass", "matched", "ok")).await;
    let decided = h.public("/v1/room?kind=embed").await.1;
    assert_eq!((decided["awaiting_verdict"].clone(), decided["room_tasks"].clone()), (json!(0), json!(5)));
}

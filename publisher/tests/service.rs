//! The service against a local Redis (db 12) and the R2 stub: claims, acknowledgements, lost uploads, withdrawals, feeds and their holes.

mod common;

use std::collections::{BTreeSet, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Int32Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use base64::Engine;
use bytes::Bytes;
use common::{page, parquet};
use desearch::r2::{self, Bucket, Credentials, PARQUET};
use desearch::stub;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use publisher::index::{Current, VersionIndex};
use publisher::service::{self, Metrics, Settings, Shared};
use publisher::snapshot::Multipart;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde_json::{json, Value};
use tokio::sync::watch;

const REDIS: &str = "redis://127.0.0.1:6379/12";
const TEMP: &str = "subnet-22";
const MODEL: &str = "desearch-4b-v1";
const PAGES: &str = "desearch-pages";
const PREFIX: &str = "tp/";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

static REDIS_TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The test database, emptied, held by one test at a time.
async fn redis_db() -> Option<(tokio::sync::MutexGuard<'static, ()>, ConnectionManager)> {
    let turn = REDIS_TURN.lock().await;
    let client = redis::Client::open(REDIS).ok()?;
    let mut redis = tokio::time::timeout(Duration::from_secs(2), client.get_connection_manager()).await.ok()?.ok()?;
    let () = redis::cmd("FLUSHDB").query_async(&mut redis).await.ok()?;
    Some((turn, redis))
}

fn bucket(addr: std::net::SocketAddr, name: &str, prefix: &str, errors: Arc<AtomicU64>) -> Bucket {
    let credentials = Credentials { access_key: "test".into(), secret_key: "test".into(), region: "auto".into() };
    Bucket::new(r2::client().unwrap(), &format!("http://{addr}"), name, prefix, credentials, errors).unwrap()
}

/// How the task API's `pack_job` stores a publish job.
fn pack(job: &Value) -> String {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(job.to_string().as_bytes()).unwrap();
    format!("z:{}", base64::engine::general_purpose::STANDARD.encode(encoder.finish().unwrap()))
}

/// What the task API's `finalize` does for an upload that passed validation.
async fn enqueue(redis: &mut ConnectionManager, job: &Value, packed: bool) {
    let task_id = job["task_id"].as_str().unwrap();
    let raw = if packed { pack(job) } else { job.to_string() };
    let () = redis.set(format!("pjob:{task_id}"), raw).await.unwrap();
    let _: i64 = redis.rpush("publish:ready", task_id).await.unwrap();
    let _: i64 = redis.zadd("publish:pending", task_id, job["completed_at"].as_f64().unwrap_or(0.0)).await.unwrap();
}

fn crawl(task_id: &str, urls: &[String], completed: f64, etag: &str) -> Value {
    json!({
        "task_id": task_id, "round_id": "r1", "miner": "5Miner", "kind": "crawl", "validator": "", "validators": [],
        "key": format!("uploads/{task_id}.parquet"), "etag": etag, "urls": urls, "skip": [],
        "completed_at": completed, "claim_ttl": 180,
    })
}

fn rows(file: &[u8]) -> Vec<RecordBatch> {
    ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(file)).unwrap().build().unwrap().map(Result::unwrap).collect()
}

fn column(batches: &[RecordBatch], name: &str) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| b.column_by_name(name).unwrap().as_string::<i32>().iter().map(|v| v.unwrap_or_default().to_string()).collect::<Vec<_>>())
        .collect()
}

fn json_object(state: &stub::State, bucket: &str, key: &str) -> Option<Value> {
    state.object(bucket, key).map(|body| serde_json::from_slice(&body).unwrap())
}

async fn publish_until_idle(shared: &Arc<Shared>) {
    let (_stopping, stop) = watch::channel(false);
    tokio::time::timeout(Duration::from_secs(60), service::run(shared.clone(), stop, 1)).await.expect("publishing finished").unwrap();
}

/// Every outcome row the feed numbered from `first` on, as (outcome, url).
fn outcome_feed(state: &stub::State, first: u64, last: u64) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for seq in first..=last {
        let index = json_object(state, TEMP, &format!("{PREFIX}outcomes/seq/{seq:012}.json")).unwrap();
        let batches = rows(&state.object(TEMP, &format!("{PREFIX}{}", index["key"].as_str().unwrap())).unwrap());
        assert_eq!(index["rows"].as_u64().unwrap() as usize, batches.iter().map(|b| b.num_rows()).sum::<usize>());
        found.extend(column(&batches, "outcome").into_iter().zip(column(&batches, "url")));
    }
    found
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_service_publishes_reports_and_takes_back() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("service");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings =
        Settings { redis_url: REDIS.into(), claim_ttl: 600.0, batch: 2, readers: 4, ranges: 4, ahead: 2, idle_exit: 0, idle_delay: Duration::from_millis(50) };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics.clone()).await.unwrap();

    let completed = service::now_us() as f64 / 1e6 - 60.0;
    let fetched = service::now_us() - 90_000_000;
    let urls = |task: &str| (0..3).map(|i| format!("https://ex{task}.com/story/{i}")).collect::<Vec<_>>();
    let upload = |task: &str, text: &str| parquet(&urls(task).iter().map(|u| page(u, &format!("{text} {u}"), fetched)).collect::<Vec<_>>(), None, None);
    for task in ["t1", "t2", "t4"] {
        temp.put(&format!("uploads/{task}.parquet"), Bytes::from(upload(task, "first")), PARQUET, None).await.unwrap();
    }
    let t1_etag = temp.head("uploads/t1.parquet").await.unwrap().unwrap().etag;
    enqueue(&mut redis, &crawl("t1", &urls("t1"), completed, &t1_etag), true).await;
    enqueue(&mut redis, &crawl("t2", &urls("t2"), completed, ""), false).await;
    enqueue(&mut redis, &crawl("t3", &urls("t3"), completed, ""), true).await;
    enqueue(&mut redis, &crawl("t4", &urls("t4"), completed, "\"not-this-one\""), true).await;
    state.fail("GET", &format!("{TEMP}/{PREFIX}uploads/t1"), 2);
    state.fail("PUT", &format!("{PAGES}/changes/seq/"), 5);

    publish_until_idle(&shared).await;

    let acked: u64 = redis.get("publish:acked").await.unwrap();
    assert_eq!(acked, 4, "t1, t2 and the lost t3 and t4");
    let claimed: Vec<String> = redis.zrange("publish:claims", 0, -1).await.unwrap();
    assert!(claimed.is_empty() && !redis.exists::<_, bool>("pjob:t1").await.unwrap());
    let lost: u64 = redis.get("publish:lost").await.unwrap();
    let lost_tasks: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert_eq!((lost, lost_tasks), (2, HashSet::from(["t3".to_string(), "t4".to_string()])), "t3 expired, t4 changed");
    assert!(state.keys(TEMP, &format!("{PREFIX}uploads/")).is_empty(), "finished jobs' uploads are deleted, the changed one too");
    assert!(metrics.r2_errors.load(Ordering::Relaxed) >= 7, "two GETs and five seq writes failed and were retried");

    let seq: u64 = redis.get("changes:seq").await.unwrap();
    assert_eq!(seq, 1);
    let holes: Vec<String> = redis.hkeys("changes:holes").await.unwrap();
    assert!(holes.is_empty(), "the hole left by the failed index write is filled in the same pass");
    let index = json_object(&state, PAGES, "changes/seq/000000000001.json").unwrap();
    assert_eq!(index["rows"], Value::Null, "a filled hole does not know its rows");
    let change_key = index["key"].as_str().unwrap().to_string();
    assert!(change_key.starts_with("changes/dt=") && change_key.ends_with(".parquet"));
    assert_eq!(state.object(PAGES, "changes/latest.json"), None, "latest moves only after a numbered index is written");
    let changes = rows(&state.object(PAGES, &change_key).unwrap());
    assert_eq!(column(&changes, "kind"), ["page"; 6]);
    assert_eq!(state.meta(PAGES, &change_key).unwrap().0, PARQUET);

    assert_eq!(redis.get::<_, u64>("outcomes:seq").await.unwrap(), 2, "the two batches that finished tasks");
    let outcomes = outcome_feed(&state, 1, 2);
    let count = |kind: &str| outcomes.iter().filter(|(k, _)| k == kind).count();
    assert_eq!((count("published"), count("failed"), outcomes.len()), (6, 6, 12));
    assert_eq!(json_object(&state, TEMP, &format!("{PREFIX}outcomes/latest.json")).unwrap(), json!({"seq": 2}));
    assert_eq!(state.meta(TEMP, &format!("{PREFIX}outcomes/latest.json")).unwrap().1, "no-store");

    temp.put("uploads/t6.parquet", Bytes::from(upload("t1", "first")), PARQUET, None).await.unwrap();
    enqueue(&mut redis, &crawl("t6", &urls("t1"), completed + 30.0, ""), true).await;
    let withdraw = json!({"task_id": "withdraw:0123456789abcdef", "kind": "withdraw", "task_ids": ["t2"], "reason": "test"});
    enqueue(&mut redis, &withdraw, false).await;
    enqueue(&mut redis, &crawl("t2", &urls("t2"), completed, ""), true).await;
    let (stopping, stop) = watch::channel(false);
    let running = tokio::spawn(service::run(shared.clone(), stop, 0));
    for _ in 0..400 {
        if redis.get::<_, u64>("publish:acked").await.unwrap() == 7 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    stopping.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(10), running).await.expect("stopping on request").unwrap().unwrap();

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 7);
    assert_eq!(json_object(&state, PAGES, "changes/latest.json").unwrap(), json!({"seq": 2}), "the refused job writes no change file");
    let second = json_object(&state, PAGES, "changes/seq/000000000002.json").unwrap();
    assert_eq!(second["rows"], 4, "t1's pages logged again, and one row for the withdrawn task");
    let written = rows(&state.object(PAGES, second["key"].as_str().unwrap()).unwrap());
    assert_eq!(column(&written, "kind"), ["page", "page", "page", "task_withdrawn"]);
    let withdrawn: Vec<String> = redis.zrange("publish:withdrawn", 0, -1).await.unwrap();
    assert_eq!(withdrawn, ["t2"], "every publisher refuses its jobs from now on");
    let last: u64 = redis.get("outcomes:seq").await.unwrap();
    let outcomes = outcome_feed(&state, 3, last);
    let of = |kind: &str| {
        let mut urls: Vec<String> = outcomes.iter().filter(|(k, _)| k == kind).map(|(_, url)| url.clone()).collect();
        urls.sort();
        urls
    };
    assert_eq!((of("published"), of("failed"), outcomes.len()), (urls("t1"), urls("t2"), 6), "the late job for t2 is refused");
    let _ = std::fs::remove_dir_all(&dir);
}

fn embed_input(texts: &[(&str, &str, &str, i32)]) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("text_id", DataType::Utf8, false),
        Field::new("page_key", DataType::Utf8, false),
        Field::new("url", DataType::Utf8, false),
        Field::new("content_sha1", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("index", DataType::Int32, false),
        Field::new("text", DataType::LargeUtf8, false),
    ]));
    let strings = |pick: fn(&(&str, &str, &str, i32)) -> String| -> ArrayRef { Arc::new(StringArray::from(texts.iter().map(pick).collect::<Vec<_>>())) };
    let columns = vec![
        strings(|t| t.0.to_string()),
        strings(|t| t.1.to_string()),
        strings(|t| format!("https://example.com/{}", t.1)),
        strings(|t| format!("sha-{}", t.1)),
        strings(|t| t.2.to_string()),
        Arc::new(Int32Array::from(texts.iter().map(|t| t.3).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(LargeStringArray::from(texts.iter().map(|t| format!("text of {}", t.0)).collect::<Vec<_>>())) as ArrayRef,
    ];
    write_parquet(RecordBatch::try_new(schema, columns).unwrap())
}

/// An embed task's input and its raw upload, the vectors in the input's order.
async fn stage(temp: &Bucket, task: &str, texts: &[(&str, &str, &str, i32)], vectors: &[Vec<u8>]) {
    temp.put(&format!("embed/inputs/{task}.parquet"), Bytes::from(embed_input(texts)), PARQUET, None).await.unwrap();
    temp.put(&format!("uploads/{task}.f16"), Bytes::from(vectors.concat()), "application/octet-stream", None).await.unwrap();
}

/// The publish job the task API queues for an embed task that passed its check.
fn embed(task: &str, texts: &[(&str, &str, &str, i32)]) -> Value {
    let pages: Vec<Value> = texts
        .iter()
        .map(|t| t.1)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|p| json!({"page_key": p, "content_sha1": format!("sha-{p}"), "url": format!("https://example.com/{p}")}))
        .collect();
    json!({"task_id": task, "round_id": "r2", "kind": "embed", "miner": "5Miner", "model": MODEL, "key": format!("uploads/{task}.f16"),
           "input_key": format!("embed/inputs/{task}.parquet"), "texts": texts.len(), "pages": pages, "validator": "5Validator",
           "validators": ["5Validator"], "completed_at": service::now_us() as f64 / 1e6})
}

/// Every file of the model's vector feed in order, with its key.
fn vector_files(state: &stub::State) -> Vec<(String, Vec<RecordBatch>)> {
    let latest = json_object(state, PAGES, &format!("vectors/{MODEL}/latest.json")).map_or(0, |latest| latest["seq"].as_u64().unwrap());
    (1..=latest)
        .map(|seq| {
            let index = json_object(state, PAGES, &format!("vectors/{MODEL}/seq/{seq:012}.json")).unwrap();
            let key = index["key"].as_str().unwrap().to_string();
            assert!(key.starts_with(&format!("vectors/{MODEL}/log/dt=")) && key.ends_with(".parquet"));
            let batches = rows(&state.object(PAGES, &key).unwrap());
            assert_eq!(index["rows"].as_u64().unwrap() as usize, batches.iter().map(|b| b.num_rows()).sum::<usize>());
            (key, batches)
        })
        .collect()
}

/// The rows a task's vector file publishes.
fn published(state: &stub::State, task: &str) -> Vec<publisher::vectors::VectorRow> {
    let files: Vec<String> = vector_files(state)
        .into_iter()
        .filter(|(_, batches)| column(batches, "kind").iter().all(|kind| kind != "task_withdrawn") && column(batches, "task_id").iter().all(|t| t == task))
        .map(|(key, _)| key)
        .collect();
    files.iter().flat_map(|key| publisher::vectors::decode(Bytes::from(state.object(PAGES, key).unwrap())).unwrap()).collect()
}

async fn entries(redis: &mut ConnectionManager, list: &str) -> Vec<Value> {
    let raw: Vec<String> = redis.lrange(list, 0, -1).await.unwrap();
    raw.iter().map(|entry| serde_json::from_str(entry).unwrap()).collect()
}

/// Each `embed:dropped` entry as its input and the keys of its pages.
fn sent_back(entries: &[Value]) -> Vec<(String, Vec<String>)> {
    let mut found: Vec<(String, Vec<String>)> = entries
        .iter()
        .map(|e| (e["input_key"].as_str().unwrap().into(), e["pages"].as_array().unwrap().iter().map(|p| p["page_key"].as_str().unwrap().into()).collect()))
        .collect();
    found.sort();
    found
}

fn write_parquet(batch: RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut out, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

/// A unit vector of float16 values pointing along `axis`.
fn unit(dims: usize, axis: usize) -> Vec<u8> {
    (0..dims).flat_map(|i| half::f16::from_f32(if i == axis { 1.0 } else { 0.0 }).to_le_bytes()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embed_jobs_publish_well_formed_vectors_and_withdraw_those_taken_back() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("vectors");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings =
        Settings { redis_url: REDIS.into(), claim_ttl: 600.0, batch: 4, readers: 4, ranges: 4, ahead: 2, idle_exit: 0, idle_delay: Duration::from_millis(50) };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics.clone()).await.unwrap();

    let texts = [("p2#page", "p2", "page", 0), ("p1#p1", "p1", "passage", 1), ("p1#page", "p1", "page", 0)];
    stage(&temp, "t7", &texts, &[unit(8, 1), unit(8, 2), unit(8, 0)]).await;
    stage(&temp, "t8", &texts, &[unit(8, 1), unit(8, 2)]).await;
    let mixed = [("p3#page", "p3", "page", 0), ("p4#page", "p4", "page", 0), ("p4#p1", "p4", "passage", 1), ("p5#page", "p5", "page", 0)];
    let mut not_a_number = unit(8, 0);
    not_a_number[2..4].copy_from_slice(&half::f16::NAN.to_le_bytes());
    let mut long = unit(8, 0);
    long[1] = 0x40;
    stage(&temp, "t9", &mixed, &[not_a_number, unit(8, 0), unit(8, 1), long]).await;
    stage(&temp, "t10", &texts[..1], &[unit(8, 3)]).await;
    let mut unchecked = embed("t10", &texts[..1]);
    unchecked["validator"] = json!("");
    unchecked["validators"] = json!([]);
    unchecked["reason"] = json!("unchecked");
    for job in [embed("t7", &texts), embed("t8", &texts), embed("t9", &mixed), unchecked.clone()] {
        enqueue(&mut redis, &job, true).await;
    }

    publish_until_idle(&shared).await;

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 4);
    let lost: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert_eq!(lost, HashSet::from(["t8".to_string()]), "an upload of the wrong size is not published");
    assert_eq!(vector_files(&state).len(), 3);
    let written = published(&state, "t7");
    let placed: Vec<(&str, &str, i32)> = written.iter().map(|r| (r.key.as_str(), r.kind.as_str(), r.index)).collect();
    assert_eq!(placed, [("p1", "page", 0), ("p1", "passage", 1), ("p2", "page", 0)], "sorted by page, each vector with its own text");
    assert_eq!([&written[0].vector, &written[1].vector, &written[2].vector], [&unit(8, 0), &unit(8, 2), &unit(8, 1)]);
    assert_eq!((written[1].url.as_str(), written[1].content_sha1.as_str()), ("https://example.com/p1", "sha-p1"));
    let kept: Vec<(String, i32)> = published(&state, "t9").into_iter().map(|r| (r.key, r.index)).collect();
    assert_eq!(kept, [("p4".to_string(), 0), ("p4".to_string(), 1)], "only the page whose vectors are all well formed");
    assert_eq!(published(&state, "t10").len(), 1, "an unchecked upload is published like a checked one");
    let dropped = entries(&mut redis, "embed:dropped").await;
    let expected = [
        ("embed/inputs/t8.parquet".to_string(), vec!["p1".to_string(), "p2".to_string()]),
        ("embed/inputs/t9.parquet".to_string(), vec!["p3".to_string(), "p5".to_string()]),
    ];
    assert_eq!(sent_back(&dropped), expected, "every page of the lost t8, and t9's pages with a malformed vector");
    let t9 = dropped.iter().find(|entry| entry["input_key"] == "embed/inputs/t9.parquet").unwrap();
    assert_eq!(t9["pages"][0], json!({"page_key": "p3", "content_sha1": "sha-p3", "url": "https://example.com/p3"}));
    assert_eq!(entries(&mut redis, "embed:malformed").await, [json!({"task_id": "t9", "miner": "5Miner", "malformed": 2})]);
    let remembered: Option<String> = redis.get("embed:published:t7").await.unwrap();
    assert_eq!(serde_json::from_str::<Value>(&remembered.unwrap()).unwrap()["model"], MODEL);
    assert!(redis.ttl::<_, i64>("embed:published:t7").await.unwrap() > 6 * 86_400, "remembered as long as take-backs are");
    assert!(
        state.keys(TEMP, &format!("{PREFIX}uploads/")).is_empty() && state.keys(TEMP, &format!("{PREFIX}embed/")).is_empty(),
        "uploads and inputs are deleted"
    );
    assert!(state.keys(PAGES, "changes/").is_empty(), "embed jobs write no change file");

    let () = redis.del("embed:dropped").await.unwrap();
    let at = service::now_us() as f64 / 1e6;
    let _: i64 = redis.zadd_multiple("publish:withdrawn", &[(at, "t7"), (at, "t10"), (at, "t11")]).await.unwrap();
    stage(&temp, "t11", &texts[..1], &[unit(8, 0)]).await;
    enqueue(&mut redis, &json!({"task_id": "withdraw:fedcba9876543210", "kind": "withdraw", "task_ids": ["t7", "t11"], "reason": "check_failed"}), false).await;
    enqueue(&mut redis, &embed("t11", &texts[..1]), true).await;
    enqueue(&mut redis, &unchecked, true).await;

    publish_until_idle(&shared).await;

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 7);
    let files = vector_files(&state);
    assert_eq!(files.len(), 5, "no vector file for t11, a withdrawal entry for t7 and one for t10");
    let mut withdrawn: Vec<(String, String)> = Vec::new();
    for (_, batches) in &files[3..] {
        assert_eq!((column(batches, "kind"), column(batches, "key")), (vec!["task_withdrawn".to_string()], vec![String::new()]));
        withdrawn.extend(column(batches, "task_id").into_iter().zip(column(batches, "miner")));
    }
    withdrawn.sort();
    assert_eq!(withdrawn, [("t10".to_string(), "5Miner".to_string()), ("t7".to_string(), "5Miner".to_string())]);
    assert!(published(&state, "t11").is_empty(), "taken back before publishing");
    let expected = [
        ("embed/inputs/t10.parquet".to_string(), vec!["p2".to_string()]),
        ("embed/inputs/t11.parquet".to_string(), vec!["p2".to_string()]),
        ("embed/inputs/t7.parquet".to_string(), vec!["p1".to_string(), "p2".to_string()]),
    ];
    assert_eq!(sent_back(&entries(&mut redis, "embed:dropped").await), expected, "every page of every task taken back, once");
    for (task, kept) in [("t7", false), ("t10", false), ("t9", true)] {
        assert_eq!(redis.exists::<_, bool>(format!("embed:published:{task}")).await.unwrap(), kept, "{task}");
    }
    assert!(state.keys(TEMP, &format!("{PREFIX}uploads/")).is_empty() && state.keys(TEMP, &format!("{PREFIX}embed/")).is_empty());
    let changes = json_object(&state, PAGES, "changes/seq/000000000001.json").unwrap();
    let changes = rows(&state.object(PAGES, changes["key"].as_str().unwrap()).unwrap());
    assert_eq!((column(&changes, "kind"), column(&changes, "task_id")), (vec!["task_withdrawn".to_string(); 2], vec!["t7".to_string(), "t11".to_string()]));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The entity tag the task API's completion names for an upload.
async fn etag(temp: &Bucket, key: &str) -> String {
    temp.head(key).await.unwrap().unwrap().etag
}

/// Writes over an upload as a miner could after completing it.
async fn rewrite(temp: &Bucket, key: &str, body: Vec<u8>, content_type: &str) {
    let before = etag(temp, key).await;
    // The stub tags an object by its length and modification time.
    tokio::time::sleep(Duration::from_millis(20)).await;
    temp.put(key, Bytes::from(body), content_type, None).await.unwrap();
    assert_ne!(etag(temp, key).await, before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_crawl_upload_is_published_only_in_the_version_its_completion_named() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("crawl-version");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings =
        Settings { redis_url: REDIS.into(), claim_ttl: 600.0, batch: 4, readers: 4, ranges: 4, ahead: 2, idle_exit: 0, idle_delay: Duration::from_millis(50) };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics).await.unwrap();

    let completed = service::now_us() as f64 / 1e6 - 60.0;
    let fetched = service::now_us() - 90_000_000;
    let urls = |task: &str| (0..3).map(|i| format!("https://{task}.com/story/{i}")).collect::<Vec<_>>();
    let upload = |task: &str, text: &str| parquet(&urls(task).iter().map(|u| page(u, &format!("{text} {u}"), fetched)).collect::<Vec<_>>(), None, None);
    let key = |task: &str| format!("uploads/{task}.parquet");
    for task in ["v1", "v2", "v3", "v4"] {
        temp.put(&key(task), Bytes::from(upload(task, "first")), PARQUET, None).await.unwrap();
    }
    enqueue(&mut redis, &crawl("v1", &urls("v1"), completed, &etag(&temp, &key("v1")).await), true).await;
    enqueue(&mut redis, &crawl("v2", &urls("v2"), completed, ""), true).await;
    enqueue(&mut redis, &crawl("v3", &urls("v3"), completed, &etag(&temp, &key("v3")).await), true).await;
    rewrite(&temp, &key("v3"), upload("v3", "rewritten"), PARQUET).await;

    publish_until_idle(&shared).await;

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 3);
    let lost: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert_eq!(lost, HashSet::from(["v3".to_string()]), "v1 named its version, v2 named none");
    let index = json_object(&state, PAGES, "changes/seq/000000000001.json").unwrap();
    let published: BTreeSet<String> = column(&rows(&state.object(PAGES, index["key"].as_str().unwrap()).unwrap()), "url").into_iter().collect();
    assert_eq!(published, urls("v1").into_iter().chain(urls("v2")).collect::<BTreeSet<_>>());
    let outcomes = outcome_feed(&state, 1, redis.get::<_, u64>("outcomes:seq").await.unwrap());
    let failed: BTreeSet<String> = outcomes.into_iter().filter(|(kind, _)| kind == "failed").map(|(_, url)| url).collect();
    assert_eq!(failed, urls("v3").into_iter().collect::<BTreeSet<_>>());

    enqueue(&mut redis, &crawl("v4", &urls("v4"), completed, &etag(&temp, &key("v4")).await), true).await;
    state.slow("GET", &format!("{TEMP}/{PREFIX}{}", key("v4")), 1, Duration::from_secs(1));
    let heads = state.count("HEAD");
    let publishing = tokio::spawn({
        let shared = shared.clone();
        async move { publish_until_idle(&shared).await }
    });
    for _ in 0..2000 {
        if state.count("HEAD") > heads {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    rewrite(&temp, &key("v4"), upload("v4", "first"), PARQUET).await;
    publishing.await.unwrap();

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 4);
    let lost: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert!(lost.contains("v4"), "rewritten between its HEAD and its read, even with the same bytes");
    assert_eq!(redis.get::<_, u64>("changes:seq").await.unwrap(), 1, "nothing of v4 is published");
    assert!(state.keys(TEMP, &format!("{PREFIX}uploads/")).is_empty(), "every upload is deleted once its job is done");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_embed_upload_changed_after_completion_is_lost_and_its_pages_go_back_to_the_planner() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("embed-version");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings =
        Settings { redis_url: REDIS.into(), claim_ttl: 600.0, batch: 4, readers: 4, ranges: 4, ahead: 2, idle_exit: 0, idle_delay: Duration::from_millis(50) };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics).await.unwrap();

    let texts = [("p2#page", "p2", "page", 0), ("p1#page", "p1", "page", 0)];
    for task in ["e1", "e2", "e3"] {
        stage(&temp, task, &texts, &[unit(8, 0), unit(8, 1)]).await;
    }
    let named = |task: &str, etag: String| {
        let mut job = embed(task, &texts);
        job["etag"] = json!(etag);
        job
    };
    enqueue(&mut redis, &named("e1", etag(&temp, "uploads/e1.f16").await), true).await;
    enqueue(&mut redis, &named("e2", etag(&temp, "uploads/e2.f16").await), true).await;
    enqueue(&mut redis, &embed("e3", &texts), true).await;
    rewrite(&temp, "uploads/e2.f16", [unit(8, 2), unit(8, 3)].concat(), "application/octet-stream").await;

    publish_until_idle(&shared).await;

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 3);
    let lost: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert_eq!(lost, HashSet::from(["e2".to_string()]));
    assert_eq!((published(&state, "e1").len(), published(&state, "e2").len(), published(&state, "e3").len()), (2, 0, 2));
    let back = sent_back(&entries(&mut redis, "embed:dropped").await);
    assert_eq!(back, [("embed/inputs/e2.parquet".to_string(), vec!["p1".to_string(), "p2".to_string()])]);
    assert!(state.keys(TEMP, &format!("{PREFIX}uploads/")).is_empty() && state.keys(TEMP, &format!("{PREFIX}embed/")).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn large_objects_go_up_in_parts_and_reads_survive_errors() {
    let dir = scratch("multipart");
    let stub = stub::start(dir.clone(), "127.0.0.1:0").await.unwrap();
    let errors = Arc::new(AtomicU64::new(0));
    let pages = bucket(stub.addr, PAGES, "pre/", errors.clone());
    let data: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();
    let (handle, uploaded) = (tokio::runtime::Handle::current(), data.clone());
    let target = pages.clone();
    tokio::task::spawn_blocking(move || {
        let mut upload = Multipart::new(target, "big.bin".into(), handle).with_part_bytes(64 << 10);
        upload.write_all(&uploaded).unwrap();
        upload.finish().unwrap();
    })
    .await
    .unwrap();
    assert_eq!(stub.state.object(PAGES, "pre/big.bin").unwrap(), data);
    assert!(stub.state.count("PUT") >= 7, "{} parts", stub.state.count("PUT"));
    let puts = stub.state.count("PUT");
    pages.put_in_parts("parts.bin", Bytes::from(data.clone()), PARQUET, 64 << 10, 3).await.unwrap();
    assert_eq!(stub.state.object(PAGES, "pre/parts.bin").unwrap(), data);
    assert_eq!(stub.state.count("PUT") - puts, 7, "seven parts of at most 64 KiB");

    stub.state.fail("GET", &format!("{PAGES}/pre/big.bin"), 3);
    let read = pages.get_range("big.bin", 4..12, None).await.unwrap();
    assert_eq!(read.as_ref(), &data[4..12]);
    assert_eq!(errors.load(Ordering::Relaxed), 3);
    assert!(matches!(pages.get_range("big.bin", 0..4, Some("\"stale\"")).await, Err(r2::Error::Changed)));
    assert!(pages.head("missing").await.unwrap().is_none());
    assert!(matches!(pages.get("missing").await, Err(r2::Error::Missing)));
    stub.state.fail("PUT", &format!("{PAGES}/pre/doomed"), 10);
    assert!(pages.put("doomed", Bytes::from_static(b"x"), PARQUET, None).await.is_err(), "gives up after its attempts");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pages_kept_with_sitemap_escapes_are_taken_down() {
    let Some((_turn, _redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("escaped");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings =
        Settings { redis_url: REDIS.into(), claim_ttl: 600.0, batch: 2, readers: 4, ranges: 4, ahead: 2, idle_exit: 0, idle_delay: Duration::from_millis(50) };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics).await.unwrap();
    let urls = ["https://ex.com/story", "https://ex.com/shop?amp%3Bp=2669&post_type=product", "https://ex.com/Q&amp;A"].map(String::from);
    let index = Arc::new(VersionIndex::open(&dir.join("index"), 8 << 20).unwrap());
    let entries: Vec<(String, Current)> = urls
        .iter()
        .enumerate()
        .map(|(row, url)| {
            let current = Current {
                url: url.clone(),
                version: format!("{row:040}"),
                fetched_at: "2026-10-08T10:00:00+00:00".into(),
                task_id: "t1".into(),
                content_sha1: "c".repeat(40),
                change_seq: Some(1),
                change_row: Some(row as i64),
            };
            (format!("pages/ex.com/{row:040x}"), current)
        })
        .collect();
    index.store(&entries).unwrap();

    assert_eq!(service::remove_escaped(&shared, &index, None).await.unwrap(), 2);
    assert_eq!(index.pages().unwrap().len(), 3, "counting changes nothing");
    let backup = dir.join("escaped.jsonl");
    assert_eq!(service::remove_escaped(&shared, &index, Some(&backup)).await.unwrap(), 2);
    let saved: Vec<Value> = std::fs::read_to_string(&backup).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(saved.len(), 2);
    assert!(saved.iter().all(|entry| entry["change_seq"].as_i64().is_some() && entry["version"].as_str().is_some_and(|v| !v.is_empty())));
    let kept: Vec<String> = index.pages().unwrap().into_iter().map(|(_, current)| current.url).collect();
    assert_eq!(kept, ["https://ex.com/story"]);
    let latest = json_object(&state, PAGES, "changes/latest.json").unwrap()["seq"].as_u64().unwrap();
    let numbered = json_object(&state, PAGES, &format!("changes/seq/{latest:012}.json")).unwrap();
    let removed = rows(&state.object(PAGES, numbered["key"].as_str().unwrap()).unwrap());
    assert_eq!(column(&removed, "kind"), ["removed"; 2]);
    let mut taken = column(&removed, "url");
    taken.sort();
    assert_eq!(taken, ["https://ex.com/Q&amp;A", "https://ex.com/shop?amp%3Bp=2669&post_type=product"]);
    assert_eq!(service::remove_escaped(&shared, &index, Some(&dir.join("again.jsonl"))).await.unwrap(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_pages_bucket_can_live_on_its_own_endpoint() {
    let env = |pairs: &[(&str, &str)]| -> std::collections::HashMap<String, String> { pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() };
    let r2 = [("CF_R2_ENDPOINT", "https://r2.example"), ("CF_R2_ACCESS_KEY_ID", "r2key"), ("CF_R2_SECRET_ACCESS_KEY", "r2secret")];
    let signed = |bucket: &Bucket| bucket.presign("GET", "k", 60, None, Duration::from_secs(1_700_000_000));

    let both = env(&r2);
    let (temp, pages) = service::buckets_from(|name| both.get(name).cloned(), Arc::default()).unwrap();
    assert_eq!((temp.bucket.as_str(), pages.bucket.as_str()), (TEMP, PAGES));
    assert!(signed(&pages).starts_with("https://r2.example/desearch-pages/k?") && signed(&pages).contains("r2key%2F20231114%2Fauto%2F"));

    let mut split = r2.to_vec();
    split.extend([
        ("PAGES_S3_ENDPOINT", "https://s3.example"),
        ("PAGES_S3_REGION", "us-east-va"),
        ("PAGES_S3_ACCESS_KEY_ID", "s3key"),
        ("PAGES_S3_SECRET_ACCESS_KEY", "s3secret"),
    ]);
    let split = env(&split);
    let (temp, pages) = service::buckets_from(|name| split.get(name).cloned(), Arc::default()).unwrap();
    assert!(signed(&temp).starts_with("https://r2.example/subnet-22/k?") && signed(&temp).contains("r2key%2F20231114%2Fauto%2F"));
    assert!(signed(&pages).starts_with("https://s3.example/desearch-pages/k?") && signed(&pages).contains("s3key%2F20231114%2Fus-east-va%2F"));

    let mut keyless = r2.to_vec();
    keyless.push(("PAGES_S3_ENDPOINT", "https://s3.example"));
    let keyless = env(&keyless);
    assert!(service::buckets_from(|name| keyless.get(name).cloned(), Arc::default()).is_err());

    let mut same = r2.to_vec();
    same.push(("CF_R2_PAGES_BUCKET", TEMP));
    let same = env(&same);
    assert!(service::buckets_from(|name| same.get(name).cloned(), Arc::default()).is_err());
}

//! The service against a local Redis (db 13) and the R2 stub: claims, acknowledgements, lost uploads, withdrawals, feeds and their holes.

mod common;

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::RecordBatch;
use base64::Engine;
use bytes::Bytes;
use common::{page, parquet};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use publisher::r2::{self, Bucket, Credentials, PARQUET};
use publisher::service::{self, Metrics, Settings, Shared};
use publisher::snapshot::Multipart;
use publisher::stub;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde_json::{json, Value};
use tokio::sync::watch;

const REDIS: &str = "redis://127.0.0.1:6379/13";
const TEMP: &str = "subnet-22";
const PAGES: &str = "desearch-pages";
const PREFIX: &str = "tp/";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn redis_db() -> Option<ConnectionManager> {
    let client = redis::Client::open(REDIS).ok()?;
    let mut redis = tokio::time::timeout(Duration::from_secs(2), client.get_connection_manager()).await.ok()?.ok()?;
    let () = redis::cmd("FLUSHDB").query_async(&mut redis).await.ok()?;
    Some(redis)
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
        "key": format!("submitted/{task_id}.parquet"), "etag": etag, "urls": urls, "skip": [],
        "completed_at": completed, "claim_ttl": 180,
    })
}

fn rows(file: &[u8]) -> Vec<RecordBatch> {
    ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(file)).unwrap().build().unwrap().map(Result::unwrap).collect()
}

fn column(batches: &[RecordBatch], name: &str) -> Vec<String> {
    batches.iter().flat_map(|b| b.column_by_name(name).unwrap().as_string::<i32>().iter().map(|v| v.unwrap_or_default().to_string()).collect::<Vec<_>>()).collect()
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
async fn the_service_publishes_reports_and_takes_back_as_the_python_one_does() {
    let Some(mut redis) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("service");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let state = stub.state.clone();
    let metrics = Arc::new(Metrics::default());
    let temp = bucket(stub.addr, TEMP, PREFIX, metrics.r2_errors.clone());
    let pages = bucket(stub.addr, PAGES, "", metrics.r2_errors.clone());
    let settings = Settings {
        redis_url: REDIS.into(),
        claim_ttl: 600.0,
        batch: 2,
        readers: 4,
        ranges: 4,
        index: dir.join("index"),
        cache_bytes: 8 << 20,
        idle_exit: 0,
        idle_delay: Duration::from_millis(50),
    };
    let shared = Shared::connect(settings, temp.clone(), pages.clone(), metrics.clone()).await.unwrap();

    let completed = service::now_us() as f64 / 1e6 - 60.0;
    let fetched = service::now_us() - 90_000_000;
    let urls = |task: &str| (0..3).map(|i| format!("https://ex{task}.com/story/{i}")).collect::<Vec<_>>();
    let upload = |task: &str, text: &str| parquet(&urls(task).iter().map(|u| page(u, &format!("{text} {u}"), fetched)).collect::<Vec<_>>(), None, None);
    for task in ["t1", "t2", "t4"] {
        temp.put(&format!("submitted/{task}.parquet"), Bytes::from(upload(task, "first")), PARQUET, None).await.unwrap();
    }
    let t1_etag = temp.head("submitted/t1.parquet").await.unwrap().unwrap().etag;
    enqueue(&mut redis, &crawl("t1", &urls("t1"), completed, &t1_etag), true).await;
    enqueue(&mut redis, &crawl("t2", &urls("t2"), completed, ""), false).await;
    enqueue(&mut redis, &crawl("t3", &urls("t3"), completed, ""), true).await;
    enqueue(&mut redis, &crawl("t4", &urls("t4"), completed, "\"not-this-one\""), true).await;
    enqueue(&mut redis, &json!({"task_id": "t5", "kind": "embed", "key": "embed/t5.parquet", "input_key": "embed-inputs/t5.parquet", "completed_at": completed}), true).await;
    state.fail("GET", &format!("{TEMP}/{PREFIX}submitted/t1"), 2);
    state.fail("PUT", &format!("{PAGES}/changes/seq/"), 5);

    publish_until_idle(&shared).await;

    let acked: u64 = redis.get("publish:acked").await.unwrap();
    assert_eq!(acked, 4, "t1, t2 and the lost t3 and t4");
    let claimed: Vec<String> = redis.zrange("publish:claims", 0, -1).await.unwrap();
    assert_eq!(claimed, ["t5"], "an embed job stays claimed for the API to give back");
    let pending: Vec<String> = redis.zrange("publish:pending", 0, -1).await.unwrap();
    assert_eq!(pending, ["t5"]);
    assert!(!redis.exists::<_, bool>("pjob:t1").await.unwrap() && redis.exists::<_, bool>("pjob:t5").await.unwrap());
    let lost: u64 = redis.get("publish:lost").await.unwrap();
    let lost_tasks: HashSet<String> = redis.smembers("publish:lost:tasks").await.unwrap();
    assert_eq!((lost, lost_tasks), (2, HashSet::from(["t3".to_string(), "t4".to_string()])), "t3 expired, t4 changed");
    assert!(state.keys(TEMP, &format!("{PREFIX}submitted/")).is_empty(), "finished jobs' uploads are deleted, the changed one too");
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
    assert_eq!(column(&changes, "kind"), ["new"; 6]);
    assert_eq!(state.meta(PAGES, &change_key).unwrap().0, PARQUET);

    assert_eq!(redis.get::<_, u64>("outcomes:seq").await.unwrap(), 2, "the two batches that finished tasks");
    let outcomes = outcome_feed(&state, 1, 2);
    let count = |kind: &str| outcomes.iter().filter(|(k, _)| k == kind).count();
    assert_eq!((count("published"), count("failed"), outcomes.len()), (6, 6, 12));
    assert_eq!(json_object(&state, TEMP, &format!("{PREFIX}outcomes/latest.json")).unwrap(), json!({"seq": 2}));
    assert_eq!(state.meta(TEMP, &format!("{PREFIX}outcomes/latest.json")).unwrap().1, "no-store");

    let snapshot_key = format!("index/snapshots/{}.parquet", service::day(service::now_us()));
    for _ in 0..100 {
        if state.object(PAGES, &snapshot_key).is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let snapshot = Bytes::from(state.object(PAGES, &snapshot_key).expect("the day's index snapshot"));
    assert_eq!(ParquetRecordBatchReaderBuilder::try_new(snapshot).unwrap().schema().fields().len(), 8);

    temp.put("submitted/t6.parquet", Bytes::from(upload("t1", "first")), PARQUET, None).await.unwrap();
    enqueue(&mut redis, &crawl("t6", &urls("t1"), completed + 30.0, ""), true).await;
    let withdraw = json!({"task_id": "withdraw:0123456789abcdef", "kind": "withdraw", "task_ids": ["t2"], "reason": "test"});
    enqueue(&mut redis, &withdraw, false).await;
    let (stopping, stop) = watch::channel(false);
    let running = tokio::spawn(service::run(shared.clone(), stop, 0));
    for _ in 0..400 {
        if redis.get::<_, u64>("publish:acked").await.unwrap() == 6 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    stopping.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(10), running).await.expect("stopping on request").unwrap().unwrap();

    assert_eq!(redis.get::<_, u64>("publish:acked").await.unwrap(), 6);
    assert_eq!(json_object(&state, PAGES, "changes/latest.json").unwrap(), json!({"seq": 2}));
    let second = json_object(&state, PAGES, "changes/seq/000000000002.json").unwrap();
    assert_eq!(second["rows"], 3);
    let removed = rows(&state.object(PAGES, second["key"].as_str().unwrap()).unwrap());
    assert_eq!(column(&removed, "kind"), ["removed"; 3]);
    let mut taken = column(&removed, "url");
    taken.sort();
    assert_eq!(taken, urls("t2"));
    let kinds: Vec<String> = outcome_feed(&state, 3, 3).into_iter().map(|(kind, _)| kind).collect();
    assert_eq!(kinds, ["unchanged", "unchanged", "unchanged", "dropped", "dropped", "dropped"]);
    assert!(shared.index.is_withdrawn("t2").unwrap());
    assert_eq!(shared.index.pages().unwrap().len(), 3, "t1's pages stay, t2's are taken back");
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

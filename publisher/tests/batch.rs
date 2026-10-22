//! Whole batches: what is published, what fails, and what a withdrawal takes back.

mod common;

use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use common::{page, page_v2, parquet, parquet_v2};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use publisher::changes::{Body, Kind};
use publisher::current::{decode_changes, ChangeRow};
use publisher::index::VersionIndex;
use publisher::local::{LocalFeed, LocalUploads};
use publisher::outcomes;
use publisher::records::{record_version, Record, SECOND};
use publisher::worker::{publish, Job};
use serde_json::json;

const COMPLETED: i64 = 1_791_310_000;
const NOW: i64 = 1_791_311_000 * SECOND;
const A: &str = "https://ex.com/a";

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn withdraw(task_ids: &[&str]) -> Job {
    serde_json::from_value(json!({"task_id": "withdraw:w1", "kind": "withdraw", "task_ids": task_ids, "reason": "test"})).unwrap()
}

fn job(task_id: &str, urls: &[&str], skip: &[&str]) -> Job {
    serde_json::from_value(json!({
        "task_id": task_id, "kind": "crawl", "miner": "5Miner", "key": format!("uploads/{task_id}.parquet"),
        "urls": urls, "skip": skip, "completed_at": COMPLETED, "claim_ttl": 180, "validator": "", "validators": [],
    }))
    .unwrap()
}

#[test]
fn a_batch_publishes_the_best_copy_of_each_page_and_publishing_again_logs_it_again() {
    let dir = scratch("batch");
    let at = |seconds: i64| (COMPLETED - seconds) * SECOND;
    let mut failed_row = page("https://ex.com/b", "text", at(60));
    failed_row.row.error = Some("timeout".into());
    let mut challenge = page("https://ex.com/c", "", at(60));
    challenge.row.text = Some("Checking your browser before accessing".into());
    challenge.row.title = Some("Just a moment...".into());
    let first = [
        page(A, "first copy", at(120)),
        failed_row,
        challenge,
        page("https://ex.com/d", "never assigned", at(60)),
        page("https://ex.com/e", "skipped", at(60)),
    ];
    std::fs::write(dir.join("t1.parquet"), parquet(&first, None, None)).unwrap();
    std::fs::write(dir.join("t2.parquet"), parquet(&[page(A, "second copy", at(30))], None, None)).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();
    let none = HashSet::new();

    let t1 = job("t1", &[A, "https://ex.com/b", "https://ex.com/c", "https://ex.com/e", "https://ex.com/x"], &["https://ex.com/e"]);
    let t2 = job("t2", &[A], &[]);
    let t3 = job("t3", &["https://ex.com/gone"], &[]);
    let batch = publish(&[t1.clone(), t2.clone(), t3], &uploads, &none, &feed, NOW, 2).unwrap();
    assert_eq!(batch.changes.len(), 1, "one row per page, the later fetch of the two copies");
    let change = &batch.changes[0];
    assert_eq!((change.kind, change.record().unwrap().task_id.as_str(), change.record().unwrap().text.as_str()), (Kind::Page, "t2", "second copy"));
    assert_eq!(change.previous_content_sha1, "");
    let failed: Vec<(&str, &str)> = batch.failed.iter().map(|f| (f.url.as_str(), f.task_id.as_str())).collect();
    assert_eq!(
        failed,
        [("https://ex.com/b", "t1"), ("https://ex.com/c", "t1"), ("https://ex.com/e", "t1"), ("https://ex.com/x", "t1"), ("https://ex.com/gone", "t3")]
    );
    assert_eq!((batch.lost.as_slice(), batch.finalized.len(), batch.change_seq), (&["t3".to_string()][..], 3, Some(1)));

    let batch = publish(&[t1], &uploads, &none, &feed, NOW + 30 * SECOND, 2).unwrap();
    assert_eq!(batch.changes.len(), 1, "no version check: an older copy is logged too, and the compactor keeps the later fetch");
    let rows = outcomes::rows(&batch.changes, &batch.failed);
    assert_eq!(rows.iter().filter(|r| r.outcome == desearch::outcomes::PUBLISHED).count(), 1);
    assert!(!desearch::outcomes::encode(&rows, NOW).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_v2_upload_publishes_its_v2_fields_beside_v1_uploads_and_the_later_fetch_wins() {
    let dir = scratch("v2");
    let at = |seconds: i64| (COMPLETED - seconds) * SECOND;
    let (b, c) = ("https://ex.com/b", "https://ex.com/c");
    std::fs::write(dir.join("old.parquet"), parquet(&[page(A, "v1 copy", at(120)), page(b, "only v1", at(60)), page(c, "newer v1", at(10))], None, None))
        .unwrap();
    std::fs::write(dir.join("new.parquet"), parquet_v2(&[page_v2(A, "# v2 copy", at(30)), page_v2(c, "# older v2", at(90))], None)).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();
    let batch = publish(&[job("old", &[A, b, c], &[]), job("new", &[A, c], &[])], &uploads, &HashSet::new(), &feed, NOW, 2).unwrap();
    let by_url: HashMap<&str, &Record> = batch.changes.iter().map(|change| (change.url().unwrap(), change.record().unwrap())).collect();
    let v2 = by_url[A];
    assert_eq!((v2.text.as_str(), v2.extraction.as_deref(), v2.text_mode.as_deref()), ("# v2 copy", Some("v2"), Some("main")));
    assert_eq!((v2.tags.len(), v2.image.as_deref(), v2.ld.as_deref()), (2, Some("https://ex.com/a/cover.png"), Some("{\"faq\": [[\"Q\", \"A\"]]}")));
    assert!(by_url[b].is_v1() && by_url[b].tags.is_empty());
    assert_eq!((by_url[c].text.as_str(), by_url[c].is_v1()), ("newer v1", true), "a later v1 fetch wins over an earlier v2 one");

    let file = std::fs::read(dir.join("changes").join(format!("{:012}.parquet", 1))).unwrap();
    let logged = decode_changes(bytes::Bytes::from(file), 1).unwrap();
    for (row, change) in logged.iter().zip(&batch.changes) {
        let (ChangeRow::Page(row), Body::Page { record, version }) = (row, &change.body) else { panic!("pages only") };
        assert_eq!((&row.record, &row.version), (record.as_ref(), version), "{} reads back as published", record.url);
        assert_eq!(*version, record_version(record));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_withdrawal_writes_one_row_per_task_whatever_its_pages_and_refuses_its_jobs_then_and_after() {
    let dir = scratch("withdraw");
    let at = (COMPLETED - 60) * SECOND;
    let urls: Vec<String> = (0..300).map(|i| format!("https://ex.com/story/{i}")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    std::fs::write(dir.join("big.parquet"), parquet(&urls.iter().map(|u| page(u, &format!("text of {u}"), at)).collect::<Vec<_>>(), None, None)).unwrap();
    std::fs::write(dir.join("small.parquet"), parquet(&[page(A, "a", at)], None, None)).unwrap();
    std::fs::copy(dir.join("small.parquet"), dir.join("late.parquet")).unwrap();
    std::fs::copy(dir.join("small.parquet"), dir.join("later.parquet")).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();
    let batch = publish(&[job("big", &refs, &[]), job("small", &[A], &[])], &uploads, &HashSet::new(), &feed, NOW, 2).unwrap();
    assert_eq!(batch.changes.len(), 301);

    let batch = publish(&[withdraw(&["big", "small", "late", "big"]), job("late", &[A], &[])], &uploads, &HashSet::new(), &feed, NOW + SECOND, 2).unwrap();
    let rows: Vec<(Kind, Option<&str>)> = batch.changes.iter().map(|c| (c.kind, c.url())).collect();
    assert_eq!(rows, [(Kind::TaskWithdrawn, None); 3], "one row per task named, none per page");
    assert_eq!(batch.withdrawn, ["big", "small", "late"]);
    let failed: Vec<(&str, &str)> = batch.failed.iter().map(|f| (f.url.as_str(), f.task_id.as_str())).collect();
    assert_eq!(failed, [(A, "late")], "a job for a task taken back in the same batch is refused");
    assert!(batch.finalized.contains(&"late".to_string()));

    let refused: HashSet<String> = ["later".to_string()].into();
    let batch = publish(&[job("later", &[A], &[])], &uploads, &refused, &feed, NOW + 2 * SECOND, 2).unwrap();
    assert!(batch.changes.is_empty(), "a job for a task taken back earlier is refused");
    assert_eq!(batch.failed.len(), 1);

    let file = std::fs::read(dir.join("changes").join(format!("{:012}.parquet", 2))).unwrap();
    let read = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(file)).unwrap().build().unwrap().next().unwrap().unwrap();
    for name in ["key", "previous_content_sha1", "url", "domain", "text", "assigned_url", "miner"] {
        assert_eq!(read.column_by_name(name).unwrap().null_count(), 3, "{name} is null");
    }
    let kinds: Vec<&str> = read.column_by_name("kind").unwrap().as_string::<i32>().iter().flatten().collect();
    let ids: Vec<&str> = read.column_by_name("task_id").unwrap().as_string::<i32>().iter().flatten().collect();
    assert_eq!((kinds, ids), (vec!["task_withdrawn"; 3], vec!["big", "small", "late"]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_index_kept_from_before_still_reads_and_takes_pages_down() {
    let dir = scratch("index");
    let index = VersionIndex::open(&dir, 8 << 20).unwrap();
    let current = |task: &str, version: &str, fetched_at: &str| publisher::index::Current {
        url: "https://ex.com/p".into(),
        version: version.into(),
        fetched_at: fetched_at.into(),
        task_id: task.into(),
        content_sha1: "c".repeat(40),
        change_seq: None,
        change_row: None,
    };
    let key = "pages/ex.com/1".to_string();
    index.store(&[(key.clone(), current("t1", &"1".repeat(40), "2026-10-01T00:00:00+00:00"))]).unwrap();
    index.store(&[(key.clone(), current("t2", &"2".repeat(40), "2026-10-02T00:00:00+00:00"))]).unwrap();
    assert_eq!(index.current(&key).unwrap().unwrap().task_id, "t2");
    index.remove(&[(key.clone(), "1".repeat(40))]).unwrap();
    assert!(index.current(&key).unwrap().is_some(), "a version that is no longer current stays");
    index.remove(&[(key.clone(), "2".repeat(40))]).unwrap();
    assert_eq!(index.current(&key).unwrap(), None);
    let _ = std::fs::remove_dir_all(&dir);
}

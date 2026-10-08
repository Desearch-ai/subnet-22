//! Whole batches against a real index: what is published, what fails, and what a withdrawal takes back.

mod common;

use std::collections::HashSet;

use arrow_array::cast::AsArray;
use common::{page, parquet};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use publisher::changes::{Body, Kind};
use publisher::index::{VersionIndex, Withdrawn};
use publisher::local::{LocalFeed, LocalUploads};
use publisher::outcomes;
use publisher::records::SECOND;
use publisher::withdrawals;
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
        "task_id": task_id, "kind": "crawl", "miner": "5Miner", "key": format!("submitted/{task_id}.parquet"),
        "urls": urls, "skip": skip, "completed_at": COMPLETED, "claim_ttl": 180, "validator": "", "validators": [],
    }))
    .unwrap()
}

#[test]
fn a_batch_publishes_the_best_copy_and_a_withdrawn_one_is_published_again_from_another_task() {
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
    let index = VersionIndex::open(&dir.join("index"), 8 << 20).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();

    let t1 = job("t1", &[A, "https://ex.com/b", "https://ex.com/c", "https://ex.com/e", "https://ex.com/x"], &["https://ex.com/e"]);
    let t2 = job("t2", &[A], &[]);
    let t3 = job("t3", &["https://ex.com/gone"], &[]);
    let batch = publish(&[t1.clone(), t2.clone(), t3], &uploads, &index, &feed, NOW, 2).unwrap();
    assert_eq!(batch.changes.len(), 1);
    let change = &batch.changes[0];
    assert_eq!((change.kind, change.record().unwrap().task_id.as_str(), change.record().unwrap().text.as_str()), (Kind::New, "t2", "second copy"));
    let failed: Vec<(&str, &str)> = batch.failed.iter().map(|f| (f.url.as_str(), f.task_id.as_str())).collect();
    assert_eq!(
        failed,
        [("https://ex.com/b", "t1"), ("https://ex.com/c", "t1"), ("https://ex.com/e", "t1"), ("https://ex.com/x", "t1"), ("https://ex.com/gone", "t3")]
    );
    assert_eq!((batch.lost.as_slice(), batch.finalized.len(), batch.change_seq), (&["t3".to_string()][..], 3, Some(1)));
    let current = index.current(&change.key).unwrap().unwrap();
    assert_eq!((current.task_id.as_str(), current.change_seq, current.change_row), ("t2", Some(1), Some(0)));

    let batch = publish(&[t1.clone(), t2.clone()], &uploads, &index, &feed, NOW + 30 * SECOND, 2).unwrap();
    assert_eq!((batch.changes.len(), batch.unchanged.len()), (0, 1), "the older first copy does not replace the newer one");

    let batch = publish(&[withdraw(&["t2"]), t1, t2], &uploads, &index, &feed, NOW + 60 * SECOND, 2).unwrap();
    assert_eq!(batch.withdrawn, ["t2"]);
    let kinds: Vec<Kind> = batch.changes.iter().map(|c| c.kind).collect();
    assert_eq!(kinds, [Kind::New, Kind::TaskWithdrawn], "the withdrawn version counts as none, so the first copy is published");
    let republished = batch.changes[0].record().unwrap();
    assert_eq!((republished.task_id.as_str(), republished.text.as_str(), batch.changes[0].previous_content_sha1.as_str()), ("t1", "first copy", ""));
    assert!(matches!(&batch.changes[1].body, Body::TaskWithdrawn { task_id } if task_id == "t2"));
    assert!(batch.failed.iter().filter(|f| f.task_id == "t2").count() == 1, "a withdrawn task's URLs fail");
    assert_eq!(index.current(&change.key).unwrap().unwrap().task_id, "t1");
    let rows = outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed);
    assert_eq!(rows.iter().filter(|r| r.outcome == desearch::outcomes::PUBLISHED).count(), 1, "a task_withdrawn row is no page");
    assert!(!desearch::outcomes::encode(&rows, NOW).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_withdrawal_writes_one_row_per_task_whatever_its_pages_and_refuses_late_jobs() {
    let dir = scratch("withdraw");
    let at = (COMPLETED - 60) * SECOND;
    let urls: Vec<String> = (0..300).map(|i| format!("https://ex.com/story/{i}")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    std::fs::write(dir.join("big.parquet"), parquet(&urls.iter().map(|u| page(u, &format!("text of {u}"), at)).collect::<Vec<_>>(), None, None)).unwrap();
    std::fs::write(dir.join("small.parquet"), parquet(&[page(A, "a", at)], None, None)).unwrap();
    std::fs::copy(dir.join("small.parquet"), dir.join("late.parquet")).unwrap();
    let index = VersionIndex::open(&dir.join("index"), 8 << 20).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();
    let batch = publish(&[job("big", &refs, &[]), job("small", &[A], &[])], &uploads, &index, &feed, NOW, 2).unwrap();
    assert_eq!(batch.changes.len(), 301);

    let batch = publish(&[withdraw(&["big", "small", "late", "big"]), job("late", &[A], &[])], &uploads, &index, &feed, NOW + SECOND, 2).unwrap();
    let rows: Vec<(Kind, Option<&str>)> = batch.changes.iter().map(|c| (c.kind, c.url())).collect();
    assert_eq!(rows, [(Kind::TaskWithdrawn, None); 3], "one row per task named, none per page");
    let tasks: Vec<&str> = batch
        .changes
        .iter()
        .filter_map(|c| match &c.body {
            Body::TaskWithdrawn { task_id } => Some(task_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tasks, ["big", "small", "late"]);
    assert!(batch.unchanged.is_empty());
    let failed: Vec<(&str, &str)> = batch.failed.iter().map(|f| (f.url.as_str(), f.task_id.as_str())).collect();
    assert_eq!(failed, [(A, "late")], "a late job for a withdrawn task is refused");
    assert!(batch.finalized.contains(&"late".to_string()));
    assert_eq!(index.pages().unwrap().len(), 301, "pages leave the index later, off the publish path");

    let file = std::fs::read(dir.join("changes").join(format!("{:012}.parquet", batch.change_seq.unwrap()))).unwrap();
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
fn withdrawn_pages_leave_the_index_in_chunks_reported_as_dropped() {
    let dir = scratch("drain");
    let at = (COMPLETED - 60) * SECOND;
    let urls: Vec<String> = (0..5).map(|i| format!("https://ex.com/story/{i}")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    std::fs::write(dir.join("t1.parquet"), parquet(&urls.iter().map(|u| page(u, &format!("text of {u}"), at)).collect::<Vec<_>>(), None, None)).unwrap();
    std::fs::write(dir.join("t2.parquet"), parquet(&[page(&urls[0], "a newer copy", at + SECOND)], None, None)).unwrap();
    let index = VersionIndex::open(&dir.join("index"), 8 << 20).unwrap();
    let uploads = LocalUploads::new(&dir);
    let feed = LocalFeed::new(dir.join("changes"), 1).unwrap();
    publish(&[job("t1", &refs, &[])], &uploads, &index, &feed, NOW, 2).unwrap();
    publish(&[withdraw(&["t1"]), job("t2", &refs[..1], &[])], &uploads, &index, &feed, NOW + SECOND, 2).unwrap();

    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut report = |pages: &[Withdrawn]| -> anyhow::Result<()> {
        chunks.push(outcomes::dropped(pages).into_iter().map(|o| o.url).collect());
        Ok(())
    };
    let mut drained = HashSet::new();
    let now = (NOW + 2 * SECOND) as f64 / SECOND as f64;
    assert_eq!(withdrawals::drain(&index, &mut drained, now, 2, &|| false, &mut report).unwrap(), 4);
    let mut expected: Vec<String> = urls[1..].to_vec();
    expected.sort();
    assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2], "at most a chunk at a time");
    let mut reported = chunks.concat();
    reported.sort();
    assert_eq!(reported, expected, "the page t2 replaced stays");
    let left: Vec<String> = index.pages().unwrap().into_iter().map(|(_, current)| current.task_id).collect();
    assert_eq!(left, ["t2"]);
    assert!(drained.contains("t1"));

    let mut again = 0;
    let mut count = |_: &[Withdrawn]| -> anyhow::Result<()> {
        again += 1;
        Ok(())
    };
    assert_eq!(withdrawals::drain(&index, &mut HashSet::new(), now, 2, &|| false, &mut count).unwrap(), 0, "a restart finds nothing left");
    assert_eq!(again, 0);
    let mut failing = |_: &[Withdrawn]| -> anyhow::Result<()> { anyhow::bail!("storage is down") };
    index.withdraw(&["t2".into()], now).unwrap();
    assert!(withdrawals::drain(&index, &mut drained, now, 2, &|| false, &mut failing).is_err());
    assert_eq!(index.pages().unwrap().len(), 1, "pages stay until the bot is told");
    assert_eq!(withdrawals::drain(&index, &mut drained, now + 8.0 * 86_400.0, 2, &|| false, &mut |_| Ok(())).unwrap(), 1);
    assert!(index.withdrawn().unwrap().is_empty(), "drained tasks are forgotten after seven days");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_index_keeps_each_task_s_pages_current() {
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
    let taken = |task: &str| {
        let mut found = Vec::new();
        let looked = index.drain(task, None, 10, |pages| {
            found.extend(pages.iter().map(|p| p.key.clone()));
            anyhow::bail!("only looking")
        });
        assert_eq!(looked.is_err(), !found.is_empty(), "a failed report keeps the pages");
        found
    };
    assert!(taken("t1").is_empty(), "a page belongs to the task of its current version only");
    assert_eq!(taken("t2"), [key.as_str()]);
    index.touch(&[(key.clone(), "2026-10-01T12:00:00+00:00".into())]).unwrap();
    assert_eq!(index.current(&key).unwrap().unwrap().fetched_at, "2026-10-02T00:00:00+00:00", "touch only moves forward");
    index.touch(&[(key.clone(), "2026-10-03T00:00:00+00:00".into()), ("pages/ex.com/none".into(), "2026-10-03T00:00:00+00:00".into())]).unwrap();
    assert_eq!(index.current(&key).unwrap().unwrap().fetched_at, "2026-10-03T00:00:00+00:00");
    assert_eq!(index.current("pages/ex.com/none").unwrap(), None);
    index.remove(&[(key.clone(), "1".repeat(40))]).unwrap();
    assert!(index.current(&key).unwrap().is_some(), "a withdrawn version that is no longer current stays");
    index.remove(&[(key.clone(), "2".repeat(40))]).unwrap();
    assert_eq!(index.current(&key).unwrap(), None);
    assert!(taken("t2").is_empty());
    index.withdraw(&["old".into()], 1_000_000.0).unwrap();
    index.withdraw(&["new".into(), "old".into()], 1_000_000.0 + 8.0 * 86_400.0).unwrap();
    assert_eq!(index.withdrawn().unwrap(), [("new".to_string(), 1_000_000.0 + 8.0 * 86_400.0), ("old".to_string(), 1_000_000.0)]);
    assert_eq!(index.withdrawn_among(&["old", "none", "new", "old"]).unwrap(), HashSet::from(["old".to_string(), "new".to_string()]));
    let _ = std::fs::remove_dir_all(&dir);
}

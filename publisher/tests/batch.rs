//! Whole batches against a real index: what is published, what fails, and what a withdrawal takes back.

mod common;

use common::{page, parquet};
use publisher::changes::{Body, Kind};
use publisher::index::VersionIndex;
use publisher::local::{LocalFeed, LocalUploads};
use publisher::outcomes;
use publisher::records::SECOND;
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

fn job(task_id: &str, urls: &[&str], skip: &[&str]) -> Job {
    serde_json::from_value(json!({
        "task_id": task_id, "kind": "crawl", "miner": "5Miner", "key": format!("submitted/{task_id}.parquet"),
        "urls": urls, "skip": skip, "completed_at": COMPLETED, "claim_ttl": 180, "validator": "", "validators": [],
    }))
    .unwrap()
}

#[test]
fn a_batch_publishes_the_best_copy_and_a_withdrawal_takes_it_back() {
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

    let withdraw: Job = serde_json::from_value(json!({"task_id": "w1", "kind": "withdraw", "task_ids": ["t2"]})).unwrap();
    let batch = publish(&[withdraw, t1, t2], &uploads, &index, &feed, NOW + 60 * SECOND, 2).unwrap();
    assert_eq!(batch.unchanged.len(), 1, "the older first copy does not replace the newer one");
    assert_eq!(batch.changes.len(), 1);
    assert!(matches!(&batch.changes[0].body, Body::Removed { url, .. } if url == A));
    assert_eq!(batch.changes[0].previous_content_sha1, current.content_sha1);
    assert!(batch.failed.iter().filter(|f| f.task_id == "t2").count() == 1, "a withdrawn task's URLs fail");
    assert_eq!(index.current(&change.key).unwrap(), None);
    assert!(index.is_withdrawn("t2").unwrap());
    let rows = outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed, &batch.removed);
    let dropped: Vec<_> = rows.iter().filter(|r| r.outcome == outcomes::DROPPED).map(|r| (r.url.as_str(), r.host.as_str())).collect();
    assert_eq!(dropped, [(A, "ex.com")]);
    assert!(!outcomes::encode(&rows, NOW).unwrap().is_empty());
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
    assert!(index.of_tasks(&["t1".into()]).unwrap().is_empty(), "a page belongs to the task of its current version only");
    assert_eq!(index.of_tasks(&["t2".into()]).unwrap().len(), 1);
    index.touch(&[(key.clone(), "2026-10-01T12:00:00+00:00".into())]).unwrap();
    assert_eq!(index.current(&key).unwrap().unwrap().fetched_at, "2026-10-02T00:00:00+00:00", "touch only moves forward");
    index.touch(&[(key.clone(), "2026-10-03T00:00:00+00:00".into()), ("pages/ex.com/none".into(), "2026-10-03T00:00:00+00:00".into())]).unwrap();
    assert_eq!(index.current(&key).unwrap().unwrap().fetched_at, "2026-10-03T00:00:00+00:00");
    assert_eq!(index.current("pages/ex.com/none").unwrap(), None);
    index.remove(&[(key.clone(), "1".repeat(40))]).unwrap();
    assert!(index.current(&key).unwrap().is_some(), "a withdrawn version that is no longer current stays");
    index.remove(&[(key.clone(), "2".repeat(40))]).unwrap();
    assert_eq!(index.current(&key).unwrap(), None);
    assert!(index.of_tasks(&["t2".into()]).unwrap().is_empty());
    index.withdraw(&["old".into()], 1_000_000.0).unwrap();
    index.withdraw(&["new".into(), "old".into()], 1_000_000.0 + 8.0 * 86_400.0).unwrap();
    assert_eq!(index.withdrawn().unwrap().into_iter().map(|(t, _)| t).collect::<Vec<_>>(), ["new"], "kept seven days");
    let _ = std::fs::remove_dir_all(&dir);
}

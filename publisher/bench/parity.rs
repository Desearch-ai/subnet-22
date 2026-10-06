//! Replays the reference harness's batches with the Rust publisher and dumps what it decided: `parity WORK_DIR`.

use std::path::PathBuf;

use publisher::changes::{Body, Change};
use publisher::index::{Current, VersionIndex};
use publisher::local::{LocalFeed, LocalUploads};
use publisher::outcomes;
use publisher::records::{Record, SECOND, SOURCE};
use publisher::worker::{self, Job, Page};
use serde::Deserialize;
use serde_json::{json, Map, Value};

const NOW: i64 = 1_791_311_000 * SECOND;

#[derive(Deserialize)]
struct Seeded {
    key: String,
    url: String,
    version: String,
    fetched_at: String,
    task_id: String,
    content_sha1: String,
    change_seq: Option<i64>,
    change_row: Option<i64>,
}

fn record(r: &Record) -> Map<String, Value> {
    let value = json!({
        "url": r.url, "domain": r.domain, "title": r.title, "published": r.published, "author": r.author,
        "lang": r.lang, "text": r.text, "html": "", "fetched_at": r.fetched_at, "lastmod": "", "etag": "",
        "content_sha1": r.content_sha1, "source": SOURCE, "captured_at": r.captured_at, "doc_id": r.doc_id,
        "assigned_url": r.assigned_url, "final_url": r.final_url, "canonical": r.canonical, "status": r.status,
        "page_type": r.page_type, "description": r.description, "json_ld_types": r.json_ld_types,
        "headings": r.headings, "text_sha256": r.text_sha256, "task_id": r.task_id, "miner": r.miner,
        "validator": r.validator, "validators": r.validators,
    });
    value.as_object().cloned().unwrap()
}

fn page(p: &Page) -> Value {
    let mut out = record(&p.record);
    out.insert("key".into(), p.key.clone().into());
    out.insert("version".into(), p.version.clone().into());
    out.into()
}

fn change(c: &Change) -> Value {
    let mut out = match &c.body {
        Body::Page { record: r, version } => {
            let mut out = record(r);
            for name in ["html", "lastmod", "etag"] {
                out.remove(name);
            }
            out.insert("version".into(), version.clone().into());
            out
        }
        Body::Removed { url, domain } => json!({"url": url, "domain": domain, "assigned_url": url}).as_object().cloned().unwrap(),
    };
    out.insert("key".into(), c.key.clone().into());
    out.insert("kind".into(), c.kind.as_str().into());
    out.insert("previous_content_sha1".into(), c.previous_content_sha1.clone().into());
    out.insert("published_at".into(), c.published_at.clone().into());
    out.into()
}

fn current(c: &Current) -> Value {
    json!({"url": c.url, "version": c.version, "fetched_at": c.fetched_at, "task_id": c.task_id, "content_sha1": c.content_sha1, "change_seq": c.change_seq, "change_row": c.change_row})
}

fn main() {
    let work = PathBuf::from(std::env::args().nth(1).expect("work directory"));
    let read = |name: &str| std::fs::read(work.join(name)).unwrap();
    let batches: Vec<Vec<Job>> = serde_json::from_slice(&read("batches.json")).unwrap();
    let seed: Vec<Seeded> = serde_json::from_slice(&read("seed.json")).unwrap();
    let index_dir = work.join("rust-index");
    let changes_dir = work.join("rust-changes");
    let _ = std::fs::remove_dir_all(&index_dir);
    let _ = std::fs::remove_dir_all(&changes_dir);
    let index = VersionIndex::open(&index_dir, 256 << 20).unwrap();
    let seeded: Vec<(String, Current)> = seed
        .into_iter()
        .map(|s| {
            let current = Current { url: s.url, version: s.version, fetched_at: s.fetched_at, task_id: s.task_id, content_sha1: s.content_sha1, change_seq: s.change_seq, change_row: s.change_row };
            (s.key, current)
        })
        .collect();
    index.store(&seeded).unwrap();
    let uploads = LocalUploads::new(work.join("uploads"));
    let feed = LocalFeed::new(&changes_dir, 1).unwrap();
    let mut dumped = Vec::new();
    for (n, jobs) in batches.iter().enumerate() {
        let now = NOW + 60 * SECOND * n as i64;
        let reads = worker::read_all(jobs, &uploads, now, 8);
        let mut per_job = Vec::new();
        for (job, read) in jobs.iter().zip(&reads) {
            let Some(read) = read else { continue };
            per_job.push(match read {
                Ok((pages, failed)) => json!({
                    "task_id": job.task_id,
                    "records": pages.iter().map(page).collect::<Vec<_>>(),
                    "failed": failed.iter().map(|f| json!({"url": f.url, "task_id": f.task_id})).collect::<Vec<_>>(),
                }),
                Err(fault) => json!({"task_id": job.task_id, "fault": format!("{fault:?}")}),
            });
        }
        let batch = worker::write(jobs, reads, &index, &feed, now).unwrap();
        let rows = outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed, &batch.removed);
        let published = batch.changes.iter().filter(|c| matches!(c.body, Body::Page { .. })).count();
        dumped.push(json!({
            "now": now / SECOND,
            "jobs": per_job,
            "changes": batch.changes[..published].iter().map(change).collect::<Vec<_>>(),
            "removals": batch.changes[published..].iter().map(change).collect::<Vec<_>>(),
            "unchanged": batch.unchanged.iter().map(|p| json!({"key": p.key, "version": p.version, "fetched_at": p.record.fetched_at})).collect::<Vec<_>>(),
            "stored": batch.change_seq.map(|seq| worker::indexed(&batch.changes[..published], seq as i64)).unwrap_or_default().iter().map(|(key, c)| {
                let mut out = current(c);
                out["key"] = key.clone().into();
                out
            }).collect::<Vec<_>>(),
            "touched": batch.unchanged.iter().map(|p| json!([p.key, p.record.fetched_at])).collect::<Vec<_>>(),
            "removed": batch.removed.iter().map(|r| json!([r.key, r.version])).collect::<Vec<_>>(),
            "change_file": batch.change_seq.map(|seq| changes_dir.join(format!("{seq:012}.parquet")).display().to_string()),
            "outcomes": rows.iter().map(|o| json!({"url": o.url, "host": o.host, "outcome": o.outcome, "task_id": o.task_id})).collect::<Vec<_>>(),
            "lost": batch.lost,
            "acked": batch.finalized,
            "retry": batch.retry,
        }));
    }
    let pages: Map<String, Value> = index.pages().unwrap().iter().map(|(key, c)| (key.clone(), current(c))).collect();
    let withdrawn: Vec<String> = index.withdrawn().unwrap().into_iter().map(|(task_id, _)| task_id).collect();
    let dump = json!({"batches": dumped, "index": {"pages": pages, "withdrawn": withdrawn}});
    std::fs::write(work.join("rust.json"), serde_json::to_vec(&dump).unwrap()).unwrap();
    println!("rust: {} batches, {} pages indexed", batches.len(), pages.len());
}

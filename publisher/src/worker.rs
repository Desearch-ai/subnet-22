//! One publish batch: read the uploads, keep publishable pages, write them to one change file with a row per task withdrawn.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as Atomic};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::blocked::looks_blocked;
use crate::changes::{self, Body, Change, Kind};
use crate::reading::{changed, read_rows, RangeRead};
use crate::records::{build_record, iso, publish_window, record_key, record_version, Context as RecordContext, Record, Row};

pub const WITHDRAW: &str = "withdraw";
pub const EMBED: &str = "embed";
/// Why an upload whose version is not the one its completion named is lost.
pub const CHANGED: &str = "changed after completion";

/// A publish job as the task API queues it.
#[derive(Clone, Debug, Deserialize)]
pub struct Job {
    pub task_id: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub miner: Option<String>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub etag: Option<String>,
    /// An embed job's input file in the temp bucket, deleted with the upload when the job is done.
    #[serde(default)]
    pub input_key: Option<String>,
    /// The embedding model an embed job's vectors come from.
    #[serde(default)]
    pub model: Option<String>,
    /// An embed job's pages, as the planner listed them, sent back to it when the task is taken back or lost before publishing.
    #[serde(default)]
    pub pages: Vec<Value>,
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub skip: Vec<String>,
    #[serde(default)]
    pub completed_at: Value,
    #[serde(default)]
    pub claim_ttl: Value,
    /// Python's `job.get("validator", "")`: absent is empty, null stays None.
    #[serde(default = "no_validator")]
    pub validator: Option<String>,
    #[serde(default)]
    pub validators: Vec<String>,
    /// The tasks a withdraw job takes back.
    #[serde(default)]
    pub task_ids: Vec<String>,
}

fn no_validator() -> Option<String> {
    Some(String::new())
}

impl Job {
    pub fn is(&self, kind: &str) -> bool {
        self.kind.as_deref() == Some(kind)
    }
}

/// An assigned URL that was not published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failed {
    pub url: String,
    pub task_id: String,
}

/// A publishable page with its index key and content version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    pub key: String,
    pub version: String,
    pub record: Record,
}

/// Why a job's upload cannot be read now.
#[derive(Debug)]
pub enum Fault {
    /// Expired, replaced or unreadable: its URLs fail and the job is done.
    Gone(&'static str),
    /// Anything else: the job stays queued and is tried again.
    Retry(String),
}

/// Where a job's upload is read from.
pub trait Uploads: Sync {
    fn open(&self, job: &Job) -> Result<Box<dyn RangeRead>, Fault>;
}

/// Where change files go, each numbered in the order written.
pub trait ChangeFeed: Sync {
    /// Keeps one change file and returns its number in the feed.
    fn append(&self, file: Vec<u8>, rows: usize) -> Result<u64>;
}

/// A job's publishable pages and the assigned URLs it could not publish.
pub fn read_job(job: &Job, uploads: &dyn Uploads, now: i64) -> Result<(Vec<Page>, Vec<Failed>), Fault> {
    if job.is(EMBED) {
        return Err(Fault::Retry("embed jobs are not published here yet".into()));
    }
    let source = uploads.open(job)?;
    let assigned = job.urls.iter().collect::<HashSet<_>>().len();
    let rows = match read_rows(&*source, assigned) {
        Ok(Some(rows)) => rows,
        Ok(None) => return Err(Fault::Gone("unreadable")),
        Err(error) if changed(&error) => return Err(Fault::Gone(CHANGED)),
        Err(error) => return Err(Fault::Retry(format!("reading the upload: {error}"))),
    };
    collect(job, &rows, now).map_err(|error| Fault::Retry(format!("{error:#}")))
}

pub fn collect(job: &Job, rows: &[Row], now: i64) -> Result<(Vec<Page>, Vec<Failed>)> {
    let window = publish_window(python_float(&job.completed_at)?, python_float(&job.claim_ttl)?, now)?;
    let miner = job.miner.as_deref();
    let context = RecordContext {
        task_id: &job.task_id,
        miner: miner.unwrap_or_default(),
        window,
        captured_at: now,
        validator: job.validator.as_deref(),
        validators: &job.validators,
    };
    let given: HashSet<&str> = job.urls.iter().map(String::as_str).collect();
    let skipped: HashSet<&str> = job.skip.iter().map(String::as_str).collect();
    let assigned: HashSet<&str> = given.difference(&skipped).copied().collect();
    let mut pages = Vec::new();
    let mut published = HashSet::new();
    for row in rows {
        if !publishable(row, &assigned) {
            continue;
        }
        published.insert(row.url.as_deref().unwrap_or_default());
        miner.context("a publish job without a miner")?;
        let record = build_record(row, &context)?;
        pages.push(Page { key: record_key(&record)?, version: record_version(&record), record });
    }
    let mut missed: Vec<&str> = given.difference(&published).copied().collect();
    missed.sort_unstable();
    Ok((pages, failed(job, missed)))
}

pub fn publishable(row: &Row, assigned: &HashSet<&str>) -> bool {
    let (Some(url), Some(text)) = (row.url.as_deref(), row.text.as_deref()) else {
        return false;
    };
    if row.error.is_some() || text.is_empty() || !assigned.contains(url) {
        return false;
    }
    !looks_blocked(row.status, text, row.title.as_deref().unwrap_or(""))
}

fn failed<'a>(job: &Job, urls: impl IntoIterator<Item = &'a str>) -> Vec<Failed> {
    urls.into_iter().map(|url| Failed { url: url.to_string(), task_id: job.task_id.clone() }).collect()
}

/// Python's `float(value)` for a value that may be absent: None where Python's truthiness says false.
fn python_float(value: &Value) -> Result<Option<f64>> {
    Ok(match value {
        Value::Null | Value::Bool(false) => None,
        Value::Bool(true) => Some(1.0),
        Value::Number(n) => n.as_f64().filter(|&f| f != 0.0),
        Value::String(s) if s.is_empty() => None,
        Value::String(s) => Some(s.trim().parse::<f64>().with_context(|| format!("{s:?} is not a number"))?),
        other => bail!("{other} is not a number"),
    })
}

fn rank(page: &Page) -> (bool, &str, &str) {
    (page.record.assigned_url == page.record.url, &page.record.fetched_at, &page.version)
}

/// Pages in the order first seen, each key holding its best-ranked page.
#[derive(Default)]
pub struct Chosen {
    pages: Vec<Page>,
    at: HashMap<String, usize>,
}

impl Chosen {
    pub fn offer(&mut self, page: Page) {
        match self.at.get(&page.key) {
            Some(&i) => {
                if rank(&page).cmp(&rank(&self.pages[i])) == Ordering::Greater {
                    self.pages[i] = page;
                }
            }
            None => {
                self.at.insert(page.key.clone(), self.pages.len());
                self.pages.push(page);
            }
        }
    }

    pub fn into_pages(self) -> Vec<Page> {
        self.pages
    }
}

/// The change rows for a batch's pages.
pub fn page_changes(pages: Vec<Page>, published_at: &str) -> Vec<Change> {
    pages
        .into_iter()
        .map(|page| Change {
            key: page.key,
            kind: Kind::Page,
            previous_content_sha1: String::new(),
            published_at: published_at.to_string(),
            body: Body::Page { record: Box::new(page.record), version: page.version },
        })
        .collect()
}

/// What one batch did, for the queue and the bot.
#[derive(Debug, Default)]
pub struct Batch {
    /// The pages, then a row per task withdrawn, as the change file holds them.
    pub changes: Vec<Change>,
    pub failed: Vec<Failed>,
    /// URLs of tasks taken back or lost: nothing about their sites, so the bot retries them without counting a failure.
    pub dropped: Vec<Failed>,
    /// Tasks taken back in this batch.
    pub withdrawn: Vec<String>,
    /// Tasks done with: acknowledged and their uploads deleted.
    pub finalized: Vec<String>,
    /// Tasks whose upload was gone before publishing.
    pub lost: Vec<String>,
    /// Tasks left queued to try again, with the reason.
    pub retry: Vec<(String, String)>,
    pub change_seq: Option<u64>,
    pub change_bytes: usize,
    /// Seconds the change file took to write and number.
    pub stored_s: f64,
}

/// Publishes one batch of jobs; `now` is microseconds since the epoch, `readers` the uploads read at once, `refused` the tasks already taken back.
pub fn publish(jobs: &[Job], uploads: &dyn Uploads, refused: &HashSet<String>, feed: &dyn ChangeFeed, now: i64, readers: usize) -> Result<Batch> {
    let reads = read_all(jobs, uploads, now, readers);
    write(jobs, reads, refused, feed, now)
}

/// The tasks a batch's withdraw jobs take back, each once, in order.
pub fn withdrawn_in(jobs: &[Job]) -> Vec<String> {
    let mut named = HashSet::new();
    jobs.iter().filter(|job| job.is(WITHDRAW)).flat_map(|job| job.task_ids.iter().cloned()).filter(|task_id| named.insert(task_id.clone())).collect()
}

/// The second half of a batch, from the jobs' reads in the jobs' order; the next batch can be read meanwhile.
pub fn write(jobs: &[Job], reads: Vec<Read>, refused: &HashSet<String>, feed: &dyn ChangeFeed, now: i64) -> Result<Batch> {
    let mut batch = Batch::default();
    let withdrawn = withdrawn_in(jobs);
    let taken_back: HashSet<&str> = withdrawn.iter().map(String::as_str).collect();
    let mut chosen = Chosen::default();
    for (job, read) in jobs.iter().zip(reads) {
        let Some(read) = read else {
            batch.finalized.push(job.task_id.clone());
            continue;
        };
        if refused.contains(&job.task_id) || taken_back.contains(job.task_id.as_str()) {
            batch.dropped.extend(failed(job, job.urls.iter().map(String::as_str)));
            batch.finalized.push(job.task_id.clone());
            continue;
        }
        match read {
            Ok((pages, missed)) => {
                for page in pages {
                    chosen.offer(page);
                }
                batch.failed.extend(missed);
                batch.finalized.push(job.task_id.clone());
            }
            Err(Fault::Gone(why)) => {
                eprintln!("task={} upload {why} before publishing", job.task_id);
                batch.lost.push(job.task_id.clone());
                if !job.is(EMBED) {
                    batch.dropped.extend(failed(job, job.urls.iter().map(String::as_str)));
                }
                batch.finalized.push(job.task_id.clone());
            }
            Err(Fault::Retry(why)) => batch.retry.push((job.task_id.clone(), why)),
        }
    }
    let published_at = iso(now);
    let mut changes = page_changes(chosen.into_pages(), &published_at);
    changes.extend(withdrawn.iter().map(|task_id| Change::task_withdrawn(task_id, &published_at)));
    if !changes.is_empty() {
        let started = Instant::now();
        let file = changes::encode(&changes)?;
        batch.change_bytes = file.len();
        batch.change_seq = Some(feed.append(file, changes.len())?);
        batch.stored_s = started.elapsed().as_secs_f64();
    }
    batch.withdrawn = withdrawn;
    batch.changes = changes;
    Ok(batch)
}

pub type Read = Option<Result<(Vec<Page>, Vec<Failed>), Fault>>;

/// Each job's pages, or what stopped it, in the jobs' order; None for withdraw jobs, which read nothing.
pub fn read_all(jobs: &[Job], uploads: &dyn Uploads, now: i64, readers: usize) -> Vec<Read> {
    let next = AtomicUsize::new(0);
    let mut found: Vec<(usize, Read)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..readers.clamp(1, jobs.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Atomic::Relaxed);
                        let Some(job) = jobs.get(i) else {
                            return done;
                        };
                        done.push((i, (!job.is(WITHDRAW)).then(|| read_job(job, uploads, now))));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().expect("upload reader panicked")).collect()
    });
    found.sort_by_key(|(i, _)| *i);
    found.into_iter().map(|(_, read)| read).collect()
}

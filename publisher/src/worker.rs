//! One publish batch: read the uploads, keep publishable pages, decide new, changed or unchanged, write the change file with a row per task withdrawn, then the index.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as Atomic};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::blocked::looks_blocked;
use crate::changes::{self, Body, Change, Kind};
use crate::index::{Current, VersionIndex};
use crate::reading::{read_rows, RangeRead};
use crate::records::{build_record, iso, publish_window, record_key, record_version, Context as RecordContext, Record, Row, SECOND};

pub const WITHDRAW: &str = "withdraw";
pub const EMBED: &str = "embed";

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

/// New and changed pages, and pages seen again unchanged, against the version index; a version from a withdrawn task counts as none.
pub fn decide(pages: Vec<Page>, index: &VersionIndex, published_at: &str) -> Result<(Vec<Change>, Vec<Page>)> {
    let keys: Vec<&str> = pages.iter().map(|page| page.key.as_str()).collect();
    let currents = index.current_many(&keys)?;
    let withdrawn = index.withdrawn_among(&currents.values().map(|current| current.task_id.as_str()).collect::<Vec<_>>())?;
    let (mut changes, mut unchanged) = (Vec::new(), Vec::new());
    for page in pages {
        let (kind, previous) = match currents.get(&page.key).filter(|current| !withdrawn.contains(&current.task_id)) {
            None => (Kind::New, String::new()),
            Some(current)
                if current.version == page.version
                    || (current.fetched_at.as_str(), current.version.as_str()) >= (page.record.fetched_at.as_str(), page.version.as_str()) =>
            {
                unchanged.push(page);
                continue;
            }
            Some(current) => (Kind::Changed, current.content_sha1.clone()),
        };
        changes.push(Change {
            key: page.key,
            kind,
            previous_content_sha1: previous,
            published_at: published_at.to_string(),
            body: Body::Page { record: Box::new(page.record), version: page.version },
        });
    }
    Ok((changes, unchanged))
}

/// What one batch did, for the queue and the bot.
#[derive(Debug, Default)]
pub struct Batch {
    /// New and changed pages, then a row per task withdrawn, as the change file holds them.
    pub changes: Vec<Change>,
    pub unchanged: Vec<Page>,
    pub failed: Vec<Failed>,
    /// Tasks taken back; their pages leave the index later, a chunk at a time.
    pub withdrawn: Vec<String>,
    /// Tasks done with: acknowledged and their uploads deleted.
    pub finalized: Vec<String>,
    /// Tasks whose upload was gone before publishing.
    pub lost: Vec<String>,
    /// Tasks left queued to try again, with the reason.
    pub retry: Vec<(String, String)>,
    pub change_seq: Option<u64>,
    pub change_bytes: usize,
    /// Seconds into the write when the pages were decided, the change file stored and the index updated.
    pub steps: [f64; 3],
}

/// Publishes one batch of jobs; `now` is microseconds since the epoch, `readers` the uploads read at once.
pub fn publish(jobs: &[Job], uploads: &dyn Uploads, index: &VersionIndex, feed: &dyn ChangeFeed, now: i64, readers: usize) -> Result<Batch> {
    let reads = read_all(jobs, uploads, now, readers);
    write(jobs, reads, index, feed, now)
}

/// The second half of a batch, from the jobs' reads in the jobs' order; the next batch can be read meanwhile.
pub fn write(jobs: &[Job], reads: Vec<Read>, index: &VersionIndex, feed: &dyn ChangeFeed, now: i64) -> Result<Batch> {
    let started = Instant::now();
    let mut batch = Batch::default();
    let mut named = HashSet::new();
    let withdrawn: Vec<String> =
        jobs.iter().filter(|job| job.is(WITHDRAW)).flat_map(|job| job.task_ids.iter().cloned()).filter(|task_id| named.insert(task_id.clone())).collect();
    if !withdrawn.is_empty() {
        index.withdraw(&withdrawn, now as f64 / SECOND as f64)?;
    }
    let mut chosen = Chosen::default();
    for (job, read) in jobs.iter().zip(reads) {
        let Some(read) = read else {
            batch.finalized.push(job.task_id.clone());
            continue;
        };
        if index.is_withdrawn(&job.task_id)? {
            batch.failed.extend(failed(job, job.urls.iter().map(String::as_str)));
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
                    batch.failed.extend(failed(job, job.urls.iter().map(String::as_str)));
                }
                batch.finalized.push(job.task_id.clone());
            }
            Err(Fault::Retry(why)) => batch.retry.push((job.task_id.clone(), why)),
        }
    }
    let published_at = iso(now);
    let (mut changes, unchanged) = decide(chosen.into_pages(), index, &published_at)?;
    batch.steps[0] = started.elapsed().as_secs_f64();
    let published = changes.len();
    changes.extend(withdrawn.iter().map(|task_id| Change::task_withdrawn(task_id, &published_at)));
    // The change file is written before the index learns of it, so a crash only replays.
    if !changes.is_empty() {
        let file = changes::encode(&changes)?;
        batch.change_bytes = file.len();
        let seq = feed.append(file, changes.len())?;
        batch.change_seq = Some(seq);
        batch.steps[1] = started.elapsed().as_secs_f64();
        index.store(&indexed(&changes[..published], seq as i64))?;
    }
    index.touch(&unchanged.iter().map(|page| (page.key.clone(), page.record.fetched_at.clone())).collect::<Vec<_>>())?;
    batch.steps[2] = started.elapsed().as_secs_f64();
    batch.withdrawn = withdrawn;
    batch.changes = changes;
    batch.unchanged = unchanged;
    Ok(batch)
}

/// The index entries for a change file's new and changed pages: where each full record sits.
pub fn indexed(changes: &[Change], seq: i64) -> Vec<(String, Current)> {
    changes
        .iter()
        .enumerate()
        .filter_map(|(row, change)| match &change.body {
            Body::Page { record, version } => Some((
                change.key.clone(),
                Current {
                    url: record.url.clone(),
                    version: version.clone(),
                    fetched_at: record.fetched_at.clone(),
                    task_id: record.task_id.clone(),
                    content_sha1: record.content_sha1.clone(),
                    change_seq: Some(seq),
                    change_row: Some(row as i64),
                },
            )),
            Body::Removed { .. } | Body::TaskWithdrawn { .. } => None,
        })
        .collect()
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

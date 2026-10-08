//! The publisher as a service: claim batches from the task API's queue, read the next ones while one is written, then report and acknowledge.

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use redis::aio::ConnectionManager;
use regex::Regex;
use tokio::runtime::Handle;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::changes::{self, Body, Change, Kind};
use crate::index::{Current, VersionIndex};
use crate::outcomes;
use crate::queue::PublishQueue;
use crate::reading::{RangeRead, Remote, Traffic};
use crate::records::iso;
use crate::snapshot;
use crate::worker::{self, ChangeFeed, Fault, Job, Read, Uploads};
use desearch::feeds::{CHANGES, OUTCOMES};
use desearch::r2::{self, Bucket, Credentials, PARQUET};

const RETRY_DELAY: Duration = Duration::from_secs(5);
const SUMMARY_EVERY: Duration = Duration::from_secs(60);
const CHANGE_PART_BYTES: usize = 8 << 20;
const CHANGE_PARTS_AT_ONCE: usize = 8;

pub struct Settings {
    pub redis_url: String,
    pub claim_ttl: f64,
    pub batch: usize,
    /// Uploads read at once.
    pub readers: usize,
    /// Ranged reads in flight for one upload.
    pub ranges: usize,
    /// Batches claimed and reading while one is written.
    pub ahead: usize,
    pub index: PathBuf,
    pub cache_bytes: usize,
    /// Stop after this many passes in a row found nothing; 0 runs until stopped.
    pub idle_exit: u32,
    pub idle_delay: Duration,
}

impl Settings {
    pub fn from_env() -> Result<Self> {
        if text("TASK_API_EMBED_TASKS", "0") == "1" {
            bail!("TASK_API_EMBED_TASKS=1: the publisher does not write embed inputs");
        }
        let batch = number("PUBLISHER_BATCH", 40)?;
        Ok(Settings {
            redis_url: text("TASK_API_REDIS", "redis://localhost:6379/15"),
            claim_ttl: number("TASK_API_PUBLISH_TTL", 600.0)?,
            batch,
            readers: number("PUBLISHER_READERS", batch)?,
            ranges: number("PUBLISHER_RANGES", 8)?,
            ahead: number("PUBLISHER_AHEAD", 2)?,
            index: PathBuf::from(text("PUBLISHER_INDEX", "publisher-index")),
            cache_bytes: number::<usize>("PUBLISHER_CACHE_MB", 1024)? << 20,
            idle_exit: number("PUBLISHER_IDLE_EXIT", 0)?,
            idle_delay: Duration::from_secs_f64(number("PUBLISHER_IDLE_DELAY_S", 2.0)?),
        })
    }
}

fn text(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn number<T: FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name).ok().filter(|v| !v.is_empty()) {
        Some(value) => value.trim().parse().map_err(|_| anyhow::anyhow!("{name}={value:?} is not a number")),
        None => Ok(default),
    }
}

/// The temp bucket uploads wait in and the permanent bucket pages are published to, as the Python service names them.
pub fn buckets_from_env(errors: Arc<AtomicU64>) -> Result<(Bucket, Bucket)> {
    let endpoint = std::env::var("CF_R2_ENDPOINT").context("CF_R2_ENDPOINT is not set")?;
    let credentials = Credentials {
        access_key: std::env::var("CF_R2_ACCESS_KEY_ID").context("CF_R2_ACCESS_KEY_ID is not set")?,
        secret_key: std::env::var("CF_R2_SECRET_ACCESS_KEY").context("CF_R2_SECRET_ACCESS_KEY is not set")?,
        region: text("CF_R2_REGION", "auto"),
    };
    let http = r2::client()?;
    let temp = Bucket::new(
        http.clone(),
        &endpoint,
        &text("CF_R2_BUCKET", "subnet-22"),
        &std::env::var("TASK_API_R2_PREFIX").unwrap_or_default(),
        credentials.clone(),
        errors.clone(),
    )?;
    let pages = Bucket::new(
        http,
        &endpoint,
        &text("CF_R2_PAGES_BUCKET", "desearch-pages"),
        &std::env::var("CF_R2_PAGES_PREFIX").unwrap_or_default(),
        credentials,
        errors,
    )?;
    if (&temp.bucket, &temp.prefix) == (&pages.bucket, &pages.prefix) {
        bail!("CF_R2_PAGES_BUCKET is {:?}, the same place uploads are kept and expired; point it at the permanent bucket", pages.bucket);
    }
    Ok((temp, pages))
}

#[derive(Default)]
pub struct Metrics {
    pub tasks: AtomicU64,
    pub batches: AtomicU64,
    pub changed: AtomicU64,
    pub unchanged: AtomicU64,
    pub removed: AtomicU64,
    pub failed_urls: AtomicU64,
    pub lost: AtomicU64,
    pub retried: AtomicU64,
    pub r2_errors: Arc<AtomicU64>,
    pub traffic: Arc<Traffic>,
}

/// What every part of the loop shares.
pub struct Shared {
    pub queue: PublishQueue,
    pub temp: Bucket,
    pub pages: Bucket,
    pub index: Arc<VersionIndex>,
    pub settings: Settings,
    pub metrics: Arc<Metrics>,
    pub handle: Handle,
}

impl Shared {
    pub async fn connect(settings: Settings, temp: Bucket, pages: Bucket, metrics: Arc<Metrics>) -> Result<Arc<Self>> {
        if settings.index.is_file() {
            bail!("PUBLISHER_INDEX {} is a file, not an index folder", settings.index.display());
        }
        let redis = redis::Client::open(settings.redis_url.as_str())?.get_connection_manager().await.context("connecting to TASK_API_REDIS")?;
        let index = Arc::new(VersionIndex::open(&settings.index, settings.cache_bytes)?);
        Ok(Arc::new(Shared { queue: PublishQueue::new(redis, settings.claim_ttl), temp, pages, index, settings, metrics, handle: Handle::current() }))
    }

    fn redis(&self) -> &ConnectionManager {
        &self.queue.redis
    }
}

/// Uploads read from the temp bucket after a HEAD that checks they are still the one validated.
struct R2Uploads {
    temp: Bucket,
    handle: Handle,
    ranges: usize,
    traffic: Arc<Traffic>,
}

impl Uploads for R2Uploads {
    fn open(&self, job: &Job) -> Result<Box<dyn RangeRead>, Fault> {
        let key = job.key.as_deref().ok_or_else(|| Fault::Retry("a publish job without an upload key".into()))?;
        let head = match self.handle.block_on(self.temp.head(key)) {
            Ok(Some(head)) => head,
            Ok(None) => return Err(Fault::Gone("expired")),
            Err(error) => return Err(Fault::Retry(format!("HEAD {key}: {error}"))),
        };
        if let Some(etag) = job.etag.as_deref().filter(|e| !e.is_empty()) {
            if head.etag.trim_matches('"') != etag.trim_matches('"') {
                return Err(Fault::Gone("changed"));
            }
        }
        Ok(Box::new(Remote {
            bucket: self.temp.clone(),
            key: key.to_string(),
            head,
            handle: self.handle.clone(),
            ranges_at_once: self.ranges,
            traffic: self.traffic.clone(),
        }))
    }
}

/// Change files in the pages bucket, numbered in the `changes` feed.
struct R2Changes {
    pages: Bucket,
    redis: ConnectionManager,
    handle: Handle,
    day: String,
}

impl ChangeFeed for R2Changes {
    fn append(&self, file: Vec<u8>, rows: usize) -> Result<u64> {
        let key = format!("changes/dt={}/{}.parquet", self.day, uuid::Uuid::new_v4().simple());
        self.handle.block_on(async {
            self.pages
                .put_in_parts(&key, Bytes::from(file), PARQUET, CHANGE_PART_BYTES, CHANGE_PARTS_AT_ONCE)
                .await
                .with_context(|| format!("writing {key}"))?;
            CHANGES.number(&self.pages, &self.redis, &key, rows).await
        })
    }
}

pub struct Claimed {
    jobs: Vec<Job>,
    reads: JoinHandle<Vec<Read>>,
    started: Instant,
}

/// Claims the next batch and starts reading its uploads.
pub async fn start_batch(shared: Arc<Shared>) -> Result<Option<Claimed>> {
    let jobs = shared.queue.claim(shared.settings.batch).await?;
    if jobs.is_empty() {
        return Ok(None);
    }
    let started = Instant::now();
    let uploads =
        R2Uploads { temp: shared.temp.clone(), handle: shared.handle.clone(), ranges: shared.settings.ranges, traffic: shared.metrics.traffic.clone() };
    let (reading, readers, now) = (jobs.clone(), shared.settings.readers, now_us());
    let reads = tokio::task::spawn_blocking(move || worker::read_all(&reading, &uploads, now, readers));
    Ok(Some(Claimed { jobs, reads, started }))
}

/// Writes a batch already read: change file, index, outcomes, acknowledgements; returns the jobs finished.
pub async fn write_batch(shared: &Shared, claimed: Claimed) -> Result<usize> {
    let Claimed { jobs, reads, started } = claimed;
    let reads = reads.await.context("reading the batch")?;
    let read_s = started.elapsed().as_secs_f64();
    let ids: Vec<&str> = jobs.iter().map(|job| job.task_id.as_str()).collect();
    shared.queue.extend_claims(&ids).await?;
    let uploads: HashMap<String, Vec<String>> =
        jobs.iter().map(|job| (job.task_id.clone(), [job.key.clone(), job.input_key.clone()].into_iter().flatten().collect())).collect();
    let now = now_us();
    let feed = R2Changes { pages: shared.pages.clone(), redis: shared.redis().clone(), handle: shared.handle.clone(), day: day(now) };
    let index = shared.index.clone();
    let batch = tokio::task::spawn_blocking(move || worker::write(&jobs, reads, &index, &feed, now)).await??;
    let written_s = started.elapsed().as_secs_f64();
    for task_id in &batch.lost {
        if let Err(error) = shared.queue.mark_lost(task_id).await {
            eprintln!("task={task_id} could not be marked lost: {error:#}");
        }
    }
    for (task_id, why) in &batch.retry {
        eprintln!("task={task_id} publish failed; it will be retried: {why}");
    }
    report_outcomes(shared, &batch, now).await;
    let uploads = &uploads;
    let finishing = batch.finalized.iter().map(|task_id| async move {
        if let Err(error) = shared.queue.ack(task_id).await {
            eprintln!("task={task_id} could not be acknowledged: {error:#}");
            return;
        }
        // Deleting a published upload is housekeeping; a slow DELETE must not hold up the next batch.
        for key in uploads.get(task_id).into_iter().flatten() {
            let (temp, key) = (shared.temp.clone(), key.clone());
            tokio::spawn(async move { temp.delete(&key).await });
        }
    });
    futures::future::join_all(finishing).await;
    let published = batch.changes.len() - batch.removed.len();
    let m = &shared.metrics;
    m.batches.fetch_add(1, Ordering::Relaxed);
    m.tasks.fetch_add(batch.finalized.len() as u64, Ordering::Relaxed);
    m.changed.fetch_add(published as u64, Ordering::Relaxed);
    m.unchanged.fetch_add(batch.unchanged.len() as u64, Ordering::Relaxed);
    m.removed.fetch_add(batch.removed.len() as u64, Ordering::Relaxed);
    m.failed_urls.fetch_add(batch.failed.len() as u64, Ordering::Relaxed);
    m.lost.fetch_add(batch.lost.len() as u64, Ordering::Relaxed);
    m.retried.fetch_add(batch.retry.len() as u64, Ordering::Relaxed);
    println!(
        "published {} tasks, {} pages new or changed, {} withdrawn in {:.1}s (read by {:.1}s, written by {:.1}s: decided {:.1}s, stored {:.1}s, indexed {:.1}s); {} unchanged, {} URLs failed, {} lost, {} left to retry, change file {}",
        batch.finalized.len(),
        published,
        batch.removed.len(),
        started.elapsed().as_secs_f64(),
        read_s,
        written_s,
        batch.steps[0],
        batch.steps[1],
        batch.steps[2],
        batch.unchanged.len(),
        batch.failed.len(),
        batch.lost.len(),
        batch.retry.len(),
        batch.change_seq.map_or("none".to_string(), |seq| format!("#{seq} ({} bytes)", batch.change_bytes)),
    );
    Ok(batch.finalized.len())
}

/// What became of every URL, for the bot; a failure here is logged and the batch still finishes.
async fn report_outcomes(shared: &Shared, batch: &worker::Batch, now: i64) {
    let rows = outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed, &batch.removed);
    if rows.is_empty() {
        return;
    }
    let written = async {
        let count = rows.len();
        let file = tokio::task::spawn_blocking(move || desearch::outcomes::encode(&rows, now)).await??;
        let key = format!("outcomes/dt={}/{}.parquet", day(now), uuid::Uuid::new_v4().simple());
        shared.temp.put(&key, Bytes::from(file), PARQUET, None).await?;
        OUTCOMES.number(&shared.temp, shared.redis(), &key, count).await?;
        anyhow::Ok(count)
    };
    if let Err(error) = written.await {
        eprintln!("could not write the batch's outcomes: {error:#}");
    }
}

#[derive(Default)]
struct Snapshots {
    day: String,
    running: Option<JoinHandle<()>>,
}

impl Snapshots {
    /// A copy of the version index a day, made in the background from a checkpoint while publishing goes on.
    fn daily(&mut self, shared: &Arc<Shared>) {
        if self.running.as_ref().is_some_and(|running| !running.is_finished()) {
            return;
        }
        let today = day(now_us());
        if today == self.day {
            return;
        }
        self.day = today.clone();
        let shared = shared.clone();
        self.running = Some(tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            match snapshot::upload(&shared.index, &shared.settings.index, &shared.pages, &today, &shared.handle) {
                Ok(Some(rows)) => println!("index snapshot {today}: {rows} pages in {:.0}s", started.elapsed().as_secs_f64()),
                Ok(None) => {}
                Err(error) => eprintln!("could not snapshot the version index: {error:#}"),
            }
        }));
    }
}

/// Publishes until `stop` turns true, then finishes the batches already claimed and claims no more; `idle_exit` passes in a row with nothing to do also stop it.
pub async fn run(shared: Arc<Shared>, mut stop: watch::Receiver<bool>, idle_exit: u32) -> Result<()> {
    let mut idle = 0;
    let mut claimed: Vec<Claimed> = Vec::new();
    let mut snapshots = Snapshots::default();
    loop {
        let finishing = *stop.borrow() || (idle_exit > 0 && idle >= idle_exit);
        while !finishing && claimed.len() <= shared.settings.ahead {
            match start_batch(shared.clone()).await {
                Ok(Some(batch)) => claimed.push(batch),
                Ok(None) => break,
                Err(error) => {
                    eprintln!("claiming a batch failed: {error:#}");
                    pause(&mut stop, RETRY_DELAY).await;
                    break;
                }
            }
        }
        if claimed.is_empty() {
            if finishing {
                break;
            }
            idle += 1;
        } else {
            idle = 0;
            let current = claimed.remove(first_read(&claimed).await);
            if let Err(error) = write_batch(&shared, current).await {
                eprintln!("publish pass failed: {error:#}");
                pause(&mut stop, RETRY_DELAY).await;
            }
        }
        snapshots.daily(&shared);
        if let Err(error) = CHANGES.fill_holes(&shared.pages, shared.redis()).await {
            eprintln!("filling holes in the change feed failed: {error:#}");
        }
        if idle > 0 && !finishing {
            pause(&mut stop, shared.settings.idle_delay).await;
        }
    }
    if let Some(running) = snapshots.running {
        if tokio::time::timeout(Duration::from_secs(3), running).await.is_err() {
            println!("stopping with the index snapshot unfinished; it is made again on the next start");
        }
    }
    Ok(())
}

/// The oldest batch in hand whose uploads are all read; a page keeps its latest fetch whichever batch lands first, so one slow upload holds up only its own batch.
async fn first_read(claimed: &[Claimed]) -> usize {
    loop {
        if let Some(i) = claimed.iter().position(|batch| batch.reads.is_finished()) {
            return i;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn pause(stop: &mut watch::Receiver<bool>, delay: Duration) {
    if *stop.borrow() {
        return;
    }
    let _ = tokio::time::timeout(delay, stop.wait_for(|stopped| *stopped)).await;
}

/// A line a minute: pace, backlog and storage errors.
pub async fn summarize(shared: Arc<Shared>, mut stop: watch::Receiver<bool>) {
    let m = shared.metrics.clone();
    let counters = || {
        [&m.tasks, &m.changed, &m.unchanged, &m.failed_urls, &m.lost, &m.retried, m.r2_errors.as_ref(), &m.traffic.bytes, &m.traffic.requests]
            .map(|c| c.load(Ordering::Relaxed))
    };
    let mut before = counters();
    loop {
        if tokio::time::timeout(SUMMARY_EVERY, stop.wait_for(|stopped| *stopped)).await.is_ok() {
            return;
        }
        let now = counters();
        let d: Vec<u64> = now.iter().zip(before).map(|(n, b)| n - b).collect();
        before = now;
        let backlog = match shared.queue.backlog().await {
            Ok(b) => format!("{} ready, {} not yet published, {} set aside, {} lost in all", b.ready, b.waiting, b.set_aside, b.lost),
            Err(error) => format!("unknown ({error})"),
        };
        println!(
            "minute: {} tasks/min, {} pages new or changed, {} unchanged, {} URLs failed, {} lost, {} left to retry; backlog {backlog}; R2 {} errors, read {:.0} MB in {} requests",
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            d[6],
            d[7] as f64 / 1e6,
            d[8]
        );
    }
}

/// `publisher serve`: everything from the environment, until SIGTERM or SIGINT.
pub async fn serve() -> Result<()> {
    let settings = Settings::from_env()?;
    let metrics = Arc::new(Metrics::default());
    let (temp, pages) = buckets_from_env(metrics.r2_errors.clone())?;
    for bucket in [&temp, &pages] {
        bucket.check().await?;
    }
    let shared = Shared::connect(settings, temp, pages, metrics).await?;
    let (stopping, stop) = watch::channel(false);
    tokio::spawn(async move {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("a SIGTERM handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
        println!("stopping: finishing the batches in hand, claiming no more");
        let _ = stopping.send(true);
    });
    println!(
        "publishing {}/{} -> {}/{}, batches of {}, {} uploads read at once, index {}",
        shared.temp.bucket,
        shared.temp.prefix,
        shared.pages.bucket,
        shared.pages.prefix,
        shared.settings.batch,
        shared.settings.readers,
        shared.settings.index.display()
    );
    tokio::spawn(summarize(shared.clone(), stop.clone()));
    run(shared.clone(), stop, shared.settings.idle_exit).await?;
    shared.index.flush()?;
    println!("publisher stopped");
    Ok(())
}

/// A page URL kept from a sitemap address whose XML escapes were never decoded, as in `?amp%3Bid=1`.
static ESCAPED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)[?&]amp%3B|&(amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-f]+);").expect("a valid pattern"));
const REMOVED_PER_FILE: usize = 10_000;

pub fn escaped(url: &str) -> bool {
    ESCAPED.is_match(url)
}

/// `remove_escaped` with the storage, Redis and index `serve` uses.
pub async fn remove_escaped_from_env(apply: bool) -> Result<()> {
    let settings = Settings::from_env()?;
    let metrics = Arc::new(Metrics::default());
    let (temp, pages) = buckets_from_env(metrics.r2_errors.clone())?;
    pages.check().await?;
    let shared = Shared::connect(settings, temp, pages, metrics).await?;
    remove_escaped(&shared, apply).await?;
    Ok(())
}

/// Takes down every page whose URL still carries a sitemap's XML escapes, in change files of `removed` rows; without `apply` it only counts them.
pub async fn remove_escaped(shared: &Shared, apply: bool) -> Result<usize> {
    let index = shared.index.clone();
    let found: Vec<(String, Current)> = tokio::task::spawn_blocking(move || {
        let mut found = Vec::new();
        index.scan(|key, current| {
            if escaped(&current.url) {
                found.push((key, current));
            }
            Ok(())
        })?;
        anyhow::Ok(found)
    })
    .await??;
    println!("{} pages carry escaped URLs", found.len());
    for (key, current) in found.iter().take(5) {
        println!("  {key} {}", current.url);
    }
    if !apply || found.is_empty() {
        return Ok(found.len());
    }
    let count = found.len();
    let now = now_us();
    let feed = R2Changes { pages: shared.pages.clone(), redis: shared.redis().clone(), handle: shared.handle.clone(), day: day(now) };
    let index = shared.index.clone();
    tokio::task::spawn_blocking(move || {
        for chunk in found.chunks(REMOVED_PER_FILE) {
            let removed: Vec<Change> = chunk
                .iter()
                .map(|(key, current)| Change {
                    key: key.clone(),
                    kind: Kind::Removed,
                    previous_content_sha1: current.content_sha1.clone(),
                    published_at: iso(now),
                    body: Body::Removed { url: current.url.clone(), domain: desearch::canonical::domain_of(&current.url).unwrap_or_default() },
                })
                .collect();
            let seq = feed.append(changes::encode(&removed)?, removed.len())?;
            index.remove(&chunk.iter().map(|(key, current)| (key.clone(), current.version.clone())).collect::<Vec<_>>())?;
            println!("removed {} pages in change file {seq}", removed.len());
        }
        index.flush()
    })
    .await??;
    Ok(count)
}

pub fn now_us() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros() as i64
}

/// The UTC day of a moment, as `YYYY-MM-DD`.
pub fn day(us: i64) -> String {
    iso(us)[..10].to_string()
}

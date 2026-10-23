//! The publisher as a service: claim batches from the task API's queue, read the next ones while one is written, then report and acknowledge.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
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
use crate::queue::{PublishQueue, PublishedVectors};
use crate::reading::{RangeRead, Remote, Traffic};
use crate::records::iso;
use crate::vectors;
use crate::worker::{self, ChangeFeed, Fault, Job, Read, Uploads, CHANGED, EMBED};
use desearch::feeds::{Feed, CHANGES, OUTCOMES};
use desearch::outcomes::Outcome;
use desearch::r2::{self, Bucket, Credentials, PARQUET};

const RETRY_DELAY: Duration = Duration::from_secs(5);
/// Failed passes in a row wait twice as long each time, up to this.
const RETRY_DELAY_MAX: Duration = Duration::from_secs(120);
const SUMMARY_EVERY: Duration = Duration::from_secs(60);
const CHANGE_PART_BYTES: usize = 8 << 20;
const CHANGE_PARTS_AT_ONCE: usize = 8;
/// Tasks taken back over a week ago are forgotten this often.
const FORGET_WITHDRAWN_EVERY: Duration = Duration::from_secs(600);

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
    /// Stop after this many passes in a row found nothing; 0 runs until stopped.
    pub idle_exit: u32,
    pub idle_delay: Duration,
}

impl Settings {
    pub fn from_env() -> Result<Self> {
        let batch = number("PUBLISHER_BATCH", 40)?;
        Ok(Settings {
            redis_url: text("TASK_API_REDIS", "redis://localhost:6379/15"),
            claim_ttl: number("TASK_API_PUBLISH_TTL", 600.0)?,
            batch,
            readers: number("PUBLISHER_READERS", batch)?,
            ranges: number("PUBLISHER_RANGES", 8)?,
            ahead: number("PUBLISHER_AHEAD", 2)?,
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
    buckets_from(|name| std::env::var(name).ok().filter(|v| !v.is_empty()), errors)
}

/// Both buckets on the R2 endpoint, unless `PAGES_S3_ENDPOINT` puts the pages bucket on its own endpoint with its own keys.
pub fn buckets_from(var: impl Fn(&str) -> Option<String>, errors: Arc<AtomicU64>) -> Result<(Bucket, Bucket)> {
    let required = |name: &str| var(name).with_context(|| format!("{name} is not set"));
    let endpoint = required("CF_R2_ENDPOINT")?;
    let credentials = Credentials {
        access_key: required("CF_R2_ACCESS_KEY_ID")?,
        secret_key: required("CF_R2_SECRET_ACCESS_KEY")?,
        region: var("CF_R2_REGION").unwrap_or_else(|| "auto".into()),
    };
    let http = r2::client()?;
    let temp = Bucket::new(
        http.clone(),
        &endpoint,
        &var("CF_R2_BUCKET").unwrap_or_else(|| "subnet-22".into()),
        &var("TASK_API_R2_PREFIX").unwrap_or_default(),
        credentials.clone(),
        errors.clone(),
    )?
    .public_objects(var("CF_R2_PUBLIC_OBJECTS").as_deref() == Some("1"));
    let pages_bucket = var("CF_R2_PAGES_BUCKET").unwrap_or_else(|| "desearch-pages".into());
    let pages_prefix = var("CF_R2_PAGES_PREFIX").unwrap_or_default();
    let pages = match var("PAGES_S3_ENDPOINT") {
        Some(own) => {
            let own_credentials = Credentials {
                access_key: required("PAGES_S3_ACCESS_KEY_ID")?,
                secret_key: required("PAGES_S3_SECRET_ACCESS_KEY")?,
                region: required("PAGES_S3_REGION")?,
            };
            Bucket::new(http, &own, &var("PAGES_S3_BUCKET").unwrap_or(pages_bucket), &pages_prefix, own_credentials, errors)?
        }
        None => {
            if (&temp.bucket, &temp.prefix) == (&pages_bucket, &pages_prefix) {
                bail!("CF_R2_PAGES_BUCKET is {pages_bucket:?}, the same place uploads are kept and expired; point it at the permanent bucket");
            }
            Bucket::new(http, &endpoint, &pages_bucket, &pages_prefix, credentials, errors)?
        }
    };
    Ok((temp, pages))
}

#[derive(Default)]
pub struct Metrics {
    pub tasks: AtomicU64,
    pub batches: AtomicU64,
    pub changed: AtomicU64,
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
    pub settings: Settings,
    pub metrics: Arc<Metrics>,
    pub handle: Handle,
}

impl Shared {
    pub async fn connect(settings: Settings, temp: Bucket, pages: Bucket, metrics: Arc<Metrics>) -> Result<Arc<Self>> {
        let redis = redis::Client::open(settings.redis_url.as_str())?.get_connection_manager().await.context("connecting to TASK_API_REDIS")?;
        Ok(Arc::new(Shared { queue: PublishQueue::new(redis, settings.claim_ttl), temp, pages, settings, metrics, handle: Handle::current() }))
    }

    fn redis(&self) -> &ConnectionManager {
        &self.queue.redis
    }
}

/// Uploads read from the temp bucket in the version their completion named, or as first found when it named none.
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
        let etag = job.etag.clone().filter(|e| !e.is_empty());
        if etag.as_deref().is_some_and(|etag| head.etag.trim_matches('"') != etag.trim_matches('"')) {
            return Err(Fault::Gone(CHANGED));
        }
        Ok(Box::new(Remote {
            bucket: self.temp.clone(),
            key: key.to_string(),
            size: head.size,
            etag: etag.or(Some(head.etag).filter(|e| !e.is_empty())),
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
    embeds: Vec<Job>,
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
    let (embeds, jobs): (Vec<Job>, Vec<Job>) = jobs.into_iter().partition(|job| job.is(EMBED));
    let uploads =
        R2Uploads { temp: shared.temp.clone(), handle: shared.handle.clone(), ranges: shared.settings.ranges, traffic: shared.metrics.traffic.clone() };
    let (reading, readers, now) = (jobs.clone(), shared.settings.readers, now_us());
    let reads = tokio::task::spawn_blocking(move || worker::read_all(&reading, &uploads, now, readers));
    Ok(Some(Claimed { jobs, embeds, reads, started }))
}

/// Writes a batch already read: change file, outcomes and acknowledgements for pages, a vector file for each embed job, a withdrawal entry for vectors taken back; returns the jobs finished.
pub async fn write_batch(shared: &Shared, claimed: Claimed) -> Result<usize> {
    let Claimed { jobs, embeds, reads, started } = claimed;
    let withdrawn = worker::withdrawn_in(&jobs);
    shared.queue.note_withdrawn(&withdrawn, now_us() as f64 / 1e6).await?;
    let embedded = publish_vectors(shared, embeds).await;
    let models = take_back_vectors(shared, &withdrawn).await.context("taking back published vectors")?;
    fill_vector_holes(shared, models).await;
    if jobs.is_empty() {
        return Ok(embedded);
    }
    let reads = reads.await.context("reading the batch")?;
    let read_s = started.elapsed().as_secs_f64();
    let ids: Vec<&str> = jobs.iter().map(|job| job.task_id.as_str()).collect();
    shared.queue.extend_claims(&ids).await?;
    let uploads: HashMap<String, Vec<String>> =
        jobs.iter().map(|job| (job.task_id.clone(), [job.key.clone(), job.input_key.clone()].into_iter().flatten().collect())).collect();
    let now = now_us();
    let refused = shared.queue.withdrawn_among(&ids).await?;
    let feed = R2Changes { pages: shared.pages.clone(), redis: shared.redis().clone(), handle: shared.handle.clone(), day: day(now) };
    let batch = tokio::task::spawn_blocking(move || worker::write(&jobs, reads, &refused, &feed, now)).await??;
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
    let published = batch.changes.len() - batch.withdrawn.len();
    let m = &shared.metrics;
    m.batches.fetch_add(1, Ordering::Relaxed);
    m.tasks.fetch_add(batch.finalized.len() as u64, Ordering::Relaxed);
    m.changed.fetch_add(published as u64, Ordering::Relaxed);
    m.failed_urls.fetch_add((batch.failed.len() + batch.dropped.len()) as u64, Ordering::Relaxed);
    m.lost.fetch_add(batch.lost.len() as u64, Ordering::Relaxed);
    m.retried.fetch_add(batch.retry.len() as u64, Ordering::Relaxed);
    println!(
        "published {} tasks, {} pages, {} tasks withdrawn in {:.1}s (read by {:.1}s, written by {:.1}s, change file stored in {:.1}s); {} URLs failed, {} lost, {} left to retry, change file {}",
        batch.finalized.len(),
        published,
        batch.withdrawn.len(),
        started.elapsed().as_secs_f64(),
        read_s,
        written_s,
        batch.stored_s,
        batch.failed.len(),
        batch.lost.len(),
        batch.retry.len(),
        batch.change_seq.map_or("none".to_string(), |seq| format!("#{seq} ({} bytes)", batch.change_bytes)),
    );
    Ok(batch.finalized.len() + embedded)
}

/// Publishes each embed job's vectors and finishes it; a job taken back first publishes nothing, a job that cannot be read any more is finished as lost with its pages sent back to the planner, any other failure leaves it claimed to be tried again.
async fn publish_vectors(shared: &Shared, jobs: Vec<Job>) -> usize {
    let ids: Vec<&str> = jobs.iter().map(|job| job.task_id.as_str()).collect();
    let refused = match shared.queue.withdrawn_among(&ids).await {
        Ok(refused) => refused,
        Err(error) => {
            eprintln!("{} embed jobs left to retry, their take-backs unknown: {error:#}", jobs.len());
            return 0;
        }
    };
    let mut finished = 0;
    let mut models = BTreeSet::new();
    for job in jobs {
        let task_id = job.task_id.as_str();
        let done = if refused.contains(task_id) { refuse_vectors(shared, &job).await } else { publish_job_vectors(shared, &job).await };
        match done {
            Ok(()) => {}
            Err(Fault::Gone(why)) => {
                eprintln!("task={task_id} embed upload lost: {why}");
                if let Err(error) = shared.queue.drop_pages(job.input_key.as_deref().unwrap_or_default(), &job.pages).await {
                    eprintln!("task={task_id} pages not sent back to the planner; it will be retried: {error:#}");
                    continue;
                }
                if let Err(error) = shared.queue.mark_lost(task_id).await {
                    eprintln!("task={task_id} could not be marked lost: {error:#}");
                }
            }
            Err(Fault::Retry(why)) => {
                eprintln!("task={task_id} vectors not published; they will be retried: {why}");
                continue;
            }
        }
        models.extend(job.model.clone());
        if let Err(error) = shared.queue.ack(task_id).await {
            eprintln!("task={task_id} could not be acknowledged: {error:#}");
            continue;
        }
        finished += 1;
        for key in [job.key.clone(), job.input_key.clone()].into_iter().flatten() {
            let temp = shared.temp.clone();
            tokio::spawn(async move { temp.delete(&key).await });
        }
    }
    fill_vector_holes(shared, models).await;
    finished
}

async fn fill_vector_holes(shared: &Shared, models: BTreeSet<String>) {
    for model in models {
        if let Err(error) = Feed::vectors(&model).fill_holes(&shared.pages, shared.redis()).await {
            eprintln!("filling holes in the {model} vectors feed failed: {error:#}");
        }
    }
}

/// A job taken back before publishing sends its pages back to the planner; vectors of it published meanwhile are withdrawn instead.
async fn refuse_vectors(shared: &Shared, job: &Job) -> Result<(), Fault> {
    let retry = |error: anyhow::Error| Fault::Retry(format!("{error:#}"));
    if !take_back_vectors(shared, std::slice::from_ref(&job.task_id)).await.map_err(retry)?.is_empty() {
        return Ok(());
    }
    shared.queue.drop_pages(job.input_key.as_deref().unwrap_or_default(), &job.pages).await.map_err(retry)?;
    println!("task={} taken back before publishing; {} pages sent back to the planner", job.task_id, job.pages.len());
    Ok(())
}

/// Withdraws the published vectors of these tasks with a `task_withdrawn` row for each in its model's feed, and sends their pages back to the planner; the models written to.
async fn take_back_vectors(shared: &Shared, task_ids: &[String]) -> Result<BTreeSet<String>> {
    let published = shared.queue.published_vectors(task_ids).await?;
    let mut by_model: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (task_id, remembered) in &published {
        by_model.entry(remembered.model.clone()).or_default().push((task_id.clone(), remembered.miner.clone()));
    }
    let now = now_us();
    for (model, tasks) in &by_model {
        let (rows, taken, published_at) = (tasks.len(), tasks.clone(), iso(now));
        let file = tokio::task::spawn_blocking(move || vectors::encode_withdrawn(&taken, &published_at)).await??;
        let key = vectors::file_key(model, &day(now), &uuid::Uuid::new_v4().simple().to_string());
        shared.pages.put(&key, Bytes::from(file), PARQUET, None).await.with_context(|| format!("writing {key}"))?;
        let seq = Feed::vectors(model).number(&shared.pages, shared.redis(), &key, rows).await?;
        println!("withdrew the vectors of {rows} tasks in {model} vectors file #{seq}");
    }
    for (task_id, remembered) in published {
        let pages = match shared.pages.get(&remembered.key).await {
            Ok(body) => tokio::task::spawn_blocking(move || vectors::pages(body)).await?.unwrap_or_else(|error| {
                eprintln!("task={task_id} vector file {} unreadable; its pages are not sent back: {error:#}", remembered.key);
                Vec::new()
            }),
            Err(r2::Error::Missing) => {
                eprintln!("task={task_id} vector file {} is gone; its pages are not sent back", remembered.key);
                Vec::new()
            }
            Err(error) => bail!("reading {}: {error}", remembered.key),
        };
        shared.queue.drop_pages(&remembered.input_key, &pages).await?;
        shared.queue.forget_vectors(&task_id).await?;
    }
    Ok(by_model.into_keys().collect())
}

/// One embed job's vector file, written, numbered in its model's feed and remembered for take-backs; pages with a malformed vector are left out, sent back to the planner and reported.
async fn publish_job_vectors(shared: &Shared, job: &Job) -> Result<(), Fault> {
    let missing = |what: &str| Fault::Retry(format!("an embed job without {what}"));
    let (key, input_key, model) = (
        job.key.as_deref().ok_or_else(|| missing("an upload key"))?,
        job.input_key.as_deref().ok_or_else(|| missing("an input"))?,
        job.model.as_deref().ok_or_else(|| missing("a model"))?,
    );
    let etag = job.etag.as_deref().filter(|e| !e.is_empty());
    let (upload, input) = (read_whole(&shared.temp, key, etag).await?, read_whole(&shared.temp, input_key, None).await?);
    let now = now_us();
    let (task_id, miner) = (job.task_id.as_str(), job.miner.as_deref().unwrap_or_default());
    let (writing, writer, published_at) = (task_id.to_string(), miner.to_string(), iso(now));
    let encoded = tokio::task::spawn_blocking(move || {
        let joined = vectors::join(input, &upload)?;
        let file = if joined.rows.is_empty() { None } else { Some(vectors::encode(&joined.rows, &writing, &writer, &published_at)?) };
        Ok::<_, anyhow::Error>((joined, file))
    })
    .await
    .map_err(|error| Fault::Retry(format!("joining the vectors: {error}")))?;
    let (joined, file) = encoded.map_err(|error| {
        eprintln!("task={task_id} vectors refused: {error:#}");
        Fault::Gone("unreadable")
    })?;
    let retry = |error: anyhow::Error| Fault::Retry(format!("{error:#}"));
    if let Some(file) = file {
        let file_key = vectors::file_key(model, &day(now), &uuid::Uuid::new_v4().simple().to_string());
        shared.pages.put(&file_key, Bytes::from(file), PARQUET, None).await.map_err(|error| Fault::Retry(format!("writing {file_key}: {error}")))?;
        Feed::vectors(model)
            .number(&shared.pages, shared.redis(), &file_key, joined.rows.len())
            .await
            .map_err(|error| Fault::Retry(format!("numbering {file_key}: {error:#}")))?;
        let published = PublishedVectors { model: model.into(), key: file_key.clone(), miner: miner.into(), input_key: input_key.into() };
        shared.queue.remember_vectors(task_id, &published).await.map_err(retry)?;
        println!("task={task_id} published {} vectors to {file_key}", joined.rows.len());
        // A take-back that came while the file was written found nothing to withdraw yet.
        if shared.queue.withdrawn_among(&[task_id]).await.map_err(retry)?.contains(task_id) {
            take_back_vectors(shared, std::slice::from_ref(&job.task_id)).await.map_err(retry)?;
        }
    }
    if joined.malformed > 0 {
        shared.queue.drop_pages(input_key, &joined.dropped).await.map_err(retry)?;
        shared.queue.note_malformed(task_id, miner, joined.malformed).await.map_err(retry)?;
        eprintln!("task={task_id} has {} malformed vectors; {} pages left out and sent back to the planner", joined.malformed, joined.dropped.len());
    }
    Ok(())
}

/// A whole object of the temp bucket, refused as changed unless it is still the version `etag` names.
async fn read_whole(temp: &Bucket, key: &str, etag: Option<&str>) -> Result<Bytes, Fault> {
    match temp.get_matching(key, etag).await {
        Ok(body) => Ok(body),
        Err(r2::Error::Missing) => Err(Fault::Gone("expired")),
        Err(r2::Error::Changed) => Err(Fault::Gone(CHANGED)),
        Err(error) => Err(Fault::Retry(format!("reading {key}: {error}"))),
    }
}

/// What became of every URL, for the bot; a failure here is logged and the batch still finishes.
async fn report_outcomes(shared: &Shared, batch: &worker::Batch, now: i64) {
    if let Err(error) = write_outcomes(shared, outcomes::rows(&batch.changes, &batch.failed, &batch.dropped), now).await {
        eprintln!("could not write the batch's outcomes: {error:#}");
    }
}

/// One outcome file, numbered in the feed.
async fn write_outcomes(shared: &Shared, rows: Vec<Outcome>, now: i64) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let count = rows.len();
    let file = tokio::task::spawn_blocking(move || desearch::outcomes::encode(&rows, now)).await??;
    let key = format!("outcomes/dt={}/{}.parquet", day(now), uuid::Uuid::new_v4().simple());
    shared.temp.put(&key, Bytes::from(file), PARQUET, None).await?;
    OUTCOMES.number(&shared.temp, shared.redis(), &key, count).await?;
    Ok(())
}

/// Publishes until `stop` turns true, then finishes the batches already claimed and claims no more; `idle_exit` passes in a row with nothing to do also stop it.
pub async fn run(shared: Arc<Shared>, mut stop: watch::Receiver<bool>, idle_exit: u32) -> Result<()> {
    let mut idle = 0;
    let mut claimed: Vec<Claimed> = Vec::new();
    let mut forgot: Option<Instant> = None;
    let mut failed_passes = 0u32;
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
            let task_ids: Vec<String> = current.jobs.iter().chain(&current.embeds).map(|job| job.task_id.clone()).collect();
            if let Err(error) = write_batch(&shared, current).await {
                failed_passes += 1;
                let delay = RETRY_DELAY.saturating_mul(1 << (failed_passes - 1).min(5)).min(RETRY_DELAY_MAX);
                eprintln!("publish pass failed ({failed_passes} in a row), its tasks go back to be tried again in {}s: {error:#}", delay.as_secs());
                pause(&mut stop, delay).await;
                if let Err(error) = shared.queue.give_back(&task_ids).await {
                    eprintln!("giving back the failed pass's tasks failed; they return when their claims expire: {error:#}");
                }
            } else {
                failed_passes = 0;
            }
        }
        if forgot.is_none_or(|at| at.elapsed() >= FORGET_WITHDRAWN_EVERY) {
            forgot = Some(Instant::now());
            if let Err(error) = shared.queue.forget_withdrawn(now_us() as f64 / 1e6).await {
                eprintln!("forgetting old take-backs failed: {error:#}");
            }
        }
        if let Err(error) = CHANGES.fill_holes(&shared.pages, shared.redis()).await {
            eprintln!("filling holes in the change feed failed: {error:#}");
        }
        if idle > 0 && !finishing {
            pause(&mut stop, shared.settings.idle_delay).await;
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
        [&m.tasks, &m.changed, &m.failed_urls, &m.lost, &m.retried, m.r2_errors.as_ref(), &m.traffic.bytes, &m.traffic.requests]
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
            "minute: {} tasks/min, {} pages, {} URLs failed, {} lost, {} left to retry; backlog {backlog}; R2 {} errors, read {:.0} MB in {} requests",
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            d[6] as f64 / 1e6,
            d[7]
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
        "publishing {}/{} -> {}/{}, batches of {}, {} uploads read at once",
        shared.temp.bucket, shared.temp.prefix, shared.pages.bucket, shared.pages.prefix, shared.settings.batch, shared.settings.readers
    );
    tokio::spawn(summarize(shared.clone(), stop.clone()));
    run(shared.clone(), stop, shared.settings.idle_exit).await?;
    println!("publisher stopped");
    Ok(())
}

/// A page URL kept from a sitemap address whose XML escapes were never decoded, as in `?amp%3Bid=1`.
static ESCAPED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)[?&]amp%3B|&(amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-f]+);").expect("a valid pattern"));
const REMOVED_PER_FILE: usize = 10_000;

pub fn escaped(url: &str) -> bool {
    ESCAPED.is_match(url)
}

/// `remove_escaped` on the version index at `index`, with the storage and Redis `serve` uses.
pub async fn remove_escaped_from_env(index: &Path, backup: Option<PathBuf>) -> Result<()> {
    let settings = Settings::from_env()?;
    let metrics = Arc::new(Metrics::default());
    let (temp, pages) = buckets_from_env(metrics.r2_errors.clone())?;
    pages.check().await?;
    let shared = Shared::connect(settings, temp, pages, metrics).await?;
    let index = Arc::new(VersionIndex::open(index, 256 << 20)?);
    remove_escaped(&shared, &index, backup.as_deref()).await?;
    Ok(())
}

/// Takes down every page whose URL still carries a sitemap's XML escapes, in change files of `removed` rows, each index entry saved to `backup` first as a JSON line; without a backup it only counts them.
pub async fn remove_escaped(shared: &Shared, index: &Arc<VersionIndex>, backup: Option<&Path>) -> Result<usize> {
    let scanning = index.clone();
    let found: Vec<(String, Current)> = tokio::task::spawn_blocking(move || {
        let mut found = Vec::new();
        scanning.scan(|key, current| {
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
    let Some(backup) = backup.filter(|_| !found.is_empty()) else {
        return Ok(found.len());
    };
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(backup).with_context(|| format!("creating {}", backup.display()))?;
    let mut saved = std::io::BufWriter::new(file);
    let count = found.len();
    let now = now_us();
    let feed = R2Changes { pages: shared.pages.clone(), redis: shared.redis().clone(), handle: shared.handle.clone(), day: day(now) };
    let index = index.clone();
    tokio::task::spawn_blocking(move || {
        for chunk in found.chunks(REMOVED_PER_FILE) {
            for (key, current) in chunk {
                let entry = serde_json::json!({
                    "key": key, "url": current.url, "version": current.version, "fetched_at": current.fetched_at, "task_id": current.task_id,
                    "content_sha1": current.content_sha1, "change_seq": current.change_seq, "change_row": current.change_row,
                });
                writeln!(saved, "{entry}")?;
            }
            saved.flush()?;
            saved.get_ref().sync_data()?;
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

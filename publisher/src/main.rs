//! publisher: publish validated uploads into the change feed, and keep the current state of every page compacted from it.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use desearch::feeds::BULK;
use desearch::r2::{self, Bucket, Credentials};
use desearch::stub;
use publisher::compactor::{self, R2Store, Split, Store};
use publisher::index::VersionIndex;
use publisher::local::{LocalFeed, LocalLog, LocalUploads};
use publisher::worker::{self, Job};
use publisher::{build, bulk, outcomes, planner, service};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
#[command(name = "publisher", about = "Publish validated uploads into the change feed and compact it into the current state")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Publish from the task API's queue in Redis to R2, configured from the environment, until SIGTERM; any number may run at once.
    Serve,
    /// Take down the pages of a version index whose URL kept a sitemap's XML escapes; counts them unless given `--backup`.
    RemoveEscaped {
        /// The version index the publisher kept before it published without one.
        #[arg(long)]
        index: PathBuf,
        /// Remove, after saving every removed index entry to this new file as JSON lines.
        #[arg(long)]
        backup: Option<PathBuf>,
    },
    /// Publish a JSON list of jobs whose uploads are `<task_id>.parquet` files in a folder.
    Local(LocalArgs),
    /// Publish a bulk run's batch files from `python -m bulk` into the bulk feed: a folder laid out as the pages bucket with `--log`, or the pages bucket with `--r2`.
    Bulk(BulkArgs),
    /// Keep the current state under `COMPACTOR_PREFIX` (default `current/`) up to the change log, a cycle at a time, until SIGTERM.
    Compact {
        /// One cycle, then exit.
        #[arg(long)]
        once: bool,
        /// Give shards without a manifest an empty one that follows the log after this change file.
        #[arg(long)]
        start_empty_at: Option<i64>,
    },
    /// Check sampled pages of change files `a..b` read back from the current state as that version or a newer one.
    CheckCurrent {
        #[arg(long, default_value = "current/")]
        prefix: String,
        #[arg(long)]
        seqs: String,
        /// Change files sampled across the range.
        #[arg(long, default_value_t = 10)]
        files: usize,
        #[arg(long, default_value_t = 5)]
        per_file: usize,
        /// Only pages of shards `a..b`, so a check reads a few shards whole rather than every shard.
        #[arg(long, default_value = "0..1024")]
        shards: String,
        /// For each page found wrong, list every version and removal of it in `seqs`.
        #[arg(long)]
        explain: bool,
    },
    /// Compare every page of the current state with a version index the publisher kept, or else with a plain fold of the change log; for test runs.
    CompareCurrent {
        #[arg(long, default_value = "current/")]
        prefix: String,
        #[arg(long)]
        index: Option<PathBuf>,
        /// Only shards `a..b`, so a check of a full-size state fits in memory.
        #[arg(long, default_value = "0..1024")]
        shards: String,
    },
    /// Build the current state once from an index snapshot: a local file, or its key in the pages bucket.
    BuildCurrent {
        #[arg(long)]
        snapshot: String,
        #[arg(long)]
        work: PathBuf,
        #[arg(long, default_value = "current/")]
        prefix: String,
        /// Shards `a..b`, for a trial on part of the data.
        #[arg(long, default_value = "0..1024")]
        shards: String,
        /// Change files `a..b`: only pages whose current row is in them, for a trial.
        #[arg(long)]
        seqs: Option<String>,
        #[arg(long, default_value_t = 16)]
        threads: usize,
        /// Build without the snapshot's pages that name no change file row, instead of refusing.
        #[arg(long)]
        allow_unlocated: bool,
    },
    /// Turn published pages that need a vector into embed inputs for the task API, following the change log until SIGTERM.
    Plan,
    /// Serve a local stand-in for R2 that keeps objects as files under a folder; for local runs only.
    StubR2 {
        #[arg(long)]
        root: PathBuf,
        #[arg(long, default_value = "127.0.0.1:9000")]
        listen: String,
    },
}

#[derive(clap::Args)]
struct BulkArgs {
    /// The run's folder, holding its manifest.json.
    dir: PathBuf,
    #[arg(long, conflicts_with = "r2", required_unless_present = "r2")]
    log: Option<PathBuf>,
    #[arg(long)]
    r2: bool,
    #[arg(long, default_value_t = 10_000)]
    max_rows: usize,
    #[arg(long, default_value_t = 32)]
    max_text_mb: usize,
    /// Pages an hour at most; 0 for no limit.
    #[arg(long, default_value_t = 0.0)]
    pages_per_hour: f64,
}

#[derive(clap::Args)]
struct LocalArgs {
    #[arg(long)]
    jobs: PathBuf,
    #[arg(long)]
    uploads: PathBuf,
    /// Change files go to `<out>/changes`, outcome files to `<out>/outcomes`.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 40)]
    batch: usize,
    /// Uploads read at once.
    #[arg(long, default_value_t = 8)]
    readers: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve => {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().max_blocking_threads(1024).build()?;
            let served = runtime.block_on(service::serve());
            runtime.shutdown_timeout(Duration::from_secs(5));
            served
        }
        Command::RemoveEscaped { index, backup } => {
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(service::remove_escaped_from_env(&index, backup))
        }
        Command::Local(args) => local(args),
        Command::Bulk(args) => bulk(args),
        Command::Compact { once, start_empty_at } => compact(once, start_empty_at),
        Command::Plan => tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(plan()),
        Command::CheckCurrent { prefix, seqs, files, per_file, shards, explain } => {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            let store = current_store(&runtime, &prefix, pages_store(&runtime)?)?;
            let seqs = range::<i64>(&seqs)?;
            let all: Vec<i64> = seqs.clone().collect();
            let step = ((seqs.end - seqs.start) as usize / files.max(1)).max(1);
            let sampled: Vec<i64> = seqs.step_by(step).take(files).collect();
            let (checked, removed, wrong) = compactor::check(&store, &prefix, &sampled, per_file, &range(&shards)?, 32)?;
            for line in &wrong {
                println!("{line}");
                if explain {
                    let key = line.split(' ').next().unwrap_or_default();
                    for event in compactor::history(&store, key, &all, 32)? {
                        println!("    {event}");
                    }
                }
            }
            println!("{checked} pages checked in change files {sampled:?}: {removed} removed by a later change, {} wrong", wrong.len());
            Ok(())
        }
        Command::CompareCurrent { prefix, index, shards } => {
            let shards = range::<usize>(&shards)?;
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            let store = current_store(&runtime, &prefix, pages_store(&runtime)?)?;
            let (compared, against) = match index {
                Some(index) => (compactor::compare(&store, &prefix, &VersionIndex::open_read_only(&index, 64 << 20)?, &shards, 16)?, "index"),
                None => (compactor::compare_log(&store, &prefix, &shards, 16)?, "change log"),
            };
            for (what, keys) in [("missing", &compared.missing), ("extra", &compared.extra), ("other version", &compared.other_version)] {
                for key in keys.iter().take(20) {
                    println!("{what}: {key}");
                }
            }
            println!(
                "{against} {} pages ({} of withdrawn tasks still draining), current {} pages: {} missing, {} extra, {} at another version",
                compared.index_pages,
                compared.draining,
                compared.current_pages,
                compared.missing.len(),
                compared.extra.len(),
                compared.other_version.len()
            );
            Ok(())
        }
        Command::BuildCurrent { snapshot, work, prefix, shards, seqs, threads, allow_unlocated } => {
            build_current(&snapshot, work, prefix, &shards, seqs.as_deref(), threads, allow_unlocated)
        }
        Command::StubR2 { root, listen } => {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            runtime.block_on(async {
                let stub = stub::start(root.clone(), &listen).await?;
                println!("serving {} as R2 on http://{}", root.display(), stub.addr);
                tokio::signal::ctrl_c().await?;
                anyhow::Ok(())
            })
        }
    }
}

fn local(args: LocalArgs) -> Result<()> {
    let jobs: Vec<Job> = serde_json::from_slice(&std::fs::read(&args.jobs).with_context(|| format!("reading {}", args.jobs.display()))?)?;
    let mut refused = HashSet::new();
    let uploads = LocalUploads::new(&args.uploads);
    let feed = LocalFeed::new(args.out.join("changes"), 1)?;
    let outcome_dir = args.out.join("outcomes");
    std::fs::create_dir_all(&outcome_dir)?;
    let mut files = 0;
    let mut write_outcomes = |rows: &[desearch::outcomes::Outcome], now: i64| -> Result<()> {
        files += 1;
        Ok(std::fs::write(outcome_dir.join(format!("{files:012}.parquet")), desearch::outcomes::encode(rows, now)?)?)
    };
    let now_us = || -> Result<i64> { Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros() as i64) };
    for jobs in jobs.chunks(args.batch.max(1)) {
        let started = Instant::now();
        let now = now_us()?;
        let batch = worker::publish(jobs, &uploads, &refused, &feed, now, args.readers)?;
        refused.extend(batch.withdrawn.iter().cloned());
        write_outcomes(&outcomes::rows(&batch.changes, &batch.failed, &batch.dropped), now)?;
        println!(
            "published {} tasks, {} pages, {} tasks withdrawn, {} failed URLs, {} to retry in {:.2}s",
            batch.finalized.len(),
            batch.changes.len() - batch.withdrawn.len(),
            batch.withdrawn.len(),
            batch.failed.len(),
            batch.retry.len(),
            started.elapsed().as_secs_f64()
        );
        for (task_id, why) in &batch.retry {
            eprintln!("task={task_id} will be retried: {why}");
        }
    }
    Ok(())
}

fn bulk(args: BulkArgs) -> Result<()> {
    let settings = bulk::Settings { max_rows: args.max_rows, max_text_bytes: args.max_text_mb << 20, pages_per_hour: args.pages_per_hour };
    let Some(log) = args.log else {
        return tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(service::bulk_from_env(args.dir, settings));
    };
    let report = bulk::publish_run(&args.dir, &LocalLog::open(log, &BULK)?, &settings, &service::now_us)?;
    println!("{} files, {} pages, bulk files {:?}", report.files, report.pages, report.change_files);
    Ok(())
}

fn env_number<T: std::str::FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name).ok().filter(|v| !v.is_empty()) {
        Some(value) => value.trim().parse().map_err(|_| anyhow::anyhow!("{name}={value:?} is not a number")),
        None => Ok(default),
    }
}

fn range<T: std::str::FromStr>(text: &str) -> Result<std::ops::Range<T>> {
    let (start, end) = text.split_once("..").with_context(|| format!("{text:?} is not a range a..b"))?;
    let parse = |part: &str| part.trim().parse::<T>().map_err(|_| anyhow::anyhow!("{text:?} is not a range a..b"));
    Ok(parse(start)?..parse(end)?)
}

fn pages_store(runtime: &tokio::runtime::Runtime) -> Result<R2Store> {
    let (_, pages) = runtime.block_on(async { service::buckets_from_env(Default::default()) })?;
    Ok(R2Store { bucket: pages, handle: runtime.handle().clone() })
}

/// The change log in the pages bucket and the current state under `prefix` in `CURRENT_S3_BUCKET` when that is set, else in the pages bucket too.
fn current_store(runtime: &tokio::runtime::Runtime, prefix: &str, log: R2Store) -> Result<Split> {
    let (current, at) = match std::env::var("CURRENT_S3_ENDPOINT").ok().filter(|e| !e.is_empty()) {
        Some(endpoint) => {
            let credentials = Credentials {
                access_key: std::env::var("CURRENT_S3_ACCESS_KEY_ID").context("CURRENT_S3_ACCESS_KEY_ID is not set")?,
                secret_key: std::env::var("CURRENT_S3_SECRET_ACCESS_KEY").context("CURRENT_S3_SECRET_ACCESS_KEY is not set")?,
                region: std::env::var("CURRENT_S3_REGION").context("CURRENT_S3_REGION is not set")?,
            };
            let bucket = std::env::var("CURRENT_S3_BUCKET").context("CURRENT_S3_BUCKET is not set")?;
            let bucket = Bucket::new(r2::client()?, &endpoint, &bucket, "", credentials, Default::default())?;
            runtime.block_on(bucket.check())?;
            (R2Store { bucket, handle: runtime.handle().clone() }, endpoint)
        }
        None => (pages_store(runtime)?, "the pages bucket".to_string()),
    };
    println!("current state under {}{prefix} at {at}; change log in {}{}", current.bucket.bucket, log.bucket.bucket, log.bucket.prefix);
    Ok(Split { prefix: prefix.to_string(), current: Box::new(current), log: Box::new(log) })
}

/// `CURRENT_S3_BUCKET` when it is set, else the pages bucket, where the backfill reads `current/`.
async fn current_bucket(pages: &Bucket) -> Result<Bucket> {
    let Some(endpoint) = std::env::var("CURRENT_S3_ENDPOINT").ok().filter(|e| !e.is_empty()) else { return Ok(pages.clone()) };
    let credentials = Credentials {
        access_key: std::env::var("CURRENT_S3_ACCESS_KEY_ID").context("CURRENT_S3_ACCESS_KEY_ID is not set")?,
        secret_key: std::env::var("CURRENT_S3_SECRET_ACCESS_KEY").context("CURRENT_S3_SECRET_ACCESS_KEY is not set")?,
        region: std::env::var("CURRENT_S3_REGION").context("CURRENT_S3_REGION is not set")?,
    };
    let bucket = std::env::var("CURRENT_S3_BUCKET").context("CURRENT_S3_BUCKET is not set")?;
    let bucket = Bucket::new(r2::client()?, &endpoint, &bucket, "", credentials, Default::default())?;
    bucket.check().await?;
    Ok(bucket)
}

const COMPACT_RETRY_S: u64 = 30;

async fn plan() -> Result<()> {
    let settings = planner::Settings::from_env()?;
    let (temp, pages) = service::buckets_from_env(Default::default())?;
    let redis_url = std::env::var("TASK_API_REDIS").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "redis://localhost:6379/15".into());
    let redis = redis::Client::open(redis_url.as_str())?.get_connection_manager().await.context("connecting to TASK_API_REDIS")?;
    let task_api = std::env::var("TASK_API_URL").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "https://api-22.desearch.ai".into());
    let current = if settings.backfill { Some(current_bucket(&pages).await?) } else { None };
    let mut planner = planner::Planner::new(settings, pages, temp, redis, planner::Room::api(&task_api)?).await?;
    planner.current = current;
    println!(
        "planning {} texts after change file {}, at most {} pages an hour, inputs of {} pages, room from {task_api}",
        planner::PLANNED_RECIPES.join(", "),
        planner.read_through(),
        planner.settings.pages_per_hour,
        planner.settings.pages_per_input
    );
    let (stopping, stop) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        if let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            term.recv().await;
        }
        let _ = stopping.send(true);
    });
    planner.run(stop).await
}

fn compact(once: bool, start_empty_at: Option<i64>) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let defaults = compactor::Settings::default();
    let settings = compactor::Settings {
        prefix: std::env::var("COMPACTOR_PREFIX").ok().filter(|p| !p.is_empty()).unwrap_or(defaults.prefix),
        work: std::env::var("COMPACTOR_WORK").map(PathBuf::from).unwrap_or(defaults.work),
        files_per_cycle: env_number("COMPACTOR_FILES_PER_CYCLE", defaults.files_per_cycle)?,
        bulk_files_per_cycle: env_number("COMPACTOR_BULK_FILES_PER_CYCLE", defaults.bulk_files_per_cycle)?,
        merge_ratio: env_number("COMPACTOR_MERGE_RATIO", defaults.merge_ratio)?,
        every_s: env_number("COMPACTOR_EVERY_S", defaults.every_s)?,
        fanout: env_number("COMPACTOR_FANOUT", defaults.fanout)?,
        merges_per_cycle: env_number("COMPACTOR_MERGES", defaults.merges_per_cycle)?,
        combines_per_cycle: env_number("COMPACTOR_COMBINES", defaults.combines_per_cycle)?,
        lag_warn_s: env_number("COMPACTOR_LAG_WARN_S", defaults.lag_warn_s)?,
        retire_after_s: env_number("COMPACTOR_RETIRE_S", defaults.retire_after_s)?,
        spill_bytes: env_number::<usize>("COMPACTOR_SPILL_MB", defaults.spill_bytes >> 20)? << 20,
        threads: env_number("COMPACTOR_THREADS", defaults.threads)?,
        merge_threads: env_number("COMPACTOR_MERGE_THREADS", defaults.merge_threads)?,
        hole_wait_s: env_number("COMPACTOR_HOLE_WAIT_S", defaults.hole_wait_s)?,
    };
    let store = current_store(&runtime, &settings.prefix, pages_store(&runtime)?)?;
    if let Some(seq) = start_empty_at {
        compactor::start_empty(&store, &settings.prefix, seq, settings.threads.max(16))?;
    }
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopping = stop.clone();
    runtime.spawn(async move {
        if let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            term.recv().await;
        }
        stopping.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let stopped = || stop.load(std::sync::atomic::Ordering::SeqCst);
    let pause = |seconds: u64| {
        for _ in 0..seconds {
            if stopped() {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    };
    loop {
        let started = Instant::now();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let latest = |report| -> Result<_> { Ok((report, compactor::latest_seq(&store)?, compactor::latest_bulk(&store)?)) };
        let (report, latest, bulk_latest) = match compactor::cycle(&store, &settings, now).and_then(latest) {
            Ok(done) => done,
            Err(error) if !once => {
                eprintln!("cycle failed, trying again in {COMPACT_RETRY_S} s: {error:#}");
                pause(COMPACT_RETRY_S);
                if stopped() {
                    return Ok(());
                }
                continue;
            }
            Err(error) => return Err(error),
        };
        println!(
            "cycle: change files {}..{} of {latest}, {} rows into {} deltas, {} withdrawn, {} bases rewritten, {} deltas combined, {} files deleted in {:.1}s{}{}{}{}{}",
            report.from,
            report.through,
            report.rows,
            report.deltas,
            report.withdrawn,
            report.merged,
            report.combined,
            report.deleted,
            started.elapsed().as_secs_f64(),
            if report.skipped.is_empty() { String::new() } else { format!(", numbers never written skipped: {:?}", report.skipped) },
            if report.waiting { format!(", waiting for change file {}", report.through + 1) } else { String::new() },
            if report.failed.is_empty() { String::new() } else { format!(", {} rewrites failed", report.failed.len()) },
            if report.delete_failed == 0 { String::new() } else { format!(", {} deletes failed and kept", report.delete_failed) },
            match &report.lag {
                compactor::Lag::CaughtUp => String::new(),
                compactor::Lag::Behind(lag) => format!(", oldest unread change file {:.1} h old", lag / 3600.0),
                compactor::Lag::Unknown(_) => ", lag unknown".into(),
            }
        );
        match &report.lag {
            compactor::Lag::Behind(lag) if *lag > settings.lag_warn_s => eprintln!(
                "warning: the compactor is behind: change file {}, the oldest not read, was written {:.1} h ago (limit {:.0} h); the change log expires its files after a set number of days",
                report.through + 1,
                lag / 3600.0,
                settings.lag_warn_s / 3600.0
            ),
            compactor::Lag::Unknown(why) => eprintln!("warning: how far the compactor is behind is unknown: {why}"),
            _ => {}
        }
        if bulk_latest > 0 {
            println!(
                "cycle: bulk files {}..{} of {bulk_latest}, {} rows{}",
                report.bulk_from,
                report.bulk_through,
                report.bulk_rows,
                if report.bulk_skipped.is_empty() { String::new() } else { format!(", numbers never written skipped: {:?}", report.bulk_skipped) }
            );
        }
        if once || stopped() {
            return Ok(());
        }
        if (report.through >= latest || report.waiting) && (report.bulk_through >= bulk_latest || report.bulk_waiting) {
            pause((settings.every_s - started.elapsed().as_secs_f64()).max(COMPACT_RETRY_S as f64) as u64);
            if stopped() {
                return Ok(());
            }
        }
    }
}

/// Copies a large object to a local file a part at a time, so it never sits whole in memory.
fn download(store: &R2Store, key: &str, path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    let head = store.handle.block_on(store.bucket.head(key)).map_err(|e| anyhow::anyhow!("{key}: {e}"))?.with_context(|| format!("{key} does not exist"))?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let part = 64u64 << 20;
    let mut at = 0;
    while at < head.size {
        let end = (at + part).min(head.size);
        let bytes = store.handle.block_on(store.bucket.get_range(key, at..end, Some(&head.etag))).map_err(|e| anyhow::anyhow!("{key}: {e}"))?;
        file.write_all(&bytes)?;
        at = end;
    }
    file.flush()?;
    Ok(())
}

fn build_current(snapshot: &str, work: PathBuf, prefix: String, shards: &str, seqs: Option<&str>, threads: usize, allow_unlocated: bool) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let log = pages_store(&runtime)?;
    std::fs::create_dir_all(&work)?;
    let local = if std::path::Path::new(snapshot).exists() {
        PathBuf::from(snapshot)
    } else {
        let local = work.join("snapshot.parquet");
        if !local.exists() {
            let started = Instant::now();
            download(&log, snapshot, &local)?;
            println!("{snapshot} downloaded in {:.0}s", started.elapsed().as_secs_f64());
        }
        local
    };
    let store = current_store(&runtime, &prefix, log)?;
    let settings = build::Settings {
        prefix,
        work,
        threads,
        shards: range(shards)?,
        seqs: seqs.map(range).transpose()?.unwrap_or(0..i64::MAX),
        spill_bytes: 2 << 30,
        allow_unlocated,
    };
    let started = Instant::now();
    let report = build::build(&store as &dyn Store, &local, &settings)?;
    println!("{report:?} in {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}

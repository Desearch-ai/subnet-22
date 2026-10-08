//! publisher: publish validated uploads into the change feed and the version index.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use desearch::stub;
use publisher::compactor::{self, R2Store, Store};
use publisher::index::VersionIndex;
use publisher::local::{LocalFeed, LocalUploads};
use publisher::worker::{self, Job};
use publisher::{build, outcomes, service, withdrawals};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
#[command(name = "publisher", about = "Publish validated uploads into the change feed and the version index")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Publish from the task API's queue in Redis to R2, configured from the environment, until SIGTERM; SIGUSR1 uploads a copy of the index to `index/snapshots/`.
    Serve,
    /// Take down pages whose URL kept a sitemap's XML escapes; counts them unless given `--backup`. Run it with `serve` stopped.
    RemoveEscaped {
        /// Remove, after saving every removed index entry to this new file as JSON lines.
        #[arg(long)]
        backup: Option<PathBuf>,
    },
    /// Publish a JSON list of jobs whose uploads are `<task_id>.parquet` files in a folder.
    Local(LocalArgs),
    /// Keep the current state under `COMPACTOR_PREFIX` (default `current/`) up to the change log, a cycle at a time, until SIGTERM.
    Compact {
        /// One cycle, then exit.
        #[arg(long)]
        once: bool,
        /// Give shards without a manifest an empty one that follows the log after this change file.
        #[arg(long)]
        start_empty_at: Option<i64>,
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
    },
    /// Serve a local stand-in for R2 that keeps objects as files under a folder; for local runs only.
    StubR2 {
        #[arg(long)]
        root: PathBuf,
        #[arg(long, default_value = "127.0.0.1:9000")]
        listen: String,
    },
}

#[derive(clap::Args)]
struct LocalArgs {
    #[arg(long)]
    jobs: PathBuf,
    #[arg(long)]
    uploads: PathBuf,
    #[arg(long)]
    index: PathBuf,
    /// Change files go to `<out>/changes`, outcome files to `<out>/outcomes`.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 40)]
    batch: usize,
    /// Uploads read at once.
    #[arg(long, default_value_t = 8)]
    readers: usize,
    #[arg(long, default_value_t = 1024)]
    cache_mb: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve => {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().max_blocking_threads(1024).build()?;
            let served = runtime.block_on(service::serve());
            // An unfinished snapshot upload is abandoned rather than holding up the exit.
            runtime.shutdown_timeout(Duration::from_secs(5));
            served
        }
        Command::RemoveEscaped { backup } => {
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(service::remove_escaped_from_env(backup))
        }
        Command::Local(args) => local(args),
        Command::Compact { once, start_empty_at } => compact(once, start_empty_at),
        Command::BuildCurrent { snapshot, work, prefix, shards, seqs, threads } => build_current(&snapshot, work, prefix, &shards, seqs.as_deref(), threads),
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
    let index = VersionIndex::open(&args.index, args.cache_mb << 20)?;
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
        let batch = worker::publish(jobs, &uploads, &index, &feed, now, args.readers)?;
        write_outcomes(&outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed), now)?;
        println!(
            "published {} tasks, {} pages new or changed, {} tasks withdrawn, {} unchanged, {} failed URLs, {} to retry in {:.2}s",
            batch.finalized.len(),
            batch.changes.len() - batch.withdrawn.len(),
            batch.withdrawn.len(),
            batch.unchanged.len(),
            batch.failed.len(),
            batch.retry.len(),
            started.elapsed().as_secs_f64()
        );
        for (task_id, why) in &batch.retry {
            eprintln!("task={task_id} will be retried: {why}");
        }
    }
    let now = now_us()?;
    let mut report = |pages: &[publisher::index::Withdrawn]| write_outcomes(&outcomes::dropped(pages), now);
    let taken = withdrawals::drain(&index, &mut HashSet::new(), now as f64 / 1e6, withdrawals::CHUNK, &|| false, &mut report)?;
    if taken > 0 {
        println!("took {taken} pages of withdrawn tasks out of the index");
    }
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

fn compact(once: bool, start_empty_at: Option<i64>) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let store = pages_store(&runtime)?;
    let defaults = compactor::Settings::default();
    let settings = compactor::Settings {
        prefix: std::env::var("COMPACTOR_PREFIX").ok().filter(|p| !p.is_empty()).unwrap_or(defaults.prefix),
        work: std::env::var("COMPACTOR_WORK").map(PathBuf::from).unwrap_or(defaults.work),
        files_per_cycle: env_number("COMPACTOR_FILES_PER_CYCLE", defaults.files_per_cycle)?,
        merge_ratio: env_number("COMPACTOR_MERGE_RATIO", defaults.merge_ratio)?,
        max_deltas: env_number("COMPACTOR_MAX_DELTAS", defaults.max_deltas)?,
        merges_per_cycle: env_number("COMPACTOR_MERGES", defaults.merges_per_cycle)?,
        retire_after_s: env_number("COMPACTOR_RETIRE_S", defaults.retire_after_s)?,
        spill_bytes: env_number::<usize>("COMPACTOR_SPILL_MB", defaults.spill_bytes >> 20)? << 20,
        threads: env_number("COMPACTOR_THREADS", defaults.threads)?,
    };
    if let Some(seq) = start_empty_at {
        compactor::start_empty(&store, &settings.prefix, seq)?;
    }
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopping = stop.clone();
    runtime.spawn(async move {
        if let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            term.recv().await;
        }
        stopping.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    loop {
        let started = Instant::now();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let report = compactor::cycle(&store, &settings, now)?;
        let latest = compactor::latest_seq(&store)?;
        println!(
            "cycle: change files {}..{} of {latest}, {} rows into {} deltas, {} withdrawn, {} shards merged, {} files deleted in {:.1}s",
            report.from,
            report.through,
            report.rows,
            report.deltas,
            report.withdrawn,
            report.merged,
            report.deleted,
            started.elapsed().as_secs_f64()
        );
        if once || stop.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }
        if report.through >= latest {
            for _ in 0..60 {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(1));
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

fn build_current(snapshot: &str, work: PathBuf, prefix: String, shards: &str, seqs: Option<&str>, threads: usize) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let store = pages_store(&runtime)?;
    std::fs::create_dir_all(&work)?;
    let local = if std::path::Path::new(snapshot).exists() {
        PathBuf::from(snapshot)
    } else {
        let local = work.join("snapshot.parquet");
        if !local.exists() {
            let started = Instant::now();
            download(&store, snapshot, &local)?;
            println!("{snapshot} downloaded in {:.0}s", started.elapsed().as_secs_f64());
        }
        local
    };
    let settings =
        build::Settings { prefix, work, threads, shards: range(shards)?, seqs: seqs.map(range).transpose()?.unwrap_or(0..i64::MAX), spill_bytes: 2 << 30 };
    let started = Instant::now();
    let report = build::build(&store as &dyn Store, &local, &settings)?;
    println!("{report:?} in {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}

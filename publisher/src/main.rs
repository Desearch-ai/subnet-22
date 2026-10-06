//! publisher: publish validated uploads into the change feed and the version index.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use publisher::index::VersionIndex;
use publisher::local::{LocalFeed, LocalUploads};
use publisher::outcomes;
use publisher::worker::{self, Job};

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
    /// Publish a JSON list of jobs whose uploads are `<task_id>.parquet` files in a folder.
    Local(LocalArgs),
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
    let Command::Local(args) = Cli::parse().command;
    let jobs: Vec<Job> = serde_json::from_slice(&std::fs::read(&args.jobs).with_context(|| format!("reading {}", args.jobs.display()))?)?;
    let index = VersionIndex::open(&args.index, args.cache_mb << 20)?;
    let uploads = LocalUploads::new(&args.uploads);
    let feed = LocalFeed::new(args.out.join("changes"), 1)?;
    let outcome_dir = args.out.join("outcomes");
    std::fs::create_dir_all(&outcome_dir)?;
    for (n, jobs) in jobs.chunks(args.batch.max(1)).enumerate() {
        let started = Instant::now();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros() as i64;
        let batch = worker::publish(jobs, &uploads, &index, &feed, now, args.readers)?;
        let rows = outcomes::rows(&batch.changes, &batch.unchanged, &batch.failed, &batch.removed);
        std::fs::write(outcome_dir.join(format!("{:012}.parquet", n + 1)), outcomes::encode(&rows, now)?)?;
        println!(
            "published {} tasks, {} pages new or changed or removed, {} unchanged, {} failed URLs, {} to retry in {:.2}s",
            batch.finalized.len(),
            batch.changes.len(),
            batch.unchanged.len(),
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

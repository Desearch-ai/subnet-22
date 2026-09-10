//! desearch-bot: the crawl loop.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use tokio::sync::{watch, Semaphore};

use desearch_bot::buckets::{Buckets, Resources, BUCKETS};
use desearch_bot::crawl::Loop;
use desearch_bot::net::{self, PublicResolver};
use desearch_bot::registry::{self, Registry};
use desearch_bot::signing::Signer;
use desearch_bot::suffixes::PublicSuffixList;
use desearch_bot::visit::{Visitor, MIN_HOST_INTERVAL};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
#[command(name = "desearch-bot", about = "The DesearchBot crawl loop")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Crawl the buckets this process owns, around the clock.
    Run(RunArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    /// Visits in flight at once.
    #[arg(long, default_value_t = 840)]
    concurrency: usize,
    #[arg(long, default_value = "/var/lib/desearch-bot/buckets")]
    buckets_dir: PathBuf,
    /// Buckets to crawl, as ranges such as 0-13,20.
    #[arg(long, default_value = "0-255")]
    buckets: String,
    /// Where public_suffix_list.dat lives.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long, default_value_t = 4096)]
    cache_mb: usize,
    #[arg(long, default_value_t = 2048)]
    memtable_mb: usize,
    /// Seconds a read may stall before the request fails.
    #[arg(long, default_value_t = 10.0)]
    timeout: f64,
    /// Crawl without reporting to Postgres or taking changes from it.
    #[arg(long)]
    no_registry: bool,
    /// Stop after this many seconds.
    #[arg(long)]
    duration: Option<u64>,
}

fn main() -> Result<()> {
    let Command::Run(args) = Cli::parse().command;
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run(args))
}

async fn run(args: RunArgs) -> Result<()> {
    let owned = parse_buckets(&args.buckets)?;
    let dsn = std::env::var("DESEARCH_DB").ok().filter(|d| !d.is_empty());
    let excluded = match &dsn {
        Some(dsn) => registry::excluded_categories(dsn).await?,
        None if args.no_registry => HashSet::new(),
        None => bail!("DESEARCH_DB is not set"),
    };
    let resources = Resources::new(args.cache_mb << 20, args.memtable_mb << 20, owned.len());
    let buckets = Arc::new(Buckets::open(&args.buckets_dir, &owned, &resources)?);
    let registry = match dsn {
        Some(dsn) if !args.no_registry => Registry::new(dsn, &buckets)?,
        _ => Registry::offline(),
    };
    let suffixes_path = args.data_dir.join("public_suffix_list.dat");
    let suffixes = Arc::new(PublicSuffixList::load(&suffixes_path).with_context(|| format!("reading {}", suffixes_path.display()))?);
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let read_timeout = Duration::from_secs_f64(args.timeout);
    let visitor = Arc::new(Visitor {
        client: net::client(PublicResolver::local(), read_timeout)?,
        buckets: buckets.clone(),
        suffixes,
        signer: Signer::from_env()?.map(Arc::new),
        floor: MIN_HOST_INTERVAL,
        connect_timeout: net::connect_timeout(read_timeout),
        cpu: Arc::new(Semaphore::new(cores * 2)),
    });
    let mut crawl = Loop::new(buckets, visitor, args.concurrency, registry, excluded);
    let scheduled = crawl.load()?;
    println!("[rs] {scheduled} domains in {} buckets", owned.len());

    let (stop, stopped) = watch::channel(false);
    tokio::spawn(async move {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        let limit = async {
            match args.duration {
                Some(seconds) => tokio::time::sleep(Duration::from_secs(seconds)).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
            _ = limit => {}
        }
        let _ = stop.send(true);
    });
    crawl.run(stopped).await?;
    println!("[rs] done {}", crawl.summary());
    Ok(())
}

fn parse_buckets(spec: &str) -> Result<Vec<usize>> {
    let mut owned = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        let (first, last): (usize, usize) = (first.parse()?, last.parse()?);
        if first > last || last >= BUCKETS {
            bail!("bucket range {part} is outside 0-{}", BUCKETS - 1);
        }
        owned.extend(first..=last);
    }
    owned.sort_unstable();
    owned.dedup();
    Ok(owned)
}

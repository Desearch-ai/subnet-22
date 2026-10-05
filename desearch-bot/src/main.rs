//! desearch-bot: the crawl loop.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use tokio::sync::watch;

use desearch_bot::allowed::Allowed;
use desearch_bot::buckets::{Buckets, Resources, BUCKETS};
use desearch_bot::crawl::Loop;
use desearch_bot::dispatch::{self, Dispatcher, Progress};
use desearch_bot::hotkey::Hotkey;
use desearch_bot::langid::LangId;
use desearch_bot::net::{self, PublicResolver};
use desearch_bot::outcomes::OutcomeFeed;
use desearch_bot::ready::DEFAULT_RECRAWL;
use desearch_bot::registry::{self, Registry};
use desearch_bot::signing::Signer;
use desearch_bot::suffixes::PublicSuffixList;
use desearch_bot::taskapi::TaskApi;
use desearch_bot::visit::{Slots, Visitor, MIN_HOST_INTERVAL};

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
    /// Where public_suffix_list.dat and langid.bin live.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long, default_value_t = 4096)]
    cache_mb: usize,
    #[arg(long, default_value_t = 2048)]
    memtable_mb: usize,
    /// Sitemap files fetched or waiting to be parsed at once, which bounds the memory their bodies take.
    #[arg(long, default_value_t = 64)]
    sitemap_slots: usize,
    /// Visits at once to domains with thousands of sitemap records, which bounds the memory those records take.
    #[arg(long, default_value_t = 16)]
    heavy_slots: usize,
    /// Percent of the visit slots kept for domains never checked; slots one kind of work leaves unused go to the others.
    #[arg(long, default_value_t = 30)]
    discovery_share: usize,
    /// Percent of the visit slots kept for reading sitemaps found but not read yet.
    #[arg(long, default_value_t = 20)]
    backlog_share: usize,
    /// Seconds a read may stall before the request fails.
    #[arg(long, default_value_t = 10.0)]
    timeout: f64,
    /// Crawl without reporting to Postgres or taking changes from it.
    #[arg(long)]
    no_registry: bool,
    /// New visits wait while the disk has less than this many GB free.
    #[arg(long, default_value_t = 30)]
    min_free_gb: u64,
    /// Rewrite every store once in the background, to reclaim space in files written by older settings.
    #[arg(long)]
    compact: bool,
    /// Stop after this many seconds.
    #[arg(long)]
    duration: Option<u64>,
    /// The task API whose queue the ready lists fill, such as https://api-22.desearch.ai; requests are signed with the key in FEEDER_KEY_URI.
    #[arg(long)]
    task_api: Option<String>,
    /// URLs sent in an hour at most, at a steady pace; 0 sends whatever the task API has room for.
    #[arg(long, default_value_t = 0)]
    urls_per_hour: u64,
    /// URLs one domain may send in an hour.
    #[arg(long, default_value_t = 2000)]
    per_domain_hourly: u64,
    /// Percent of each domain's sends kept for pages never sent, newest found first; what one lane leaves unused goes to the others.
    #[arg(long, default_value_t = 50)]
    new_share: usize,
    /// Percent kept for retries and scheduled re-crawls; pages whose lastmod moved get the rest.
    #[arg(long, default_value_t = 10)]
    retry_share: usize,
    /// Where the task API publishes what became of each URL, such as https://r2.desearch.ai.
    #[arg(long)]
    outcomes_url: Option<String>,
    /// Days after a crawl that a page without a lastmod goes out again.
    #[arg(long, default_value_t = DEFAULT_RECRAWL / 86_400)]
    recrawl_days: u32,
    /// Walk every store once to put pages never sent, or changed since sent, on the ready lists; it resumes where it stopped.
    #[arg(long)]
    backfill_ready: bool,
    /// URL records the backfill reads per second.
    #[arg(long, default_value_t = 50_000)]
    backfill_per_sec: u64,
    /// Pages the Python feeder sent, as tab-separated host, path, lastmod and Unix time, marked sent before the backfill.
    #[arg(long)]
    backfill_sent: Option<PathBuf>,
    /// Only these domains' pages are queued for the task API: a JSON list of names or of {host, rank}, read again when it changes.
    #[arg(long)]
    domains: Option<PathBuf>,
}

fn main() -> Result<()> {
    let Command::Run(args) = Cli::parse().command;
    // Parsing is capped by its own slots; more blocking threads would only hold more allocator caches.
    tokio::runtime::Builder::new_multi_thread().enable_all().max_blocking_threads(128).build()?.block_on(run(args))
}

async fn run(args: RunArgs) -> Result<()> {
    let owned = parse_buckets(&args.buckets)?;
    if args.discovery_share + args.backlog_share > 100 {
        bail!("--discovery-share and --backlog-share add up to more than 100");
    }
    if args.new_share + args.retry_share > 100 {
        bail!("--new-share and --retry-share add up to more than 100");
    }
    let dsn = std::env::var("DESEARCH_DB").ok().filter(|d| !d.is_empty());
    let excluded = match &dsn {
        Some(dsn) => registry::excluded_categories(dsn).await?,
        None if args.no_registry => HashSet::new(),
        None => bail!("DESEARCH_DB is not set"),
    };
    let resources = Resources::new(args.cache_mb << 20, args.memtable_mb << 20, owned.len());
    let buckets = Arc::new(Buckets::open(&args.buckets_dir, &owned, &resources)?);
    if let Some(path) = &args.domains {
        let allowed = Allowed::from_file(path)?;
        buckets.allow(&allowed);
        println!("[rs] queueing pages of {} listed domains only", allowed.count().unwrap_or(0));
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            match allowed.reload() {
                Ok(true) => println!("[rs] domain list changed: {} domains", allowed.count().unwrap_or(0)),
                Ok(false) => {}
                Err(error) => eprintln!("[rs] reading the domain list failed, keeping the last one: {error:#}"),
            }
        });
    }
    let registry = match dsn {
        Some(dsn) if !args.no_registry => Registry::new(dsn, &buckets)?,
        _ => Registry::offline(),
    };
    let suffixes_path = args.data_dir.join("public_suffix_list.dat");
    let suffixes = Arc::new(PublicSuffixList::load(&suffixes_path).with_context(|| format!("reading {}", suffixes_path.display()))?);
    let model = LangId::load(&args.data_dir.join("langid.bin"))?;
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let read_timeout = Duration::from_secs_f64(args.timeout);
    let visitor = Arc::new(Visitor {
        client: net::client(PublicResolver::local(), read_timeout)?,
        buckets: buckets.clone(),
        suffixes,
        signer: Signer::from_env()?.map(Arc::new),
        language: Arc::new(move |text: &str| text.chars().nth(19).map(|_| model.classify(text).to_string())),
        floor: MIN_HOST_INTERVAL,
        connect_timeout: net::connect_timeout(read_timeout),
        cpu: Slots::new(cores * 2),
        bodies: Slots::new(args.sitemap_slots),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(args.heavy_slots),
    });
    if args.compact {
        let stores = buckets.clone();
        std::thread::spawn(move || {
            let all: Vec<_> = stores.stores().collect();
            for (done, store) in all.iter().enumerate() {
                store.compact();
                if (done + 1) % 32 == 0 || done + 1 == all.len() {
                    println!("[rs] compacted {} of {} stores", done + 1, all.len());
                }
            }
        });
    }
    let mut crawl = Loop::new(buckets.clone(), visitor, args.concurrency, registry, excluded)
        .with_min_free_disk(args.min_free_gb << 30)
        .with_shares(args.discovery_share, args.backlog_share);
    let scheduled = crawl.load()?;
    println!("[rs] {scheduled} domains in {} buckets", owned.len());

    let (stop, stopped) = watch::channel(false);
    let progress = Arc::new(Progress::default());
    let halt = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let backfill = args.backfill_ready.then(|| {
        let (buckets, progress, halt, sent) = (buckets.clone(), progress.clone(), halt.clone(), args.backfill_sent.clone());
        let per_second = args.backfill_per_sec;
        std::thread::spawn(move || {
            if let Some(path) = sent {
                match dispatch::import_sent(&buckets, &path) {
                    Ok(known) => println!("[rs] marked {known} pages the earlier feeder sent"),
                    Err(error) => eprintln!("[rs] importing sent pages failed: {error:#}"),
                }
            }
            match dispatch::backfill(&buckets, per_second, &progress, &halt) {
                Ok(()) => println!("[rs] ready-list backfill stopped or done"),
                Err(error) => eprintln!("[rs] ready-list backfill failed: {error:#}"),
            }
        })
    });
    if let Some(base) = &args.task_api {
        let uri = std::env::var("FEEDER_KEY_URI").ok().filter(|u| !u.is_empty()).context("FEEDER_KEY_URI is not set")?;
        let api = TaskApi::new(base, Hotkey::from_uri(&uri).context("FEEDER_KEY_URI")?)?;
        let dispatcher = Dispatcher::load(
            buckets.clone(),
            api,
            args.per_domain_hourly,
            dispatch::shares(args.new_share, args.retry_share),
            progress.clone(),
        ).await?
        .paced(args.urls_per_hour);
        tokio::spawn(dispatcher.run(stopped.clone()));
    }
    if let Some(base) = &args.outcomes_url {
        let feed = OutcomeFeed::new(base)?;
        tokio::spawn(dispatch::follow_outcomes(buckets.clone(), feed, args.recrawl_days * 86_400, progress.clone(), stopped.clone()));
    }
    if args.task_api.is_some() || args.outcomes_url.is_some() || args.backfill_ready {
        let (progress, mut stopped) = (progress.clone(), stopped.clone());
        tokio::spawn(async move {
            let mut last = (std::time::Instant::now(), 0u64);
            loop {
                tokio::select! {
                    _ = stopped.changed() => return,
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {}
                }
                let sent = progress.dispatched.load(std::sync::atomic::Ordering::Relaxed);
                let per_minute = (sent - last.1) as f64 * 60.0 / last.0.elapsed().as_secs_f64().max(1.0);
                last = (std::time::Instant::now(), sent);
                println!("{}", progress.line(per_minute));
            }
        });
    }
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
    halt.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(backfill) = backfill {
        let _ = tokio::task::spawn_blocking(move || backfill.join()).await;
    }
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

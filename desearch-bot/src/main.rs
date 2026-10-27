//! desearch-bot: the crawl loop.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use tokio::sync::watch;

use desearch_bot::allowed::Allowed;
use desearch_bot::buckets::{bucket_of, BucketStore, Buckets, Escaped, Resources, BUCKETS};
use desearch_bot::crawl::Loop;
use desearch_bot::dispatch::{self, Dispatcher, Progress};
use desearch_bot::hotkey::Hotkey;
use desearch_bot::inbox;
use desearch_bot::langid::LangId;
use desearch_bot::net::{self, PublicResolver};
use desearch_bot::outcomes::{OutcomeFeed, OutcomeReader};
use desearch_bot::recrawl::{Recrawl, DAY, DEFAULT_EVERY, DEFAULT_GONE};
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
    Run(Box<RunArgs>),
    /// Delete pages and sitemaps kept with a sitemap's XML escapes; counts them unless given `--backup-dir`. Run it with the crawl loop stopped.
    RemoveEscaped(RemoveEscapedArgs),
    /// Put back what `remove-escaped` deleted, from its backup folder. Run it with the crawl loop stopped.
    RestoreEscaped(RestoreEscapedArgs),
    /// Delete every page kept for these domains and mark them excluded; counts them unless given `--delete`. Run it with the crawl loop stopped.
    DropDomains(DropDomainsArgs),
}

#[derive(clap::Args)]
struct DropDomainsArgs {
    #[arg(long, default_value = "/var/lib/desearch-bot/buckets")]
    buckets_dir: PathBuf,
    /// Comma-separated domains, as the store keys them.
    #[arg(long)]
    domains: String,
    #[arg(long, default_value = "junk")]
    reason: String,
    #[arg(long)]
    delete: bool,
}

#[derive(clap::Args)]
struct RemoveEscapedArgs {
    #[arg(long, default_value = "/var/lib/desearch-bot/buckets")]
    buckets_dir: PathBuf,
    #[arg(long, default_value = "0-255")]
    buckets: String,
    /// Delete, after saving every deleted entry here, one file per store; without it nothing is deleted.
    #[arg(long)]
    backup_dir: Option<PathBuf>,
}

#[derive(clap::Args)]
struct RestoreEscapedArgs {
    #[arg(long, default_value = "/var/lib/desearch-bot/buckets")]
    buckets_dir: PathBuf,
    #[arg(long, default_value = "0-255")]
    buckets: String,
    #[arg(long)]
    backup_dir: PathBuf,
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
    /// The public URL of the S3 bucket where the task API publishes what became of each URL (its `outcomes/` feed).
    #[arg(long)]
    outcomes_url: Option<String>,
    /// Days after a send that a page without a lastmod, or one that failed, goes out again.
    #[arg(long, default_value_t = DEFAULT_EVERY / DAY)]
    recrawl_days: u32,
    /// Re-crawl intervals for listed domains and their subdomains, as `domain days` lines such as `bbc.co.uk 1`.
    #[arg(long)]
    recrawl_overrides: Option<PathBuf>,
    /// Days a page that failed on two sends in a row waits before it goes out again, unless its lastmod moves.
    #[arg(long, default_value_t = DEFAULT_GONE / DAY)]
    gone_days: u32,
    /// Pages of lost or taken-back tasks put back each hour; a burst beyond it is spread over the following hours.
    #[arg(long, default_value_t = 200_000)]
    dropped_per_hour: u64,
    /// Walk every store once to put pages never sent, or changed since sent, on the ready lists; it resumes where it stopped.
    #[arg(long)]
    backfill_ready: bool,
    /// URL records the backfill reads per second.
    #[arg(long, default_value_t = 50_000)]
    backfill_per_sec: u64,
    /// URL records read per second while queuing the stored pages of domains new to the --domains list; 0 turns the walk off.
    #[arg(long, default_value_t = 100_000)]
    join_per_sec: u64,
    /// Pages the Python feeder sent, as tab-separated host, path, lastmod and Unix time, marked sent before the backfill.
    #[arg(long)]
    backfill_sent: Option<PathBuf>,
    /// Only these domains' pages are queued for the task API: a JSON list of names or of {host, rank}, read again when it changes.
    #[arg(long)]
    domains: Option<PathBuf>,
    /// A directory of URL lists, `<host>.txt` with a URL and optionally a tab and its date a line, taken in as the host's listing.
    #[arg(long)]
    inbox: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = match Cli::parse().command {
        Command::Run(args) => *args,
        Command::RemoveEscaped(args) => return remove_escaped(args),
        Command::RestoreEscaped(args) => return restore_escaped(args),
        Command::DropDomains(args) => return drop_domains(args),
    };
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
    let mut recrawl = Recrawl::new(args.recrawl_days.saturating_mul(DAY), args.gone_days.saturating_mul(DAY));
    if let Some(path) = &args.recrawl_overrides {
        recrawl = recrawl.with_overrides_file(path)?;
        println!("[rs] {} domains re-crawled on their own interval", recrawl.overridden());
    }
    let dsn = std::env::var("DESEARCH_DB").ok().filter(|d| !d.is_empty());
    let excluded = match &dsn {
        Some(dsn) => registry::excluded_categories(dsn).await?,
        None if args.no_registry => HashSet::new(),
        None => bail!("DESEARCH_DB is not set"),
    };
    let resources = Resources::new(args.cache_mb << 20, args.memtable_mb << 20, owned.len());
    let buckets = Arc::new(Buckets::open(&args.buckets_dir, &owned, &resources)?);
    let listed = args.domains.as_ref().map(|path| Allowed::from_file(path)).transpose()?;
    if let Some(allowed) = listed.clone() {
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
    let joining = listed.filter(|_| args.join_per_sec > 0).map(|allowed| {
        let (buckets, progress, halt, per_second) = (buckets.clone(), progress.clone(), halt.clone(), args.join_per_sec);
        std::thread::spawn(move || match dispatch::join_listed(&buckets, &allowed, per_second, &progress, &halt) {
            Ok(()) => println!("[rs] list walker stopped"),
            Err(error) => eprintln!("[rs] list walker failed: {error:#}"),
        })
    });
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
    let inbox = args.inbox.clone().map(|dir| {
        let (buckets, halt) = (buckets.clone(), halt.clone());
        std::thread::spawn(move || inbox::keep_taking(&buckets, &dir, &halt))
    });
    let wanted = Arc::new(dispatch::StockWanted::default());
    let stocking = args.task_api.as_ref().map(|_| {
        let (buckets, progress, halt, wanted, per_second) = (buckets.clone(), progress.clone(), halt.clone(), wanted.clone(), args.join_per_sec.max(1));
        std::thread::spawn(move || match dispatch::keep_stocks(&buckets, &wanted, per_second, &progress, &halt) {
            Ok(()) => println!("[rs] refresh stock keeper stopped"),
            Err(error) => eprintln!("[rs] refresh stock keeper failed: {error:#}"),
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
            recrawl.clone(),
            progress.clone(),
        ).await?
        .paced(args.urls_per_hour)
        .refreshing(wanted.clone());
        tokio::spawn(dispatcher.run(stopped.clone()));
    }
    if let Some(base) = &args.outcomes_url {
        let feed = OutcomeFeed::new(base)?;
        let reader = OutcomeReader::new(recrawl.clone(), args.dropped_per_hour);
        tokio::spawn(dispatch::follow_outcomes(buckets.clone(), feed, reader, progress.clone(), stopped.clone()));
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
    if let Some(stocking) = stocking {
        let _ = tokio::task::spawn_blocking(move || stocking.join()).await;
    }
    if let Some(joining) = joining {
        let _ = tokio::task::spawn_blocking(move || joining.join()).await;
    }
    if let Some(inbox) = inbox {
        let _ = tokio::task::spawn_blocking(move || inbox.join()).await;
    }
    println!("[rs] done {}", crawl.summary());
    Ok(())
}

fn remove_escaped(args: RemoveEscapedArgs) -> Result<()> {
    if let Some(dir) = &args.backup_dir {
        std::fs::create_dir_all(dir)?;
    }
    let owned = parse_buckets(&args.buckets)?;
    let resources = Resources::new(1 << 30, 512 << 20, owned.len());
    let buckets = Buckets::open(&args.buckets_dir, &owned, &resources)?;
    let stores: Vec<&BucketStore> = buckets.stores().collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let backup_dir = args.backup_dir.as_deref();
    let found = std::thread::scope(|scope| {
        let handles: Vec<_> = stores
            .chunks(stores.len().div_ceil(threads).max(1))
            .map(|chunk| {
                scope.spawn(move || {
                    chunk.iter().try_fold(Escaped::default(), |sum, store| {
                        let backup = backup_dir.map(|dir| dir.join(format!("{}.bin", store.name())));
                        let found = store.remove_escaped(backup.as_deref())?;
                        anyhow::Ok(Escaped { pages: sum.pages + found.pages, sitemaps: sum.sitemaps + found.sitemaps })
                    })
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a store pass panicked")).try_fold(Escaped::default(), |sum, found| {
            let found = found?;
            anyhow::Ok(Escaped { pages: sum.pages + found.pages, sitemaps: sum.sitemaps + found.sitemaps })
        })
    })?;
    let done = if backup_dir.is_some() { "deleted" } else { "found" };
    println!("{done} {} pages and {} sitemaps with escaped addresses in {} stores", found.pages, found.sitemaps, stores.len());
    Ok(())
}

fn drop_domains(args: DropDomainsArgs) -> Result<()> {
    let domains: Vec<String> = args.domains.split(',').map(|d| d.trim().to_lowercase()).filter(|d| !d.is_empty()).collect();
    let mut owned: Vec<usize> = domains.iter().map(|d| bucket_of(d)).collect();
    owned.sort_unstable();
    owned.dedup();
    let resources = Resources::new(1 << 30, 512 << 20, owned.len());
    let buckets = Buckets::open(&args.buckets_dir, &owned, &resources)?;
    for domain in &domains {
        let started = std::time::Instant::now();
        let found = buckets.store(domain).drop_domain(domain, &args.reason, args.delete)?;
        if !found.known {
            println!("{domain}: not kept in its store");
            continue;
        }
        let done = if args.delete { format!("deleted in {:.0}s", started.elapsed().as_secs_f64()) } else { "nothing deleted without --delete".into() };
        println!("{domain}: {} ready pages, {} URLs in its last sitemaps; {done}", found.ready, found.urls);
    }
    Ok(())
}

fn restore_escaped(args: RestoreEscapedArgs) -> Result<()> {
    let owned = parse_buckets(&args.buckets)?;
    let resources = Resources::new(1 << 30, 512 << 20, owned.len());
    let buckets = Buckets::open(&args.buckets_dir, &owned, &resources)?;
    let mut restored = 0;
    for store in buckets.stores() {
        let backup = args.backup_dir.join(format!("{}.bin", store.name()));
        if backup.exists() {
            restored += store.restore(&backup)?;
        }
    }
    println!("restored {restored} entries");
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

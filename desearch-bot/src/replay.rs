//! Hours of the dispatcher on copied stores: passes on a simulated clock against an in-process task API while the list walker joins listed domains and the stock keeper chooses refresh stocks, with outcomes drawn from a real day's and sitemap reads simulated.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rocksdb::{Direction, IteratorMode, Options, WriteBatch, DB};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::allowed::Allowed;
use crate::buckets::{BucketStore, Buckets, Resources, URL};
use crate::dispatch::{self, Dispatcher, Progress, StockWanted};
use crate::hotkey::Hotkey;
use crate::outcomes::{OutcomeReader, OutcomeRow, ResentPages};
use crate::ready::{bounded, fetchable, prefixed, Outcome};
use crate::recrawl::Recrawl;
use crate::records;
use crate::taskapi::TaskApi;
use crate::urls::{self, Record};

const PASS: u32 = 5;
const HOUR: u32 = 3_600;
const FLOOR: u32 = 86_400;
/// Seconds from a send to its outcome, as the miners, validators and publisher take.
const OUTCOME_AFTER: u32 = 600;
/// The sitemap every simulated read is filed under.
const REPLAY_SITEMAP: i64 = 900_000_001;
/// Pages one simulated sitemap read lists, and the part of them it re-dates.
const LISTED_PER_READ: usize = 500;
const REDATED: f64 = 0.05;
/// Pages the re-stamping site's read lists, every one re-dated.
const RESTAMP_LISTED: usize = 20_000;

/// Keys read between two catch-ups with the live store: its compactions delete the files a long read still holds.
const COPY_CHUNK: usize = 2_000_000;
/// Chunks read again after a file of the live store vanished under them.
const COPY_RETRIES: usize = 20;

/// Copy one bucket store key by key from a live store, opened read-only beside the process that owns it; returns the keys copied.
pub fn copy_bucket(from: &Path, secondary: &Path, to: &Path, resources: &Resources) -> Result<u64> {
    let mut options = Options::default();
    options.set_max_open_files(256);
    let source = DB::open_as_secondary(&options, from, secondary).with_context(|| format!("opening {} read-only", from.display()))?;
    let target = BucketStore::open(to, resources)?;
    let (mut copied, mut last, mut retries): (u64, Option<Vec<u8>>, usize) = (0, None, 0);
    loop {
        source.try_catch_up_with_primary()?;
        let start = last.as_ref().map(|key| [key.as_slice(), b"\0"].concat());
        let mode = start.as_deref().map_or(IteratorMode::Start, |key| IteratorMode::From(key, rocksdb::Direction::Forward));
        let (mut batch, mut read, mut chunk_last) = (WriteBatch::default(), 0usize, None);
        let mut failed = None;
        for item in source.iterator(mode).take(COPY_CHUNK) {
            match item {
                Ok((key, value)) => {
                    batch.put(&key, &value);
                    chunk_last = Some(key.to_vec());
                    read += 1;
                }
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failed {
            retries += 1;
            if retries > COPY_RETRIES {
                return Err(error.into());
            }
            continue;
        }
        target.db.write(batch)?;
        copied += read as u64;
        if read < COPY_CHUNK {
            break;
        }
        last = chunk_last;
    }
    target.db.flush()?;
    drop(source);
    std::fs::remove_dir_all(secondary).ok();
    Ok(copied)
}

pub struct ReplayArgs {
    pub buckets_dir: PathBuf,
    pub buckets: Vec<usize>,
    pub domains: PathBuf,
    /// A real day's outcomes: host, hour of the day, published, failed, dropped, tab-separated.
    pub outcomes: PathBuf,
    pub start: u32,
    pub hours: u32,
    pub per_hour: u64,
    /// A site whose pages all fail from the middle of the day on.
    pub refusing: Option<String>,
    /// A site whose sitemap re-dates every page each hour; the listed site with most pages if none.
    pub restamping: Option<String>,
    /// Stored records the list walker and the stock keeper each read per second, beside the passes.
    pub join_per_sec: u64,
    /// Pages the resend walk lists per second for domains whose `resend_before` changed.
    pub resend_per_sec: u64,
    /// Hours into the replay at which the crawl list becomes `switch_domains`.
    pub switch_at: Option<u32>,
    pub switch_domains: Option<PathBuf>,
    /// Domains re-crawled on their own interval, as the bot's `--recrawl-overrides`.
    pub recrawl_overrides: Option<PathBuf>,
    /// Where the hourly report goes.
    pub log: PathBuf,
}

/// The bars, counted as the replay goes: per page its last send, how that send ended and when a sitemap last re-dated it, per site its sends of the last hour.
#[derive(Default)]
struct Bars {
    last: HashMap<u64, (u32, u8)>,
    redated: HashMap<u64, u32>,
    sites: HashMap<String, VecDeque<u32>>,
    /// Sends within a day of the last that nothing explains: a refresh gone early.
    early: u64,
    /// Refreshes on an interval under a day, such as a news article under two days old.
    due_within_day: u64,
    early_redated: u64,
    early_after_failure: u64,
    early_after_drop: u64,
    restamped_early: u64,
    over_hour: u64,
    over_spread: u64,
    /// Sends this hour to sites that ask for a delay between requests, and those sites.
    delayed_sent: u64,
    delayed_sites: HashSet<String>,
    refusing: HashMap<u32, u64>,
}

const PENDING: u8 = 0;
const FAILED: u8 = 1;
const DONE: u8 = 2;
const DROPPED: u8 = 3;

fn page_id(url: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    hasher.finish()
}

impl Bars {
    /// A send of `url`; `interval` gives the page's refresh interval as of a send at the time it is given.
    fn sent(&mut self, url: &str, at: u32, restamping: bool, interval: impl FnOnce(u32) -> u32) {
        let id = page_id(url);
        if let Some((before, how)) = self.last.insert(id, (at, PENDING)) {
            let redated = self.redated.get(&id).is_some_and(|&when| when >= before);
            if at.saturating_sub(before) < FLOOR {
                // A sitemap date newer than the last send sends the page at once, whatever that send's outcome.
                match how {
                    _ if redated && restamping => self.restamped_early += 1,
                    _ if redated => self.early_redated += 1,
                    FAILED => self.early_after_failure += 1,
                    DROPPED => self.early_after_drop += 1,
                    _ if at.saturating_sub(before) >= interval(before) => self.due_within_day += 1,
                    _ => self.early += 1,
                }
            }
        }
    }

    /// A send to `host`, held to `limit` an hour; `delayed` for a site that asks for a delay between requests.
    fn spaced(&mut self, host: &str, at: u32, limit: u64, delayed: bool) {
        let times = self.sites.entry(host.to_string()).or_default();
        while times.front().is_some_and(|&t| t + HOUR <= at) {
            times.pop_front();
        }
        times.push_back(at);
        if times.len() as u64 > limit {
            self.over_hour += 1;
        }
        // A delayed site gets what its delay allows over a pass, rounded up; any other the share saved up and what refills while it goes.
        let (within, most) = if delayed {
            self.delayed_sent += 1;
            self.delayed_sites.insert(host.to_string());
            (1, (u64::from(PASS) * limit).div_ceil(u64::from(HOUR)))
        } else {
            (dispatch::SPREAD_SECONDS, dispatch::most_at_once(limit) + limit * u64::from(dispatch::SPREAD_SECONDS) / u64::from(HOUR))
        };
        let times = &self.sites[host];
        if times.iter().rev().take_while(|&&t| t + within > at).count() as u64 > most {
            self.over_spread += 1;
        }
    }

    fn ended(&mut self, url: &str, sent_at: u32, outcome: Outcome) {
        if let Some(entry) = self.last.get_mut(&page_id(url)).filter(|entry| entry.0 == sent_at) {
            entry.1 = match outcome {
                Outcome::Failed => FAILED,
                Outcome::Dropped => DROPPED,
                _ => DONE,
            };
        }
    }
}

/// What one round of simulated sitemap reads did.
#[derive(Default)]
struct Read {
    reads: usize,
    new: usize,
    redated: Vec<String>,
}

/// A sitemap read of each domain, from its stored pages: a few re-dated to now and a few new; the re-stamping site re-dates every page it lists.
fn read_sitemaps(buckets: &Buckets, domains: Vec<String>, restamping: &str, now: u32, seed: u64) -> Result<Read> {
    let rounds = dispatch::on_readers(domains, |domain| {
        let store = buckets.store(&domain);
        let restamps = domain == restamping;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ page_id(&domain));
        let stored = stored_pages(store, &domain, if restamps { RESTAMP_LISTED } else { LISTED_PER_READ })?;
        let mut entries = Vec::with_capacity(stored.len() + 8);
        let mut redated = Vec::new();
        for (i, (rest, record)) in stored.iter().enumerate() {
            let url = fetchable(rest, record.flags);
            let Some(parsed) = urls::parse(&url, &domain) else {
                continue;
            };
            let lastmod = if restamps || rng.gen_bool(REDATED) {
                redated.push(url);
                now.saturating_sub(i as u32)
            } else {
                record.lastmod
            };
            entries.push((parsed, lastmod, false));
        }
        let new = (stored.len() / 100).max(1);
        for n in 0..new {
            if let Some(url) = urls::parse(&format!("https://{domain}/replay-{now}-{n}"), &domain) {
                entries.push((url, now, false));
            }
        }
        store.record_listing(&domain, REPLAY_SITEMAP, entries, now)?;
        Ok(Read { reads: 1, new, redated })
    })?;
    Ok(rounds.into_iter().fold(Read::default(), |mut sum, one| {
        sum.reads += one.reads;
        sum.new += one.new;
        sum.redated.extend(one.redated);
        sum
    }))
}

fn stored_pages(store: &BucketStore, domain: &str, most: usize) -> Result<Vec<(Vec<u8>, Record)>> {
    let prefix = prefixed(URL, domain.as_bytes(), b"");
    let mut pages = Vec::new();
    for item in store.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)).take(most) {
        let (key, raw) = item?;
        if let Some(record) = Record::unpack(&raw) {
            pages.push((key[prefix.len()..].to_vec(), record));
        }
    }
    Ok(pages)
}

/// What became of one sent page in the replay.
struct Sent {
    url: String,
    host: String,
    at: u32,
}

/// The fake task API: room for whatever is asked, and every URL taken kept for the replay to judge.
#[derive(Default)]
struct Taken {
    urls: Vec<(String, String)>,
}

pub async fn replay(args: ReplayArgs) -> Result<()> {
    let resources = Resources::new(512 << 20, 1 << 30, args.buckets.len());
    let buckets = Arc::new(Buckets::open(&args.buckets_dir, &args.buckets, &resources)?);
    // The replay rewrites its own copy of the list when it switches, never the file it was given.
    let working = args.log.with_extension("domains.json");
    std::fs::copy(&args.domains, &working)?;
    let list = Allowed::from_file(&working)?;
    buckets.allow(&list);
    let rates = read_rates(&args.outcomes)?;
    let recrawl = match &args.recrawl_overrides {
        Some(path) => Recrawl::default().with_overrides_file(path)?,
        None => Recrawl::default(),
    };
    let mut log = std::io::BufWriter::new(std::fs::File::create(&args.log)?);
    let mut bars = Bars::default();
    let listed: Vec<String> = buckets.listed();
    let urls_of = |domain: &String| buckets.store(domain).domain(domain).ok().flatten().and_then(|record| records::int(record.get("urls"))).unwrap_or(0);
    let restamping = args.restamping.clone().or_else(|| listed.iter().max_by_key(|domain| urls_of(domain)).cloned()).unwrap_or_default();
    println!("[replay] {} listed domains; {restamping} re-stamps its sitemap every hour", listed.len());

    let progress = Arc::new(Progress::default());
    let stop = Arc::new(AtomicBool::new(false));
    let clock = Arc::new(AtomicU32::new(args.start));
    let walking = Instant::now();
    let walker = {
        let (buckets, list, progress, stop) = (buckets.clone(), list.clone(), progress.clone(), stop.clone());
        let per_second = args.join_per_sec;
        std::thread::spawn(move || dispatch::join_listed(&buckets, &list, per_second, &progress, &stop))
    };
    let resender = {
        let (buckets, list, progress, stop) = (buckets.clone(), list.clone(), progress.clone(), stop.clone());
        let (read_per_sec, write_per_sec) = (dispatch::resend_reads(args.join_per_sec), args.resend_per_sec);
        std::thread::spawn(move || dispatch::resend_listed(&buckets, &list, read_per_sec, write_per_sec, &progress, &stop))
    };
    let resent = Arc::new(ResentPages::default());
    let wanted = Arc::new(StockWanted::default());
    let keeper = {
        let (buckets, wanted, progress, stop, clock) = (buckets.clone(), wanted.clone(), progress.clone(), stop.clone(), clock.clone());
        let per_second = args.join_per_sec;
        std::thread::spawn(move || dispatch::keep_stocks_on(&buckets, &wanted, per_second, &progress, &stop, &|| clock.load(Ordering::Relaxed)))
    };

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let taken = Arc::new(Mutex::new(Taken::default()));
    tokio::spawn(serve(listener, taken.clone()));
    let api = TaskApi::new(&base, Hotkey::from_uri("//replay")?)?;
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, dispatch::shares(50, 10), recrawl.clone(), progress.clone())
        .await?
        .paced_from(args.per_hour, args.start)
        .refreshing(wanted.clone())
        .resent_into(resent.clone());
    let mut reader = OutcomeReader::new(recrawl.clone(), 200_000).with_resent(resent.clone());
    let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
    let mut waiting: VecDeque<Sent> = VecDeque::new();
    let mut delays: HashMap<String, Option<u64>> = HashMap::new();
    let end = args.start + args.hours * HOUR;
    let mut now = args.start;
    let mut pass_ms: Vec<u64> = Vec::new();
    let mut hour_sent = 0u64;
    let mut last = Counts::of(&progress);
    let mut walked: Option<f64> = None;
    let mut listings = (0, 0, 0);
    while now < end {
        clock.store(now, Ordering::Relaxed);
        if args.switch_at.is_some_and(|hours| now == args.start + hours * HOUR) {
            if let Some(path) = &args.switch_domains {
                let next = working.with_extension("next");
                std::fs::copy(path, &next)?;
                std::fs::rename(&next, &working)?;
                list.reload()?;
                let line = format!("[replay] the crawl list is now {} at hour {}", path.display(), (now - args.start) / HOUR);
                println!("{line}");
                writeln!(log, "{line}")?;
            }
        }
        if (now - args.start) % HOUR == 0 {
            // Each listed domain's sitemap is read about once a day, the re-stamping site's every hour.
            let mut domains: Vec<String> = listed.choose_multiple(&mut rng, listed.len() / 24).cloned().collect();
            domains.retain(|domain| *domain != restamping);
            domains.push(restamping.clone());
            let (reading, seed) = (buckets.clone(), u64::from(now));
            let restamps = restamping.clone();
            let read = tokio::task::spawn_blocking(move || read_sitemaps(&reading, domains, &restamps, now, seed)).await??;
            for url in &read.redated {
                bars.redated.insert(page_id(url), now);
            }
            listings = (read.reads, read.new, read.redated.len());
        }
        let started = Instant::now();
        dispatcher.pass(now).await?;
        let took = started.elapsed().as_millis() as u64;
        pass_ms.push(took);
        if took > 2_000 {
            let ms: Vec<u64> = progress.pass_ms.iter().map(|ms| ms.load(Ordering::Relaxed)).collect();
            let line = format!("[replay] slow pass at hour {:.2}: {took} ms (absorb {} ms, pick {} ms, send {} ms)", f64::from(now - args.start) / f64::from(HOUR), ms[0], ms[1], ms[2]);
            println!("{line}");
            writeln!(log, "{line}")?;
        }
        for (host, url) in std::mem::take(&mut taken.lock().unwrap_or_else(|e| e.into_inner()).urls) {
            let listed = buckets.terms(&host).hourly.unwrap_or(1_000_000);
            let delay = *delays.entry(host.clone()).or_insert_with(|| buckets.store_for(&host).and_then(|store| store.domain(&host).ok().flatten()).as_ref().and_then(dispatch::delay_hourly));
            bars.spaced(&host, now, delay.map_or(listed, |per_hour| per_hour.min(listed)), delay.is_some());
            let interval = |_: u32| recrawl.interval(&host);
            bars.sent(&url, now, host == restamping, interval);
            if args.refusing.as_deref() == Some(host.as_str()) {
                *bars.refusing.entry((now - args.start) / HOUR).or_default() += 1;
            }
            hour_sent += 1;
            waiting.push_back(Sent { url, host, at: now });
        }
        let mut rows = Vec::new();
        while waiting.front().is_some_and(|sent| sent.at + OUTCOME_AFTER <= now) {
            let sent = waiting.pop_front().expect("a front");
            let hour = (sent.at % 86_400) / HOUR;
            let refusing = args.refusing.as_deref() == Some(sent.host.as_str()) && now >= args.start + args.hours * HOUR / 2;
            let outcome = if refusing { Outcome::Failed } else { draw(&rates, &sent.host, hour, &mut rng) };
            bars.ended(&sent.url, sent.at, outcome);
            rows.push(OutcomeRow { url: sent.url, host: sent.host, outcome, at: i64::from(sent.at + OUTCOME_AFTER) * 1_000_000 });
        }
        if !rows.is_empty() {
            let buckets = buckets.clone();
            reader = tokio::task::spawn_blocking(move || -> Result<OutcomeReader> {
                reader.apply(&buckets, &rows, now)?;
                Ok(reader)
            })
            .await??;
        }
        let (done, of) = (progress.join_done.load(Ordering::Relaxed), progress.join_of.load(Ordering::Relaxed));
        if walked.is_none() && of > 0 && done >= of {
            let seconds = walking.elapsed().as_secs_f64();
            walked = Some(seconds);
            let line = format!(
                "[replay] the list walker joined {done} domains, {} pages queued, in {seconds:.0}s of wall time, by simulated hour {:.1}",
                progress.join_queued.load(Ordering::Relaxed),
                f64::from(now - args.start) / f64::from(HOUR)
            );
            println!("{line}");
            writeln!(log, "{line}")?;
        }
        now += PASS;
        if (now - args.start) % HOUR == 0 {
            let counts = Counts::of(&progress);
            let mut sorted = pass_ms.clone();
            sorted.sort_unstable();
            let p = |q: f64| sorted.get(((sorted.len() as f64 * q) as usize).min(sorted.len().saturating_sub(1))).copied().unwrap_or(0);
            let line = format!(
                "[replay] hour {:>2}: {hour_sent} sent (new {} refresh {} retry {} resend {} fill {}), resend walk listed {} waiting {}, pass p50 {} ms p99 {} ms max {} ms, {} ready in {} domains, walker {done}/{of} queued {} before the floor {} cleared at caps {}, stocks chosen {} fill domains {}, sitemaps {} read {} new {} re-dated, rss {} MiB; over hour {} over spread {} early {} due within a day {} early re-dated {} re-stamped early {} early after a failure {} after a drop {}, sites with a delay {} sent to {}",
                (now - args.start) / HOUR,
                counts.new - last.new,
                counts.refresh - last.refresh,
                counts.retry - last.retry,
                counts.resend - last.resend,
                counts.fill - last.fill,
                progress.join_resent.load(Ordering::Relaxed),
                progress.resend_waiting.load(Ordering::Relaxed),
                p(0.5),
                p(0.99),
                sorted.last().copied().unwrap_or(0),
                progress.ready.load(Ordering::Relaxed),
                progress.ready_domains.load(Ordering::Relaxed),
                progress.join_queued.load(Ordering::Relaxed),
                progress.join_before_floor.load(Ordering::Relaxed),
                progress.capped_cleared.load(Ordering::Relaxed),
                progress.stocked.load(Ordering::Relaxed),
                progress.fill_domains.load(Ordering::Relaxed),
                listings.0,
                listings.1,
                listings.2,
                rss_mib(),
                bars.over_hour,
                bars.over_spread,
                bars.early,
                bars.due_within_day,
                bars.early_redated,
                bars.restamped_early,
                bars.early_after_failure,
                bars.early_after_drop,
                bars.delayed_sent,
                bars.delayed_sites.len()
            );
            println!("{line}");
            writeln!(log, "{line}")?;
            log.flush()?;
            last = counts;
            hour_sent = 0;
            bars.delayed_sent = 0;
            bars.delayed_sites.clear();
            pass_ms.clear();
        }
    }
    stop.store(true, Ordering::Relaxed);
    walker.join().map_err(|_| anyhow::anyhow!("the list walker panicked"))??;
    resender.join().map_err(|_| anyhow::anyhow!("the resend walker panicked"))??;
    keeper.join().map_err(|_| anyhow::anyhow!("the stock keeper panicked"))??;
    if let Some(host) = &args.refusing {
        let mut hours: Vec<(u32, u64)> = bars.refusing.into_iter().collect();
        hours.sort_unstable();
        writeln!(log, "[replay] {host} sends by hour, refusing from hour {}: {hours:?}", args.hours / 2)?;
        println!("[replay] {host} sends by hour, refusing from hour {}: {hours:?}", args.hours / 2);
    }
    log.flush()?;
    println!("[replay] done; the report is in {}", args.log.display());
    Ok(())
}

#[derive(Clone, Copy)]
struct Counts {
    new: u64,
    refresh: u64,
    retry: u64,
    resend: u64,
    fill: u64,
}

impl Counts {
    fn of(progress: &Progress) -> Counts {
        let get = |counter: &std::sync::atomic::AtomicU64| counter.load(std::sync::atomic::Ordering::Relaxed);
        Counts { new: get(&progress.sent_new), refresh: get(&progress.sent_refresh), retry: get(&progress.sent_retry), resend: get(&progress.sent_resend), fill: get(&progress.sent_fill) }
    }
}

/// Each host's published, failed and dropped counts by hour of the day.
fn read_rates(path: &Path) -> Result<HashMap<String, [[u32; 3]; 24]>> {
    let mut rates: HashMap<String, [[u32; 3]; 24]> = HashMap::new();
    for line in std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        let [host, hour, published, failed, dropped] = fields[..] else {
            continue;
        };
        let (Ok(hour), Ok(published), Ok(failed), Ok(dropped)) = (hour.parse::<usize>(), published.parse(), failed.parse(), dropped.parse()) else {
            continue;
        };
        rates.entry(host.to_string()).or_insert([[0; 3]; 24])[hour.min(23)] = [published, failed, dropped];
    }
    Ok(rates)
}

/// An outcome as the host's real outcomes went that hour; a host the day never saw publishes three in four.
fn draw(rates: &HashMap<String, [[u32; 3]; 24]>, host: &str, hour: u32, rng: &mut impl Rng) -> Outcome {
    let counts = rates.get(host).map(|day| {
        let hour = day[hour as usize % 24];
        if hour.iter().sum::<u32>() > 0 {
            hour
        } else {
            day.iter().fold([0; 3], |sum, h| [sum[0] + h[0], sum[1] + h[1], sum[2] + h[2]])
        }
    });
    let [published, failed, dropped] = counts.filter(|c| c.iter().sum::<u32>() > 0).unwrap_or([3, 1, 0]);
    let pick = rng.gen_range(0..published + failed + dropped);
    if pick < published {
        Outcome::Published
    } else if pick < published + failed {
        Outcome::Failed
    } else {
        Outcome::Dropped
    }
}

fn rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| status.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map_or(0, |kb| kb / 1024)
}

/// The task API's two calls the dispatcher makes: room for anything, and every enqueued URL kept.
async fn serve(listener: TcpListener, taken: Arc<Mutex<Taken>>) {
    while let Ok((mut socket, _)) = listener.accept().await {
        let taken = taken.clone();
        tokio::spawn(async move {
            let mut buffer = Vec::new();
            let end = loop {
                if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end;
                }
                let mut chunk = [0u8; 8192];
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                }
            };
            let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
            let length: usize =
                head.lines().filter_map(|l| l.split_once(':')).find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.trim().parse().ok()).unwrap_or(0);
            let mut body = buffer[end + 4..].to_vec();
            while body.len() < length {
                let mut chunk = [0u8; 65536];
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => body.extend_from_slice(&chunk[..n]),
                }
            }
            let reply = if head.starts_with("GET /v1/room") {
                br#"{"room_tasks":100000,"queue":0,"unrevealed":0,"refusing":false}"#.to_vec()
            } else {
                let sent: Value = serde_json::from_slice(&body).unwrap_or_default();
                let urls = sent["urls"].as_array().into_iter().flatten().filter_map(|u| Some((u["host"].as_str()?.to_string(), u["url"].as_str()?.to_string())));
                taken.lock().unwrap_or_else(|e| e.into_inner()).urls.extend(urls);
                br#"{"queued":true}"#.to_vec()
            };
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reply.len());
            let _ = socket.write_all(&[head.into_bytes(), reply].concat()).await;
        });
    }
}

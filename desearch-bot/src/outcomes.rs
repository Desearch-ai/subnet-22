//! What became of the URLs the bot queued, read in order from the task API's public outcome feed.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use serde_json::Value;

use crate::buckets::{bucket_of, Buckets};
use crate::dispatch::on_readers;
use crate::ready::{Outcome, PageSchedule, Scheduled, Why, Yield};
use crate::recrawl::Recrawl;
use crate::urls::Normaliser;

const TIMEOUT: Duration = Duration::from_secs(60);
/// Hosts whose domain is remembered; the memory starts over past this many, or an hour on, so domain list changes count.
const HOSTS_KEPT: usize = 1_000_000;
const HOSTS_FOR: Duration = Duration::from_secs(3600);
/// Bits in each of the two sets of pages sent to go once more: 64 MiB, a few percent false positives at a day's resends.
const RESENT_BITS: usize = 1 << 29;
/// How long each set takes inserts before the older one is dropped; outcomes come back well within it.
const RESENT_ROTATE: Duration = Duration::from_secs(6 * 3600);
const RESENT_HASHES: u64 = 3;

/// Pages lately sent to go once more, so their outcomes stay out of the yields that pace new pages: two sets of bits, the older dropped every `RESENT_ROTATE`, allocated on the first insert. A false positive only leaves one outcome out of a yield.
#[derive(Default)]
pub struct ResentPages {
    sets: Mutex<Option<Resent>>,
}

struct Resent {
    current: Vec<u64>,
    older: Vec<u64>,
    since: Instant,
}

impl ResentPages {
    pub fn insert(&self, domain: &str, rest: &[u8]) {
        let mut sets = self.sets.lock().unwrap_or_else(|e| e.into_inner());
        let sets = sets.get_or_insert_with(|| Resent { current: vec![0; RESENT_BITS / 64], older: vec![0; RESENT_BITS / 64], since: Instant::now() });
        if sets.since.elapsed() >= RESENT_ROTATE {
            sets.older = std::mem::replace(&mut sets.current, vec![0; RESENT_BITS / 64]);
            sets.since = Instant::now();
        }
        for bit in bits(domain, rest) {
            sets.current[bit / 64] |= 1 << (bit % 64);
        }
    }

    pub fn contains(&self, domain: &str, rest: &[u8]) -> bool {
        let sets = self.sets.lock().unwrap_or_else(|e| e.into_inner());
        let Some(sets) = sets.as_ref() else {
            return false;
        };
        let has = |set: &[u64]| bits(domain, rest).iter().all(|&bit| set[bit / 64] & (1 << (bit % 64)) != 0);
        has(&sets.current) || has(&sets.older)
    }

    /// Whether anything was inserted yet; until then outcomes need no page key.
    pub fn active(&self) -> bool {
        self.sets.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

fn bits(domain: &str, rest: &[u8]) -> [usize; RESENT_HASHES as usize] {
    let hash = |seed: u64| {
        let mut hasher = DefaultHasher::new();
        (seed, domain, rest).hash(&mut hasher);
        hasher.finish()
    };
    let (first, step) = (hash(1), hash(2) | 1);
    std::array::from_fn(|i| (first.wrapping_add(step.wrapping_mul(i as u64)) % RESENT_BITS as u64) as usize)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeRow {
    pub url: String,
    /// The URL's host without `www.`, as the task API wrote it.
    pub host: String,
    pub outcome: Outcome,
    /// Epoch microseconds.
    pub at: i64,
}

/// The rows of one outcome file; rows with an outcome this bot does not know are skipped.
pub fn read_parquet(data: Bytes) -> Result<Vec<OutcomeRow>> {
    let reader = SerializedFileReader::new(data).context("not a Parquet file")?;
    let mut rows = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let (mut url, mut host, mut outcome, mut at) = (None, None, None, None);
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("url", Field::Str(value)) => url = Some(value.clone()),
                ("host", Field::Str(value)) => host = Some(value.clone()),
                ("outcome", Field::Str(value)) => outcome = Outcome::parse(value),
                ("at", Field::TimestampMicros(us)) => at = Some(*us),
                ("at", Field::TimestampMillis(ms)) => at = Some(ms * 1000),
                ("at", Field::Long(value)) => at = Some(if *value > 100_000_000_000_000_000 { value / 1000 } else { *value }),
                _ => {}
            }
        }
        if let (Some(url), Some(host), Some(outcome), Some(at)) = (url, host, outcome, at) {
            rows.push(OutcomeRow { url, host, outcome, at });
        }
    }
    Ok(rows)
}

/// What one outcome file held, and what reading it did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counted {
    pub published: u64,
    pub failed: u64,
    pub dropped: u64,
    /// Dropped pages put back now, within this hour's budget; the rest follow in later hours.
    pub dropped_now: u64,
    /// Rows of hosts no domain here was sent under.
    pub unknown: u64,
}

/// Spreads pages over the coming clock hours, at most `per_hour` in each.
#[derive(Clone, Copy, Debug)]
pub struct HourlySpread {
    per_hour: u64,
    hour: u32,
    used: u64,
}

impl HourlySpread {
    pub fn new(per_hour: u64) -> Self {
        HourlySpread { per_hour, hour: 0, used: 0 }
    }

    /// When the next page goes: now while this hour has room, else the start of the first later hour that has.
    pub fn due(&mut self, now: u32) -> u32 {
        if self.per_hour == 0 {
            return u32::MAX;
        }
        if self.hour < now / 3600 {
            (self.hour, self.used) = (now / 3600, 0);
        }
        if self.used >= self.per_hour {
            (self.hour, self.used) = (self.hour + 1, 0);
        }
        self.used += 1;
        if self.hour == now / 3600 {
            now
        } else {
            self.hour.saturating_mul(3600)
        }
    }
}

/// Reads outcome files into per-domain yields and schedules for failed and dropped pages, without looking up a single page; pages sent once more for a `resend_before` stay out of the yields.
pub struct OutcomeReader {
    recrawl: Recrawl,
    dropped: HourlySpread,
    domains: HashMap<String, Option<String>>,
    domains_since: Instant,
    resent: Option<Arc<ResentPages>>,
}

impl OutcomeReader {
    pub fn new(recrawl: Recrawl, dropped_per_hour: u64) -> Self {
        OutcomeReader { recrawl, dropped: HourlySpread::new(dropped_per_hour), domains: HashMap::new(), domains_since: Instant::now(), resent: None }
    }

    /// Leave the outcomes of pages in `resent`, sent to go once more, out of the yields.
    pub fn with_resent(mut self, resent: Arc<ResentPages>) -> Self {
        self.resent = Some(resent);
        self
    }

    /// The domain a host's pages were sent under: the host itself or its nearest parent that this process holds and may queue.
    pub fn domain_of(&mut self, buckets: &Buckets, host: &str) -> Option<String> {
        let host = host.split_once(':').map_or(host, |(name, _)| name);
        let host = host.strip_prefix("www.").unwrap_or(host);
        if let Some(found) = self.domains.get(host) {
            return found.clone();
        }
        if self.domains.len() >= HOSTS_KEPT || self.domains_since.elapsed() >= HOSTS_FOR {
            self.domains.clear();
            self.domains_since = Instant::now();
        }
        let mut name = host;
        let found = loop {
            let known = buckets.allows(name) && buckets.store_for(name).is_some_and(|store| store.has_domain(name).unwrap_or(false));
            if known {
                break Some(name.to_string());
            }
            match name.split_once('.') {
                Some((_, parent)) if parent.contains('.') => name = parent,
                _ => break None,
            }
        };
        self.domains.insert(host.to_string(), found.clone());
        found
    }

    /// Tally a file's rows per domain and schedule its failed and dropped pages, a few writes per store.
    pub fn apply(&mut self, buckets: &Buckets, rows: &[OutcomeRow], now: u32) -> Result<Counted> {
        let mut counted = Counted::default();
        let mut tallies: HashMap<String, Yield> = HashMap::new();
        let mut schedules: HashMap<usize, Vec<PageSchedule>> = HashMap::new();
        for row in rows {
            match row.outcome {
                Outcome::Published | Outcome::Unchanged => counted.published += 1,
                Outcome::Failed => counted.failed += 1,
                Outcome::Dropped => counted.dropped += 1,
            }
            let Some(domain) = self.domain_of(buckets, &row.host) else {
                counted.unknown += 1;
                continue;
            };
            let at = row.at.div_euclid(1_000_000).clamp(0, u32::MAX.into()) as u32;
            let resent = self.resent.as_ref().filter(|resent| resent.active());
            let published = matches!(row.outcome, Outcome::Published | Outcome::Unchanged);
            let rest = (resent.is_some() || !published).then(|| Normaliser::new(&domain).ok().and_then(|n| n.parse(&row.url)).map(|key| key.key[domain.len() + 1..].to_vec())).flatten();
            // A dropped task never reached the site, so it says nothing about the domain; a page sent once more for its domain's resend_before says little about its new pages.
            let tally = tallies.entry(domain.clone()).or_default();
            if !resent.zip(rest.as_ref()).is_some_and(|(resent, rest)| resent.contains(&domain, rest)) {
                match row.outcome {
                    Outcome::Published | Outcome::Unchanged => tally.ok += 1.0,
                    Outcome::Failed => tally.failed += 1.0,
                    Outcome::Dropped => {}
                }
            }
            tally.at = tally.at.max(at);
            let interval = self.recrawl.interval(&domain);
            let next = match row.outcome {
                Outcome::Published | Outcome::Unchanged => continue,
                Outcome::Failed => Scheduled { why: Why::Failed, due: at.saturating_add(interval), at },
                // A page whose task was lost goes again soon, but a burst of them is spread out, never later than its re-crawl.
                Outcome::Dropped => {
                    let due = self.dropped.due(now).min(at.saturating_add(interval));
                    counted.dropped_now += u64::from(due <= now);
                    Scheduled { why: Why::Dropped, due, at }
                }
            };
            let Some(rest) = rest else {
                continue;
            };
            schedules.entry(bucket_of(&domain)).or_default().push(PageSchedule { domain, rest, next });
        }
        let mut by_bucket: HashMap<usize, (Vec<PageSchedule>, Vec<(String, Yield)>)> = HashMap::new();
        for (domain, tally) in tallies {
            by_bucket.entry(bucket_of(&domain)).or_default().1.push((domain, tally));
        }
        for (bucket, pages) in schedules {
            by_bucket.entry(bucket).or_default().0 = pages;
        }
        let groups: Vec<(Vec<PageSchedule>, Vec<(String, Yield)>)> = by_bucket.into_values().collect();
        // Every scheduled page's domain was tallied first, so each group has a domain to find its store by.
        on_readers(groups, |(pages, tallies)| buckets.store(&tallies[0].0).note_outcomes(&pages, &tallies))?;
        Ok(counted)
    }
}

/// One numbered outcome file, nothing yet, a number not indexed while later ones are, or an index whose file expired unread.
pub enum Next {
    Wait,
    Missing,
    Gone,
    File { rows: Vec<OutcomeRow>, published_at: Option<i64> },
}

/// A store that keeps its bucket private answers 403, not 404, for a key it does not have.
fn absent(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::FORBIDDEN
}

#[derive(Clone)]
pub struct OutcomeFeed {
    base: String,
    http: reqwest::Client,
}

impl OutcomeFeed {
    /// The newest number the task API has written, from `outcomes/latest.json`.
    pub async fn latest(&self) -> Result<Option<u64>> {
        let found = self.http.get(format!("{}/outcomes/latest.json", self.base)).send().await?;
        if absent(found.status()) {
            return Ok(None);
        }
        if !found.status().is_success() {
            bail!("outcomes/latest.json: HTTP {}", found.status());
        }
        let latest: Value = serde_json::from_slice(&found.bytes().await?).context("outcomes/latest.json that is not JSON")?;
        Ok(latest.get("seq").and_then(Value::as_u64))
    }
}

impl OutcomeFeed {
    pub fn new(base: &str) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(TIMEOUT).build()?;
        Ok(OutcomeFeed { base: base.trim_end_matches('/').to_string(), http })
    }

    /// File number `seq`; a number not published yet means wait, since files are applied in order.
    pub async fn fetch(&self, seq: u64) -> Result<Next> {
        self.fetch_below(seq, None).await
    }

    /// As `fetch`, with the newest number already read when the caller has it.
    pub async fn fetch_below(&self, seq: u64, latest: Option<u64>) -> Result<Next> {
        let index = self.http.get(format!("{}/outcomes/seq/{seq:012}.json", self.base)).send().await?;
        if absent(index.status()) {
            let latest = match latest {
                Some(latest) => Some(latest),
                None => self.latest().await?,
            };
            return Ok(if latest.is_some_and(|latest| latest > seq) { Next::Missing } else { Next::Wait });
        }
        if !index.status().is_success() {
            bail!("outcome index {seq}: HTTP {}", index.status());
        }
        let published_at = index
            .headers()
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| chrono::DateTime::parse_from_rfc2822(v).ok())
            .map(|t| t.timestamp());
        let index: Value = serde_json::from_slice(&index.bytes().await?).context("an outcome index that is not JSON")?;
        let key = index.get("key").and_then(Value::as_str).context("an outcome index without a key")?;
        let file = self.http.get(format!("{}/{}", self.base, key.trim_start_matches('/'))).send().await?;
        if absent(file.status()) {
            return Ok(Next::Gone);
        }
        if !file.status().is_success() {
            bail!("outcome file {key}: HTTP {}", file.status());
        }
        let rows = read_parquet(file.bytes().await?)?;
        let published_at = published_at.or_else(|| rows.iter().map(|r| r.at.div_euclid(1_000_000)).max());
        Ok(Next::File { rows, published_at })
    }
}

//! Pages waiting to go to the task API, a list per domain in the bucket store, and what became of the pages sent.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::bytes::{Regex, RegexBuilder};
use rocksdb::{Direction, IteratorMode, ReadOptions, WriteBatch};
use serde_json::{json, Value};

use crate::allowed::Terms;
use crate::buckets::{meta_key, BucketStore, Changes, LISTED, URL};
use crate::recrawl::Recrawl;
use crate::records;
use crate::robots::Paths;
use crate::urls::{Record, HTTPS, WWW};

/// Pages waiting to be sent, under their domain, in the order they go.
const READY: u8 = b'R';
/// How many pages each domain has waiting.
const READY_COUNT: u8 = b'Q';
/// The lastmod each page was last sent with, and its failures since.
pub(crate) const SENT: u8 = b'P';
/// Retries and re-crawls by the time they join their ready list again.
const WAITING: u8 = b'W';
/// Each page's latest schedule; a waiting entry another schedule replaced is skipped when it comes due.
const NEXT: u8 = b'N';
/// Deadlines left by earlier versions, which resent every page with no outcome a day after it went.
const EXPECTING: u8 = b'X';
/// Each domain's recent outcomes, so the dispatcher sends less where pages do not come back.
const YIELD: u8 = b'Y';
/// The same outcomes fading within hours, so a domain that starts refusing is slowed the same hour.
const YIELD_RECENT: u8 = b'Z';
/// How many distinct pages of each domain were ever sent, for its cap.
const SENT_COUNT: u8 = b'C';
/// The cap each listed domain's stored pages were last queued under; a domain is walked again only when it changes.
const JOINED: u8 = b'J';
/// Keys a domain's new-page lane loses per locked write when it is cleared.
const TRIM_CHUNK: usize = 50_000;
/// Pages queued per locked write while a domain is walked.
const JOIN_CHUNK: usize = 20_000;
/// An outcome counts half as much a day later.
pub const YIELD_HALF_LIFE_S: f64 = 86_400.0;
/// In the recent count, half as much an hour later.
pub const YIELD_RECENT_HALF_LIFE_S: f64 = 3_600.0;
const BACKFILL: &str = "backfill_ready";

/// A sitemap read that moves more than this share of its pages, and more than RESTAMP_MIN, past what they were sent with is re-stamping dates, not changing pages.
pub const RESTAMP_SHARE: f64 = 0.02;
pub const RESTAMP_MIN: usize = 50;
pub const RESTAMP_KEEP: usize = 10;
/// An outcome this much older than the latest send of its page belongs to an earlier send.
const STALE_OUTCOME: u32 = 60;
/// Failed sends in a row after which a page waits the long `Recrawl::gone` interval.
pub const GONE_AFTER: u8 = 2;

/// Listing, search and file pages: nothing to read on them.
static JUNK: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(concat!(
        r"/(tags?|categor(y|ies)|authors?|page/[0-9]+|search|feeds?|rss|amp|wp-json|attachment|print|login|cart|checkout)(/|$)",
        r"|\.(pdf|jpe?g|png|gif|webp|svg|zip|gz|mp[34]|mov|xml|json|css|js)$",
        r"|[?&](page|p|s|q|replytocom)=",
    ))
    .case_insensitive(true)
    .build()
    .expect("a valid pattern")
});

pub fn junk(rest: &[u8]) -> bool {
    JUNK.is_match(rest)
}

pub fn restamped(changed: usize, listed: usize) -> bool {
    changed as f64 > (RESTAMP_MIN as f64).max(RESTAMP_SHARE * listed as f64)
}

/// The address a miner requests, in the scheme and form the site listed it.
pub fn fetchable(rest: &[u8], flags: u8) -> String {
    let scheme = if flags & HTTPS != 0 { "https" } else { "http" };
    format!("{scheme}://{}{}", if flags & WWW != 0 { "www." } else { "" }, String::from_utf8_lossy(rest))
}

/// A domain's ready list is three lanes, by the first byte of the order: refreshes, retries and new pages.
pub const CHANGED: usize = 0;
pub const REQUEUED: usize = 1;
pub const FRESH: usize = 2;
pub const LANES: usize = 3;
/// Who takes what a lane leaves unused: new pages first.
const SPILL: [usize; LANES] = [FRESH, CHANGED, REQUEUED];

/// A page's place in its domain's ready list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Order {
    /// Sent before and listed with a newer lastmod since: newest lastmod first.
    Changed { lastmod: u32 },
    /// A retry or a scheduled re-crawl: longest due first.
    Requeued { due: u32 },
    /// Never sent: newest found first, then newest lastmod.
    Fresh { first_seen: u32, lastmod: u32 },
}

impl Order {
    pub fn lane(self) -> usize {
        match self {
            Order::Changed { .. } => CHANGED,
            Order::Requeued { .. } => REQUEUED,
            Order::Fresh { .. } => FRESH,
        }
    }

    fn bytes(self) -> [u8; 9] {
        let (class, first, second) = match self {
            Order::Changed { lastmod } => (0, !lastmod, 0),
            Order::Requeued { due } => (1, due, 0),
            Order::Fresh { first_seen, lastmod } => (2, !first_seen, !lastmod),
        };
        let mut out = [class; 9];
        out[1..5].copy_from_slice(&first.to_be_bytes());
        out[5..].copy_from_slice(&second.to_be_bytes());
        out
    }

    fn parse(bytes: &[u8]) -> Option<Order> {
        let word = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
        match bytes.first()? {
            _ if bytes.len() < 9 => None,
            0 => Some(Order::Changed { lastmod: !word(1) }),
            1 => Some(Order::Requeued { due: word(1) }),
            2 => Some(Order::Fresh { first_seen: !word(1), lastmod: !word(5) }),
            _ => None,
        }
    }
}

/// One page on a ready list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub domain: String,
    pub rest: Vec<u8>,
    pub order: Order,
}

impl Entry {
    pub fn key(&self) -> Vec<u8> {
        [&[READY], self.domain.as_bytes(), b"\0", &self.order.bytes(), &self.rest].concat()
    }

    pub fn parse(key: &[u8]) -> Option<Entry> {
        let (domain, after) = split_key(key.strip_prefix(&[READY])?)?;
        Some(Entry { domain: String::from_utf8(domain.to_vec()).ok()?, rest: after.get(9..)?.to_vec(), order: Order::parse(after)? })
    }

    fn url_key(&self) -> Vec<u8> {
        prefixed(URL, self.domain.as_bytes(), &self.rest)
    }

    fn sent_key(&self) -> Vec<u8> {
        prefixed(SENT, self.domain.as_bytes(), &self.rest)
    }
}

/// What a page was last sent with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sent {
    pub lastmod: u32,
    /// Sends in a row that failed, since it was last sent for a new lastmod.
    pub failures: u8,
    /// Back on its ready list for a retry or a re-crawl.
    pub requeued: bool,
}

impl Sent {
    /// Failures share the last byte with the requeued flag; the byte before held a count that also took in outcomes that never came, and is left at zero.
    fn pack(self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[..4].copy_from_slice(&self.lastmod.to_le_bytes());
        out[5] = u8::from(self.requeued) | self.failures.min(127) << 1;
        out
    }

    fn unpack(data: &[u8]) -> Option<Sent> {
        (data.len() == 6).then(|| Sent { lastmod: u32::from_le_bytes(data[..4].try_into().unwrap()), failures: data[5] >> 1, requeued: data[5] & 1 != 0 })
    }
}

/// Why a page goes out now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Fresh,
    Changed,
    Requeued,
}

/// Whether a page on a ready list still needs sending, and why; None for an entry that went stale.
pub fn reason(record: &Record, sent: Option<&Sent>) -> Option<Reason> {
    if record.pushed_at == 0 {
        return Some(Reason::Fresh);
    }
    match sent {
        None => Some(Reason::Changed),
        Some(sent) if record.lastmod > sent.lastmod => Some(Reason::Changed),
        Some(sent) if sent.requeued => Some(Reason::Requeued),
        Some(_) => None,
    }
}

/// A page taken off its ready list to be sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pick {
    pub entry: Entry,
    pub url: String,
    pub lastmod: u32,
    pub reason: Reason,
    pub failures: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Published,
    Unchanged,
    Failed,
    Dropped,
}

impl Outcome {
    pub fn parse(value: &str) -> Option<Outcome> {
        match value {
            "published" => Some(Outcome::Published),
            "unchanged" => Some(Outcome::Unchanged),
            "failed" => Some(Outcome::Failed),
            "dropped" => Some(Outcome::Dropped),
            _ => None,
        }
    }
}

/// Why a sent page goes out again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// A page without a lastmod, once its domain's interval has passed since it was sent.
    Recrawl,
    /// A page whose send failed.
    Failed,
    /// A page whose task was lost or taken back before it reached the site.
    Dropped,
    /// A page that failed on sends in a row, after the long wait.
    Gone,
}

/// When a page goes out again, and the send (re-crawl, gone) or outcome (failed, dropped) that set it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scheduled {
    pub why: Why,
    pub due: u32,
    pub at: u32,
}

impl Scheduled {
    fn pack(self) -> [u8; 9] {
        let mut out = [0u8; 9];
        out[0] = match self.why {
            Why::Recrawl => b'r',
            Why::Failed => b'f',
            Why::Dropped => b'd',
            Why::Gone => b'g',
        };
        out[1..5].copy_from_slice(&self.due.to_be_bytes());
        out[5..].copy_from_slice(&self.at.to_be_bytes());
        out
    }

    fn unpack(raw: &[u8]) -> Option<Scheduled> {
        let word = |at: usize| u32::from_be_bytes(raw[at..at + 4].try_into().unwrap());
        let why = match raw.first()? {
            _ if raw.len() != 9 => return None,
            b'r' => Why::Recrawl,
            b'f' => Why::Failed,
            b'd' => Why::Dropped,
            b'g' => Why::Gone,
            _ => return None,
        };
        Some(Scheduled { why, due: word(1), at: word(5) })
    }

    /// Whether the page went out again after what set this schedule, which then says nothing about it.
    fn outdated(self, pushed_at: u32) -> bool {
        match self.why {
            Why::Recrawl | Why::Gone => pushed_at != self.at,
            Why::Failed | Why::Dropped => pushed_at > self.at.saturating_add(STALE_OUTCOME),
        }
    }
}

/// A page to send again, as its store keys it, written without reading the page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageSchedule {
    pub domain: String,
    pub rest: Vec<u8>,
    pub next: Scheduled,
}

/// A page an earlier sender sent, with the lastmod it had then.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentPage {
    pub domain: String,
    pub rest: Vec<u8>,
    pub lastmod: u32,
    pub at: u32,
}

/// A domain's published and failed outcomes, each decayed by its age.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Yield {
    pub ok: f64,
    pub failed: f64,
    pub at: u32,
}

impl Yield {
    /// The counts as they stand at `now`.
    pub fn at(self, now: u32) -> Yield {
        self.faded(now, YIELD_HALF_LIFE_S)
    }

    /// The recent counts as they stand at `now`.
    pub fn recent_at(self, now: u32) -> Yield {
        self.faded(now, YIELD_RECENT_HALF_LIFE_S)
    }

    fn faded(self, now: u32, half_life: f64) -> Yield {
        let kept = 0.5f64.powf(now.saturating_sub(self.at) as f64 / half_life);
        Yield { ok: self.ok * kept, failed: self.failed * kept, at: self.at.max(now) }
    }

    pub fn judged(self) -> f64 {
        self.ok + self.failed
    }

    fn pack(self) -> Vec<u8> {
        [(self.ok as f32).to_be_bytes(), (self.failed as f32).to_be_bytes(), self.at.to_be_bytes()].concat()
    }

    fn unpack(raw: &[u8]) -> Option<Yield> {
        let word = |at: usize| raw.get(at..at + 4).and_then(|b| b.try_into().ok());
        Some(Yield { ok: f32::from_be_bytes(word(0)?).into(), failed: f32::from_be_bytes(word(4)?).into(), at: u32::from_be_bytes(word(8)?) })
    }
}

/// What a dropped domain held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    pub known: bool,
    pub ready: u64,
    pub urls: i64,
}

/// What walking one listed domain's stored pages found and queued.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Joined {
    pub scanned: u64,
    /// Pages of the domain sent at some point, which count against its cap.
    pub sent: u64,
    pub queued: u64,
    /// New-page entries cleared first, for a domain with a cap.
    pub cleared: u64,
    /// Stopped before the walk ended; the domain is walked again next time.
    pub stopped: bool,
}

/// What one look at the due schedules did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Requeued {
    /// Put back on their ready lists.
    pub moved: usize,
    /// Failed again, so set to wait the long interval.
    pub gone: usize,
}

/// Where a store's one-time walk for the ready lists is, and what it found.
#[derive(Clone, Debug, Default)]
pub struct Backfill {
    pub started: u32,
    pub scanned: u64,
    pub added: u64,
    pub done: bool,
    after: Option<Vec<u8>>,
    domain: Option<DomainScan>,
}

#[derive(Clone, Debug)]
struct DomainScan {
    domain: Vec<u8>,
    listed: usize,
    changed: Vec<(Vec<u8>, u32)>,
}

impl BucketStore {
    /// Domains with pages waiting, and how many each has.
    pub fn ready_domains(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&[READY_COUNT], Direction::Forward), bounded(&[READY_COUNT])) {
            let (key, value) = item?;
            out.push((String::from_utf8_lossy(&key[1..]).into_owned(), count_value(Some(&value))));
        }
        Ok(out)
    }

    pub fn ready_count(&self, domain: &str) -> Result<u64> {
        Ok(count_value(self.db.get_pinned(count_key(domain))?.as_deref()))
    }

    /// Domains whose ready lists grew since the last call, and by how much; the first call starts the count.
    pub fn take_noticed(&self) -> HashMap<String, u64> {
        self.noticed.lock().unwrap_or_else(|e| e.into_inner()).replace(HashMap::new()).unwrap_or_default()
    }

    /// The first pages of a domain's ready list, in the order they go.
    pub fn ready_list(&self, domain: &str, limit: usize) -> Result<Vec<Entry>> {
        Ok(self.ready_keys(domain, None, limit)?.iter().filter_map(|key| Entry::parse(key)).collect())
    }

    pub fn sent(&self, domain: &str, rest: &[u8]) -> Result<Option<Sent>> {
        Ok(self.db.get_pinned(prefixed(SENT, domain.as_bytes(), rest))?.as_deref().and_then(Sent::unpack))
    }

    /// Pages waiting to join their ready lists, by when.
    pub fn waiting(&self) -> Result<Vec<(u32, String, Vec<u8>)>> {
        let mut out = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&[WAITING], Direction::Forward), bounded(&[WAITING])) {
            let (key, _) = item?;
            out.extend(timed_entry(&key).map(|(due, domain, rest)| (due, String::from_utf8_lossy(domain).into_owned(), rest.to_vec())));
        }
        Ok(out)
    }

    /// Pages of a domain that still need sending: each lane up to its quota, then lanes with pages left fill what the others could not; entries gone stale are dropped on the way.
    pub fn take_ready(&self, domain: &str, mut quotas: [usize; LANES]) -> Result<Vec<Pick>> {
        let want: usize = quotas.iter().sum();
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let mut taking = Taking { robots, ..Taking::default() };
        // A domain at its cap sends no more new pages; refreshes and retries of pages it sent go on.
        let room = self.fresh_room(domain)?.map(|room| room.min(usize::MAX as u64) as usize);
        if let Some(room) = room {
            quotas[FRESH] = quotas[FRESH].min(room);
        }
        for (lane, quota) in quotas.into_iter().enumerate() {
            self.take_lane(domain, lane, quota, &mut taking)?;
        }
        for lane in SPILL {
            if taking.picks.len() >= want {
                break;
            }
            let mut more = want - taking.picks.len();
            if let (FRESH, Some(room)) = (lane, room) {
                more = more.min(room.saturating_sub(taking.picks.iter().filter(|p| p.entry.order.lane() == FRESH).count()));
            }
            self.take_lane(domain, lane, more, &mut taking)?;
        }
        if !taking.stale.is_empty() {
            let _guard = self.lock_ready();
            let mut batch = WriteBatch::default();
            let mut counts = HashMap::new();
            self.remove_ready(&mut batch, &taking.stale, &mut counts)?;
            self.write_counts(&mut batch, &counts)?;
            self.db.write(batch)?;
        }
        Ok(taking.picks)
    }

    /// Up to `want` more pages from one lane, best first, carrying on from where the lane was last read.
    fn take_lane(&self, domain: &str, lane: usize, want: usize, taking: &mut Taking) -> Result<()> {
        let goal = taking.picks.len() + want;
        while taking.picks.len() < goal && !taking.drained[lane] {
            let asked = (goal - taking.picks.len()).clamp(64, 4096);
            let started = taking.after[lane].clone();
            let keys = self.lane_keys(domain, lane, started.as_deref(), asked)?;
            let read = keys.len();
            let mut entries = Vec::with_capacity(read);
            for key in keys {
                match Entry::parse(&key) {
                    Some(entry) => entries.push((key, entry)),
                    None => taking.stale.push(key),
                }
            }
            let records = self.records(entries.iter().map(|(_, e)| e.url_key()))?;
            let sents = self.sents(entries.iter().map(|(_, e)| e.sent_key()))?;
            let mut stopped_early = false;
            for (((key, entry), record), sent) in entries.into_iter().zip(records).zip(sents) {
                if taking.picks.len() >= goal {
                    stopped_early = true;
                    break;
                }
                taking.after[lane] = Some(key.clone());
                let allowed = taking.robots.allows_page(domain, &entry.rest);
                let due = record.filter(|_| allowed).and_then(|r| reason(&r, sent.as_ref()).map(|why| (r, why)));
                // A page listed twice, under an older and a newer lastmod, goes once.
                let Some((record, why)) = due.filter(|_| taking.picked.insert(entry.rest.clone())) else {
                    taking.stale.push(key);
                    continue;
                };
                let failures = sent.map_or(0, |s| s.failures);
                taking.picks.push(Pick { url: fetchable(&entry.rest, record.flags), lastmod: record.lastmod, reason: why, failures, entry });
            }
            // A read that moved nothing forward would only repeat itself.
            if (read < asked && !stopped_early) || taking.after[lane] == started {
                taking.drained[lane] = true;
            }
        }
        Ok(())
    }

    /// The task API took these pages: stamp them sent, with the lastmod they went with, and take them off their lists.
    pub fn mark_sent(&self, picks: &[Pick], at: u32, recrawl: &Recrawl) -> Result<()> {
        if picks.is_empty() {
            return Ok(());
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        let mut counts = HashMap::new();
        let keys: Vec<Vec<u8>> = picks.iter().map(|p| p.entry.key()).collect();
        self.remove_ready(&mut batch, &keys, &mut counts)?;
        let records = self.records(picks.iter().map(|p| p.entry.url_key()))?;
        let mut moved = Vec::new();
        for (pick, record) in picks.iter().zip(records) {
            if let Some(mut record) = record {
                record.pushed_at = at;
                batch.put(pick.entry.url_key(), record.pack());
                if record.lastmod > pick.lastmod {
                    moved.push(Entry { order: Order::Changed { lastmod: record.lastmod }, ..pick.entry.clone() });
                }
                // No lastmod will ever say such a page changed, so it goes again on its domain's interval.
                if record.lastmod == 0 {
                    let next = Scheduled { why: Why::Recrawl, due: at.saturating_add(recrawl.interval(&pick.entry.domain)), at };
                    put_schedule(&mut batch, pick.entry.domain.as_bytes(), &pick.entry.rest, next);
                }
            }
            let failures = if pick.reason == Reason::Requeued { pick.failures } else { 0 };
            batch.put(pick.entry.sent_key(), Sent { lastmod: pick.lastmod, failures, requeued: false }.pack());
        }
        // A lastmod that moved while the page was in flight is a change the API has not seen.
        self.add_ready(&mut batch, moved, &mut counts)?;
        self.write_counts(&mut batch, &counts)?;
        let mut first_sends: HashMap<&str, u64> = HashMap::new();
        for pick in picks.iter().filter(|p| p.reason == Reason::Fresh) {
            *first_sends.entry(pick.entry.domain.as_str()).or_default() += 1;
        }
        for (domain, more) in first_sends {
            // A domain not counted yet is counted when the list walker reaches it.
            let Some(sent) = self.sent_count(domain)? else {
                continue;
            };
            batch.put(count_key_of(SENT_COUNT, domain), (sent + more).to_le_bytes());
            if self.terms(domain).send_cap.is_some_and(|cap| sent < cap && sent + more >= cap) {
                self.capped.lock().unwrap_or_else(|e| e.into_inner()).insert(domain.to_string());
            }
        }
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(())
    }

    /// Distinct pages of the domain ever sent, once the list walker counted them.
    pub fn sent_count(&self, domain: &str) -> Result<Option<u64>> {
        Ok(self.db.get_pinned(count_key_of(SENT_COUNT, domain))?.and_then(|raw| raw.as_ref().try_into().ok().map(u64::from_le_bytes)))
    }

    /// New pages the domain may still send under its cap; None when it has no cap or was not counted yet.
    pub fn fresh_room(&self, domain: &str) -> Result<Option<u64>> {
        let Some(cap) = self.terms(domain).send_cap else {
            return Ok(None);
        };
        Ok(self.sent_count(domain)?.map(|sent| cap.saturating_sub(sent)))
    }

    /// The cap the domain's stored pages were last queued under: None if never walked, Some(None) if walked without a cap.
    pub fn joined(&self, domain: &str) -> Result<Option<Option<u64>>> {
        let raw = self.db.get_pinned(count_key_of(JOINED, domain))?;
        Ok(raw.and_then(|raw| raw.as_ref().try_into().ok().map(u64::from_le_bytes)).map(|cap| (cap != u64::MAX).then_some(cap)))
    }

    /// Domains that reached their cap since the last call.
    pub fn take_capped(&self) -> Vec<String> {
        self.capped.lock().unwrap_or_else(|e| e.into_inner()).drain().collect()
    }

    /// Empties the domain's new-page lane, a locked chunk at a time with exact counts; returns how many entries went.
    pub fn clear_fresh(&self, domain: &str) -> Result<u64> {
        let prefix = [&[READY], domain.as_bytes(), b"\0", &[FRESH as u8]].concat();
        let mut cleared = 0;
        loop {
            let _guard = self.lock_ready();
            let keys = self.keys_under(&prefix, None, TRIM_CHUNK)?;
            let (Some(first), Some(last)) = (keys.first(), keys.last()) else {
                break;
            };
            let mut batch = WriteBatch::default();
            // Every key between the first and the last was just read under the lock, so the range holds exactly these.
            batch.delete_range(first.as_slice(), [last.as_slice(), b"\0"].concat().as_slice());
            self.write_counts(&mut batch, &HashMap::from([(domain.to_string(), -(keys.len() as i64))]))?;
            self.db.write(batch)?;
            cleared += keys.len() as u64;
            if keys.len() < TRIM_CHUNK {
                break;
            }
        }
        Ok(cleared)
    }

    /// Queue the pages a listed domain stored but never sent, newest first, as many as its cap leaves; counts its sent pages for the cap.
    pub fn join_domain(&self, domain: &str, terms: Terms, started: u32, pace: &mut dyn FnMut(u64) -> bool) -> Result<Joined> {
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(URL, domain.as_bytes(), b"");
        let mut joined = Joined::default();
        let mut streamed: Vec<Entry> = Vec::new();
        // With a cap or a backlog only the newest pages go: a heap of the oldest kept, never more than the bound itself.
        let bound = terms.send_cap.or(terms.backlog);
        let mut newest: BinaryHeap<Reverse<(u32, u32, Vec<u8>)>> = BinaryHeap::new();
        let mut since_pace = 0u64;
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)) {
            let (key, raw) = item?;
            joined.scanned += 1;
            since_pace += 1;
            if since_pace >= 10_000 {
                if !pace(since_pace) {
                    joined.stopped = true;
                    return Ok(joined);
                }
                since_pace = 0;
            }
            let Some(record) = Record::unpack(&raw) else {
                continue;
            };
            if record.pushed_at != 0 {
                joined.sent += 1;
                continue;
            }
            let rest = &key[prefix.len()..];
            // Pages first seen after the walk began went on their list when found.
            if record.first_seen >= started || junk(rest) || !robots.allows_page(domain, rest) {
                continue;
            }
            match bound {
                None => {
                    let order = Order::Fresh { first_seen: record.first_seen, lastmod: record.lastmod };
                    streamed.push(Entry { domain: domain.to_string(), rest: rest.to_vec(), order });
                    if streamed.len() >= JOIN_CHUNK {
                        joined.queued += self.queue_entries(std::mem::take(&mut streamed))? as u64;
                    }
                }
                Some(most) => {
                    newest.push(Reverse((record.first_seen, record.lastmod, rest.to_vec())));
                    if newest.len() as u64 > most {
                        newest.pop();
                    }
                }
            }
        }
        if !pace(since_pace) {
            joined.stopped = true;
            return Ok(joined);
        }
        if let Some(most) = bound {
            let room = match terms.send_cap {
                Some(cap) => {
                    joined.cleared = self.clear_fresh(domain)?;
                    cap.saturating_sub(joined.sent)
                }
                None => most,
            };
            let mut kept: Vec<(u32, u32, Vec<u8>)> = newest.into_iter().map(|Reverse(page)| page).collect();
            kept.sort_unstable_by(|a, b| b.cmp(a));
            kept.truncate(room as usize);
            for chunk in kept.chunks(JOIN_CHUNK) {
                let entries = chunk.iter().map(|(first_seen, lastmod, rest)| Entry { domain: domain.to_string(), rest: rest.clone(), order: Order::Fresh { first_seen: *first_seen, lastmod: *lastmod } });
                joined.queued += self.queue_entries(entries.collect())? as u64;
            }
        } else {
            joined.queued += self.queue_entries(streamed)? as u64;
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        batch.put(count_key_of(SENT_COUNT, domain), joined.sent.to_le_bytes());
        batch.put(count_key_of(JOINED, domain), terms.send_cap.unwrap_or(u64::MAX).to_le_bytes());
        self.db.write(batch)?;
        Ok(joined)
    }

    /// Counts the pages of a domain already sending and applies its cap, without walking its stored pages: their stock is queued already.
    pub fn cap_domain(&self, domain: &str, terms: Terms, pace: &mut dyn FnMut(u64) -> bool) -> Result<Joined> {
        let mut joined = Joined::default();
        // Without a cap there is nothing to apply; a cap given later walks the domain again, and counts it then.
        if terms.send_cap.is_none() {
            let _guard = self.lock_ready();
            self.db.put(count_key_of(JOINED, domain), u64::MAX.to_le_bytes())?;
            return Ok(joined);
        }
        let prefix = prefixed(SENT, domain.as_bytes(), b"");
        let mut since_pace = 0u64;
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)) {
            item?;
            joined.sent += 1;
            since_pace += 1;
            if since_pace >= 10_000 {
                if !pace(since_pace) {
                    joined.stopped = true;
                    return Ok(joined);
                }
                since_pace = 0;
            }
        }
        if !pace(since_pace) {
            joined.stopped = true;
            return Ok(joined);
        }
        joined.scanned = joined.sent;
        {
            let _guard = self.lock_ready();
            let mut batch = WriteBatch::default();
            batch.put(count_key_of(SENT_COUNT, domain), joined.sent.to_le_bytes());
            batch.put(count_key_of(JOINED, domain), terms.send_cap.unwrap_or(u64::MAX).to_le_bytes());
            self.db.write(batch)?;
        }
        if terms.send_cap.is_some_and(|cap| joined.sent >= cap) {
            joined.cleared = self.clear_fresh(domain)?;
        }
        Ok(joined)
    }

    fn queue_entries(&self, entries: Vec<Entry>) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        let mut counts = HashMap::new();
        let added = self.add_ready(&mut batch, entries, &mut counts)?;
        self.write_counts(&mut batch, &counts)?;
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(added)
    }

    /// Deletes the deadlines earlier versions kept for every sent page; whether there were any.
    pub fn forget_expecting(&self) -> Result<bool> {
        let (from, to) = ([EXPECTING], [EXPECTING + 1]);
        if self.db.iterator_opt(IteratorMode::From(&from, Direction::Forward), bounded(&from)).next().transpose()?.is_none() {
            return Ok(false);
        }
        let mut changes = Changes::default();
        changes.delete_range(&from, &to);
        self.write(changes)?;
        self.db.delete_file_in_range(from, to)?;
        Ok(true)
    }

    /// Put pages whose schedule came due back on their ready lists: a page that failed again waits the long interval instead.
    pub fn requeue_due(&self, now: u32, limit: usize, recrawl: &Recrawl) -> Result<Requeued> {
        let mut options = ReadOptions::default();
        options.set_iterate_upper_bound([&[WAITING][..], &now.saturating_add(1).to_be_bytes()].concat());
        let mut keys = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&[WAITING], Direction::Forward), options).take(limit) {
            keys.push(item?.0.to_vec());
        }
        let mut done = Requeued::default();
        if keys.is_empty() {
            return Ok(done);
        }
        let _guard = self.lock_ready();
        let parsed: Vec<(u32, &[u8], &[u8])> = keys.iter().filter_map(|key| timed_entry(key)).collect();
        let records = self.records(parsed.iter().map(|(_, domain, rest)| prefixed(URL, domain, rest)))?;
        let sents = self.sents(parsed.iter().map(|(_, domain, rest)| prefixed(SENT, domain, rest)))?;
        let schedules = self.schedules(parsed.iter().map(|(_, domain, rest)| prefixed(NEXT, domain, rest)))?;
        let mut batch = WriteBatch::default();
        // Deleted first, so a new schedule that lands on the same second survives the batch.
        for key in &keys {
            batch.delete(key);
        }
        let mut entries = Vec::new();
        for ((((due, domain, rest), record), sent), next) in parsed.iter().zip(records).zip(sents).zip(schedules) {
            if next.is_some_and(|next| next.due != *due) {
                continue;
            }
            if next.is_some() {
                batch.delete(prefixed(NEXT, domain, rest));
            }
            let Some(record) = record else {
                continue;
            };
            let mut sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
            // Entries without a schedule were written by earlier versions and go back as they are.
            match next {
                Some(next) if next.outdated(record.pushed_at) => continue,
                Some(Scheduled { why: Why::Recrawl, .. }) => sent.failures = 0,
                Some(Scheduled { why: Why::Failed, at, .. }) => {
                    sent.failures = sent.failures.saturating_add(1);
                    if sent.failures >= GONE_AFTER {
                        put_schedule(&mut batch, domain, rest, Scheduled { why: Why::Gone, due: at.saturating_add(recrawl.gone), at: record.pushed_at });
                        batch.put(prefixed(SENT, domain, rest), sent.pack());
                        done.gone += 1;
                        continue;
                    }
                }
                Some(Scheduled { why: Why::Dropped | Why::Gone, .. }) | None => {}
            }
            sent.requeued = true;
            batch.put(prefixed(SENT, domain, rest), sent.pack());
            entries.push(Entry { domain: String::from_utf8_lossy(domain).into_owned(), rest: rest.to_vec(), order: Order::Requeued { due: *due } });
        }
        let mut counts = HashMap::new();
        done.moved = self.add_ready(&mut batch, entries, &mut counts)?;
        self.write_counts(&mut batch, &counts)?;
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(done)
    }

    /// Keep one outcome file's part for this store: schedules for failed and dropped pages, written without reading a page, and each domain's yield.
    pub fn note_outcomes(&self, schedules: &[PageSchedule], tallies: &[(String, Yield)]) -> Result<()> {
        let mut batch = WriteBatch::default();
        for (domain, tally) in tallies.iter().filter(|(_, tally)| tally.judged() > 0.0) {
            let kept = self.domain_yield(domain)?.unwrap_or_default().at(tally.at);
            batch.put(yield_key(domain), Yield { ok: kept.ok + tally.ok, failed: kept.failed + tally.failed, at: kept.at }.pack());
            let recent = self.domain_yield_recent(domain)?.unwrap_or_default().recent_at(tally.at);
            batch.put(count_key_of(YIELD_RECENT, domain), Yield { ok: recent.ok + tally.ok, failed: recent.failed + tally.failed, at: recent.at }.pack());
        }
        for PageSchedule { domain, rest, next } in schedules {
            put_schedule(&mut batch, domain.as_bytes(), rest, *next);
        }
        let _guard = self.lock_ready();
        self.db.write(batch)?;
        Ok(())
    }

    /// The page's latest schedule, if one is waiting.
    pub fn next_send(&self, domain: &str, rest: &[u8]) -> Result<Option<Scheduled>> {
        Ok(self.db.get_pinned(prefixed(NEXT, domain.as_bytes(), rest))?.as_deref().and_then(Scheduled::unpack))
    }

    /// Deletes every page kept for `domain` and marks it excluded so no visit lists them again; with `delete` false only counts.
    pub fn drop_domain(&self, domain: &str, reason: &str, delete: bool) -> Result<Dropped> {
        let Some(mut record) = self.domain(domain)? else {
            return Ok(Dropped::default());
        };
        let found = Dropped { known: true, ready: self.ready_count(domain)?, urls: records::int(record.get("urls")).unwrap_or(0) };
        if !delete {
            return Ok(found);
        }
        let ranges: Vec<(Vec<u8>, Vec<u8>)> = [URL, READY, SENT, LISTED, NEXT]
            .into_iter()
            .map(|kind| ([&[kind], domain.as_bytes(), b"\0"].concat(), [&[kind], domain.as_bytes(), b"\x01"].concat()))
            .collect();
        let _guard = self.lock_ready();
        let mut changes = Changes::default();
        for (from, to) in &ranges {
            changes.delete_range(from, to);
        }
        changes.delete(&count_key(domain));
        changes.delete(&yield_key(domain));
        changes.delete(&count_key_of(YIELD_RECENT, domain));
        changes.delete(&count_key_of(SENT_COUNT, domain));
        changes.delete(&count_key_of(JOINED, domain));
        record.insert("state".into(), "excluded".into());
        record.insert("reason".into(), reason.into());
        record.insert("due".into(), Value::Null);
        record.insert("urls".into(), 0.into());
        // The pages go and the domain is marked excluded in one write, so a crash never leaves it to be listed again.
        changes.domain(domain, &record);
        self.write(changes)?;
        for (from, to) in &ranges {
            self.db.delete_file_in_range(from, to)?;
        }
        Ok(found)
    }

    /// The domain's outcomes as last counted, not yet decayed to now.
    pub fn domain_yield(&self, domain: &str) -> Result<Option<Yield>> {
        Ok(self.db.get(yield_key(domain))?.as_deref().and_then(Yield::unpack))
    }

    pub fn domain_yield_recent(&self, domain: &str) -> Result<Option<Yield>> {
        Ok(self.db.get(count_key_of(YIELD_RECENT, domain))?.as_deref().and_then(Yield::unpack))
    }

    /// Stamp pages another sender already sent; returns how many the store knows.
    pub fn import_sent(&self, rows: &[SentPage]) -> Result<usize> {
        let _guard = self.lock_ready();
        let records = self.records(rows.iter().map(|row| prefixed(URL, row.domain.as_bytes(), &row.rest)))?;
        let sents = self.sents(rows.iter().map(|row| prefixed(SENT, row.domain.as_bytes(), &row.rest)))?;
        let mut batch = WriteBatch::default();
        let mut known = 0;
        for ((SentPage { domain, rest, lastmod, at }, record), sent) in rows.iter().zip(records).zip(sents) {
            let Some(mut record) = record else {
                continue;
            };
            known += 1;
            if record.pushed_at == 0 {
                record.pushed_at = *at;
                batch.put(prefixed(URL, domain.as_bytes(), rest), record.pack());
            }
            if sent.is_none() {
                batch.put(prefixed(SENT, domain.as_bytes(), rest), Sent { lastmod: *lastmod, ..Sent::default() }.pack());
            }
        }
        self.db.write(batch)?;
        Ok(known)
    }

    /// Where this store's walk for the ready lists stands, starting one now if none ever did.
    pub fn backfill_state(&self, now: u32) -> Result<Backfill> {
        let Some(saved) = self.meta(BACKFILL)? else {
            return Ok(Backfill { started: now, ..Backfill::default() });
        };
        let after = saved.get("after").and_then(Value::as_str).and_then(decode_hex);
        Ok(Backfill {
            started: saved.get("started").and_then(Value::as_u64).unwrap_or(now.into()) as u32,
            scanned: saved.get("scanned").and_then(Value::as_u64).unwrap_or(0),
            added: saved.get("added").and_then(Value::as_u64).unwrap_or(0),
            done: saved.get("done").and_then(Value::as_bool).unwrap_or(false),
            after,
            domain: None,
        })
    }

    /// Read up to `limit` more URL records and put every page never sent, or changed since sent, on its ready list.
    pub fn backfill_step(&self, state: &mut Backfill, limit: usize) -> Result<()> {
        if state.done {
            return Ok(());
        }
        let upper = vec![URL + 1];
        let mut options = ReadOptions::default();
        options.set_iterate_upper_bound(upper);
        let start = state.after.clone().map_or_else(|| vec![URL], |mut key| {
            key.push(0);
            key
        });
        let mut entries = Vec::new();
        let (mut read, mut skipped) = (0, false);
        for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), options).take(limit) {
            let (key, raw) = item?;
            read += 1;
            let Some((domain, rest)) = split_key(&key[1..]) else {
                continue;
            };
            if !self.allows(&String::from_utf8_lossy(domain)) {
                if let Some(done) = state.domain.take() {
                    entries.extend(changed_entries(done));
                }
                // No page path holds 0xFF, so the walk goes on from the next domain.
                state.after = Some([&[URL], domain, b"\0\xff"].concat());
                skipped = true;
                break;
            }
            if state.domain.as_ref().is_none_or(|d| d.domain != domain) {
                if let Some(done) = state.domain.take() {
                    entries.extend(changed_entries(done));
                }
                state.domain = Some(DomainScan { domain: domain.to_vec(), listed: 0, changed: Vec::new() });
            }
            let scan = state.domain.as_mut().expect("set above");
            scan.listed += 1;
            state.after = Some(key.to_vec());
            let Some(record) = Record::unpack(&raw) else {
                continue;
            };
            if record.pushed_at == 0 {
                // Pages first seen since the walk began went on their lists when they were found.
                if record.first_seen < state.started && !junk(rest) {
                    let order = Order::Fresh { first_seen: record.first_seen, lastmod: record.lastmod };
                    entries.push(Entry { domain: String::from_utf8_lossy(domain).into_owned(), rest: rest.to_vec(), order });
                }
            } else {
                let sent = self.db.get_pinned(prefixed(SENT, domain, rest))?.as_deref().and_then(Sent::unpack);
                if sent.is_none_or(|s| record.lastmod > s.lastmod) {
                    scan.changed.push((rest.to_vec(), record.lastmod));
                }
            }
        }
        state.scanned += read as u64;
        if read < limit && !skipped {
            state.done = true;
            if let Some(done) = state.domain.take() {
                entries.extend(changed_entries(done));
            }
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        let mut counts = HashMap::new();
        state.added += self.add_ready(&mut batch, entries, &mut counts)? as u64;
        self.write_counts(&mut batch, &counts)?;
        // A walk stopped inside a domain starts that domain again, so its re-stamp count sees all of it.
        let resume = state.domain.as_ref().map(|d| [&[URL], &d.domain[..]].concat()).or(state.after.clone());
        let saved = json!({
            "started": state.started,
            "scanned": state.scanned,
            "added": state.added,
            "done": state.done,
            "after": resume.map(|k| hex(&k)),
        });
        batch.put(meta_key(BACKFILL), serde_json::to_vec(&saved)?);
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(())
    }

    /// Ready entries for a sitemap read: pages new to the store, and sent pages whose lastmod moved past what they were sent with.
    pub(crate) fn ready_from_listing(
        &self,
        batch: &mut WriteBatch,
        fresh: &[(Vec<u8>, Record)],
        moved: &[(Vec<u8>, u32)],
        listed: usize,
    ) -> Result<(usize, HashMap<String, i64>)> {
        let mut entries = Vec::new();
        let mut room: HashMap<Vec<u8>, bool> = HashMap::new();
        for (key, record) in fresh {
            if let Some((domain, rest)) = split_key(&key[1..]).filter(|(_, rest)| !junk(rest)) {
                // A domain at its cap queues no more new pages.
                let open = match room.get(domain) {
                    Some(&open) => open,
                    None => {
                        let open = self.fresh_room(&String::from_utf8_lossy(domain))? != Some(0);
                        room.insert(domain.to_vec(), open);
                        open
                    }
                };
                if !open {
                    continue;
                }
                let order = Order::Fresh { first_seen: record.first_seen, lastmod: record.lastmod };
                entries.push(Entry { domain: String::from_utf8_lossy(domain).into_owned(), rest: rest.to_vec(), order });
            }
        }
        if !moved.is_empty() {
            let sents = self.sents(moved.iter().map(|(key, _)| [&[SENT], &key[1..]].concat()))?;
            let changed: Vec<(Vec<u8>, u32)> = moved
                .iter()
                .zip(sents)
                .filter(|((_, lastmod), sent)| sent.is_none_or(|s| *lastmod > s.lastmod))
                .map(|((key, lastmod), _)| (key[1..].to_vec(), *lastmod))
                .collect();
            if let Some((domain, _)) = changed.first().and_then(|(key, _)| split_key(key)) {
                let domain = domain.to_vec();
                let changed = changed.iter().filter_map(|(key, lastmod)| Some((split_key(key)?.1.to_vec(), *lastmod))).collect();
                entries.extend(changed_entries(DomainScan { domain, listed, changed }));
            }
        }
        let mut counts = HashMap::new();
        let added = self.add_ready(batch, entries, &mut counts)?;
        self.write_counts(batch, &counts)?;
        Ok((added, counts))
    }

    pub(crate) fn notice(&self, counts: &HashMap<String, i64>) {
        let mut noticed = self.noticed.lock().unwrap_or_else(|e| e.into_inner());
        let Some(noticed) = noticed.as_mut() else {
            return;
        };
        for (domain, &added) in counts.iter().filter(|(_, &n)| n > 0) {
            *noticed.entry(domain.clone()).or_default() += added as u64;
        }
    }

    /// Entries not on their lists yet go on; the caller holds the ready lock and writes the batch.
    fn add_ready(&self, batch: &mut WriteBatch, mut entries: Vec<Entry>, counts: &mut HashMap<String, i64>) -> Result<usize> {
        entries.retain(|entry| self.allows(&entry.domain));
        let mut keys: Vec<(Vec<u8>, String)> = entries.drain(..).map(|e| (e.key(), e.domain)).collect();
        keys.sort_unstable();
        keys.dedup_by(|a, b| a.0 == b.0);
        let cf = self.db.cf_handle("default").context("default column family")?;
        let found = self.db.batched_multi_get_cf(cf, keys.iter().map(|(key, _)| key), false);
        let mut added = 0;
        for ((key, domain), present) in keys.into_iter().zip(found) {
            if present?.is_none() {
                batch.put(&key, []);
                *counts.entry(domain).or_default() += 1;
                added += 1;
            }
        }
        Ok(added)
    }

    fn remove_ready(&self, batch: &mut WriteBatch, keys: &[Vec<u8>], counts: &mut HashMap<String, i64>) -> Result<()> {
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        let cf = self.db.cf_handle("default").context("default column family")?;
        let found = self.db.batched_multi_get_cf(cf, &keys, false);
        for (key, present) in keys.iter().zip(found) {
            if present?.is_some() {
                batch.delete(key);
                if let Some((domain, _)) = split_key(&key[1..]) {
                    *counts.entry(String::from_utf8_lossy(domain).into_owned()).or_default() -= 1;
                }
            }
        }
        Ok(())
    }

    fn write_counts(&self, batch: &mut WriteBatch, counts: &HashMap<String, i64>) -> Result<()> {
        for (domain, &delta) in counts.iter().filter(|(_, &d)| d != 0) {
            let now = self.ready_count(domain)? as i64 + delta;
            if now > 0 {
                batch.put(count_key(domain), (now as u64).to_le_bytes());
            } else {
                batch.delete(count_key(domain));
            }
        }
        Ok(())
    }

    fn ready_keys(&self, domain: &str, after: Option<&[u8]>, limit: usize) -> Result<Vec<Vec<u8>>> {
        self.keys_under(&[&[READY], domain.as_bytes(), b"\0"].concat(), after, limit)
    }

    fn lane_keys(&self, domain: &str, lane: usize, after: Option<&[u8]>, limit: usize) -> Result<Vec<Vec<u8>>> {
        self.keys_under(&[&[READY], domain.as_bytes(), b"\0", &[lane as u8]].concat(), after, limit)
    }

    fn keys_under(&self, prefix: &[u8], after: Option<&[u8]>, limit: usize) -> Result<Vec<Vec<u8>>> {
        let prefix = prefix.to_vec();
        let start = after.map_or_else(|| prefix.clone(), |key| [key, b"\0"].concat());
        let mut keys = Vec::with_capacity(limit.min(4096));
        for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix)).take(limit) {
            keys.push(item?.0.to_vec());
        }
        Ok(keys)
    }

    fn records(&self, keys: impl Iterator<Item = Vec<u8>>) -> Result<Vec<Option<Record>>> {
        let keys: Vec<Vec<u8>> = keys.collect();
        let cf = self.db.cf_handle("default").context("default column family")?;
        self.db.batched_multi_get_cf(cf, &keys, false).into_iter().map(|raw| Ok(raw?.as_deref().and_then(Record::unpack))).collect()
    }

    fn sents(&self, keys: impl Iterator<Item = Vec<u8>>) -> Result<Vec<Option<Sent>>> {
        let keys: Vec<Vec<u8>> = keys.collect();
        let cf = self.db.cf_handle("default").context("default column family")?;
        self.db.batched_multi_get_cf(cf, &keys, false).into_iter().map(|raw| Ok(raw?.as_deref().and_then(Sent::unpack))).collect()
    }

    fn schedules(&self, keys: impl Iterator<Item = Vec<u8>>) -> Result<Vec<Option<Scheduled>>> {
        let keys: Vec<Vec<u8>> = keys.collect();
        let cf = self.db.cf_handle("default").context("default column family")?;
        self.db.batched_multi_get_cf(cf, &keys, false).into_iter().map(|raw| Ok(raw?.as_deref().and_then(Scheduled::unpack))).collect()
    }
}

#[derive(Default)]
struct Taking {
    robots: Paths,
    picks: Vec<Pick>,
    stale: Vec<Vec<u8>>,
    picked: std::collections::HashSet<Vec<u8>>,
    after: [Option<Vec<u8>>; LANES],
    drained: [bool; LANES],
}

/// A domain's changed pages once all of it was read, cut to the newest few when it re-stamps.
fn changed_entries(scan: DomainScan) -> Vec<Entry> {
    let restamping = restamped(scan.changed.len(), scan.listed);
    let mut changed: Vec<(Vec<u8>, u32)> = scan.changed.into_iter().filter(|(rest, _)| !junk(rest)).collect();
    if restamping {
        changed.sort_by_key(|(_, lastmod)| Reverse(*lastmod));
        changed.truncate(RESTAMP_KEEP);
    }
    let domain = String::from_utf8_lossy(&scan.domain).into_owned();
    changed.into_iter().map(|(rest, lastmod)| Entry { domain: domain.clone(), rest, order: Order::Changed { lastmod } }).collect()
}

/// A URL key without its type byte, split into domain and the rest.
fn split_key(key: &[u8]) -> Option<(&[u8], &[u8])> {
    let at = memchr::memchr(0, key)?;
    Some((&key[..at], &key[at + 1..]))
}

fn prefixed(kind: u8, domain: &[u8], rest: &[u8]) -> Vec<u8> {
    [&[kind], domain, b"\0", rest].concat()
}

fn count_key(domain: &str) -> Vec<u8> {
    count_key_of(READY_COUNT, domain)
}

fn count_key_of(kind: u8, domain: &str) -> Vec<u8> {
    [&[kind], domain.as_bytes()].concat()
}

fn count_value(raw: Option<&[u8]>) -> u64 {
    raw.and_then(|r| r.try_into().ok()).map_or(0, u64::from_le_bytes)
}

fn yield_key(domain: &str) -> Vec<u8> {
    [&[YIELD], domain.as_bytes()].concat()
}

/// A page's schedule and the waiting entry that brings it due, replacing any schedule it had.
fn put_schedule(batch: &mut WriteBatch, domain: &[u8], rest: &[u8], next: Scheduled) {
    batch.put(prefixed(NEXT, domain, rest), next.pack());
    batch.put(timed_key(WAITING, next.due, domain, rest), []);
}

fn timed_key(kind: u8, due: u32, domain: &[u8], rest: &[u8]) -> Vec<u8> {
    [&[kind], &due.to_be_bytes()[..], domain, b"\0", rest].concat()
}

fn timed_entry(key: &[u8]) -> Option<(u32, &[u8], &[u8])> {
    let due = u32::from_be_bytes(key.get(1..5)?.try_into().ok()?);
    let (domain, rest) = split_key(&key[5..])?;
    Some((due, domain, rest))
}

fn bounded(prefix: &[u8]) -> ReadOptions {
    let mut upper = prefix.to_vec();
    *upper.last_mut().expect("a prefix") += 1;
    let mut options = ReadOptions::default();
    options.set_iterate_upper_bound(upper);
    options
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn junk_matches_the_python_feeder() {
        for path in ["example.com/tag/x", "example.com/category/news/", "example.com/page/2", "example.com/a.PDF", "example.com/s?page=3", "example.com/x?q=a"] {
            assert!(junk(path.as_bytes()), "{path}");
        }
        for path in ["example.com/2026/10/story", "example.com/tagline", "example.com/a.pdf?x=1", "example.com/?id=3"] {
            assert!(!junk(path.as_bytes()), "{path}");
        }
    }

    #[test]
    fn keys_group_by_lane_each_in_its_own_order() {
        let entry = |order| Entry { domain: "example.com".into(), rest: b"example.com/a".to_vec(), order };
        let mut keys = [
            entry(Order::Fresh { first_seen: 100, lastmod: 5 }),
            entry(Order::Fresh { first_seen: 200, lastmod: 5 }),
            entry(Order::Fresh { first_seen: 200, lastmod: 9 }),
            entry(Order::Requeued { due: 50 }),
            entry(Order::Requeued { due: 10 }),
            entry(Order::Changed { lastmod: 7 }),
            entry(Order::Changed { lastmod: 8 }),
        ]
        .map(|e| e.key());
        keys.sort();
        let orders: Vec<Order> = keys.iter().map(|k| Entry::parse(k).unwrap().order).collect();
        assert_eq!(
            orders,
            [
                Order::Changed { lastmod: 8 },
                Order::Changed { lastmod: 7 },
                Order::Requeued { due: 10 },
                Order::Requeued { due: 50 },
                Order::Fresh { first_seen: 200, lastmod: 9 },
                Order::Fresh { first_seen: 200, lastmod: 5 },
                Order::Fresh { first_seen: 100, lastmod: 5 },
            ]
        );
        assert_eq!(Entry::parse(&keys[0]).unwrap().rest, b"example.com/a");
    }

    #[test]
    fn deadlines_of_earlier_versions_are_deleted_once_and_their_waiting_entries_still_come_back() {
        let dir = std::env::temp_dir().join(format!("desearch-bot-ready-old-keys-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = BucketStore::open(&dir, &crate::buckets::Resources::new(8 << 20, 8 << 20, 1)).unwrap();
        let page = |path: &str| (crate::urls::parse(&format!("https://example.com/{path}"), "example.com").unwrap(), 10, false);
        store.record_listing("example.com", 7, vec![page("a"), page("b")], 100).unwrap();
        let picks = store.take_ready("example.com", [0, 0, 10]).unwrap();
        store.mark_sent(&picks, 1000, &Recrawl::default()).unwrap();
        for (due, path) in [(1000 + 86_400, "a"), (1000 + 86_400, "b")] {
            store.db.put(timed_key(EXPECTING, due, b"example.com", format!("example.com/{path}").as_bytes()), []).unwrap();
        }
        store.db.put(timed_key(WAITING, 1500, b"example.com", b"example.com/b"), []).unwrap();

        assert!(store.forget_expecting().unwrap());
        assert!(store.db.iterator_opt(IteratorMode::From(&[EXPECTING], Direction::Forward), bounded(&[EXPECTING])).next().is_none());
        assert!(!store.forget_expecting().unwrap(), "nothing is left to delete on the next start");
        assert_eq!(store.requeue_due(1000 + 86_400, 100, &Recrawl::default()).unwrap(), Requeued { moved: 1, gone: 0 }, "a waiting entry with no schedule goes back as before");
        let again = store.take_ready("example.com", [0, 10, 0]).unwrap();
        assert_eq!(again.iter().map(|p| (p.url.as_str(), p.reason)).collect::<Vec<_>>(), [("https://example.com/b", Reason::Requeued)]);
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn failures_share_the_requeued_byte_and_the_count_of_earlier_versions_is_not_read() {
        let sent = Sent { lastmod: 7, failures: 2, requeued: false };
        assert_eq!(Sent::unpack(&sent.pack()), Some(sent));
        assert_eq!(sent.pack()[4..], [0, 4], "earlier versions read it as requeued, which only matters for a page on its list");
        let earlier = [7, 0, 0, 0, 3, 1];
        assert_eq!(Sent::unpack(&earlier), Some(Sent { lastmod: 7, failures: 0, requeued: true }), "three retries counted before, outcomes that never came among them");
    }

    #[test]
    fn a_schedule_packs_and_unpacks() {
        for why in [Why::Recrawl, Why::Failed, Why::Dropped, Why::Gone] {
            let next = Scheduled { why, due: 1_800_000_000, at: 1_700_000_000 };
            assert_eq!(Scheduled::unpack(&next.pack()), Some(next));
        }
        assert_eq!(Scheduled::unpack(b"x12345678"), None);
        assert_eq!(Scheduled::unpack(b""), None);
    }

    #[test]
    fn the_restamp_guard_needs_two_percent_and_more_than_fifty() {
        assert!(!restamped(50, 100));
        assert!(restamped(51, 100));
        assert!(!restamped(200, 10_000));
        assert!(restamped(201, 10_000));
    }
}

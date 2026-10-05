//! Pages waiting to go to the task API, a list per domain in the bucket store, and what became of the pages sent.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::bytes::{Regex, RegexBuilder};
use rocksdb::{Direction, IteratorMode, ReadOptions, WriteBatch};
use serde_json::{json, Value};

use crate::buckets::{meta_key, BucketStore, URL};
use crate::schedule::DAY;
use crate::urls::{Record, HTTPS, WWW};

/// Pages waiting to be sent, under their domain, in the order they go.
const READY: u8 = b'R';
/// How many pages each domain has waiting.
const READY_COUNT: u8 = b'Q';
/// The lastmod each page was last sent with, and its failures since.
const SENT: u8 = b'P';
/// Retries and re-crawls by the time they join their ready list again.
const WAITING: u8 = b'W';
/// Sent pages by the time their outcome is overdue.
const EXPECTING: u8 = b'X';
const BACKFILL: &str = "backfill_ready";

/// A sitemap read that moves more than this share of its pages, and more than RESTAMP_MIN, past what they were sent with is re-stamping dates, not changing pages.
pub const RESTAMP_SHARE: f64 = 0.02;
pub const RESTAMP_MIN: usize = 50;
pub const RESTAMP_KEEP: usize = 10;
/// Waits before each retry of a page that failed; after the last, it waits for its lastmod to change.
pub const BACKOFF: [u32; 3] = [3600, 6 * 3600, 24 * 3600];
/// An outcome this much older than the latest send of its page belongs to an earlier send.
const STALE_OUTCOME: u32 = 60;
pub const DEFAULT_RECRAWL: u32 = (7 * DAY / 1_000_000) as u32;
/// A sent page with no outcome by then counts as dropped: the task API lost it, or its outcome file expired unread.
pub const NO_OUTCOME: u32 = 24 * 3600;

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
    /// Failures since it was last sent for a new lastmod.
    pub retries: u8,
    /// Back on its ready list for a retry or a re-crawl.
    pub requeued: bool,
}

impl Sent {
    fn pack(self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[..4].copy_from_slice(&self.lastmod.to_le_bytes());
        out[4] = self.retries;
        out[5] = self.requeued.into();
        out
    }

    fn unpack(data: &[u8]) -> Option<Sent> {
        (data.len() == 6).then(|| Sent { lastmod: u32::from_le_bytes(data[..4].try_into().unwrap()), retries: data[4], requeued: data[5] != 0 })
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
    pub retries: u8,
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

/// What became of one sent page, as its store keys it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageOutcome {
    pub domain: String,
    pub rest: Vec<u8>,
    pub outcome: Outcome,
    pub at: u32,
}

/// A page an earlier sender sent, with the lastmod it had then.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentPage {
    pub domain: String,
    pub rest: Vec<u8>,
    pub lastmod: u32,
    pub at: u32,
}

/// What a batch of outcomes did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub crawled: u64,
    pub retried: u64,
    pub gave_up: u64,
    /// Outcomes of an earlier send, or of pages the store does not know.
    pub ignored: u64,
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
    pub fn take_ready(&self, domain: &str, quotas: [usize; LANES]) -> Result<Vec<Pick>> {
        let want: usize = quotas.iter().sum();
        let mut taking = Taking::default();
        for (lane, quota) in quotas.into_iter().enumerate() {
            self.take_lane(domain, lane, quota, &mut taking)?;
        }
        for lane in SPILL {
            if taking.picks.len() >= want {
                break;
            }
            self.take_lane(domain, lane, want - taking.picks.len(), &mut taking)?;
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
                let due = record.and_then(|r| reason(&r, sent.as_ref()).map(|why| (r, why)));
                // A page listed twice, under an older and a newer lastmod, goes once.
                let Some((record, why)) = due.filter(|_| taking.picked.insert(entry.rest.clone())) else {
                    taking.stale.push(key);
                    continue;
                };
                let retries = sent.map_or(0, |s| s.retries);
                taking.picks.push(Pick { url: fetchable(&entry.rest, record.flags), lastmod: record.lastmod, reason: why, retries, entry });
            }
            // A read that moved nothing forward would only repeat itself.
            if (read < asked && !stopped_early) || taking.after[lane] == started {
                taking.drained[lane] = true;
            }
        }
        Ok(())
    }

    /// The task API took these pages: stamp them sent, with the lastmod they went with, and take them off their lists.
    pub fn mark_sent(&self, picks: &[Pick], at: u32) -> Result<()> {
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
            }
            let retries = if pick.reason == Reason::Requeued { pick.retries } else { 0 };
            batch.put(pick.entry.sent_key(), Sent { lastmod: pick.lastmod, retries, requeued: false }.pack());
            batch.put(timed_key(EXPECTING, at.saturating_add(NO_OUTCOME), pick.entry.domain.as_bytes(), &pick.entry.rest), []);
        }
        // A lastmod that moved while the page was in flight is a change the API has not seen.
        self.add_ready(&mut batch, moved, &mut counts)?;
        self.write_counts(&mut batch, &counts)?;
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(())
    }

    /// Sent pages whose outcome never came are retried like a dropped page; returns how many.
    pub fn expire_unanswered(&self, now: u32, limit: usize) -> Result<usize> {
        let mut options = ReadOptions::default();
        options.set_iterate_upper_bound([&[EXPECTING][..], &now.saturating_add(1).to_be_bytes()].concat());
        let mut keys = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&[EXPECTING], Direction::Forward), options).take(limit) {
            keys.push(item?.0.to_vec());
        }
        if keys.is_empty() {
            return Ok(0);
        }
        let _guard = self.lock_ready();
        let parsed: Vec<(u32, &[u8], &[u8])> = keys.iter().filter_map(|key| timed_entry(key)).collect();
        let records = self.records(parsed.iter().map(|(_, domain, rest)| prefixed(URL, domain, rest)))?;
        let sents = self.sents(parsed.iter().map(|(_, domain, rest)| prefixed(SENT, domain, rest)))?;
        let mut batch = WriteBatch::default();
        let mut expired = 0;
        for (((due, domain, rest), record), sent) in parsed.iter().zip(records).zip(sents) {
            // Only the latest send of a page can be overdue; an older deadline is left over from a resend.
            let Some(record) = record.filter(|r| r.pushed_at.saturating_add(NO_OUTCOME) == *due) else {
                continue;
            };
            let mut lost = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
            lost.retries = lost.retries.saturating_add(1);
            batch.put(prefixed(SENT, domain, rest), lost.pack());
            if usize::from(lost.retries) <= BACKOFF.len() {
                batch.put(timed_key(WAITING, now, domain, rest), []);
                expired += 1;
            }
        }
        for key in &keys {
            batch.delete(key);
        }
        self.db.write(batch)?;
        Ok(expired)
    }

    /// Put retries and re-crawls that are due back on their ready lists; returns how many moved.
    pub fn requeue_due(&self, now: u32, limit: usize) -> Result<usize> {
        let mut options = ReadOptions::default();
        options.set_iterate_upper_bound([&[WAITING][..], &now.saturating_add(1).to_be_bytes()].concat());
        let mut keys = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&[WAITING], Direction::Forward), options).take(limit) {
            keys.push(item?.0.to_vec());
        }
        if keys.is_empty() {
            return Ok(0);
        }
        let _guard = self.lock_ready();
        let parsed: Vec<(u32, &[u8], &[u8])> = keys.iter().filter_map(|key| timed_entry(key)).collect();
        let records = self.records(parsed.iter().map(|(_, domain, rest)| prefixed(URL, domain, rest)))?;
        let sents = self.sents(parsed.iter().map(|(_, domain, rest)| prefixed(SENT, domain, rest)))?;
        let mut batch = WriteBatch::default();
        let mut entries = Vec::new();
        for (((due, domain, rest), record), sent) in parsed.iter().zip(records).zip(sents) {
            let Some(record) = record else {
                continue;
            };
            let sent = Sent { requeued: true, ..sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() }) };
            batch.put(prefixed(SENT, domain, rest), sent.pack());
            entries.push(Entry { domain: String::from_utf8_lossy(domain).into_owned(), rest: rest.to_vec(), order: Order::Requeued { due: *due } });
        }
        for key in &keys {
            batch.delete(key);
        }
        let mut counts = HashMap::new();
        self.add_ready(&mut batch, entries, &mut counts)?;
        self.write_counts(&mut batch, &counts)?;
        self.db.write(batch)?;
        self.notice(&counts);
        Ok(keys.len())
    }

    /// Keep what became of sent pages: a crawl stamps crawled_at, a failure comes back after a backoff.
    pub fn apply_outcomes(&self, rows: &[PageOutcome], recrawl_after: u32) -> Result<Applied> {
        let mut applied = Applied::default();
        if rows.is_empty() {
            return Ok(applied);
        }
        let _guard = self.lock_ready();
        let url_keys: Vec<Vec<u8>> = rows.iter().map(|row| prefixed(URL, row.domain.as_bytes(), &row.rest)).collect();
        let mut records: HashMap<Vec<u8>, Option<Record>> = HashMap::new();
        let mut sents: HashMap<Vec<u8>, Option<Sent>> = HashMap::new();
        let distinct: Vec<&Vec<u8>> = {
            let mut seen = std::collections::HashSet::new();
            url_keys.iter().filter(|k| seen.insert(*k)).collect()
        };
        let found = self.records(distinct.iter().map(|k| (*k).clone()))?;
        let found_sent = self.sents(distinct.iter().map(|k| [&[SENT], &k[1..]].concat()))?;
        for ((key, record), sent) in distinct.iter().zip(found).zip(found_sent) {
            records.insert((*key).clone(), record);
            sents.insert((*key).clone(), sent);
        }
        let mut batch = WriteBatch::default();
        for (PageOutcome { domain, rest, outcome, at }, key) in rows.iter().zip(&url_keys) {
            let Some(Some(record)) = records.get_mut(key) else {
                applied.ignored += 1;
                continue;
            };
            if record.pushed_at <= at.saturating_add(STALE_OUTCOME) {
                batch.delete(timed_key(EXPECTING, record.pushed_at.saturating_add(NO_OUTCOME), domain.as_bytes(), rest));
            }
            let sent = sents.get_mut(key).expect("looked up with its record");
            match outcome {
                Outcome::Published | Outcome::Unchanged => {
                    record.crawled_at = record.crawled_at.max(*at);
                    if let Some(sent) = sent.as_mut() {
                        sent.retries = 0;
                        sent.requeued = false;
                    }
                    if record.lastmod == 0 {
                        batch.put(timed_key(WAITING, at.saturating_add(recrawl_after), domain.as_bytes(), rest), []);
                    }
                    applied.crawled += 1;
                }
                Outcome::Failed | Outcome::Dropped => {
                    if record.pushed_at > at.saturating_add(STALE_OUTCOME) {
                        applied.ignored += 1;
                        continue;
                    }
                    let failed = sent.get_or_insert(Sent { lastmod: record.lastmod, ..Sent::default() });
                    failed.retries = failed.retries.saturating_add(1);
                    match BACKOFF.get(usize::from(failed.retries) - 1) {
                        Some(wait) => {
                            batch.put(timed_key(WAITING, at.saturating_add(*wait), domain.as_bytes(), rest), []);
                            applied.retried += 1;
                        }
                        None => applied.gave_up += 1,
                    }
                }
            }
        }
        for (key, record) in &records {
            if let Some(record) = record {
                batch.put(key, record.pack());
            }
            if let Some(Some(sent)) = sents.get(key) {
                batch.put([&[SENT], &key[1..]].concat(), sent.pack());
            }
        }
        self.db.write(batch)?;
        Ok(applied)
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
        let mut read = 0;
        for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), options).take(limit) {
            let (key, raw) = item?;
            read += 1;
            let Some((domain, rest)) = split_key(&key[1..]) else {
                continue;
            };
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
        if read < limit {
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
        for (key, record) in fresh {
            if let Some((domain, rest)) = split_key(&key[1..]).filter(|(_, rest)| !junk(rest)) {
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
}

#[derive(Default)]
struct Taking {
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
    [&[READY_COUNT], domain.as_bytes()].concat()
}

fn count_value(raw: Option<&[u8]>) -> u64 {
    raw.and_then(|r| r.try_into().ok()).map_or(0, u64::from_le_bytes)
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
    fn the_restamp_guard_needs_two_percent_and_more_than_fifty() {
        assert!(!restamped(50, 100));
        assert!(restamped(51, 100));
        assert!(!restamped(200, 10_000));
        assert!(restamped(201, 10_000));
    }
}

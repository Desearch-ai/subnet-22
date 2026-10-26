//! Each listed domain's set of pages as a due queue: every page has one due time, and a pass takes what is due.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{Context, Result};
use rocksdb::{Direction, IteratorMode, WriteBatch};

use crate::allowed::{SiteType, Terms};
use crate::buckets::{BucketStore, URL};
use crate::ready::{
    bounded, count_key_of, fetchable, junk, prefixed, restamped, timed_key, Entry, Order, PageSchedule, Pick, Reason, Scheduled, Sent, Why, GONE_AFTER, JOINED,
    NEXT, RESTAMP_KEEP, SENT, STALE_OUTCOME, STOCK, WAITING,
};
use crate::records;
use crate::recrawl::Recrawl;
use crate::robots::Paths;
use crate::urls::Record;

/// A domain's set as a queue: one key per page, in the order the pages fall due. D and S are domains and sitemaps.
pub(crate) const DUE: u8 = b'T';
/// How many pages a domain's set holds; written once its queue is built.
pub(crate) const SET_SIZE: u8 = b'K';

const HOUR: u32 = 3_600;
const DAY: u32 = 86_400;
/// Stored records read at a time while a domain's queue is built.
const BUILD_CHUNK: usize = 4096;
/// Queue keys a take may pass over as stale beyond what it wants.
const TAKE_SLACK: usize = 64;
/// Pages admitted per locked write.
const ADMIT_CHUNK: usize = 20_000;
/// A failed page goes again this long after the failure.
const RETRY_AFTER: u32 = DAY;
/// Failures a page that left its set is marked with, past any count of real ones, so a rebuild leaves it out too.
pub(crate) const LEFT: u8 = 127;

/// Seconds after a send that a page falls due again, by its site's type and, for news, how long ago the article appeared.
pub fn interval_for(site: SiteType, first_seen: u32, at: u32) -> u32 {
    match site {
        SiteType::News => match at.saturating_sub(first_seen) {
            age if age < 2 * DAY => 12 * HOUR,
            age if age < 7 * DAY => DAY,
            age if age < 30 * DAY => 7 * DAY,
            _ => 30 * DAY,
        },
        SiteType::Reference | SiteType::Docs | SiteType::Academic | SiteType::Public | SiteType::Other => 7 * DAY,
        SiteType::Media | SiteType::Community => 14 * DAY,
        SiteType::Company | SiteType::Organization | SiteType::Personal => 30 * DAY,
    }
}

/// The interval of one page: its domain's own line in the overrides, else the table.
pub fn page_interval(recrawl: &Recrawl, site: SiteType, domain: &str, first_seen: u32, at: u32) -> u32 {
    recrawl.own_interval(domain).unwrap_or_else(|| interval_for(site, first_seen, at))
}

/// A news site's home or section page: the root, or one short path segment without digits, such as /world or /us-news.
pub fn front_page(rest: &[u8]) -> bool {
    let Some(slash) = rest.iter().position(|&b| b == b'/') else {
        return true;
    };
    let path = rest[slash..].split(|&b| b == b'?' || b == b'#').next().unwrap_or_default();
    let segment: Vec<u8> = path.iter().copied().skip_while(|&b| b == b'/').collect();
    let segment = segment.strip_suffix(b"/").unwrap_or(&segment);
    segment.is_empty()
        || (!segment.contains(&b'/')
            && segment.len() <= 20
            && !segment.iter().any(|b| b.is_ascii_digit() || *b == b'.')
            && segment.iter().filter(|&&b| b == b'-').count() <= 1)
}

/// The most pages a domain's set may hold: news rolls within its backlog, the rest fill up to their cap.
pub fn set_cap(terms: Terms) -> Option<u64> {
    if terms.site == SiteType::News {
        terms.backlog
    } else {
        terms.send_cap
    }
}

pub(crate) fn due_key(domain: &str, due: u32, first: bool, rest: &[u8]) -> Vec<u8> {
    // A first crawl sorts before any other page due the same second.
    [&[DUE], domain.as_bytes(), b"\0", &due.to_be_bytes(), &[u8::from(!first)], rest].concat()
}

fn due_parts(key: &[u8], prefix: usize) -> Option<(u32, bool, &[u8])> {
    let due = u32::from_be_bytes(key.get(prefix..prefix + 4)?.try_into().ok()?);
    Some((due, *key.get(prefix + 4)? == 0, key.get(prefix + 5..)?))
}

fn wanted(site: SiteType, domain: &str, rest: &[u8], robots: &Paths) -> bool {
    !junk(rest) && robots.allows_page(domain, rest) && !(site == SiteType::News && front_page(rest))
}

/// What one build of a domain's queue found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Built {
    pub scanned: u64,
    /// Sent pages kept, each due after its interval or as its schedule says.
    pub members: u64,
    /// Pages never sent that joined, due now as first crawls.
    pub admitted: u64,
    pub stopped: bool,
}

/// What reached a domain while its queue was being built, applied once it is.
#[derive(Default)]
pub struct Held {
    outcomes: Vec<PageSchedule>,
    fresh: Vec<Vec<u8>>,
    moved: Vec<(Vec<u8>, u32)>,
}

/// Where a domain's sends, outcomes and sitemap changes go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Route {
    Lists,
    /// Its queue is being built: they wait for it.
    Held,
    Queue,
}

/// Takes a domain off the held map however its build ends.
struct Building<'a> {
    store: &'a BucketStore,
    domain: &'a str,
}

impl Drop for Building<'_> {
    fn drop(&mut self) {
        self.store.held.lock().unwrap_or_else(|e| e.into_inner()).remove(self.domain);
    }
}

/// What a take from a domain's due queue found.
#[derive(Debug, Default)]
pub struct DueTake {
    pub picks: Vec<Pick>,
    /// When the first page left after the picks is due; None when none is left.
    pub head: Option<u32>,
    /// For a pull that found nothing to take: pages read were sent within the floor, and the first may go from this time.
    pub floor_until: Option<u32>,
}

impl BucketStore {
    /// Pages in the domain's set; None until its queue is built.
    pub fn set_size(&self, domain: &str) -> Result<Option<u64>> {
        Ok(self.db.get_pinned(count_key_of(SET_SIZE, domain))?.and_then(|raw| raw.as_ref().try_into().ok().map(u64::from_le_bytes)))
    }

    /// When the front of the domain's queue is due; None when it is empty.
    pub fn due_head(&self, domain: &str) -> Result<Option<u32>> {
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let first = self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)).next().transpose()?;
        Ok(first.and_then(|(key, _)| due_parts(&key, prefix.len()).map(|(due, _, _)| due)))
    }

    /// Every domain whose queue is built.
    pub fn queued_domains(&self) -> Result<Vec<String>> {
        let prefix = [SET_SIZE];
        let mut domains = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)) {
            domains.push(String::from_utf8_lossy(&item?.0[1..]).into_owned());
        }
        Ok(domains)
    }

    /// Where the domain's sends, outcomes and sitemap changes go; read under the ready lock, so a build cannot finish between the look and the write.
    pub(crate) fn route(&self, domain: &str) -> Result<Route> {
        if self.set_size(domain)?.is_some() {
            Ok(Route::Queue)
        } else if self.held.lock().unwrap_or_else(|e| e.into_inner()).contains_key(domain) {
            Ok(Route::Held)
        } else {
            Ok(Route::Lists)
        }
    }

    /// Keep a sitemap read's pages of a domain whose queue is being built; a re-stamping read keeps only its newest changes.
    pub(crate) fn hold_listing(&self, domain: &str, fresh: &[&[u8]], moved: &[(&[u8], u32)], listed: usize) {
        let mut moved: Vec<(Vec<u8>, u32)> = moved.iter().map(|(rest, lastmod)| (rest.to_vec(), *lastmod)).collect();
        if restamped(moved.len(), listed) {
            moved.sort_by_key(|(_, lastmod)| Reverse(*lastmod));
            moved.truncate(RESTAMP_KEEP);
        }
        if let Some(held) = self.held.lock().unwrap_or_else(|e| e.into_inner()).get_mut(domain) {
            held.fresh.extend(fresh.iter().map(|rest| rest.to_vec()));
            held.moved.extend(moved);
        }
    }

    pub(crate) fn hold_outcome(&self, page: &PageSchedule) {
        if let Some(held) = self.held.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&page.domain) {
            held.outcomes.push(page.clone());
        }
    }

    /// Domains whose queue fronts may have moved since the dispatcher last looked.
    pub fn take_due_noticed(&self) -> Vec<String> {
        self.due_noticed.lock().unwrap_or_else(|e| e.into_inner()).drain().collect()
    }

    fn notice_due(&self, domain: &str) {
        self.due_noticed.lock().unwrap_or_else(|e| e.into_inner()).insert(domain.to_string());
    }

    /// Build the domain's queue from its stored pages: every sent page not gone, due as its schedule says or after its interval, and the newest pages never sent up to the cap, due now.
    pub fn build_queue(&self, domain: &str, terms: Terms, recrawl: &Recrawl, now: u32, pace: &mut dyn FnMut(u64) -> bool) -> Result<Built> {
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(URL, domain.as_bytes(), b"");
        let cap = set_cap(terms);
        let mut built = Built::default();
        // The newest pages never sent, at most the cap: a heap of the oldest kept.
        let mut newest: BinaryHeap<Reverse<(u32, u32, Vec<u8>)>> = BinaryHeap::new();
        let mut last: Option<Vec<u8>> = None;
        self.begin_build(domain)?;
        let _building = Building { store: self, domain };
        loop {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let mut chunk: Vec<(Vec<u8>, Record)> = Vec::with_capacity(BUILD_CHUNK);
            let mut read = 0;
            for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix)).take(BUILD_CHUNK) {
                let (key, raw) = item?;
                read += 1;
                last = Some(key.to_vec());
                if let Some(record) = Record::unpack(&raw) {
                    chunk.push((key.to_vec(), record));
                }
            }
            built.scanned += read as u64;
            let (sent, unsent): (Vec<_>, Vec<_>) = chunk.into_iter().partition(|(_, record)| record.pushed_at != 0);
            let rests: Vec<&[u8]> = sent.iter().map(|(key, _)| &key[prefix.len()..]).collect();
            let sents = self.sents(rests.iter().map(|rest| prefixed(SENT, domain.as_bytes(), rest)))?;
            let nexts = self.schedules(rests.iter().map(|rest| prefixed(NEXT, domain.as_bytes(), rest)))?;
            let mut batch = WriteBatch::default();
            for ((((_, record), rest), sent), next) in sent.iter().zip(&rests).zip(sents).zip(nexts) {
                if !wanted(terms.site, domain, rest, &robots) {
                    continue;
                }
                let sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
                let current = next.filter(|next| !next.outdated(record.pushed_at));
                if sent.failures >= GONE_AFTER || current.is_some_and(|next| next.why == Why::Gone) {
                    continue;
                }
                if current.is_some_and(|next| next.why == Why::Leave) {
                    batch.put(prefixed(SENT, domain.as_bytes(), rest), Sent { failures: LEFT, ..sent }.pack());
                    continue;
                }
                let schedule = match current {
                    Some(next) if matches!(next.why, Why::Failed | Why::Dropped) => next,
                    _ => standing(recrawl, terms.site, domain, rest, record, &sent, now),
                };
                batch.put(due_key(domain, schedule.due, false, rest), []);
                batch.put(prefixed(NEXT, domain.as_bytes(), rest), schedule.pack());
                built.members += 1;
            }
            for (key, record) in unsent {
                let rest = &key[prefix.len()..];
                if wanted(terms.site, domain, rest, &robots) {
                    newest.push(Reverse((record.first_seen, record.lastmod, rest.to_vec())));
                    if cap.is_some_and(|cap| newest.len() as u64 > cap) {
                        newest.pop();
                    }
                }
            }
            if !batch.is_empty() {
                let _guard = self.lock_ready();
                self.db.write(batch)?;
            }
            if !pace(read as u64) {
                built.stopped = true;
                return Ok(built);
            }
            if read < BUILD_CHUNK {
                break;
            }
        }
        let room = cap.map_or(usize::MAX, |cap| cap.saturating_sub(built.members).min(usize::MAX as u64) as usize);
        let mut kept: Vec<(u32, u32, Vec<u8>)> = newest.into_iter().map(|Reverse(page)| page).collect();
        kept.sort_unstable_by(|a, b| b.cmp(a));
        kept.truncate(room);
        for chunk in kept.chunks(ADMIT_CHUNK) {
            let mut batch = WriteBatch::default();
            for (_, _, rest) in chunk {
                admit_into(&mut batch, domain, rest, now);
            }
            let _guard = self.lock_ready();
            self.db.write(batch)?;
        }
        built.admitted = kept.len() as u64;
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        batch.put(count_key_of(SET_SIZE, domain), (built.members + built.admitted).to_le_bytes());
        batch.put(count_key_of(JOINED, domain), cap.unwrap_or(u64::MAX).to_le_bytes());
        // The refresh stocks are retired: the queue is already in the order they kept.
        batch.delete_range(prefixed(STOCK, domain.as_bytes(), b""), [&[STOCK], domain.as_bytes(), b"\x01"].concat());
        self.clear_ready(&mut batch, domain);
        self.db.write(batch)?;
        let held = self.held.lock().unwrap_or_else(|e| e.into_inner()).remove(domain).unwrap_or_default();
        let mut batch = WriteBatch::default();
        let fresh: Vec<&[u8]> = held.fresh.iter().map(Vec::as_slice).collect();
        let moved: Vec<(&[u8], u32)> = held.moved.iter().map(|(rest, lastmod)| (rest.as_slice(), *lastmod)).collect();
        self.listing_into_queue(&mut batch, domain, &fresh, &moved, usize::MAX, now)?;
        self.outcomes_into_queues(&mut batch, &held.outcomes.iter().collect::<Vec<_>>())?;
        self.db.write(batch)?;
        drop(_guard);
        self.notice_due(domain);
        // More sent pages than the cap: the set keeps the newest.
        if cap.is_some_and(|cap| built.members > cap) {
            let shrunk = self.shrink_set(domain, terms, pace)?;
            built.scanned += shrunk.scanned;
            built.stopped = shrunk.stopped;
        }
        Ok(built)
    }

    /// Admit the newest stored pages not in the set, up to the room its cap leaves: pages never sent as first crawls, and sent pages a cap cut left out on their refresh schedule.
    pub fn admit_more(&self, domain: &str, terms: Terms, recrawl: &Recrawl, now: u32, pace: &mut dyn FnMut(u64) -> bool) -> Result<Built> {
        let mut built = Built::default();
        let Some(size) = self.recount_set(domain, pace)? else {
            return Ok(built);
        };
        let cap = set_cap(terms);
        let room = cap.map_or(u64::MAX, |cap| cap.saturating_sub(size));
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(URL, domain.as_bytes(), b"");
        let mut newest: BinaryHeap<Reverse<(u32, u32, Vec<u8>, [u8; 9])>> = BinaryHeap::new();
        let mut last: Option<Vec<u8>> = None;
        while room > 0 {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let mut chunk: Vec<(Vec<u8>, Record)> = Vec::with_capacity(BUILD_CHUNK);
            let mut read = 0;
            for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix)).take(BUILD_CHUNK) {
                let (key, raw) = item?;
                read += 1;
                last = Some(key.to_vec());
                if let Some(record) = Record::unpack(&raw).filter(|_| wanted(terms.site, domain, &key[prefix.len()..], &robots)) {
                    chunk.push((key[prefix.len()..].to_vec(), record));
                }
            }
            built.scanned += read as u64;
            let nexts = self.schedules(chunk.iter().map(|(rest, _)| prefixed(NEXT, domain.as_bytes(), rest)))?;
            let sents = self.sents(chunk.iter().map(|(rest, _)| prefixed(SENT, domain.as_bytes(), rest)))?;
            let keys: Vec<Option<Vec<u8>>> = chunk.iter().zip(&nexts).map(|((rest, _), next)| next.map(|next| due_key(domain, next.due, next.why == Why::Admit, rest))).collect();
            let mut members = self.present(keys.iter().flatten())?.into_iter();
            for ((((rest, record), key), next), sent) in chunk.into_iter().zip(keys).zip(nexts).zip(sents) {
                if key.is_some() && members.next().unwrap_or(false) {
                    continue;
                }
                let sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
                let current = next.filter(|next| !next.outdated(record.pushed_at));
                if sent.failures >= GONE_AFTER || current.is_some_and(|next| matches!(next.why, Why::Gone | Why::Leave)) {
                    continue;
                }
                let schedule = standing(recrawl, terms.site, domain, &rest, &record, &sent, now);
                newest.push(Reverse((record.first_seen, record.lastmod, rest, schedule.pack())));
                if newest.len() as u64 > room {
                    newest.pop();
                }
            }
            if !pace(read as u64) {
                built.stopped = true;
                return Ok(built);
            }
            if read < BUILD_CHUNK {
                break;
            }
        }
        let mut kept: Vec<(u32, u32, Vec<u8>, [u8; 9])> = newest.into_iter().map(|Reverse(page)| page).collect();
        kept.sort_unstable_by(|a, b| b.cmp(a));
        for chunk in kept.chunks(ADMIT_CHUNK) {
            let _guard = self.lock_ready();
            // Sitemap reads may have taken some of the room meanwhile.
            let size = self.set_size(domain)?.unwrap_or(0);
            let fits = cap.map_or(chunk.len() as u64, |cap| cap.saturating_sub(size).min(chunk.len() as u64)) as usize;
            let mut batch = WriteBatch::default();
            for (_, _, rest, schedule) in &chunk[..fits] {
                let Some(schedule) = Scheduled::unpack(schedule) else {
                    continue;
                };
                batch.put(due_key(domain, schedule.due, schedule.why == Why::Admit, rest), []);
                batch.put(prefixed(NEXT, domain.as_bytes(), rest), schedule.pack());
            }
            batch.put(count_key_of(SET_SIZE, domain), (size + fits as u64).to_le_bytes());
            self.db.write(batch)?;
            built.admitted += fits as u64;
            if fits < chunk.len() {
                break;
            }
        }
        self.db.put(count_key_of(JOINED, domain), cap.unwrap_or(u64::MAX).to_le_bytes())?;
        if built.admitted > 0 {
            self.notice_due(domain);
        }
        Ok(built)
    }

    /// A sitemap read of a domain with a built queue: new pages join while its cap leaves room (news rolls, its latest-due page leaving), and changed pages fall due, at once if changed since their last send, else a day after it; a sitemap re-stamping its dates moves only its newest. The caller holds the lock and writes the batch.
    pub(crate) fn listing_into_queue(
        &self,
        batch: &mut WriteBatch,
        domain: &str,
        fresh: &[&[u8]],
        moved: &[(&[u8], u32)],
        listed: usize,
        now: u32,
    ) -> Result<usize> {
        let terms = self.terms(domain);
        let Some(mut size) = self.set_size(domain)? else {
            return Ok(0);
        };
        let cap = set_cap(terms);
        let candidate = |rest: &[u8]| !junk(rest) && !(terms.site == SiteType::News && front_page(rest));
        let mut admitted = 0;
        let mut rolled = 0;
        // A page with a schedule is already in: the build that held this read admitted it.
        let scheduled = self.schedules(fresh.iter().map(|rest| prefixed(NEXT, domain.as_bytes(), rest)))?;
        for (rest, scheduled) in fresh.iter().zip(scheduled) {
            if !candidate(rest) || scheduled.is_some() {
                continue;
            }
            if cap.is_some_and(|cap| size >= cap) {
                if terms.site != SiteType::News {
                    continue;
                }
                rolled += 1;
            } else {
                size += 1;
            }
            admit_into(batch, domain, rest, now);
            admitted += 1;
        }
        if !moved.is_empty() {
            let nexts = self.schedules(moved.iter().map(|(rest, _)| prefixed(NEXT, domain.as_bytes(), rest)))?;
            let sents = self.sents(moved.iter().map(|(rest, _)| prefixed(SENT, domain.as_bytes(), rest)))?;
            let records = self.records(moved.iter().map(|(rest, _)| prefixed(URL, domain.as_bytes(), rest)))?;
            let keys: Vec<Option<Vec<u8>>> = moved.iter().zip(&nexts).map(|((rest, _), next)| next.map(|next| due_key(domain, next.due, next.why == Why::Admit, rest))).collect();
            let members = self.present(keys.iter().flatten())?;
            let mut members = members.into_iter();
            let mut changed = Vec::new();
            for ((((rest, lastmod), key), sent), record) in moved.iter().zip(keys).zip(sents).zip(records) {
                // A schedule says nothing of membership; its queue key does.
                let member = key.filter(|_| members.next().unwrap_or(false));
                if sent.is_some_and(|sent| *lastmod <= sent.lastmod || sent.failures == LEFT) || (member.is_none() && !candidate(rest)) {
                    continue;
                }
                changed.push((*rest, *lastmod, member, record.map_or(0, |record| record.pushed_at)));
            }
            if restamped(changed.len(), listed) {
                changed.sort_by_key(|(_, lastmod, _, _)| Reverse(*lastmod));
                changed.truncate(RESTAMP_KEEP);
            }
            for (rest, lastmod, member, pushed_at) in changed {
                match member {
                    Some(key) => batch.delete(key),
                    None if cap.is_none_or(|cap| size < cap) => size += 1,
                    None => continue,
                }
                let due = if lastmod > pushed_at { now } else { now.max(pushed_at.saturating_add(DAY)) };
                batch.put(due_key(domain, due, false, rest), []);
                batch.put(prefixed(NEXT, domain.as_bytes(), rest), Scheduled { why: Why::Changed, due, at: pushed_at }.pack());
                admitted += 1;
            }
        }
        if rolled > 0 {
            self.evict_latest(batch, domain, rolled)?;
        }
        batch.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
        if admitted > 0 {
            self.notice_due(domain);
        }
        Ok(admitted)
    }

    fn present<'a>(&self, keys: impl Iterator<Item = &'a Vec<u8>>) -> Result<Vec<bool>> {
        let keys: Vec<&Vec<u8>> = keys.collect();
        let cf = self.db.cf_handle("default").context("default column family")?;
        self.db.batched_multi_get_cf(cf, &keys, false).into_iter().map(|found| Ok(found?.is_some())).collect()
    }

    /// Take the `count` pages due last off a rolling set: in news they are the oldest articles, on the longest intervals.
    fn evict_latest(&self, batch: &mut WriteBatch, domain: &str, count: usize) -> Result<()> {
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let end = [&[DUE], domain.as_bytes(), b"\x01"].concat();
        for item in self.db.iterator_opt(IteratorMode::From(&end, Direction::Reverse), bounded(&prefix)).take(count) {
            let key = item?.0;
            if let Some((_, _, rest)) = due_parts(&key, prefix.len()) {
                batch.delete(prefixed(NEXT, domain.as_bytes(), rest));
            }
            batch.delete(&key);
        }
        Ok(())
    }

    /// Up to `want` pages from the front of the domain's queue due by `now`; with a `floor`, the soonest due whatever their time, passing over pages sent within `floor`. A key out of step with its page moves to where the page stands, and a page gone from the store or marked gone leaves the set.
    pub fn take_due(&self, domain: &str, want: usize, now: u32, floor: Option<u32>, recrawl: &Recrawl) -> Result<DueTake> {
        let site = self.terms(domain).site;
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let mut taken = DueTake::default();
        let mut keys: Vec<(Vec<u8>, u32, bool)> = Vec::new();
        // A pull may pass over as many pages sent within the floor as it wants.
        let slack = if floor.is_some() { want.max(TAKE_SLACK) } else { TAKE_SLACK };
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)) {
            let key = item?.0.to_vec();
            let Some((due, first, _)) = due_parts(&key, prefix.len()) else {
                continue;
            };
            if (floor.is_none() && due > now) || keys.len() >= want + slack {
                taken.head = Some(due);
                break;
            }
            keys.push((key, due, first));
        }
        let rests: Vec<Vec<u8>> = keys.iter().map(|(key, _, _)| key[prefix.len() + 5..].to_vec()).collect();
        let records = self.records(rests.iter().map(|rest| prefixed(URL, domain.as_bytes(), rest)))?;
        let sents = self.sents(rests.iter().map(|rest| prefixed(SENT, domain.as_bytes(), rest)))?;
        let nexts = self.schedules(rests.iter().map(|rest| prefixed(NEXT, domain.as_bytes(), rest)))?;
        let mut fixes = WriteBatch::default();
        let mut left: HashSet<Vec<u8>> = HashSet::new();
        let later = |head: &mut Option<u32>, due: u32| *head = Some(head.map_or(due, |head| head.min(due)));
        for ((((key, due, first), rest), record), (sent, next)) in keys.into_iter().zip(rests).zip(records).zip(sents.into_iter().zip(nexts)) {
            if taken.picks.len() >= want {
                later(&mut taken.head, due);
                break;
            }
            let Some(record) = record else {
                fixes.delete(&key);
                fixes.delete(prefixed(NEXT, domain.as_bytes(), &rest));
                left.insert(rest);
                continue;
            };
            let sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
            // A schedule set before the page's latest send says nothing of it: the page went out by its ready list while its queue was built.
            let current = next.filter(|next| !next.outdated(record.pushed_at));
            let schedule = match current {
                _ if sent.failures == LEFT => None,
                Some(next) if matches!(next.why, Why::Gone | Why::Leave) => None,
                Some(next) => Some(next),
                None if sent.failures >= GONE_AFTER => None,
                None => Some(standing(recrawl, site, domain, &rest, &record, &sent, now)),
            };
            let Some(schedule) = schedule else {
                if current.is_some_and(|next| next.why == Why::Leave) {
                    fixes.put(prefixed(SENT, domain.as_bytes(), &rest), Sent { failures: LEFT, ..sent }.pack());
                }
                fixes.delete(&key);
                fixes.delete(prefixed(NEXT, domain.as_bytes(), &rest));
                left.insert(rest);
                continue;
            };
            if schedule.due != due || (schedule.why == Why::Admit) != first || next != Some(schedule) {
                fixes.delete(&key);
                fixes.put(due_key(domain, schedule.due, schedule.why == Why::Admit, &rest), []);
                fixes.put(prefixed(NEXT, domain.as_bytes(), &rest), schedule.pack());
                later(&mut taken.head, schedule.due);
                continue;
            }
            // A page sent within the floor, such as a failure waiting its retry, waits; pages behind it may go.
            if let Some(floor) = floor {
                if record.pushed_at != 0 && now.saturating_sub(record.pushed_at) < floor {
                    later(&mut taken.floor_until, record.pushed_at.saturating_add(floor));
                    later(&mut taken.head, due);
                    continue;
                }
            }
            // A page robots now disallows moves back a day, so it does not hold the front.
            if !robots.allows_page(domain, &rest) {
                let moved = Scheduled { due: now.saturating_add(DAY), ..schedule };
                fixes.delete(&key);
                fixes.put(due_key(domain, moved.due, moved.why == Why::Admit, &rest), []);
                fixes.put(prefixed(NEXT, domain.as_bytes(), &rest), moved.pack());
                later(&mut taken.head, moved.due);
                continue;
            }
            let reason = match (floor, schedule.why) {
                (Some(_), _) => Reason::Pulled,
                (None, Why::Admit) => Reason::Fresh,
                (None, Why::Changed) => Reason::Changed,
                (None, Why::Failed | Why::Dropped) => Reason::Requeued,
                (None, _) => Reason::Refresh,
            };
            let entry = Entry { domain: domain.to_string(), rest: rest.clone(), order: Order::Due { due, first } };
            taken.picks.push(Pick { url: fetchable(&rest, record.flags), lastmod: record.lastmod, reason, failures: sent.failures, entry });
        }
        if !taken.picks.is_empty() {
            taken.floor_until = None;
        }
        if !fixes.is_empty() {
            let _guard = self.lock_ready();
            if !left.is_empty() {
                let size = self.set_size(domain)?.unwrap_or(0).saturating_sub(left.len() as u64);
                fixes.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
            }
            self.db.write(fixes)?;
        }
        Ok(taken)
    }

    /// Pages due by `now` at the front of the domain's queue, counted up to `most`.
    pub fn due_count(&self, domain: &str, now: u32, most: usize) -> Result<usize> {
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let mut options = bounded(&prefix);
        options.set_iterate_upper_bound([prefix.as_slice(), &now.saturating_add(1).to_be_bytes()].concat());
        let mut count = 0;
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), options).take(most) {
            item?;
            count += 1;
        }
        Ok(count)
    }

    /// A page taken from a due queue went out: it falls due again after its interval, or at once if it changed in flight.
    pub(crate) fn sent_from_queue(&self, batch: &mut WriteBatch, pick: &Pick, record: &Record, at: u32, recrawl: &Recrawl) {
        let Order::Due { due, first } = pick.entry.order else {
            return;
        };
        let domain = &pick.entry.domain;
        let next = if record.lastmod > pick.lastmod {
            Scheduled { why: Why::Changed, due: at, at }
        } else {
            let every = page_interval(recrawl, self.terms(domain).site, domain, record.first_seen, at);
            Scheduled { why: Why::Recrawl, due: at.saturating_add(every), at }
        };
        batch.delete(due_key(domain, due, first, &pick.entry.rest));
        batch.put(due_key(domain, next.due, false, &pick.entry.rest), []);
        batch.put(prefixed(NEXT, domain.as_bytes(), &pick.entry.rest), next.pack());
    }

    /// Outcomes for pages of domains with built queues: a failure falls due again after a day, the second in a row leaves the set, a page the index does not take leaves it at once, and a dropped page falls due as the reader set it.
    pub(crate) fn outcomes_into_queues(&self, batch: &mut WriteBatch, pages: &[&PageSchedule]) -> Result<()> {
        let records = self.records(pages.iter().map(|p| prefixed(URL, p.domain.as_bytes(), &p.rest)))?;
        let sents = self.sents(pages.iter().map(|p| prefixed(SENT, p.domain.as_bytes(), &p.rest)))?;
        let nexts = self.schedules(pages.iter().map(|p| prefixed(NEXT, p.domain.as_bytes(), &p.rest)))?;
        let keys: Vec<Option<Vec<u8>>> = pages.iter().zip(&nexts).map(|(p, next)| next.map(|next| due_key(&p.domain, next.due, next.why == Why::Admit, &p.rest))).collect();
        let mut members = self.present(keys.iter().flatten())?.into_iter();
        let mut left: HashMap<&str, u64> = HashMap::new();
        for (((page, record), sent), key) in pages.iter().zip(records).zip(sents).zip(keys) {
            let Some(record) = record else {
                continue;
            };
            // An outcome of an earlier send says nothing about the page's latest one.
            if record.pushed_at > page.next.at.saturating_add(STALE_OUTCOME) {
                continue;
            }
            let (domain, rest) = (page.domain.as_str(), page.rest.as_slice());
            let Some(key) = key.filter(|_| members.next().unwrap_or(false)) else {
                // A page outside the set stays out, and one the index does not take never joins; a take that meets an old key of it lets it go.
                if page.next.why == Why::Leave {
                    batch.put(prefixed(SENT, domain.as_bytes(), rest), Sent { failures: LEFT, ..sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() }) }.pack());
                    batch.delete(prefixed(NEXT, domain.as_bytes(), rest));
                }
                continue;
            };
            batch.delete(key);
            let next = match page.next.why {
                Why::Leave => {
                    let mut sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
                    sent.failures = LEFT;
                    batch.put(prefixed(SENT, domain.as_bytes(), rest), sent.pack());
                    batch.delete(prefixed(NEXT, domain.as_bytes(), rest));
                    *left.entry(domain).or_default() += 1;
                    continue;
                }
                Why::Failed => {
                    let mut sent = sent.unwrap_or(Sent { lastmod: record.lastmod, ..Sent::default() });
                    sent.failures = sent.failures.saturating_add(1);
                    batch.put(prefixed(SENT, domain.as_bytes(), rest), sent.pack());
                    if sent.failures >= GONE_AFTER {
                        batch.delete(prefixed(NEXT, domain.as_bytes(), rest));
                        *left.entry(domain).or_default() += 1;
                        continue;
                    }
                    Scheduled { why: Why::Failed, due: page.next.at.saturating_add(RETRY_AFTER), at: page.next.at }
                }
                _ => page.next,
            };
            batch.put(due_key(domain, next.due, false, rest), []);
            batch.put(prefixed(NEXT, domain.as_bytes(), rest), next.pack());
            self.notice_due(domain);
        }
        for (domain, gone) in left {
            let size = self.set_size(domain)?.unwrap_or(0).saturating_sub(gone);
            batch.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
        }
        Ok(())
    }

    /// A cap cut below the set: the pages seen longest ago leave until the set fits, their stored records kept but no longer crawled.
    pub fn shrink_set(&self, domain: &str, terms: Terms, pace: &mut dyn FnMut(u64) -> bool) -> Result<Built> {
        let mut built = Built::default();
        let (Some(size), Some(cap)) = (self.recount_set(domain, pace)?, set_cap(terms)) else {
            return Ok(built);
        };
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        // The newest `cap` pages stay: a heap of the newest kept, holding at most the cap.
        let mut newest: BinaryHeap<Reverse<(u32, Vec<u8>)>> = BinaryHeap::new();
        let mut leaving: Vec<Vec<u8>> = Vec::new();
        let mut last: Option<Vec<u8>> = None;
        while size > cap {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let keys: Vec<Vec<u8>> = self
                .db
                .iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix))
                .take(BUILD_CHUNK)
                .map(|item| item.map(|(key, _)| key.to_vec()))
                .collect::<Result<_, _>>()?;
            let read = keys.len();
            last = keys.last().cloned();
            built.scanned += read as u64;
            let records = self.records(keys.iter().filter_map(|key| due_parts(key, prefix.len())).map(|(_, _, rest)| prefixed(URL, domain.as_bytes(), rest)))?;
            for (key, record) in keys.into_iter().zip(records) {
                let first_seen = record.map_or(0, |record| record.first_seen);
                newest.push(Reverse((first_seen, key)));
                if newest.len() as u64 > cap {
                    if let Some(Reverse((_, key))) = newest.pop() {
                        leaving.push(key);
                    }
                }
            }
            if !pace(read as u64) {
                built.stopped = true;
                return Ok(built);
            }
            if read < BUILD_CHUNK {
                break;
            }
        }
        for chunk in leaving.chunks(ADMIT_CHUNK) {
            let mut batch = WriteBatch::default();
            for key in chunk {
                if let Some((_, _, rest)) = due_parts(key, prefix.len()) {
                    batch.delete(prefixed(NEXT, domain.as_bytes(), rest));
                }
                batch.delete(key);
            }
            let _guard = self.lock_ready();
            let size = self.set_size(domain)?.unwrap_or(0).saturating_sub(chunk.len() as u64);
            batch.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
            self.db.write(batch)?;
        }
        self.db.put(count_key_of(JOINED, domain), cap.to_le_bytes())?;
        self.notice_due(domain);
        Ok(built)
    }

    /// Count the domain's set from its queue keys, every page with one counted once, and correct its stored size by what the count found, keeping what changed it meanwhile.
    pub fn recount_set(&self, domain: &str, pace: &mut dyn FnMut(u64) -> bool) -> Result<Option<u64>> {
        let guard = self.lock_ready();
        let Some(before) = self.set_size(domain)? else {
            return Ok(None);
        };
        let snapshot = self.db.snapshot();
        drop(guard);
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let schedules = |keys: Vec<Vec<u8>>| -> Result<Vec<Option<Scheduled>>> {
            snapshot.multi_get(keys).into_iter().map(|raw| Ok(raw?.as_deref().and_then(Scheduled::unpack))).collect()
        };
        let mut counted = 0u64;
        // Pages whose schedule names no key of theirs: each counts once however many keys it has.
        let mut unnamed: HashSet<Vec<u8>> = HashSet::new();
        let mut last: Option<Vec<u8>> = None;
        loop {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let keys: Vec<Vec<u8>> = snapshot
                .iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix))
                .take(BUILD_CHUNK)
                .map(|item| item.map(|(key, _)| key.to_vec()))
                .collect::<Result<_, _>>()?;
            let read = keys.len();
            last = keys.last().cloned();
            let parts: Vec<(u32, bool, &[u8])> = keys.iter().filter_map(|key| due_parts(key, prefix.len())).collect();
            let nexts = schedules(parts.iter().map(|(_, _, rest)| prefixed(NEXT, domain.as_bytes(), rest)).collect())?;
            let mut named = Vec::new();
            for ((due, first, rest), next) in parts.iter().zip(nexts) {
                match next {
                    Some(next) if next.due == *due && (next.why == Why::Admit) == *first => counted += 1,
                    Some(next) => named.push((rest.to_vec(), due_key(domain, next.due, next.why == Why::Admit, rest))),
                    None => {
                        unnamed.insert(rest.to_vec());
                    }
                }
            }
            let found = snapshot.multi_get(named.iter().map(|(_, key)| key));
            for ((rest, _), found) in named.into_iter().zip(found) {
                if found?.is_none() {
                    unnamed.insert(rest);
                }
            }
            if !pace(read as u64) {
                return Ok(Some(before));
            }
            if read < BUILD_CHUNK {
                break;
            }
        }
        counted += unnamed.len() as u64;
        let _guard = self.lock_ready();
        let stored = self.set_size(domain)?.unwrap_or(before);
        let size = (stored + counted).saturating_sub(before);
        if size != stored {
            self.db.put(count_key_of(SET_SIZE, domain), size.to_le_bytes())?;
        }
        Ok(Some(size))
    }

    /// Undo the domain's queue for a rollback to a bot without queues: each schedule goes back on the waiting list or, for a change, the ready list, and the walker of that bot walks the domain again.
    pub fn unbuild_queue(&self, domain: &str) -> Result<u64> {
        let prefix = prefixed(NEXT, domain.as_bytes(), b"");
        let mut moved = 0;
        let mut last: Option<Vec<u8>> = None;
        loop {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let mut schedules: Vec<(Vec<u8>, Scheduled)> = Vec::new();
            let mut read = 0;
            for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix)).take(BUILD_CHUNK) {
                let (key, raw) = item?;
                read += 1;
                last = Some(key.to_vec());
                if let Some(next) = Scheduled::unpack(&raw) {
                    schedules.push((key[prefix.len()..].to_vec(), next));
                }
            }
            let records = self.records(schedules.iter().map(|(rest, _)| prefixed(URL, domain.as_bytes(), rest)))?;
            let sents = self.sents(schedules.iter().map(|(rest, _)| prefixed(SENT, domain.as_bytes(), rest)))?;
            let _guard = self.lock_ready();
            let mut batch = WriteBatch::default();
            let mut changed = Vec::new();
            for (((rest, next), record), sent) in schedules.into_iter().zip(records).zip(sents) {
                match next.why {
                    Why::Recrawl | Why::Dropped | Why::Gone => batch.put(timed_key(WAITING, next.due, domain.as_bytes(), &rest), []),
                    Why::Failed => {
                        // The queue counted the failure when it came; the waiting list counts it again when the retry is due.
                        if let Some(sent) = sent {
                            batch.put(prefixed(SENT, domain.as_bytes(), &rest), Sent { failures: sent.failures.saturating_sub(1), ..sent }.pack());
                        }
                        batch.put(timed_key(WAITING, next.due, domain.as_bytes(), &rest), []);
                    }
                    Why::Admit | Why::Leave => batch.delete(prefixed(NEXT, domain.as_bytes(), &rest)),
                    Why::Changed => {
                        batch.delete(prefixed(NEXT, domain.as_bytes(), &rest));
                        if let Some(record) = record {
                            changed.push(Entry { domain: domain.to_string(), rest, order: Order::Changed { lastmod: record.lastmod } });
                        }
                    }
                }
                moved += 1;
            }
            self.ready_back(&mut batch, changed)?;
            self.db.write(batch)?;
            if read < BUILD_CHUNK {
                break;
            }
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        batch.delete_range(prefixed(DUE, domain.as_bytes(), b""), [&[DUE], domain.as_bytes(), b"\x01"].concat());
        batch.delete(count_key_of(SET_SIZE, domain));
        batch.delete(count_key_of(JOINED, domain));
        self.db.write(batch)?;
        Ok(moved)
    }

    /// Clear what an earlier build left, and hold the domain's outcomes and sitemap changes from now until its queue is built.
    fn begin_build(&self, domain: &str) -> Result<()> {
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        batch.delete_range(prefixed(DUE, domain.as_bytes(), b""), [&[DUE], domain.as_bytes(), b"\x01"].concat());
        batch.delete(count_key_of(SET_SIZE, domain));
        self.db.write(batch)?;
        self.held.lock().unwrap_or_else(|e| e.into_inner()).insert(domain.to_string(), Held::default());
        Ok(())
    }
}

/// A page's fixed offset within `every` seconds, so overdue pages of one domain spread evenly.
fn spread(rest: &[u8], every: u32) -> u32 {
    let hash = rest.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    (hash % u64::from(every.max(1))) as u32
}

/// Where a page stands without a schedule to go by: a first crawl now, a change since its send now, else a refresh an interval after its send, spread over the next interval if already overdue. Each schedule's `at` is the send it follows.
fn standing(recrawl: &Recrawl, site: SiteType, domain: &str, rest: &[u8], record: &Record, sent: &Sent, now: u32) -> Scheduled {
    if record.pushed_at == 0 {
        return Scheduled { why: Why::Admit, due: now, at: 0 };
    }
    if record.lastmod > sent.lastmod {
        return Scheduled { why: Why::Changed, due: now, at: record.pushed_at };
    }
    let every = page_interval(recrawl, site, domain, record.first_seen, record.pushed_at);
    let mut due = record.pushed_at.saturating_add(every);
    // A refresh already overdue goes some time within its next interval, not all at once ahead of first crawls.
    if due < now {
        due = now.saturating_add(spread(rest, every));
    }
    Scheduled { why: Why::Recrawl, due, at: record.pushed_at }
}

fn admit_into(batch: &mut WriteBatch, domain: &str, rest: &[u8], now: u32) {
    batch.put(due_key(domain, now, true, rest), []);
    batch.put(prefixed(NEXT, domain.as_bytes(), rest), Scheduled { why: Why::Admit, due: now, at: 0 }.pack());
}

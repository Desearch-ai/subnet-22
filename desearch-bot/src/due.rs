//! Each listed domain's set of pages as a due queue: every page has one due time, and a pass takes what is due.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use anyhow::Result;
use rocksdb::{Direction, IteratorMode, WriteBatch};

use crate::allowed::{SiteType, Terms};
use crate::buckets::{BucketStore, URL};
use crate::ready::{
    bounded, count_key_of, fetchable, junk, prefixed, Entry, Order, PageSchedule, Pick, Reason, Scheduled, Sent, Why, GONE_AFTER, JOINED, NEXT, SENT,
    STALE_OUTCOME, STOCK,
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
const LEFT: u8 = 127;

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

/// What a take from a domain's due queue found.
#[derive(Debug, Default)]
pub struct DueTake {
    pub picks: Vec<Pick>,
    /// When the first page left after the picks is due; None when none is left.
    pub head: Option<u32>,
    /// For a take that pulls refreshes forward: the front page was sent within the floor and may go from this time.
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
        self.clear_queue(domain)?;
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
                let schedule = match current {
                    Some(next) if matches!(next.why, Why::Failed | Why::Dropped) => next,
                    _ if record.lastmod > sent.lastmod => Scheduled { why: Why::Changed, due: now, at: now },
                    _ => {
                        let every = page_interval(recrawl, terms.site, domain, record.first_seen, record.pushed_at);
                        let mut due = record.pushed_at.saturating_add(every);
                        // A refresh already overdue goes some time within its next interval, not all at once ahead of first crawls.
                        if due < now {
                            due = now.saturating_add(spread(rest, every));
                        }
                        Scheduled { why: Why::Recrawl, due, at: record.pushed_at }
                    }
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
        self.db.write(batch)?;
        drop(_guard);
        self.notice_due(domain);
        Ok(built)
    }

    /// Admit the newest stored pages never sent and not yet in the set, up to the room its cap leaves.
    pub fn admit_more(&self, domain: &str, terms: Terms, now: u32, pace: &mut dyn FnMut(u64) -> bool) -> Result<Built> {
        let mut built = Built::default();
        let Some(size) = self.set_size(domain)? else {
            return Ok(built);
        };
        let room = set_cap(terms).map_or(u64::MAX, |cap| cap.saturating_sub(size));
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(URL, domain.as_bytes(), b"");
        let mut newest: BinaryHeap<Reverse<(u32, u32, Vec<u8>)>> = BinaryHeap::new();
        let mut last: Option<Vec<u8>> = None;
        while room > 0 {
            let start = last.as_ref().map_or_else(|| prefix.clone(), |key| [key.as_slice(), b"\0"].concat());
            let mut unsent: Vec<(Vec<u8>, Record)> = Vec::new();
            let mut read = 0;
            for item in self.db.iterator_opt(IteratorMode::From(&start, Direction::Forward), bounded(&prefix)).take(BUILD_CHUNK) {
                let (key, raw) = item?;
                read += 1;
                last = Some(key.to_vec());
                if let Some(record) = Record::unpack(&raw).filter(|record| record.pushed_at == 0) {
                    unsent.push((key.to_vec(), record));
                }
            }
            built.scanned += read as u64;
            let nexts = self.schedules(unsent.iter().map(|(key, _)| prefixed(NEXT, domain.as_bytes(), &key[prefix.len()..])))?;
            for ((key, record), next) in unsent.into_iter().zip(nexts) {
                let rest = &key[prefix.len()..];
                if next.is_none() && wanted(terms.site, domain, rest, &robots) {
                    newest.push(Reverse((record.first_seen, record.lastmod, rest.to_vec())));
                    if newest.len() as u64 > room {
                        newest.pop();
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
        let mut kept: Vec<(u32, u32, Vec<u8>)> = newest.into_iter().map(|Reverse(page)| page).collect();
        kept.sort_unstable_by(|a, b| b.cmp(a));
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
        let size = self.set_size(domain)?.unwrap_or(size) + built.admitted;
        let mut batch = WriteBatch::default();
        batch.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
        batch.put(count_key_of(JOINED, domain), set_cap(terms).unwrap_or(u64::MAX).to_le_bytes());
        self.db.write(batch)?;
        drop(_guard);
        if built.admitted > 0 {
            self.notice_due(domain);
        }
        Ok(built)
    }

    /// A sitemap read of a domain with a built queue: new pages join while its cap leaves room (news rolls, its latest-due page leaving), and changed pages fall due now; the caller holds the lock and writes the batch.
    pub(crate) fn listing_into_queue(&self, batch: &mut WriteBatch, domain: &str, fresh: &[(&[u8], &Record)], moved: &[(&[u8], u32)], now: u32) -> Result<usize> {
        let terms = self.terms(domain);
        let Some(mut size) = self.set_size(domain)? else {
            return Ok(0);
        };
        let cap = set_cap(terms);
        let mut admitted = 0;
        let mut rolled = 0;
        for (rest, _) in fresh {
            if junk(rest) || (terms.site == SiteType::News && front_page(rest)) {
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
            for (((rest, lastmod), next), sent) in moved.iter().zip(nexts).zip(sents) {
                if sent.is_some_and(|sent| *lastmod <= sent.lastmod || sent.failures == LEFT) {
                    continue;
                }
                match next {
                    Some(next) => {
                        batch.delete(due_key(domain, next.due, next.why == Why::Admit, rest));
                    }
                    // A gone page listed again with a newer lastmod is a candidate once more.
                    None if cap.is_none_or(|cap| size < cap) && !junk(rest) => size += 1,
                    None => continue,
                }
                let changed = Scheduled { why: Why::Changed, due: now, at: now };
                batch.put(due_key(domain, now, false, rest), []);
                batch.put(prefixed(NEXT, domain.as_bytes(), rest), changed.pack());
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

    /// Up to `want` pages from the front of the domain's queue due by `now`; with a `floor`, the soonest due whatever their time, if last sent at least `floor` ago. Stale keys go.
    pub fn take_due(&self, domain: &str, want: usize, now: u32, floor: Option<u32>) -> Result<DueTake> {
        let robots = self.domain(domain)?.map(|record| records::robots_paths(&record)).unwrap_or_default();
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        let mut taken = DueTake::default();
        let mut keys: Vec<(Vec<u8>, u32, bool)> = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(&prefix, Direction::Forward), bounded(&prefix)) {
            let key = item?.0.to_vec();
            let Some((due, first, _)) = due_parts(&key, prefix.len()) else {
                continue;
            };
            if (floor.is_none() && due > now) || keys.len() >= want + TAKE_SLACK {
                taken.head = Some(due);
                break;
            }
            keys.push((key, due, first));
        }
        let rests: Vec<Vec<u8>> = keys.iter().map(|(key, _, _)| key[prefix.len() + 5..].to_vec()).collect();
        let records = self.records(rests.iter().map(|rest| prefixed(URL, domain.as_bytes(), rest)))?;
        let sents = self.sents(rests.iter().map(|rest| prefixed(SENT, domain.as_bytes(), rest)))?;
        let nexts = self.schedules(rests.iter().map(|rest| prefixed(NEXT, domain.as_bytes(), rest)))?;
        let mut stale = WriteBatch::default();
        let mut left = 0;
        for ((((key, due, first), rest), record), (sent, next)) in keys.into_iter().zip(rests).zip(records).zip(sents.into_iter().zip(nexts)) {
            if taken.picks.len() >= want {
                taken.head = Some(taken.head.map_or(due, |head| head.min(due)));
                break;
            }
            // The page's schedule is its authority: a key it no longer names is stale.
            let Some(next) = next.filter(|next| next.due == due) else {
                stale.delete(&key);
                continue;
            };
            let Some(record) = record else {
                stale.delete(&key);
                stale.delete(prefixed(NEXT, domain.as_bytes(), &rest));
                left += 1;
                continue;
            };
            if let Some(floor) = floor {
                if record.pushed_at != 0 && now.saturating_sub(record.pushed_at) < floor {
                    taken.floor_until = Some(record.pushed_at.saturating_add(floor));
                    taken.head = Some(due);
                    break;
                }
            }
            // A page robots now disallows moves back a day, so it does not hold the front.
            if !robots.allows_page(domain, &rest) {
                let later = Scheduled { due: now.saturating_add(DAY), ..next };
                stale.delete(&key);
                stale.put(due_key(domain, later.due, false, &rest), []);
                stale.put(prefixed(NEXT, domain.as_bytes(), &rest), later.pack());
                continue;
            }
            let reason = match (floor, next.why) {
                (Some(_), _) => Reason::Pulled,
                (None, Why::Admit) => Reason::Fresh,
                (None, Why::Changed) => Reason::Changed,
                (None, Why::Failed | Why::Dropped) => Reason::Requeued,
                (None, _) => Reason::Refresh,
            };
            let failures = sent.map_or(0, |sent| sent.failures);
            let entry = Entry { domain: domain.to_string(), rest: rest.clone(), order: Order::Due { due, first } };
            taken.picks.push(Pick { url: fetchable(&rest, record.flags), lastmod: record.lastmod, reason, failures, entry });
        }
        if left > 0 {
            stale.put(count_key_of(SET_SIZE, domain), self.set_size(domain)?.unwrap_or(0).saturating_sub(left).to_le_bytes());
        }
        if !stale.is_empty() {
            let _guard = self.lock_ready();
            self.db.write(stale)?;
        }
        Ok(taken)
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
        let mut left: HashMap<&str, u64> = HashMap::new();
        for (((page, record), sent), current) in pages.iter().zip(records).zip(sents).zip(nexts) {
            let (Some(record), Some(current)) = (record, current) else {
                continue;
            };
            // An outcome of an earlier send says nothing about the page's latest one.
            if record.pushed_at > page.next.at.saturating_add(STALE_OUTCOME) {
                continue;
            }
            let (domain, rest) = (page.domain.as_str(), page.rest.as_slice());
            batch.delete(due_key(domain, current.due, current.why == Why::Admit, rest));
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
        let (Some(size), Some(cap)) = (self.set_size(domain)?, set_cap(terms)) else {
            return Ok(built);
        };
        let over = size.saturating_sub(cap);
        let prefix = prefixed(DUE, domain.as_bytes(), b"");
        // The newest `cap` pages stay: a heap of the newest kept, holding at most the cap.
        let mut newest: BinaryHeap<Reverse<(u32, Vec<u8>)>> = BinaryHeap::new();
        let mut leaving: Vec<Vec<u8>> = Vec::new();
        let mut last: Option<Vec<u8>> = None;
        while over > 0 {
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
            self.db.write(batch)?;
        }
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        let size = self.set_size(domain)?.unwrap_or(size).saturating_sub(leaving.len() as u64);
        batch.put(count_key_of(SET_SIZE, domain), size.to_le_bytes());
        batch.put(count_key_of(JOINED, domain), cap.to_le_bytes());
        self.db.write(batch)?;
        drop(_guard);
        self.notice_due(domain);
        Ok(built)
    }

    fn clear_queue(&self, domain: &str) -> Result<()> {
        let _guard = self.lock_ready();
        let mut batch = WriteBatch::default();
        batch.delete_range(prefixed(DUE, domain.as_bytes(), b""), [&[DUE], domain.as_bytes(), b"\x01"].concat());
        batch.delete(count_key_of(SET_SIZE, domain));
        Ok(self.db.write(batch)?)
    }
}

/// A page's fixed offset within `every` seconds, so overdue pages of one domain spread evenly.
fn spread(rest: &[u8], every: u32) -> u32 {
    let hash = rest.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    (hash % u64::from(every.max(1))) as u32
}

fn admit_into(batch: &mut WriteBatch, domain: &str, rest: &[u8], now: u32) {
    batch.put(due_key(domain, now, true, rest), []);
    batch.put(prefixed(NEXT, domain.as_bytes(), rest), Scheduled { why: Why::Admit, due: now, at: now }.pack());
}

//! The crawl loop: many visits in flight across the buckets this process owns, and what they learn kept.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::Rng;
use serde_json::Value;
use tokio::sync::{watch, OwnedSemaphorePermit};
use tokio::task::{Id, JoinError, JoinHandle, JoinSet};
use tokio::time::Instant;

use crate::buckets::{Buckets, Changes, Json};
use crate::links::{first_wave, LinkPace, LinkQueue};
use crate::registry::{Change, Registry, Report};
use crate::schedule::{HOUR, SECOND};
use crate::states::{self, Outcome, State};
use crate::suffixes::tld_group;
use crate::timetable::{placed, HeavyQueue, Lane, Timetable};
use crate::visit::{is_listing, robots_allow, Crash, FollowLinks, Known, LinksAt, Visit, Visitor, LINKS_KIND};
use crate::{exclusions, records};

const TICK: Duration = Duration::from_secs(1);
const FLUSH_SECONDS: Duration = Duration::from_secs(5);
const FLUSH_VISITS: usize = 200;
/// Postgres hears from the loop in bulk, never once per visit.
const SYNC_SECONDS: Duration = Duration::from_secs(30);
const PRINT_SECONDS: Duration = Duration::from_secs(30);
const CRASH_RETRY: i64 = HOUR;
/// On shutdown, visits still running after this long are dropped; they are simply due again.
const STOP_GRACE: Duration = Duration::from_secs(30);
/// A visit cut short by failing requests leaves the site alone this long.
const CUT_SHORT_WAIT: i64 = HOUR;
/// A domain with more sitemap records than this needs a heavy slot to be visited.
const HEAVY_RECORDS: usize = 2_000;
/// New visits wait while the disk is this close to full; they resume with 5 GB more to spare.
const RESUME_MARGIN: u64 = 5 << 30;
const DISK_CHECK: Duration = Duration::from_secs(10);
/// Percent of the visit slots each lane can count on: refresh, backlog, discovery.
const SHARES: [usize; 3] = [50, 20, 30];
/// Listed domains woken at once are spread this far apart, over a day at most, or over three days when there are this many, as on a first start.
const WAKE_GAP: i64 = 2 * SECOND;
const WAKE_SPREAD: i64 = 24 * HOUR;
const WAKE_MANY: usize = 20_000;
/// Domains made due on one pass of the loop, which writes each one's record.
const WAKE_CHUNK: usize = 1_000;
/// Sites whose link work waits its turn, at most; one turned away is offered again by its next visit.
const LINKS_QUEUED: usize = 200_000;
/// A site visited when its link work comes up takes it this many seconds later.
const LINKS_BUSY_WAIT: i64 = 60;

#[derive(Clone, Debug)]
pub struct DomainWrite {
    pub host: String,
    pub state: State,
    pub reason: Option<String>,
    pub failures: i64,
    pub next_due_at: Option<i64>,
    pub last_ok_at: Option<i64>,
    pub canonical_host: Option<String>,
    pub checked_at: i64,
    /// Due again at once, to keep reading sitemaps found but not read yet.
    pub backlog: bool,
    pub visit: Visit,
}

/// The state a visit leaves a domain in, and when the loop should come back to it.
pub fn plan(known: &Known, visit: Visit, now: i64, rng: &mut impl Rng) -> DomainWrite {
    // A refusal that may pass, on a recheck, is tried again before its outcome counts.
    let retry = (visit.passing && known.state == State::New).then(|| states::recheck_retry(known.failures, now, rng)).flatten();
    let decision = retry.unwrap_or_else(|| states::decide(known.state, visit.outcome, known.failures, known.last_ok_at, now, rng));
    let mut due = decision.next_check_at;
    if decision.state == State::Active {
        let earliest = if visit.deferred.is_empty() { earliest_sitemap(known, &visit, now) } else { Some(now) };
        if let Some(earliest) = earliest.filter(|&e| due.is_none_or(|d| e < d)) {
            due = Some(earliest);
        }
        if visit.cut_short {
            due = due.map(|d| d.max(now + CUT_SHORT_WAIT));
        }
        // A visit that asked nothing read every file it could: due again at once, it would ask nothing again.
        if visit.requests == 0 && !visit.paused && due.is_some_and(|d| d <= now) {
            due = Some(now + CUT_SHORT_WAIT);
        }
    }
    let canonical_host = match visit.outcome {
        Outcome::Redirect => visit.canonical_host.clone(),
        outcome if outcome.answered() => None,
        _ => known.canonical_host.clone(),
    };
    let last_ok_at = if visit.outcome.answered() && visit.requests - visit.aside > 0 { Some(now) } else { known.last_ok_at };
    DomainWrite {
        host: known.host.clone(),
        state: decision.state,
        reason: visit.reason.clone(),
        failures: decision.failures,
        next_due_at: due,
        last_ok_at,
        canonical_host,
        checked_at: now,
        backlog: decision.state == State::Active && due.is_some_and(|d| d <= now),
        visit,
    }
}

/// What link work alone leaves a domain: its standing and its due time as they were.
pub fn links_plan(known: &Known, visit: Visit, now: i64) -> DomainWrite {
    DomainWrite {
        host: known.host.clone(),
        state: known.state,
        reason: None,
        failures: known.failures,
        next_due_at: None,
        last_ok_at: known.last_ok_at,
        canonical_host: known.canonical_host.clone(),
        checked_at: now,
        backlog: false,
        visit,
    }
}

/// A visit that failed on our side leaves the domain as it was, to be tried again later.
pub fn crashed(known: &Known, error: &str, now: i64) -> DomainWrite {
    DomainWrite {
        host: known.host.clone(),
        state: known.state,
        reason: Some(format!("crashed: {error}")),
        failures: known.failures,
        next_due_at: Some(now + CRASH_RETRY),
        last_ok_at: known.last_ok_at,
        canonical_host: known.canonical_host.clone(),
        checked_at: now,
        backlog: false,
        visit: Visit::new(&known.host),
    }
}

/// How a domain reached through a redirect joins the list: its group, state, and why.
pub fn adopted(target: &str) -> (&'static str, State, Option<&'static str>) {
    let group = tld_group(target);
    let reason = exclusions::exclusion_reason(target, group);
    (group, if reason.is_some() { State::Excluded } else { State::New }, reason)
}

fn earliest_sitemap(known: &Known, visit: &Visit, now: i64) -> Option<i64> {
    let bare = known.host.strip_prefix("www.").unwrap_or(&known.host);
    let updated: HashMap<&str, Option<i64>> = visit.sitemaps.iter().map(|u| (u.url.as_str(), u.next_check_at)).collect();
    let mut times: Vec<i64> = known
        .sitemaps
        .values()
        // A homepage or link listing left unread while due sets no time: it is read once the site qualifies again.
        .filter(|s| updated.contains_key(s.url.as_str()) || !is_listing(s.kind.as_deref()) || s.next_check_at.is_some_and(|next| next > now))
        // Nor does a file robots.txt disallows, never read while it does; the next robots.txt read decides again.
        .filter(|s| updated.contains_key(s.url.as_str()) || robots_allow(&visit.robots_paths, bare, &s.url))
        .map(|s| updated.get(s.url.as_str()).copied().unwrap_or(s.next_check_at).unwrap_or(now))
        .collect();
    times.extend(updated.iter().filter(|(url, _)| !known.sitemaps.contains_key(**url)).filter_map(|(_, when)| *when));
    times.into_iter().min()
}

/// Seconds that many woken domains are spread over.
fn wake_spread(hosts: usize) -> i64 {
    let spread = if hosts > WAKE_MANY { 3 * WAKE_SPREAD } else { (hosts as i64 * WAKE_GAP).min(WAKE_SPREAD) };
    spread / SECOND
}

/// Listed domains found without sitemaps on a visit made while unlisted.
fn unread(buckets: &Buckets) -> Vec<String> {
    buckets.listed().into_iter().filter(|host| buckets.store(host).domain(host).ok().flatten().is_some_and(|record| records::listed_unread(&record))).collect()
}

pub fn now_micros() -> i64 {
    chrono::Utc::now().timestamp_micros()
}

/// Visit one domain and decide what comes next; a failure on our side never escapes.
pub async fn once(known: Arc<Known>, visitor: &Visitor, excluded: &HashSet<String>) -> DomainWrite {
    let visit = if known.categories.iter().any(|c| excluded.contains(c)) || exclusions::blocked_operator(&known.host) {
        Ok(Visit { outcome: Outcome::Excluded, reason: Some("excluded".into()), ..Visit::new(&known.host) })
    } else {
        visitor.visit(known.clone(), now_micros()).await
    };
    match visit {
        Ok(visit) => plan(&known, visit, now_micros(), &mut rand::thread_rng()),
        Err(Crash(error)) => crashed(&known, &error, now_micros()),
    }
}

enum Finished {
    Skipped,
    /// A heavy domain visited without a heavy slot; it waits in the heavy queue for one, not in a visit slot.
    Busy,
    Visited(Box<DomainWrite>, Arc<Known>),
}

#[derive(Default)]
pub struct Stats {
    pub visited: u64,
    pub requests: i64,
    pub new: i64,
    pub listed: i64,
    pub outcomes: BTreeMap<String, u64>,
    /// Visits finished in each lane.
    pub lanes: [u64; 3],
    pub link_errors: u64,
    /// Link runs that read the site, and link work that found it no longer qualifies or waited.
    pub link_runs: u64,
    pub link_dropped: u64,
    pub link_waits: u64,
}

/// Keeps many visits in flight across the buckets one process owns.
pub struct Loop {
    buckets: Arc<Buckets>,
    visitor: Arc<Visitor>,
    concurrency: usize,
    registry: Registry,
    excluded: Arc<HashSet<String>>,
    timetable: Timetable,
    heavy: HeavyQueue,
    shares: [usize; 3],
    running: [usize; 3],
    visits: JoinSet<Finished>,
    tasks: HashMap<Id, (Arc<str>, Lane)>,
    busy: HashSet<Arc<str>>,
    pending: Vec<(Box<DomainWrite>, Arc<Known>)>,
    unreported: Vec<Report>,
    pub stats: Stats,
    started: Instant,
    printed: Instant,
    min_free_disk: u64,
    disk_checked: Option<Instant>,
    disk_low: bool,
    /// The list version whose domains were last looked at for ones visited while it did not name them, and that look while it runs.
    woken: u64,
    waking: Option<JoinHandle<Vec<String>>>,
    /// Domains that look found still to be made due, by when at the latest, in Unix seconds.
    to_wake: Vec<String>,
    wake_by: i64,
    /// Requests made per domain, kept only when asked for, as a probe does.
    tally: Option<HashMap<String, i64>>,
    /// Link work waiting its turn, and the runners that take it, apart from the visits.
    links: LinkQueue,
    link_pace: LinkPace,
    link_runs: JoinSet<LinkFinished>,
    link_tasks: HashMap<Id, Arc<str>>,
    links_spread: i64,
}

impl Loop {
    pub fn new(buckets: Arc<Buckets>, visitor: Arc<Visitor>, concurrency: usize, registry: Registry, excluded: HashSet<String>) -> Self {
        Loop {
            buckets,
            visitor,
            concurrency,
            registry,
            excluded: Arc::new(excluded),
            timetable: Timetable::default(),
            heavy: HeavyQueue::default(),
            shares: SHARES,
            running: [0; 3],
            visits: JoinSet::new(),
            tasks: HashMap::new(),
            busy: HashSet::new(),
            pending: Vec::new(),
            unreported: Vec::new(),
            stats: Stats::default(),
            started: Instant::now(),
            printed: Instant::now(),
            min_free_disk: 0,
            disk_checked: None,
            disk_low: false,
            woken: 0,
            waking: None,
            to_wake: Vec::new(),
            wake_by: 0,
            tally: None,
            links: LinkQueue::new(LINKS_QUEUED),
            link_pace: LinkPace::new(0),
            link_runs: JoinSet::new(),
            link_tasks: HashMap::new(),
            links_spread: 0,
        }
    }

    /// Link runs started an hour at most, 0 for no limit, and how long after a site first qualifies its first run may come.
    pub fn with_links(mut self, per_hour: u64, first_spread: i64) -> Self {
        self.link_pace = LinkPace::new(per_hour);
        self.links_spread = first_spread;
        self
    }

    /// Sites whose link work waits its turn, and the bytes the queue holds.
    pub fn links_queued(&self) -> (usize, usize) {
        (self.links.len(), self.links.bytes())
    }

    /// Stop starting visits while the disk has less than this many bytes free.
    pub fn with_min_free_disk(mut self, bytes: u64) -> Self {
        self.min_free_disk = bytes;
        self
    }

    /// Keep these percents of the visit slots for domains never checked and for sitemap backlogs; refresh gets the rest.
    /// Keep each domain's request count, for `requests_by_domain`.
    pub fn with_tally(mut self) -> Self {
        self.tally = Some(HashMap::new());
        self
    }

    pub fn requests_by_domain(&self) -> Option<&HashMap<String, i64>> {
        self.tally.as_ref()
    }

    pub fn with_shares(mut self, discovery: usize, backlog: usize) -> Self {
        self.shares = [100usize.saturating_sub(discovery + backlog), backlog, discovery];
        self
    }

    /// Put every domain the stores hold into the timetable; returns how many are due ever.
    pub fn load(&mut self) -> Result<usize> {
        let stores: Vec<_> = self.buckets.stores().collect();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let schedules = std::thread::scope(|scope| {
            let handles: Vec<_> = stores
                .chunks(stores.len().div_ceil(threads).max(1))
                .map(|chunk| scope.spawn(move || chunk.iter().map(|store| store.schedule()).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("loading a store panicked")).collect::<Vec<_>>()
        });
        let following = self.visitor.follow_links != FollowLinks::Off;
        for schedule in schedules {
            for (host, due) in schedule? {
                if let Some(state) = State::parse(&due.state) {
                    let place = self.place(&host, due.rank);
                    self.timetable.set(&host, Lane::of(state, false), due.due, Some(place));
                }
                if let Some(at) = due.links_due.filter(|_| following) {
                    self.links.push(&host, at);
                }
            }
        }
        Ok(self.timetable.len())
    }

    pub async fn run(&mut self, mut stop: watch::Receiver<bool>) -> Result<()> {
        self.started = Instant::now();
        self.sync().await;
        let (mut flushed, mut synced) = (Instant::now(), Instant::now());
        while !*stop.borrow() {
            self.fill();
            tokio::select! {
                _ = stop.changed() => {}
                Some(done) = self.visits.join_next_with_id(), if !self.visits.is_empty() => {
                    self.finish(done);
                    while let Some(done) = self.visits.try_join_next_with_id() {
                        self.finish(done);
                    }
                }
                Some(done) = self.link_runs.join_next_with_id(), if !self.link_runs.is_empty() => {
                    self.finish_links(done);
                }
                _ = tokio::time::sleep(TICK) => {}
            }
            if self.pending.len() >= FLUSH_VISITS || (!self.pending.is_empty() && flushed.elapsed() >= FLUSH_SECONDS) {
                self.flush()?;
                flushed = Instant::now();
            }
            if synced.elapsed() >= SYNC_SECONDS {
                self.sync().await;
                synced = Instant::now();
            }
            if self.printed.elapsed() >= PRINT_SECONDS {
                self.print();
            }
        }
        let deadline = Instant::now() + STOP_GRACE;
        while !self.visits.is_empty() {
            match tokio::time::timeout_at(deadline, self.visits.join_next_with_id()).await {
                Ok(Some(done)) => self.finish(done),
                _ => break,
            }
        }
        self.visits.abort_all();
        while !self.link_runs.is_empty() {
            match tokio::time::timeout_at(deadline, self.link_runs.join_next_with_id()).await {
                Ok(Some(done)) => self.finish_links(done),
                _ => break,
            }
        }
        self.link_runs.abort_all();
        self.flush()?;
        self.sync().await;
        Ok(())
    }

    /// Visit everything due now, wait for those visits, and keep what they learned.
    pub async fn step(&mut self) -> Result<usize> {
        self.fill();
        let mut finished = 0;
        while let Some(done) = self.visits.join_next_with_id().await {
            self.finish(done);
            finished += 1;
        }
        self.flush()?;
        // Link work taken up now runs to its end, as the visits did.
        loop {
            self.fill_links(now_micros() / SECOND);
            match self.link_runs.join_next_with_id().await {
                Some(done) => {
                    self.finish_links(done);
                    self.flush()?;
                }
                None => break,
            }
        }
        self.sync().await;
        if let Some(scan) = self.waking.take() {
            self.woke(scan.await.unwrap_or_default());
        }
        while !self.to_wake.is_empty() {
            self.wake();
        }
        Ok(finished)
    }

    fn fill(&mut self) {
        if self.disk_checked.is_none_or(|at| at.elapsed() >= DISK_CHECK) {
            self.disk_checked = Some(Instant::now());
            if let Some(free) = self.buckets.free_disk() {
                let low = if self.disk_low { free < self.min_free_disk + RESUME_MARGIN } else { free < self.min_free_disk };
                if low != self.disk_low {
                    println!("[rs] {} new visits: {} GB free on disk", if low { "pausing" } else { "resuming" }, free >> 30);
                }
                self.disk_low = low;
                self.visitor.pause.store(low, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if self.disk_low {
            return;
        }
        // Heavy slots go to the best placed heavy domains waiting, never by luck of timing.
        while self.visits.len() < self.concurrency && !self.heavy.is_empty() {
            let Some(permit) = self.visitor.heavy.try_take() else {
                break;
            };
            let Some((host, lane)) = self.heavy.pop(now_micros() / SECOND) else {
                break;
            };
            self.start_with(host, lane, Some(permit));
        }
        let mut free = self.concurrency.saturating_sub(self.visits.len());
        let now = now_micros() / SECOND;
        // Each lane first fills its own share; slots a lane cannot use go to the lanes that can.
        for share_only in [true, false] {
            for lane in Lane::ALL {
                let room = if share_only { self.share(lane).saturating_sub(self.running[lane.index()]).min(free) } else { free };
                if room == 0 {
                    continue;
                }
                for host in self.timetable.take(lane, room, now) {
                    if self.start(host, lane) {
                        free -= 1;
                    }
                }
            }
        }
        self.fill_links(now);
    }

    /// Runners free and the hourly pace allowing, take the earliest due link work; a site being visited waits a minute.
    fn fill_links(&mut self, now: i64) {
        if self.disk_low || self.visitor.follow_links == FollowLinks::Off {
            return;
        }
        while self.link_runs.len() < self.visitor.links.size() && self.link_pace.ready() {
            let Some(host) = self.links.pop_due(now) else {
                break;
            };
            if !self.busy.insert(host.clone()) {
                self.links.push(&host, now + LINKS_BUSY_WAIT);
                continue;
            }
            self.link_pace.take();
            let handle = self.link_runs.spawn(links_one(host.clone(), self.buckets.clone(), self.visitor.clone(), self.excluded.clone()));
            self.link_tasks.insert(handle.id(), host);
        }
    }

    fn finish_links(&mut self, done: Result<(Id, LinkFinished), JoinError>) {
        let id = match &done {
            Ok((id, _)) => *id,
            Err(error) => error.id(),
        };
        let Some(host) = self.link_tasks.remove(&id) else {
            return;
        };
        match done {
            // Waiting its turn costs no write: only the queue knows.
            Ok((_, LinkFinished::Done(write, _))) if write.visit.links_wait.is_some() && write.visit.sitemaps.is_empty() && !write.visit.robots_read => {
                self.stats.link_waits += 1;
                self.link_pace.refund();
                self.links.push(&host, write.visit.links_wait.unwrap_or_default() / SECOND);
                self.release(&host);
            }
            Ok((_, LinkFinished::Done(write, known))) => self.pending.push((write, known)),
            Ok((_, LinkFinished::Gone)) => self.release(&host),
            Err(error) => {
                eprintln!("[rs] link work on {host} failed: {error}");
                self.release(&host);
            }
        }
    }

    /// Frees a site link work held without writing it, back in the timetable by its record, as a visit due meanwhile was turned away.
    fn release(&mut self, host: &str) {
        self.busy.remove(host);
        if let Ok(Some(record)) = self.buckets.store(host).domain(host) {
            let backlog = self.backlog_of(host, &record);
            self.schedule(host, &record, backlog);
        }
    }

    /// Whether a site goes back in the backlog lane: where it waits now, or, taken off meanwhile, active and due already.
    fn backlog_of(&self, host: &str, record: &Json) -> bool {
        match self.timetable.lane(host) {
            Some(lane) => lane == Lane::Backlog,
            None => record.get("state").and_then(Value::as_str) == Some(State::Active.as_str()) && records::int(record.get("due")).is_some_and(|due| due <= now_micros() / SECOND),
        }
    }

    fn share(&self, lane: Lane) -> usize {
        self.concurrency * self.shares[lane.index()] / 100
    }

    fn start(&mut self, host: Arc<str>, lane: Lane) -> bool {
        self.start_with(host, lane, None)
    }

    fn start_with(&mut self, host: Arc<str>, lane: Lane, heavy: Option<OwnedSemaphorePermit>) -> bool {
        if !self.busy.insert(host.clone()) {
            return false;
        }
        let job = visit_one(host.clone(), self.buckets.clone(), self.visitor.clone(), self.excluded.clone(), heavy);
        let handle = self.visits.spawn(job);
        self.tasks.insert(handle.id(), (host, lane));
        self.running[lane.index()] += 1;
        true
    }

    fn finish(&mut self, done: Result<(Id, Finished), JoinError>) {
        let id = match &done {
            Ok((id, _)) => *id,
            Err(error) => error.id(),
        };
        let Some((host, lane)) = self.tasks.remove(&id) else {
            return;
        };
        self.running[lane.index()] -= 1;
        match done {
            Ok((_, Finished::Skipped)) => {
                self.busy.remove(&host);
            }
            Ok((_, Finished::Busy)) => {
                self.busy.remove(&host);
                if let Ok(Some(record)) = self.buckets.store(&host).domain(&host) {
                    let place = self.place(&host, records::int(record.get("rank")));
                    self.heavy.push(host, lane, place, now_micros() / SECOND);
                }
            }
            Ok((_, Finished::Visited(write, known))) => {
                self.stats.visited += 1;
                self.stats.lanes[lane.index()] += 1;
                self.pending.push((write, known));
            }
            Err(error) => {
                eprintln!("[rs] visit to {host} failed: {error}");
                self.busy.remove(&host);
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        for (write, known) in std::mem::take(&mut self.pending) {
            self.save(*write, &known)?;
        }
        Ok(())
    }

    /// Write a finished visit to its store and put the domain back in the timetable.
    fn save(&mut self, write: DomainWrite, known: &Known) -> Result<()> {
        self.busy.remove(write.host.as_str());
        self.stats.requests += write.visit.requests;
        if let Some(tally) = self.tally.as_mut() {
            *tally.entry(write.host.clone()).or_default() += write.visit.requests;
        }
        self.stats.new += write.visit.new;
        self.stats.listed += write.visit.listed;
        if !write.visit.links_only {
            let outcome = format!("{} {}", write.state.as_str(), write.reason.as_deref().unwrap_or("-"));
            *self.stats.outcomes.entry(outcome).or_default() += 1;
        }
        if let Some(error) = &write.visit.links_error {
            self.stats.link_errors += 1;
            eprintln!("[rs] link work on {} failed, its verdict kept: {error}", write.host);
        }
        let buckets = self.buckets.clone();
        let store = buckets.store(&write.host);
        let Some(current) = store.domain(&write.host)? else {
            return Ok(());
        };
        if current.get("state").and_then(Value::as_str) == Some(State::Excluded.as_str()) {
            return Ok(());
        }
        let mut changes = Changes::default();
        // Each file's URL count, and whether it answered; a file counts toward the domain only if it did, and a link run's listing never.
        let mut totals: HashMap<&str, (i64, bool)> = known.sitemaps.iter().map(|(url, s)| (url.as_str(), (s.url_count, s.ok && s.kind.as_deref() != Some(LINKS_KIND)))).collect();
        for update in &write.visit.sitemaps {
            let previous = store.sitemap(&write.host, &update.url)?;
            let record = records::written_sitemap(previous.as_ref(), update, write.checked_at);
            changes.sitemap(&write.host, &update.url, &record);
            let ok = record.get("status").and_then(Value::as_str) == Some("ok") && update.kind.as_deref() != Some(LINKS_KIND);
            totals.insert(&update.url, (records::int(record.get("urls")).unwrap_or(0), ok));
        }
        if write.state == State::Active {
            for (url, depth, parent) in &write.visit.deferred {
                if !totals.contains_key(url.as_str()) {
                    changes.sitemap(&write.host, url, &records::unread_sitemap(url, *depth, *parent));
                    totals.insert(url, (0, true));
                }
            }
        }
        let urls = totals.values().filter(|(_, ok)| *ok).map(|(count, _)| count).sum();
        let mut record = records::written_domain(&current, &write, urls);
        self.queue_links(&write, &mut record);
        changes.domain(&write.host, &record);
        store.write(changes)?;
        // Link work alone keeps the site in the lane it waits in.
        let backlog = if write.visit.links_only { self.backlog_of(&write.host, &record) } else { write.backlog };
        self.schedule(&write.host, &record, backlog);
        if !write.visit.links_only {
            self.unreported.push(Report::of(&write, urls));
        }
        Ok(())
    }

    /// Keeps when the domain's link work comes next, in its record and on the queue; a site's first run takes a spot of its own in the spread.
    fn queue_links(&mut self, write: &DomainWrite, record: &mut Json) {
        let now = write.checked_at / SECOND;
        if let Some(at) = write.visit.links_wait {
            self.stats.link_waits += 1;
            record.insert("links_due".into(), (at / SECOND).into());
            self.links.push(&write.host, at / SECOND);
            return;
        }
        let at = match write.visit.links_at {
            None => return,
            Some(LinksAt::Never) => None,
            Some(LinksAt::At(at)) => Some(at / SECOND),
            Some(LinksAt::First) => Some(records::int(record.get("links_due")).unwrap_or(now + first_wave(&write.host, self.links_spread))),
        };
        if write.visit.links_only {
            if write.visit.sitemaps.iter().any(|u| u.kind.as_deref() == Some(LINKS_KIND)) {
                self.stats.link_runs += 1;
            } else if at.is_none() {
                self.stats.link_dropped += 1;
            }
        }
        match at {
            Some(at) => {
                record.insert("links_due".into(), at.into());
                self.links.push(&write.host, at);
            }
            None => {
                record.remove("links_due");
                self.links.remove(&write.host);
            }
        }
    }

    /// Send the registry what visits found, and take in what changed there.
    async fn sync(&mut self) {
        let visits = std::mem::take(&mut self.unreported);
        if let Err(error) = self.registry.report(&visits, now_micros()).await {
            eprintln!("[rs] reporting {} visits failed, will retry: {error:#}", visits.len());
            self.unreported = visits;
        }
        match self.registry.changes(&self.buckets).await {
            Ok(changes) => {
                for change in changes {
                    if let Err(error) = self.apply(&change) {
                        eprintln!("[rs] applying a change to {} failed: {error:#}", change.host);
                    }
                }
            }
            Err(error) => eprintln!("[rs] reading registry changes failed: {error:#}"),
        }
        if let Some(scan) = self.waking.take_if(|scan| scan.is_finished()) {
            self.woke(scan.await.unwrap_or_default());
        }
        self.wake();
        self.look_for_unread();
    }

    /// When the list changes, looks in the background for listed domains found without sitemaps on a visit made while it did not name them, and makes them due.
    fn look_for_unread(&mut self) {
        let version = self.buckets.list_version();
        if version == 0 || version == self.woken || self.waking.is_some() {
            return;
        }
        self.woken = version;
        let buckets = self.buckets.clone();
        self.waking = Some(tokio::task::spawn_blocking(move || unread(&buckets)));
    }

    fn woke(&mut self, hosts: Vec<String>) {
        if !hosts.is_empty() {
            let spread = wake_spread(hosts.len());
            self.wake_by = now_micros() / SECOND + spread;
            println!("[rs] {} listed domains without sitemaps, not yet visited as listed, come due within {} minutes", hosts.len(), spread / 60);
        }
        self.to_wake = hosts;
    }

    /// Makes some of the domains found unread due, each by its record as it stands now, which only its due time changes.
    fn wake(&mut self) {
        let now = now_micros() / SECOND;
        let mut rng = rand::thread_rng();
        let at = self.to_wake.len().saturating_sub(WAKE_CHUNK);
        for host in self.to_wake.split_off(at) {
            if self.busy.contains(host.as_str()) {
                continue;
            }
            let Ok(Some(mut record)) = self.buckets.store(&host).domain(&host) else {
                continue;
            };
            if !records::listed_unread(&record) || records::int(record.get("due")).is_some_and(|due| due <= self.wake_by) {
                continue;
            }
            record.insert("due".into(), (now + rng.gen_range(0..=(self.wake_by - now).max(0))).into());
            let mut changes = Changes::default();
            changes.domain(&host, &record);
            match self.buckets.store(&host).write(changes) {
                Ok(()) => self.schedule(&host, &record, false),
                Err(error) => eprintln!("[rs] waking {host} failed: {error:#}"),
            }
        }
    }

    fn apply(&mut self, change: &Change) -> Result<()> {
        let store = self.buckets.store(&change.host);
        let excluded = change.state == State::Excluded.as_str();
        let categories = change.categories.clone().unwrap_or_default();
        let record = match store.domain(&change.host)? {
            None => records::new_domain(
                change.rank,
                change.tld_group.as_deref(),
                &categories,
                if excluded { State::Excluded } else { State::New },
                change.state_reason.as_deref().filter(|_| excluded),
                (!excluded).then(now_micros),
            ),
            Some(mut record) => {
                let mut sorted = categories;
                sorted.sort();
                record.insert("rank".into(), change.rank.into());
                record.insert("group".into(), change.tld_group.clone().into());
                record.insert("categories".into(), sorted.into());
                if excluded {
                    record.insert("state".into(), State::Excluded.as_str().into());
                    record.insert("reason".into(), change.state_reason.clone().into());
                    record.insert("due".into(), Value::Null);
                } else if records::wants_recheck(&change.state, &record) {
                    records::reset_for_recheck(&mut record, now_micros());
                }
                record
            }
        };
        let mut changes = Changes::default();
        changes.domain(&change.host, &record);
        store.write(changes)?;
        if !self.busy.contains(change.host.as_str()) {
            self.schedule(&change.host, &record, false);
        }
        Ok(())
    }

    fn schedule(&mut self, host: &str, record: &Json, backlog: bool) {
        if let Some(state) = record.get("state").and_then(Value::as_str).and_then(State::parse) {
            let place = self.place(host, records::int(record.get("rank")));
            self.timetable.set(host, Lane::of(state, backlog), records::int(record.get("due")), Some(place));
        }
    }

    fn place(&self, host: &str, rank: Option<i64>) -> i64 {
        placed(self.buckets.allows(host), self.buckets.terms(host).news, rank)
    }

    fn print(&mut self) {
        self.printed = Instant::now();
        let elapsed = self.started.elapsed().as_secs_f64().max(1.0);
        let memory = self.buckets.memory();
        let disk = self.buckets.free_disk().unwrap_or(0);
        let heap = heap();
        let lanes: Vec<String> = Lane::ALL
            .iter()
            .map(|&lane| format!("{} {}/{} ({} due)", lane.as_str(), self.running[lane.index()], self.share(lane), thousands(self.timetable.queued(lane) as i64)))
            .collect();
        println!(
            "[rs] {} visited  {:.1}/s  {:.1} req/s  in flight {}  lanes {}  new urls {}  scheduled {}  sitemap slots {}/{} parsing {}/{} heavy {}/{} ({} waiting)  rocksdb readers {} MB memtables {} MB cache {} MB  heap {} MB in use {} MB resident  disk free {} GB",
            thousands(self.stats.visited as i64),
            self.stats.visited as f64 / elapsed,
            self.stats.requests as f64 / elapsed,
            self.visits.len(),
            lanes.join(" "),
            thousands(self.stats.new),
            thousands(self.timetable.len() as i64),
            self.visitor.bodies.busy(),
            self.visitor.bodies.size(),
            self.visitor.cpu.busy(),
            self.visitor.cpu.size(),
            self.visitor.heavy.busy(),
            self.visitor.heavy.size(),
            thousands(self.heavy.len() as i64),
            memory.table_readers >> 20,
            memory.memtables >> 20,
            memory.block_cache >> 20,
            heap.0 >> 20,
            heap.1 >> 20,
            disk >> 30,
        );
    }

    /// Totals since the loop started, as one JSON line.
    pub fn summary(&self) -> Value {
        serde_json::json!({
            "seconds": self.started.elapsed().as_secs_f64(),
            "visited": self.stats.visited,
            "requests": self.stats.requests,
            "new_urls": self.stats.new,
            "listed_urls": self.stats.listed,
            "outcomes": self.stats.outcomes,
            "lanes": Lane::ALL.iter().map(|&lane| (lane.as_str(), self.stats.lanes[lane.index()])).collect::<BTreeMap<_, _>>(),
            "link_errors": self.stats.link_errors,
            "link_runs": self.stats.link_runs,
            "link_dropped": self.stats.link_dropped,
            "link_waits": self.stats.link_waits,
            "links_queued": self.links.len(),
            "handoff_errors": self.buckets.handoff_errors.load(std::sync::atomic::Ordering::Relaxed),
        })
    }
}

/// A heavy domain started without a heavy slot comes back Busy, for the loop to hand it one in its turn.
async fn visit_one(host: Arc<str>, buckets: Arc<Buckets>, visitor: Arc<Visitor>, excluded: Arc<HashSet<String>>, heavy: Option<OwnedSemaphorePermit>) -> Finished {
    if heavy.is_none() {
        let (counting, counted) = (buckets.clone(), host.clone());
        let records = tokio::task::spawn_blocking(move || counting.store(&counted).sitemap_count(&counted).unwrap_or(0)).await.unwrap_or(0);
        if records > HEAVY_RECORDS {
            return Finished::Busy;
        }
    }
    let _heavy = heavy;
    let reading = host.clone();
    let loaded = tokio::task::spawn_blocking(move || load_known(&buckets, &reading)).await;
    let Ok(Some(known)) = loaded else {
        return Finished::Skipped;
    };
    let known = Arc::new(known);
    let write = once(known.clone(), &visitor, &excluded).await;
    Finished::Visited(Box::new(write), known)
}

enum LinkFinished {
    Done(Box<DomainWrite>, Arc<Known>),
    Gone,
}

/// Link work for a site taken off the queue; one excluded since, or no longer listed, leaves the queue at no cost to the site.
async fn links_one(host: Arc<str>, buckets: Arc<Buckets>, visitor: Arc<Visitor>, excluded: Arc<HashSet<String>>) -> LinkFinished {
    let reading = host.clone();
    let loaded = tokio::task::spawn_blocking(move || load_known(&buckets, &reading)).await;
    let Ok(Some(known)) = loaded else {
        return LinkFinished::Gone;
    };
    let known = Arc::new(known);
    let now = now_micros();
    let visit = if known.categories.iter().any(|c| excluded.contains(c)) || exclusions::blocked_operator(&known.host) || !matches!(known.state, State::Active | State::NoSitemap) {
        Visit { links_only: true, links_at: Some(LinksAt::Never), ..Visit::new(&known.host) }
    } else {
        visitor.links(known.clone(), now).await
    };
    if visit.revisit {
        // The visit reads robots.txt again for itself and leaves link work on the site as it decides, or ends it.
        let fresh = Arc::new(Known { robots_checked_at: None, ..(*known).clone() });
        let mut write = once(fresh.clone(), &visitor, &excluded).await;
        write.visit.links_at = write.visit.links_at.or(Some(LinksAt::Never));
        return LinkFinished::Done(Box::new(write), fresh);
    }
    LinkFinished::Done(Box::new(links_plan(&known, visit, now)), known)
}

/// What a visit needs to know about a domain, from its store; None once excluded.
fn load_known(buckets: &Buckets, host: &str) -> Option<Known> {
    let store = buckets.store(host);
    let record = store.domain(host).ok()??;
    if record.get("state").and_then(Value::as_str) == Some(State::Excluded.as_str()) {
        return None;
    }
    records::known(host, &record, &store.sitemaps(host).ok()?)
}

/// Bytes the allocator has handed out, and bytes it keeps resident, so fragmentation shows.
fn heap() -> (usize, usize) {
    #[cfg(not(target_env = "msvc"))]
    {
        use tikv_jemalloc_ctl::{epoch, stats};
        if epoch::advance().is_ok() {
            return (stats::allocated::read().unwrap_or(0), stats::resident::read().unwrap_or(0));
        }
    }
    (0, 0)
}

fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        out.insert(0, '-');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn many_woken_domains_spread_over_three_days() {
        assert_eq!(wake_spread(100), 200);
        assert_eq!(wake_spread(20_000), 40_000, "two seconds apart");
        assert_eq!(wake_spread(40_020), 3 * 24 * 3_600, "a first start wakes every listed site without sitemaps");
    }
}

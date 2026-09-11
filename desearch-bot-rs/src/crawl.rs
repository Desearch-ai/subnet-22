//! The crawl loop: many visits in flight across the buckets this process owns, and what they learn kept.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::Rng;
use serde_json::Value;
use tokio::sync::watch;
use tokio::task::{Id, JoinError, JoinSet};
use tokio::time::Instant;

use crate::buckets::{Buckets, Changes, Json};
use crate::registry::{Change, Registry, Report};
use crate::schedule::{HOUR, SECOND};
use crate::states::{self, Outcome, State};
use crate::suffixes::tld_group;
use crate::timetable::Timetable;
use crate::visit::{Crash, Known, Visit, Visitor};
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
    pub visit: Visit,
}

/// The state a visit leaves a domain in, and when the loop should come back to it.
pub fn plan(known: &Known, visit: Visit, now: i64, rng: &mut impl Rng) -> DomainWrite {
    let decision = states::decide(known.state, visit.outcome, known.failures, known.last_ok_at, now, rng);
    let mut due = decision.next_check_at;
    if decision.state == State::Active {
        let earliest = if visit.deferred.is_empty() { earliest_sitemap(known, &visit, now) } else { Some(now) };
        if let Some(earliest) = earliest.filter(|&e| due.is_none_or(|d| e < d)) {
            due = Some(earliest);
        }
        if visit.cut_short {
            due = due.map(|d| d.max(now + CUT_SHORT_WAIT));
        }
    }
    let canonical_host = match visit.outcome {
        Outcome::Redirect => visit.canonical_host.clone(),
        outcome if outcome.answered() => None,
        _ => known.canonical_host.clone(),
    };
    let last_ok_at = if visit.outcome.answered() && visit.requests > 0 { Some(now) } else { known.last_ok_at };
    DomainWrite {
        host: known.host.clone(),
        state: decision.state,
        reason: visit.reason.clone(),
        failures: decision.failures,
        next_due_at: due,
        last_ok_at,
        canonical_host,
        checked_at: now,
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
    let updated: HashMap<&str, Option<i64>> = visit.sitemaps.iter().map(|u| (u.url.as_str(), u.next_check_at)).collect();
    let mut times: Vec<i64> = known
        .sitemaps
        .values()
        .map(|s| updated.get(s.url.as_str()).copied().unwrap_or(s.next_check_at).unwrap_or(now))
        .collect();
    times.extend(updated.iter().filter(|(url, _)| !known.sitemaps.contains_key(**url)).filter_map(|(_, when)| *when));
    times.into_iter().min()
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
    Skipped(Arc<str>),
    Visited(Box<DomainWrite>, Arc<Known>),
}

#[derive(Default)]
pub struct Stats {
    pub visited: u64,
    pub requests: i64,
    pub new: i64,
    pub listed: i64,
    pub outcomes: BTreeMap<String, u64>,
}

/// Keeps many visits in flight across the buckets one process owns.
pub struct Loop {
    buckets: Arc<Buckets>,
    visitor: Arc<Visitor>,
    concurrency: usize,
    registry: Registry,
    excluded: Arc<HashSet<String>>,
    timetable: Timetable,
    visits: JoinSet<Finished>,
    tasks: HashMap<Id, Arc<str>>,
    busy: HashSet<Arc<str>>,
    pending: Vec<(Box<DomainWrite>, Arc<Known>)>,
    unreported: Vec<Report>,
    pub stats: Stats,
    started: Instant,
    printed: Instant,
    min_free_disk: u64,
    disk_checked: Option<Instant>,
    disk_low: bool,
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
        }
    }

    /// Stop starting visits while the disk has less than this many bytes free.
    pub fn with_min_free_disk(mut self, bytes: u64) -> Self {
        self.min_free_disk = bytes;
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
        for schedule in schedules {
            for (host, due) in schedule? {
                if let Some(state) = State::parse(&due.state) {
                    self.timetable.set(&host, state, due.due, due.rank);
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
        self.sync().await;
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
        let free = self.concurrency.saturating_sub(self.visits.len());
        if free == 0 {
            return;
        }
        for host in self.timetable.take(free, now_micros() / SECOND) {
            if !self.busy.insert(host.clone()) {
                continue;
            }
            let job = visit_one(host.clone(), self.buckets.clone(), self.visitor.clone(), self.excluded.clone());
            let handle = self.visits.spawn(job);
            self.tasks.insert(handle.id(), host);
        }
    }

    fn finish(&mut self, done: Result<(Id, Finished), JoinError>) {
        match done {
            Ok((id, Finished::Skipped(host))) => {
                self.tasks.remove(&id);
                self.busy.remove(&host);
            }
            Ok((id, Finished::Visited(write, known))) => {
                self.tasks.remove(&id);
                self.stats.visited += 1;
                self.pending.push((write, known));
            }
            Err(error) => {
                if let Some(host) = self.tasks.remove(&error.id()) {
                    eprintln!("[rs] visit to {host} failed: {error}");
                    self.busy.remove(&host);
                }
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
        self.stats.new += write.visit.new;
        self.stats.listed += write.visit.listed;
        let outcome = format!("{} {}", write.state.as_str(), write.reason.as_deref().unwrap_or("-"));
        *self.stats.outcomes.entry(outcome).or_default() += 1;
        let store = self.buckets.store(&write.host);
        let Some(current) = store.domain(&write.host)? else {
            return Ok(());
        };
        if current.get("state").and_then(Value::as_str) == Some(State::Excluded.as_str()) {
            return Ok(());
        }
        let mut changes = Changes::default();
        // Each file's URL count, and whether it answered; a file counts toward the domain only if it did.
        let mut totals: HashMap<&str, (i64, bool)> = known.sitemaps.iter().map(|(url, s)| (url.as_str(), (s.url_count, s.ok))).collect();
        for update in &write.visit.sitemaps {
            let previous = store.sitemap(&write.host, &update.url)?;
            let record = records::written_sitemap(previous.as_ref(), update, write.checked_at);
            changes.sitemap(&write.host, &update.url, &record);
            let ok = record.get("status").and_then(Value::as_str) == Some("ok");
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
        let record = records::written_domain(&current, &write, urls);
        changes.domain(&write.host, &record);
        store.write(changes)?;
        self.schedule(&write.host, &record);
        self.unreported.push(Report::of(&write, urls));
        Ok(())
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
                }
                record
            }
        };
        let mut changes = Changes::default();
        changes.domain(&change.host, &record);
        store.write(changes)?;
        if !self.busy.contains(change.host.as_str()) {
            self.schedule(&change.host, &record);
        }
        Ok(())
    }

    fn schedule(&mut self, host: &str, record: &Json) {
        if let Some(state) = record.get("state").and_then(Value::as_str).and_then(State::parse) {
            self.timetable.set(host, state, records::int(record.get("due")), records::int(record.get("rank")));
        }
    }

    fn print(&mut self) {
        self.printed = Instant::now();
        let elapsed = self.started.elapsed().as_secs_f64().max(1.0);
        let memory = self.buckets.memory();
        let disk = self.buckets.free_disk().unwrap_or(0);
        println!(
            "[rs] {} visited  {:.1}/s  {:.1} req/s  in flight {}  new urls {}  scheduled {}  sitemap slots {}/{} parsing {}/{} heavy {}/{}  rocksdb readers {} MB memtables {} MB cache {} MB  disk free {} GB",
            thousands(self.stats.visited as i64),
            self.stats.visited as f64 / elapsed,
            self.stats.requests as f64 / elapsed,
            self.visits.len(),
            thousands(self.stats.new),
            thousands(self.timetable.len() as i64),
            self.visitor.bodies.busy(),
            self.visitor.bodies.size(),
            self.visitor.cpu.busy(),
            self.visitor.cpu.size(),
            self.visitor.heavy.busy(),
            self.visitor.heavy.size(),
            memory.table_readers >> 20,
            memory.memtables >> 20,
            memory.block_cache >> 20,
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
        })
    }
}

async fn visit_one(host: Arc<str>, buckets: Arc<Buckets>, visitor: Arc<Visitor>, excluded: Arc<HashSet<String>>) -> Finished {
    let (counting, counted) = (buckets.clone(), host.clone());
    let records = tokio::task::spawn_blocking(move || counting.store(&counted).sitemap_count(&counted).unwrap_or(0)).await.unwrap_or(0);
    let _heavy = if records > HEAVY_RECORDS {
        match visitor.heavy.take().await {
            Ok(permit) => Some(permit),
            Err(_) => return Finished::Skipped(host),
        }
    } else {
        None
    };
    let reading = host.clone();
    let loaded = tokio::task::spawn_blocking(move || load_known(&buckets, &reading)).await;
    let Ok(Some(known)) = loaded else {
        return Finished::Skipped(host);
    };
    let known = Arc::new(known);
    let write = once(known.clone(), &visitor, &excluded).await;
    Finished::Visited(Box::new(write), known)
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

//! Fills the task API's queue from the ready lists: as much as it has room for, round robin across domains by rank, within an hourly cap per domain.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::watch;

use crate::buckets::{bucket_of, Buckets};
use crate::outcomes::{self, Next, OutcomeFeed};
use crate::ready::{Entry, Order, Pick, Reason, SentPage, CHANGED, FRESH, LANES, REQUEUED};
use crate::records;
use crate::taskapi::{QueuedUrl, TaskApi};
use crate::timetable::UNRANKED;

pub const PASS_SECONDS: Duration = Duration::from_secs(15);
/// How long a missing outcome number may stay missing, while later ones exist, before it is skipped.
const MISSING_GRACE: Duration = Duration::from_secs(120);
/// URLs in one task, as the API counts its room.
pub const TASK_URLS: u64 = 1000;
pub const BATCH_URLS: usize = 10_000;
/// Batches in flight to the API at once; each waits there for seconds, so one at a time leaves its queue short.
pub const SENDS_AT_ONCE: usize = 4;
/// Threads reading the stores at once: a network disk answers many reads together far faster than one after another.
const READERS: usize = 16;
const HOUR: u32 = 3600;
const RANKS_SECONDS: Duration = Duration::from_secs(600);
/// Retries and re-crawls moved back onto ready lists per store per pass.
const REQUEUE_PER_PASS: usize = 50_000;
/// A pass tries again with what is left when stale entries left the first allotment short.
const ALLOT_TRIES: usize = 3;
const PENDING: &str = "dispatch_pending";
const TOP_DOMAINS: usize = 5;

/// Slots a domain gets at each turn: 7 for the ten best ranks, one fewer for each tenfold, 1 from a million on and for unranked domains.
pub fn slots(rank: i64) -> u64 {
    7u64.saturating_sub(u64::from(rank.max(1).ilog10())).max(1)
}

/// What one domain could send this pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Want {
    pub rank: i64,
    pub ready: u64,
    pub allowance: u64,
}

/// Each domain's share of one pass and where the next pass starts: turns go round the domains, sorted best first, from `start`, each taking its slots.
pub fn allot(wants: &[Want], budget: u64, start: usize) -> (Vec<u64>, usize) {
    let mut given = vec![0u64; wants.len()];
    if wants.is_empty() {
        return (given, 0);
    }
    let cap = |i: usize| wants[i].ready.min(wants[i].allowance);
    let start = start % wants.len();
    let mut turns: VecDeque<usize> = (start..wants.len()).chain(0..start).filter(|&i| cap(i) > 0).collect();
    let mut left = budget;
    let mut next = start;
    while left > 0 {
        let Some(i) = turns.pop_front() else {
            break;
        };
        let take = slots(wants[i].rank).min(cap(i) - given[i]).min(left);
        given[i] += take;
        left -= take;
        next = (i + 1) % wants.len();
        if given[i] < cap(i) {
            turns.push_back(i);
        }
    }
    (given, next)
}

/// Each lane's part of what a domain sends, from percents for new pages and retries; refreshes get the rest.
pub fn shares(new_percent: usize, retry_percent: usize) -> [f64; LANES] {
    let mut shares = [0.0; LANES];
    shares[FRESH] = new_percent as f64 / 100.0;
    shares[REQUEUED] = retry_percent as f64 / 100.0;
    shares[CHANGED] = (100usize.saturating_sub(new_percent + retry_percent)) as f64 / 100.0;
    shares
}

/// A domain's want this pass split across its lanes by their shares, with credit carried between passes so a few pages a pass still come out even.
pub fn split(credit: &mut [f64; LANES], shares: &[f64; LANES], want: usize) -> [usize; LANES] {
    for (owed, share) in credit.iter_mut().zip(shares) {
        *owed += share * want as f64;
    }
    let mut quotas = [0; LANES];
    for _ in 0..want {
        let lane = (0..LANES).max_by(|&a, &b| credit[a].total_cmp(&credit[b])).expect("lanes");
        quotas[lane] += 1;
        credit[lane] -= 1.0;
    }
    quotas
}

/// A lane that came up short is owed nothing for it, and one that filled another's gap owes nothing back: no credit or debt beyond one page carries over.
pub fn settle(credit: &mut [f64; LANES], quotas: [usize; LANES], taken: [usize; LANES]) {
    for lane in 0..LANES {
        credit[lane] = (credit[lane] + quotas[lane] as f64 - taken[lane] as f64).clamp(-1.0, 1.0);
    }
}

/// A stable name for a batch's content, so the API ignores the same batch sent again after a timeout.
pub fn batch_id(urls: &[QueuedUrl]) -> String {
    let mut hash = Sha256::new();
    for url in urls {
        hash.update(url.host.as_bytes());
        hash.update(b"\t");
        hash.update(url.url.as_bytes());
        hash.update(b"\n");
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The first of every list, then the second of every list, so a batch spans many domains.
pub fn interleave<T>(lists: Vec<Vec<T>>) -> Vec<T> {
    let mut iters: Vec<_> = lists.into_iter().map(Vec::into_iter).collect();
    let mut out = Vec::new();
    loop {
        let before = out.len();
        out.extend(iters.iter_mut().filter_map(Iterator::next));
        if out.len() == before {
            return out;
        }
    }
}

/// URLs each domain sent within the last hour.
#[derive(Default)]
pub struct HourlyCap {
    limit: u64,
    sent: HashMap<String, VecDeque<(u32, u64)>>,
}

impl HourlyCap {
    pub fn new(limit: u64) -> Self {
        HourlyCap { limit, sent: HashMap::new() }
    }

    pub fn allowance(&mut self, domain: &str, now: u32) -> u64 {
        let Some(sent) = self.sent.get_mut(domain) else {
            return self.limit;
        };
        while sent.front().is_some_and(|(at, _)| *at + HOUR <= now) {
            sent.pop_front();
        }
        let total: u64 = sent.iter().map(|(_, n)| n).sum();
        if sent.is_empty() {
            self.sent.remove(domain);
        }
        self.limit.saturating_sub(total)
    }

    pub fn record(&mut self, domain: &str, now: u32, count: u64) {
        self.sent.entry(domain.to_string()).or_default().push_back((now, count));
    }
}

/// What the status line reports, shared by the dispatcher, the outcome feed and the backfill.
#[derive(Default)]
pub struct Progress {
    pub ready: AtomicU64,
    pub ready_domains: AtomicU64,
    pub top: Mutex<Vec<(String, u64)>>,
    pub dispatched: AtomicU64,
    /// Sent pages by why they went: new, refreshed, retried.
    pub sent_new: AtomicU64,
    pub sent_refresh: AtomicU64,
    pub sent_retry: AtomicU64,
    pub batches: AtomicU64,
    pub room_tasks: AtomicI64,
    pub requeued: AtomicU64,
    /// Sent pages retried because no outcome came within a day.
    pub overdue: AtomicU64,
    /// Outcome files gone before they were read.
    pub skipped: AtomicU64,
    pub outcome_seq: AtomicU64,
    pub outcome_lag: AtomicI64,
    pub outcomes: AtomicU64,
    pub retried: AtomicU64,
    pub backfill_stores: AtomicU64,
    pub backfill_of: AtomicU64,
    pub backfill_scanned: AtomicU64,
    /// The last pass's milliseconds in each step: absorbing, picking, sending.
    pub pass_ms: [AtomicU64; 3],
    pub backfill_added: AtomicU64,
}

impl Progress {
    /// One status line; `per_minute` is URLs dispatched per minute since the previous line.
    pub fn line(&self, per_minute: f64) -> String {
        let top = self.top.lock().unwrap_or_else(|e| e.into_inner());
        let top: Vec<String> = top.iter().map(|(domain, n)| format!("{domain} {}", thousands(*n))).collect();
        format!(
            "[rs] dispatch ready {} in {} domains (top {})  sent {} ({:.0}/min, {} batches; new {} refresh {} retry {})  room {} tasks  requeued {} overdue {}  outcomes seq {} lag {}s applied {} retried {} skipped {}  backfill {}/{} stores {} read {} added",
            thousands(self.ready.load(Ordering::Relaxed)),
            thousands(self.ready_domains.load(Ordering::Relaxed)),
            if top.is_empty() { "-".to_string() } else { top.join(", ") },
            thousands(self.dispatched.load(Ordering::Relaxed)),
            per_minute,
            self.batches.load(Ordering::Relaxed),
            thousands(self.sent_new.load(Ordering::Relaxed)),
            thousands(self.sent_refresh.load(Ordering::Relaxed)),
            thousands(self.sent_retry.load(Ordering::Relaxed)),
            self.room_tasks.load(Ordering::Relaxed),
            thousands(self.requeued.load(Ordering::Relaxed)),
            thousands(self.overdue.load(Ordering::Relaxed)),
            self.outcome_seq.load(Ordering::Relaxed),
            self.outcome_lag.load(Ordering::Relaxed),
            thousands(self.outcomes.load(Ordering::Relaxed)),
            thousands(self.retried.load(Ordering::Relaxed)),
            self.skipped.load(Ordering::Relaxed),
            self.backfill_stores.load(Ordering::Relaxed),
            self.backfill_of.load(Ordering::Relaxed),
            thousands(self.backfill_scanned.load(Ordering::Relaxed)),
            thousands(self.backfill_added.load(Ordering::Relaxed)),
        ) + &format!(
            "  last pass absorb {:.1}s pick {:.1}s send {:.1}s",
            self.pass_ms[0].load(Ordering::Relaxed) as f64 / 1000.0,
            self.pass_ms[1].load(Ordering::Relaxed) as f64 / 1000.0,
            self.pass_ms[2].load(Ordering::Relaxed) as f64 / 1000.0,
        )
    }
}

/// One domain's part of a pass: its lane quotas, the pages taken, and how many it has left.
struct Taken {
    domain: String,
    quotas: [usize; LANES],
    picks: Vec<Pick>,
    left: u64,
}

/// A picked page as kept while its batch is in flight, so a restart resends the same batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Sending {
    domain: String,
    rest: String,
    order: (u8, u32, u32),
    lastmod: u32,
    reason: u8,
    retries: u8,
    url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Pending {
    batch_id: String,
    at: u32,
    picks: Vec<Sending>,
}

impl Pending {
    fn of(picks: &[Pick], at: u32) -> Pending {
        let picks: Vec<Sending> = picks.iter().map(Sending::of).collect();
        let urls: Vec<QueuedUrl> = picks.iter().map(|p| QueuedUrl { host: p.domain.clone(), url: p.url.clone() }).collect();
        Pending { batch_id: batch_id(&urls), at, picks }
    }

    fn urls(&self) -> Vec<QueuedUrl> {
        self.picks.iter().map(|p| QueuedUrl { host: p.domain.clone(), url: p.url.clone() }).collect()
    }
}

impl Sending {
    fn of(pick: &Pick) -> Sending {
        let order = match pick.entry.order {
            Order::Changed { lastmod } => (0, lastmod, 0),
            Order::Requeued { due } => (1, due, 0),
            Order::Fresh { first_seen, lastmod } => (2, first_seen, lastmod),
        };
        let reason = match pick.reason {
            Reason::Fresh => 0,
            Reason::Changed => 1,
            Reason::Requeued => 2,
        };
        Sending {
            domain: pick.entry.domain.clone(),
            rest: String::from_utf8_lossy(&pick.entry.rest).into_owned(),
            order,
            lastmod: pick.lastmod,
            reason,
            retries: pick.retries,
            url: pick.url.clone(),
        }
    }

    fn pick(&self) -> Pick {
        let order = match self.order {
            (0, lastmod, _) => Order::Changed { lastmod },
            (1, due, _) => Order::Requeued { due },
            (_, first_seen, lastmod) => Order::Fresh { first_seen, lastmod },
        };
        let reason = match self.reason {
            0 => Reason::Fresh,
            1 => Reason::Changed,
            _ => Reason::Requeued,
        };
        Pick {
            entry: Entry { domain: self.domain.clone(), rest: self.rest.clone().into_bytes(), order },
            url: self.url.clone(),
            lastmod: self.lastmod,
            reason,
            retries: self.retries,
        }
    }
}

struct Domain {
    rank: i64,
    ready: u64,
}

pub struct Dispatcher {
    buckets: Arc<Buckets>,
    api: Arc<TaskApi>,
    cap: HourlyCap,
    domains: HashMap<String, Domain>,
    /// The domain whose turn comes first next pass, as (rank, domain).
    resume: Option<(i64, String)>,
    /// The batches in flight, kept so a restart sends them again unchanged.
    pending: Vec<Pending>,
    progress: Arc<Progress>,
    ranks_read: Instant,
    shares: [f64; LANES],
    credit: HashMap<String, [f64; LANES]>,
}

impl Dispatcher {
    /// Every domain with pages waiting, from the per-domain counts; nothing is scanned.
    pub async fn load(
        buckets: Arc<Buckets>,
        api: TaskApi,
        per_domain_hourly: u64,
        shares: [f64; LANES],
        progress: Arc<Progress>,
    ) -> Result<Self> {
        let reading = buckets.clone();
        let (domains, pending) = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut domains = HashMap::new();
            for store in reading.stores() {
                // The counts already hold what grew before now.
                store.take_noticed();
                for (domain, ready) in store.ready_domains()? {
                    if !reading.allows(&domain) {
                        continue;
                    }
                    let rank = rank_of(&reading, &domain);
                    domains.insert(domain, Domain { rank, ready });
                }
            }
            let pending: Vec<Pending> = match reading.stores().next().map(|store| store.meta(PENDING)).transpose()?.flatten() {
                Some(Value::Array(saved)) => saved.into_iter().map(serde_json::from_value).collect::<Result<_, _>>()?,
                Some(Value::Null) | None => Vec::new(),
                Some(one) => vec![serde_json::from_value(one)?],
            };
            Ok((domains, pending))
        })
        .await??;
        let dispatcher = Dispatcher {
            buckets,
            api: Arc::new(api),
            cap: HourlyCap::new(per_domain_hourly),
            domains,
            resume: None,
            pending,
            progress,
            ranks_read: Instant::now(),
            shares,
            credit: HashMap::new(),
        };
        dispatcher.report();
        Ok(dispatcher)
    }

    pub async fn run(mut self, mut stop: watch::Receiver<bool>) {
        println!("[rs] dispatching as {} to the task API, {} domains with pages ready", self.api.hotkey(), self.domains.len());
        let mut tick = tokio::time::interval(PASS_SECONDS);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !*stop.borrow() {
            tokio::select! {
                _ = stop.changed() => break,
                _ = tick.tick() => {}
            }
            if let Err(error) = self.pass(now()).await {
                eprintln!("[rs] dispatch pass failed: {error:#}");
            }
        }
    }

    /// One look at the room and as many batches as fit.
    pub async fn pass(&mut self, now: u32) -> Result<u64> {
        let mut spent = [Duration::ZERO; 3];
        let sent = self.timed_pass(now, &mut spent).await;
        for (slot, took) in self.progress.pass_ms.iter().zip(spent) {
            slot.store(took.as_millis() as u64, Ordering::Relaxed);
        }
        sent
    }

    async fn timed_pass(&mut self, now: u32, spent: &mut [Duration; 3]) -> Result<u64> {
        let started = Instant::now();
        self.absorb(now).await?;
        spent[0] = started.elapsed();
        let started = Instant::now();
        let (mut sent, cleared) = self.send_pending().await?;
        spent[2] += started.elapsed();
        if !cleared {
            return Ok(sent);
        }
        let room = match self.api.room().await {
            Ok(room) => room,
            Err(error) => {
                eprintln!("[rs] reading the task API's room failed: {error}");
                return Ok(sent);
            }
        };
        self.progress.room_tasks.store(room.room_tasks, Ordering::Relaxed);
        if room.refusing || room.room_tasks <= 0 {
            self.report();
            return Ok(sent);
        }
        let mut budget = room.room_tasks as u64 * TASK_URLS;
        for _ in 0..ALLOT_TRIES {
            let started = Instant::now();
            let picks = self.pick(budget, now).await?;
            spent[1] += started.elapsed();
            if picks.is_empty() {
                break;
            }
            for wave in picks.chunks(BATCH_URLS * SENDS_AT_ONCE) {
                self.pending = wave.chunks(BATCH_URLS).map(|batch| Pending::of(batch, now)).collect();
                let started = Instant::now();
                let (taken, cleared) = self.send_pending().await?;
                spent[2] += started.elapsed();
                sent += taken;
                budget = budget.saturating_sub(taken);
                if !cleared {
                    self.report();
                    return Ok(sent);
                }
            }
            if budget == 0 {
                break;
            }
        }
        self.report();
        Ok(sent)
    }

    /// Take in what grew since the last pass: new pages on the lists, retries and re-crawls now due, and ranks now and then.
    async fn absorb(&mut self, now: u32) -> Result<()> {
        let buckets = self.buckets.clone();
        let refresh: Vec<String> =
            if self.ranks_read.elapsed() >= RANKS_SECONDS { self.domains.keys().cloned().collect() } else { Vec::new() };
        if !refresh.is_empty() {
            self.ranks_read = Instant::now();
        }
        let (grown, ranks, requeued, overdue) = tokio::task::spawn_blocking(move || -> Result<_> {
            let (mut requeued, mut overdue) = (0, 0);
            let mut grown: HashMap<String, u64> = HashMap::new();
            let swept = on_readers(buckets.stores().collect(), |store| {
                Ok((store.expire_unanswered(now, REQUEUE_PER_PASS)?, store.requeue_due(now, REQUEUE_PER_PASS)?, store.take_noticed()))
            })?;
            for (expired, due, noticed) in swept {
                overdue += expired;
                requeued += due;
                for (domain, added) in noticed {
                    *grown.entry(domain).or_default() += added;
                }
            }
            let ranks: Vec<(String, i64)> =
                refresh.iter().chain(grown.keys()).map(|domain| (domain.clone(), rank_of(&buckets, domain))).collect();
            Ok((grown, ranks, requeued, overdue))
        })
        .await??;
        self.progress.requeued.fetch_add(requeued as u64, Ordering::Relaxed);
        self.progress.overdue.fetch_add(overdue as u64, Ordering::Relaxed);
        let ranks: HashMap<String, i64> = ranks.into_iter().collect();
        for (domain, added) in grown {
            let rank = ranks.get(&domain).copied().unwrap_or(UNRANKED);
            self.domains.entry(domain).or_insert(Domain { rank, ready: 0 }).ready += added;
        }
        for (domain, rank) in ranks {
            if let Some(known) = self.domains.get_mut(&domain) {
                known.rank = rank;
            }
        }
        Ok(())
    }

    /// Pages for one pass, interleaved across domains best first.
    async fn pick(&mut self, budget: u64, now: u32) -> Result<Vec<Pick>> {
        let buckets = self.buckets.clone();
        let mut order: Vec<(i64, String)> = self
            .domains
            .iter()
            .filter(|(name, d)| d.ready > 0 && buckets.allows(name))
            .map(|(name, d)| (d.rank, name.clone()))
            .collect();
        order.sort_unstable();
        let wants: Vec<Want> = order
            .iter()
            .map(|(rank, name)| Want { rank: *rank, ready: self.domains[name].ready, allowance: self.cap.allowance(name, now) })
            .collect();
        let start = self.resume.as_ref().map_or(0, |resume| order.partition_point(|key| key <= resume));
        let (given, next) = allot(&wants, budget, start);
        self.resume = order.get(next.checked_sub(1).unwrap_or(order.len().saturating_sub(1))).cloned();
        let asks: Vec<(String, [usize; LANES])> = order
            .iter()
            .zip(&given)
            .filter(|(_, &n)| n > 0)
            .map(|((_, name), &n)| (name.clone(), split(self.credit.entry(name.clone()).or_default(), &self.shares, n as usize)))
            .collect();
        if asks.is_empty() {
            return Ok(Vec::new());
        }
        let buckets = self.buckets.clone();
        let taken = tokio::task::spawn_blocking(move || -> Result<Vec<Taken>> {
            on_readers(asks, |(domain, quotas)| {
                let Some(store) = buckets.store_for(&domain) else {
                    return Ok(Taken { domain, quotas, picks: Vec::new(), left: 0 });
                };
                let picks = store.take_ready(&domain, quotas)?;
                let left = store.ready_count(&domain)?;
                Ok(Taken { domain, quotas, picks, left })
            })
        })
        .await??;
        let mut lists = Vec::with_capacity(taken.len());
        for Taken { domain, quotas, picks, left } in taken {
            let mut by_lane = [0; LANES];
            for pick in &picks {
                by_lane[pick.entry.order.lane()] += 1;
            }
            if let Some(credit) = self.credit.get_mut(&domain) {
                settle(credit, quotas, by_lane);
            }
            let rank = self.domains.get(&domain).map_or(UNRANKED, |d| d.rank);
            self.set_ready(&domain, left, rank);
            lists.push(picks);
        }
        Ok(interleave(lists))
    }

    /// Send the batches in flight together; how many URLs the API took, and whether all went (the rest go again first next pass).
    async fn send_pending(&mut self) -> Result<(u64, bool)> {
        if self.pending.is_empty() {
            return Ok((0, true));
        }
        self.save_pending().await?;
        let mut sending = tokio::task::JoinSet::new();
        for pending in self.pending.clone() {
            let api = self.api.clone();
            sending.spawn(async move {
                let sent = api.enqueue(&pending.urls(), &pending.batch_id).await;
                (pending, sent)
            });
        }
        let (mut taken, mut failed) = (0, Vec::new());
        while let Some(joined) = sending.join_next().await {
            match joined? {
                (pending, Ok(_)) => taken += self.landed(pending).await?,
                (pending, Err(error)) => {
                    eprintln!("[rs] enqueueing batch {} of {} URLs failed, will send it again: {error}", &pending.batch_id[..12], pending.picks.len());
                    failed.push(pending);
                }
            }
        }
        let cleared = failed.is_empty();
        self.pending = failed;
        self.save_pending().await?;
        Ok((taken, cleared))
    }

    async fn save_pending(&self) -> Result<()> {
        let saved = if self.pending.is_empty() { Value::Null } else { serde_json::to_value(&self.pending)? };
        let buckets = self.buckets.clone();
        tokio::task::spawn_blocking(move || buckets.stores().next().map(|s| s.set_meta(PENDING, &saved)).transpose()).await??;
        Ok(())
    }

    /// A batch the API took: its pages leave the ready lists and count as sent.
    async fn landed(&mut self, pending: Pending) -> Result<u64> {
        let buckets = self.buckets.clone();
        let picks: Vec<Pick> = pending.picks.iter().map(Sending::pick).collect();
        let at = pending.at;
        let left = tokio::task::spawn_blocking(move || -> Result<Vec<(String, u64, i64)>> {
            let mut by_bucket: HashMap<usize, Vec<Pick>> = HashMap::new();
            for pick in picks {
                by_bucket.entry(bucket_of(&pick.entry.domain)).or_default().push(pick);
            }
            let per_store = on_readers(by_bucket.into_values().collect(), |picks| {
                let mut left = Vec::new();
                let Some(store) = buckets.store_for(&picks[0].entry.domain) else {
                    return Ok(left);
                };
                store.mark_sent(&picks, at)?;
                let mut domains: Vec<&str> = picks.iter().map(|p| p.entry.domain.as_str()).collect();
                domains.sort_unstable();
                domains.dedup();
                for domain in domains {
                    left.push((domain.to_string(), store.ready_count(domain)?, rank_of(&buckets, domain)));
                }
                Ok(left)
            })?;
            Ok(per_store.into_iter().flatten().collect())
        })
        .await??;
        for (domain, ready, rank) in left {
            self.set_ready(&domain, ready, rank);
        }
        let mut per_domain: HashMap<&str, u64> = HashMap::new();
        for sending in &pending.picks {
            *per_domain.entry(&sending.domain).or_default() += 1;
        }
        for (domain, count) in per_domain {
            self.cap.record(domain, pending.at, count);
        }
        for sending in &pending.picks {
            let counter = match sending.reason {
                0 => &self.progress.sent_new,
                1 => &self.progress.sent_refresh,
                _ => &self.progress.sent_retry,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
        let count = pending.picks.len() as u64;
        self.progress.dispatched.fetch_add(count, Ordering::Relaxed);
        self.progress.batches.fetch_add(1, Ordering::Relaxed);
        Ok(count)
    }

    fn set_ready(&mut self, domain: &str, ready: u64, rank: i64) {
        if ready == 0 {
            self.domains.remove(domain);
        } else {
            let known = self.domains.entry(domain.to_string()).or_insert(Domain { rank, ready });
            known.ready = ready;
        }
    }

    fn report(&self) {
        let total: u64 = self.domains.values().map(|d| d.ready).sum();
        let mut top: Vec<(String, u64)> = self.domains.iter().map(|(name, d)| (name.clone(), d.ready)).collect();
        let keep = TOP_DOMAINS.min(top.len());
        if keep > 0 {
            top.select_nth_unstable_by(keep - 1, |a, b| b.1.cmp(&a.1));
            top.truncate(keep);
            top.sort_by_key(|(_, ready)| std::cmp::Reverse(*ready));
        }
        self.progress.ready.store(total, Ordering::Relaxed);
        self.progress.ready_domains.store(self.domains.len() as u64, Ordering::Relaxed);
        *self.progress.top.lock().unwrap_or_else(|e| e.into_inner()) = top;
    }
}

/// Read the outcome feed in order for as long as the process runs.
pub async fn follow_outcomes(buckets: Arc<Buckets>, feed: OutcomeFeed, recrawl_after: u32, progress: Arc<Progress>, mut stop: watch::Receiver<bool>) {
    let reading = buckets.clone();
    let saved = tokio::task::spawn_blocking(move || outcomes::saved_cursor(&reading)).await.ok().and_then(Result::ok).flatten();
    // A bot reading the feed for the first time has sent nothing the older files could be about.
    let mut next = match saved {
        Some(next) => next,
        None => feed.latest().await.ok().flatten().map_or(1, |latest| latest + 1),
    };
    progress.outcome_seq.store(next, Ordering::Relaxed);
    let mut missing: Option<(u64, Instant)> = None;
    let mut skipping = 0u64;
    while !*stop.borrow() {
        let wait = match feed.fetch(next).await {
            Ok(Next::Missing) => {
                let since = missing.filter(|(seq, _)| *seq == next).map_or_else(Instant::now, |(_, at)| at);
                missing = Some((next, since));
                // A number written late is filled in within seconds; one still gone after that expired unread.
                if skipping > 0 || since.elapsed() >= MISSING_GRACE {
                    skipping += 1;
                    next += 1;
                    progress.outcome_seq.store(next, Ordering::Relaxed);
                    progress.skipped.fetch_add(1, Ordering::Relaxed);
                    false
                } else {
                    true
                }
            }
            Ok(Next::File { rows, published_at }) => {
                if skipping > 0 {
                    eprintln!("[rs] skipped {skipping} outcome files gone before they were read; their pages come back once overdue");
                    skipping = 0;
                }
                let lag = published_at.map_or(0, |at| (chrono::Utc::now().timestamp() - at).max(0));
                progress.outcome_lag.store(lag, Ordering::Relaxed);
                let applying = buckets.clone();
                let count = rows.len() as u64;
                let applied = tokio::task::spawn_blocking(move || -> Result<_> {
                    let applied = outcomes::apply(&applying, &rows, recrawl_after)?;
                    outcomes::save_cursor(&applying, next + 1)?;
                    Ok(applied)
                })
                .await;
                match applied {
                    Ok(Ok(applied)) => {
                        next += 1;
                        progress.outcome_seq.store(next, Ordering::Relaxed);
                        progress.outcomes.fetch_add(count, Ordering::Relaxed);
                        progress.retried.fetch_add(applied.retried, Ordering::Relaxed);
                        false
                    }
                    Ok(Err(error)) => {
                        eprintln!("[rs] applying outcome file {next} failed: {error:#}");
                        true
                    }
                    Err(error) => {
                        eprintln!("[rs] applying outcome file {next} failed: {error}");
                        true
                    }
                }
            }
            Ok(Next::Wait) => {
                progress.outcome_lag.store(0, Ordering::Relaxed);
                true
            }
            Err(error) => {
                eprintln!("[rs] reading outcome file {next} failed: {error:#}");
                true
            }
        };
        if wait {
            tokio::select! {
                _ = stop.changed() => {}
                _ = tokio::time::sleep(PASS_SECONDS) => {}
            }
        }
    }
}

/// Mark what an earlier sender already sent, from tab-separated host, path, lastmod and Unix time lines.
pub fn import_sent(buckets: &Buckets, path: &Path) -> Result<usize> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut by_bucket: HashMap<usize, Vec<SentPage>> = HashMap::new();
    let mut known = 0;
    let mut flush = |by_bucket: &mut HashMap<usize, Vec<SentPage>>| -> Result<()> {
        for (_, rows) in by_bucket.drain() {
            known += buckets.store(&rows[0].domain).import_sent(&rows)?;
        }
        Ok(())
    };
    let mut held = 0;
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        let mut fields = line.split('\t');
        let (Some(host), Some(rest), Some(lastmod), Some(at)) = (fields.next(), fields.next(), fields.next(), fields.next()) else {
            continue;
        };
        if !buckets.owns(host) {
            continue;
        }
        let (Ok(lastmod), Ok(at)) = (lastmod.trim().parse::<f64>(), at.trim().parse::<f64>()) else {
            continue;
        };
        let page = SentPage { domain: host.to_string(), rest: rest.as_bytes().to_vec(), lastmod: lastmod as u32, at: at as u32 };
        by_bucket.entry(bucket_of(host)).or_default().push(page);
        held += 1;
        if held >= 50_000 {
            flush(&mut by_bucket)?;
            held = 0;
        }
    }
    flush(&mut by_bucket)?;
    Ok(known)
}

/// Walk every store once at a bounded pace and put each page never sent, or changed since sent, on its ready list.
pub fn backfill(buckets: &Buckets, per_second: u64, progress: &Progress, stop: &AtomicBool) -> Result<()> {
    let stores: Vec<_> = buckets.stores().collect();
    progress.backfill_of.store(stores.len() as u64, Ordering::Relaxed);
    let began = now();
    let chunk = per_second.clamp(1, 10_000) as usize;
    let started = Instant::now();
    let mut read: u64 = 0;
    for store in stores {
        let mut state = store.backfill_state(began)?;
        while !state.done {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let (scanned, added) = (state.scanned, state.added);
            store.backfill_step(&mut state, chunk)?;
            read += state.scanned - scanned;
            progress.backfill_scanned.fetch_add(state.scanned - scanned, Ordering::Relaxed);
            progress.backfill_added.fetch_add(state.added - added, Ordering::Relaxed);
            let due = Duration::from_secs_f64(read as f64 / per_second.max(1) as f64);
            if let Some(ahead) = due.checked_sub(started.elapsed()) {
                std::thread::sleep(ahead);
            }
        }
        progress.backfill_stores.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Each item's work on one of the reader threads, results in the items' order.
fn on_readers<T: Send, R: Send>(items: Vec<T>, work: impl Fn(T) -> Result<R> + Sync) -> Result<Vec<R>> {
    let items: Vec<Mutex<Option<T>>> = items.into_iter().map(|item| Mutex::new(Some(item))).collect();
    let done: Vec<Mutex<Option<Result<R>>>> = items.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..READERS.min(items.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i).and_then(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).take()) else {
                    break;
                };
                *done[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(work(item));
            });
        }
    });
    done.into_iter().map(|slot| slot.into_inner().unwrap_or_else(|e| e.into_inner()).expect("every item ran")).collect()
}

fn rank_of(buckets: &Buckets, domain: &str) -> i64 {
    let record = buckets.store_for(domain).and_then(|store| store.domain(domain).ok().flatten());
    record.and_then(|d| records::int(d.get("rank"))).unwrap_or(UNRANKED)
}

pub fn now() -> u32 {
    chrono::Utc::now().timestamp().clamp(0, u32::MAX.into()) as u32
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

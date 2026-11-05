//! Background work in loops of their own, so a backlog in one never holds up the others; only the copy holding the lease runs them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use desearch::feeds::OUTCOMES;
use desearch::time::now;
use redis::{AsyncCommands, Script};
use tokio::sync::watch;
use tokio::task::JoinSet;

use crate::budgets::{self, hour_of, KEEP_H};
use crate::clock::Tick;
use crate::http::ApiError;
use crate::queues::KINDS;
use crate::seeds::Seeds;
use crate::state::State;
use crate::{anchor, canary, checks, hours, journal, lifecycle, logs, outages, roundstore, snapshot, uploadlog, validations};
use serde_json::{json, Value};

/// The copy of the API that runs the passes; a rolling deploy briefly runs two copies.
pub const LEASE_KEY: &str = "janitor:leader";
/// Outlasts any one pass; a copy that stops hands the lease over at once instead.
const LEASE_MS: u64 = 60_000;
/// Leaders that queue rounds from URL lists staged in Redis; an older leader would queue such a round empty.
pub const STAGING_LEADER: &str = "staging:";

static HOLD: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"local holder = redis.call('GET', KEYS[1])
if holder and holder ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return 1",
    )
});
static RELEASE: LazyLock<Script> =
    LazyLock::new(|| Script::new(r"if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"));

const LEASE_EVERY: Duration = Duration::from_secs(1);
/// Every copy keeps the overview's pages this fresh, whoever holds the lease.
const WARM_EVERY: Duration = Duration::from_secs(15);
const SEAL_EVERY: Duration = Duration::from_secs(5);
/// Held tasks moved to the ready queue in one step at most; a backlog drains over a few steps.
const RELEASE_AT_ONCE: usize = 2_000;
const UPLOAD_LOG_EVERY: Duration = Duration::from_secs(30);
/// Log files published per pass at most, so a backlog drains over a few passes.
const PUBLISH_AT_ONCE: usize = 10;
const BEST_BLOCK_EVERY: Duration = Duration::from_secs(30);
/// The open list carries every open upload's URLs, so it is rewritten at most this often.
const OPEN_LIST_EVERY: Duration = Duration::from_secs(5);
/// A verdict drops its publish job a day after it was finalized.
const PUBLISH_COPY_KEEP_S: f64 = 86_400.0;
const SEAL_ROUNDS: i64 = 500;
/// Batches read per sealing step, so a pass over rounds that still hold embed page lists stays short.
const SEAL_BYTES: i64 = 16 << 20;
const DROP_VERDICTS: i64 = 200;
/// Per-URL details cleared per pass: each frees ~50 KB, ~2 ms of writer time.
const PRUNE_DETAILS: i64 = 10;
/// Verdict rows counted into the hourly totals per pass; a first start catches up on the kept week this way.
const ROLL_UP_ROWS: i64 = 5000;
/// Log entries tallied per pass, ~15 ms of writer time.
const ROLL_UP_ENTRIES: i64 = 5000;

#[derive(Clone, Copy, Debug)]
enum Duty {
    Journal,
    Anchor,
    Reclaim,
    Settle,
    Finalize,
    Publish,
    Rounds,
    Release,
    Upkeep,
    /// Its own loop, since a probe of a failing store may take as long as its timeouts.
    Canary,
}

const DUTIES: [Duty; 10] =
    [Duty::Journal, Duty::Anchor, Duty::Reclaim, Duty::Settle, Duty::Finalize, Duty::Publish, Duty::Rounds, Duty::Release, Duty::Upkeep, Duty::Canary];
/// How often a canary that is turned off looks again whether it still is.
const CANARY_OFF_EVERY: Duration = Duration::from_secs(60);

#[derive(Default)]
struct Due {
    anchoring: anchor::Anchoring,
    journaled_at: Option<Instant>,
    shared_at: Option<Instant>,
    snapshot_at: Option<Instant>,
    sealed_at: Option<Instant>,
    logged_at: Option<Instant>,
    kept_block_at: Option<Instant>,
    probed_at: Option<Instant>,
    /// The last storage probe that worked: an outage is dated from it.
    storage_ok: Option<Tick>,
    listed_at: Option<Instant>,
    pruned_hour: Option<i64>,
}

fn elapsed(at: Option<Instant>, interval: Duration) -> bool {
    at.is_none_or(|at| at.elapsed() >= interval)
}

impl Duty {
    fn every(self, state: &State) -> Duration {
        match self {
            Duty::Rounds | Duty::Anchor => Duration::from_secs(2),
            Duty::Canary if state.settings.canary_every > 0.0 => Duration::from_secs_f64(state.settings.canary_every),
            Duty::Canary => CANARY_OFF_EVERY,
            _ => Duration::from_secs(1),
        }
    }

    async fn pass(self, state: &Arc<State>, due: &mut Due) -> Result<()> {
        match self {
            Duty::Journal => journal_pass(state, due).await,
            Duty::Anchor => anchor::pass(state, &mut due.anchoring).await,
            Duty::Reclaim => lifecycle::reclaim_expired(state, state.rule_now().await?).await.map(drop),
            // A pause settles no upload and finalizes no verdict, so nobody is paid or penalized while it lasts.
            Duty::Settle | Duty::Finalize if state.custody.paused().is_some() => Ok(()),
            Duty::Settle => {
                lifecycle::recheck_malformed(state).await?;
                lifecycle::settle_seeded(state).await.map(drop)
            }
            Duty::Finalize => lifecycle::finalize_due(state, state.rule_now().await?).await.map(drop),
            Duty::Publish => {
                note_publish_rate(state).await?;
                if elapsed(due.listed_at, OPEN_LIST_EVERY) {
                    due.listed_at = Some(Instant::now());
                    lifecycle::publish_open(state, state.rule_now().await?).await?;
                }
                lifecycle::return_expired_publishes(state).await.map(drop)
            }
            Duty::Rounds => {
                // Each step runs whatever another one hit, so a chain error opening embed rounds holds up no reveal.
                let steps = [
                    ("opening embed rounds", lifecycle::open_embed_rounds(state).await.map(drop)),
                    ("filling revealed rounds", lifecycle::fill_missing(state).await.map(drop)),
                    ("revealing rounds", lifecycle::reveal_pending(state).await.map(drop)),
                    ("closing rounds", lifecycle::close_finished(state).await.map(drop)),
                ];
                for (step, done) in steps {
                    if let Err(error) = done {
                        eprintln!("janitor Rounds: {step} failed: {error:#}");
                    }
                }
                Ok(())
            }
            Duty::Release => {
                for kind in KINDS {
                    state.tasks(kind).release_due(&state.redis, now(), RELEASE_AT_ONCE).await?;
                }
                Ok(())
            }
            Duty::Upkeep => upkeep(state, due).await,
            Duty::Canary if state.settings.canary_every > 0.0 => canary::run(state).await.map(drop),
            Duty::Canary => Ok(()),
        }
    }
}

/// Writes the request hashes, the shares and the snapshots into the log, and publishes what it holds.
async fn journal_pass(state: &Arc<State>, due: &mut Due) -> Result<()> {
    let settings = &state.settings;
    if !elapsed(due.journaled_at, Duration::from_secs_f64(settings.journal_every)) {
        return Ok(());
    }
    due.journaled_at = Some(Instant::now());
    journal::log_requests(state).await?;
    // The first pass of a leader waits a whole interval, so copies taking turns do not each write the shares at once.
    if due.shared_at.is_none() {
        due.shared_at = Some(Instant::now());
    } else if elapsed(due.shared_at, Duration::from_secs_f64(settings.journal_shares_every)) {
        due.shared_at = Some(Instant::now());
        journal::log_shares(state).await?;
    }
    if due.snapshot_at.is_none() {
        due.snapshot_at = Some(Instant::now());
    } else if elapsed(due.snapshot_at, Duration::from_secs_f64(settings.snapshot_every)) {
        due.snapshot_at = Some(Instant::now());
        state.db.run(|conn| snapshot::record(conn, now())).await?;
    }
    for _ in 0..PUBLISH_AT_ONCE {
        if journal::publish(state).await?.is_none() {
            break;
        }
    }
    Ok(())
}

async fn upkeep(state: &Arc<State>, due: &mut Due) -> Result<()> {
    state.db.run(|conn| validations::prune_urls(conn, now(), PRUNE_DETAILS)).await?;
    state.db.run(|conn| validations::roll_up(conn, ROLL_UP_ROWS)).await?;
    state.db.run(|conn| hours::roll_up_claims(conn, ROLL_UP_ENTRIES, now())).await?;
    state.db.run(|conn| hours::roll_up_checks(conn, ROLL_UP_ROWS, now())).await?;
    if elapsed(due.sealed_at, SEAL_EVERY) {
        due.sealed_at = Some(Instant::now());
        OUTCOMES.fill_holes(&state.storage, &state.redis).await?;
        // Each step is its own transaction, so claims and verdicts get the writer between them.
        let deleted = state.db.run(|conn| roundstore::delete_expired(conn, now(), SEAL_ROUNDS, SEAL_BYTES)).await?;
        let (sealed, walked) = state.db.run(|conn| roundstore::seal_closed(conn, now(), SEAL_ROUNDS, SEAL_BYTES)).await?;
        if deleted + sealed > 0 {
            eprintln!("janitor rounds: deleted {deleted}, sealed {sealed}");
        }
        if let Some(closed_at) = walked {
            state.db.run(move |conn| roundstore::mark_sealed(conn, closed_at)).await?;
        }
        state.db.run(|conn| validations::drop_publish_copies(conn, now() - PUBLISH_COPY_KEEP_S, DROP_VERDICTS)).await?;
    }
    if elapsed(due.logged_at, UPLOAD_LOG_EVERY) {
        due.logged_at = Some(Instant::now());
        uploadlog::flush(state, &state.storage).await?;
    }
    if elapsed(due.kept_block_at, BEST_BLOCK_EVERY) {
        due.kept_block_at = Some(Instant::now());
        state.keep_best_block().await?;
        watch_finality(state).await?;
        watch_clock(state, true).await?;
    }
    let probe_every = Duration::from_secs_f64(state.settings.storage_probe_s.max(1.0));
    if state.probe_storage.swap(false, Ordering::AcqRel) || elapsed(due.probed_at, probe_every) {
        due.probed_at = Some(Instant::now());
        probe_storage(state, due).await?;
    }
    let at = state.rule_now().await?;
    if due.pruned_hour != Some(hour_of(at)) {
        state
            .db
            .run(move |conn| {
                budgets::prune(conn, KEEP_H, at)?;
                validations::prune_hours(conn, now())?;
                hours::prune(conn, now())?;
                canary::prune(conn, now())?;
                outages::prune(conn, at)?;
                crate::rules::prune(conn, at - 7.0 * 86_400.0)?;
                checks::prune(conn, at)?;
                // In rule time, as a replay prunes its own accounts at this entry.
                journal::append(conn, journal::Record::new("prune", at, json!({"keep_h": KEEP_H}), Value::Null))?;
                Ok(())
            })
            .await?;
        due.pruned_hour = Some(hour_of(at));
        let keep = state.settings.journal_keep;
        state.db.run(move |conn| journal::prune(conn, now(), keep)).await?;
    }
    Ok(())
}

/// Takes or renews the lease for `holder`; false while another copy holds it.
/// Writes and reads back the probe object; a failure opens a storage outage dated from the last probe that worked.
async fn probe_storage(state: &State, due: &mut Due) -> Result<()> {
    let tick = state.tick().await?;
    match outages::probe_storage(state, tick.at).await {
        Ok(()) => {
            if state.db.run(move |conn| outages::end_logged(conn, outages::STORAGE, tick.at, tick.block)).await? {
                eprintln!("storage outage over at block {}", tick.block);
            }
            due.storage_ok = Some(tick);
        }
        Err(why) => {
            let start = due.storage_ok.unwrap_or(tick);
            let detail = why.clone();
            if state.db.run(move |conn| outages::begin_logged(conn, outages::STORAGE, start.at, start.block, &detail)).await? {
                eprintln!("storage outage from block {}: {why}; claims alive in it lapse at no cost", start.block);
            }
        }
    }
    Ok(())
}

/// The host clock against the chain's. Once it is off by more than `TASK_API_CLOCK_SKEW_S`, signed requests are timed by the
/// chain, and a clock outage dated from the last check that agreed forgives every claim alive in it: a request refused for a
/// timestamp before the check caught it was refused for no fault of its miner. Every copy times its own requests; the copy that
/// leads (`log`) records the outage.
pub async fn watch_clock(state: &State, log: bool) -> Result<()> {
    let (Seeds::Chain(chain), limit) = (&state.seeds, state.settings.clock_skew_s) else { return Ok(()) };
    if limit <= 0.0 {
        return Ok(());
    }
    let (tick, best) = (state.tick().await?, chain.current_block().await?);
    let Some(block_time) = chain.block_time(tick.block).await? else { return Ok(()) };
    let offset = outages::host_offset(now(), block_time, best, tick.block, state.seeds.block_seconds());
    if offset.abs() > limit {
        state.set_clock_offset(offset);
        if !log {
            return Ok(());
        }
        let start = state.clock_ok.lock().expect("clock check").unwrap_or(tick);
        let detail = format!("the host clock is {offset:+.0}s off block {}'s time {block_time:.0}", tick.block);
        if state.db.run(move |conn| outages::begin_logged(conn, outages::CLOCK, start.at, start.block, &detail)).await? {
            eprintln!("host clock {offset:+.0}s off the chain's at block {}; requests are timed by the chain, claims alive lapse at no cost", tick.block);
        }
    } else {
        state.set_clock_offset(0.0);
        *state.clock_ok.lock().expect("clock check") = Some(tick);
        if log && state.db.run(move |conn| outages::end_logged(conn, outages::CLOCK, tick.at, tick.block)).await? {
            eprintln!("host clock agrees with the chain again at block {}", tick.block);
        }
    }
    Ok(())
}

/// Finality that falls behind the best block stops every deadline; the claims alive then lapse at no cost once it catches up.
async fn watch_finality(state: &State) -> Result<()> {
    let (Seeds::Chain(chain), limit) = (&state.seeds, state.settings.finality_stall_blocks) else { return Ok(()) };
    let (best, tick) = (chain.current_block().await?, state.tick().await?);
    let lag = best - tick.block;
    if lag > limit {
        let detail = format!("finalized block {} is {lag} blocks behind best {best}", tick.block);
        if state.db.run(move |conn| outages::begin_logged(conn, outages::FINALITY, tick.at, tick.block, &detail)).await? {
            eprintln!("finality stalled {lag} blocks behind best at block {}; claims alive now lapse at no cost", tick.block);
        }
    } else if state.db.run(move |conn| outages::end_logged(conn, outages::FINALITY, tick.at, tick.block)).await? {
        eprintln!("finality caught up at block {}", tick.block);
    }
    Ok(())
}

pub async fn hold(state: &State, holder: &str) -> Result<bool> {
    let held: i64 = HOLD.key(LEASE_KEY).arg(holder).arg(LEASE_MS).invoke_async(&mut state.redis.clone()).await?;
    Ok(held == 1)
}

/// False while a leader from before staged URL lists still runs, as in a rolling deploy.
pub async fn stages_rounds(state: &State) -> Result<bool> {
    let holder: Option<String> = state.redis.clone().get(LEASE_KEY).await?;
    Ok(holder.is_none_or(|holder| holder.starts_with(STAGING_LEADER)))
}

pub async fn release(state: &State, holder: &str) -> Result<()> {
    let _: i64 = RELEASE.key(LEASE_KEY).arg(holder).invoke_async(&mut state.redis.clone()).await?;
    Ok(())
}

/// Holds the lease and runs every duty while it does, until `stop` turns true; the lease goes only once every duty has stopped.
pub async fn run(state: Arc<State>, stop: watch::Receiver<bool>) {
    let holder = format!("{STAGING_LEADER}{}", uuid::Uuid::new_v4().simple());
    let leading = Arc::new(AtomicBool::new(false));
    let mut duties = JoinSet::new();
    for duty in DUTIES {
        duties.spawn(keep(duty, state.clone(), leading.clone(), stop.clone()));
    }
    let warming = tokio::spawn(keep_warm(state.clone(), stop.clone()));
    let mut stopping = stop.clone();
    while !*stopping.borrow() {
        if let Err(error) = crate::custody::refresh_pause(&state).await {
            eprintln!("could not read the pause: {error:#}");
        }
        // A copy that does not hold the log signs nothing, so it runs none of the passes.
        let held = if state.custody.signing() { hold(&state, &holder).await } else { Ok(false) };
        leading.store(matches!(held, Ok(true)), Ordering::Release);
        let followed = match held {
            Ok(true) => Ok(()),
            Ok(false) => follow(&state).await,
            Err(error) => Err(error),
        };
        if let Err(error) = followed {
            eprintln!("janitor lease failed: {error:#}");
        }
        tokio::select! {
            _ = tokio::time::sleep(LEASE_EVERY) => {}
            _ = stopping.changed() => {}
        }
    }
    leading.store(false, Ordering::Release);
    while duties.join_next().await.is_some() {}
    let _ = warming.await;
    if let Err(error) = release(&state, &holder).await {
        eprintln!("could not release the janitor lease: {error:#}");
    }
}

async fn keep_warm(state: Arc<State>, mut stop: watch::Receiver<bool>) {
    while !*stop.borrow() {
        if let Err(error) = watch_clock(&state, false).await {
            eprintln!("could not check the host clock against the chain's: {error:#}");
        }
        match logs::warm(&state).await {
            Err(ApiError::Internal(error)) => eprintln!("could not make the log pages: {error:#}"),
            Err(ApiError::Status { status, detail, .. }) => eprintln!("could not make the log pages: {status} {detail}"),
            Err(ApiError::Unavailable) | Ok(()) => {}
        }
        tokio::select! {
            _ = tokio::time::sleep(WARM_EVERY) => {}
            _ = stop.changed() => {}
        }
    }
}

async fn keep(duty: Duty, state: Arc<State>, leading: Arc<AtomicBool>, mut stop: watch::Receiver<bool>) {
    let mut due = Due::default();
    while !*stop.borrow() {
        if leading.load(Ordering::Acquire) {
            if let Err(error) = duty.pass(&state, &mut due).await {
                eprintln!("janitor {duty:?} failed: {error:#}");
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(duty.every(&state)) => {}
            _ = stop.changed() => {}
        }
    }
}

/// What a copy without the lease still needs to answer requests: the publish pace and the rounds to log refusals in.
async fn follow(state: &State) -> Result<()> {
    note_publish_rate(state).await?;
    let current = state.lookups.read(roundstore::latest_revealed).await?;
    *state.current.lock().expect("current rounds") = current;
    Ok(())
}

async fn note_publish_rate(state: &State) -> Result<()> {
    let finished = state.publish.finished(&state.redis).await?;
    state.publish_rate.lock().expect("publish rate").note(finished, now());
    Ok(())
}

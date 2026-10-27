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
use crate::http::ApiError;
use crate::queues::KINDS;
use crate::state::State;
use crate::{checks, hours, lifecycle, logs, roundstore, uploadlog, validations};

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
    Reclaim,
    Settle,
    Finalize,
    Publish,
    Rounds,
    Release,
    Upkeep,
}

const DUTIES: [Duty; 7] = [Duty::Reclaim, Duty::Settle, Duty::Finalize, Duty::Publish, Duty::Rounds, Duty::Release, Duty::Upkeep];

#[derive(Default)]
struct Due {
    sealed_at: Option<Instant>,
    logged_at: Option<Instant>,
    kept_block_at: Option<Instant>,
    listed_at: Option<Instant>,
    pruned_hour: Option<i64>,
}

fn elapsed(at: Option<Instant>, interval: Duration) -> bool {
    at.is_none_or(|at| at.elapsed() >= interval)
}

impl Duty {
    fn every(self) -> Duration {
        match self {
            Duty::Rounds => Duration::from_secs(2),
            _ => Duration::from_secs(1),
        }
    }

    async fn pass(self, state: &Arc<State>, due: &mut Due) -> Result<()> {
        match self {
            Duty::Reclaim => lifecycle::reclaim_expired(state, now()).await.map(drop),
            Duty::Settle => {
                lifecycle::recheck_malformed(state).await?;
                lifecycle::settle_seeded(state).await.map(drop)
            }
            Duty::Finalize => lifecycle::finalize_due(state, now()).await.map(drop),
            Duty::Publish => {
                note_publish_rate(state).await?;
                if elapsed(due.listed_at, OPEN_LIST_EVERY) {
                    due.listed_at = Some(Instant::now());
                    lifecycle::publish_open(state, now()).await?;
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
        }
    }
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
        state.db.run(|conn| roundstore::delete_expired(conn, now(), SEAL_ROUNDS, SEAL_BYTES)).await?;
        let (_, walked) = state.db.run(|conn| roundstore::seal_closed(conn, now(), SEAL_ROUNDS, SEAL_BYTES)).await?;
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
    }
    if due.pruned_hour != Some(hour_of(now())) {
        state
            .db
            .run(|conn| {
                let at = now();
                budgets::prune(conn, KEEP_H, at)?;
                validations::prune_hours(conn, at)?;
                hours::prune(conn, at)?;
                checks::prune(conn, at)
            })
            .await?;
        due.pruned_hour = Some(hour_of(now()));
    }
    Ok(())
}

/// Takes or renews the lease for `holder`; false while another copy holds it.
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
        let held = hold(&state, &holder).await;
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
            _ = tokio::time::sleep(duty.every()) => {}
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

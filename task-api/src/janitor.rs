//! The background pass every second: reclaims, settles, finalizes and publishes; rounds every few seconds; logs and pruning less often.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use desearch::feeds::OUTCOMES;
use desearch::time::now;
use redis::Script;
use tokio::sync::watch;

use crate::budgets::{self, hour_of, KEEP_H};
use crate::state::State;
use crate::{checks, lifecycle, roundstore, uploadlog, validations};

/// The copy of the API that runs the passes; a rolling deploy briefly runs two copies.
pub const LEASE_KEY: &str = "janitor:leader";
/// Outlasts any one pass; a copy that stops hands the lease over at once instead.
const LEASE_MS: u64 = 60_000;

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

const INTERVAL: Duration = Duration::from_secs(1);
const ROUNDS_INTERVAL: Duration = Duration::from_secs(5);
const UPLOAD_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// A day after closing, a round keeps only each batch's URL count and hash, and a verdict drops its publish job.
const LISTS_KEEP_S: f64 = 86_400.0;
const SEAL_ROUNDS: i64 = 20;
const DROP_VERDICTS: i64 = 200;

#[derive(Default)]
struct Due {
    rounds_at: Option<Instant>,
    logged_at: Option<Instant>,
    pruned_hour: Option<i64>,
}

fn elapsed(at: Option<Instant>, interval: Duration) -> bool {
    at.is_none_or(|at| at.elapsed() >= interval)
}

/// Takes or renews the lease for `holder`; false while another copy holds it.
pub async fn hold(state: &State, holder: &str) -> Result<bool> {
    let held: i64 = HOLD.key(LEASE_KEY).arg(holder).arg(LEASE_MS).invoke_async(&mut state.redis.clone()).await?;
    Ok(held == 1)
}

pub async fn release(state: &State, holder: &str) -> Result<()> {
    let _: i64 = RELEASE.key(LEASE_KEY).arg(holder).invoke_async(&mut state.redis.clone()).await?;
    Ok(())
}

/// Runs passes until `stop` turns true, then lets the next copy take over.
pub async fn run(state: Arc<State>, mut stop: watch::Receiver<bool>) {
    let holder = uuid::Uuid::new_v4().simple().to_string();
    let mut due = Due::default();
    while !*stop.borrow() {
        let passed = match hold(&state, &holder).await {
            Ok(true) => pass(&state, &mut due).await,
            Ok(false) => follow(&state).await,
            Err(error) => Err(error),
        };
        if let Err(error) = passed {
            eprintln!("janitor pass failed: {error:#}");
        }
        tokio::select! {
            _ = tokio::time::sleep(INTERVAL) => {}
            _ = stop.changed() => {}
        }
    }
    if let Err(error) = release(&state, &holder).await {
        eprintln!("could not release the janitor lease: {error:#}");
    }
}

/// What a copy without the lease still needs to answer requests: the publish pace and the rounds to log refusals in.
async fn follow(state: &State) -> Result<()> {
    note_publish_rate(state).await?;
    let current = state.reader.read(roundstore::latest_revealed).await?;
    *state.current.lock().expect("current rounds") = current;
    Ok(())
}

async fn note_publish_rate(state: &State) -> Result<()> {
    let finished = state.publish.finished(&state.redis).await?;
    state.publish_rate.lock().expect("publish rate").note(finished, now());
    Ok(())
}

async fn pass(state: &Arc<State>, due: &mut Due) -> Result<()> {
    note_publish_rate(state).await?;
    lifecycle::reclaim_expired(state, now()).await?;
    lifecycle::settle_seeded(state).await?;
    lifecycle::finalize_due(state, now()).await?;
    lifecycle::publish_open(state, now()).await?;
    lifecycle::return_expired_publishes(state).await?;
    if elapsed(due.rounds_at, ROUNDS_INTERVAL) {
        lifecycle::open_embed_rounds(state).await?;
        lifecycle::fill_missing(state).await?;
        lifecycle::reveal_pending(state).await?;
        lifecycle::close_finished(state).await?;
        OUTCOMES.fill_holes(&state.storage, &state.redis).await?;
        let kept_after = now() - LISTS_KEEP_S;
        state
            .db
            .run(move |conn| {
                roundstore::seal_closed(conn, kept_after, SEAL_ROUNDS)?;
                validations::drop_publish_copies(conn, kept_after, DROP_VERDICTS)
            })
            .await?;
        due.rounds_at = Some(Instant::now());
    }
    if elapsed(due.logged_at, UPLOAD_LOG_INTERVAL) {
        due.logged_at = Some(Instant::now());
        uploadlog::flush(state, &state.storage).await?;
    }
    if due.pruned_hour != Some(hour_of(now())) {
        state
            .db
            .run(|conn| {
                let at = now();
                budgets::prune(conn, KEEP_H, at)?;
                validations::prune_urls(conn, at)?;
                checks::prune(conn, at)
            })
            .await?;
        due.pruned_hour = Some(hour_of(now()));
    }
    Ok(())
}

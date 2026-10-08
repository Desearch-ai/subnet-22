//! The background pass every second: reclaims, settles, finalizes and publishes; rounds every few seconds; logs and pruning less often.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use desearch::feeds::OUTCOMES;
use desearch::time::now;

use crate::budgets::{self, hour_of, KEEP_H};
use crate::state::State;
use crate::{checks, lifecycle, roundstore, uploadlog, validations};

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

pub async fn run(state: Arc<State>) {
    let mut due = Due::default();
    loop {
        if let Err(error) = pass(&state, &mut due).await {
            eprintln!("janitor pass failed: {error:#}");
        }
        tokio::time::sleep(INTERVAL).await;
    }
}

async fn pass(state: &Arc<State>, due: &mut Due) -> Result<()> {
    let finished = state.publish.finished(&state.redis).await?;
    state.publish_rate.lock().expect("publish rate").note(finished, now());
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

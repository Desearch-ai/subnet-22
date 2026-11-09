//! The waiting line for embed tasks: a released task goes to the place that has waited longest, however often anyone asks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use redis::AsyncCommands;
use tokio::sync::{watch, Notify};

use crate::budgets::EMBED;
use crate::queues::{line_alive_key, line_key, ready_key, LINE_SLOTS};
use crate::state::Redis;
use crate::state::State;

/// Longest a claim waits for a task: well inside the 30 s miners' clients allow and the 60 s a signed request is good for when it arrives.
pub const MAX_WAIT_S: f64 = 20.0;
/// What a polling miner in line is told to wait: well inside what a place lasts, and short enough that its turn does not stand long.
pub const LINE_POLL_S: i64 = 5;
/// A waiting claim says it is still there this often, whatever else wakes it.
pub const STILL_THERE: Duration = Duration::from_secs(3);
/// How long a released task waits to be noticed by a claim waiting here.
const WAKE_EVERY: Duration = Duration::from_millis(100);

/// The claims waiting in this copy of the API, by hotkey.
#[derive(Default)]
pub struct Waiters(Mutex<HashMap<String, (Arc<Notify>, usize)>>);

/// A claim's wait, ended when it is dropped, however the request ends.
pub struct Waiting<'a> {
    waiters: &'a Waiters,
    hotkey: String,
    pub woken: Arc<Notify>,
    redis: Redis,
    /// Set once the claim is answered: a claim dropped before that is a miner gone, whose places stand in others' way.
    answered: bool,
}

impl Waiting<'_> {
    pub fn answered(&mut self) {
        self.answered = true;
    }
}

impl Waiters {
    pub fn wait(&self, hotkey: &str, redis: Redis) -> Waiting<'_> {
        let mut waiting = self.0.lock().expect("waiting claims");
        let (woken, count) = waiting.entry(hotkey.to_string()).or_default();
        *count += 1;
        Waiting { waiters: self, hotkey: hotkey.to_string(), woken: woken.clone(), redis, answered: false }
    }

    pub fn is_empty(&self) -> bool {
        self.0.lock().expect("waiting claims").is_empty()
    }

    /// Hotkeys with a claim waiting here.
    pub fn len(&self) -> usize {
        self.0.lock().expect("waiting claims").len()
    }

    /// Wakes the claims waiting here for these places.
    fn wake(&self, places: &[String]) {
        let waiting = self.0.lock().expect("waiting claims");
        for place in places {
            let hotkey = place.rsplit_once(':').map_or(place.as_str(), |(hotkey, _)| hotkey);
            if let Some((woken, _)) = waiting.get(hotkey) {
                woken.notify_one();
            }
        }
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut waiting = self.waiters.0.lock().expect("waiting claims");
        if let Some((_, count)) = waiting.get_mut(&self.hotkey) {
            *count -= 1;
            if *count == 0 {
                waiting.remove(&self.hotkey);
            }
        }
        if self.answered {
            return;
        }
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let (mut redis, hotkey) = (self.redis.clone(), self.hotkey.clone());
            runtime.spawn(async move {
                let places: Vec<String> = (0..LINE_SLOTS).map(|slot| format!("{hotkey}:{slot}")).collect();
                let leaving: redis::RedisResult<()> =
                    redis::pipe().zrem(line_key(EMBED), &places).ignore().zrem(line_alive_key(EMBED), &places).ignore().query_async(&mut redis).await;
                if let Err(error) = leaving {
                    eprintln!("could not let go of the places of {hotkey}, gone from its wait: {error}");
                }
            });
        }
    }
}

/// Wakes the claims whose places have come up, one round trip each step however many wait.
pub async fn keep_waking(state: Arc<State>, mut stop: watch::Receiver<bool>) {
    while !*stop.borrow() {
        if let Err(error) = wake_step(&state).await {
            eprintln!("waking waiting claims failed: {error:#}");
        }
        tokio::select! {
            _ = tokio::time::sleep(WAKE_EVERY) => {}
            _ = stop.changed() => {}
        }
    }
}

async fn wake_step(state: &State) -> Result<()> {
    if state.waiters.is_empty() {
        return Ok(());
    }
    let kind = EMBED;
    let ready: isize = state.redis.clone().zcard(ready_key(kind)).await?;
    if ready > 0 {
        let places: Vec<String> = state.redis.clone().zrange(line_key(kind), 0, ready - 1).await?;
        state.waiters.wake(&places);
    }
    Ok(())
}

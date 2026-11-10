//! The waiting lines, one for each kind of task: a released task goes to the place that has waited longest, however often anyone asks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use redis::AsyncCommands;
use tokio::sync::{watch, Notify};

use crate::queues::{line_alive_key, line_key, ready_key, KINDS, LINE_SLOTS};
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

/// A kind and a hotkey: one hotkey may wait for both kinds at once.
type Waiter = (&'static str, String);

/// The claims waiting in this copy of the API, each with what wakes it and how many there are.
#[derive(Default)]
pub struct Waiters(Mutex<HashMap<Waiter, (Arc<Notify>, usize)>>);

/// A claim's wait, ended when it is dropped, however the request ends.
pub struct Waiting<'a> {
    waiters: &'a Waiters,
    kind: &'static str,
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
    pub fn wait(&self, kind: &'static str, hotkey: &str, redis: Redis) -> Waiting<'_> {
        let mut waiting = self.0.lock().expect("waiting claims");
        let (woken, count) = waiting.entry((kind, hotkey.to_string())).or_default();
        *count += 1;
        Waiting { waiters: self, kind, hotkey: hotkey.to_string(), woken: woken.clone(), redis, answered: false }
    }

    pub fn is_empty(&self) -> bool {
        self.0.lock().expect("waiting claims").is_empty()
    }

    /// Hotkeys with a claim waiting here, counted once for each kind they wait for.
    pub fn len(&self) -> usize {
        self.0.lock().expect("waiting claims").len()
    }

    /// Whether any claim here waits for this kind.
    fn any_for(&self, kind: &str) -> bool {
        self.0.lock().expect("waiting claims").keys().any(|(waits_for, _)| *waits_for == kind)
    }

    /// Wakes the claims waiting here for these places in this kind's line.
    fn wake(&self, kind: &'static str, places: &[String]) {
        let waiting = self.0.lock().expect("waiting claims");
        for place in places {
            let hotkey = place.rsplit_once(':').map_or(place.as_str(), |(hotkey, _)| hotkey);
            if let Some((woken, _)) = waiting.get(&(kind, hotkey.to_string())) {
                woken.notify_one();
            }
        }
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut waiting = self.waiters.0.lock().expect("waiting claims");
        let key = (self.kind, self.hotkey.clone());
        if let Some((_, count)) = waiting.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                waiting.remove(&key);
            }
        }
        if self.answered {
            return;
        }
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let (mut redis, kind, hotkey) = (self.redis.clone(), self.kind, self.hotkey.clone());
            runtime.spawn(async move {
                let places: Vec<String> = (0..LINE_SLOTS).map(|slot| format!("{hotkey}:{slot}")).collect();
                let leaving: redis::RedisResult<()> =
                    redis::pipe().zrem(line_key(kind), &places).ignore().zrem(line_alive_key(kind), &places).ignore().query_async(&mut redis).await;
                if let Err(error) = leaving {
                    eprintln!("could not let go of the {kind} places of {hotkey}, gone from its wait: {error}");
                }
            });
        }
    }
}

/// Wakes the claims whose places have come up, one round trip a line each step however many wait.
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
    for kind in KINDS {
        if !state.settings.line(kind).on || !state.waiters.any_for(kind) {
            continue;
        }
        let ready: isize = state.redis.clone().zcard(ready_key(kind)).await?;
        if ready > 0 {
            let places: Vec<String> = state.redis.clone().zrange(line_key(kind), 0, ready - 1).await?;
            state.waiters.wake(kind, &places);
        }
    }
    Ok(())
}

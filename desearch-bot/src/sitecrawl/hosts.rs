//! Politeness per host: requests in flight and their pace, back off when overloaded, stop when refused.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use tokio::time::Instant;

pub const MIN_DELAY: f64 = 1.0;
pub const MAX_DELAY: f64 = 60.0;
pub const STOP: [u16; 4] = [401, 402, 403, 451];
pub const MAX_RATE_LIMITS: u32 = 2;
pub const MAX_SERVER_ERRORS: u32 = 5;
pub const OVERLOAD_SHARE: f64 = 0.05;
pub const OVERLOAD_WINDOW: usize = 100;
pub const OVERLOAD_MIN: usize = 20;
pub const BACKOFF_SLOTS: usize = 2;
pub const FORBIDDEN_RUN: u32 = 3;
pub const FORBIDDEN_MAX: usize = 5;
pub const FORBIDDEN_WINDOW: usize = 50;
const SLOT_POLL: Duration = Duration::from_millis(50);

struct State {
    delay: f64,
    limit: usize,
    active: usize,
    recent: VecDeque<bool>,
    forbidden: VecDeque<bool>,
    forbidden_run: u32,
    next_at: Option<Instant>,
    stopped: Option<String>,
    rate_limits: u32,
    server_errors: u32,
}

pub struct Host {
    pub name: String,
    pub slots: usize,
    max_rate_limits: u32,
    state: Mutex<State>,
}

/// Python's `float(text)`, 0 for anything it would not read.
pub fn seconds(text: Option<&str>) -> f64 {
    super::robots::py_float(text.unwrap_or("0")).unwrap_or(0.0)
}

fn push_window(window: &mut VecDeque<bool>, most: usize, value: bool) {
    if window.len() == most {
        window.pop_front();
    }
    window.push_back(value);
}

impl Host {
    /// `slots` and `min_delay` default to one request in flight and a second between starts.
    pub fn new(name: &str, delay: Option<f64>, slots: Option<usize>, min_delay: Option<f64>) -> Self {
        let delay_given = delay.unwrap_or(0.0);
        let min_delay = min_delay.unwrap_or(MIN_DELAY);
        // A Crawl-delay is a pace for one request at a time, so it keeps a host to one in flight.
        let slots = if delay_given != 0.0 { 1 } else { slots.unwrap_or(1) };
        Host {
            name: name.to_string(),
            slots,
            max_rate_limits: if slots == 1 { MAX_RATE_LIMITS } else { MAX_RATE_LIMITS * 10 },
            state: Mutex::new(State {
                delay: delay_given.max(min_delay).min(MAX_DELAY),
                limit: slots,
                active: 0,
                recent: VecDeque::with_capacity(OVERLOAD_WINDOW),
                forbidden: VecDeque::with_capacity(FORBIDDEN_WINDOW),
                forbidden_run: 0,
                next_at: None,
                stopped: None,
                rate_limits: 0,
                server_errors: 0,
            }),
        }
    }

    pub fn stopped(&self) -> Option<String> {
        self.state.lock().unwrap().stopped.clone()
    }

    pub fn delay(&self) -> f64 {
        self.state.lock().unwrap().delay
    }

    pub fn limit(&self) -> usize {
        self.state.lock().unwrap().limit
    }

    /// One of the host's requests in flight; waits while it has `limit` out.
    pub async fn slot(&self) -> Slot<'_> {
        loop {
            {
                let mut st = self.state.lock().unwrap();
                if st.active < st.limit {
                    st.active += 1;
                    return Slot(self);
                }
            }
            tokio::time::sleep(SLOT_POLL).await;
        }
    }

    /// Waits until the next request to this host is due.
    pub async fn turn(&self) {
        let next_at = self.state.lock().unwrap().next_at;
        if let Some(at) = next_at.filter(|at| *at > Instant::now()) {
            tokio::time::sleep_until(at).await;
        }
        let mut st = self.state.lock().unwrap();
        st.next_at = Some(Instant::now() + Duration::from_secs_f64(st.delay));
    }

    pub fn answer(&self, status: u16, retry_after: Option<&str>) {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        push_window(&mut st.recent, OVERLOAD_WINDOW, status == 429 || status == 0 || status >= 500);
        let troubled = st.recent.iter().filter(|&&b| b).count();
        if st.recent.len() >= OVERLOAD_MIN && troubled as f64 > OVERLOAD_SHARE * st.recent.len() as f64 {
            st.limit = st.limit.min(BACKOFF_SLOTS);
        }
        push_window(&mut st.forbidden, FORBIDDEN_WINDOW, status == 403);
        st.forbidden_run = if status == 403 { st.forbidden_run + 1 } else { 0 };
        if status == 403 {
            if st.forbidden_run >= FORBIDDEN_RUN || st.forbidden.iter().filter(|&&b| b).count() > FORBIDDEN_MAX {
                st.stopped = Some("refused_403".into());
            }
        } else if STOP.contains(&status) {
            st.stopped = Some(format!("refused_{status}"));
        } else if status == 429 {
            st.rate_limits += 1;
            st.limit = st.limit.min(BACKOFF_SLOTS);
            if st.rate_limits >= self.max_rate_limits {
                st.stopped = Some("rate_limited".into());
            } else {
                st.delay = (st.delay * 4.0).max(seconds(retry_after)).clamp(MIN_DELAY, MAX_DELAY);
                st.next_at = Some(Instant::now() + Duration::from_secs_f64(st.delay));
            }
        } else if status == 0 || status >= 500 {
            st.server_errors += 1;
            if st.server_errors >= MAX_SERVER_ERRORS {
                st.stopped = Some(format!("server_errors_{status}"));
            }
        } else {
            st.server_errors = 0;
        }
    }
}

pub struct Slot<'a>(&'a Host);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().active -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn pace_takes_the_robots_delay_with_our_floor() {
        assert_eq!(Host::new("a", None, None, None).delay(), MIN_DELAY);
        let h = Host::new("a", Some(5.0), None, None);
        let start = Instant::now();
        let mut starts = Vec::new();
        for _ in 0..3 {
            h.turn().await;
            starts.push((Instant::now() - start).as_secs_f64());
        }
        assert_eq!(starts, [0.0, 5.0, 10.0]);
        assert_eq!(Host::new("a", Some(500.0), None, None).delay(), MAX_DELAY);
        assert_eq!(Host::new("a", Some(0.0), Some(3), None).slots, 3, "a zero delay is no delay");
        assert_eq!(Host::new("a", Some(0.5), Some(3), None).slots, 1);
    }

    #[test]
    fn stops_when_refused_or_rate_limited() {
        let h = Host::new("a", None, None, None);
        h.answer(401, None);
        assert_eq!(h.stopped().as_deref(), Some("refused_401"));
        let h = Host::new("a", None, None, None);
        h.answer(429, Some("30"));
        assert_eq!((h.stopped(), h.delay()), (None, 30.0));
        h.answer(429, None);
        assert_eq!(h.stopped().as_deref(), Some("rate_limited"));
        let h = Host::new("a", None, None, None);
        h.answer(429, Some("soon"));
        assert_eq!(h.delay(), 4.0, "an unreadable Retry-After waits four times the pace");
        let h = Host::new("a", None, None, None);
        for _ in 0..5 {
            h.answer(503, None);
        }
        assert_eq!(h.stopped().as_deref(), Some("server_errors_503"));
        let h = Host::new("a", None, None, None);
        for status in [500, 500, 500, 500, 404, 500, 0, 0, 0] {
            h.answer(status, None);
        }
        assert_eq!(h.stopped(), None, "a good answer starts the count again");
        h.answer(0, None);
        assert_eq!(h.stopped().as_deref(), Some("server_errors_0"));
        let h = Host::new("a", None, None, None);
        for status in [500, 500, 500, 500, 403, 429] {
            h.answer(status, None);
        }
        h.answer(502, None);
        assert_eq!(h.stopped().as_deref(), Some("server_errors_502"), "a 403 or 429 does not reset the server errors");
    }

    #[test]
    fn stops_on_forbidden_answers_in_a_row_or_too_many_of_late() {
        let (in_a_row, scattered, once) = (Host::new("a", None, None, None), Host::new("b", None, None, None), Host::new("c", None, None, None));
        for status in [200, 403, 403, 403] {
            in_a_row.answer(status, None);
        }
        for status in [403, 200].repeat(6) {
            scattered.answer(status, None);
        }
        for status in [403, 200, 200, 403, 200] {
            once.answer(status, None);
        }
        assert_eq!(in_a_row.stopped().as_deref(), Some("refused_403"));
        assert_eq!(scattered.stopped().as_deref(), Some("refused_403"));
        assert_eq!(once.stopped(), None);
    }

    #[test]
    fn backs_off_to_two_in_flight_when_overloaded() {
        let host = Host::new("h", None, Some(20), Some(0.0));
        for status in [vec![200; 18], vec![503, 503]].concat() {
            host.answer(status, None);
        }
        assert_eq!((host.limit(), host.stopped()), (BACKOFF_SLOTS, None));
        assert_eq!(Host::new("h", None, Some(20), None).max_rate_limits, MAX_RATE_LIMITS * 10);
    }
}

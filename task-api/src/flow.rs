//! How much work the system can take, from how fast the publisher has been finishing it.

use std::collections::VecDeque;

/// Work in the system the bot may fill to, in seconds of publishing at the measured rate; well above the ~10 min a task takes end to end.
pub const LAG_TARGET_S: f64 = 1800.0;
/// Claims stop once this much is waiting to be published.
pub const LAG_LIMIT_S: f64 = 1800.0;
/// Assumed until the publisher has been measured, so a fresh start is not held at zero.
pub const RATE_FLOOR: f64 = 100.0 / 60.0;
pub const WINDOW_S: f64 = 300.0;

/// Tasks a second the publisher finished over the last few minutes.
pub struct PublishRate {
    pub lag_s: f64,
    pub limit_s: f64,
    samples: VecDeque<(f64, i64)>,
}

impl PublishRate {
    pub fn new(lag_s: f64, limit_s: f64) -> Self {
        PublishRate { lag_s, limit_s, samples: VecDeque::new() }
    }

    pub fn note(&mut self, acked: i64, at: f64) {
        self.samples.push_back((at, acked));
        while self.samples.len() > 2 && self.samples[1].0 <= at - WINDOW_S {
            self.samples.pop_front();
        }
    }

    pub fn per_second(&self) -> f64 {
        let (Some(&(first_at, first)), Some(&(last_at, last))) = (self.samples.front(), self.samples.back()) else {
            return RATE_FLOOR;
        };
        if self.samples.len() < 2 || last_at <= first_at {
            return RATE_FLOOR;
        }
        RATE_FLOOR.max((last - first) as f64 / (last_at - first_at))
    }

    /// Tasks that may still enter before the work ahead of the publisher passes the target.
    pub fn room(&self, in_system: i64) -> i64 {
        0.max((self.per_second() * self.lag_s) as i64 - in_system)
    }

    pub fn overloaded(&self, waiting: i64) -> bool {
        waiting as f64 > self.per_second() * self.limit_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rate_is_measured_over_a_window() {
        let mut rate = PublishRate::new(LAG_TARGET_S, LAG_LIMIT_S);
        assert_eq!(rate.per_second(), RATE_FLOOR);
        rate.note(0, 0.0);
        rate.note(600, 60.0);
        assert_eq!(rate.per_second(), 10.0);
        assert_eq!(rate.room(1000), 17_000);
        assert!(rate.overloaded(18_001) && !rate.overloaded(18_000));
        rate.note(600, 400.0);
        rate.note(1800, 460.0);
        assert_eq!(rate.per_second(), 1200.0 / 400.0);
    }
}

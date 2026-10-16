//! How fast the publisher has been finishing work, reported on `/v1/room`.

use std::collections::VecDeque;

pub const WINDOW_S: f64 = 300.0;

/// Tasks a second the publisher finished over the last few minutes.
#[derive(Default)]
pub struct PublishRate {
    samples: VecDeque<(f64, i64)>,
}

impl PublishRate {
    pub fn note(&mut self, acked: i64, at: f64) {
        self.samples.push_back((at, acked));
        while self.samples.len() > 2 && self.samples[1].0 <= at - WINDOW_S {
            self.samples.pop_front();
        }
    }

    pub fn per_second(&self) -> f64 {
        let (Some(&(first_at, first)), Some(&(last_at, last))) = (self.samples.front(), self.samples.back()) else {
            return 0.0;
        };
        if last_at <= first_at {
            return 0.0;
        }
        (last - first) as f64 / (last_at - first_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rate_is_measured_over_a_window() {
        let mut rate = PublishRate::default();
        assert_eq!(rate.per_second(), 0.0);
        rate.note(0, 0.0);
        rate.note(600, 60.0);
        assert_eq!(rate.per_second(), 10.0);
        rate.note(600, 400.0);
        rate.note(1800, 460.0);
        assert_eq!(rate.per_second(), 1200.0 / 400.0);
    }
}

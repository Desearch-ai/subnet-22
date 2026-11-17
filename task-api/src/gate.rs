use std::collections::HashMap;
use std::sync::Mutex;

const SWEEP_AT: usize = 50_000;

#[derive(Default)]
pub struct Gate {
    counts: Mutex<HashMap<String, (f64, i64)>>,
}

impl Gate {
    pub fn take(&self, key: &str, window_s: f64, limit: i64, now: f64) -> Option<f64> {
        let ends = ((now / window_s).floor() + 1.0) * window_s;
        let mut counts = self.counts.lock().expect("gate");
        if counts.len() > SWEEP_AT {
            counts.retain(|_, (until, _)| *until > now);
        }
        let entry = counts.entry(key.to_string()).or_insert((ends, 0));
        if entry.0 != ends {
            *entry = (ends, 0);
        }
        entry.1 += 1;
        (entry.1 > limit).then(|| (ends - now).max(0.01))
    }

    pub fn count(&self, key: &str, window_s: f64, now: f64) -> i64 {
        let ends = ((now / window_s).floor() + 1.0) * window_s;
        self.counts.lock().expect("gate").get(key).filter(|(until, _)| *until == ends).map_or(0, |(_, n)| *n)
    }
}

#[cfg(test)]
mod tests {
    use super::Gate;

    #[test]
    fn a_window_admits_its_limit_and_then_waits_for_the_next() {
        let gate = Gate::default();
        assert!((0..5).all(|_| gate.take("k", 10.0, 5, 101.0).is_none()));
        assert_eq!(gate.take("k", 10.0, 5, 104.0), Some(6.0));
        assert_eq!(gate.count("k", 10.0, 104.0), 6);
        assert!(gate.take("k", 10.0, 5, 110.0).is_none());
        assert!(gate.take("other", 10.0, 5, 104.0).is_none());
    }
}

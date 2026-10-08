//! How a verdict turns into paid rows, the one rule the task API and every validator share (`desearch.credit`).

use crate::py::round;

pub const COVERAGE_GATE: f64 = 0.85;
pub const EVIDENCE: [&str; 4] = ["matched", "mismatched", "errors_confirmed", "errors_unconfirmed"];
pub const RECENT_CHECKS: i64 = 10;
pub const FAILS_FOR_PENALTY: i64 = 2;
pub const PENALTY_WINDOW_S: f64 = 24.0 * 3600.0;
/// A miner's checks of this long set the rate its unchecked uploads are paid at.
pub const RATE_WINDOW_S: f64 = 72.0 * 3600.0;
/// Reported content rows may exceed what a check counts by this share of the task before it fails.
pub const REPORTED_SLACK: f64 = 0.02;

/// How many samples had each outcome.
pub trait Outcomes {
    fn count(&self, outcome: &str) -> i64;
}

/// Paid at the sample's rate, so an unsampled forgery still costs.
pub fn credited_urls(ok_rows: i64, error_rows: i64, outcomes: &impl Outcomes) -> i64 {
    let compared = outcomes.count("matched") + outcomes.count("mismatched");
    let judged = outcomes.count("errors_confirmed") + outcomes.count("errors_unconfirmed");
    let pages = if compared > 0 { round((ok_rows * outcomes.count("matched")) as f64 / compared as f64) } else { 0 };
    let errors = if judged > 0 { round((error_rows * outcomes.count("errors_confirmed")) as f64 / judged as f64) } else { 0 };
    pages + errors
}

/// More content rows in the miner's report than a check counted in its file.
pub fn overstated(reported_ok: i64, counted_content: i64, assigned: usize) -> bool {
    reported_ok as f64 > counted_content as f64 + (REPORTED_SLACK * assigned as f64).max(1.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    impl Outcomes for HashMap<&str, i64> {
        fn count(&self, outcome: &str) -> i64 {
            self.get(outcome).copied().unwrap_or(0)
        }
    }

    #[test]
    fn credit_is_paid_at_the_sample_rate() {
        let outcomes = HashMap::from([("matched", 3), ("mismatched", 1), ("errors_confirmed", 1), ("errors_unconfirmed", 1)]);
        assert_eq!(credited_urls(10, 5, &outcomes), 8 + 2);
        assert_eq!(credited_urls(1, 0, &HashMap::from([("matched", 1), ("mismatched", 1)])), 0);
        assert_eq!(credited_urls(3, 0, &HashMap::from([("matched", 1), ("mismatched", 1)])), 2);
        assert!(overstated(103, 100, 100) && !overstated(102, 100, 100) && overstated(14, 1, 600) && !overstated(13, 1, 600));
    }
}

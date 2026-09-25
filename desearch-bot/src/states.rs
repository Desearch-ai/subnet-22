//! What a domain is, and when to look at it again. Times are epoch microseconds.

use rand::Rng;

use crate::schedule::{DAY, HOUR, MINUTE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    New,
    Active,
    Failing,
    Down,
    Unreachable,
    NoSitemap,
    Redirects,
    Blocked,
    Ineligible,
    Excluded,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::New => "new",
            State::Active => "active",
            State::Failing => "failing",
            State::Down => "down",
            State::Unreachable => "unreachable",
            State::NoSitemap => "no_sitemap",
            State::Redirects => "redirects",
            State::Blocked => "blocked",
            State::Ineligible => "ineligible",
            State::Excluded => "excluded",
        }
    }

    pub fn parse(value: &str) -> Option<State> {
        Some(match value {
            "new" => State::New,
            "active" => State::Active,
            "failing" => State::Failing,
            "down" => State::Down,
            "unreachable" => State::Unreachable,
            "no_sitemap" => State::NoSitemap,
            "redirects" => State::Redirects,
            "blocked" => State::Blocked,
            "ineligible" => State::Ineligible,
            "excluded" => State::Excluded,
            _ => return None,
        })
    }

    /// Known sites the loop keeps fresh, as opposed to domains still being discovered.
    pub fn refreshes(self) -> bool {
        matches!(self, State::Active | State::Failing | State::Down)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Sitemap,
    NoSitemap,
    Redirect,
    Blocked,
    Ineligible,
    Unreachable,
    Excluded,
}

impl Outcome {
    /// The site itself answered, whatever it said.
    pub fn answered(self) -> bool {
        matches!(self, Outcome::Sitemap | Outcome::NoSitemap | Outcome::Redirect | Outcome::Blocked | Outcome::Ineligible)
    }
}

const RETRY: [i64; 5] = [15 * MINUTE, 30 * MINUTE, HOUR, 2 * HOUR, 4 * HOUR];
const DOWN_AFTER: i64 = 7 * DAY;
/// Spread rechecks so domains visited together do not all fall due in the same second.
pub const JITTER: f64 = 0.1;

fn recheck(state: State) -> i64 {
    match state {
        State::Active | State::Down => DAY,
        State::Ineligible => 90 * DAY,
        _ => 30 * DAY,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub state: State,
    pub failures: i64,
    pub next_check_at: Option<i64>,
}

/// The state a visit leaves a domain in, and when to look at it again.
pub fn decide(
    previous: State,
    outcome: Outcome,
    failures: i64,
    last_ok_at: Option<i64>,
    now: i64,
    rng: &mut impl Rng,
) -> Decision {
    let settled = match outcome {
        Outcome::Excluded => return Decision { state: State::Excluded, failures: 0, next_check_at: None },
        Outcome::Unreachable => return failed(previous, failures + 1, last_ok_at, now, rng),
        Outcome::Sitemap => State::Active,
        Outcome::NoSitemap => State::NoSitemap,
        Outcome::Redirect => State::Redirects,
        Outcome::Blocked => State::Blocked,
        Outcome::Ineligible => State::Ineligible,
    };
    Decision { state: settled, failures: 0, next_check_at: Some(after(now, recheck(settled), rng)) }
}

fn failed(previous: State, failures: i64, last_ok_at: Option<i64>, now: i64, rng: &mut impl Rng) -> Decision {
    let down = |rng: &mut _| Decision { state: State::Down, failures, next_check_at: Some(after(now, recheck(State::Down), rng)) };
    match previous {
        State::Active | State::Failing => {
            if last_ok_at.is_some_and(|ok| now - ok >= DOWN_AFTER) {
                return down(rng);
            }
            let retry = RETRY[(failures.clamp(1, RETRY.len() as i64) - 1) as usize];
            Decision { state: State::Failing, failures, next_check_at: Some(now + retry) }
        }
        State::Down => down(rng),
        _ => Decision {
            state: State::Unreachable,
            failures,
            next_check_at: Some(after(now, recheck(State::Unreachable), rng)),
        },
    }
}

fn after(now: i64, interval: i64, rng: &mut impl Rng) -> i64 {
    now + (interval as f64 * rng.gen_range(1.0 - JITTER..=1.0 + JITTER)).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_back_off_then_go_down() {
        let mut rng = rand::thread_rng();
        let now = 1_700_000_000_000_000;
        let first = decide(State::Active, Outcome::Unreachable, 0, Some(now - HOUR), now, &mut rng);
        assert_eq!((first.state, first.failures, first.next_check_at), (State::Failing, 1, Some(now + 15 * MINUTE)));
        let later = decide(State::Failing, Outcome::Unreachable, 9, Some(now - HOUR), now, &mut rng);
        assert_eq!(later.next_check_at, Some(now + 4 * HOUR));
        let down = decide(State::Failing, Outcome::Unreachable, 3, Some(now - 8 * DAY), now, &mut rng);
        assert_eq!(down.state, State::Down);
        let new = decide(State::New, Outcome::Unreachable, 0, None, now, &mut rng);
        assert_eq!(new.state, State::Unreachable);
        assert!(new.next_check_at.unwrap() >= now + 27 * DAY);
    }
}

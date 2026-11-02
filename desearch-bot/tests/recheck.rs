//! A recheck's failed visits: tried again soon, spread out, then judged as before.

use desearch_bot::crawl::plan;
use desearch_bot::records;
use desearch_bot::states::{Outcome, State};
use desearch_bot::visit::{Known, Visit};
use rand::SeedableRng;
use serde_json::json;

const NOW: i64 = 1_800_000_000_000_000;
const SECOND: i64 = 1_000_000;
const RETRIES: [i64; 3] = [900, 7_200, 43_200];

fn known(state: &str, failures: i64, ok: Option<i64>) -> Known {
    let mut record = json!({"state": state, "failures": failures, "categories": []});
    if let Some(ok) = ok {
        record["ok"] = ok.into();
    }
    records::known("a.test", record.as_object().unwrap(), &[]).unwrap()
}

fn refused(passing: bool) -> Visit {
    Visit { outcome: Outcome::Ineligible, reason: Some("homepage_error".into()), home_status: Some(403), requests: 3, passing, ..Visit::new("a.test") }
}

/// The states a recheck goes through, visit after visit, and how long after `NOW` each retry is due.
fn chain(visit: impl Fn() -> Visit, seed: u64) -> Vec<(State, Option<i64>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut failures = 0;
    let mut out = Vec::new();
    for _ in 0..5 {
        let write = plan(&known("new", failures, Some(1_700_000_000)), visit(), NOW, &mut rng);
        out.push((write.state, write.next_due_at.map(|due| (due - NOW) / SECOND)));
        if write.state != State::New {
            break;
        }
        failures = write.failures;
    }
    out
}

fn assert_retried(chain: &[(State, Option<i64>)], last: State) {
    assert_eq!(chain.len(), 4, "{chain:?}");
    for ((state, due), wait) in chain.iter().zip(RETRIES) {
        assert_eq!(*state, State::New);
        let due = due.unwrap();
        assert!((wait..=wait * 3 / 2).contains(&due), "a retry due {due} s on, against {wait} s");
    }
    assert_eq!(chain[3].0, last);
}

#[test]
fn a_first_refusal_makes_the_site_ineligible() {
    let write = plan(&known("new", 0, None), refused(false), NOW, &mut rand::thread_rng());
    assert_eq!(write.state, State::Ineligible);
}

#[test]
fn a_refused_recheck_is_tried_three_times_then_ineligible() {
    assert_retried(&chain(|| refused(true), 1), State::Ineligible);
}

#[test]
fn a_timed_out_recheck_is_tried_three_times_then_unreachable() {
    let timeout = || Visit { outcome: Outcome::Unreachable, reason: Some("TimeoutError".into()), requests: 1, ..Visit::new("a.test") };
    assert_retried(&chain(timeout, 2), State::Unreachable);
}

#[test]
fn a_recheck_whose_sitemaps_failed_is_tried_three_times_then_has_none() {
    let refused_sitemap = || Visit { outcome: Outcome::NoSitemap, reason: Some("sitemap_429".into()), requests: 4, passing: true, ..Visit::new("a.test") };
    assert_retried(&chain(refused_sitemap, 3), State::NoSitemap);
}

#[test]
fn retries_of_rechecks_failing_together_spread_out() {
    let dues: std::collections::HashSet<i64> = (0..20).map(|seed| chain(|| refused(true), seed)[0].1.unwrap()).collect();
    assert!(dues.len() > 10, "{} distinct first retries of 20", dues.len());
}

#[test]
fn a_visit_that_asked_nothing_is_not_due_again_at_once_while_one_that_read_waits_for_nothing() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let stale = json!({"id": 1, "kind": "urlset", "depth": 0, "status": "ok", "urls": 3, "next": (NOW / SECOND) - 86_400});
    let record = json!({"state": "active", "failures": 0, "categories": [], "lang": "en"});
    let known = records::known("a.test", record.as_object().unwrap(), &[("https://a.test/s.xml".into(), stale.as_object().unwrap().clone())]).unwrap();
    let idle = plan(&known, Visit { requests: 0, ..Visit::new("a.test") }, NOW, &mut rng);
    assert_eq!(idle.next_due_at, Some(NOW + 3600 * SECOND), "it asked nothing, so it would ask nothing again at once");
    assert!(!idle.backlog);
    let busy = plan(&known, Visit { requests: 3, ..Visit::new("a.test") }, NOW, &mut rng);
    assert!(busy.next_due_at.is_some_and(|due| due <= NOW) && busy.backlog, "a visit that read files keeps reading the backlog at once");
}

#[test]
fn a_walk_stopped_for_the_disk_waits_no_hour_for_the_files_it_left() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let stale = json!({"id": 1, "kind": "urlset", "depth": 0, "status": "ok", "urls": 3, "next": (NOW / SECOND) - 86_400});
    let record = json!({"state": "active", "failures": 0, "categories": [], "lang": "en"});
    let known = records::known("a.test", record.as_object().unwrap(), &[("https://a.test/s.xml".into(), stale.as_object().unwrap().clone())]).unwrap();
    let paused = plan(&known, Visit { requests: 0, paused: true, ..Visit::new("a.test") }, NOW, &mut rng);
    assert!(paused.next_due_at.is_some_and(|due| due <= NOW), "its due files wait only for the disk");
}

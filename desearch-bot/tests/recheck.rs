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

/// A domain's record and stored sitemaps, as round B may have left them, and a visit to it: off mode schedules each as round A does.
fn off_mode_cases() -> Vec<(&'static str, Known, Visit)> {
    let sitemap = |url: &str, kind: &str, next: i64| (url.to_string(), json!({"id": 1, "kind": kind, "depth": 0, "status": "ok", "urls": 3, "next": NOW / SECOND + next}).as_object().unwrap().clone());
    let domain = |state: &str, b_fields: bool| {
        let mut record = json!({"state": state, "failures": 0, "categories": [], "lang": "en", "ok": NOW / SECOND - 86_400, "checked": NOW / SECOND - 3_600});
        if b_fields {
            for (name, at) in [("recheck", NOW / SECOND + 600), ("recheck_of", NOW / SECOND - 3_600), ("links_due", NOW / SECOND - 60)] {
                record[name] = at.into();
            }
        }
        record
    };
    let known = |state: &str, b_fields: bool, sitemaps: &[(String, serde_json::Map<String, serde_json::Value>)]| records::known("a.test", domain(state, b_fields).as_object().unwrap(), sitemaps).unwrap();
    let urlset = sitemap("https://a.test/s.xml", "urlset", 6 * 3_600);
    let later = sitemap("https://a.test/s.xml", "urlset", 5 * 86_400);
    let stale = sitemap("https://a.test/s.xml", "urlset", -86_400);
    let links_soon = sitemap("https://a.test/#links", "links", 2 * 86_400);
    let links_past = sitemap("https://a.test/#links", "links", -600);
    let read = |outcome: Outcome, requests: i64| Visit { outcome, requests, ..Visit::new("a.test") };
    vec![
        ("active, sitemap in 6 h", known("active", false, &[urlset.clone()]), read(Outcome::Sitemap, 2)),
        ("active, sitemap due, read", known("active", false, &[stale.clone()]), read(Outcome::Sitemap, 3)),
        ("active, sitemap due, nothing asked", known("active", false, &[stale]), read(Outcome::Sitemap, 0)),
        ("active, links run in 2 days", known("active", false, &[later, links_soon.clone()]), read(Outcome::Sitemap, 2)),
        ("active, links run past", known("active", false, &[urlset.clone(), links_past.clone()]), read(Outcome::Sitemap, 2)),
        ("active, round B fields", known("active", true, &[urlset.clone(), links_past.clone()]), read(Outcome::Sitemap, 2)),
        ("no_sitemap, links run past", known("no_sitemap", true, &[links_past]), read(Outcome::NoSitemap, 5)),
        ("no_sitemap, links run soon", known("no_sitemap", false, &[links_soon]), read(Outcome::NoSitemap, 5)),
        ("new, found a sitemap", known("new", false, &[]), read(Outcome::Sitemap, 4)),
        ("new, none found", known("new", true, &[]), read(Outcome::NoSitemap, 6)),
    ]
}

#[test]
fn off_mode_schedules_each_case_as_round_a_did() {
    let got: Vec<(&str, Option<i64>, bool)> = off_mode_cases()
        .into_iter()
        .map(|(name, known, visit)| {
            let write = plan(&known, visit, NOW, &mut rand::rngs::StdRng::seed_from_u64(11));
            (name, write.next_due_at.map(|due| (due - NOW) / SECOND), write.backlog)
        })
        .collect();
    // Seconds from now to the next visit, and whether it is backlog, as round A with the idle-visit fix schedules them.
    let round_a = [
        ("active, sitemap in 6 h", Some(21_600), false),
        ("active, sitemap due, read", Some(-86_400), true),
        ("active, sitemap due, nothing asked", Some(3_600), false),
        ("active, links run in 2 days", Some(82_528), false),
        ("active, links run past", Some(21_600), false),
        ("active, round B fields", Some(21_600), false),
        ("no_sitemap, links run past", Some(2_475_858), false),
        ("no_sitemap, links run soon", Some(2_475_858), false),
        ("new, found a sitemap", Some(82_528), false),
        ("new, none found", Some(2_475_858), false),
    ];
    assert_eq!(got, round_a);
}

#[test]
fn a_visit_a_crawl_delay_stopped_decides_nothing_and_comes_back_after_the_delay() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for state in ["new", "active", "no_sitemap"] {
        let mut record = json!({"state": state, "failures": 2, "reason": "TimeoutError", "categories": []});
        record["ok"] = 1_700_000_000.into();
        let known = records::known("a.test", record.as_object().unwrap(), &[]).unwrap();
        let visit = Visit { paced: true, quiet_until: Some(NOW + 600 * SECOND), deferred: vec![("http://a.test/more.xml".into(), 1, None)], requests: 1, ..Visit::new("a.test") };
        let write = plan(&known, visit, NOW, &mut rng);
        assert_eq!((write.state, write.failures, write.reason.as_deref()), (known.state, 2, Some("TimeoutError")), "{state}");
        assert!(write.next_due_at.unwrap() >= NOW + 600 * SECOND && !write.backlog, "{state}: sitemaps left for later wait for the delay too");
    }
    let known = records::known("a.test", json!({"state": "active", "delay": 43_200.0, "categories": []}).as_object().unwrap(), &[]).unwrap();
    assert!(desearch_bot::crawl::crashed(&known, "StoreError", NOW).next_due_at.unwrap() >= NOW + 43_200 * SECOND, "a crash retries after the delay, not the hour");
}

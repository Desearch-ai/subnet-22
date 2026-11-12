//! The refresh commands' readers and checks against the Python crawler's answers in `tests/data/refresh/units.json`.

use std::collections::HashSet;

use chrono::NaiveDate;
use indexmap::IndexSet;
use serde_json::{json, Value};

use desearch_bot::refresh::{check_edgar, check_fred};
use desearch_bot::setrule::SetRule;
use desearch_bot::sitecrawl::edgar::{filings, index_ciks, Pace, FORMS};
use desearch_bot::sitecrawl::lists::group_urls;
use desearch_bot::sitecrawl::Counts;

fn units() -> Value {
    serde_json::from_str(include_str!("data/refresh/units.json")).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn counts(pairs: &Value) -> Counts {
    let mut c = Counts::default();
    for pair in pairs.as_array().unwrap() {
        c.add(pair[0].as_str().unwrap(), pair[1].as_i64().unwrap());
    }
    c
}

fn pairs(c: &Counts) -> Value {
    json!(c.0.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>())
}

fn date(v: &Value) -> Option<NaiveDate> {
    v.as_str().map(|s| s.parse().unwrap())
}

fn check(kind: &str, case: impl std::fmt::Display, got: Value, want: &Value, wrong: &mut Vec<String>) {
    if &got != want {
        wrong.push(format!("{kind} {case}\n  python {want}\n  rust   {got}"));
    }
}

#[test]
fn indexes_and_submissions_read_as_the_python_crawler() {
    let u = units();
    let mut wrong = Vec::new();
    for case in u["index_ciks"].as_array().unwrap() {
        let forms = if case[1].is_null() { FORMS.iter().map(|f| f.to_string()).collect() } else { strings(&case[1]) };
        let forms: Vec<&str> = forms.iter().map(String::as_str).collect();
        let got = index_ciks(case[0].as_str().unwrap(), &forms);
        check("index_ciks", json!([case[0], case[1]]), json!(got), &case[2], &mut wrong);
    }
    for case in u["filings"].as_array().unwrap() {
        let got = match filings(&case[0], date(&case[1]).unwrap(), &FORMS, date(&case[2])) {
            Ok(found) => json!(found.iter().map(|f| json!([f.cik, f.company, f.form, f.filed, f.accession, f.document, f.url()])).collect::<Vec<_>>()),
            Err(crash) => json!({"error": crash.0}),
        };
        check("filings", json!([case[0], case[1], case[2]]), got, &case[3], &mut wrong);
    }
    assert!(wrong.is_empty(), "{} differences:\n{}", wrong.len(), wrong.join("\n"));
}

#[tokio::test(start_paused = true)]
async fn the_pace_starts_requests_when_python_does() {
    for case in units()["pace"].as_array().unwrap() {
        let pace = Pace::new(case[0].as_f64().unwrap()).unwrap();
        let start = tokio::time::Instant::now();
        let mut starts = Vec::new();
        for idle in case[1].as_array().unwrap() {
            tokio::time::sleep(std::time::Duration::from_secs_f64(idle.as_f64().unwrap())).await;
            pace.turn().await;
            starts.push((tokio::time::Instant::now() - start).as_secs_f64());
        }
        let want: Vec<f64> = case[2].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        // Timers fire on whole milliseconds, so each wait may start up to 1 ms later, never sooner.
        let close = starts.iter().zip(&want).enumerate().all(|(i, (a, b))| *a >= b - 1e-6 && *a <= b + 1e-3 * (i + 1) as f64 + 1e-6);
        assert!(starts.len() == want.len() && close, "{case}: rust {starts:?}");
    }
}

#[test]
fn checks_fail_the_runs_python_fails_with_its_message() {
    let u = units();
    let mut wrong = Vec::new();
    for case in u["check_edgar"].as_array().unwrap() {
        let got = check_edgar(&counts(&case[0])).err().map(|e| e.to_string());
        check("check_edgar", &case[0], json!(got), &case[1], &mut wrong);
    }
    for case in u["check_fred"].as_array().unwrap() {
        let got = check_fred(&counts(&case[0]), case[1].as_u64().unwrap() as usize).err().map(|e| e.to_string());
        check("check_fred", json!([case[0], case[1]]), json!(got), &case[2], &mut wrong);
    }
    assert!(wrong.is_empty(), "{} differences:\n{}", wrong.len(), wrong.join("\n"));
}

#[test]
fn url_lists_group_by_host_as_python_does() {
    let mut wrong = Vec::new();
    for case in units()["groups"].as_array().unwrap() {
        let (urls, skip) = (strings(&case[0]), strings(&case[1]).into_iter().collect::<HashSet<_>>());
        let set = (!case[2].is_null()).then(|| SetRule::parse(case[2].to_string().as_bytes()).unwrap());
        let (first, by_host) = group_urls(&urls, &skip, |url| set.as_ref().is_none_or(|s| s.may_admit(url)));
        let (mut refused, mut opened, mut fetched) = (first.clone(), first, Vec::new());
        for (host, todo) in &by_host {
            refused.add(&format!("robots_{host}"), todo.len() as i64);
            let once: IndexSet<&String> = todo.iter().collect();
            opened.add("status_404", once.len() as i64);
            fetched.extend(once.into_iter().map(|url| json!([host, url])));
        }
        let want = &case[3];
        check("refused", &case[2], pairs(&refused), &want["refused"], &mut wrong);
        check("opened", &case[2], pairs(&opened), &want["opened"], &mut wrong);
        check("fetched", &case[2], json!(fetched), &want["fetched"], &mut wrong);
    }
    assert!(wrong.is_empty(), "{} differences:\n{}", wrong.len(), wrong.join("\n"));
}

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;

use bulk_core::worker::Worker;
use bulk_core::Batches;
use serde_json::Value;

const INNER: &str = "\u{0}INNER\u{0}";

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/wikipedia")
}

fn hidden() -> HashSet<String> {
    let text = std::fs::read_to_string(fixtures().join("input/hidden.txt")).unwrap();
    bulk_core::text::splitlines(&text).into_iter().map(str::to_string).collect()
}

fn without_inner(frame: &str) -> String {
    let marker = frame.find(r#"<div class="mw-content-ltr mw-parser-output""#).unwrap();
    let start = marker + frame[marker..].find('>').unwrap() + 1;
    let end = frame.rfind(r#"</div></div><div id="catlinks">"#).unwrap();
    format!("{}{INNER}{}", &frame[..start], &frame[end..])
}

#[test]
fn frames_match_page_html() {
    let frames: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(fixtures().join("frames.json")).unwrap()).unwrap();
    let hidden = hidden();
    for case in &frames {
        let article = &case["article"];
        let (page, disambiguation) = bulk_wikipedia::frame(article, &hidden).unwrap().expect("a body to cut");
        assert_eq!(without_inner(&page), case["frame"].as_str().unwrap(), "{}", article["name"]);
        assert_eq!(disambiguation, case["disambiguation"].as_bool().unwrap(), "{}", article["name"]);
        let body = article["article_body"]["html"].as_str().unwrap();
        let inner = bulk_wikipedia::html::body_inner(body).unwrap();
        let at = inner.as_ptr() as usize - body.as_ptr() as usize;
        assert!(page.contains(&*bulk_wikipedia::html::as_lxml_writes(inner, &bulk_wikipedia::html::start_tags(body), at)));
    }
}

#[test]
fn hidden_categories_match_the_golden_run() {
    let mut out = Vec::new();
    bulk_wikipedia::hidden_categories(&[fixtures().join("input/ns14.ndjson")], &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), std::fs::read_to_string(fixtures().join("hidden-stdout.txt")).unwrap());
}

#[test]
fn run_matches_the_golden_run() {
    let Some(python) = std::env::var("BULK_PYTHON").ok().filter(|p| !p.is_empty()) else {
        eprintln!("skipped: set BULK_PYTHON to a Python with the shared extraction to run the page worker");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let expected = fixtures().join("expected");
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rust");
    let worker_args: Vec<OsString> = vec!["--hidden".into(), fixtures().join("input/hidden.txt").into_os_string()];
    let spawn = || Worker::spawn(&python, &root, &worker_args);
    let as_of = bulk_core::rows::parse_as_of("2026-11-01T00:00:00Z").unwrap();
    let mut batches = Batches::new(&out, bulk_wikipedia::SOURCE, "fixture", as_of).unwrap();
    let counts = bulk_wikipedia::run(&[fixtures().join("input/articles.ndjson")], &mut batches, as_of, &hidden(), 0, 1, 3, &spawn).unwrap();
    let summary = batches.close(&counts).unwrap();
    let line = bulk_core::run::summary_line(&expected, &summary, &counts);
    let expected_line = std::fs::read_to_string(fixtures().join("stdout.txt")).unwrap();
    assert_eq!(line.replace(&expected.to_string_lossy().into_owned(), "bulk/fixtures/wikipedia/expected"), expected_line.trim_end());
    let diff = bulk_core::runs::diff(&out, &expected, 20).unwrap();
    assert!(diff.manifest_differences.is_empty() && diff.rows_differing == 0 && diff.rows_a == diff.rows_b, "{diff:#?}");
}

#[test]
fn limit_stops_at_the_golden_page() {
    let Some(python) = std::env::var("BULK_PYTHON").ok().filter(|p| !p.is_empty()) else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = tempfile::tempdir().unwrap();
    let worker_args: Vec<OsString> = vec!["--hidden".into(), fixtures().join("input/hidden.txt").into_os_string()];
    let spawn = || Worker::spawn(&python, &root, &worker_args);
    let as_of = bulk_core::rows::parse_as_of("2026-11-01T00:00:00Z").unwrap();
    // The golden counts at --limit 33: with --workers 4 one window holds every job, with 1 page 33 is in the second.
    for (workers, want) in [
        (4, [("other_namespace", 1), ("deleted", 1), ("older_copy", 1), ("no_html", 1), ("pages", 33), ("disambiguation", 1)]),
        (1, [("other_namespace", 1), ("pages", 33), ("disambiguation", 1), ("deleted", 1), ("older_copy", 1), ("no_html", 1)]),
    ] {
        let mut batches = Batches::new(&dir.path().join(format!("rust-{workers}")), bulk_wikipedia::SOURCE, "fixture", as_of).unwrap();
        let counts = bulk_wikipedia::run(&[fixtures().join("input/articles.ndjson")], &mut batches, as_of, &hidden(), 33, workers, 2, &spawn).unwrap();
        assert_eq!(counts.iter().collect::<Vec<_>>(), want, "--workers {workers}");
        assert_eq!(batches.rows(), 33);
    }
}

use std::path::{Path, PathBuf};

use anyhow::Result;
use bulk_core::worker::{Worker, PAGE_WORKER};
use bulk_core::{Batches, Counts};
use serde_json::Value;

const AS_OF: &str = "2026-11-11T08:45:08Z";

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/bittensor")
}

fn python() -> Option<String> {
    let found = std::env::var("BULK_PYTHON").ok().filter(|p| !p.is_empty());
    if found.is_none() {
        eprintln!("skipped: set BULK_PYTHON to a Python with the shared extraction to run the page worker");
    }
    found
}

fn check_run(source: &str, line: usize, run: impl FnOnce(&mut Batches, i64) -> Result<Counts>) {
    let expected = fixtures().join("python").join(source);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rust");
    let as_of = bulk_core::rows::parse_as_of(AS_OF).unwrap();
    let mut batches = Batches::new(&out, source, "fixture", as_of).unwrap();
    let counts = run(&mut batches, as_of).unwrap();
    let summary = batches.close(&counts).unwrap();
    let stdout = std::fs::read_to_string(fixtures().join("stdout.txt")).unwrap();
    let python_line = stdout.lines().nth(line).unwrap();
    let line = bulk_core::run::summary_line(&expected, &summary, &counts);
    assert_eq!(line.replace(&expected.to_string_lossy().into_owned(), &format!("bulk-rs/fixtures/bittensor/python/{source}")), python_line);
    let diff = bulk_core::runs::diff(&out, &expected, 20).unwrap();
    assert!(diff.same(), "{diff:#?}");
    assert_eq!(std::fs::read(out.join("removed.txt")).unwrap(), std::fs::read(expected.join("removed.txt")).unwrap());
}

fn spawn(python: &str) -> impl Fn() -> Result<Worker> + Sync + '_ {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    move || Worker::spawn(python, &root, PAGE_WORKER, &[])
}

#[test]
fn urls_clean_as_python() {
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("urls.json")).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let (url, cleaned, app) = (case[0].as_str().unwrap(), case[1].as_str().unwrap(), case[2].as_bool().unwrap());
        assert_eq!(bulk_bittensor::without_tracking(url).unwrap(), cleaned, "{url}");
        assert_eq!(bulk_bittensor::app_page(cleaned).unwrap(), app, "{url}");
    }
}

#[test]
fn subnets_match_the_python_run() {
    let chain = fixtures().join("input/subnets-chain.json");
    check_run(bulk_bittensor::SUBNETS_SOURCE, 0, |batches, as_of| bulk_bittensor::run_subnets(&chain, batches, as_of));
}

#[test]
fn sites_match_the_python_run() {
    let Some(python) = python() else { return };
    let folder = fixtures().join("input/sites");
    check_run(bulk_bittensor::SITES_SOURCE, 1, |batches, as_of| bulk_bittensor::run_sites(&folder, batches, as_of, 3, &spawn(&python)));
}

#[test]
fn github_matches_the_python_run() {
    let Some(python) = python() else { return };
    let folder = fixtures().join("input/github");
    check_run(bulk_bittensor::GITHUB_SOURCE, 2, |batches, as_of| bulk_bittensor::run_github(&folder, batches, as_of, 2, &spawn(&python)));
}

#[test]
fn no_pages_need_no_worker() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("sites.json"), "[]").unwrap();
    std::fs::write(dir.path().join("github.json"), r#"[{"netuid": 1, "error": "not a repo url"}]"#).unwrap();
    let as_of = bulk_core::rows::parse_as_of(AS_OF).unwrap();
    let unspawnable = || -> Result<Worker> { anyhow::bail!("no worker should start") };
    let mut batches = Batches::new(&dir.path().join("sites"), bulk_bittensor::SITES_SOURCE, "t", as_of).unwrap();
    assert!(bulk_bittensor::run_sites(dir.path(), &mut batches, as_of, 4, &unspawnable).unwrap().is_empty());
    let mut batches = Batches::new(&dir.path().join("github"), bulk_bittensor::GITHUB_SOURCE, "t", as_of).unwrap();
    let counts = bulk_bittensor::run_github(dir.path(), &mut batches, as_of, 4, &unspawnable).unwrap();
    assert_eq!(counts.iter().collect::<Vec<_>>(), [("no_repo", 1)]);
}

use std::sync::Mutex;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray};
use parquet::arrow::ArrowWriter;
use serde_json::json;

use crate::diff::{self, Refused};
use crate::run::{self, Hooks};
use crate::sources::{Build, Source};

const DAY1: &str = "2026-11-09T02:00:00Z";
const DAY2: &str = "2026-11-16T02:00:00Z";
const DAY2_SECS: u64 = 1_794_794_400;
const FULL: Build = Build { name: "subnets", argv: &[], full_list: true, volatile: false, volatile_lines: &[], volatile_columns: &[] };
const PARTIAL: Build = Build { name: "subnets", argv: &[], full_list: false, volatile: false, volatile_lines: &[], volatile_columns: &[] };

fn run_folder(path: &Path, pages: &[(String, String)], fetched_us: i64, removed: &[&str]) -> std::path::PathBuf {
    std::fs::create_dir_all(path).unwrap();
    let urls: ArrayRef = Arc::new(StringArray::from(pages.iter().map(|p| p.0.clone()).collect::<Vec<_>>()));
    let texts: ArrayRef = Arc::new(LargeStringArray::from(pages.iter().map(|p| p.1.clone()).collect::<Vec<_>>()));
    let times: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![fetched_us; pages.len()]).with_timezone("UTC"));
    let batch = RecordBatch::try_from_iter([("url", urls), ("text", texts), ("fetched_at", times)]).unwrap();
    let mut writer = ArrowWriter::try_new(std::fs::File::create(path.join("subnets-000001.parquet")).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    if !removed.is_empty() {
        std::fs::write(path.join(diff::REMOVED), removed.iter().map(|u| format!("{u}\n")).collect::<String>()).unwrap();
    }
    let manifest = json!({"source": "subnets", "run": "r", "as_of": DAY1, "files": ["subnets-000001.parquet"], "rows": pages.len()});
    std::fs::write(path.join(diff::MANIFEST), manifest.to_string()).unwrap();
    path.to_path_buf()
}

fn urls(n: usize) -> Vec<(String, String)> {
    (0..n).map(|i| (format!("https://taostats.io/subnets/{i}"), format!("subnet {i}"))).collect()
}

fn with(mut pages: Vec<(String, String)>, url: &str, text: &str) -> Vec<(String, String)> {
    match pages.iter_mut().find(|p| p.0 == url) {
        Some(page) => page.1 = text.to_string(),
        None => pages.push((url.to_string(), text.to_string())),
    }
    pages
}

#[test]
fn a_seeded_source_publishes_only_new_and_changed_pages() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    assert_eq!(diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &urls(40), 1, &[])]).unwrap(), 40);

    let second = with(with(urls(40), "https://taostats.io/subnets/3", "subnet 3, new owner"), "https://taostats.io/subnets/40", "new");
    let out = tmp.path().join("out");
    let counts = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r2"), &second, 2, &[]), &out, "r2", DAY2).unwrap();
    assert_eq!((counts.seen, counts.new, counts.changed, counts.unchanged, counts.removed), (41, 1, 1, 39, 0));
    let mut sent = diff::read_urls(&out).unwrap();
    sent.sort();
    assert_eq!(sent, ["https://taostats.io/subnets/3", "https://taostats.io/subnets/40"]);
    assert!(!counts.large);
}

#[test]
fn the_fetch_time_alone_is_no_change() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &urls(5), 1, &[])]).unwrap();
    let counts = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r2"), &urls(5), 999, &[]), &tmp.path().join("o"), "r2", DAY2).unwrap();
    assert_eq!(counts.unchanged, 5);
}

#[test]
fn pages_gone_from_a_full_list_are_hidden_and_come_back() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &urls(40), 1, &[])]).unwrap();

    let less: Vec<_> = urls(40).into_iter().filter(|p| p.0 != "https://taostats.io/subnets/7").collect();
    let counts = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r2"), &less, 2, &[]), &tmp.path().join("o2"), "r2", DAY2).unwrap();
    diff::keep(&mut con, &FULL, &counts, "r2", DAY2).unwrap();
    assert_eq!(counts.gone, ["https://taostats.io/subnets/7"]);
    assert!(diff::read_urls(&tmp.path().join("o2")).unwrap().is_empty());
    assert_eq!(diff::hidden_urls(&con).unwrap(), ["https://taostats.io/subnets/7"]);

    let counts = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r3"), &urls(40), 3, &[]), &tmp.path().join("o3"), "r3", DAY2).unwrap();
    diff::keep(&mut con, &FULL, &counts, "r3", DAY2).unwrap();
    assert_eq!(diff::read_urls(&tmp.path().join("o3")).unwrap(), ["https://taostats.io/subnets/7"]);
    assert!(diff::hidden_urls(&con).unwrap().is_empty());
}

#[test]
fn a_short_fetch_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &urls(40), 1, &[])]).unwrap();
    let err = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r2"), &urls(37), 2, &[]), &tmp.path().join("o"), "r2", DAY2).unwrap_err();
    assert!(err.downcast_ref::<Refused>().is_some());
}

#[test]
fn a_large_change_is_flagged_and_removed_txt_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &urls(10), 1, &[])]).unwrap();
    let edited: Vec<_> = urls(10).into_iter().map(|(u, t)| (u, t + " edited")).collect();
    let folder = run_folder(&tmp.path().join("r2"), &edited, 2, &["https://taostats.io/subnets/99"]);
    let counts = diff::diff(&con, &PARTIAL, &folder, &tmp.path().join("o"), "r2", DAY2).unwrap();
    assert!(counts.large && counts.changed == 10);
    assert_eq!(counts.gone, ["https://taostats.io/subnets/99"]);
}

fn copy_source(name: &'static str, prepared: &Path, fetch: &'static [&'static [&'static str]]) -> Source {
    let argv: &'static [&'static str] =
        Box::leak(vec!["cp", "-R", Box::leak(prepared.display().to_string().into_boxed_str()) as &str, "{out}"].into_boxed_slice());
    let builds: &'static [Build] = Box::leak(vec![Build { name: "subnets", argv, full_list: true, volatile: false, volatile_lines: &[], volatile_columns: &[] }].into_boxed_slice());
    Source { name, fetch, builds, cwd: "/", python: "/usr/bin/true", memory_bytes: 0, timeout_s: 60 }
}

#[test]
fn refresh_runs_the_commands_publishes_and_hides() {
    let tmp = tempfile::tempdir().unwrap();
    let prepared = run_folder(&tmp.path().join("prepared"), &urls(3), 1, &[]);
    let source = copy_source("test", &prepared, &[]);
    let sent = Mutex::new(Vec::new());
    let hidden = Mutex::new(Vec::new());
    let publish = |folder: &Path, _: &Path| {
        sent.lock().unwrap().push(diff::read_urls(folder).unwrap());
        Ok("rc=0".to_string())
    };
    let hide = |name: &str, urls: &[String]| {
        hidden.lock().unwrap().push((name.to_string(), urls.to_vec()));
        Ok(())
    };
    let hooks = Hooks { publish: &publish, hide: &hide };
    let home = tmp.path().join("home");

    let report = run::refresh(&source, &home, false, DAY2_SECS, "", None, &hooks).unwrap();
    assert_eq!(report["builds"]["subnets"]["new"], 3);
    assert_eq!(sent.lock().unwrap().len(), 1);
    assert_eq!(*hidden.lock().unwrap(), [("test".to_string(), Vec::<String>::new())]);

    let again = run::refresh(&source, &home, false, DAY2_SECS + 7 * 86_400, "", None, &hooks).unwrap();
    assert_eq!(again["builds"]["subnets"]["unchanged"], 3);
    assert_eq!(sent.lock().unwrap().len(), 1);
}

#[test]
fn a_dry_run_publishes_and_records_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let prepared = run_folder(&tmp.path().join("prepared"), &urls(3), 1, &[]);
    let source = copy_source("test", &prepared, &[]);
    let publish = |_: &Path, _: &Path| -> anyhow::Result<String> { panic!("published on a dry run") };
    let hide = |_: &str, _: &[String]| -> anyhow::Result<()> { panic!("hid on a dry run") };
    let report = run::refresh(&source, &tmp.path().join("home"), true, DAY2_SECS, "", None, &Hooks { publish: &publish, hide: &hide }).unwrap();
    assert_eq!(report["builds"]["subnets"]["new"], 3);
    let con = diff::open_state(&tmp.path().join("home/state/test.sqlite")).unwrap();
    assert_eq!(con.query_row("SELECT count(*) FROM pages", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
}

#[test]
fn an_incremental_source_fetches_from_the_last_published_run() {
    let tmp = tempfile::tempdir().unwrap();
    let prepared = run_folder(&tmp.path().join("prepared"), &urls(2), 1, &[]);
    let calls = tmp.path().join("calls");
    let record: &'static [&'static str] = Box::leak(
        vec!["sh", "-c", Box::leak(format!("echo $0 $1 >> {}", calls.display()).into_boxed_str()) as &str, "{since}", "{until}"]
            .into_boxed_slice(),
    );
    let fetch: &'static [&'static [&'static str]] = Box::leak(vec![record].into_boxed_slice());
    let source = copy_source("inc", &prepared, fetch);
    let publish = |_: &Path, _: &Path| Ok("rc=0".to_string());
    let hide = |_: &str, _: &[String]| Ok(());
    let hooks = Hooks { publish: &publish, hide: &hide };
    let home = tmp.path().join("home");

    assert!(run::refresh(&source, &home, false, DAY2_SECS, "", None, &hooks).unwrap_err().downcast_ref::<Refused>().is_some());
    let mut report = run::refresh(&source, &home, false, DAY2_SECS, "2026-11-10", None, &hooks).unwrap();
    report["ok"] = json!(true);
    std::fs::write(home.join(run::STATUS), format!("{report}\n")).unwrap();
    run::refresh(&source, &home, false, DAY2_SECS + 86_400, "", None, &hooks).unwrap();
    assert_eq!(std::fs::read_to_string(calls).unwrap(), "2026-11-10 2026-11-16\n2026-11-16 2026-11-17\n");
}

#[test]
fn wildcard_paths_expand_to_the_files_they_match() {
    let tmp = tempfile::tempdir().unwrap();
    for name in ["b.xml", "a.xml", "c.txt"] {
        std::fs::write(tmp.path().join(name), "").unwrap();
    }
    let values = [("work", tmp.path().display().to_string())].into_iter().collect();
    let found = run::expand("{work}/*.xml", &values);
    assert_eq!(found, [tmp.path().join("a.xml").display().to_string(), tmp.path().join("b.xml").display().to_string()]);
}

#[test]
fn publish_hands_the_run_to_the_outbox_and_reads_its_receipt() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let folder = run_folder(&root.join("work/refresh-subnets-20261116-0200"), &urls(2), 1, &[]);
    let planner = {
        let root = root.clone();
        std::thread::spawn(move || {
            let ready = root.join("outbox/refresh-subnets-20261116-0200.ready");
            while !ready.exists() {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            std::fs::create_dir_all(root.join("receipts")).unwrap();
            std::fs::write(root.join("receipts/refresh-subnets-20261116-0200.txt"), "rc=0\n1 files, 2 pages, bulk files [4700]\n0\n").unwrap();
        })
    };
    let line = run::publish(&folder, &root, std::time::Duration::from_millis(10)).unwrap();
    planner.join().unwrap();
    assert_eq!(line, "1 files, 2 pages, bulk files [4700]");
    assert!(root.join("outbox/refresh-subnets-20261116-0200/manifest.json").exists() && !folder.exists());

    let failed = run_folder(&root.join("work/refresh-subnets-x"), &urls(1), 1, &[]);
    std::fs::write(root.join("receipts/refresh-subnets-x.txt"), "rc=1\nerror: no room\n").unwrap();
    let err = run::publish(&failed, &root, std::time::Duration::from_millis(10)).unwrap_err();
    assert!(format!("{err}").contains("no room"));
}

#[test]
fn the_discord_line_says_what_was_published_or_why_not() {
    let ok = json!({
        "source": "bittensor", "run": "20261116-0200", "ok": true, "hidden": 1, "seconds": 812,
        "builds": {"subnets": {"new": 2, "changed": 5, "removed": 1, "seen": 130, "published": "1 files, 7 pages, bulk files [4701]", "large": false}},
    });
    assert_eq!(
        run::summary(&ok),
        "refresh bittensor 20261116-0200: subnets 2 new, 5 changed, 1 removed of 130, 1 files, 7 pages, bulk files [4701]; 1 hidden; 812 s"
    );
    let bad = json!({"source": "arxiv", "ok": false, "refused": true, "error": "short", "seconds": 3});
    assert_eq!(run::summary(&bad), "⚠️ refresh arxiv refused after 3 s: short");
}

#[test]
fn iso_times_are_utc() {
    assert_eq!(run::iso(DAY2_SECS), DAY2);
    assert_eq!(run::iso(0), "1970-01-01T00:00:00Z");
}


#[test]
fn a_site_missing_whole_is_held_until_named_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let mut con = diff::open_state(&tmp.path().join("s.sqlite")).unwrap();
    let mut pages = urls(40);
    pages.push(("https://down.example/a".into(), "a".into()));
    pages.push(("https://down.example/b".into(), "b".into()));
    diff::seed(&mut con, &FULL, &[run_folder(&tmp.path().join("r1"), &pages, 1, &[])]).unwrap();
    let counts = diff::diff(&con, &FULL, &run_folder(&tmp.path().join("r2"), &urls(40), 2, &[]), &tmp.path().join("o2"), "r2", DAY2).unwrap();
    assert_eq!((counts.removed, counts.held), (0, 2));
    let named = run_folder(&tmp.path().join("r3"), &urls(40), 3, &["https://down.example/a"]);
    let counts = diff::diff(&con, &FULL, &named, &tmp.path().join("o3"), "r3", DAY2).unwrap();
    assert_eq!((counts.gone.clone(), counts.held), (vec!["https://down.example/a".to_string()], 1));
}


#[test]
fn a_finished_build_can_be_published_without_fetching_again() {
    let tmp = tempfile::tempdir().unwrap();
    let prepared = run_folder(&tmp.path().join("prepared"), &urls(3), 1, &[]);
    let source = copy_source("test", &prepared, &[]);
    let publish = |_: &Path, _: &Path| -> anyhow::Result<String> { panic!("published on a dry run") };
    let hide = |_: &str, _: &[String]| -> anyhow::Result<()> { panic!("hid on a dry run") };
    let home = tmp.path().join("home");
    let report = run::refresh(&source, &home, true, DAY2_SECS, "", None, &Hooks { publish: &publish, hide: &hide }).unwrap();
    let work = home.join("work/test").join(report["run"].as_str().unwrap());
    std::fs::remove_dir_all(&prepared).unwrap();
    let sent = Mutex::new(0);
    let publish = |_: &Path, _: &Path| {
        *sent.lock().unwrap() += 1;
        Ok("rc=0".to_string())
    };
    let hide = |_: &str, _: &[String]| Ok(());
    let again = run::refresh(&source, &home, false, DAY2_SECS + 60, "", Some(&work), &Hooks { publish: &publish, hide: &hide }).unwrap();
    assert_eq!((again["builds"]["subnets"]["new"].as_u64(), *sent.lock().unwrap()), (Some(3), 1));
}


#[test]
fn a_home_path_is_expanded() {
    let home = std::env::var("HOME").unwrap();
    assert_eq!(run::expand("~/refresh/bin/bulk", &Default::default()), [format!("{home}/refresh/bin/bulk")]);
}

#[test]
fn a_volatile_line_alone_is_no_change() {
    const DATED: Build = Build { name: "subnets", argv: &[], full_list: false, volatile: false, volatile_lines: &["Source: dated"], volatile_columns: &[] };
    let dir = tempfile::tempdir().unwrap();
    let page = |day: &str, body: &str| vec![("https://clinicaltrials.gov/study/NCT1".to_string(), format!("Title\nSource: dated, processed on {day}.\n{body}"))];
    let first = run_folder(&dir.path().join("a"), &page("2026-11-12", "Phase 2"), 1, &[]);
    let mut con = diff::open_state(&dir.path().join("state.sqlite")).unwrap();
    diff::seed(&mut con, &DATED, &[first]).unwrap();
    let same = run_folder(&dir.path().join("b"), &page("2026-11-19", "Phase 2"), 2, &[]);
    let counts = diff::diff(&con, &DATED, &same, &dir.path().join("out-b"), "b", DAY2).unwrap();
    assert_eq!((counts.unchanged, counts.changed), (1, 0));
    let changed = run_folder(&dir.path().join("c"), &page("2026-11-19", "Phase 3"), 2, &[]);
    let counts = diff::diff(&con, &DATED, &changed, &dir.path().join("out-c"), "c", DAY2).unwrap();
    assert_eq!((counts.unchanged, counts.changed), (0, 1));
}

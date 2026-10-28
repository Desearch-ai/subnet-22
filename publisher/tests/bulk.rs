//! Bulk runs: published into their own feed, folded with the change log, replaced by a later crawl or a corrected run, and never anyone's work.

#[allow(dead_code)]
mod common;

use common::{page_v2, parquet_v2};
use desearch::feeds::{BULK, CHANGES};
use publisher::bulk::{self, Settings, MANIFEST, PUBLISHED};
use publisher::changes::Change;
use publisher::compactor::{self, feed_key, LocalStore, Store};
use publisher::current::{decode_changes, ChangeRow, BULK_ORDER};
use publisher::local::{LocalLog, LocalUploads};
use publisher::records::{page_key, SECOND};
use publisher::worker::{publish, Job};
use serde_json::json;

const AS_OF: i64 = 1_793_059_200 * SECOND;
const NOW: i64 = 1_793_100_000 * SECOND;
const WIKI: &str = "https://en.wikipedia.org/wiki/Squirrel";
const OTHER: &str = "https://en.wikipedia.org/wiki/Tbilisi";

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-bulk-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn settings() -> Settings {
    Settings { max_rows: 2, max_text_bytes: 32 << 20, pages_per_hour: 0.0 }
}

/// A run folder as `python -m bulk` leaves it, its pages fetched at `as_of`.
fn run(dir: &std::path::Path, as_of: i64, files: &[(&str, &[(&str, &str)])]) {
    std::fs::create_dir_all(dir).unwrap();
    for (name, pages) in files {
        let rows: Vec<_> = pages.iter().map(|(url, text)| page_v2(url, text, as_of)).collect();
        std::fs::write(dir.join(name), parquet_v2(&rows, None)).unwrap();
    }
    let manifest = json!({
        "source": "wikipedia", "run": "20261101", "as_of": "2026-11-01T00:00:00Z",
        "files": files.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join(MANIFEST), manifest.to_string()).unwrap();
}

/// A miner's crawl of `url`, published into the change log.
fn crawl(root: &std::path::Path, log: &LocalLog, task_id: &str, url: &str, text: &str, fetched_at: i64) {
    let uploads = root.join("uploads");
    std::fs::create_dir_all(&uploads).unwrap();
    std::fs::write(uploads.join(format!("{task_id}.parquet")), parquet_v2(&[page_v2(url, text, fetched_at)], None)).unwrap();
    let job: Job = serde_json::from_value(json!({
        "task_id": task_id, "kind": "crawl", "miner": "5Miner", "urls": [url],
        "completed_at": (fetched_at / SECOND + 10) as f64, "claim_ttl": 180, "validator": "5Validator",
    }))
    .unwrap();
    publish(&[job], &LocalUploads::new(&uploads), &Default::default(), log, fetched_at + 20 * SECOND, 1).unwrap();
}

fn text_of(store: &dyn Store, url: &str) -> Option<String> {
    compactor::lookup(store, "current/", &page_key(url).unwrap()).unwrap().map(|row| row.record.text)
}

fn compact(store: &LocalStore, work: &std::path::Path) -> compactor::Report {
    let settings = compactor::Settings { work: work.into(), threads: 2, ..compactor::Settings::default() };
    compactor::cycle(store, &settings, (NOW / SECOND) as f64).unwrap()
}

#[test]
fn a_bulk_run_goes_to_its_own_feed_one_id_a_row_and_names_no_miner() {
    let root = scratch("rows");
    run(&root.join("run"), AS_OF, &[("wikipedia-000001.parquet", &[(WIKI, "snapshot text"), (OTHER, "tbilisi"), ("https://en.wikipedia.org/wiki/C", "c")])]);
    let feed = LocalLog::open(root.join("bucket"), &BULK).unwrap();
    let report = bulk::publish_run(&root.join("run"), &feed, &settings(), &|| NOW).unwrap();
    assert_eq!((report.files, report.pages, report.change_files.len()), (1, 3, 2), "two pages a bulk file at most");
    let store = LocalStore { root: root.join("bucket") };
    assert!(store.get(&CHANGES.latest_key()).unwrap().is_none(), "nothing went into the change log");
    let rows = decode_changes(store.get(&feed_key(&store, &BULK, 1).unwrap().unwrap()).unwrap().unwrap(), BULK_ORDER + 1).unwrap();
    let ids: Vec<&str> = rows.iter().map(|row| if let ChangeRow::Page(page) = row { page.record.task_id.as_str() } else { "" }).collect();
    assert_eq!(ids, ["bulk-wikipedia-20261101-000001-0", "bulk-wikipedia-20261101-000001-1"]);
    let ChangeRow::Page(page) = &rows[0] else { panic!("a page row") };
    assert_eq!((page.record.miner.as_str(), page.record.validator.as_deref(), page.record.validators.len()), ("", None, 0));
    assert_eq!(page.record.fetched_at, "2026-10-27T00:00:00+00:00", "the snapshot's time in the publisher's own format");
    assert_eq!(page.record.extraction.as_deref(), Some("v2"));

    let again = bulk::publish_run(&root.join("run"), &feed, &settings(), &|| NOW).unwrap();
    assert_eq!(again.files, 0, "a published file is not published twice");
    assert!(std::fs::read_to_string(root.join("run").join(PUBLISHED)).unwrap().starts_with("wikipedia-000001.parquet 1 2"));
}

#[test]
fn the_compactor_folds_both_feeds_and_a_later_crawl_replaces_the_snapshot() {
    let root = scratch("replace");
    let (log, feed) = (LocalLog::open(root.join("bucket"), &CHANGES).unwrap(), LocalLog::open(root.join("bucket"), &BULK).unwrap());
    crawl(&root, &log, "0a1b2c3d4e5f6071", "https://news.example/a", "news", NOW);
    run(&root.join("run"), AS_OF, &[("wikipedia-000001.parquet", &[(WIKI, "snapshot text"), (OTHER, "tbilisi")])]);
    bulk::publish_run(&root.join("run"), &feed, &settings(), &|| NOW).unwrap();
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 2).unwrap();
    let report = compact(&store, &root.join("work"));
    assert_eq!((report.through, report.rows, report.bulk_through, report.bulk_rows), (1, 1, 1, 2));
    assert_eq!(text_of(&store, WIKI).as_deref(), Some("snapshot text"));

    crawl(&root, &log, "7be798edabde4b06", WIKI, "crawled text", NOW + 60 * SECOND);
    let report = compact(&store, &root.join("work"));
    assert_eq!((report.rows, report.bulk_rows), (1, 0), "each feed is read once");
    assert_eq!(text_of(&store, WIKI).as_deref(), Some("crawled text"), "the crawl is newer than the snapshot");
    assert_eq!(text_of(&store, "https://news.example/a").as_deref(), Some("news"));

    run(&root.join("fixed"), AS_OF + SECOND, &[("wikipedia-000001.parquet", &[(OTHER, "tbilisi, fixed")])]);
    bulk::publish_run(&root.join("fixed"), &feed, &settings(), &|| NOW).unwrap();
    compact(&store, &root.join("work"));
    assert_eq!(text_of(&store, OTHER).as_deref(), Some("tbilisi, fixed"), "a corrected run a second later replaces the first");
    let compared = compactor::compare_log(&store, "current/", &(0..1024), 2).unwrap();
    assert_eq!((compared.missing.len(), compared.extra.len(), compared.other_version.len()), (0, 0, 0), "the state is a plain fold of both feeds");
}

#[test]
fn a_crawl_older_than_the_snapshot_gives_way_and_only_the_change_log_removes() {
    let root = scratch("older");
    let (log, feed) = (LocalLog::open(root.join("bucket"), &CHANGES).unwrap(), LocalLog::open(root.join("bucket"), &BULK).unwrap());
    crawl(&root, &log, "0a1b2c3d4e5f6071", WIKI, "older crawl", AS_OF - 3600 * SECOND);
    run(&root.join("run"), AS_OF, &[("wikipedia-000001.parquet", &[(WIKI, "snapshot text"), (OTHER, "tbilisi")])]);
    bulk::publish_run(&root.join("run"), &feed, &settings(), &|| NOW).unwrap();
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 2).unwrap();
    compact(&store, &root.join("work"));
    assert_eq!(text_of(&store, WIKI).as_deref(), Some("snapshot text"));

    let removed = Change {
        key: page_key(OTHER).unwrap(),
        kind: publisher::changes::Kind::Removed,
        previous_content_sha1: String::new(),
        published_at: "2026-10-28T00:00:00+00:00".into(),
        body: publisher::changes::Body::Removed { url: OTHER.into(), domain: "en.wikipedia.org".into() },
    };
    publisher::worker::ChangeFeed::append(&log, publisher::changes::encode(&[removed.clone()]).unwrap(), 1).unwrap();
    publisher::worker::ChangeFeed::append(&feed, publisher::changes::encode(&[Change { key: page_key(WIKI).unwrap(), ..removed }]).unwrap(), 1).unwrap();
    compact(&store, &root.join("work"));
    assert_eq!(text_of(&store, OTHER), None, "a removal in the change log takes the bulk version down");
    assert_eq!(text_of(&store, WIKI).as_deref(), Some("snapshot text"), "the bulk feed carries pages only; a removal there is ignored");
    let compared = compactor::compare_log(&store, "current/", &(0..1024), 2).unwrap();
    assert_eq!((compared.missing.len(), compared.extra.len(), compared.other_version.len()), (0, 0, 0));
}

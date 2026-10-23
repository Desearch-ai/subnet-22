//! The current state built from the change log: shards, newest wins, removals, withdrawals, merges and cleanup.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use bytes::Bytes;
use desearch::feeds::CHANGES;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use publisher::changes::{self, Body, Change, Kind};
use publisher::compactor::{self, LocalStore, Settings, Split, Store};
use publisher::current::{self, shard_of, ChangeRow, PageRow, Withdrawals, SHARDS};
use publisher::records::{page_key, record_version, Record, ROW_COLUMNS_V2};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-current-{name}-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn record(url: &str, fetched_at: &str, task_id: &str, text: &str) -> Record {
    Record {
        url: url.into(),
        domain: "news.example".into(),
        title: format!("title of {url}"),
        text: text.into(),
        fetched_at: fetched_at.into(),
        content_sha1: format!("sha1 of {text}"),
        assigned_url: url.into(),
        task_id: task_id.into(),
        miner: "5Miner".into(),
        validators: vec!["5Validator".into()],
        json_ld_types: vec![Some("NewsArticle".into())],
        status: Some(200),
        ..Record::default()
    }
}

fn v2_record(url: &str, fetched_at: &str, task_id: &str, text: &str) -> Record {
    Record {
        extraction: Some("v2".into()),
        text_mode: Some("main".into()),
        modified: Some("2026-10-07".into()),
        site_name: Some("News".into()),
        image: Some(format!("{url}.png")),
        section: Some("World".into()),
        tags: vec![Some("a".into()), None],
        breadcrumbs: vec![Some("Home".into())],
        entity: Some("{\"name\": \"News\"}".into()),
        ld: Some("{\"faq\": []}".into()),
        paywalled: Some(true),
        robots: vec![Some("noarchive".into())],
        ..record(url, fetched_at, task_id, text)
    }
}

fn page(url: &str, fetched_at: &str, task_id: &str, text: &str) -> Change {
    change_of(record(url, fetched_at, task_id, text))
}

fn v2_page(url: &str, fetched_at: &str, task_id: &str, text: &str) -> Change {
    change_of(v2_record(url, fetched_at, task_id, text))
}

fn change_of(record: Record) -> Change {
    Change {
        key: page_key(&record.url).unwrap(),
        kind: Kind::New,
        previous_content_sha1: String::new(),
        published_at: "2026-10-08T12:00:00+00:00".into(),
        body: Body::Page { version: record_version(&record), record: Box::new(record) },
    }
}

fn removed(url: &str) -> Change {
    Change {
        key: page_key(url).unwrap(),
        kind: Kind::Removed,
        previous_content_sha1: String::new(),
        published_at: "2026-10-08T12:00:00+00:00".into(),
        body: Body::Removed { url: url.into(), domain: "news.example".into() },
    }
}

/// A change log in a folder, numbered as the publisher numbers it.
struct Log {
    store: LocalStore,
    next: i64,
}

impl Log {
    fn new(root: PathBuf) -> Log {
        Log { store: LocalStore { root }, next: 1 }
    }

    fn append(&mut self, changes: &[Change]) -> i64 {
        let seq = self.next;
        self.next += 1;
        let key = format!("changes/dt=2026-10-08/{seq}.parquet");
        self.store.put(&key, Bytes::from(changes::encode(changes).unwrap())).unwrap();
        self.store.put(&CHANGES.seq_key(seq as u64), Bytes::from(format!("{{\"key\": \"{key}\", \"rows\": {}}}", changes.len()))).unwrap();
        self.store.put(&CHANGES.latest_key(), Bytes::from(format!("{{\"seq\": {seq}}}"))).unwrap();
        seq
    }
}

fn settings(work: PathBuf) -> Settings {
    Settings { work, threads: 4, spill_bytes: 4096, merge_ratio: 100.0, fanout: 100, retire_after_s: 3600.0, ..Settings::default() }
}

fn text_of(store: &dyn Store, url: &str) -> Option<String> {
    compactor::lookup(store, "current/", &page_key(url).unwrap()).unwrap().map(|row| row.record.text)
}

#[test]
fn a_key_s_shard_is_the_first_ten_bits_of_its_url_hash() {
    assert_eq!(shard_of("pages/news.example/000fffffffffffffffffffffffffffffffffffff"), Some(0));
    assert_eq!(shard_of("pages/news.example/004fffffffffffffffffffffffffffffffffffff"), Some(1));
    assert_eq!(shard_of("pages/news.example/fffaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), Some(SHARDS - 1));
    assert_eq!(shard_of("pages/news.example/zz"), None);
    let spread: HashSet<usize> = (0..20_000).filter_map(|i| shard_of(&page_key(&format!("https://news.example/{i}")).unwrap())).collect();
    assert!(spread.len() > 1000, "keys spread over the shards: {}", spread.len());
}

#[test]
fn a_change_file_reads_back_row_for_row_and_a_pages_file_round_trips() {
    let changes = vec![
        page("https://news.example/a", "2026-10-08T10:00:00+00:00", "t1", "first"),
        removed("https://news.example/b"),
        Change::task_withdrawn("t9", "2026-10-08T12:00:00+00:00"),
    ];
    let rows = current::decode_changes(Bytes::from(changes::encode(&changes).unwrap()), 7).unwrap();
    let ChangeRow::Page(first) = &rows[0] else { panic!("a page first") };
    assert_eq!((first.change_seq, first.record.text.as_str(), first.record.json_ld_types.clone()), (7, "first", vec![Some("NewsArticle".to_string())]));
    assert_eq!(first.version, record_version(&first.record));
    assert_eq!(rows[1], ChangeRow::Removed(page_key("https://news.example/b").unwrap()));
    assert_eq!(rows[2], ChangeRow::TaskWithdrawn("t9".into()));
    let pages: Vec<PageRow> = vec![*first.clone()];
    assert_eq!(current::decode(Bytes::from(current::encode(&pages).unwrap())).unwrap(), pages);
}

#[test]
fn the_later_fetch_wins_whatever_order_it_was_logged_in_and_a_removal_goes_by_log_order() {
    let row = |fetched_at: &str, seq: i64, text: &str| {
        let ChangeRow::Page(row) =
            current::decode_changes(Bytes::from(changes::encode(&[page("https://news.example/a", fetched_at, "t1", text)]).unwrap()), seq).unwrap().remove(0)
        else {
            panic!("a page")
        };
        *row
    };
    let newer = row("2026-10-08T11:00:00+00:00", 1, "newer");
    let older_logged_later = row("2026-10-08T09:00:00+00:00", 2, "older");
    let merged = current::merge(vec![newer.clone(), older_logged_later], &Withdrawals::new(), false);
    assert_eq!(merged[0].record.text, "newer");
    let gone = PageRow { key: newer.key.clone(), removed: true, change_seq: 3, version: String::new(), published_at: String::new(), record: Record::default() };
    assert!(current::merge(vec![newer.clone(), gone.clone()], &Withdrawals::new(), false).is_empty());
    assert_eq!(current::merge(vec![newer.clone(), gone], &Withdrawals::new(), true).len(), 1, "a delta keeps the removal for the base");
    let withdrawn: Withdrawals = [("t1".to_string(), newer.change_seq)].into();
    assert!(current::merge(vec![newer.clone()], &withdrawn, false).is_empty());
    let taken_back_before: Withdrawals = [("t1".to_string(), newer.change_seq - 1)].into();
    assert!(current::merge(vec![newer], &taken_back_before, true).is_empty(), "logged after its take-back: never happened, not a removal");
}

#[test]
fn the_compactor_follows_the_log_into_shards_and_readers_see_the_newest_version() {
    let root = scratch("follow");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let urls: Vec<String> = (0..300).map(|i| format!("https://news.example/story-{i}")).collect();
    log.append(&urls.iter().map(|u| page(u, "2026-10-08T10:00:00+00:00", "t1", "v1")).collect::<Vec<_>>());
    log.append(&[page(&urls[0], "2026-10-08T11:00:00+00:00", "t2", "v2"), page(&urls[1], "2026-10-08T09:00:00+00:00", "t2", "stale")]);
    let settings = settings(root.join("work"));
    let report = compactor::cycle(&store, &settings, 1_000.0).unwrap();
    assert_eq!((report.from, report.through, report.rows), (1, 2, 302));
    assert_eq!(text_of(&store, &urls[0]).as_deref(), Some("v2"));
    assert_eq!(text_of(&store, &urls[1]).as_deref(), Some("v1"), "an older fetch logged later does not win");
    assert_eq!(text_of(&store, "https://news.example/never").as_deref(), None);

    log.append(&[removed(&urls[2]), page(&urls[3], "2026-10-08T12:00:00+00:00", "t3", "v3"), Change::task_withdrawn("t2", "2026-10-08T13:00:00+00:00")]);
    let report = compactor::cycle(&store, &settings, 2_000.0).unwrap();
    assert_eq!((report.from, report.through, report.withdrawn), (3, 3, 1));
    assert_eq!(text_of(&store, &urls[2]), None, "removed");
    assert_eq!(text_of(&store, &urls[3]).as_deref(), Some("v3"));
    assert_eq!(text_of(&store, &urls[0]), None, "its current version came from the withdrawn task, so it is gone");
    let again = compactor::cycle(&store, &settings, 3_000.0).unwrap();
    assert_eq!((again.rows, again.deltas), (0, 0), "nothing new, nothing written");
    let all: usize = (0..SHARDS).map(|shard| compactor::shard_rows(&store, "current/", shard).unwrap().len()).sum();
    assert_eq!(all, 298);
    assert!(compactor::load_withdrawn(&store, "current/").unwrap().tasks.contains_key("t2"), "unmerged deltas still hold its rows");
}

#[test]
fn the_current_state_can_live_in_another_store_than_the_change_log() {
    let root = scratch("split");
    let mut log = Log::new(root.join("log"));
    let store =
        Split { prefix: "current/".into(), current: Box::new(LocalStore { root: root.join("state") }), log: Box::new(LocalStore { root: root.join("log") }) };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let urls: Vec<String> = (0..50).map(|i| format!("https://news.example/split-{i}")).collect();
    let first = log.append(&urls.iter().map(|u| page(u, "2026-10-08T10:00:00+00:00", "t1", "v1")).collect::<Vec<_>>());
    let report = compactor::cycle(&store, &settings(root.join("work")), 1_000.0).unwrap();
    assert_eq!((report.through, report.rows), (first, 50));
    assert_eq!(text_of(&store, &urls[7]).as_deref(), Some("v1"));
    assert!(root.join("state/current").is_dir() && !root.join("state/changes").exists(), "the state store holds only the current state");
    assert!(root.join("log/changes").is_dir() && !root.join("log/current").exists(), "the log store is only read");
    let (checked, _, wrong) = compactor::check(&store, "current/", &[first], 10, &(0..SHARDS), 4).unwrap();
    assert!(checked > 0 && wrong.is_empty(), "{wrong:?}");
}

#[test]
fn merges_fold_deltas_into_a_base_and_replaced_files_go_after_their_grace() {
    let root = scratch("merge");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let url = "https://news.example/one";
    let shard = shard_of(&page_key(url).unwrap()).unwrap();
    let mut settings = settings(root.join("work"));
    settings.merge_ratio = 1.0;
    log.append(&[page(url, "2026-10-08T10:00:00+00:00", "t1", "v1")]);
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    let manifests = compactor::load_manifests(&store, "current/", 4).unwrap();
    let manifest = &manifests[shard];
    assert!(manifest.base.is_some() && manifest.deltas.is_empty(), "a delta outgrowing an empty base merges at once");
    assert_eq!(manifest.retired.len(), 1);
    let retired = manifest.retired[0].key.clone();
    assert!(store.get(&retired).unwrap().is_some(), "kept for readers mid-read");

    log.append(&[page(url, "2026-10-08T11:00:00+00:00", "t2", "v2"), Change::task_withdrawn("t9", "2026-10-08T11:00:00+00:00")]);
    let report = compactor::cycle(&store, &settings, 1_000.0 + 3_600.0).unwrap();
    assert!(report.deleted >= 1);
    assert!(store.get(&retired).unwrap().is_none(), "deleted after its grace");
    assert_eq!(text_of(&store, url).as_deref(), Some("v2"));
    assert!(!compactor::load_withdrawn(&store, "current/").unwrap().tasks.contains_key("t9"), "every shard merged past it, or holds nothing");
}

#[test]
fn a_number_not_written_yet_ends_the_cycle_without_skipping_it() {
    let root = scratch("hole");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    log.append(&[page("https://news.example/a", "2026-10-08T10:00:00+00:00", "t1", "a")]);
    let hole = log.append(&[page("https://news.example/b", "2026-10-08T10:00:00+00:00", "t1", "b")]);
    log.append(&[page("https://news.example/c", "2026-10-08T10:00:00+00:00", "t1", "c")]);
    store.delete(&CHANGES.seq_key(hole as u64)).unwrap();
    let settings = settings(root.join("work"));
    assert_eq!(compactor::cycle(&store, &settings, 1.0).unwrap().through, 1);
    assert_eq!(text_of(&store, "https://news.example/c"), None);
    let key = "changes/dt=2026-10-08/2.parquet";
    store.put(&CHANGES.seq_key(hole as u64), Bytes::from(format!("{{\"key\": \"{key}\", \"rows\": 1}}"))).unwrap();
    assert_eq!(compactor::cycle(&store, &settings, 2.0).unwrap().through, 3);
    assert_eq!(text_of(&store, "https://news.example/b").as_deref(), Some("b"));
    assert_eq!(text_of(&store, "https://news.example/c").as_deref(), Some("c"));
}

#[test]
fn a_build_from_an_index_snapshot_matches_it_and_the_compactor_carries_on_after_it() {
    let root = scratch("build");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    let index = publisher::index::VersionIndex::open(&root.join("index"), 64 << 20).unwrap();
    let urls: Vec<String> = (0..200).map(|i| format!("https://news.example/built-{i}")).collect();
    let first: Vec<Change> = urls.iter().map(|u| page(u, "2026-10-08T10:00:00+00:00", "t1", "v1")).collect();
    let seq = log.append(&first);
    index.store(&publisher::index::entries(&first, seq)).unwrap();
    let second: Vec<Change> = urls[..50].iter().map(|u| page(u, "2026-10-08T11:00:00+00:00", "t2", "v2")).collect();
    let through = log.append(&second);
    index.store(&publisher::index::entries(&second, through)).unwrap();
    let snapshot = root.join("snapshot.parquet");
    publisher::snapshot::write_parquet(&index, std::fs::File::create(&snapshot).unwrap()).unwrap();

    let settings = publisher::build::Settings {
        prefix: "current/".into(),
        work: root.join("build-work"),
        threads: 4,
        shards: 0..SHARDS,
        seqs: 0..i64::MAX,
        spill_bytes: 4096,
        allow_unlocated: false,
    };
    let report = publisher::build::build(&store, &snapshot, &settings).unwrap();
    assert_eq!((report.wanted, report.written, report.misplaced, report.other_version, report.through_seq), (200, 200, 0, 0, through));
    assert_eq!(text_of(&store, &urls[0]).as_deref(), Some("v2"));
    assert_eq!(text_of(&store, &urls[100]).as_deref(), Some("v1"));

    log.append(&[page(&urls[100], "2026-10-08T12:00:00+00:00", "t3", "v3")]);
    let carried = compactor::cycle(&store, &self::settings(root.join("work")), 1.0).unwrap();
    assert_eq!((carried.from, carried.rows), (through + 1, 1));
    assert_eq!(text_of(&store, &urls[100]).as_deref(), Some("v3"));
}

#[test]
fn the_check_counts_a_page_removed_later_as_removed_and_a_lost_page_as_wrong() {
    let root = scratch("check");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let mut settings = settings(root.join("work"));
    settings.merge_ratio = 1.0;
    let (gone, lost) = ("https://news.example/gone", "https://news.example/lost");
    let first = log.append(&[page(gone, "2026-10-08T10:00:00+00:00", "t1", "v1"), page(lost, "2026-10-08T10:00:00+00:00", "t1", "v1")]);
    log.append(&[removed(gone)]);
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    assert!(compactor::winner(&store, "current/", &page_key(gone).unwrap()).unwrap().is_none(), "a merged base keeps no removals");
    assert_eq!(compactor::check(&store, "current/", &[first], 10, &(0..SHARDS), 4).unwrap(), (2, 1, Vec::new()));

    let shard = shard_of(&page_key(lost).unwrap()).unwrap();
    let mut manifest = compactor::load_manifests(&store, "current/", 4).unwrap().swap_remove(shard);
    (manifest.base, manifest.deltas) = (None, Vec::new());
    store.put(&compactor::manifest_key("current/", shard), Bytes::from(serde_json::to_vec(&manifest).unwrap())).unwrap();
    let (checked, removed, wrong) = compactor::check(&store, "current/", &[first], 10, &(0..SHARDS), 4).unwrap();
    assert_eq!((checked, removed), (2, 1));
    assert_eq!(wrong, vec![format!("{} in change file {first}: not in the current state", page_key(lost).unwrap())]);
}

#[test]
fn a_page_fetched_earlier_and_published_after_a_take_back_is_the_current_one() {
    let root = scratch("takeback-order");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let url = "https://news.example/retaken";
    log.append(&[page(url, "2026-10-08T10:00:00+00:00", "t0", "first")]);
    log.append(&[page(url, "2026-10-08T12:00:00+00:00", "t2", "taken back")]);
    log.append(&[Change::task_withdrawn("t2", "2026-10-08T12:30:00+00:00")]);
    log.append(&[page(url, "2026-10-08T11:00:00+00:00", "t1", "published after")]);
    let mut settings = settings(root.join("work"));
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    assert_eq!(text_of(&store, url).as_deref(), Some("published after"), "as the publisher's index holds it");
    settings.merge_ratio = 0.0;
    log.append(&[page("https://news.example/other", "2026-10-08T13:00:00+00:00", "t3", "x")]);
    assert!(compactor::cycle(&store, &settings, 2_000.0).unwrap().merged > 0);
    assert_eq!(text_of(&store, url).as_deref(), Some("published after"), "and after the shard merges");
}

#[test]
fn a_number_never_written_is_skipped_once_a_later_one_is_old_enough() {
    let root = scratch("never-written");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    log.append(&[page("https://news.example/a", "2026-10-08T10:00:00+00:00", "t1", "a")]);
    let hole = log.append(&[page("https://news.example/b", "2026-10-08T10:00:00+00:00", "t1", "b")]);
    log.append(&[page("https://news.example/c", "2026-10-08T10:00:00+00:00", "t1", "c")]);
    store.delete(&CHANGES.seq_key(hole as u64)).unwrap();
    let mut settings = settings(root.join("work"));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    let waited = compactor::cycle(&store, &settings, now).unwrap();
    assert_eq!((waited.through, waited.waiting), (1, true));
    settings.hole_wait_s = 0.0;
    let skipped = compactor::cycle(&store, &settings, now).unwrap();
    assert_eq!((skipped.through, skipped.skipped.clone(), skipped.waiting), (3, vec![hole], false));
    assert_eq!(text_of(&store, "https://news.example/c").as_deref(), Some("c"));
}

#[test]
fn a_removal_between_two_versions_holds_whichever_cycles_the_rows_land_in() {
    let root = scratch("removal-between");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let url = "https://news.example/refetched";
    let mut settings = settings(root.join("work"));
    log.append(&[page(url, "2026-10-08T10:00:00+00:00", "t1", "removed version")]);
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    log.append(&[removed(url)]);
    log.append(&[page(url, "2026-10-08T09:00:00+00:00", "t2", "fetched earlier, logged after")]);
    settings.merge_ratio = 100.0;
    settings.fanout = 100;
    let report = compactor::cycle(&store, &settings, 2_000.0).unwrap();
    assert_eq!(report.merged, 0, "read from the base and a delta");
    assert_eq!(text_of(&store, url).as_deref(), Some("fetched earlier, logged after"));
    settings.merge_ratio = 0.0;
    log.append(&[page("https://news.example/other", "2026-10-08T13:00:00+00:00", "t3", "x")]);
    compactor::cycle(&store, &settings, 3_000.0).unwrap();
    assert_eq!(text_of(&store, url).as_deref(), Some("fetched earlier, logged after"), "and once merged");
}

#[test]
fn a_snapshot_page_with_no_change_file_row_is_read_from_its_page_object_and_one_with_neither_stops_the_build() {
    let root = scratch("unlocated");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    let index = publisher::index::VersionIndex::open(&root.join("index"), 64 << 20).unwrap();
    let located = vec![page("https://news.example/located", "2026-10-08T10:00:00+00:00", "t1", "from the log")];
    let seq = log.append(&located);
    index.store(&publisher::index::entries(&located, seq)).unwrap();
    for (url, kept) in [("https://news.example/kept-object", true), ("https://news.example/no-object", false)] {
        let record = record(url, "2026-10-05T10:00:00+00:00", "t0", &format!("old text of {url}"));
        let key = page_key(url).unwrap();
        if kept {
            let doc = serde_json::json!({
                "url": record.url, "domain": record.domain, "title": record.title, "text": record.text, "fetched_at": record.fetched_at,
                "content_sha1": record.content_sha1, "assigned_url": record.assigned_url, "task_id": record.task_id, "miner": record.miner,
                "validators": record.validators, "json_ld_types": ["NewsArticle"], "status": 200, "published": "", "author": "", "lang": "",
                "captured_at": "", "doc_id": "", "headings": [], "source": "subnet22", "final_url": null, "canonical": null,
                "page_type": null, "description": null, "text_sha256": null, "validator": null
            });
            store.put(&key, Bytes::from(zstd::encode_all(serde_json::to_vec(&doc).unwrap().as_slice(), 3).unwrap())).unwrap();
        }
        let current = publisher::index::Current {
            url: url.into(),
            version: record_version(&record),
            fetched_at: record.fetched_at.clone(),
            task_id: "t0".into(),
            content_sha1: record.content_sha1.clone(),
            change_seq: None,
            change_row: None,
        };
        index.store(&[(key, current)]).unwrap();
    }
    let snapshot = root.join("snapshot.parquet");
    publisher::snapshot::write_parquet(&index, std::fs::File::create(&snapshot).unwrap()).unwrap();
    let settings = |work: &str, allow_unlocated: bool| publisher::build::Settings {
        prefix: "current/".into(),
        work: root.join(work),
        threads: 4,
        shards: 0..SHARDS,
        seqs: 0..i64::MAX,
        spill_bytes: 4096,
        allow_unlocated,
    };
    let refused = publisher::build::build(&store, &snapshot, &settings("refused", false)).unwrap_err();
    assert!(refused.to_string().starts_with("1 pages of the snapshot have neither"), "{refused}");
    let report = publisher::build::build(&store, &snapshot, &settings("allowed", true)).unwrap();
    assert_eq!((report.unlocated, report.filled, report.other_version, report.written), (2, 1, 0, 2));
    assert_eq!(text_of(&store, "https://news.example/kept-object").as_deref(), Some("old text of https://news.example/kept-object"));
    assert_eq!(text_of(&store, "https://news.example/located").as_deref(), Some("from the log"));
}

#[test]
fn small_deltas_combine_without_rewriting_the_base_and_a_removal_in_them_still_hides_the_base() {
    let root = scratch("combine");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let urls: Vec<String> = (0..40).map(|i| format!("https://news.example/base-{i}")).collect();
    log.append(&urls.iter().map(|u| page(u, "2026-10-08T10:00:00+00:00", "t1", "first")).collect::<Vec<_>>());
    let mut settings = settings(root.join("work"));
    settings.fanout = 3;
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    let based: Vec<_> = compactor::load_manifests(&store, "current/", 4).unwrap().into_iter().map(|m| m.base.map(|b| b.key)).collect();

    let (changed, gone) = (&urls[0], &urls[1]);
    let shard = shard_of(&page_key(changed).unwrap()).unwrap();
    log.append(&[page(changed, "2026-10-08T11:00:00+00:00", "t2", "second")]);
    compactor::cycle(&store, &settings, 2_000.0).unwrap();
    log.append(&[removed(gone)]);
    compactor::cycle(&store, &settings, 3_000.0).unwrap();
    for i in 0..3 {
        log.append(&[page(changed, &format!("2026-10-08T0{i}:00:00+00:00"), "t3", "older fetch")]);
        compactor::cycle(&store, &settings, 4_000.0 + f64::from(i)).unwrap();
    }
    let manifests = compactor::load_manifests(&store, "current/", 4).unwrap();
    assert_eq!(manifests.iter().map(|m| m.base.clone().map(|b| b.key)).collect::<Vec<_>>(), based, "no base rewritten");
    assert!(manifests[shard].deltas.iter().any(|d| d.level == 1), "the shard's small deltas were combined");
    assert!(manifests.iter().all(|m| m.deltas.iter().filter(|d| d.level == 0).count() < 3));
    assert_eq!(text_of(&store, changed).as_deref(), Some("second"));
    assert_eq!(text_of(&store, gone), None);
    assert_eq!(text_of(&store, &urls[2]).as_deref(), Some("first"));
}

#[test]
fn a_withdrawn_task_s_row_logged_after_its_take_back_stays_out_until_the_base_is_rewritten_past_it() {
    let root = scratch("late-row");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let (first, late, kept) = ("https://news.example/first", "https://news.example/late", "https://news.example/kept");
    let based: Vec<Change> = [first, late, kept].iter().map(|u| page(u, "2026-10-08T09:00:00+00:00", "t0", "older")).collect();
    log.append(&based);
    let mut settings = settings(root.join("work"));
    compactor::cycle(&store, &settings, 500.0).unwrap();
    log.append(&[page(first, "2026-10-08T10:00:00+00:00", "t5", "t5's"), page(kept, "2026-10-08T10:00:00+00:00", "t1", "kept")]);
    log.append(&[Change::task_withdrawn("t5", "2026-10-08T11:00:00+00:00")]);
    let late_seq = log.append(&[page(late, "2026-10-08T10:30:00+00:00", "t5", "t5's, published late")]);
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    assert_eq!((text_of(&store, first), text_of(&store, late).as_deref()), (None, Some("older")), "the late row never happened");
    let listed = compactor::load_withdrawn(&store, "current/").unwrap().tasks["t5"];
    assert_eq!((listed.seq, listed.last), (late_seq - 1, late_seq));

    settings.merge_ratio = 100.0;
    log.append(&[page("https://news.example/other", "2026-10-08T12:00:00+00:00", "t6", "x")]);
    compactor::cycle(&store, &settings, 2_000.0).unwrap();
    assert!(compactor::load_withdrawn(&store, "current/").unwrap().tasks.contains_key("t5"), "kept while a base may still hold its rows");

    settings.merge_ratio = 0.0;
    log.append(&[first, late, kept].iter().map(|u| page(u, "2026-10-08T13:00:00+00:00", "t7", "t7's")).collect::<Vec<_>>());
    compactor::cycle(&store, &settings, 3_000.0).unwrap();
    assert!(!compactor::load_withdrawn(&store, "current/").unwrap().tasks.contains_key("t5"), "every base rewritten past its last row");
    assert_eq!(text_of(&store, late).as_deref(), Some("t7's"));
}

#[test]
fn the_compacted_state_matches_a_plain_fold_of_the_log_through_combines_rewrites_and_take_backs() {
    let root = scratch("fold");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let mut settings = settings(root.join("work"));
    (settings.fanout, settings.merge_ratio) = (2, 3.0);
    let url = |i: usize| format!("https://news.example/fold-{i}");
    for round in 0..8usize {
        let task = format!("t{round}");
        let mut changes: Vec<Change> = (0..30)
            .map(|i| page(&url((i * 7 + round * 3) % 60), &format!("2026-10-08T{:02}:00:00+00:00", 10 + (round + i) % 9), &task, &format!("{task}-{i}")))
            .collect();
        changes.push(removed(&url(round * 5)));
        if round == 5 {
            changes.push(Change::task_withdrawn("t3", "2026-10-08T15:00:00+00:00"));
            changes.push(page(&url(59), "2026-10-08T23:00:00+00:00", "t3", "late"));
        }
        log.append(&changes);
        compactor::cycle(&store, &settings, 1_000.0 * round as f64).unwrap();
    }
    let compared = compactor::compare_log(&store, "current/", &(0..SHARDS), 4).unwrap();
    assert!(compared.index_pages > 20);
    assert_eq!((compared.missing, compared.extra, compared.other_version), (vec![], vec![], vec![]));
}

#[test]
fn a_take_back_takes_down_only_the_version_that_stands() {
    let row = |task: &str, seq: i64, fetched_at: &str| {
        let record = record("https://news.example/one", fetched_at, task, task);
        PageRow { key: page_key(&record.url).unwrap(), removed: false, change_seq: seq, version: record_version(&record), published_at: String::new(), record }
    };
    let rows = vec![row("t1", 2, "2026-10-08T17:00:00+00:00"), row("t2", 3, "2026-10-08T18:00:00+00:00"), row("t3", 4, "2026-10-08T10:00:00+00:00")];
    let withdrawn: Withdrawals = [("t3".to_string(), 6)].into();
    let standing = current::merge(rows.clone(), &withdrawn, false);
    assert_eq!(standing.iter().map(|r| r.record.task_id.as_str()).collect::<Vec<_>>(), ["t2"], "t3 never stood, so its take-back hides nothing");
    let withdrawn: Withdrawals = [("t2".to_string(), 6)].into();
    assert!(current::merge(rows.clone(), &withdrawn, false).is_empty(), "t2 stood: the page is down until fetched again");
    let mut later = rows;
    later.push(row("t4", 7, "2026-10-08T09:00:00+00:00"));
    assert_eq!(current::merge(later, &withdrawn, false)[0].record.task_id, "t4", "a fetch logged after the take-back stands again");
}

fn record_of(store: &dyn Store, url: &str) -> Option<Record> {
    compactor::lookup(store, "current/", &page_key(url).unwrap()).unwrap().map(|row| row.record)
}

/// A file as written before the v2 columns existed.
fn without_v2_columns(file: Vec<u8>) -> Vec<u8> {
    let batches: Vec<RecordBatch> = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(file)).unwrap().build().unwrap().map(Result::unwrap).collect();
    let schema = batches[0].schema();
    let kept: Vec<usize> = (0..schema.fields().len()).filter(|&i| !ROW_COLUMNS_V2.contains(&schema.field(i).name().as_str())).collect();
    assert_eq!(kept.len(), schema.fields().len() - ROW_COLUMNS_V2.len());
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, Arc::new(schema.project(&kept).unwrap()), None).unwrap();
    for batch in batches {
        writer.write(&batch.project(&kept).unwrap()).unwrap();
    }
    writer.close().unwrap();
    out
}

#[test]
fn a_log_of_v1_and_v2_pages_folds_by_fetch_time_and_the_compactor_keeps_every_v2_field() {
    let root = scratch("mixed");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0, 4).unwrap();
    let (a, b, c, d) = ("https://news.example/a", "https://news.example/b", "https://news.example/c", "https://news.example/d");
    let at = |hour: u32| format!("2026-10-23T{hour:02}:00:00+00:00");
    log.append(&[page(a, &at(10), "t1", "v1 a"), v2_page(b, &at(10), "t1", "# v2 b"), page(c, &at(12), "t1", "v1 c")]);
    log.append(&[v2_page(a, &at(11), "t2", "# v2 a"), page(b, &at(9), "t2", "older v1 b"), v2_page(c, &at(11), "t2", "# older v2 c")]);
    let settings = settings(root.join("work"));
    compactor::cycle(&store, &settings, 1_000.0).unwrap();
    let manifests = compactor::load_manifests(&store, "current/", 4).unwrap();
    assert!(manifests.iter().all(|m| m.deltas.is_empty()) && manifests.iter().any(|m| m.base.is_some()), "the first cycle writes bases only");
    assert_eq!(record_of(&store, a), Some(v2_record(a, &at(11), "t2", "# v2 a")), "a later v2 fetch wins, with every v2 field");
    assert_eq!(record_of(&store, b), Some(v2_record(b, &at(10), "t1", "# v2 b")), "an older v1 fetch logged later does not win");
    assert_eq!(record_of(&store, c), Some(record(c, &at(12), "t1", "v1 c")), "a later v1 fetch wins over a v2 one");

    log.append(&[v2_page(d, &at(13), "t3", "# v2 d"), page(a, &at(14), "t3", "v1 a again")]);
    compactor::cycle(&store, &settings, 2_000.0).unwrap();
    let manifests = compactor::load_manifests(&store, "current/", 4).unwrap();
    assert!(manifests.iter().any(|m| !m.deltas.is_empty()), "the second cycle writes deltas");
    assert_eq!(record_of(&store, d), Some(v2_record(d, &at(13), "t3", "# v2 d")), "a delta keeps every v2 field");
    assert_eq!(record_of(&store, a), Some(record(a, &at(14), "t3", "v1 a again")), "a v1 page is v1 again, nothing of the v2 one left");
    let compared = compactor::compare_log(&store, "current/", &(0..SHARDS), 4).unwrap();
    assert_eq!((compared.index_pages, compared.missing, compared.extra, compared.other_version), (4, vec![], vec![], vec![]));
}

#[test]
fn files_written_before_the_v2_columns_read_as_v1_and_merge_with_v2_rows() {
    let v1 = page("https://news.example/old", "2026-10-08T10:00:00+00:00", "t1", "old");
    let old_log = without_v2_columns(changes::encode(std::slice::from_ref(&v1)).unwrap());
    let rows = current::decode_changes(Bytes::from(old_log), 4).unwrap();
    let ChangeRow::Page(old) = &rows[0] else { panic!("a page") };
    let Body::Page { record, version } = &v1.body else { panic!("a page") };
    assert_eq!((&old.record, &old.version), (record.as_ref(), version), "an old change file reads as the v1 record it held");

    let written = changes::encode(&[v1.clone(), v2_page("https://news.example/new", "2026-10-23T12:00:00+00:00", "t2", "# new")]).unwrap();
    let batch = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(written)).unwrap().build().unwrap().next().unwrap().unwrap();
    for name in ROW_COLUMNS_V2 {
        assert_eq!(batch.column_by_name(name).unwrap().null_count(), 1, "{name}: null for the v1 page only");
    }

    let base = without_v2_columns(current::encode(&[*old.clone()]).unwrap());
    let ChangeRow::Page(new) =
        current::decode_changes(Bytes::from(changes::encode(&[v2_page("https://news.example/new", "2026-10-23T12:00:00+00:00", "t2", "# new")]).unwrap()), 5)
            .unwrap()
            .remove(0)
    else {
        panic!("a page")
    };
    let delta = current::encode(&[*new.clone()]).unwrap();
    let (merged, count) = current::merge_files(vec![Bytes::from(base), Bytes::from(delta)], &Withdrawals::new(), false).unwrap();
    assert_eq!(count, 2);
    let merged = current::decode(Bytes::from(merged)).unwrap();
    let by_url = |url: &str| merged.iter().find(|row| row.record.url == url).unwrap().clone();
    assert_eq!(by_url("https://news.example/old"), *old.clone());
    assert_eq!(by_url("https://news.example/new"), *new);
}

//! The current state built from the change log: shards, newest wins, removals, withdrawals, merges and cleanup.

use std::collections::HashSet;
use std::path::PathBuf;

use bytes::Bytes;
use desearch::feeds::CHANGES;
use publisher::changes::{self, Body, Change, Kind};
use publisher::compactor::{self, LocalStore, Settings, Store};
use publisher::current::{self, shard_of, ChangeRow, PageRow, SHARDS};
use publisher::records::{page_key, record_version, Record};

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

fn page(url: &str, fetched_at: &str, task_id: &str, text: &str) -> Change {
    let record = record(url, fetched_at, task_id, text);
    Change {
        key: page_key(url).unwrap(),
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
    Settings { work, threads: 4, spill_bytes: 4096, merge_ratio: 100.0, max_deltas: 100, retire_after_s: 3600.0, ..Settings::default() }
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
    let merged = current::merge(vec![newer.clone(), older_logged_later], &HashSet::new(), false);
    assert_eq!(merged[0].record.text, "newer");
    let gone = PageRow { key: newer.key.clone(), removed: true, change_seq: 3, version: String::new(), published_at: String::new(), record: Record::default() };
    assert!(current::merge(vec![newer.clone(), gone.clone()], &HashSet::new(), false).is_empty());
    assert_eq!(current::merge(vec![newer.clone(), gone], &HashSet::new(), true).len(), 1, "a delta keeps the removal for the base");
    let withdrawn: HashSet<String> = ["t1".to_string()].into();
    assert!(current::merge(vec![newer], &withdrawn, false).is_empty());
}

#[test]
fn the_compactor_follows_the_log_into_shards_and_readers_see_the_newest_version() {
    let root = scratch("follow");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0).unwrap();
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
fn merges_fold_deltas_into_a_base_and_replaced_files_go_after_their_grace() {
    let root = scratch("merge");
    let mut log = Log::new(root.join("bucket"));
    let store = LocalStore { root: root.join("bucket") };
    compactor::start_empty(&store, "current/", 0).unwrap();
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
    compactor::start_empty(&store, "current/", 0).unwrap();
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
    index.store(&publisher::worker::indexed(&first, seq)).unwrap();
    let second: Vec<Change> = urls[..50].iter().map(|u| page(u, "2026-10-08T11:00:00+00:00", "t2", "v2")).collect();
    let through = log.append(&second);
    index.store(&publisher::worker::indexed(&second, through)).unwrap();
    let snapshot = root.join("snapshot.parquet");
    publisher::snapshot::write_parquet(&index, std::fs::File::create(&snapshot).unwrap()).unwrap();

    let settings = publisher::build::Settings {
        prefix: "current/".into(),
        work: root.join("build-work"),
        threads: 4,
        shards: 0..SHARDS,
        seqs: 0..i64::MAX,
        spill_bytes: 4096,
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

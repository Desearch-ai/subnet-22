//! The SQLite index moved into RocksDB and back without losing a field.

use publisher::index::{Current, VersionIndex};
use publisher::sqlite;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn page(i: u64) -> (String, Current) {
    let current = Current {
        url: format!("https://ex.com/{i}?q=\u{e9}"),
        version: format!("{:040x}", i * 7919),
        fetched_at: if i.is_multiple_of(5) { "yesterday".into() } else { format!("2026-10-0{}T12:00:00+00:00", i % 9 + 1) },
        task_id: format!("task{}", i % 3),
        content_sha1: format!("{:040x}", i),
        change_seq: i.is_multiple_of(2).then_some(i as i64),
        change_row: i.is_multiple_of(2).then_some((i % 1000) as i64),
    };
    (format!("pages/ex.com/{:040x}", i * 31), current)
}

#[test]
fn an_index_survives_export_and_import() {
    let dir = scratch("sqlite");
    let original = VersionIndex::open(&dir.join("original"), 8 << 20).unwrap();
    let pages: Vec<_> = (0..250_001).map(page).collect();
    original.store(&pages).unwrap();
    original.withdraw(&["task1".into(), "gone".into()], 1_791_311_000.5).unwrap();
    let file = dir.join("index.sqlite");
    let exported = sqlite::export(&original, &file, |_| {}).unwrap();
    assert_eq!((exported.pages, exported.withdrawn), (250_001, 2));

    let db = rusqlite::Connection::open(&file).unwrap();
    let columns: Vec<String> = db.prepare("PRAGMA table_info(pages)").unwrap().query_map([], |r| r.get(1)).unwrap().map(Result::unwrap).collect();
    assert_eq!(columns, ["key", "url", "version", "fetched_at", "task_id", "content_sha1", "change_seq", "change_row"]);
    let indexes: Vec<String> = db.prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'pages'").unwrap().query_map([], |r| r.get(0)).unwrap().map(Result::unwrap).collect();
    assert!(indexes.contains(&"pages_task".to_string()));
    let nulls: i64 = db.query_row("SELECT COUNT(*) FROM pages WHERE change_seq IS NULL", [], |r| r.get(0)).unwrap();
    assert_eq!(nulls, 125_000);
    drop(db);
    assert!(sqlite::export(&original, &file, |_| {}).is_err(), "never overwrites a file");

    let copy = VersionIndex::open(&dir.join("copy"), 8 << 20).unwrap();
    let mut reported = Vec::new();
    let imported = sqlite::import(&file, &copy, |done| reported.push(done)).unwrap();
    assert_eq!((imported.pages, imported.withdrawn), (250_001, 2));
    assert_eq!(reported, [100_000, 200_000]);
    assert_eq!(copy.pages().unwrap(), original.pages().unwrap());
    assert_eq!(copy.withdrawn().unwrap(), original.withdrawn().unwrap());
    assert_eq!(copy.of_tasks(&["task1".into()]).unwrap(), original.of_tasks(&["task1".into()]).unwrap());
    assert!(sqlite::import(&file, &copy, |_| {}).is_err(), "only into an empty index");
    let _ = std::fs::remove_dir_all(&dir);
}

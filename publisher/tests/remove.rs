use bytes::Bytes;
use publisher::current::{decode_changes, ChangeRow};
use publisher::local::LocalFeed;
use publisher::records::page_key;
use publisher::remove;

#[test]
fn a_url_list_becomes_removed_rows_saved_to_a_new_backup_first() {
    let dir = std::env::temp_dir().join(format!("publisher-remove-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let a = "https://en.wikipedia.org/wiki/2022_Argentine_Primera_División";
    let b = "https://ex.com/b?x=1";
    let lines = [a, "", a, "not a url", "https:///no-host", b].map(String::from);
    let (rows, refused) = remove::removals(&lines, "2026-11-23T00:00:00Z");
    assert_eq!(refused, ["not a url", "https:///no-host"]);
    assert_eq!(rows.iter().map(|r| r.url().unwrap()).collect::<Vec<_>>(), [a, b]);

    let feed = LocalFeed::new(dir.join("changes"), 7).unwrap();
    let backup = dir.join("backup.tsv");
    assert_eq!(remove::take_down(&rows, &feed, &backup).unwrap(), [7]);
    let (ka, kb) = (page_key(a).unwrap(), page_key(b).unwrap());
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), format!("{ka}\t{a}\n{kb}\t{b}\n"));
    let file = std::fs::read(dir.join("changes/000000000007.parquet")).unwrap();
    assert_eq!(decode_changes(Bytes::from(file), 7).unwrap(), [ChangeRow::Removed(ka), ChangeRow::Removed(kb)]);

    assert!(remove::take_down(&rows, &feed, &backup).is_err(), "an existing backup is never overwritten");
    assert!(!dir.join("changes/000000000008.parquet").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

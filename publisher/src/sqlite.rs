//! The Python publisher's SQLite index moved into RocksDB and back, a chunk at a time.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rusqlite::{Connection, OpenFlags};

use crate::index::{Current, VersionIndex};

const CHUNK: usize = 100_000;
/// `publisher/index.py`'s schema, created the way it creates it, so the Python publisher opens the file unchanged.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS pages (
    key          TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    version      TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    content_sha1 TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS withdrawn (
    task_id TEXT PRIMARY KEY,
    at      REAL NOT NULL
) WITHOUT ROWID;
ALTER TABLE pages ADD COLUMN change_seq INTEGER;
ALTER TABLE pages ADD COLUMN change_row INTEGER;
";

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Moved {
    pub pages: u64,
    pub withdrawn: u64,
}

/// Reads the SQLite index into an empty RocksDB index, reporting progress every `CHUNK` pages.
pub fn import(sqlite: &Path, index: &VersionIndex, mut progress: impl FnMut(u64)) -> Result<Moved> {
    if !index.is_empty()? {
        bail!("the RocksDB index already holds pages; import only into an empty one");
    }
    let db = Connection::open_with_flags(sqlite, OpenFlags::SQLITE_OPEN_READ_ONLY).with_context(|| format!("opening {}", sqlite.display()))?;
    let columns: Vec<String> = db.prepare("PRAGMA table_info(pages)")?.query_map([], |row| row.get(1))?.collect::<Result<_, _>>()?;
    let located = |name: &str| if columns.iter().any(|c| c == name) { name.to_string() } else { format!("NULL AS {name}") };
    let query = format!("SELECT key, url, version, fetched_at, task_id, content_sha1, {}, {} FROM pages", located("change_seq"), located("change_row"));
    let mut statement = db.prepare(&query)?;
    let mut rows = statement.query([])?;
    let mut moved = Moved::default();
    let mut chunk = Vec::with_capacity(CHUNK);
    while let Some(row) = rows.next()? {
        let current = Current {
            url: row.get(1)?,
            version: row.get(2)?,
            fetched_at: row.get(3)?,
            task_id: row.get(4)?,
            content_sha1: row.get(5)?,
            change_seq: row.get(6)?,
            change_row: row.get(7)?,
        };
        chunk.push((row.get::<_, String>(0)?, current));
        if chunk.len() == CHUNK {
            index.load(&chunk)?;
            moved.pages += chunk.len() as u64;
            chunk.clear();
            progress(moved.pages);
        }
    }
    index.load(&chunk)?;
    moved.pages += chunk.len() as u64;
    let withdrawn: Vec<(String, f64)> = db.prepare("SELECT task_id, at FROM withdrawn")?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<Result<_, _>>()?;
    index.load_withdrawn(&withdrawn)?;
    moved.withdrawn = withdrawn.len() as u64;
    index.flush()?;
    Ok(moved)
}

/// Writes the RocksDB index as a new SQLite file in `publisher/index.py`'s schema, for going back to the Python publisher.
pub fn export(index: &VersionIndex, sqlite: &Path, mut progress: impl FnMut(u64)) -> Result<Moved> {
    if sqlite.exists() {
        bail!("{} already exists", sqlite.display());
    }
    let mut db = Connection::open(sqlite).with_context(|| format!("creating {}", sqlite.display()))?;
    db.pragma_update(None, "journal_mode", "WAL")?;
    db.pragma_update(None, "synchronous", "NORMAL")?;
    db.execute_batch(SCHEMA)?;
    let mut moved = Moved::default();
    let mut chunk = Vec::with_capacity(CHUNK);
    index.scan(|key, current| {
        chunk.push((key, current));
        if chunk.len() == CHUNK {
            insert(&mut db, &chunk)?;
            moved.pages += chunk.len() as u64;
            chunk.clear();
            progress(moved.pages);
        }
        Ok(())
    })?;
    insert(&mut db, &chunk)?;
    moved.pages += chunk.len() as u64;
    db.execute_batch("CREATE INDEX IF NOT EXISTS pages_task ON pages (task_id);")?;
    let transaction = db.transaction()?;
    for (task_id, at) in index.withdrawn()? {
        transaction.execute("INSERT INTO withdrawn VALUES (?, ?)", rusqlite::params![task_id, at])?;
        moved.withdrawn += 1;
    }
    transaction.commit()?;
    db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok(moved)
}

fn insert(db: &mut Connection, chunk: &[(String, Current)]) -> Result<()> {
    let transaction = db.transaction()?;
    {
        let mut insert = transaction.prepare_cached(
            "INSERT INTO pages (key, url, version, fetched_at, task_id, content_sha1, change_seq, change_row) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        for (key, c) in chunk {
            insert.execute(rusqlite::params![key, c.url, c.version, c.fetched_at, c.task_id, c.content_sha1, c.change_seq, c.change_row])?;
        }
    }
    Ok(transaction.commit()?)
}

/// A progress line at most every ten seconds, for long moves.
pub fn reporter(what: &'static str) -> impl FnMut(u64) {
    let started = Instant::now();
    let mut last = Instant::now();
    move |done| {
        if last.elapsed().as_secs() < 10 {
            return;
        }
        last = Instant::now();
        let seconds = started.elapsed().as_secs_f64();
        println!("{what}: {done} pages in {seconds:.0}s ({:.0} a second)", done as f64 / seconds.max(1e-9));
    }
}

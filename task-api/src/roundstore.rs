//! Rounds as SQLite keeps them, URLs inline until Redis holds the round's tasks.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde_json::{json, Map, Value};

use crate::db::{optional_real, real};
use crate::rounds::{Batch, Round, Url};

const COLUMNS: &str = "round_id, kind, manifest_hash, seed_block, opened_at, seed, serve_order, closed_at, batches";

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS rounds (
            round_id      TEXT PRIMARY KEY,
            kind          TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            seed_block    INTEGER NOT NULL,
            opened_at     REAL NOT NULL,
            seed          TEXT,
            serve_order   TEXT NOT NULL DEFAULT '[]',
            closed_at     REAL,
            batches       TEXT NOT NULL,
            filled_at     REAL,
            tasks         INTEGER
        );
        CREATE INDEX IF NOT EXISTS rounds_opened ON rounds (opened_at);
        CREATE INDEX IF NOT EXISTS rounds_pending ON rounds (seed, closed_at);
        CREATE INDEX IF NOT EXISTS rounds_unfilled ON rounds (opened_at)
            WHERE seed IS NOT NULL AND filled_at IS NULL AND closed_at IS NULL;
        CREATE INDEX IF NOT EXISTS rounds_open_filled ON rounds (round_id)
            WHERE seed IS NOT NULL AND filled_at IS NOT NULL AND closed_at IS NULL;
        CREATE INDEX IF NOT EXISTS rounds_closed ON rounds (closed_at)
            WHERE closed_at IS NOT NULL;
        CREATE TABLE IF NOT EXISTS retention (name TEXT PRIMARY KEY, mark REAL NOT NULL);
        ",
    )?;
    if !conn.prepare("SELECT 1 FROM pragma_table_info('rounds') WHERE name = 'tasks'")?.exists([])? {
        conn.execute("ALTER TABLE rounds ADD COLUMN tasks INTEGER", [])?;
    }
    conn.execute("UPDATE rounds SET tasks = (SELECT COUNT(*) FROM json_each(batches)) WHERE seed IS NULL AND tasks IS NULL", [])?;
    Ok(())
}

/// A batch as the rounds table keeps it.
pub fn stored(batch: &Batch) -> Value {
    let urls: Vec<Value> = batch.urls.iter().map(|u| json!([u.host, u.url])).collect();
    let mut entry = Map::new();
    entry.insert("urls".into(), urls.into());
    entry.insert("extra".into(), Value::Object(batch.extra.clone()));
    if let Some(sealed) = &batch.sealed {
        entry.insert("sealed".into(), sealed.clone());
    }
    Value::Object(entry)
}

fn stored_batches(round: &Round) -> String {
    let batches: Map<String, Value> = round.batches.iter().map(|b| (b.batch_id.clone(), stored(b))).collect();
    Value::Object(batches).to_string()
}

fn batch_of(batch_id: &str, entry: &Value) -> Batch {
    let urls = entry["urls"]
        .as_array()
        .map(|urls| {
            urls.iter().map(|pair| Url { host: pair[0].as_str().unwrap_or_default().into(), url: pair[1].as_str().unwrap_or_default().into() }).collect()
        })
        .unwrap_or_default();
    let extra = entry["extra"].as_object().cloned().unwrap_or_default();
    Batch { batch_id: batch_id.into(), urls, extra, sealed: entry.get("sealed").filter(|s| !s.is_null()).cloned() }
}

fn round_of(row: &Row) -> rusqlite::Result<Round> {
    let order: String = row.get(6)?;
    let batches: String = row.get(8)?;
    let batches: Map<String, Value> = serde_json::from_str(&batches).unwrap_or_default();
    Ok(Round {
        round_id: row.get(0)?,
        kind: row.get(1)?,
        manifest_hash: row.get(2)?,
        seed_block: row.get(3)?,
        opened_at: real(row, 4)?,
        seed: row.get(5)?,
        order: serde_json::from_str(&order).unwrap_or_default(),
        closed_at: optional_real(row, 7)?,
        batches: batches.iter().map(|(id, entry)| batch_of(id, entry)).collect(),
    })
}

pub fn save(conn: &Connection, round: &Round) -> Result<()> {
    conn.execute(
        &format!(
            "INSERT INTO rounds ({COLUMNS}, tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (round_id) DO UPDATE SET seed = excluded.seed, serve_order = excluded.serve_order, closed_at = excluded.closed_at"
        ),
        params![
            round.round_id,
            round.kind,
            round.manifest_hash,
            round.seed_block,
            round.opened_at,
            round.seed,
            serde_json::to_string(&round.order)?,
            round.closed_at,
            stored_batches(round),
            round.batches.len()
        ],
    )?;
    Ok(())
}

pub fn get(conn: &Connection, round_id: &str) -> Result<Option<Round>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM rounds WHERE round_id = ?"), [round_id], round_of).optional()?)
}

/// Up to `limit` rounds not revealed yet whose seed block `by_block` has reached, oldest first.
pub fn unrevealed(conn: &Connection, by_block: i64, limit: i64) -> Result<Vec<Round>> {
    let mut statement = conn.prepare(&format!("SELECT {COLUMNS} FROM rounds WHERE seed IS NULL AND seed_block <= ? ORDER BY opened_at LIMIT ?"))?;
    let rounds = statement.query_map([by_block, limit], round_of)?;
    Ok(rounds.collect::<rusqlite::Result<_>>()?)
}

pub fn unrevealed_tasks(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COALESCE(SUM(tasks), 0) FROM rounds WHERE seed IS NULL", [], |row| row.get(0))?)
}

pub fn mark_revealed(conn: &Connection, round: &Round) -> Result<()> {
    conn.execute(
        "UPDATE rounds SET seed = ?, serve_order = ? WHERE round_id = ?",
        params![round.seed, serde_json::to_string(&round.order)?, round.round_id],
    )?;
    Ok(())
}

/// Revealed rounds whose tasks never reached the queue.
pub fn unfilled(conn: &Connection) -> Result<Vec<Round>> {
    let mut statement = conn.prepare(&format!(
        "SELECT {COLUMNS} FROM rounds INDEXED BY rounds_unfilled WHERE seed IS NOT NULL AND filled_at IS NULL AND closed_at IS NULL ORDER BY opened_at"
    ))?;
    let rounds = statement.query_map([], round_of)?;
    Ok(rounds.collect::<rusqlite::Result<_>>()?)
}

/// The round's batches with their URLs dropped, keeping each batch's count and hash.
pub fn sealed_batches(round: &Round) -> String {
    let batches: Map<String, Value> = round.batches.iter().map(|b| (b.batch_id.clone(), stored(&sealed(b)))).collect();
    Value::Object(batches).to_string()
}

/// Once Redis holds a round's tasks, it keeps only `sealed_batches`.
pub fn mark_filled(conn: &Connection, round_id: &str, sealed_batches: &str, at: f64) -> Result<()> {
    conn.execute("UPDATE rounds SET filled_at = ?, batches = ? WHERE round_id = ?", params![at, sealed_batches, round_id])?;
    Ok(())
}

fn sealed(batch: &Batch) -> Batch {
    Batch { urls: Vec::new(), sealed: Some(batch.seal()), ..batch.clone() }
}

pub fn open_revealed(conn: &Connection) -> Result<Vec<String>> {
    let mut statement =
        conn.prepare("SELECT round_id FROM rounds INDEXED BY rounds_open_filled WHERE seed IS NOT NULL AND filled_at IS NOT NULL AND closed_at IS NULL")?;
    let ids = statement.query_map([], |row| row.get(0))?;
    Ok(ids.collect::<rusqlite::Result<_>>()?)
}

/// The newest revealed round of each kind that is still open.
pub fn latest_revealed(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut statement = conn.prepare("SELECT kind, round_id FROM rounds WHERE seed IS NOT NULL AND closed_at IS NULL ORDER BY opened_at")?;
    let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Drops the URLs rounds closed before `closed_before` still keep, keeping each batch's count and hash.
pub fn seal_closed(conn: &Connection, closed_before: f64, limit: i64) -> Result<usize> {
    let mark: f64 = conn.query_row("SELECT COALESCE((SELECT mark FROM retention WHERE name = 'rounds'), 0)", [], |row| real(row, 0))?;
    let mut statement = conn.prepare(
        "SELECT round_id, closed_at, batches FROM rounds INDEXED BY rounds_closed WHERE closed_at >= ? AND closed_at < ? ORDER BY closed_at LIMIT ?",
    )?;
    let rows = statement.query_map(params![mark, closed_before, limit], |row| Ok((row.get::<_, String>(0)?, real(row, 1)?, row.get::<_, String>(2)?)))?;
    let rows: Vec<(String, f64, String)> = rows.collect::<rusqlite::Result<_>>()?;
    for (round_id, _, batches) in &rows {
        let batches: Map<String, Value> = serde_json::from_str(batches).context("a round's batches")?;
        let sealed: Map<String, Value> = batches.iter().map(|(id, entry)| (id.clone(), stored(&sealed(&batch_of(id, entry))))).collect();
        conn.execute("UPDATE rounds SET batches = ? WHERE round_id = ?", params![Value::Object(sealed).to_string(), round_id])?;
    }
    if let Some((_, closed_at, _)) = rows.last() {
        conn.execute("INSERT INTO retention VALUES ('rounds', ?) ON CONFLICT (name) DO UPDATE SET mark = excluded.mark", [closed_at])?;
    }
    Ok(rows.len())
}

pub fn close(conn: &Connection, round_id: &str, at: f64) -> Result<()> {
    conn.execute("UPDATE rounds SET closed_at = ? WHERE round_id = ?", params![at, round_id])?;
    Ok(())
}

pub fn recent(conn: &Connection, limit: i64) -> Result<Vec<Value>> {
    let mut statement =
        conn.prepare("SELECT round_id, kind, manifest_hash, seed_block, opened_at, seed IS NOT NULL, closed_at FROM rounds ORDER BY opened_at DESC LIMIT ?")?;
    let rows = statement.query_map([limit], |row| {
        Ok(json!({
            "round_id": row.get::<_, String>(0)?,
            "kind": row.get::<_, String>(1)?,
            "manifest_hash": row.get::<_, String>(2)?,
            "seed_block": row.get::<_, i64>(3)?,
            "opened_at": real(row, 4)?,
            "revealed": row.get::<_, bool>(5)?,
            "closed_at": optional_real(row, 6)?,
        }))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        conn
    }

    fn round(round_id: &str, batches: usize, seed_block: i64, opened_at: f64, seed: Option<&str>) -> Round {
        let batches = (0..batches)
            .map(|n| Batch::new(format!("b{n}"), vec![Url { host: "site.example".into(), url: "https://site.example/".into() }], Map::new()))
            .collect();
        Round {
            round_id: round_id.into(),
            batches,
            manifest_hash: "h".into(),
            seed_block,
            opened_at,
            kind: "crawl".into(),
            seed: seed.map(String::from),
            order: Vec::new(),
            closed_at: None,
        }
    }

    #[test]
    fn open_rounds_are_found_through_their_own_indexes() {
        let conn = store();
        for (n, (seed, filled, closed)) in
            [(None, None, None), (Some("s"), None, None), (Some("s"), Some(1.0), None), (Some("s"), Some(1.0), Some(2.0))].into_iter().enumerate()
        {
            let saved = round(&format!("r{n}"), 1, 1, n as f64, seed);
            save(&conn, &saved).unwrap();
            if let Some(at) = filled {
                mark_filled(&conn, &saved.round_id, &sealed_batches(&saved), at).unwrap();
            }
            if let Some(at) = closed {
                close(&conn, &format!("r{n}"), at).unwrap();
            }
        }
        assert_eq!(unfilled(&conn).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["r1"]);
        assert_eq!(open_revealed(&conn).unwrap(), ["r2"]);
    }

    #[test]
    fn only_rounds_whose_seed_block_has_come_are_loaded_for_reveal() {
        let conn = store();
        save(&conn, &round("due", 3, 10, 1.0, None)).unwrap();
        save(&conn, &round("later", 3, 20, 2.0, None)).unwrap();
        assert_eq!(unrevealed(&conn, 15, 10).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due"]);
        assert_eq!(unrevealed(&conn, i64::MAX, 10).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due", "later"]);
        assert_eq!(unrevealed(&conn, i64::MAX, 1).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due"]);
        assert_eq!(unrevealed_tasks(&conn).unwrap(), 6);
    }

    #[test]
    fn a_rounds_table_from_before_task_counts_gets_them_for_rounds_still_to_reveal() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE rounds (round_id TEXT PRIMARY KEY, kind TEXT NOT NULL, manifest_hash TEXT NOT NULL, seed_block INTEGER NOT NULL,
             opened_at REAL NOT NULL, seed TEXT, serve_order TEXT NOT NULL DEFAULT '[]', closed_at REAL, batches TEXT NOT NULL, filled_at REAL)",
        )
        .unwrap();
        for (round_id, seed) in [("waiting", None), ("revealed", Some("s"))] {
            conn.execute(
                "INSERT INTO rounds (round_id, kind, manifest_hash, seed_block, opened_at, seed, batches) VALUES (?, 'crawl', 'h', 1, 1.0, ?, ?)",
                params![round_id, seed, r#"{"b1": {}, "b2": {}, "b3": {}}"#],
            )
            .unwrap();
        }
        create(&conn).unwrap();
        assert_eq!(unrevealed_tasks(&conn).unwrap(), 3);
        save(&conn, &round("new", 2, 1, 2.0, None)).unwrap();
        assert_eq!(unrevealed_tasks(&conn).unwrap(), 5);
    }

    #[test]
    fn a_filled_round_keeps_its_manifest_without_its_urls() {
        let conn = store();
        let urls: Vec<Url> = (0..2500).map(|i| Url { host: "example.com".into(), url: format!("https://example.com/{i}") }).collect();
        let mut filled = crate::rounds::open_batches(crate::rounds::pack(urls, 1000), 100, "crawl", 1_800_000_000.0);
        crate::rounds::reveal(&mut filled, &"00".repeat(32));
        save(&conn, &filled).unwrap();
        close(&conn, &filled.round_id, 1_800_000_100.0).unwrap();
        let before = get(&conn, &filled.round_id).unwrap().unwrap().public_view();
        mark_filled(&conn, &filled.round_id, &sealed_batches(&filled), 1_800_000_010.0).unwrap();
        let batches: String = conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [&filled.round_id], |row| row.get(0)).unwrap();
        assert_eq!(batches.matches("https://example.com/").count(), 0);
        assert_eq!(get(&conn, &filled.round_id).unwrap().unwrap().public_view(), before);
    }

    #[test]
    fn a_round_closed_a_day_ago_keeps_its_manifest_without_its_urls() {
        let conn = store();
        let now = 1_800_000_000.0;
        let urls: Vec<Url> = (0..2500).map(|i| Url { host: "example.com".into(), url: format!("https://example.com/{i}") }).collect();
        let mut old = crate::rounds::open_batches(crate::rounds::pack(urls.clone(), 1000), 100, "crawl", now);
        old.round_id = "old".into();
        let mut recent = crate::rounds::open_batches(crate::rounds::pack(urls[..1200].to_vec(), 1000), 100, "crawl", now);
        recent.round_id = "recent".into();
        for round in [&mut old, &mut recent] {
            crate::rounds::reveal(round, &"00".repeat(32));
            save(&conn, round).unwrap();
        }
        close(&conn, "old", now - 2.0 * 86_400.0).unwrap();
        close(&conn, "recent", now - 60.0).unwrap();
        let before = get(&conn, "old").unwrap().unwrap().public_view();
        assert_eq!(seal_closed(&conn, now - 86_400.0, 10).unwrap(), 1);
        let stored = |round_id: &str| -> usize {
            let batches: String = conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [round_id], |row| row.get(0)).unwrap();
            batches.matches("https://example.com/").count()
        };
        assert_eq!((stored("old"), stored("recent")), (0, 1200));
        assert_eq!(get(&conn, "old").unwrap().unwrap().public_view(), before);
        assert_eq!(seal_closed(&conn, now - 86_400.0, 10).unwrap(), 1, "the last sealed round is looked at again, harmlessly");
        assert_eq!(get(&conn, "old").unwrap().unwrap().public_view(), before);
    }
}

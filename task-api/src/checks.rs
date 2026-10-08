//! What validators' checks found about each hotkey: its passes, its recent fails and its re-checks.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};

use crate::credit::{RATE_WINDOW_S, RECENT_CHECKS};
use crate::sampling::RECHECK_UPLOADS;

pub const KEEP_DAYS: f64 = 30.0;
const DAY: f64 = 86_400.0;

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS checks (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            hotkey             TEXT NOT NULL,
            task_id            TEXT NOT NULL,
            at                 REAL NOT NULL,
            upload_at          REAL NOT NULL,
            passed             INTEGER NOT NULL,
            errors_judged      INTEGER NOT NULL DEFAULT 0,
            errors_unconfirmed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS checks_hotkey ON checks (hotkey, at);
        CREATE TABLE IF NOT EXISTS check_state (
            hotkey       TEXT PRIMARY KEY,
            recheck_left INTEGER NOT NULL DEFAULT 0,
            last_pass_at REAL NOT NULL DEFAULT 0
        );
        ",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn record(
    conn: &Connection,
    hotkey: &str,
    task_id: &str,
    passed: bool,
    upload_at: f64,
    errors_judged: i64,
    errors_unconfirmed: i64,
    at: f64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO checks (hotkey, task_id, at, upload_at, passed, errors_judged, errors_unconfirmed) VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![hotkey, task_id, at, upload_at, i64::from(passed), errors_judged, errors_unconfirmed],
    )?;
    if passed {
        conn.execute(
            "INSERT INTO check_state (hotkey, last_pass_at) VALUES (?, ?)
             ON CONFLICT (hotkey) DO UPDATE SET last_pass_at = MAX(last_pass_at, excluded.last_pass_at)",
            params![hotkey, upload_at],
        )?;
    }
    Ok(())
}

pub fn passes(conn: &Connection, hotkey: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM checks WHERE hotkey = ? AND passed = 1", [hotkey], |row| row.get(0))?)
}

pub fn fails_in_recent(conn: &Connection, hotkey: &str) -> Result<i64> {
    let mut statement = conn.prepare("SELECT passed FROM checks WHERE hotkey = ? ORDER BY at DESC, id DESC LIMIT ?")?;
    let passed = statement.query_map(params![hotkey, RECENT_CHECKS], |row| row.get::<_, i64>(0))?;
    Ok(passed.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().filter(|p| *p == 0).count() as i64)
}

pub fn last_pass_at(conn: &Connection, hotkey: &str) -> Result<f64> {
    Ok(conn.query_row("SELECT last_pass_at FROM check_state WHERE hotkey = ?", [hotkey], |row| crate::db::real(row, 0)).optional()?.unwrap_or(0.0))
}

pub fn recheck_left(conn: &Connection, hotkey: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT recheck_left FROM check_state WHERE hotkey = ?", [hotkey], |row| row.get(0)).optional()?.unwrap_or(0))
}

pub fn start_recheck(conn: &Connection, hotkey: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO check_state (hotkey, recheck_left) VALUES (?, ?) ON CONFLICT (hotkey) DO UPDATE SET recheck_left = excluded.recheck_left",
        params![hotkey, RECHECK_UPLOADS],
    )?;
    Ok(())
}

pub fn took_recheck(conn: &Connection, hotkey: &str) -> Result<()> {
    conn.execute("UPDATE check_state SET recheck_left = MAX(recheck_left - 1, 0) WHERE hotkey = ?", [hotkey])?;
    Ok(())
}

/// Share of the hotkey's reported failures that checks reproduced, one of each assumed before any.
pub fn error_share(conn: &Connection, hotkey: &str, now: f64) -> Result<f64> {
    let (judged, unconfirmed): (i64, i64) = conn.query_row(
        "SELECT COALESCE(SUM(errors_judged), 0), COALESCE(SUM(errors_unconfirmed), 0) FROM checks WHERE hotkey = ? AND at >= ?",
        params![hotkey, now - RATE_WINDOW_S],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((judged - unconfirmed + 1) as f64 / (judged + 2) as f64)
}

pub fn prune(conn: &Connection, now: f64) -> Result<()> {
    conn.execute("DELETE FROM checks WHERE at < ?", [now - KEEP_DAYS * DAY])?;
    Ok(())
}

//! What validators' checks found about each hotkey in each kind of task: its passes, its recent fails and its re-checks.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use crate::budgets::CRAWL;
use crate::credit::{RATE_WINDOW_S, RECENT_CHECKS};
use crate::sampling::RECHECK_UPLOADS;

pub const KEEP_DAYS: f64 = 30.0;
const DAY: f64 = 86_400.0;

fn has_table(conn: &Connection, name: &str) -> Result<bool> {
    Ok(conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?", [name], |row| row.get::<_, i64>(0))? > 0)
}

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    Ok(conn.query_row("SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?", [table, column], |row| row.get::<_, i64>(0))? > 0)
}

/// Checks kept before they had a kind were all crawl checks; their per-hotkey state is copied once into the per-kind table.
pub fn create(conn: &Connection) -> Result<()> {
    let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let had_states = has_table(&tx, "check_states")?;
    tx.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS checks (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            hotkey             TEXT NOT NULL,
            task_id            TEXT NOT NULL,
            at                 REAL NOT NULL,
            upload_at          REAL NOT NULL,
            passed             INTEGER NOT NULL,
            errors_judged      INTEGER NOT NULL DEFAULT 0,
            errors_unconfirmed INTEGER NOT NULL DEFAULT 0,
            kind               TEXT NOT NULL DEFAULT '{CRAWL}'
        );
        CREATE TABLE IF NOT EXISTS check_states (
            hotkey       TEXT NOT NULL,
            kind         TEXT NOT NULL,
            recheck_left INTEGER NOT NULL DEFAULT 0,
            last_pass_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (hotkey, kind)
        );
        "
    ))?;
    if !has_column(&tx, "checks", "kind")? {
        tx.execute_batch(&format!("ALTER TABLE checks ADD COLUMN kind TEXT NOT NULL DEFAULT '{CRAWL}'"))?;
    }
    tx.execute_batch("CREATE INDEX IF NOT EXISTS checks_kind ON checks (hotkey, kind, at)")?;
    if !had_states && has_table(&tx, "check_state")? {
        tx.execute(
            "INSERT INTO check_states (hotkey, kind, recheck_left, last_pass_at) SELECT hotkey, ?, recheck_left, last_pass_at FROM check_state",
            [CRAWL],
        )?;
    }
    tx.commit()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn record(
    conn: &Connection,
    hotkey: &str,
    kind: &str,
    task_id: &str,
    passed: bool,
    upload_at: f64,
    errors_judged: i64,
    errors_unconfirmed: i64,
    at: f64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO checks (hotkey, kind, task_id, at, upload_at, passed, errors_judged, errors_unconfirmed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![hotkey, kind, task_id, at, upload_at, i64::from(passed), errors_judged, errors_unconfirmed],
    )?;
    if passed {
        conn.execute(
            "INSERT INTO check_states (hotkey, kind, last_pass_at) VALUES (?, ?, ?)
             ON CONFLICT (hotkey, kind) DO UPDATE SET last_pass_at = MAX(last_pass_at, excluded.last_pass_at)",
            params![hotkey, kind, upload_at],
        )?;
    }
    Ok(())
}

pub fn passes(conn: &Connection, hotkey: &str, kind: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM checks WHERE hotkey = ? AND kind = ? AND passed = 1", [hotkey, kind], |row| row.get(0))?)
}

pub fn fails_in_recent(conn: &Connection, hotkey: &str, kind: &str) -> Result<i64> {
    let mut statement = conn.prepare("SELECT passed FROM checks WHERE hotkey = ? AND kind = ? ORDER BY at DESC, id DESC LIMIT ?")?;
    let passed = statement.query_map(params![hotkey, kind, RECENT_CHECKS], |row| row.get::<_, i64>(0))?;
    Ok(passed.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().filter(|p| *p == 0).count() as i64)
}

pub fn last_pass_at(conn: &Connection, hotkey: &str, kind: &str) -> Result<f64> {
    Ok(conn
        .query_row("SELECT last_pass_at FROM check_states WHERE hotkey = ? AND kind = ?", [hotkey, kind], |row| crate::db::real(row, 0))
        .optional()?
        .unwrap_or(0.0))
}

pub fn recheck_left(conn: &Connection, hotkey: &str, kind: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT recheck_left FROM check_states WHERE hotkey = ? AND kind = ?", [hotkey, kind], |row| row.get(0)).optional()?.unwrap_or(0))
}

pub fn start_recheck(conn: &Connection, hotkey: &str, kind: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO check_states (hotkey, kind, recheck_left) VALUES (?, ?, ?) ON CONFLICT (hotkey, kind) DO UPDATE SET recheck_left = excluded.recheck_left",
        params![hotkey, kind, RECHECK_UPLOADS],
    )?;
    Ok(())
}

pub fn took_recheck(conn: &Connection, hotkey: &str, kind: &str) -> Result<()> {
    conn.execute("UPDATE check_states SET recheck_left = MAX(recheck_left - 1, 0) WHERE hotkey = ? AND kind = ?", [hotkey, kind])?;
    Ok(())
}

/// Share of the hotkey's reported crawl failures that checks reproduced, one of each assumed before any.
pub fn error_share(conn: &Connection, hotkey: &str, now: f64) -> Result<f64> {
    let (judged, unconfirmed): (i64, i64) = conn.query_row(
        "SELECT COALESCE(SUM(errors_judged), 0), COALESCE(SUM(errors_unconfirmed), 0) FROM checks WHERE hotkey = ? AND kind = ? AND at >= ?",
        params![hotkey, CRAWL, now - RATE_WINDOW_S],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((judged - unconfirmed + 1) as f64 / (judged + 2) as f64)
}

pub fn prune(conn: &Connection, now: f64) -> Result<()> {
    conn.execute("DELETE FROM checks WHERE at < ?", [now - KEEP_DAYS * DAY])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budgets::EMBED;

    const NOW: f64 = 1_800_000_000.0;

    fn before_kinds() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, hotkey TEXT NOT NULL, task_id TEXT NOT NULL, at REAL NOT NULL, upload_at REAL NOT NULL,
                passed INTEGER NOT NULL, errors_judged INTEGER NOT NULL DEFAULT 0, errors_unconfirmed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX checks_hotkey ON checks (hotkey, at);
            CREATE TABLE check_state (hotkey TEXT PRIMARY KEY, recheck_left INTEGER NOT NULL DEFAULT 0, last_pass_at REAL NOT NULL DEFAULT 0);
            INSERT INTO checks (hotkey, task_id, at, upload_at, passed) VALUES ('m', 't1', 1, 1, 1), ('m', 't2', 2, 2, 0);
            INSERT INTO check_state VALUES ('m', 7, 1.0);
            ",
        )
        .unwrap();
        conn
    }

    #[test]
    fn checks_kept_before_kinds_are_crawl_checks_and_their_state_is_copied_once() {
        let conn = before_kinds();
        create(&conn).unwrap();
        assert_eq!((passes(&conn, "m", CRAWL).unwrap(), fails_in_recent(&conn, "m", CRAWL).unwrap()), (1, 1));
        assert_eq!((passes(&conn, "m", EMBED).unwrap(), fails_in_recent(&conn, "m", EMBED).unwrap()), (0, 0));
        assert_eq!((recheck_left(&conn, "m", CRAWL).unwrap(), last_pass_at(&conn, "m", CRAWL).unwrap()), (7, 1.0));
        assert_eq!(recheck_left(&conn, "m", EMBED).unwrap(), 0);
        took_recheck(&conn, "m", CRAWL).unwrap();
        create(&conn).unwrap();
        assert_eq!(recheck_left(&conn, "m", CRAWL).unwrap(), 6, "copied only the first time");
    }

    #[test]
    fn each_kind_keeps_its_own_passes_fails_and_rechecks() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        record(&conn, "m", CRAWL, "c1", true, NOW - 10.0, 2, 1, NOW).unwrap();
        record(&conn, "m", EMBED, "e1", false, NOW - 5.0, 0, 0, NOW).unwrap();
        start_recheck(&conn, "m", EMBED).unwrap();
        assert_eq!((passes(&conn, "m", CRAWL).unwrap(), passes(&conn, "m", EMBED).unwrap()), (1, 0));
        assert_eq!((fails_in_recent(&conn, "m", CRAWL).unwrap(), fails_in_recent(&conn, "m", EMBED).unwrap()), (0, 1));
        assert_eq!((last_pass_at(&conn, "m", CRAWL).unwrap(), last_pass_at(&conn, "m", EMBED).unwrap()), (NOW - 10.0, 0.0));
        assert_eq!((recheck_left(&conn, "m", CRAWL).unwrap(), recheck_left(&conn, "m", EMBED).unwrap()), (0, RECHECK_UPLOADS));
        record(&conn, "m", EMBED, "e2", false, NOW - 4.0, 0, 0, NOW).unwrap();
        assert_eq!(error_share(&conn, "m", NOW).unwrap(), 2.0 / 4.0, "only crawl checks judge reported failures");
    }
}

//! Log, verdict and vote rows go two weeks after they were written, oldest first, a few thousand per table per pass.

use anyhow::Result;
use rusqlite::Connection;

use crate::db::real;

pub const KEPT_S: f64 = 14.0 * 86_400.0;

/// Each table with the column that dates its rows; rows are written in that order, so the lowest rowids are the oldest.
pub const DATED: [(&str, &str); 6] = [
    ("entries", "served_at"),
    ("anchors", "at"),
    ("validations", "scored_at"),
    ("votes", "finalized_at"),
    ("final_verdicts", "finalized_at"),
    ("budget_events", "at"),
];

/// Deletes up to `limit` of the oldest rows of `table` dated before `before`, stopping at the first newer one; returns how many went.
pub fn drop_older(conn: &Connection, table: &str, column: &str, before: f64, limit: i64) -> Result<usize> {
    let mut statement = conn.prepare(&format!("SELECT rowid, {column} FROM {table} ORDER BY rowid LIMIT ?"))?;
    let rows = statement.query_map([limit], |row| Ok((row.get::<_, i64>(0)?, real(row, 1)?)))?;
    let mut last = None;
    for row in rows {
        let (rowid, at) = row?;
        if at >= before {
            break;
        }
        last = Some(rowid);
    }
    Ok(match last {
        Some(rowid) => conn.execute(&format!("DELETE FROM {table} WHERE rowid <= ?"), [rowid])?,
        None => 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{budgets, roundlog, validations};

    fn dated(ats: &[f64]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL)", []).unwrap();
        for at in ats {
            conn.execute("INSERT INTO t (at) VALUES (?)", [at]).unwrap();
        }
        conn
    }

    fn left(conn: &Connection) -> Vec<f64> {
        conn.prepare("SELECT at FROM t ORDER BY id").unwrap().query_map([], |row| real(row, 0)).unwrap().map(Result::unwrap).collect()
    }

    #[test]
    fn only_rows_older_than_the_cut_go() {
        let conn = dated(&[1.0, 2.0, 3.0, 10.0, 11.0]);
        assert_eq!(drop_older(&conn, "t", "at", 5.0, 100).unwrap(), 3);
        assert_eq!(left(&conn), [10.0, 11.0]);
        assert_eq!(drop_older(&conn, "t", "at", 5.0, 100).unwrap(), 0);
    }

    #[test]
    fn a_pass_takes_at_most_the_limit() {
        let conn = dated(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(drop_older(&conn, "t", "at", 5.0, 3).unwrap(), 3);
        assert_eq!(left(&conn), [4.0]);
    }

    #[test]
    fn a_newer_row_stops_the_pass_even_with_older_ones_after_it() {
        let conn = dated(&[1.0, 9.0, 2.0]);
        assert_eq!(drop_older(&conn, "t", "at", 5.0, 100).unwrap(), 1);
        assert_eq!(left(&conn), [9.0, 2.0]);
    }

    #[test]
    fn times_stored_as_integers_count() {
        let conn = dated(&[]);
        conn.execute("INSERT INTO t (at) VALUES (1)", []).unwrap();
        assert_eq!(drop_older(&conn, "t", "at", 5.0, 100).unwrap(), 1);
    }

    #[test]
    fn every_dated_table_and_column_exists() {
        let conn = Connection::open_in_memory().unwrap();
        budgets::create(&conn).unwrap();
        roundlog::create(&conn).unwrap();
        validations::create(&conn).unwrap();
        for (table, column) in DATED {
            assert_eq!(drop_older(&conn, table, column, 1.0, 10).unwrap(), 0, "{table}.{column}");
        }
    }
}

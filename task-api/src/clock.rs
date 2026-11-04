//! Rule time: every deadline, window and lockout is a count of finalized blocks, written as seconds from a fixed anchor so stored times stay readable.
//!
//! A block's rule time is `anchor_at + (block - anchor_block) * block_seconds`. The anchor is the first finalized block this
//! API read and the wall time it read it at, kept in SQLite so every copy and every restart uses the same one; tests and the
//! devnet count blocks from `TASK_API_GENESIS`. Wall time is kept only for what people read.

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::{json, Value};

/// How a block height becomes rule time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    pub anchor_block: i64,
    pub anchor_at: f64,
    pub block_seconds: f64,
}

/// A finalized height and its rule time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tick {
    pub block: i64,
    pub at: f64,
}

impl Clock {
    pub fn at(&self, block: i64) -> f64 {
        self.anchor_at + (block - self.anchor_block) as f64 * self.block_seconds
    }

    pub fn tick(&self, block: i64) -> Tick {
        Tick { block, at: self.at(block) }
    }

    /// The blocks a duration takes, rounded up so a deadline is never shorter than its seconds.
    pub fn blocks(&self, seconds: f64) -> i64 {
        (seconds / self.block_seconds).ceil().max(0.0) as i64
    }

    /// The last block whose rule time is at or before `at`.
    pub fn block_of(&self, at: f64) -> i64 {
        self.anchor_block + ((at - self.anchor_at) / self.block_seconds + 1e-9).floor() as i64
    }

    /// The rule time a deadline of `seconds` from `from` ends at: whole blocks, plus one for the block `from` was read in.
    pub fn deadline(&self, from: Tick, seconds: f64) -> Tick {
        self.tick(from.block + self.blocks(seconds) + 1)
    }

    /// What `GET /v1/clock` shows, so anyone can turn a rule time back into the block it stands for.
    pub fn view(&self, now: Tick, wall: f64) -> Value {
        json!({
            "block": now.block,
            "rule_time": now.at,
            "wall_time": wall,
            "anchor_block": self.anchor_block,
            "anchor_at": self.anchor_at,
            "block_seconds": self.block_seconds,
        })
    }
}

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS clock (name TEXT PRIMARY KEY, block INTEGER NOT NULL, at REAL NOT NULL);")?;
    Ok(())
}

/// The anchor kept in SQLite, or this one when there is none yet; the first copy to save one wins.
pub fn anchor(conn: &Connection, block: i64, at: f64) -> Result<(i64, f64)> {
    let saved = conn.execute("INSERT OR IGNORE INTO clock (name, block, at) VALUES ('anchor', ?, ?)", params![block, (at * 1000.0).round() / 1000.0])?;
    let anchor: (i64, f64) = conn.query_row("SELECT block, at FROM clock WHERE name = 'anchor'", [], |row| Ok((row.get(0)?, crate::db::real(row, 1)?)))?;
    if saved > 0 {
        // Every rule time from here on is counted from it.
        let input = serde_json::json!({"anchor_block": anchor.0, "anchor_at": anchor.1});
        crate::journal::append(conn, crate::journal::Record::new("clock", anchor.1, input, serde_json::Value::Null))?;
    }
    Ok(anchor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_time_counts_whole_blocks_from_the_anchor() {
        let clock = Clock { anchor_block: 1_000, anchor_at: 1_800_000_000.0, block_seconds: 12.0 };
        assert_eq!(clock.at(1_000), 1_800_000_000.0);
        assert_eq!(clock.at(1_015), 1_800_000_180.0);
        assert_eq!(clock.blocks(180.0), 15);
        assert_eq!(clock.blocks(181.0), 16, "a deadline is never shorter than its seconds");
        assert_eq!(clock.block_of(1_800_000_179.9), 1_014);
        assert_eq!(clock.block_of(clock.at(1_234)), 1_234);
        let claimed = clock.tick(1_000);
        assert_eq!(clock.deadline(claimed, 190.0), clock.tick(1_017), "16 blocks for 190 s and one for the block the claim was read in");
    }

    #[test]
    fn a_stalled_chain_stalls_every_deadline() {
        let clock = Clock { anchor_block: 0, anchor_at: 0.0, block_seconds: 12.0 };
        let claimed = clock.tick(50);
        let due = clock.deadline(claimed, 180.0);
        // However long the wall clock runs, no deadline passes until blocks are finalized.
        assert!(clock.at(50) < due.at && clock.at(65) < due.at && clock.at(66) == due.at);
    }

    #[test]
    fn the_first_anchor_saved_is_kept() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        crate::journal::create(&conn).unwrap();
        assert_eq!(anchor(&conn, 500, 1_800_000_000.123_4).unwrap(), (500, 1_800_000_000.123));
        assert_eq!(anchor(&conn, 900, 1_900_000_000.0).unwrap(), (500, 1_800_000_000.123), "a second copy or a restart reads the first anchor");
        let logged: Vec<String> =
            crate::journal::entries_after(&conn, 0, 10).unwrap().into_iter().map(|(e, _)| e["kind"].as_str().unwrap().to_string()).collect();
        assert_eq!(logged, ["clock"], "only the anchor saved is logged");
    }
}

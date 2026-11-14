//! The signed log of every claim, completion, refusal and reclaim, anchored by a Merkle root when its round closes.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, ToSql};
use serde_json::{json, Map, Value};

use crate::db::{json_of, real};
use crate::proofs;
use crate::py::canonical;

/// How many ids one page of an outcome-only search scans.
const OUTCOME_WINDOW: i64 = 200_000;

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id     TEXT NOT NULL,
            hotkey       TEXT NOT NULL,
            requested_at REAL NOT NULL,
            served_at    REAL NOT NULL,
            outcome      TEXT NOT NULL,
            task_id      TEXT,
            refusal      TEXT,
            receipt_sig  TEXT NOT NULL,
            seq          INTEGER NOT NULL DEFAULT 0,
            cause        TEXT,
            block        INTEGER
        );
        CREATE INDEX IF NOT EXISTS entries_round ON entries (round_id, id);
        CREATE INDEX IF NOT EXISTS entries_hotkey ON entries (hotkey, id);
        CREATE INDEX IF NOT EXISTS entries_task ON entries (task_id, id);
        CREATE TABLE IF NOT EXISTS anchors (
            round_id TEXT PRIMARY KEY,
            root     TEXT NOT NULL,
            at       REAL NOT NULL
        );
        ",
    )?;
    Ok(())
}

/// What a receipt says, before it is signed.
#[derive(Clone, Debug, Default)]
pub struct Receipt {
    pub round_id: String,
    pub hotkey: String,
    pub requested_at: f64,
    pub outcome: &'static str,
    pub seq: i64,
    pub task_id: Option<String>,
    pub refusal: Option<Value>,
    pub cause: Option<String>,
    pub block: Option<i64>,
}

impl Receipt {
    /// The receipt's fields that are set, as `tools/verify_round.py` rebuilds them.
    pub fn body(&self) -> Value {
        let mut body = Map::new();
        body.insert("round_id".into(), self.round_id.clone().into());
        body.insert("hotkey".into(), self.hotkey.clone().into());
        body.insert("requested_at".into(), self.requested_at.into());
        body.insert("outcome".into(), self.outcome.into());
        body.insert("seq".into(), self.seq.into());
        if let Some(task_id) = &self.task_id {
            body.insert("task_id".into(), task_id.clone().into());
        }
        if let Some(refusal) = &self.refusal {
            body.insert("refusal".into(), refusal.clone());
        }
        if let Some(cause) = &self.cause {
            body.insert("cause".into(), cause.clone().into());
        }
        if let Some(block) = self.block {
            body.insert("block".into(), block.into());
        }
        Value::Object(body)
    }
}

pub fn record(conn: &Connection, receipt: &Receipt, signature: &str, served_at: f64) -> Result<()> {
    let refusal = receipt.refusal.as_ref().filter(|r| r.as_object().is_none_or(|o| !o.is_empty())).map(|r| r.to_string());
    conn.execute(
        "INSERT INTO entries (round_id, hotkey, requested_at, served_at, outcome, task_id, refusal, receipt_sig, seq, cause, block)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            receipt.round_id,
            receipt.hotkey,
            receipt.requested_at,
            served_at,
            receipt.outcome,
            receipt.task_id,
            refusal,
            signature,
            receipt.seq,
            receipt.cause,
            receipt.block
        ],
    )?;
    Ok(())
}

pub fn entries(conn: &Connection, round_id: &str) -> Result<Vec<Value>> {
    let mut statement = conn.prepare(
        "SELECT hotkey, requested_at, served_at, outcome, task_id, refusal, receipt_sig, seq, cause, block FROM entries WHERE round_id = ? ORDER BY seq, id",
    )?;
    let rows = statement.query_map([round_id], |row| {
        let mut entry = Map::new();
        entry.insert("hotkey".into(), json_of(row.get_ref(0)?));
        entry.insert("requested_at".into(), real(row, 1)?.into());
        entry.insert("served_at".into(), real(row, 2)?.into());
        entry.insert("outcome".into(), json_of(row.get_ref(3)?));
        entry.insert("receipt_sig".into(), json_of(row.get_ref(6)?));
        entry.insert("seq".into(), json_of(row.get_ref(7)?));
        if let Some(task_id) = row.get::<_, Option<String>>(4)?.filter(|t| !t.is_empty()) {
            entry.insert("task_id".into(), task_id.into());
        }
        if let Some(refusal) = row.get::<_, Option<String>>(5)?.filter(|r| !r.is_empty()) {
            entry.insert("refusal".into(), serde_json::from_str(&refusal).unwrap_or(Value::Null));
        }
        if let Some(cause) = row.get::<_, Option<String>>(8)?.filter(|c| !c.is_empty()) {
            entry.insert("cause".into(), cause.into());
        }
        if let Some(block) = row.get::<_, Option<i64>>(9)? {
            entry.insert("block".into(), block.into());
        }
        Ok(Value::Object(entry))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Entries newest first, and the cursor of the next page when there is one.
pub fn events(
    conn: &Connection,
    miner: Option<&str>,
    task_id: Option<&str>,
    outcome: Option<&str>,
    before: Option<i64>,
    limit: i64,
) -> Result<(Vec<Value>, Option<i64>)> {
    let mut clauses = vec!["1"];
    let mut args: Vec<&dyn ToSql> = Vec::new();
    for (clause, value) in [("hotkey = ?", &miner), ("task_id = ?", &task_id), ("outcome = ?", &outcome)] {
        if let Some(value) = value.as_ref().filter(|v| !v.is_empty()) {
            clauses.push(clause);
            args.push(value);
        }
    }
    if let Some(before) = &before {
        clauses.push("id < ?");
        args.push(before);
    }
    // No index covers outcome alone, so each page scans at most a window of ids and hands back where it stopped.
    let floor = match (outcome.is_some_and(|o| !o.is_empty()), miner.is_some_and(|m| !m.is_empty()), task_id.is_some_and(|t| !t.is_empty())) {
        (true, false, false) => {
            let top = match before {
                Some(before) => before,
                None => conn.query_row("SELECT COALESCE(MAX(id), 0) + 1 FROM entries", [], |row| row.get::<_, i64>(0))?,
            };
            Some((top - OUTCOME_WINDOW).max(1))
        }
        _ => None,
    };
    if let Some(floor) = &floor {
        clauses.push("id >= ?");
        args.push(floor);
    }
    args.push(&limit);
    let sql = format!(
        "SELECT id, round_id, hotkey, served_at, outcome, task_id, refusal, cause FROM entries WHERE {} ORDER BY id DESC LIMIT ?",
        clauses.join(" AND ")
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(args.as_slice(), |row| {
        let refusal = row.get::<_, Option<String>>(6)?.filter(|r| !r.is_empty());
        Ok((
            row.get::<_, i64>(0)?,
            json!({
                "id": row.get::<_, i64>(0)?,
                "round_id": json_of(row.get_ref(1)?),
                "hotkey": json_of(row.get_ref(2)?),
                "at": real(row, 3)?,
                "outcome": json_of(row.get_ref(4)?),
                "task_id": json_of(row.get_ref(5)?),
                "refusal": refusal.map_or(Value::Null, |r| serde_json::from_str(&r).unwrap_or(Value::Null)),
                "cause": json_of(row.get_ref(7)?),
            }),
        ))
    })?;
    let rows: Vec<(i64, Value)> = rows.collect::<rusqlite::Result<_>>()?;
    let next = match floor {
        Some(floor) if (rows.len() as i64) < limit => (floor > 1).then_some(floor),
        _ => (rows.len() as i64 == limit).then(|| rows.last().map(|(id, _)| *id)).flatten(),
    };
    Ok((rows.into_iter().map(|(_, event)| event).collect(), next))
}

pub fn anchor(conn: &Connection, round_id: &str, now: f64) -> Result<String> {
    let leaves: Vec<Vec<u8>> = entries(conn, round_id)?.iter().map(canonical).collect();
    let root = proofs::merkle_root(&leaves);
    conn.execute("INSERT OR REPLACE INTO anchors (round_id, root, at) VALUES (?, ?, ?)", params![round_id, root, now])?;
    Ok(root)
}

pub fn anchored_root(conn: &Connection, round_id: &str) -> Result<Option<String>> {
    Ok(conn.query_row("SELECT root FROM anchors WHERE round_id = ?", [round_id], |row| row.get(0)).optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_outcome_search_pages_through_windows_and_finds_old_entries() {
        let mut conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        let tx = conn.transaction().unwrap();
        let total = OUTCOME_WINDOW * 2 + 10;
        for id in 1..=total {
            let outcome = if id == 5 { "refused" } else { "issued" };
            tx.execute(
                "INSERT INTO entries (id, round_id, hotkey, requested_at, served_at, outcome, receipt_sig) VALUES (?, 'r', 'h', 0, 0, ?, '')",
                params![id, outcome],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let (mut found, mut before, mut pages) = (Vec::new(), None, 0);
        loop {
            let (events, next) = events(&conn, None, None, Some("refused"), before, 50).unwrap();
            found.extend(events);
            pages += 1;
            match next {
                Some(next) => before = Some(next),
                None => break,
            }
        }
        assert_eq!(found.len(), 1);
        assert_eq!(found[0]["id"], 5);
        assert_eq!(pages, 3);
        let (recent, next) = events(&conn, None, None, Some("issued"), None, 2).unwrap();
        assert_eq!(recent.iter().map(|e| e["id"].as_i64().unwrap()).collect::<Vec<_>>(), [total, total - 1]);
        assert_eq!(next, Some(total - 1));
    }
}

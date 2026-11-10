//! The signed log of every claim, completion, refusal and reclaim, anchored by a Merkle root when its round closes.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, ToSql};
use serde_json::{json, Map, Value};

use crate::db::{json_of, real};
use crate::proofs;
use crate::py::canonical;

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
            block        INTEGER,
            waited_from  REAL
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
    let has_it: i64 = conn.query_row("SELECT COUNT(*) FROM pragma_table_info('entries') WHERE name = 'waited_from'", [], |row| row.get(0))?;
    if has_it == 0 {
        // Another copy starting at the same moment may have added it first.
        if let Err(error) = conn.execute_batch("ALTER TABLE entries ADD COLUMN waited_from REAL") {
            if !error.to_string().contains("duplicate column") {
                return Err(error.into());
            }
        }
    }
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
    /// When the place in line the task was given to was taken, to the millisecond.
    pub waited_from: Option<f64>,
}

impl Receipt {
    /// The receipt's fields that are set, as `app.roundlog.receipt_body` builds them.
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
        if let Some(waited_from) = self.waited_from {
            body.insert("waited_from".into(), waited_from.into());
        }
        Value::Object(body)
    }
}

pub fn record(conn: &Connection, receipt: &Receipt, signature: &str, served_at: f64) -> Result<()> {
    let refusal = receipt.refusal.as_ref().filter(|r| r.as_object().is_none_or(|o| !o.is_empty())).map(|r| r.to_string());
    conn.execute(
        "INSERT INTO entries (round_id, hotkey, requested_at, served_at, outcome, task_id, refusal, receipt_sig, seq, cause, block, waited_from)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            receipt.block,
            receipt.waited_from
        ],
    )?;
    Ok(())
}

pub fn entries(conn: &Connection, round_id: &str) -> Result<Vec<Value>> {
    let mut statement = conn.prepare(
        "SELECT hotkey, requested_at, served_at, outcome, task_id, refusal, receipt_sig, seq, cause, block, waited_from FROM entries WHERE round_id = ? ORDER BY seq, id",
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
        if let Some(waited_from) = row.get::<_, Option<f64>>(10)? {
            entry.insert("waited_from".into(), waited_from.into());
        }
        Ok(Value::Object(entry))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// What `events` may be narrowed to.
#[derive(Default)]
pub struct EventFilter<'a> {
    pub miner: Option<&'a str>,
    pub task_id: Option<&'a str>,
    pub outcome: Option<&'a str>,
    /// The kind of the entry's round; an entry outside any round, or of a round no longer kept, has none.
    pub kind: Option<&'a str>,
    pub before: Option<i64>,
}

/// Entries newest first, each with the kind of its round, and the cursor of the next page when there is one.
pub fn events(conn: &Connection, filter: &EventFilter<'_>, limit: i64) -> Result<(Vec<Value>, Option<i64>)> {
    let mut clauses = vec!["1"];
    let mut args: Vec<&dyn ToSql> = Vec::new();
    for (clause, value) in
        [("e.hotkey = ?", &filter.miner), ("e.task_id = ?", &filter.task_id), ("e.outcome = ?", &filter.outcome), ("r.kind = ?", &filter.kind)]
    {
        if let Some(value) = value.as_ref().filter(|v| !v.is_empty()) {
            clauses.push(clause);
            args.push(value);
        }
    }
    if let Some(before) = &filter.before {
        clauses.push("e.id < ?");
        args.push(before);
    }
    args.push(&limit);
    let sql = format!(
        "SELECT e.id, e.round_id, e.hotkey, e.served_at, e.outcome, e.task_id, e.refusal, e.cause, e.waited_from, r.kind
         FROM entries e LEFT JOIN rounds r ON r.round_id = e.round_id WHERE {} ORDER BY e.id DESC LIMIT ?",
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
                "waited_from": json_of(row.get_ref(8)?),
                "kind": json_of(row.get_ref(9)?),
            }),
        ))
    })?;
    let rows: Vec<(i64, Value)> = rows.collect::<rusqlite::Result<_>>()?;
    let next = (rows.len() as i64 == limit).then(|| rows.last().map(|(id, _)| *id)).flatten();
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
    fn a_log_from_before_the_line_gains_its_column_and_keeps_its_entries_as_they_were_signed() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, round_id TEXT NOT NULL, hotkey TEXT NOT NULL, requested_at REAL NOT NULL, served_at REAL NOT NULL,
                outcome TEXT NOT NULL, task_id TEXT, refusal TEXT, receipt_sig TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0, cause TEXT, block INTEGER
            );
            INSERT INTO entries (round_id, hotkey, requested_at, served_at, outcome, receipt_sig, seq) VALUES ('r', 'h', 1.0, 2.0, 'issued', 'ab', 7);",
        )
        .unwrap();
        create(&conn).unwrap();
        create(&conn).unwrap();
        let before = entries(&conn, "r").unwrap();
        assert!(before[0].get("waited_from").is_none(), "an old entry reads as it was signed, so its round's root stands");
        let receipt = Receipt { round_id: "r".into(), hotkey: "h".into(), outcome: "issued", seq: 8, waited_from: Some(1.5), ..Receipt::default() };
        record(&conn, &receipt, "cd", 3.0).unwrap();
        assert_eq!(receipt.body()["waited_from"], 1.5);
        assert_eq!(entries(&conn, "r").unwrap()[1]["waited_from"], 1.5);
        crate::roundstore::create(&conn).unwrap();
        let issued = EventFilter { outcome: Some("issued"), ..EventFilter::default() };
        assert_eq!(events(&conn, &issued, 1).unwrap().0[0]["waited_from"], 1.5);
    }

    #[test]
    fn events_name_the_kind_of_their_round_and_can_be_narrowed_to_one() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        crate::roundstore::create(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO rounds (round_id, kind, manifest_hash, seed_block, opened_at, batches) VALUES
                ('c', 'crawl', 'h', 1, 1.0, '[]'), ('e', 'embed', 'h', 1, 1.0, '[]');",
        )
        .unwrap();
        for (seq, round_id) in [(1, "c"), (2, "e"), (3, "c"), (4, "")] {
            let receipt = Receipt { round_id: round_id.into(), hotkey: "h".into(), outcome: "issued", seq, ..Receipt::default() };
            record(&conn, &receipt, "ab", 1.0).unwrap();
        }
        let of = |kind: Option<&str>| {
            let filter = EventFilter { kind, ..EventFilter::default() };
            events(&conn, &filter, 10).unwrap().0.iter().map(|event| (event["round_id"].clone(), event["kind"].clone())).collect::<Vec<_>>()
        };
        assert_eq!(of(Some("crawl")), [(json!("c"), json!("crawl")), (json!("c"), json!("crawl"))]);
        assert_eq!(of(Some("embed")), [(json!("e"), json!("embed"))]);
        assert_eq!(of(None).len(), 4, "unnarrowed, an entry outside any round is still listed, with no kind");
        assert_eq!(of(None)[0], (json!(""), Value::Null));
    }
}

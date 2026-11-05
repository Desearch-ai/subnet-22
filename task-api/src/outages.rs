//! Outages the API saw on its own side: object storage failing its probe, and the chain's finality falling behind its best block.
//! A claim alive during one ends at no cost to its miner, as a claim held through a restart does.
//!
//! Each outage is a span of rule time from the last probe that worked to the first one that worked again, so a claim that
//! lapsed anywhere near it is forgiven. Spans are public at `GET /v1/outages`.
//!
//! Validators' failed storage witnesses make outages too, whatever the API's own probes say: wherever `quorum` distinct
//! validators reported storage failing within `window` seconds of each other, storage was down for miners. Those spans are
//! not opened or closed by the API; they are a function of the witnesses the log holds, so a replay finds the same ones.

use anyhow::Result;
use bytes::Bytes;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::state::State;

pub const STORAGE: &str = "storage";
pub const FINALITY: &str = "finality";
/// Object storage refused to lock a completed upload's version, so the completion was refused.
pub const UPLOAD_LOCK: &str = "upload_lock";
/// Where the storage probe writes; it is overwritten every probe.
pub const PROBE_KEY: &str = "probe/storage.json";
/// The cause a lapse forgiven on validators' witnesses names.
pub const WITNESSED: &str = "storage_witnessed";
pub const KEEP_DAYS: f64 = 7.0;
const DAY: f64 = 86_400.0;

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS outages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cause       TEXT NOT NULL,
            start       REAL NOT NULL,
            end         REAL,
            start_block INTEGER NOT NULL,
            end_block   INTEGER,
            detail      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS outages_span ON outages (start, end);
        CREATE TABLE IF NOT EXISTS storage_witnesses (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            validator TEXT NOT NULL,
            at        REAL NOT NULL,
            block     INTEGER NOT NULL,
            detail    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS storage_witnesses_at ON storage_witnesses (at);
        ",
    )?;
    Ok(())
}

/// Opens a span of this cause from `start` unless one is open already; true when it opened one.
pub fn begin(conn: &Connection, cause: &str, start: f64, start_block: i64, detail: &str) -> Result<bool> {
    if open(conn, cause)?.is_some() {
        return Ok(false);
    }
    conn.execute("INSERT INTO outages (cause, start, start_block, detail) VALUES (?, ?, ?, ?)", params![cause, start, start_block, detail])?;
    Ok(true)
}

/// Closes the open span of this cause at `end`; true when one was open.
pub fn end(conn: &Connection, cause: &str, end: f64, end_block: i64) -> Result<bool> {
    let Some(id) = open(conn, cause)? else { return Ok(false) };
    conn.execute("UPDATE outages SET end = ?, end_block = ? WHERE id = ?", params![end, end_block, id])?;
    Ok(true)
}

/// Opens a span and writes it into the log, so a replay forgives the same claims; true when it opened one.
pub fn begin_logged(conn: &Connection, cause: &str, start: f64, start_block: i64, detail: &str) -> Result<bool> {
    let opened = begin(conn, cause, start, start_block, detail)?;
    if opened {
        let input = serde_json::json!({"event": "begin", "cause": cause, "at": start, "block": start_block, "detail": detail});
        crate::journal::append(conn, crate::journal::Record::new("outage", start, input, Value::Null))?;
    }
    Ok(opened)
}

/// Closes a span and writes it into the log; true when one was open.
pub fn end_logged(conn: &Connection, cause: &str, at: f64, block: i64) -> Result<bool> {
    let closed = end(conn, cause, at, block)?;
    if closed {
        let input = serde_json::json!({"event": "end", "cause": cause, "at": at, "block": block});
        crate::journal::append(conn, crate::journal::Record::new("outage", at, input, Value::Null))?;
    }
    Ok(closed)
}

fn open(conn: &Connection, cause: &str) -> Result<Option<i64>> {
    Ok(conn.query_row("SELECT id FROM outages WHERE cause = ? AND end IS NULL ORDER BY id DESC LIMIT 1", [cause], |row| row.get(0)).optional()?)
}

/// The cause of an outage any part of `from..=to` fell in, the earliest first; an open span reaches to now.
pub fn during(conn: &Connection, from: f64, to: f64) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT cause FROM outages WHERE start <= ? AND (end IS NULL OR end >= ?) ORDER BY start LIMIT 1", params![to, from], |row| row.get(0))
        .optional()?)
}

/// A validator's report that storage failed it, at the rule time it reached the API.
pub fn note_witness(conn: &Connection, validator: &str, at: f64, block: i64, detail: &str) -> Result<()> {
    conn.execute("INSERT INTO storage_witnesses (validator, at, block, detail) VALUES (?, ?, ?, ?)", params![validator, at, block, detail])?;
    Ok(())
}

/// One stretch validators saw storage down: from the first report of a quorum to `window` after the last, with how many
/// distinct validators reported in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Witnessed {
    pub start: f64,
    pub end: f64,
    pub start_block: i64,
    pub end_block: i64,
    pub validators: usize,
}

/// The stretches of `from..=to` that `quorum` distinct validators' failed storage witnesses cover: each report closes a window
/// of `window` seconds back, and wherever one holds reports from `quorum` validators, storage was down from the first of them
/// to `window` after the last. Overlapping stretches are one.
pub fn witnessed(conn: &Connection, from: f64, to: f64, window: f64, quorum: i64) -> Result<Vec<Witnessed>> {
    if quorum <= 0 {
        return Ok(Vec::new());
    }
    let mut statement = conn.prepare("SELECT validator, at, block FROM storage_witnesses WHERE at >= ? AND at <= ? ORDER BY at, id")?;
    let reports: Vec<(String, f64, i64)> =
        statement.query_map(params![from - 2.0 * window, to], |row| Ok((row.get(0)?, crate::db::real(row, 1)?, row.get(2)?)))?.collect::<rusqlite::Result<_>>()?;
    let mut spans: Vec<Witnessed> = Vec::new();
    for (i, (_, at, block)) in reports.iter().enumerate() {
        let within: Vec<&(String, f64, i64)> = reports[..=i].iter().filter(|(_, then, _)| *then >= at - window).collect();
        let validators = within.iter().map(|(validator, _, _)| validator.as_str()).collect::<std::collections::BTreeSet<_>>().len();
        if (validators as i64) < quorum {
            continue;
        }
        let (start, start_block) = (within[0].1, within[0].2);
        let span = Witnessed { start, end: at + window, start_block, end_block: block + (window / 12.0).ceil() as i64, validators };
        match spans.last_mut() {
            Some(last) if last.end >= span.start => {
                last.end = last.end.max(span.end);
                last.end_block = last.end_block.max(span.end_block);
                last.validators = last.validators.max(span.validators);
            }
            _ => spans.push(span),
        }
    }
    Ok(spans.into_iter().filter(|span| span.start <= to && span.end >= from).collect())
}

pub fn recent(conn: &Connection, since: f64, limit: i64) -> Result<Vec<Value>> {
    let mut statement =
        conn.prepare("SELECT cause, start, end, start_block, end_block, detail FROM outages WHERE end IS NULL OR end >= ? ORDER BY start DESC LIMIT ?")?;
    let rows = statement.query_map(params![since, limit], |row| {
        Ok(json!({
            "cause": row.get::<_, String>(0)?,
            "start": crate::db::real(row, 1)?,
            "end": crate::db::optional_real(row, 2)?,
            "start_block": row.get::<_, i64>(3)?,
            "end_block": row.get::<_, Option<i64>>(4)?,
            "detail": row.get::<_, String>(5)?,
        }))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn prune(conn: &Connection, now: f64) -> Result<()> {
    conn.execute("DELETE FROM outages WHERE end IS NOT NULL AND end < ?", [now - KEEP_DAYS * DAY])?;
    conn.execute("DELETE FROM storage_witnesses WHERE at < ?", [now - KEEP_DAYS * DAY])?;
    Ok(())
}

/// Writes the probe object and reads it back; the error says what failed.
pub async fn probe_storage(state: &State, at: f64) -> Result<(), String> {
    let body = Bytes::from(json!({"at": at, "signer": state.signer()}).to_string());
    state.storage.put(PROBE_KEY, body, "application/json", Some("no-store")).await.map_err(|e| format!("PUT {PROBE_KEY}: {e}"))?;
    match state.storage.head(PROBE_KEY).await {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(format!("HEAD {PROBE_KEY}: not found right after it was written")),
        Err(error) => Err(format!("HEAD {PROBE_KEY}: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_claim_alive_during_an_outage_is_forgiven_and_one_after_it_is_not() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert!(begin(&conn, STORAGE, 1_000.0, 10, "PUT probe: timed out").unwrap());
        assert!(!begin(&conn, STORAGE, 1_030.0, 12, "still down").unwrap(), "one open span per cause");
        assert_eq!(during(&conn, 900.0, 1_100.0).unwrap().as_deref(), Some(STORAGE), "an open span reaches to now");
        assert!(end(&conn, STORAGE, 1_060.0, 15).unwrap());
        assert_eq!(during(&conn, 1_050.0, 1_200.0).unwrap().as_deref(), Some(STORAGE));
        assert_eq!(during(&conn, 800.0, 999.0).unwrap(), None, "a claim that lapsed before the last good probe");
        assert_eq!(during(&conn, 1_061.0, 1_300.0).unwrap(), None, "a claim made after storage came back");
        assert!(begin(&conn, FINALITY, 2_000.0, 40, "finalized 12 blocks behind best").unwrap());
        assert_eq!(during(&conn, 1_990.0, 2_000.0).unwrap().as_deref(), Some(FINALITY));
        let listed = recent(&conn, 0.0, 10).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!((listed[1]["cause"].clone(), listed[1]["end_block"].clone()), (json!(STORAGE), json!(15)));
    }

    #[test]
    fn validators_witnesses_make_an_outage_where_a_quorum_of_them_saw_storage_fail() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        note_witness(&conn, "v1", 1_000.0, 10, "probe put: HTTP 503").unwrap();
        note_witness(&conn, "v1", 1_060.0, 15, "probe put: HTTP 503").unwrap();
        assert_eq!(witnessed(&conn, 0.0, 5_000.0, 300.0, 2).unwrap(), vec![], "one validator alone is not a quorum");
        note_witness(&conn, "v2", 1_200.0, 27, "HEAD uploads/x: timed out").unwrap();
        let spans = witnessed(&conn, 0.0, 5_000.0, 300.0, 2).unwrap();
        assert_eq!(spans, vec![Witnessed { start: 1_000.0, end: 1_500.0, start_block: 10, end_block: 52, validators: 2 }]);
        assert_eq!(witnessed(&conn, 1_400.0, 1_600.0, 300.0, 2).unwrap().len(), 1, "a claim alive at its tail");
        assert!(witnessed(&conn, 1_501.0, 2_000.0, 300.0, 2).unwrap().is_empty(), "a claim made after it");
        note_witness(&conn, "v2", 2_000.0, 90, "probe get: HTTP 500").unwrap();
        assert_eq!(witnessed(&conn, 0.0, 5_000.0, 300.0, 2).unwrap().len(), 1, "a lone report later opens nothing");
        assert!(witnessed(&conn, 0.0, 5_000.0, 300.0, 0).unwrap().is_empty(), "quorum 0 turns it off");
    }
}

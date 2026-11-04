//! A probe of the uploads bucket every minute through the path miners use: a presigned PUT of an object named and
//! shaped like a miner's upload, then a presigned GET of it. A failed probe marks the storage down at that moment,
//! so a claim that could not upload then can cost its miner nothing (`lifecycle::forgiven_cause`), and every
//! probe is public, so an outage is too.

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use bytes::Bytes;
use desearch::r2::PARQUET;
use desearch::time::now;
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::db::json_of;
use crate::routes::upload_name;
use crate::state::State;
use crate::storage::delete_quietly;

pub const OUTAGE_CAUSE: &str = "storage_outage";
/// One attempt each, never retried: a retry would hide the failure a miner's upload meets.
const STEP_TIMEOUT: Duration = Duration::from_secs(20);
/// Links live as long as a probe may take.
const LINK_S: u64 = 60;
/// A probe's size is drawn from this range, so it can't be told from a small upload by its size.
const MIN_BYTES: usize = 16 << 10;
const MAX_BYTES: usize = 256 << 10;
const MAGIC: &[u8; 4] = b"PAR1";
pub const KEEP_S: f64 = 7.0 * 86_400.0;
const MAX_PROBES: i64 = 1_000;

static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| desearch::r2::client().expect("an HTTP client"));

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS storage_probes (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            at      REAL NOT NULL,
            block   INTEGER,
            ok      INTEGER NOT NULL,
            step    TEXT,
            detail  TEXT,
            took_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS storage_probes_at ON storage_probes (at);
        ",
    )?;
    Ok(())
}

/// One probe: when it started, the block then, and the step that failed if one did.
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    pub at: f64,
    pub block: Option<i64>,
    pub failed: Option<(&'static str, String)>,
    pub took_ms: i64,
}

/// Bytes framed like a Parquet file, random in between and of a random size.
fn body() -> Bytes {
    let size = MIN_BYTES + rand::random::<usize>() % (MAX_BYTES - MIN_BYTES);
    let mut body: Vec<u8> = Vec::with_capacity(size);
    body.extend(MAGIC);
    body.extend((0..size - 2 * MAGIC.len()).map(|_| rand::random::<u8>()));
    body.extend(MAGIC);
    Bytes::from(body)
}

/// Writes and reads back one object the way a miner and a validator do; never fails, it reports.
pub async fn probe(state: &State) -> Probe {
    let at = now();
    let block = state.seeds.current_block().await.ok();
    let key = format!("uploads/{}.parquet", upload_name());
    let since = Duration::from_secs_f64(at);
    let started = Instant::now();
    let sent = body();
    let put = state.storage.presign("PUT", &key, LINK_S, Some(PARQUET), since);
    let failed = match HTTP.put(put).header("content-type", PARQUET).body(sent.clone()).timeout(STEP_TIMEOUT).send().await {
        Err(error) => Some(("put", error.to_string())),
        Ok(answer) if !answer.status().is_success() => Some(("put", format!("HTTP {}", answer.status().as_u16()))),
        Ok(_) => read_back(state, &key, since, &sent).await.err(),
    };
    let took_ms = started.elapsed().as_millis() as i64;
    if failed.as_ref().is_none_or(|(step, _)| *step != "put") {
        delete_quietly(&state.storage, &key).await;
    }
    Probe { at, block, failed, took_ms }
}

async fn read_back(state: &State, key: &str, since: Duration, sent: &Bytes) -> Result<(), (&'static str, String)> {
    let get = state.storage.presign("GET", key, LINK_S, None, since);
    let answer = HTTP.get(get).timeout(STEP_TIMEOUT).send().await.map_err(|error| ("get", error.to_string()))?;
    if !answer.status().is_success() {
        return Err(("get", format!("HTTP {}", answer.status().as_u16())));
    }
    let read = answer.bytes().await.map_err(|error| ("get", error.to_string()))?;
    if read != *sent {
        return Err(("compare", format!("read back {} bytes, wrote {}", read.len(), sent.len())));
    }
    Ok(())
}

pub fn record(conn: &Connection, probe: &Probe) -> Result<()> {
    let (step, detail) = probe.failed.clone().map_or((None, None), |(step, detail)| (Some(step), Some(detail)));
    conn.execute(
        "INSERT INTO storage_probes (at, block, ok, step, detail, took_ms) VALUES (?, ?, ?, ?, ?, ?)",
        params![probe.at, probe.block, probe.failed.is_none(), step, detail, probe.took_ms],
    )?;
    Ok(())
}

/// Runs a probe and keeps it; a failure is logged as it happens.
pub async fn run(state: &State) -> Result<Probe> {
    let found = probe(state).await;
    if let Some((step, detail)) = &found.failed {
        eprintln!("storage canary: the {step} step failed after {} ms: {detail}", found.took_ms);
    }
    let kept = found.clone();
    state.db.run(move |conn| record(conn, &kept)).await?;
    Ok(found)
}

/// Whether a probe failed between `from` and `to`, so a claim held then may not have been able to upload.
pub fn outage_during(conn: &Connection, from: f64, to: f64) -> Result<bool> {
    let failed: i64 = conn.query_row("SELECT COUNT(*) FROM storage_probes WHERE ok = 0 AND at >= ? AND at <= ?", params![from, to], |row| row.get(0))?;
    Ok(failed > 0)
}

pub fn prune(conn: &Connection, now: f64) -> Result<()> {
    conn.execute("DELETE FROM storage_probes WHERE at < ?", [now - KEEP_S])?;
    Ok(())
}

/// The probes since `since`, newest first, and the outages they show: each run of failed probes, from the last
/// probe that passed before it to the first that passed after it (none while it lasts).
pub fn report(conn: &Connection, since: f64, limit: i64) -> Result<Value> {
    let limit = limit.clamp(1, MAX_PROBES);
    let mut statement = conn.prepare("SELECT at, block, ok, step, detail, took_ms FROM storage_probes WHERE at >= ? ORDER BY at DESC LIMIT ?")?;
    let probes: Vec<Value> = statement
        .query_map(params![since, limit], |row| {
            Ok(json!({
                "at": row.get::<_, f64>(0)?,
                "block": json_of(row.get_ref(1)?),
                "ok": row.get::<_, bool>(2)?,
                "step": json_of(row.get_ref(3)?),
                "detail": json_of(row.get_ref(4)?),
                "took_ms": row.get::<_, i64>(5)?,
            }))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut outages: Vec<Value> = Vec::new();
    let mut open: Option<Value> = None;
    // Oldest first; a run of failures closes at the first probe that passes.
    let mut passed: Option<&Value> = None;
    for probe in probes.iter().rev() {
        match (probe["ok"].as_bool() == Some(true), open.as_mut()) {
            (false, None) => {
                open = Some(json!({
                    "from_at": passed.map_or(probe["at"].clone(), |p| p["at"].clone()),
                    "from_block": passed.map_or(probe["block"].clone(), |p| p["block"].clone()),
                    "to_at": null,
                    "to_block": null,
                    "failed_probes": 1,
                }));
            }
            (false, Some(outage)) => outage["failed_probes"] = (outage["failed_probes"].as_i64().unwrap_or(0) + 1).into(),
            (true, Some(outage)) => {
                outage["to_at"] = probe["at"].clone();
                outage["to_block"] = probe["block"].clone();
                outages.push(open.take().expect("an open outage"));
            }
            (true, None) => {}
        }
        if probe["ok"].as_bool() == Some(true) {
            passed = Some(probe);
        }
    }
    outages.extend(open);
    outages.reverse();
    Ok(json!({"probes": probes, "outages": outages}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(conn: &Connection, at: f64, ok: bool) {
        let failed = (!ok).then(|| ("put", "HTTP 503".to_string()));
        record(conn, &Probe { at, block: Some(at as i64 / 12), failed, took_ms: 5 }).unwrap();
    }

    #[test]
    fn a_failed_probe_inside_a_claim_marks_it_and_outages_run_from_the_last_pass_to_the_next() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        for (at, ok) in [(60.0, true), (120.0, false), (180.0, false), (240.0, true), (300.0, true), (360.0, false)] {
            probe(&conn, at, ok);
        }
        assert!(outage_during(&conn, 100.0, 130.0).unwrap());
        assert!(!outage_during(&conn, 240.0, 350.0).unwrap(), "only passes then");
        let shown = report(&conn, 0.0, 100).unwrap();
        assert_eq!(shown["probes"].as_array().unwrap().len(), 6);
        let outages = shown["outages"].as_array().unwrap();
        assert_eq!(outages.len(), 2);
        assert_eq!((outages[0]["from_at"].as_f64(), outages[0]["to_at"].as_f64(), outages[0]["failed_probes"].as_i64()), (Some(300.0), None, Some(1)));
        assert_eq!((outages[1]["from_at"].as_f64(), outages[1]["to_at"].as_f64(), outages[1]["failed_probes"].as_i64()), (Some(60.0), Some(240.0), Some(2)));
    }

    #[test]
    fn a_probe_looks_like_an_upload_and_its_size_varies() {
        let (a, b) = (body(), body());
        assert!(a.starts_with(MAGIC) && a.ends_with(MAGIC) && (MIN_BYTES..MAX_BYTES).contains(&a.len()));
        assert_ne!(a, b);
    }
}

//! Finalized verdicts, every validator's vote, validators' audit standing, and the queries the log pages read.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use anyhow::Result;
use desearch::time::utc_day;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rusqlite::types::{Value as Sql, ValueRef};
use rusqlite::{params, params_from_iter, Connection, ErrorCode, OptionalExtension, ToSql};
use serde_json::{json, Map, Value};

use crate::budgets::hour_of;
use crate::db::{json_of, object, real};

pub const COUNTS: [&str; 11] = [
    "returned",
    "missing",
    "duplicates",
    "sampled",
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "reextract_mismatch",
    "credited",
];
pub const FIELDS: [&str; 24] = [
    "task_id",
    "kind",
    "round_id",
    "miner",
    "validator",
    "verdict",
    "reason",
    "returned",
    "missing",
    "duplicates",
    "sampled",
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "reextract_mismatch",
    "credited",
    "upload_key",
    "page_key",
    "report_key",
    "claimed_at",
    "completed_at",
    "scored_at",
];
pub const VOTE_COUNTS: [&str; 8] = ["returned", "sampled", "matched", "mismatched", "unverifiable", "errors_confirmed", "errors_unconfirmed", "credited"];
pub const VOTE_FIELDS: [&str; 20] = [
    "task_id",
    "kind",
    "miner",
    "validator",
    "verdict",
    "reason",
    "returned",
    "sampled",
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "credited",
    "final_verdict",
    "final_credited",
    "agreed",
    "decided",
    "voted_at",
    "finalized_at",
];
pub const VERDICTS: [&str; 3] = ["pass", "fail", "void"];
pub const WITHDRAWN: &str = "withdrawn";
pub const MIN_AUDITS: i64 = 10;
pub const URL_DETAIL_DAYS: f64 = 2.0;
pub const MAX_DISAGREEMENT: f64 = 0.3;
const VERDICT_SUMS: &str = "SUM(verdict = 'pass'), SUM(verdict = 'fail'), SUM(verdict = 'void')";

pub fn create(conn: &Connection) -> Result<()> {
    let counts: String = COUNTS.iter().map(|name| format!("{name} INTEGER NOT NULL DEFAULT 0, ")).collect();
    let vote_counts: String = VOTE_COUNTS.iter().map(|name| format!("{name} INTEGER NOT NULL DEFAULT 0, ")).collect();
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS validations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL,
            kind       TEXT NOT NULL,
            round_id   TEXT NOT NULL,
            miner      TEXT NOT NULL,
            validator  TEXT NOT NULL,
            verdict    TEXT NOT NULL,
            reason     TEXT NOT NULL,
            {counts}
            upload_key TEXT NOT NULL,
            page_key   TEXT,
            report_key TEXT NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            scored_at  REAL NOT NULL,
            report     TEXT NOT NULL,
            urls       TEXT
        );
        CREATE INDEX IF NOT EXISTS validations_task ON validations (task_id, id);
        CREATE INDEX IF NOT EXISTS validations_miner ON validations (miner, verdict);
        CREATE INDEX IF NOT EXISTS validations_scored ON validations (scored_at);
        CREATE INDEX IF NOT EXISTS validations_miner_scored ON validations (miner, scored_at);
        CREATE INDEX IF NOT EXISTS validations_validator_scored ON validations (validator, scored_at);
        CREATE TABLE IF NOT EXISTS verdict_counts (
            miner   TEXT NOT NULL,
            verdict TEXT NOT NULL,
            n       INTEGER NOT NULL,
            PRIMARY KEY (miner, verdict)
        );
        CREATE TABLE IF NOT EXISTS validator_audits (
            hotkey        TEXT PRIMARY KEY,
            audits        INTEGER NOT NULL DEFAULT 0,
            disagreements INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS final_verdicts (
            task_id    TEXT NOT NULL,
            upload_key TEXT NOT NULL,
            verdict    TEXT NOT NULL,
            credited   INTEGER NOT NULL,
            publish    TEXT,
            finalized_at REAL NOT NULL,
            PRIMARY KEY (task_id, upload_key)
        );
        CREATE TABLE IF NOT EXISTS retention (name TEXT PRIMARY KEY, mark REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS miner_hours (
            miner          TEXT NOT NULL,
            kind           TEXT NOT NULL,
            hour           INTEGER NOT NULL,
            tasks          INTEGER NOT NULL DEFAULT 0,
            pass           INTEGER NOT NULL DEFAULT 0,
            fail           INTEGER NOT NULL DEFAULT 0,
            void           INTEGER NOT NULL DEFAULT 0,
            returned       INTEGER NOT NULL DEFAULT 0,
            missing        INTEGER NOT NULL DEFAULT 0,
            credited       INTEGER NOT NULL DEFAULT 0,
            last_scored_at REAL,
            PRIMARY KEY (miner, kind, hour)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS miner_hours_hour ON miner_hours (hour);
        CREATE TABLE IF NOT EXISTS votes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id        TEXT NOT NULL,
            kind           TEXT NOT NULL,
            miner          TEXT NOT NULL,
            validator      TEXT NOT NULL,
            upload_key     TEXT NOT NULL,
            verdict        TEXT NOT NULL,
            reason         TEXT NOT NULL,
            {vote_counts}
            final_verdict  TEXT NOT NULL,
            final_credited INTEGER NOT NULL,
            agreed         INTEGER,
            decided        INTEGER NOT NULL,
            voted_at       REAL NOT NULL,
            finalized_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS votes_task ON votes (task_id, id);
        CREATE INDEX IF NOT EXISTS votes_validator ON votes (validator, id);
        CREATE INDEX IF NOT EXISTS votes_miner ON votes (miner, id);
        CREATE INDEX IF NOT EXISTS votes_finalized ON votes (finalized_at);
        "
    ))?;
    if conn.query_row("SELECT 1 FROM verdict_counts LIMIT 1", [], |_| Ok(())).optional()?.is_none() {
        conn.execute("INSERT INTO verdict_counts SELECT miner, verdict, COUNT(*) FROM validations GROUP BY miner, verdict", [])?;
    }
    Ok(())
}

/// A JSON value as the column Python's sqlite3 would have written for it.
pub fn sql(value: &Value) -> Sql {
    match value {
        Value::Null => Sql::Null,
        Value::Bool(b) => Sql::Integer(i64::from(*b)),
        Value::Number(n) => n.as_i64().map(Sql::Integer).unwrap_or_else(|| Sql::Real(n.as_f64().unwrap_or(0.0))),
        Value::String(s) => Sql::Text(s.clone()),
        other => Sql::Text(other.to_string()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalVerdict {
    pub verdict: String,
    pub credited: i64,
    pub publish: Option<Value>,
}

/// False when this upload was finalized before, so nothing is paid twice.
pub fn finalize(conn: &Connection, task_id: &str, upload_key: &str, verdict: &str, credited: i64, publish: Option<&Value>, now: f64) -> Result<bool> {
    // Without its URLs: the copy is only read back while the upload's job, which has them, is still open.
    let copy = publish.map(|job| {
        let mut copy = job.clone();
        if let Some(fields) = copy.as_object_mut() {
            fields.remove("urls");
        }
        copy.to_string()
    });
    let inserted = conn.execute("INSERT INTO final_verdicts VALUES (?, ?, ?, ?, ?, ?)", params![task_id, upload_key, verdict, credited, copy, now]);
    match inserted {
        Ok(_) => Ok(true),
        Err(rusqlite::Error::SqliteFailure(error, _)) if error.code == ErrorCode::ConstraintViolation => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub fn final_verdict(conn: &Connection, task_id: &str, upload_key: &str) -> Result<Option<FinalVerdict>> {
    Ok(conn
        .query_row("SELECT verdict, credited, publish FROM final_verdicts WHERE task_id = ? AND upload_key = ?", params![task_id, upload_key], |row| {
            let publish: Option<String> = row.get(2)?;
            Ok(FinalVerdict { verdict: row.get(0)?, credited: row.get(1)?, publish: publish.and_then(|p| serde_json::from_str(&p).ok()) })
        })
        .optional()?)
}

/// A check's per-URL detail as the validations table keeps it, compressed about tenfold.
pub fn pack_details(urls: Option<&Value>) -> Option<Vec<u8>> {
    let Some(Value::Array(urls)) = urls.filter(|urls| urls.as_array().is_some_and(|urls| !urls.is_empty())) else { return None };
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&serde_json::to_vec(urls).ok()?).ok()?;
    encoder.finish().ok()
}

/// Rows counted into the hourly totals with each verdict: its own and any written before it.
const ROLL_UP_WITH_VERDICT: i64 = 20;

pub fn record(conn: &Connection, report: &Map<String, Value>, details: Option<&[u8]>) -> Result<()> {
    let columns: Vec<&str> = FIELDS.iter().copied().chain(["report", "urls"]).collect();
    let mut values: Vec<Sql> = FIELDS.iter().map(|name| sql(report.get(*name).unwrap_or(&Value::Null))).collect();
    values.push(Sql::Text(Value::Object(report.clone()).to_string()));
    values.push(details.map_or(Sql::Null, |details| Sql::Blob(details.to_vec())));
    conn.execute(&format!("INSERT INTO validations ({}) VALUES ({})", columns.join(", "), vec!["?"; columns.len()].join(", ")), params_from_iter(values))?;
    conn.execute(
        "INSERT INTO verdict_counts VALUES (?, ?, 1) ON CONFLICT (miner, verdict) DO UPDATE SET n = n + 1",
        params![report["miner"].as_str(), report["verdict"].as_str()],
    )?;
    roll_up(conn, ROLL_UP_WITH_VERDICT)?;
    Ok(())
}

/// Every validator's vote on one upload; `decided` is false when no vote carried it.
pub fn record_votes(conn: &Connection, report: &Map<String, Value>, votes: &[Value], disagreed: &[String], decided: bool) -> Result<()> {
    let columns: Vec<&str> = VOTE_FIELDS[..4].iter().copied().chain(["upload_key"]).chain(VOTE_FIELDS[4..].iter().copied()).collect();
    let insert = format!("INSERT INTO votes ({}) VALUES ({})", columns.join(", "), vec!["?"; columns.len()].join(", "));
    for vote in votes {
        let result = &vote["result"];
        let validator = vote["validator"].as_str().unwrap_or_default();
        let mut values = vec![
            sql(&report["task_id"]),
            sql(&report["kind"]),
            sql(&report["miner"]),
            Sql::Text(validator.into()),
            sql(&report["upload_key"]),
            sql(&vote["verdict"]),
            Sql::Text(result.get("reason").and_then(Value::as_str).unwrap_or_default().into()),
        ];
        values.extend(VOTE_COUNTS.iter().map(|name| sql(result.get(*name).unwrap_or(&json!(0)))));
        values.push(sql(&report["verdict"]));
        values.push(sql(&report["credited"]));
        values.push(if decided { Sql::Integer(i64::from(!disagreed.iter().any(|d| d == validator))) } else { Sql::Null });
        values.push(Sql::Integer(i64::from(decided && report["validator"] == validator)));
        values.push(sql(vote.get("at").unwrap_or(&report["scored_at"])));
        values.push(sql(&report["scored_at"]));
        conn.execute(&insert, params_from_iter(values))?;
    }
    Ok(())
}

fn as_vote(row: &rusqlite::Row, from: usize) -> rusqlite::Result<Value> {
    let mut vote = object(row, &VOTE_FIELDS, from)?;
    let agreed = vote["agreed"].as_i64().map(|a| Value::Bool(a != 0)).unwrap_or(Value::Null);
    vote.insert("agreed".into(), agreed);
    let decided = vote["decided"].as_i64().unwrap_or(0) != 0;
    vote.insert("decided".into(), decided.into());
    Ok(Value::Object(vote))
}

/// Filters for the votes page.
#[derive(Default)]
pub struct VoteFilter {
    pub validator: Option<String>,
    pub miner: Option<String>,
    pub task_id: Option<String>,
    pub verdict: Option<String>,
    pub agreed: Option<bool>,
    pub until: Option<f64>,
    pub before: Option<i64>,
    pub limit: i64,
}

/// Votes newest first, and the cursor of the next page when there is one.
pub fn votes(conn: &Connection, filter: &VoteFilter) -> Result<(Vec<Value>, Option<i64>)> {
    let mut clauses = vec!["1".to_string()];
    let mut args: Vec<Sql> = Vec::new();
    for (column, value) in [("validator", &filter.validator), ("miner", &filter.miner), ("task_id", &filter.task_id), ("verdict", &filter.verdict)] {
        if let Some(value) = value {
            clauses.push(format!("{column} = ?"));
            args.push(Sql::Text(value.clone()));
        }
    }
    if let Some(agreed) = filter.agreed {
        clauses.push("agreed = ?".into());
        args.push(Sql::Integer(i64::from(agreed)));
    }
    if let Some(until) = filter.until {
        clauses.push("finalized_at <= ?".into());
        args.push(Sql::Real(until));
    }
    if let Some(before) = filter.before {
        clauses.push("id < ?".into());
        args.push(Sql::Integer(before));
    }
    args.push(Sql::Integer(filter.limit));
    let mut statement = conn.prepare(&format!("SELECT id, {} FROM votes WHERE {} ORDER BY id DESC LIMIT ?", VOTE_FIELDS.join(", "), clauses.join(" AND ")))?;
    let rows = statement.query_map(params_from_iter(args), |row| Ok((row.get::<_, i64>(0)?, as_vote(row, 1)?)))?;
    let rows: Vec<(i64, Value)> = rows.collect::<rusqlite::Result<_>>()?;
    let next = (rows.len() as i64 == filter.limit).then(|| rows.last().map(|(id, _)| *id)).flatten();
    Ok((rows.into_iter().map(|(_, vote)| vote).collect(), next))
}

pub fn votes_on(conn: &Connection, task_id: &str, upload_key: &str) -> Result<Vec<Value>> {
    let mut statement = conn.prepare(&format!("SELECT {} FROM votes WHERE task_id = ? AND upload_key = ? ORDER BY id", VOTE_FIELDS.join(", ")))?;
    let rows = statement.query_map(params![task_id, upload_key], |row| as_vote(row, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn int_or_zero(row: &rusqlite::Row, at: usize) -> rusqlite::Result<i64> {
    Ok(row.get::<_, Option<i64>>(at)?.unwrap_or(0))
}

pub fn validator_totals(conn: &Connection, since: f64, until: f64) -> Result<BTreeMap<String, Value>> {
    let mut statement = conn.prepare(&format!(
        "SELECT validator, COUNT(*), {VERDICT_SUMS}, SUM(agreed = 1), SUM(agreed = 0), SUM(decided), MAX(voted_at) FROM votes
         WHERE finalized_at >= ? AND finalized_at <= ? GROUP BY validator"
    ))?;
    let rows = statement.query_map(params![since, until], |row| {
        Ok((
            row.get::<_, String>(0)?,
            json!({
                "votes": row.get::<_, i64>(1)?,
                "pass": json_of(row.get_ref(2)?),
                "fail": json_of(row.get_ref(3)?),
                "void": json_of(row.get_ref(4)?),
                "agreed": int_or_zero(row, 5)?,
                "disagreed": int_or_zero(row, 6)?,
                "decided": json_of(row.get_ref(7)?),
                "last_vote_at": json_of(row.get_ref(8)?),
            }),
        ))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

const MINER_TOTAL_NAMES: [&str; 8] = ["tasks", "pass", "fail", "void", "returned", "missing", "credited", "last_scored_at"];

/// Each miner's verdicts scored in `since..=until`: whole hours from the hourly totals, the part hours at either end from the rows.
pub fn miner_totals(conn: &Connection, since: f64, until: f64, kind: &str) -> Result<BTreeMap<String, Value>> {
    let (first, last) = (hour_of(since), hour_of(until));
    let mut totals = BTreeMap::new();
    if first < last {
        add_totals(
            &mut totals,
            conn,
            "SELECT miner, SUM(tasks), SUM(pass), SUM(fail), SUM(void), SUM(returned), SUM(missing), SUM(credited), MAX(last_scored_at) FROM miner_hours
             WHERE hour > ? AND hour < ? AND kind = ? GROUP BY miner",
            params![first, last, kind],
        )?;
        add_scored(&mut totals, conn, since, (first + 1) as f64 * 3600.0, false, kind)?;
        add_scored(&mut totals, conn, last as f64 * 3600.0, until, true, kind)?;
    } else {
        add_scored(&mut totals, conn, since, until, true, kind)?;
    }
    Ok(totals)
}

fn add_scored(totals: &mut BTreeMap<String, Value>, conn: &Connection, since: f64, until: f64, inclusive: bool, kind: &str) -> Result<()> {
    let sql = format!(
        "SELECT miner, COUNT(*), {VERDICT_SUMS}, SUM(returned), SUM(missing), SUM(credited), MAX(scored_at) FROM validations
         WHERE scored_at >= ? AND scored_at {} ? AND kind = ? GROUP BY miner",
        if inclusive { "<=" } else { "<" }
    );
    add_totals(totals, conn, &sql, params![since, until, kind])
}

fn add_totals(totals: &mut BTreeMap<String, Value>, conn: &Connection, sql: &str, args: impl rusqlite::Params) -> Result<()> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement.query_map(args, |row| Ok((row.get::<_, String>(0)?, object(row, &MINER_TOTAL_NAMES, 1)?)))?;
    for row in rows {
        let (miner, add) = row?;
        let Value::Object(total) = totals.entry(miner).or_insert_with(|| Value::Object(Map::new())) else { continue };
        for (name, value) in add {
            let merged = match (total.get(&name), name.as_str()) {
                (Some(had), "last_scored_at") => Value::from(had.as_f64().unwrap_or(0.0).max(value.as_f64().unwrap_or(0.0))),
                (Some(had), _) => Value::from(had.as_i64().unwrap_or(0) + value.as_i64().unwrap_or(0)),
                (None, _) => value,
            };
            total.insert(name, merged);
        }
    }
    Ok(())
}

pub fn task_series(conn: &Connection, bucket_s: i64, since: f64, until: f64, kind: &str, miner: Option<&str>) -> Result<BTreeMap<i64, Value>> {
    let sql = format!(
        "SELECT CAST(scored_at / ? AS INTEGER), COUNT(*), {VERDICT_SUMS}, SUM(returned), SUM(credited) FROM validations
         WHERE scored_at >= ? AND scored_at <= ? AND kind = ?{} GROUP BY 1",
        if miner.is_some() { " AND miner = ?" } else { "" }
    );
    let mut args: Vec<Sql> = vec![Sql::Integer(bucket_s), Sql::Real(since), Sql::Real(until), Sql::Text(kind.into())];
    args.extend(miner.map(|m| Sql::Text(m.into())));
    series(conn, &sql, args, &["tasks", "pass", "fail", "void", "returned", "credited"])
}

pub fn vote_series(conn: &Connection, bucket_s: i64, since: f64, until: f64, validator: &str) -> Result<BTreeMap<i64, Value>> {
    let sql = format!(
        "SELECT CAST(finalized_at / ? AS INTEGER), COUNT(*), {VERDICT_SUMS}, COALESCE(SUM(agreed = 1), 0), COALESCE(SUM(agreed = 0), 0) FROM votes
         WHERE finalized_at >= ? AND finalized_at <= ? AND validator = ? GROUP BY 1"
    );
    let args = vec![Sql::Integer(bucket_s), Sql::Real(since), Sql::Real(until), Sql::Text(validator.into())];
    series(conn, &sql, args, &["votes", "pass", "fail", "void", "agreed", "disagreed"])
}

fn series(conn: &Connection, sql: &str, args: Vec<Sql>, names: &[&str]) -> Result<BTreeMap<i64, Value>> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement.query_map(params_from_iter(args), |row| Ok((row.get::<_, i64>(0)?, Value::Object(object(row, names, 1)?))))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn reports(conn: &Connection, sql: &str, args: &[&dyn ToSql]) -> Result<Vec<Map<String, Value>>> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement.query_map(args, |row| object(row, &FIELDS, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Every finalized upload of a task, newest first.
pub fn uploads(conn: &Connection, task_id: &str) -> Result<Vec<Map<String, Value>>> {
    reports(conn, &format!("SELECT {} FROM validations WHERE task_id = ? ORDER BY id DESC", FIELDS.join(", ")), &[&task_id])
}

/// The per-URL detail of the task's last check: compressed, or plain JSON as kept before.
pub fn urls(conn: &Connection, task_id: &str) -> Result<Value> {
    let found: Option<Option<Value>> = conn
        .query_row("SELECT urls FROM validations WHERE task_id = ? ORDER BY id DESC LIMIT 1", [task_id], |row| {
            Ok(match row.get_ref(0)? {
                ValueRef::Text(text) => serde_json::from_slice(text).ok(),
                ValueRef::Blob(packed) => {
                    let mut json = Vec::new();
                    ZlibDecoder::new(packed).read_to_end(&mut json).ok().and_then(|_| serde_json::from_slice(&json).ok())
                }
                _ => None,
            })
        })
        .optional()?;
    Ok(found.flatten().unwrap_or_else(|| json!([])))
}

/// Clears the publish job kept with each verdict once it was finalized before `finalized_before`.
pub fn drop_publish_copies(conn: &Connection, finalized_before: f64, limit: i64) -> Result<i64> {
    let mark: f64 = conn.query_row("SELECT COALESCE((SELECT mark FROM retention WHERE name = 'final_verdicts'), 0)", [], |row| real(row, 0))?;
    let mark = mark as i64;
    // Rows are inserted as uploads finalize, so row order is time order.
    let mut statement = conn.prepare("SELECT rowid, finalized_at FROM final_verdicts WHERE rowid > ? ORDER BY rowid LIMIT ?")?;
    let rows = statement.query_map(params![mark, limit], |row| Ok((row.get::<_, i64>(0)?, real(row, 1)?)))?;
    let mut last = mark;
    for row in rows {
        let (rowid, at) = row?;
        if at >= finalized_before {
            break;
        }
        last = rowid;
    }
    if last > mark {
        conn.execute("UPDATE final_verdicts SET publish = NULL WHERE rowid > ? AND rowid <= ? AND publish IS NOT NULL", params![mark, last])?;
        conn.execute("INSERT INTO retention VALUES ('final_verdicts', ?) ON CONFLICT (name) DO UPDATE SET mark = excluded.mark", [last])?;
    }
    Ok(last - mark)
}

/// Clears the per-URL detail of up to `limit` verdicts older than `URL_DETAIL_DAYS`, oldest first, so one call never holds the writer long.
pub fn prune_urls(conn: &Connection, now: f64, limit: i64) -> Result<usize> {
    // Details older than a week were already cleared, so the first call starts there.
    let mark: f64 =
        conn.query_row("SELECT COALESCE((SELECT mark FROM retention WHERE name = 'url_details'), ?)", [now - 8.0 * 86_400.0], |row| real(row, 0))?;
    let mut statement = conn.prepare(
        "SELECT rowid, scored_at FROM validations INDEXED BY validations_scored
         WHERE scored_at >= ? AND scored_at < ? AND urls IS NOT NULL ORDER BY scored_at LIMIT ?",
    )?;
    let rows = statement.query_map(params![mark, now - URL_DETAIL_DAYS * 86_400.0, limit], |row| Ok((row.get::<_, i64>(0)?, real(row, 1)?)))?;
    let rows: Vec<(i64, f64)> = rows.collect::<rusqlite::Result<_>>()?;
    for (rowid, _) in &rows {
        conn.execute("UPDATE validations SET urls = NULL WHERE rowid = ?", [rowid])?;
    }
    if let Some((_, at)) = rows.last() {
        conn.execute("INSERT INTO retention VALUES ('url_details', ?) ON CONFLICT (name) DO UPDATE SET mark = excluded.mark", [at])?;
    }
    Ok(rows.len())
}

/// Filters for the tasks page and a miner's own verdicts.
#[derive(Default)]
pub struct TaskFilter {
    pub miner: Option<String>,
    pub validator: Option<String>,
    pub since: f64,
    pub before: Option<f64>,
    pub limit: i64,
    pub verdict: Option<String>,
    pub kind: Option<String>,
}

pub fn recent(conn: &Connection, filter: &TaskFilter) -> Result<Vec<Map<String, Value>>> {
    let mut clauses = vec!["scored_at >= ?".to_string()];
    let mut args: Vec<Sql> = vec![Sql::Real(filter.since)];
    if let Some(before) = filter.before {
        clauses.push("scored_at < ?".into());
        args.push(Sql::Real(before));
    }
    for (column, value) in [("miner", &filter.miner), ("validator", &filter.validator), ("verdict", &filter.verdict), ("kind", &filter.kind)] {
        if let Some(value) = value.as_ref().filter(|v| !v.is_empty()) {
            clauses.push(format!("{column} = ?"));
            args.push(Sql::Text(value.clone()));
        }
    }
    args.push(Sql::Integer(filter.limit));
    let mut statement =
        conn.prepare(&format!("SELECT {} FROM validations WHERE {} ORDER BY scored_at DESC LIMIT ?", FIELDS.join(", "), clauses.join(" AND ")))?;
    let rows = statement.query_map(params_from_iter(args), |row| object(row, &FIELDS, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn verdicts(conn: &Connection, miner: Option<&str>) -> Result<Map<String, Value>> {
    let mut counts = Map::new();
    counts.insert("pass".into(), 0.into());
    counts.insert("fail".into(), 0.into());
    let rows: Vec<(String, i64)> = match miner {
        None => {
            let mut statement = conn.prepare("SELECT verdict, SUM(n) FROM verdict_counts GROUP BY verdict")?;
            let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        }
        Some(miner) => {
            let mut statement = conn.prepare("SELECT verdict, n FROM verdict_counts WHERE miner = ?")?;
            let rows = statement.query_map([miner], |row| Ok((row.get(0)?, row.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        }
    };
    for (verdict, n) in rows {
        counts.insert(verdict, n.into());
    }
    Ok(counts)
}

pub fn judged_since(conn: &Connection, miner: &str, since: f64, kind: &str) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM validations WHERE miner = ? AND kind = ? AND scored_at >= ? AND verdict IN ('pass', 'fail')",
        params![miner, kind, since],
        |row| row.get(0),
    )?)
}

/// A passed upload taken back: its task, the credit it had, and the round it came from.
pub struct Taken {
    pub task_id: String,
    pub credited: i64,
    pub round_id: String,
}

/// A miner's passed uploads of one kind completed after `since`, taken back; only those no validator checked unless `checked_too`.
pub fn withdraw(conn: &Connection, miner: &str, kind: &str, since: f64, checked_too: bool) -> Result<Vec<Taken>> {
    let mut statement = conn.prepare(
        "SELECT id, task_id, credited, scored_at, round_id FROM validations
         WHERE miner = ? AND kind = ? AND verdict = 'pass' AND scored_at > ? AND COALESCE(completed_at, scored_at) > ? AND (? OR validator = '')",
    )?;
    let rows = statement.query_map(params![miner, kind, since, since, checked_too], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?, real(row, 3)?, row.get::<_, Option<String>>(4)?.unwrap_or_default()))
    })?;
    let rows: Vec<(i64, String, i64, f64, String)> = rows.collect::<rusqlite::Result<_>>()?;
    let rolled = rolled_through(conn)?;
    for (id, _, _, scored_at, _) in &rows {
        conn.execute("UPDATE validations SET verdict = ? WHERE id = ?", params![WITHDRAWN, id])?;
        if *id <= rolled {
            conn.execute("UPDATE miner_hours SET pass = pass - 1 WHERE miner = ? AND kind = ? AND hour = ?", params![miner, kind, hour_of(*scored_at)])?;
        }
    }
    if !rows.is_empty() {
        conn.execute("UPDATE verdict_counts SET n = n - ? WHERE miner = ? AND verdict = 'pass'", params![rows.len() as i64, miner])?;
        conn.execute(
            "INSERT INTO verdict_counts VALUES (?, ?, ?) ON CONFLICT (miner, verdict) DO UPDATE SET n = n + excluded.n",
            params![miner, WITHDRAWN, rows.len() as i64],
        )?;
    }
    Ok(rows.into_iter().map(|(_, task_id, credited, _, round_id)| Taken { task_id, credited, round_id }).collect())
}

/// The last verdict row counted into the hourly totals; at first, the last one before the kept window.
fn rolled_through(conn: &Connection) -> Result<i64> {
    if let Some(mark) = conn.query_row("SELECT mark FROM retention WHERE name = 'miner_hours'", [], |row| real(row, 0)).optional()? {
        return Ok(mark as i64);
    }
    let since = (hour_of(desearch::time::now()) - HOURS_KEPT) as f64 * 3600.0;
    let first: Option<i64> = conn
        .query_row("SELECT id FROM validations INDEXED BY validations_scored WHERE scored_at >= ? ORDER BY scored_at LIMIT 1", [since], |row| row.get(0))
        .optional()?;
    let start = first.map_or_else(|| conn.query_row("SELECT COALESCE(MAX(id), 0) FROM validations", [], |row| row.get(0)), |id| Ok(id - 1))?;
    conn.execute("INSERT INTO retention VALUES ('miner_hours', ?)", [start as f64])?;
    Ok(start)
}

/// Hours of verdict totals kept, a week and a day so a week's view always has whole hours.
const HOURS_KEPT: i64 = 24 * 8;

/// Counts up to `limit` more verdict rows into the hourly totals, in row order, and returns how many.
pub fn roll_up(conn: &Connection, limit: i64) -> Result<usize> {
    let rolled = rolled_through(conn)?;
    let mut statement =
        conn.prepare("SELECT id, miner, kind, verdict, returned, missing, credited, scored_at FROM validations WHERE id > ? ORDER BY id LIMIT ?")?;
    let rows = statement.query_map(params![rolled, limit], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            [row.get::<_, Option<i64>>(4)?.unwrap_or(0), row.get::<_, Option<i64>>(5)?.unwrap_or(0), row.get::<_, Option<i64>>(6)?.unwrap_or(0)],
            real(row, 7)?,
        ))
    })?;
    let rows: Vec<_> = rows.collect::<rusqlite::Result<_>>()?;
    let Some(last) = rows.last().map(|row| row.0) else { return Ok(0) };
    let mut totals: BTreeMap<(String, String, i64), [i64; 7]> = BTreeMap::new();
    let mut latest: BTreeMap<(String, String, i64), f64> = BTreeMap::new();
    for (_, miner, kind, verdict, [returned, missing, credited], scored_at) in &rows {
        let key = (miner.clone(), kind.clone(), hour_of(*scored_at));
        let total = totals.entry(key.clone()).or_default();
        let counts = [1, i64::from(verdict == "pass"), i64::from(verdict == "fail"), i64::from(verdict == "void"), *returned, *missing, *credited];
        for (sum, add) in total.iter_mut().zip(counts) {
            *sum += add;
        }
        let at = latest.entry(key).or_insert(*scored_at);
        *at = at.max(*scored_at);
    }
    for ((miner, kind, hour), [tasks, pass, fail, void, returned, missing, credited]) in &totals {
        conn.execute(
            "INSERT INTO miner_hours VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (miner, kind, hour) DO UPDATE SET tasks = tasks + excluded.tasks, pass = pass + excluded.pass, fail = fail + excluded.fail,
             void = void + excluded.void, returned = returned + excluded.returned, missing = missing + excluded.missing,
             credited = credited + excluded.credited, last_scored_at = MAX(COALESCE(last_scored_at, 0), excluded.last_scored_at)",
            params![miner, kind, hour, tasks, pass, fail, void, returned, missing, credited, latest[&(miner.clone(), kind.clone(), *hour)]],
        )?;
    }
    conn.execute("UPDATE retention SET mark = ? WHERE name = 'miner_hours'", [last as f64])?;
    Ok(rows.len())
}

/// Verdicts per hour from `since`'s hour to `until`, summed over miners or for one; the last hour only up to `until`.
pub fn hour_series(conn: &Connection, since: f64, until: f64, kind: &str, miner: Option<&str>) -> Result<BTreeMap<i64, Value>> {
    let last = hour_of(until);
    let sql = format!(
        "SELECT hour, SUM(tasks), SUM(pass), SUM(fail), SUM(void), SUM(returned), SUM(credited) FROM miner_hours WHERE hour >= ? AND hour < ? AND kind = ?{} GROUP BY hour",
        if miner.is_some() { " AND miner = ?" } else { "" }
    );
    let mut args: Vec<Sql> = vec![Sql::Integer(hour_of(since)), Sql::Integer(last), Sql::Text(kind.into())];
    args.extend(miner.map(|m| Sql::Text(m.into())));
    let mut hours = series(conn, &sql, args, &["tasks", "pass", "fail", "void", "returned", "credited"])?;
    hours.extend(task_series(conn, 3600, since.max(last as f64 * 3600.0), until, kind, miner)?);
    Ok(hours)
}

pub fn prune_hours(conn: &Connection, now: f64) -> Result<()> {
    conn.execute("DELETE FROM miner_hours WHERE hour < ?", [hour_of(now) - HOURS_KEPT])?;
    Ok(())
}

pub fn record_audit(conn: &Connection, agreed: &[String], disagreed: &[String]) -> Result<()> {
    for (hotkey, disagreement) in agreed.iter().map(|h| (h, 0)).chain(disagreed.iter().map(|h| (h, 1))) {
        conn.execute(
            "INSERT INTO validator_audits VALUES (?, 1, ?) ON CONFLICT (hotkey) DO UPDATE
             SET audits = audits + 1, disagreements = disagreements + excluded.disagreements",
            params![hotkey, disagreement],
        )?;
    }
    Ok(())
}

pub fn audit_standing(conn: &Connection) -> Result<BTreeMap<String, Value>> {
    let mut statement = conn.prepare("SELECT hotkey, audits, disagreements FROM validator_audits")?;
    let rows = statement.query_map([], |row| {
        let (audits, disagreements): (i64, i64) = (row.get(1)?, row.get(2)?);
        let excluded = audits >= MIN_AUDITS && disagreements as f64 / audits as f64 > MAX_DISAGREEMENT;
        Ok((row.get::<_, String>(0)?, json!({"audits": audits, "disagreements": disagreements, "excluded": excluded})))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn is_excluded(conn: &Connection, hotkey: &str) -> Result<bool> {
    let found: Option<(i64, i64)> =
        conn.query_row("SELECT audits, disagreements FROM validator_audits WHERE hotkey = ?", [hotkey], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
    Ok(found.is_some_and(|(audits, disagreements)| audits >= MIN_AUDITS && disagreements as f64 / audits as f64 > MAX_DISAGREEMENT))
}

/// The report of one finalized upload, as written to the pages bucket and the validations table.
pub fn build_report(task_id: &str, job: &Value, validator: &str, result: &Map<String, Value>, votes: &[Value], now: f64) -> Map<String, Value> {
    let mut report: Map<String, Value> = COUNTS.iter().map(|name| (name.to_string(), 0.into())).collect();
    report.extend(result.iter().filter(|(name, _)| *name != "urls").map(|(name, value)| (name.clone(), value.clone())));
    let fields = [
        ("task_id", Value::from(task_id)),
        ("kind", job.get("kind").cloned().unwrap_or_else(|| "crawl".into())),
        ("round_id", job["round_id"].clone()),
        ("miner", job["miner"].clone()),
        ("validator", validator.into()),
        ("upload_key", job["key"].clone()),
        ("upload_etag", job.get("etag").cloned().unwrap_or_else(|| "".into())),
        ("upload_bytes", job.get("size").cloned().unwrap_or_else(|| 0.into())),
        ("page_key", Value::Null),
        ("claimed_at", job.get("claimed_at").cloned().unwrap_or(Value::Null)),
        ("completed_at", job.get("completed_at").cloned().unwrap_or(Value::Null)),
        ("report_key", format!("reports/dt={}/task={task_id}.json", utc_day(now)).into()),
        ("scored_at", now.into()),
    ];
    report.extend(fields.into_iter().map(|(name, value)| (name.to_string(), value)));
    if votes.len() > 1 {
        let summary: Vec<Value> = votes
            .iter()
            .map(|vote| {
                json!({
                    "validator": vote["validator"],
                    "verdict": vote["verdict"],
                    "credited": vote["credited"],
                    "reason": vote["result"].get("reason").cloned().unwrap_or_else(|| "".into()),
                })
            })
            .collect();
        report.insert("votes".into(), summary.into());
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        conn
    }

    fn verdict(task_id: &str, scored_at: f64) -> Map<String, Value> {
        let job = json!({"round_id": "r", "miner": "m", "key": "k", "urls": ["https://a.example/"]});
        let result = json!({"verdict": "pass", "reason": "ok", "urls": details()});
        let mut report = build_report(task_id, &job, "v", result.as_object().unwrap(), &[], scored_at);
        report.insert("scored_at".into(), scored_at.into());
        report
    }

    fn details() -> Value {
        json!([{"url": "https://a.example/", "status": 200, "miner_snippet": "Story"}])
    }

    #[test]
    fn hourly_totals_match_the_verdicts_and_follow_take_backs_and_rows_written_elsewhere() {
        let conn = store();
        let base = 1_800_000_000.0 - 1_800_000_000.0 % 3600.0;
        for (i, at) in [base + 10.0, base + 20.0, base + 3_700.0].into_iter().enumerate() {
            let mut report = verdict(&format!("t{i}"), at);
            report.insert("returned".into(), 3.into());
            report.insert("credited".into(), 2.into());
            record(&conn, &report, None).unwrap();
        }
        let totals = miner_totals(&conn, base - 1.0, base + 7_200.0, "crawl").unwrap();
        assert_eq!(
            (totals["m"]["tasks"].clone(), totals["m"]["pass"].clone(), totals["m"]["returned"].clone(), totals["m"]["credited"].clone()),
            (json!(3), json!(3), json!(9), json!(6))
        );
        conn.execute(
            "INSERT INTO validations (task_id, kind, round_id, miner, validator, verdict, reason, upload_key, report_key, scored_at, report)
             VALUES ('older-copy', 'crawl', 'r', 'm', 'v', 'fail', 'x', 'k', 'rk', ?, '{}')",
            [base + 30.0],
        )
        .unwrap();
        assert_eq!(roll_up(&conn, 100).unwrap(), 1, "a row an older copy wrote is counted later");
        withdraw(&conn, "m", "crawl", 0.0, true).unwrap();
        let totals = miner_totals(&conn, base - 1.0, base + 7_200.0, "crawl").unwrap();
        assert_eq!((totals["m"]["tasks"].clone(), totals["m"]["pass"].clone(), totals["m"]["fail"].clone()), (json!(4), json!(0), json!(1)));
        let hours = hour_series(&conn, base, base + 7_199.0, "crawl", None).unwrap();
        assert_eq!((hours[&hour_of(base)]["tasks"].clone(), hours[&(hour_of(base) + 1)]["tasks"].clone()), (json!(3), json!(1)));
    }

    #[test]
    fn totals_count_exactly_the_verdicts_scored_in_the_window_not_whole_hours_around_it() {
        let conn = store();
        let base = 1_800_000_000.0 - 1_800_000_000.0 % 3600.0;
        for (i, at) in [base + 10.0, base + 30.0, base + 3_700.0, base + 7_210.0, base + 7_300.0].into_iter().enumerate() {
            record(&conn, &verdict(&format!("t{i}"), at), None).unwrap();
        }
        let totals = miner_totals(&conn, base + 20.0, base + 7_250.0, "crawl").unwrap();
        assert_eq!((totals["m"]["tasks"].clone(), totals["m"]["last_scored_at"].clone()), (json!(3), json!(base + 7_210.0)));
        assert_eq!(miner_totals(&conn, base + 20.0, base + 40.0, "crawl").unwrap()["m"]["tasks"], json!(1), "inside one hour");
        let hours = hour_series(&conn, base, base + 7_250.0, "crawl", Some("m")).unwrap();
        assert_eq!(
            (hours[&hour_of(base)]["tasks"].clone(), hours[&(hour_of(base) + 1)]["tasks"].clone(), hours[&(hour_of(base) + 2)]["tasks"].clone()),
            (json!(2), json!(1), json!(1)),
            "the last hour only up to the end of the window"
        );
    }

    #[test]
    fn the_per_url_detail_is_kept_beside_the_verdict_not_in_the_report() {
        let conn = store();
        let report = verdict("t1", 1_800_000_000.0);
        record(&conn, &report, pack_details(Some(&details())).as_deref()).unwrap();
        assert!(!report.contains_key("urls"));
        assert_eq!(urls(&conn, "t1").unwrap(), details());
        assert_eq!(urls(&conn, "missing").unwrap(), json!([]));
        record(&conn, &verdict("plain", 1_800_000_000.0), None).unwrap();
        conn.execute("UPDATE validations SET urls = ? WHERE task_id = 'plain'", [details().to_string()]).unwrap();
        assert_eq!(urls(&conn, "plain").unwrap(), details(), "kept as JSON text before it was compressed");
    }

    #[test]
    fn the_detail_is_pruned_after_two_days_and_the_verdict_stays() {
        let conn = store();
        let now = 1_800_000_000.0;
        record(&conn, &verdict("old", now - URL_DETAIL_DAYS * 86_400.0 - 60.0), pack_details(Some(&details())).as_deref()).unwrap();
        record(&conn, &verdict("new", now), pack_details(Some(&details())).as_deref()).unwrap();
        assert_eq!(prune_urls(&conn, now, 50).unwrap(), 1);
        assert_eq!((urls(&conn, "old").unwrap(), urls(&conn, "new").unwrap()), (json!([]), details()));
        assert_eq!(uploads(&conn, "old").unwrap()[0]["verdict"], "pass");
        let filter = TaskFilter { limit: 50, ..TaskFilter::default() };
        assert_eq!(recent(&conn, &filter).unwrap().iter().map(|t| t["task_id"].clone()).collect::<Vec<_>>(), [json!("new"), json!("old")]);
    }

    #[test]
    fn details_are_pruned_a_few_at_a_time_from_where_the_last_call_stopped() {
        let conn = store();
        let now = 1_800_000_000.0;
        let old = now - URL_DETAIL_DAYS * 86_400.0;
        for (task_id, at) in [("a", old - 300.0), ("b", old - 200.0), ("c", old - 100.0), ("new", now)] {
            record(&conn, &verdict(task_id, at), pack_details(Some(&details())).as_deref()).unwrap();
        }
        assert_eq!(prune_urls(&conn, now, 2).unwrap(), 2);
        assert_eq!((urls(&conn, "a").unwrap(), urls(&conn, "b").unwrap(), urls(&conn, "c").unwrap()), (json!([]), json!([]), details()));
        assert_eq!(prune_urls(&conn, now, 2).unwrap(), 1);
        assert_eq!(prune_urls(&conn, now, 2).unwrap(), 0);
        assert_eq!(urls(&conn, "new").unwrap(), details());
    }

    #[test]
    fn a_verdict_drops_its_publish_job_a_day_after_it_was_finalized() {
        let conn = store();
        let now = 1_800_000_000.0;
        for (task_id, at) in [("old1", now - 2.0 * 86_400.0), ("old2", now - 2.0 * 86_400.0), ("new1", now)] {
            let job = json!({"task_id": task_id, "urls": ["https://example.com/a"]});
            finalize(&conn, task_id, &format!("submitted/{task_id}"), "pass", 100, Some(&job), at).unwrap();
        }
        drop_publish_copies(&conn, now - 86_400.0, 100).unwrap();
        let kept = |task_id: &str| final_verdict(&conn, task_id, &format!("submitted/{task_id}")).unwrap().unwrap();
        assert!(kept("old1").publish.is_none() && kept("old2").publish.is_none());
        let copy = kept("new1").publish.unwrap();
        assert_eq!((copy["task_id"].clone(), copy.get("urls")), (json!("new1"), None), "kept without the URLs its job holds");
        assert_eq!(kept("old1").verdict, "pass");
    }
}

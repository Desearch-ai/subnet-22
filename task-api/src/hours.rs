//! Hourly tallies the log pages read instead of rows: claims from the signed log, verdicts by reason and how long they took, and every take-back.

use std::collections::BTreeMap;

use anyhow::Result;
use rusqlite::types::Value as Sql;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::budgets::{hour_of, HOUR, SHARE_WINDOW_H};
use crate::db::{optional_real, real};

/// A week and a day, as the verdict totals keep.
pub const HOURS_KEPT: i64 = 24 * 8;
/// A first start counts back a day and an hour, so a day's view is whole within minutes; the rest of the week fills as it passes.
const FIRST_COUNTED_H: i64 = SHARE_WINDOW_H + 1;
/// Causes of a reclaim the miner let happen; the others are tasks sent back after a verdict.
pub const LAPSED: [&str; 2] = ["expired", "abandoned"];
const CLAIMS_MARK: &str = "claim_hours";
const CHECKS_MARK: &str = "check_hours";
const LONGEST_WAIT_S: f64 = 86_400.0;
pub const QUANTILES: [(&str, f64); 2] = [("p50", 0.5), ("p90", 0.9)];

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS retention (name TEXT PRIMARY KEY, mark REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS claim_hours (
            hour    INTEGER NOT NULL,
            kind    TEXT NOT NULL,
            outcome TEXT NOT NULL,
            code    TEXT NOT NULL,
            hotkey  TEXT NOT NULL,
            n       INTEGER NOT NULL,
            PRIMARY KEY (hour, kind, outcome, code, hotkey)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS claim_hours_hotkey ON claim_hours (hotkey, hour);
        CREATE TABLE IF NOT EXISTS check_hours (
            hour    INTEGER NOT NULL,
            kind    TEXT NOT NULL,
            checked INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            reason  TEXT NOT NULL,
            n       INTEGER NOT NULL,
            PRIMARY KEY (hour, kind, checked, verdict, reason)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS wait_hours (
            hour    INTEGER NOT NULL,
            kind    TEXT NOT NULL,
            checked INTEGER NOT NULL,
            bucket  INTEGER NOT NULL,
            n       INTEGER NOT NULL,
            PRIMARY KEY (hour, kind, checked, bucket)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS take_backs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            miner    TEXT NOT NULL,
            kind     TEXT NOT NULL,
            reason   TEXT NOT NULL,
            tasks    INTEGER NOT NULL,
            credited INTEGER NOT NULL,
            at       REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS take_backs_miner ON take_backs (miner, id);
        CREATE INDEX IF NOT EXISTS take_backs_at ON take_backs (at);
        ",
    )?;
    Ok(())
}

fn mark(conn: &Connection, name: &str) -> Result<Option<i64>> {
    Ok(conn.query_row("SELECT mark FROM retention WHERE name = ?", [name], |row| real(row, 0)).optional()?.map(|mark| mark as i64))
}

fn set_mark(conn: &Connection, name: &str, id: i64) -> Result<()> {
    conn.execute("INSERT INTO retention VALUES (?, ?) ON CONFLICT (name) DO UPDATE SET mark = excluded.mark", params![name, id as f64])?;
    Ok(())
}

fn first_counted(now: f64) -> f64 {
    (hour_of(now) - FIRST_COUNTED_H) as f64 * HOUR
}

/// The last entry served before `at`, found by halving: entries are written in the order they are served.
fn last_entry_before(conn: &Connection, at: f64) -> Result<i64> {
    // Apart, so each is one step down the primary key; MIN and MAX in one SELECT read every row.
    let (mut low, mut high): (i64, i64) =
        conn.query_row("SELECT COALESCE((SELECT MIN(id) FROM entries), 1), COALESCE((SELECT MAX(id) FROM entries), 0)", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
    let served = |id: i64| -> Result<f64> { Ok(conn.query_row("SELECT served_at FROM entries WHERE id >= ? ORDER BY id LIMIT 1", [id], |row| real(row, 0))?) };
    high += 1;
    while low < high {
        let middle = low + (high - low) / 2;
        if served(middle)? < at {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(low - 1)
}

fn last_verdict_before(conn: &Connection, at: f64) -> Result<i64> {
    let first: Option<i64> = conn
        .query_row("SELECT id FROM validations INDEXED BY validations_scored WHERE scored_at >= ? ORDER BY scored_at LIMIT 1", [at], |row| row.get(0))
        .optional()?;
    Ok(match first {
        Some(id) => id - 1,
        None => conn.query_row("SELECT COALESCE(MAX(id), 0) FROM validations", [], |row| row.get(0))?,
    })
}

fn last_of(conn: &Connection, table: &str, after: i64, limit: i64) -> Result<Option<i64>> {
    Ok(conn.query_row(&format!("SELECT MAX(id) FROM (SELECT id FROM {table} WHERE id > ? ORDER BY id LIMIT ?)"), params![after, limit], |row| row.get(0))?)
}

/// Counts up to `limit` more signed log entries into the claim tallies, in row order.
pub fn roll_up_claims(conn: &Connection, limit: i64, now: f64) -> Result<i64> {
    let rolled = match mark(conn, CLAIMS_MARK)? {
        Some(rolled) => rolled,
        None => last_entry_before(conn, first_counted(now))?,
    };
    let Some(last) = last_of(conn, "entries", rolled, limit)? else {
        set_mark(conn, CLAIMS_MARK, rolled)?;
        return Ok(0);
    };
    conn.execute(
        "INSERT INTO claim_hours (hour, kind, outcome, code, hotkey, n)
         SELECT CAST(e.served_at / 3600 AS INTEGER), COALESCE(r.kind, ''), e.outcome, COALESCE(json_extract(e.refusal, '$.code'), e.cause, ''), e.hotkey, COUNT(*)
         FROM entries e LEFT JOIN rounds r ON r.round_id = e.round_id
         WHERE e.id > ? AND e.id <= ?
         GROUP BY 1, 2, 3, 4, 5
         ON CONFLICT (hour, kind, outcome, code, hotkey) DO UPDATE SET n = n + excluded.n",
        params![rolled, last],
    )?;
    set_mark(conn, CLAIMS_MARK, last)?;
    Ok(last - rolled)
}

/// The upper end of the wait's bucket in seconds: 15 s steps to 5 min, then 1 min to 30, 10 min to 2 h, hours to a day.
pub fn bucket(wait_s: f64) -> i64 {
    let wait = wait_s.clamp(0.0, LONGEST_WAIT_S);
    ((wait / step(wait)).ceil() * step(wait)) as i64
}

fn step(wait: f64) -> f64 {
    match wait {
        w if w <= 300.0 => 15.0,
        w if w <= 1_800.0 => 60.0,
        w if w <= 7_200.0 => 600.0,
        _ => HOUR,
    }
}

/// Counts up to `limit` more verdict rows into the reason and wait tallies, in row order.
pub fn roll_up_checks(conn: &Connection, limit: i64, now: f64) -> Result<i64> {
    let rolled = match mark(conn, CHECKS_MARK)? {
        Some(rolled) => rolled,
        None => last_verdict_before(conn, first_counted(now))?,
    };
    let Some(last) = last_of(conn, "validations", rolled, limit)? else {
        set_mark(conn, CHECKS_MARK, rolled)?;
        return Ok(0);
    };
    let mut reasons: BTreeMap<(i64, String, bool, String, String), i64> = BTreeMap::new();
    let mut waits: BTreeMap<(i64, String, bool, i64), i64> = BTreeMap::new();
    let mut statement = conn.prepare("SELECT kind, validator != '', verdict, reason, scored_at, completed_at FROM validations WHERE id > ? AND id <= ?")?;
    let mut rows = statement.query(params![rolled, last])?;
    while let Some(row) = rows.next()? {
        let (kind, checked, scored_at): (String, bool, f64) = (row.get(0)?, row.get(1)?, real(row, 4)?);
        let verdict: String = row.get(2)?;
        // A take-back rewrites a pass later; it is counted when it happens.
        let verdict = if verdict == crate::validations::WITHDRAWN { "pass".to_string() } else { verdict };
        let hour = hour_of(scored_at);
        *reasons.entry((hour, kind.clone(), checked, verdict, row.get(3)?)).or_default() += 1;
        if let Some(completed_at) = optional_real(row, 5)? {
            *waits.entry((hour, kind, checked, bucket(scored_at - completed_at))).or_default() += 1;
        }
    }
    for ((hour, kind, checked, verdict, reason), n) in reasons {
        conn.execute(
            "INSERT INTO check_hours VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (hour, kind, checked, verdict, reason) DO UPDATE SET n = n + excluded.n",
            params![hour, kind, checked, verdict, reason, n],
        )?;
    }
    for ((hour, kind, checked, bucket), n) in waits {
        conn.execute(
            "INSERT INTO wait_hours VALUES (?, ?, ?, ?, ?) ON CONFLICT (hour, kind, checked, bucket) DO UPDATE SET n = n + excluded.n",
            params![hour, kind, checked, bucket, n],
        )?;
    }
    set_mark(conn, CHECKS_MARK, last)?;
    Ok(last - rolled)
}

pub fn record_take_back(conn: &Connection, miner: &str, kind: &str, reason: &str, tasks: i64, credited: i64, now: f64) -> Result<()> {
    conn.execute(
        "INSERT INTO take_backs (miner, kind, reason, tasks, credited, at) VALUES (?, ?, ?, ?, ?, ?)",
        params![miner, kind, reason, tasks, credited, now],
    )?;
    Ok(())
}

pub fn prune(conn: &Connection, now: f64) -> Result<()> {
    let oldest = hour_of(now) - HOURS_KEPT;
    for table in ["claim_hours", "check_hours", "wait_hours"] {
        conn.execute(&format!("DELETE FROM {table} WHERE hour < ?"), [oldest])?;
    }
    conn.execute("DELETE FROM take_backs WHERE at < ?", [oldest as f64 * HOUR])?;
    Ok(())
}

fn hour_range(first: i64, last: i64, kind: &str) -> Vec<Sql> {
    vec![Sql::Integer(first), Sql::Integer(last), Sql::Text(kind.into())]
}

fn add(counts: &mut Map<String, Value>, name: &str, n: i64) {
    let had = counts.get(name).and_then(Value::as_i64).unwrap_or(0);
    counts.insert(name.into(), (had + n).into());
}

pub fn no_claims() -> Map<String, Value> {
    let mut counts: Map<String, Value> =
        ["issued", "completed", "lapsed", "requeued", "dropped"].into_iter().map(|name| (name.to_string(), Value::from(0))).collect();
    counts.insert("refused".into(), json!({}));
    counts
}

/// Each hour's claims of one kind from `first` to `last`, for every miner or one: issued, completed, lapsed, sent back, dropped, and refusals by code.
pub fn claims(conn: &Connection, first: i64, last: i64, kind: &str, miner: Option<&str>) -> Result<BTreeMap<i64, Map<String, Value>>> {
    let mut args = hour_range(first, last, kind);
    let sql = match miner {
        Some(miner) => {
            args.push(Sql::Text(miner.into()));
            "SELECT hour, outcome, code, n, 0 FROM claim_hours INDEXED BY claim_hours_hotkey WHERE hour >= ? AND hour <= ? AND kind = ? AND hotkey = ?"
        }
        None => "SELECT hour, outcome, code, SUM(n), COUNT(*) FROM claim_hours WHERE hour >= ? AND hour <= ? AND kind = ? GROUP BY hour, outcome, code",
    };
    let mut statement = conn.prepare(sql)?;
    let mut rows = statement.query(params_from_iter(args))?;
    let mut hours: BTreeMap<i64, Map<String, Value>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let (hour, outcome, code, n, hotkeys): (i64, String, String, i64, i64) = (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?);
        let counts = hours.entry(hour).or_insert_with(no_claims);
        match outcome.as_str() {
            "refused" => {
                if let Some(Value::Object(refused)) = counts.get_mut("refused") {
                    add(refused, &code, n);
                }
            }
            "reclaimed" if LAPSED.contains(&code.as_str()) => add(counts, "lapsed", n),
            "reclaimed" => add(counts, "requeued", n),
            "issued" => {
                add(counts, "issued", n);
                if miner.is_none() {
                    counts.insert("miners".into(), hotkeys.into());
                }
            }
            "completed" | "dropped" => add(counts, &outcome, n),
            _ => {}
        }
    }
    Ok(hours)
}

/// Claims summed over the hours, refusals by code.
pub fn total_claims(hours: &BTreeMap<i64, Map<String, Value>>) -> Map<String, Value> {
    let mut total = no_claims();
    for counts in hours.values() {
        for (name, value) in counts {
            match (name.as_str(), value) {
                ("miners", _) => {}
                ("refused", Value::Object(refused)) => {
                    if let Some(Value::Object(sum)) = total.get_mut("refused") {
                        for (code, n) in refused {
                            add(sum, code, n.as_i64().unwrap_or(0));
                        }
                    }
                }
                (name, n) => add(&mut total, name, n.as_i64().unwrap_or(0)),
            }
        }
    }
    total
}

/// Verdicts of one kind scored from `first` to `last` by verdict, reason and whether a validator checked them, most first.
pub fn reasons(conn: &Connection, first: i64, last: i64, kind: &str) -> Result<Vec<Value>> {
    let mut statement = conn.prepare(
        "SELECT verdict, reason, checked, SUM(n) FROM check_hours WHERE hour >= ? AND hour <= ? AND kind = ? GROUP BY verdict, reason, checked ORDER BY SUM(n) DESC, verdict, reason",
    )?;
    let rows = statement.query_map(params_from_iter(hour_range(first, last, kind)), |row| {
        Ok(json!({"verdict": row.get::<_, String>(0)?, "reason": row.get::<_, String>(1)?, "checked": row.get::<_, bool>(2)?, "tasks": row.get::<_, i64>(3)?}))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Uploads a validator checked, per hour.
pub fn checked(conn: &Connection, first: i64, last: i64, kind: &str) -> Result<BTreeMap<i64, i64>> {
    let mut statement = conn.prepare("SELECT hour, SUM(n) FROM check_hours WHERE hour >= ? AND hour <= ? AND kind = ? AND checked = 1 GROUP BY hour")?;
    let rows = statement.query_map(params_from_iter(hour_range(first, last, kind)), |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The value below which a share `q` of the tallied waits fall, spread evenly within its bucket.
pub fn quantile(buckets: &BTreeMap<i64, i64>, q: f64) -> Option<f64> {
    let total: i64 = buckets.values().sum();
    if total == 0 {
        return None;
    }
    let rank = q * total as f64;
    let mut below = 0;
    for (&upper, &n) in buckets {
        if (below + n) as f64 >= rank && n > 0 {
            let lower = (upper as f64 - step(upper as f64 - 1.0)).max(0.0);
            return Some((lower + (upper as f64 - lower) * (rank - below as f64) / n as f64).round());
        }
        below += n;
    }
    buckets.keys().last().map(|upper| *upper as f64)
}

fn summary(buckets: Option<&BTreeMap<i64, i64>>) -> Value {
    let Some(buckets) = buckets.filter(|b| b.values().sum::<i64>() > 0) else { return Value::Null };
    let mut summary = Map::new();
    summary.insert("tasks".into(), buckets.values().sum::<i64>().into());
    for (name, q) in QUANTILES {
        summary.insert(name.into(), quantile(buckets, q).map_or(Value::Null, Value::from));
    }
    Value::Object(summary)
}

fn waits_by(conn: &Connection, first: i64, last: i64, kind: &str) -> Result<BTreeMap<(i64, bool), BTreeMap<i64, i64>>> {
    let mut statement = conn.prepare("SELECT hour, checked, bucket, n FROM wait_hours WHERE hour >= ? AND hour <= ? AND kind = ?")?;
    let mut rows = statement.query(params_from_iter(hour_range(first, last, kind)))?;
    let mut found: BTreeMap<(i64, bool), BTreeMap<i64, i64>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        *found.entry((row.get(0)?, row.get(1)?)).or_default().entry(row.get(2)?).or_default() += row.get::<_, i64>(3)?;
    }
    Ok(found)
}

fn by_check(unchecked: Option<&BTreeMap<i64, i64>>, checked: Option<&BTreeMap<i64, i64>>) -> Value {
    json!({"unchecked": summary(unchecked), "checked": summary(checked)})
}

/// Seconds from upload to final verdict per hour, apart for uploads a validator checked: how many, the median and the 90th percentile.
pub fn waits(conn: &Connection, first: i64, last: i64, kind: &str) -> Result<BTreeMap<i64, Value>> {
    let found = waits_by(conn, first, last, kind)?;
    Ok((first..=last).map(|hour| (hour, by_check(found.get(&(hour, false)), found.get(&(hour, true))))).collect())
}

/// The same over all the hours together.
pub fn total_waits(conn: &Connection, first: i64, last: i64, kind: &str) -> Result<Value> {
    let mut merged: BTreeMap<bool, BTreeMap<i64, i64>> = BTreeMap::new();
    for ((_, checked), buckets) in waits_by(conn, first, last, kind)? {
        let into = merged.entry(checked).or_default();
        for (bucket, n) in buckets {
            *into.entry(bucket).or_default() += n;
        }
    }
    Ok(by_check(merged.get(&false), merged.get(&true)))
}

/// Passes of one kind taken back since `since`: how many, from how many miners, and the credit they had.
pub fn taken_back(conn: &Connection, since: f64, kind: &str) -> Result<Value> {
    Ok(conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(tasks), 0), COUNT(DISTINCT miner), COALESCE(SUM(credited), 0) FROM take_backs WHERE at >= ? AND kind = ?",
        params![since, kind],
        |row| {
            Ok(json!({"take_backs": row.get::<_, i64>(0)?, "tasks": row.get::<_, i64>(1)?, "miners": row.get::<_, i64>(2)?, "credited": row.get::<_, i64>(3)?}))
        },
    )?)
}

/// One miner's take-backs of both kinds, newest first.
pub fn take_backs_of(conn: &Connection, miner: &str, limit: i64) -> Result<Vec<Value>> {
    let mut statement = conn.prepare("SELECT kind, reason, tasks, credited, at FROM take_backs WHERE miner = ? ORDER BY id DESC LIMIT ?")?;
    let rows = statement.query_map(params![miner, limit], |row| {
        Ok(json!({"kind": row.get::<_, String>(0)?, "reason": row.get::<_, String>(1)?, "tasks": row.get::<_, i64>(2)?, "credited": row.get::<_, i64>(3)?, "at": real(row, 4)?}))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roundlog::{self, Receipt};
    use crate::{roundstore, validations};

    const NOW: f64 = 1_800_000_000.0;

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        roundlog::create(&conn).unwrap();
        roundstore::create(&conn).unwrap();
        validations::create(&conn).unwrap();
        create(&conn).unwrap();
        for (round_id, kind) in [("rc", "crawl"), ("re", "embed")] {
            conn.execute(
                "INSERT INTO rounds (round_id, kind, manifest_hash, seed_block, opened_at, batches) VALUES (?, ?, 'h', 0, 0, '[]')",
                params![round_id, kind],
            )
            .unwrap();
        }
        conn
    }

    fn log(conn: &Connection, round_id: &str, hotkey: &str, outcome: &'static str, at: f64, refusal: Option<&str>, cause: Option<&str>) {
        let receipt = Receipt {
            round_id: round_id.into(),
            hotkey: hotkey.into(),
            requested_at: at,
            outcome,
            refusal: refusal.map(|code| json!({"code": code, "inputs": {}})),
            cause: cause.map(String::from),
            ..Receipt::default()
        };
        roundlog::record(conn, &receipt, "sig", at).unwrap();
    }

    fn verdict(conn: &Connection, kind: &str, validator: &str, verdict: &str, reason: &str, completed_at: f64, scored_at: f64) {
        let job = json!({"round_id": "rc", "miner": "m", "key": "k", "kind": kind, "completed_at": completed_at});
        let result = json!({"verdict": verdict, "reason": reason});
        let report = validations::build_report("t", &job, validator, result.as_object().unwrap(), &[], scored_at);
        validations::record(conn, &report, None).unwrap();
    }

    #[test]
    fn claims_are_tallied_by_hour_kind_and_code_and_only_once() {
        let conn = store();
        let hour = hour_of(NOW);
        let at = hour as f64 * HOUR;
        log(&conn, "rc", "a", "refused", at - FIRST_COUNTED_H as f64 * HOUR - 10.0, Some("QUEUE_EMPTY"), None);
        log(&conn, "rc", "a", "issued", at + 1.0, None, None);
        log(&conn, "rc", "b", "issued", at + 2.0, None, None);
        log(&conn, "rc", "a", "completed", at + 3.0, None, None);
        log(&conn, "rc", "b", "reclaimed", at + 4.0, None, Some("expired"));
        log(&conn, "rc", "b", "reclaimed", at + 5.0, None, Some("fail"));
        log(&conn, "rc", "c", "refused", at + 6.0, Some("QUEUE_EMPTY"), None);
        log(&conn, "rc", "c", "refused", at + 66.0, Some("QUEUE_EMPTY"), None);
        log(&conn, "re", "c", "refused", at + 7.0, Some("NO_CAPACITY"), None);
        log(&conn, "", "c", "refused", at + 8.0, Some("QUEUE_EMPTY"), None);
        log(&conn, "rc", "a", "issued", at + HOUR + 1.0, None, None);

        assert_eq!(roll_up_claims(&conn, 4, NOW).unwrap(), 4, "a first start skips entries from before the day it counts");
        assert_eq!(roll_up_claims(&conn, 100, NOW).unwrap(), 6);
        assert_eq!(roll_up_claims(&conn, 100, NOW).unwrap(), 0);

        let crawl = claims(&conn, hour, hour + 1, "crawl", None).unwrap();
        assert_eq!(
            Value::Object(crawl[&hour].clone()),
            json!({"issued": 2, "completed": 1, "lapsed": 1, "requeued": 1, "dropped": 0, "refused": {"QUEUE_EMPTY": 2}, "miners": 2})
        );
        assert_eq!(crawl[&(hour + 1)]["issued"], 1);
        assert_eq!(claims(&conn, hour, hour, "embed", None).unwrap()[&hour]["refused"], json!({"NO_CAPACITY": 1}));
        let mine = claims(&conn, hour, hour + 1, "crawl", Some("a")).unwrap();
        assert_eq!((mine[&hour]["issued"].clone(), mine[&hour]["completed"].clone(), mine[&hour].get("miners")), (json!(1), json!(1), None));
        let total = total_claims(&crawl);
        assert_eq!((total["issued"].clone(), total["refused"].clone(), total.get("miners")), (json!(3), json!({"QUEUE_EMPTY": 2}), None));
    }

    #[test]
    fn verdicts_are_tallied_by_reason_with_how_long_each_took_and_a_take_back_counts_as_its_pass() {
        let conn = store();
        let hour = hour_of(NOW);
        let at = hour as f64 * HOUR + 600.0;
        verdict(&conn, "crawl", "", "pass", "unchecked", at - 116.0, at);
        verdict(&conn, "crawl", "", "pass", "unchecked", at - 119.0, at);
        verdict(&conn, "crawl", "v", "pass", "ok", at - 140.0, at);
        verdict(&conn, "crawl", "v", "fail", "content_mismatch", at - 700.0, at);
        verdict(&conn, "embed", "v", "void", "validators_disagree", at - 30.0, at);
        validations::withdraw(&conn, "m", "crawl", 0.0, false).unwrap();

        assert_eq!(roll_up_checks(&conn, 100, NOW).unwrap(), 5);
        assert_eq!(roll_up_checks(&conn, 100, NOW).unwrap(), 0);
        let found = reasons(&conn, hour, hour, "crawl").unwrap();
        assert_eq!(
            found,
            [
                json!({"verdict": "pass", "reason": "unchecked", "checked": false, "tasks": 2}),
                json!({"verdict": "fail", "reason": "content_mismatch", "checked": true, "tasks": 1}),
                json!({"verdict": "pass", "reason": "ok", "checked": true, "tasks": 1}),
            ]
        );
        assert_eq!(checked(&conn, hour, hour, "crawl").unwrap()[&hour], 2);
        let waited = &waits(&conn, hour, hour, "crawl").unwrap()[&hour];
        assert_eq!(waited["unchecked"], json!({"tasks": 2, "p50": 113.0, "p90": 119.0}));
        assert_eq!((waited["checked"]["tasks"].clone(), waited["checked"]["p90"].clone()), (json!(2), json!(708.0)));
        assert_eq!(total_waits(&conn, hour, hour, "embed").unwrap()["unchecked"], Value::Null);
    }

    #[test]
    fn waits_fall_in_buckets_that_widen_as_they_grow() {
        assert_eq!([0.0, 1.0, 15.0, 16.0, 300.0, 301.0, 1_801.0, 7_201.0, 1e9].map(bucket), [0, 15, 15, 30, 300, 360, 2_400, 10_800, 86_400]);
        let one_bucket = BTreeMap::from([(120, 4)]);
        assert_eq!((quantile(&one_bucket, 0.5), quantile(&one_bucket, 1.0)), (Some(113.0), Some(120.0)));
        assert_eq!(quantile(&BTreeMap::from([(15, 9), (2_400, 1)]), 0.9), Some(15.0));
        assert_eq!(quantile(&BTreeMap::new(), 0.5), None);
    }

    #[test]
    fn take_backs_are_kept_a_week_and_a_day() {
        let conn = store();
        record_take_back(&conn, "m", "crawl", "check_failed", 12, 11_000, NOW - HOURS_KEPT as f64 * HOUR - HOUR).unwrap();
        record_take_back(&conn, "m", "crawl", "check_failed", 3, 2_900, NOW - 60.0).unwrap();
        record_take_back(&conn, "n", "embed", "repeated_fails", 1, 5, NOW).unwrap();
        assert_eq!(taken_back(&conn, NOW - 86_400.0, "crawl").unwrap(), json!({"take_backs": 1, "tasks": 3, "miners": 1, "credited": 2_900}));
        prune(&conn, NOW).unwrap();
        let kept = take_backs_of(&conn, "m", 10).unwrap();
        assert_eq!(kept, [json!({"kind": "crawl", "reason": "check_failed", "tasks": 3, "credited": 2_900, "at": NOW - 60.0})]);
    }
}

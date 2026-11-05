//! The accounts as decisions read them, dumped in a fixed order, so the API and a replay can show they hold the same by one hash.
//!
//! A replay starts from the first snapshot in the log and, at every later one, compares the digest of its own accounts with the API's.
//! A crawl round's loadability, neutral hosts and the uploads waiting for its close ride along so a replay that joins late judges
//! the round's uploads alike; they are loaded, not compared, since the API drops them with the round and a replay keeps no rounds.

use anyhow::Result;
use rusqlite::{params_from_iter, types::Value as Sql, Connection};
use serde_json::{json, Map, Value};

use crate::credit::PENALTY_WINDOW_S;
use crate::db::json_of;
use crate::proofs::sha256;
use crate::py::canonical;

/// Each table, the columns decisions read, and which rows: every row of a table the API prunes itself (prunes are logged and
/// replayed), of verdicts only those a take-back, a strike or a lockout can still reach, and of final verdicts those of the last
/// day, which a validation still open can read back and `/v1/tasks` serves.
const TABLES: [(&str, &str, &str); 14] = [
    ("miners", "hotkey, pool, budget, verified", "1"),
    ("credits", "pool, hotkey, hour, amount", "1"),
    ("coverage", "hotkey, hour, assigned, returned", "1"),
    ("strikes", "hotkey, pool, reason, task_id, at", "1"),
    ("lockouts", "hotkey, pool, until, reason", "1"),
    ("lockout_history", "hotkey, pool, at", "1"),
    ("checks", "hotkey, kind, task_id, at, upload_at, passed, errors_judged, errors_unconfirmed", "1"),
    ("check_states", "hotkey, kind, recheck_left, last_pass_at", "1"),
    ("validator_audits", "hotkey, audits, disagreements", "1"),
    // Outages a lapse's forgiveness reads, and the wrong reports that exclude a validator.
    ("outages", "cause, start, end, start_block, end_block, detail", "1"),
    ("wrong_reports", "validator, task_id, why, at", "1"),
    // Validators' storage witnesses, which forgive lapses wherever a quorum of them saw storage fail.
    ("storage_witnesses", "validator, at, block, detail", "1"),
    (
        "validations",
        "task_id, kind, round_id, miner, validator, verdict, credited, completed_at, scored_at, upload_key",
        "verdict != 'void' AND (scored_at >= ?1 OR (verdict = 'pass' AND scored_at > COALESCE((SELECT c.last_pass_at FROM check_states c WHERE c.hotkey = validations.miner AND c.kind = validations.kind), 0)))",
    ),
    // The verdict an upload was finalized with; the publish copy the janitor clears is left out.
    ("final_verdicts", "task_id, upload_key, verdict, credited, finalized_at", "finalized_at >= ?1"),
];

/// What crawl rounds' checked uploads built up, as the rules judge the next upload of the round, and the uploads whose accounts
/// wait for their round to close.
const ROUNDS: [(&str, &str, &str); 3] = [
    ("round_loads", "round_id, sampled, loaded", "1"),
    ("round_hosts", "round_id, host, uploads, failed", "1"),
    ("round_uploads", "seq, round_id, job, basis, coldkeys, published, finalized_at", "1"),
];

fn rows_of(conn: &Connection, tables: &[(&str, &str, &str)], at: f64) -> Result<Map<String, Value>> {
    let mut state = Map::new();
    for (table, columns, rows) in tables.iter().copied() {
        let sql = format!("SELECT {columns} FROM {table} WHERE {rows} ORDER BY {columns}");
        let mut statement = conn.prepare(&sql)?;
        let width = statement.column_count();
        let args: Vec<Sql> = if rows.contains("?1") { vec![Sql::Real(at - PENALTY_WINDOW_S)] } else { Vec::new() };
        let found =
            statement.query_map(params_from_iter(args), |row| (0..width).map(|i| Ok(json_of(row.get_ref(i)?))).collect::<rusqlite::Result<Vec<Value>>>())?;
        let found: Vec<Value> = found.map(|row| row.map(Value::Array)).collect::<rusqlite::Result<_>>()?;
        state.insert(table.into(), found.into());
    }
    Ok(state)
}

/// The accounts at `at`: each table's rows as arrays, sorted; and the rounds' loadability beside them.
pub fn take(conn: &Connection, at: f64) -> Result<Value> {
    Ok(json!({"at": at, "tables": rows_of(conn, &TABLES, at)?, "rounds": rows_of(conn, &ROUNDS, at)?}))
}

/// Writes the accounts as they are now into the log; the blob holding them is named by their digest.
pub fn record(conn: &Connection, at: f64) -> Result<String> {
    let state = take(conn, at)?;
    let blob = crate::journal::Blob::of(canonical(&state));
    let digest = digest(&state);
    let output = json!({"digest": digest, "state": blob.reference()});
    crate::journal::append(conn, crate::journal::Record::new("snapshot", at, json!({"at": at}), output).with_blobs(vec![blob]))?;
    Ok(digest)
}

/// The hash of the accounts a snapshot holds: its time and its tables.
pub fn digest(snapshot: &Value) -> String {
    sha256(&canonical(&json!({"at": snapshot["at"], "tables": snapshot["tables"]})))
}

/// Fills an empty store with a snapshot, as a replay starts.
pub fn load(conn: &Connection, snapshot: &Value) -> Result<()> {
    for (part, (table, columns, _)) in TABLES.iter().map(|t| ("tables", *t)).chain(ROUNDS.iter().map(|t| ("rounds", *t))) {
        let rows = snapshot[part][table].as_array().cloned().unwrap_or_default();
        let (columns, extra) = if table == "validations" {
            // What verdict pages show is left out of the snapshot; only the columns decisions read are kept.
            (format!("{columns}, reason, report_key, report"), vec![Sql::Text(String::new()), Sql::Text(String::new()), Sql::Text("{}".into())])
        } else {
            (columns.to_string(), Vec::new())
        };
        let width = columns.split(',').count();
        // Outages a replay read before its first snapshot are in the snapshot too.
        conn.execute(&format!("DELETE FROM {table}"), [])?;
        let insert = format!("INSERT INTO {table} ({columns}) VALUES ({})", vec!["?"; width].join(", "));
        for row in rows {
            let mut values: Vec<Sql> = row.as_array().into_iter().flatten().map(sql_of).collect();
            values.extend(extra.iter().cloned());
            conn.execute(&insert, params_from_iter(values))?;
        }
    }
    Ok(())
}

fn sql_of(value: &Value) -> Sql {
    match value {
        Value::Null => Sql::Null,
        Value::Number(n) if n.is_f64() => Sql::Real(n.as_f64().unwrap_or(0.0)),
        Value::Number(n) => n.as_i64().map(Sql::Integer).unwrap_or_else(|| Sql::Real(n.as_f64().unwrap_or(0.0))),
        Value::String(s) => Sql::Text(s.clone()),
        Value::Bool(b) => Sql::Integer(i64::from(*b)),
        other => Sql::Text(other.to_string()),
    }
}

/// The tables the accounts live in, empty, as a replay needs them.
pub fn create(conn: &Connection) -> Result<()> {
    crate::canary::create(conn)?;
    crate::outages::create(conn)?;
    crate::rules::create(conn)?;
    crate::budgets::create(conn)?;
    crate::checks::create(conn)?;
    crate::validations::create(conn)?;
    crate::hours::create(conn)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budgets::{self, CRAWL};
    use crate::{checks, validations};

    const NOW: f64 = 1_800_000_000.0;

    fn accounts() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        budgets::reward(&conn, "m", "t1", 900, true, CRAWL, NOW - 10.0).unwrap();
        budgets::strike(&conn, "m", "coverage", "t2", 3, CRAWL, NOW - 5.0).unwrap();
        budgets::lock_out(&conn, "n", CRAWL, 1.0, "coverage", NOW).unwrap();
        checks::record(&conn, "m", CRAWL, "t3", true, NOW - 100.0, 2, 1, NOW - 50.0).unwrap();
        validations::finalize(&conn, "new", "k-new", "pass", 7, Some(&json!({"urls": ["u"]})), NOW - 60.0).unwrap();
        validations::finalize(&conn, "old", "k-old", "pass", 7, None, NOW - 200_000.0).unwrap();
        for (task, at, verdict, validator) in [("old", NOW - 200_000.0, "pass", ""), ("new", NOW - 60.0, "pass", ""), ("checked", NOW - 1000.0, "fail", "v")] {
            let job = json!({"round_id": "r", "miner": "m", "key": format!("k-{task}"), "completed_at": at});
            let result = json!({"verdict": verdict, "reason": "ok", "credited": 7});
            validations::record(&conn, &validations::build_report(task, &job, validator, result.as_object().unwrap(), &[], at), None).unwrap();
        }
        conn
    }

    #[test]
    fn a_loaded_snapshot_digests_the_same() {
        let conn = accounts();
        let taken = take(&conn, NOW).unwrap();
        let rows = |table: &str| taken["tables"][table].as_array().unwrap().len();
        assert_eq!((rows("miners"), rows("strikes"), rows("lockouts"), rows("checks")), (1, 1, 1, 1));
        assert_eq!(rows("validations"), 2, "a pass from before the last passed check and outside the day is out of reach");
        assert_eq!(rows("final_verdicts"), 1, "a final verdict of more than a day ago is out of reach");
        let copy = Connection::open_in_memory().unwrap();
        create(&copy).unwrap();
        load(&copy, &taken).unwrap();
        assert_eq!(digest(&take(&copy, NOW).unwrap()), digest(&taken));
        budgets::reward(&copy, "m", "t9", 1, false, CRAWL, NOW).unwrap();
        assert_ne!(digest(&take(&copy, NOW).unwrap()), digest(&taken), "one credited row shows");
        let edited = Connection::open_in_memory().unwrap();
        create(&edited).unwrap();
        load(&edited, &taken).unwrap();
        edited.execute("UPDATE final_verdicts SET verdict = 'fail' WHERE task_id = 'new'", []).unwrap();
        assert_ne!(digest(&take(&edited, NOW).unwrap()), digest(&taken), "a final verdict edited in place shows");
    }
}

//! Each miner's budget per pool, its hourly credits and coverage, and the strikes and lockouts that hold it back.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::py::round_to;

pub const START: i64 = 1;
pub const CEILING: i64 = 100;
pub const GROWTH: f64 = 0.5;
pub const WAITING_PER_BUDGET: i64 = 2;
pub const COVERAGE_GATE: f64 = 0.85;
pub const CRAWL: &str = "crawl";
pub const EMBED: &str = "embed";
pub const SHARE_WINDOW_H: i64 = 24;
pub const KEEP_H: i64 = SHARE_WINDOW_H * 7;
pub const HOUR: f64 = 3600.0;

/// Fetch failures are left out on purpose: the miner still tried.
pub const REWARD: &str = "verified";

/// A crash or a short outage lets several claims lapse at once; together they are one strike.
pub const EXPIRY_REASONS: [&str; 2] = ["claim_expired", "abandoned"];
pub const STRIKE_BURST_S: f64 = 300.0;
/// Only fails the miner caused count; "unscorable" is the validator's own timeout or crash.
pub const STRIKE_REASONS: [&str; 12] = [
    "claim_expired",
    "abandoned",
    "extra_rows",
    "coverage",
    "text_not_from_html",
    "content_mismatch",
    "errors_not_reproducible",
    "reported_rows",
    "unreadable",
    "vectors_missing",
    "vectors_malformed",
    "vectors_mismatch",
];
pub const STRIKES_TO_LOCK: i64 = 2;
pub const STRIKE_SHARE: f64 = 0.05;
pub const STRIKE_WINDOW_H: f64 = 24.0;
pub const LOCKOUT_STEPS_H: [f64; 3] = [1.0, 12.0, 48.0];
pub const LOCKOUT_MEMORY_H: f64 = 7.0 * 24.0;
pub const HOSTILE_LOCKOUT_H: f64 = 7.0 * 24.0;
pub const FULL_PENALTY_LOCKOUT_H: f64 = 48.0;

#[derive(Clone, Debug, PartialEq)]
pub struct MinerBudget {
    pub hotkey: String,
    pub pool: String,
    pub budget: i64,
    pub verified: i64,
}

pub fn hour_of(at: f64) -> i64 {
    (at / HOUR).floor() as i64
}

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS miners (
            hotkey   TEXT NOT NULL,
            pool     TEXT NOT NULL,
            budget   INTEGER NOT NULL DEFAULT 1,
            verified INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (hotkey, pool)
        );
        CREATE TABLE IF NOT EXISTS budget_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            hotkey     TEXT NOT NULL,
            pool       TEXT NOT NULL,
            old_budget INTEGER NOT NULL,
            new_budget INTEGER NOT NULL,
            cause      TEXT NOT NULL,
            task_id    TEXT,
            at         REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS budget_events_hotkey ON budget_events (hotkey, id);
        CREATE TABLE IF NOT EXISTS credits (
            pool   TEXT NOT NULL,
            hotkey TEXT NOT NULL,
            hour   INTEGER NOT NULL,
            amount INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (pool, hotkey, hour)
        );
        CREATE INDEX IF NOT EXISTS credits_hour ON credits (hour);
        CREATE TABLE IF NOT EXISTS coverage (
            hotkey   TEXT NOT NULL,
            hour     INTEGER NOT NULL,
            assigned INTEGER NOT NULL DEFAULT 0,
            returned INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (hotkey, hour)
        );
        CREATE INDEX IF NOT EXISTS coverage_hour ON coverage (hour);
        CREATE TABLE IF NOT EXISTS strikes (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            hotkey  TEXT NOT NULL,
            pool    TEXT NOT NULL,
            reason  TEXT NOT NULL,
            task_id TEXT,
            at      REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS strikes_hotkey ON strikes (hotkey, pool, at);
        CREATE TABLE IF NOT EXISTS lockouts (
            hotkey TEXT NOT NULL,
            pool   TEXT NOT NULL,
            until  REAL NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (hotkey, pool)
        );
        CREATE TABLE IF NOT EXISTS lockout_history (
            hotkey TEXT NOT NULL,
            pool   TEXT NOT NULL,
            at     REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS lockout_history_hotkey ON lockout_history (hotkey, pool, at);
        ",
    )?;
    Ok(())
}

/// The budget as it stood, the row created at the starting budget if there was none.
pub fn get_or_create(conn: &Connection, hotkey: &str, pool: &str) -> Result<MinerBudget> {
    let miner = get(conn, hotkey, pool)?;
    conn.execute("INSERT OR IGNORE INTO miners (hotkey, pool, budget) VALUES (?, ?, ?)", params![hotkey, pool, START])?;
    Ok(miner)
}

pub fn get(conn: &Connection, hotkey: &str, pool: &str) -> Result<MinerBudget> {
    let found = conn
        .query_row("SELECT budget, verified FROM miners WHERE hotkey = ? AND pool = ?", params![hotkey, pool], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?;
    let (budget, verified) = found.unwrap_or((START, 0));
    Ok(MinerBudget { hotkey: hotkey.into(), pool: pool.into(), budget, verified })
}

/// In-flight work never counts toward coverage.
pub fn record_coverage(conn: &Connection, hotkey: &str, assigned: i64, returned: i64, now: f64) -> Result<()> {
    conn.execute(
        "INSERT INTO coverage (hotkey, hour, assigned, returned) VALUES (?, ?, ?, ?)
         ON CONFLICT (hotkey, hour) DO UPDATE SET assigned = assigned + excluded.assigned,
         returned = returned + excluded.returned",
        params![hotkey, hour_of(now), assigned, returned.min(assigned)],
    )?;
    Ok(())
}

fn set_budget(conn: &Connection, hotkey: &str, pool: &str, new: i64, cause: &str, task_id: Option<&str>, now: f64) -> Result<MinerBudget> {
    let miner = get_or_create(conn, hotkey, pool)?;
    let new = new.clamp(1, CEILING);
    conn.execute("UPDATE miners SET budget = ? WHERE hotkey = ? AND pool = ?", params![new, hotkey, pool])?;
    conn.execute(
        "INSERT INTO budget_events (hotkey, pool, old_budget, new_budget, cause, task_id, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![hotkey, pool, miner.budget, new, cause, task_id, now],
    )?;
    Ok(MinerBudget { budget: new, ..miner })
}

pub fn reward(conn: &Connection, hotkey: &str, task_id: &str, amount: i64, ramp: bool, pool: &str, now: f64) -> Result<MinerBudget> {
    let miner = get_or_create(conn, hotkey, pool)?;
    conn.execute("UPDATE miners SET verified = verified + ? WHERE hotkey = ? AND pool = ?", params![amount, hotkey, pool])?;
    credit(conn, hotkey, amount, pool, now)?;
    if !ramp {
        return get_or_create(conn, hotkey, pool);
    }
    let grown = miner.budget + 1.max((miner.budget as f64 * GROWTH) as i64);
    set_budget(conn, hotkey, pool, grown, REWARD, Some(task_id), now)
}

/// Rows toward the miner's share this hour; a bad task takes its URLs back.
pub fn credit(conn: &Connection, hotkey: &str, amount: i64, pool: &str, now: f64) -> Result<()> {
    conn.execute(
        "INSERT INTO credits (pool, hotkey, hour, amount) VALUES (?, ?, ?, ?)
         ON CONFLICT (pool, hotkey, hour) DO UPDATE SET amount = amount + excluded.amount",
        params![pool, hotkey, hour_of(now), amount],
    )?;
    Ok(())
}

pub fn wipe_credits(conn: &Connection, hotkey: &str, since: f64, pool: &str) -> Result<()> {
    conn.execute("DELETE FROM credits WHERE hotkey = ? AND pool = ? AND hour >= ?", params![hotkey, pool, hour_of(since)])?;
    Ok(())
}

pub fn penalise(conn: &Connection, hotkey: &str, task_id: &str, cause: &str, pool: &str, now: f64) -> Result<MinerBudget> {
    let halved = get_or_create(conn, hotkey, pool)?.budget / 2;
    set_budget(conn, hotkey, pool, halved, cause, Some(task_id), now)
}

/// The lockout's end if this strike, among `judged` recent verdicts, starts one.
pub fn strike(conn: &Connection, hotkey: &str, reason: &str, task_id: &str, judged: i64, pool: &str, now: f64) -> Result<Option<f64>> {
    if EXPIRY_REASONS.contains(&reason) && lapsed_recently(conn, hotkey, pool, now)? {
        return locked_until(conn, hotkey, pool, now);
    }
    conn.execute("INSERT INTO strikes (hotkey, pool, reason, task_id, at) VALUES (?, ?, ?, ?, ?)", params![hotkey, pool, reason, task_id, now])?;
    let recent: i64 =
        conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ? AND pool = ? AND at > ?", params![hotkey, pool, now - STRIKE_WINDOW_H * HOUR], |row| {
            row.get(0)
        })?;
    if recent < STRIKES_TO_LOCK || (recent as f64) < STRIKE_SHARE * judged as f64 {
        return Ok(None);
    }
    let before: i64 = conn.query_row(
        "SELECT COUNT(*) FROM lockout_history WHERE hotkey = ? AND pool = ? AND at > ?",
        params![hotkey, pool, now - LOCKOUT_MEMORY_H * HOUR],
        |row| row.get(0),
    )?;
    let hours = LOCKOUT_STEPS_H[(before as usize).min(LOCKOUT_STEPS_H.len() - 1)];
    Ok(Some(lock_out(conn, hotkey, pool, hours, reason, now)?))
}

fn lapsed_recently(conn: &Connection, hotkey: &str, pool: &str, now: f64) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM strikes WHERE hotkey = ? AND pool = ? AND at > ? AND reason IN (?, ?) LIMIT 1",
            params![hotkey, pool, now - STRIKE_BURST_S, EXPIRY_REASONS[0], EXPIRY_REASONS[1]],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

pub fn lock_out(conn: &Connection, hotkey: &str, pool: &str, hours: f64, reason: &str, now: f64) -> Result<f64> {
    let until = now + hours * HOUR;
    conn.execute(
        "INSERT INTO lockouts (hotkey, pool, until, reason) VALUES (?, ?, ?, ?)
         ON CONFLICT (hotkey, pool) DO UPDATE SET until = MAX(until, excluded.until), reason = excluded.reason",
        params![hotkey, pool, until, reason],
    )?;
    conn.execute("INSERT INTO lockout_history (hotkey, pool, at) VALUES (?, ?, ?)", params![hotkey, pool, now])?;
    Ok(locked_until(conn, hotkey, pool, now)?.unwrap_or(until))
}

pub fn locked_until(conn: &Connection, hotkey: &str, pool: &str, now: f64) -> Result<Option<f64>> {
    let until: Option<f64> = conn.query_row("SELECT until FROM lockouts WHERE hotkey = ? AND pool = ?", params![hotkey, pool], |row| row.get(0)).optional()?;
    Ok(until.filter(|until| *until > now))
}

pub fn all(conn: &Connection, pool: &str) -> Result<Vec<MinerBudget>> {
    let mut statement = conn.prepare("SELECT hotkey, pool, budget, verified FROM miners WHERE pool = ?")?;
    let rows = statement.query_map([pool], |row| Ok(MinerBudget { hotkey: row.get(0)?, pool: row.get(1)?, budget: row.get(2)?, verified: row.get(3)? }))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn lockouts(conn: &Connection, pool: &str, now: f64) -> Result<HashMap<String, f64>> {
    let mut statement = conn.prepare("SELECT hotkey, until FROM lockouts WHERE pool = ? AND until > ?")?;
    let rows = statement.query_map(params![pool, now], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn pools_of(conn: &Connection, hotkey: &str) -> Result<Vec<MinerBudget>> {
    let mut statement = conn.prepare("SELECT hotkey, pool, budget, verified FROM miners WHERE hotkey = ? ORDER BY pool")?;
    let rows = statement.query_map([hotkey], |row| Ok(MinerBudget { hotkey: row.get(0)?, pool: row.get(1)?, budget: row.get(2)?, verified: row.get(3)? }))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn history(conn: &Connection, hotkey: &str, limit: i64) -> Result<Vec<Value>> {
    let mut statement = conn.prepare("SELECT pool, old_budget, new_budget, cause, task_id, at FROM budget_events WHERE hotkey = ? ORDER BY id DESC LIMIT ?")?;
    let rows = statement.query_map(params![hotkey, limit], |row| {
        Ok(json!({
            "pool": row.get::<_, String>(0)?,
            "old": row.get::<_, i64>(1)?,
            "new": row.get::<_, i64>(2)?,
            "cause": row.get::<_, String>(3)?,
            "task_id": row.get::<_, Option<String>>(4)?,
            "at": crate::db::real(row, 5)?,
        }))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Each pool's miners by their share of the credits earned in the window.
pub fn shares(conn: &Connection, window_hours: i64, now: f64) -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let since = hour_of(now) - window_hours + 1;
    let mut statement = conn.prepare("SELECT pool, hotkey, SUM(amount) FROM credits WHERE hour >= ? AND hour <= ? GROUP BY pool, hotkey")?;
    let rows = statement.query_map(params![since, hour_of(now)], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?)))?;
    let mut earned: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for row in rows {
        let (pool, hotkey, amount) = row?;
        if amount > 0 {
            earned.entry(pool).or_default().insert(hotkey, amount);
        }
    }
    Ok(earned
        .into_iter()
        .map(|(pool, miners)| {
            let total: i64 = miners.values().sum();
            (pool, miners.into_iter().map(|(hotkey, amount)| (hotkey, amount as f64 / total as f64)).collect())
        })
        .collect())
}

pub fn coverage_report(conn: &Connection, window_hours: i64, now: f64) -> Result<BTreeMap<String, Value>> {
    let since = hour_of(now) - window_hours + 1;
    let mut statement =
        conn.prepare("SELECT hotkey, SUM(assigned), SUM(returned) FROM coverage WHERE hour >= ? AND hour <= ? GROUP BY hotkey HAVING SUM(assigned) > 0")?;
    let rows = statement.query_map(params![since, hour_of(now)], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)))?;
    let mut report = BTreeMap::new();
    for row in rows {
        let (hotkey, assigned, returned) = row?;
        report.insert(hotkey, json!({"assigned": assigned, "returned": returned, "coverage": round_to(returned as f64 / assigned as f64, 4)}));
    }
    Ok(report)
}

pub fn prune(conn: &Connection, keep_hours: i64, now: f64) -> Result<()> {
    let oldest = hour_of(now) - keep_hours;
    conn.execute("DELETE FROM credits WHERE hour < ?", [oldest])?;
    conn.execute("DELETE FROM coverage WHERE hour < ?", [oldest])?;
    conn.execute("DELETE FROM strikes WHERE at < ?", [oldest as f64 * HOUR])?;
    conn.execute("DELETE FROM lockout_history WHERE at < ?", [now - LOCKOUT_MEMORY_H * HOUR])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: f64 = 1_800_000_000.0;

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        conn
    }

    fn strike_at(conn: &Connection, reason: &str, task: &str, judged: i64, now: f64) -> Option<f64> {
        strike(conn, "m", reason, task, judged, CRAWL, now).unwrap()
    }

    #[test]
    fn one_strike_is_a_warning_and_the_second_locks_the_miner_out() {
        let conn = store();
        assert_eq!(strike_at(&conn, "content_mismatch", "t1", 1, NOW), None);
        assert_eq!(locked_until(&conn, "m", CRAWL, NOW).unwrap(), None);
        let until = strike_at(&conn, "errors_not_reproducible", "t2", 2, NOW + HOUR).unwrap();
        assert_eq!(until, NOW + HOUR + LOCKOUT_STEPS_H[0] * HOUR);
        assert_eq!(locked_until(&conn, "m", CRAWL, NOW + HOUR + 60.0).unwrap(), Some(until));
        assert_eq!(locked_until(&conn, "m", CRAWL, until).unwrap(), None);
        assert_eq!(locked_until(&conn, "other", CRAWL, NOW + HOUR + 60.0).unwrap(), None);
    }

    #[test]
    fn a_busy_miner_with_a_few_bad_batches_is_not_locked() {
        let conn = store();
        strike_at(&conn, "content_mismatch", "t1", 50, NOW);
        assert_eq!(strike_at(&conn, "content_mismatch", "t2", 100, NOW + 60.0), None);
        assert!(strike_at(&conn, "coverage", "t3", 60, NOW + 120.0).is_some());
    }

    #[test]
    fn strikes_a_day_apart_never_lock() {
        let conn = store();
        strike_at(&conn, "content_mismatch", "t1", 1, NOW);
        assert_eq!(strike_at(&conn, "content_mismatch", "t2", 1, NOW + STRIKE_WINDOW_H * HOUR + 1.0), None);
    }

    #[test]
    fn each_lockout_within_a_week_lasts_longer() {
        let conn = store();
        let (mut lengths, mut now) = (Vec::new(), NOW);
        for step in 0..4 {
            strike_at(&conn, "coverage", &format!("a{step}"), 0, now);
            let until = strike_at(&conn, "coverage", &format!("b{step}"), 0, now + 60.0).unwrap();
            lengths.push(until - (now + 60.0));
            now = until + STRIKE_WINDOW_H * HOUR;
        }
        let [first, second, third] = LOCKOUT_STEPS_H.map(|h| h * HOUR);
        assert_eq!(lengths, [first, second, third, third]);
    }

    #[test]
    fn claims_that_lapse_together_are_one_strike() {
        let conn = store();
        for n in 0..5 {
            assert_eq!(strike_at(&conn, "claim_expired", &format!("t{n}"), 10, NOW + n as f64), None);
        }
        let later = NOW + STRIKE_BURST_S + 1.0;
        assert_eq!(strike_at(&conn, "abandoned", "t9", 10, later), Some(later + LOCKOUT_STEPS_H[0] * HOUR));
    }

    #[test]
    fn a_hotkey_that_only_hoards_is_locked_out_on_its_second_lapse() {
        let conn = store();
        strike_at(&conn, "claim_expired", "t1", 0, NOW);
        assert_eq!(strike_at(&conn, "claim_expired", "t2", 0, NOW + 600.0), Some(NOW + 600.0 + LOCKOUT_STEPS_H[0] * HOUR));
    }

    #[test]
    fn failed_uploads_are_never_merged_into_one_strike() {
        let conn = store();
        strike_at(&conn, "content_mismatch", "t1", 2, NOW);
        assert!(strike_at(&conn, "content_mismatch", "t2", 2, NOW + 1.0).is_some());
    }

    fn crawl_shares(conn: &Connection, now: f64) -> BTreeMap<String, f64> {
        shares(conn, SHARE_WINDOW_H, now).unwrap().remove(CRAWL).unwrap_or_default()
    }

    fn work(conn: &Connection, hotkey: &str, assigned: i64, returned: i64, verified: i64) {
        record_coverage(conn, hotkey, assigned, returned, NOW).unwrap();
        if verified > 0 {
            reward(conn, hotkey, "t", verified, true, CRAWL, NOW).unwrap();
        }
    }

    fn earned(conn: &Connection, hotkey: &str, urls: i64, hours_ago: i64) {
        record_coverage(conn, hotkey, urls, urls, NOW).unwrap();
        reward(conn, hotkey, &format!("task-{hotkey}-{hours_ago}"), urls, false, CRAWL, NOW).unwrap();
        conn.execute("UPDATE credits SET hour = hour - ? WHERE hotkey = ? AND hour = ?", params![hours_ago, hotkey, hour_of(NOW)]).unwrap();
    }

    fn close(found: BTreeMap<String, f64>, want: &[(&str, f64)]) {
        assert_eq!(found.len(), want.len(), "{found:?}");
        for (hotkey, share) in want {
            assert!((found[*hotkey] - share).abs() < 1e-12, "{hotkey}: {found:?}");
        }
    }

    #[test]
    fn shares_follow_verified_work_and_coverage() {
        let conn = store();
        work(&conn, "a", 100, 100, 100);
        close(crawl_shares(&conn, NOW), &[("a", 1.0)]);
        work(&conn, "omitter", 100, 80, 80);
        close(crawl_shares(&conn, NOW), &[("a", 100.0 / 180.0), ("omitter", 80.0 / 180.0)]);
        record_coverage(&conn, "hoarder", 500, 0, NOW).unwrap();
        assert!(!crawl_shares(&conn, NOW).contains_key("hoarder"));
        assert_eq!(coverage_report(&conn, SHARE_WINDOW_H, NOW).unwrap()["hoarder"]["coverage"], 0.0);
    }

    #[test]
    fn a_bad_task_takes_its_urls_back_from_the_day() {
        let conn = store();
        work(&conn, "a", 3000, 3000, 3000);
        work(&conn, "b", 1000, 1000, 1000);
        credit(&conn, "a", -1000, CRAWL, NOW).unwrap();
        close(crawl_shares(&conn, NOW), &[("a", 2.0 / 3.0), ("b", 1.0 / 3.0)]);
    }

    #[test]
    fn the_budget_grows_by_half_with_each_pass_up_to_the_ceiling() {
        let conn = store();
        let grown: Vec<i64> = (0..14).map(|n| reward(&conn, "a", &format!("t{n}"), 1, true, CRAWL, NOW).unwrap().budget).collect();
        assert_eq!(grown, [2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 94, CEILING, CEILING, CEILING]);
        assert_eq!(penalise(&conn, "a", "t99", "verification_failed", CRAWL, NOW).unwrap().budget, CEILING / 2);
    }

    #[test]
    fn work_not_yet_decided_does_not_count_against_coverage() {
        let conn = store();
        get_or_create(&conn, "busy", CRAWL).unwrap();
        assert!(coverage_report(&conn, SHARE_WINDOW_H, NOW).unwrap().is_empty());
        work(&conn, "busy", 100, 100, 100);
        assert_eq!(coverage_report(&conn, SHARE_WINDOW_H, NOW).unwrap()["busy"]["coverage"], 1.0);
    }

    #[test]
    fn a_low_credit_pass_does_not_ramp_the_budget() {
        let conn = store();
        reward(&conn, "a", "t1", 25, true, CRAWL, NOW).unwrap();
        reward(&conn, "a", "t2", 13, false, CRAWL, NOW).unwrap();
        let miner = get(&conn, "a", CRAWL).unwrap();
        assert_eq!((miner.budget, miner.verified), (2, 38));
    }

    #[test]
    fn peeking_at_an_unknown_miner_stores_nothing() {
        let conn = store();
        assert_eq!(get(&conn, "stranger", CRAWL).unwrap().budget, 1);
        assert!(all(&conn, CRAWL).unwrap().is_empty());
    }

    #[test]
    fn history_is_newest_first_and_bounded() {
        let conn = store();
        for n in 0..5 {
            reward(&conn, "a", &format!("t{n}"), 1, true, CRAWL, NOW).unwrap();
        }
        let found: Vec<serde_json::Value> = history(&conn, "a", 3).unwrap().into_iter().map(|e| e["task_id"].clone()).collect();
        assert_eq!(found, ["t4", "t3", "t2"]);
    }

    #[test]
    fn work_older_than_the_window_no_longer_pays_and_a_week_is_kept() {
        let conn = store();
        earned(&conn, "old", 1000, SHARE_WINDOW_H + 1);
        earned(&conn, "new", 10, 0);
        close(crawl_shares(&conn, NOW), &[("new", 1.0)]);
        assert_eq!(get(&conn, "old", CRAWL).unwrap().verified, 1000);
        earned(&conn, "ancient", 5, KEEP_H + 5);
        prune(&conn, KEEP_H, NOW).unwrap();
        let kept: Vec<String> =
            conn.prepare("SELECT DISTINCT hotkey FROM credits ORDER BY hotkey").unwrap().query_map([], |row| row.get(0)).unwrap().map(Result::unwrap).collect();
        assert_eq!(kept, ["new", "old"]);
        assert!(shares(&store(), SHARE_WINDOW_H, NOW).unwrap().is_empty());
    }

    #[test]
    fn a_lapse_costs_its_urls_for_as_long_as_the_window_holds_it() {
        let conn = store();
        credit(&conn, "recovered", -750, CRAWL, NOW).unwrap();
        conn.execute("UPDATE credits SET hour = hour - ?", [SHARE_WINDOW_H + 1]).unwrap();
        earned(&conn, "recovered", 1000, 0);
        close(crawl_shares(&conn, NOW), &[("recovered", 1.0)]);
        credit(&conn, "recovered", -750, CRAWL, NOW).unwrap();
        earned(&conn, "steady", 250, 0);
        close(crawl_shares(&conn, NOW), &[("recovered", 0.5), ("steady", 0.5)]);
        credit(&conn, "recovered", -1000, CRAWL, NOW).unwrap();
        close(crawl_shares(&conn, NOW), &[("steady", 1.0)]);
    }

    #[test]
    fn each_pool_is_shared_out_on_its_own() {
        let conn = store();
        earned(&conn, "crawler", 300, 0);
        record_coverage(&conn, "hoarder", 1000, 0, NOW).unwrap();
        for (hotkey, amount) in [("embedder", 40), ("second", 10), ("hoarder", 10)] {
            reward(&conn, hotkey, &format!("t-{hotkey}"), amount, false, EMBED, NOW).unwrap();
        }
        let pools = shares(&conn, SHARE_WINDOW_H, NOW).unwrap();
        close(pools[CRAWL].clone(), &[("crawler", 1.0)]);
        close(pools[EMBED].clone(), &[("embedder", 4.0 / 6.0), ("second", 1.0 / 6.0), ("hoarder", 1.0 / 6.0)]);
    }
}

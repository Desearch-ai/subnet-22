//! A log database at production rates, and what the log pages' reads cost on it: `cargo run --release --example log_reads -- build <dir> [days]`, `-- tally <dir>`, then `-- time <dir>`.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};
use desearch::time::now;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rusqlite::{params, Connection, OpenFlags};
use serde_json::{json, Map, Value};
use task_api::budgets::{self, hour_of, HOUR, SHARE_WINDOW_H};
use task_api::roundlog::{self, Receipt};
use task_api::validations::{self, TaskFilter, VoteFilter};
use task_api::{checks, db, hours, registry, roundstore};

const MINERS: usize = 250;
const CRAWL_PER_HOUR: usize = 20_000;
const EMBED_PER_HOUR: usize = 1_000;
const CHECKED_SHARE: f64 = 0.03;
const FAIL_SHARE_OF_CHECKED: f64 = 10.0 / 15_000.0;
const VOID_SHARE_OF_CHECKED: f64 = 20.0 / 15_000.0;
const REFUSED_PER_HOUR: usize = 14_000;
const RECLAIMED_PER_HOUR: usize = 40;
const TASKS_PER_ROUND: usize = 9;
const VALIDATORS: [&str; 2] = ["5CkmTfQH8UbYAbohFr2m3jPm5gWJ6L3XrSqHd1ijKuzz4iZ5", "5FBk8NbxWLRboCqEL2oj1KXwSmmaC1nFmBBL4AzLEN5Qjs22"];
const RUNS: usize = 5;

fn hotkey(n: usize) -> String {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let mut rng = StdRng::seed_from_u64(n as u64);
    std::iter::once('5').chain((0..47).map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char)).collect()
}

fn hex_id(rng: &mut StdRng) -> String {
    format!("{:016x}", rng.gen::<u64>())
}

fn create(conn: &Connection) -> Result<()> {
    budgets::create(conn)?;
    roundlog::create(conn)?;
    roundstore::create(conn)?;
    validations::create(conn)?;
    checks::create(conn)?;
    registry::create(conn)?;
    hours::create(conn)?;
    Ok(())
}

struct Task {
    task_id: String,
    kind: &'static str,
    round_id: String,
    miner: String,
    claimed_at: f64,
    completed_at: f64,
    scored_at: f64,
    validator: &'static str,
    verdict: &'static str,
    reason: &'static str,
}

fn result_of(task: &Task, rng: &mut StdRng) -> Value {
    if task.kind == budgets::EMBED {
        let chars = rng.gen_range(2_000_000..4_000_000);
        return json!({"verdict": task.verdict, "reason": task.reason, "credited": chars, "returned": 1_500, "samples": []});
    }
    let returned = rng.gen_range(980..=1_000);
    let credited = if task.verdict == "pass" { returned - rng.gen_range(0..40) } else { 0 };
    let mut result = json!({
        "verdict": task.verdict, "reason": task.reason, "returned": returned, "missing": 1_000 - returned,
        "error_rows": rng.gen_range(0..30), "credited": credited, "samples": [], "rejected": [],
    });
    if !task.validator.is_empty() {
        let samples: Vec<Value> =
            (0..10).map(|i| json!({"url": format!("https://site{i}.example/a/{}", rng.gen::<u32>()), "outcome": "matched", "similarity": 0.99})).collect();
        result["sampled"] = 10.into();
        result["matched"] = 10.into();
        result["samples"] = samples.into();
    }
    result
}

fn details(rng: &mut StdRng) -> Vec<u8> {
    let urls: Vec<Value> = (0..100)
        .map(|i| json!({"url": format!("https://site{i}.example/article/{}", rng.gen::<u64>()), "status": 200, "text_chars": rng.gen_range(500..20_000), "sampled": i < 10, "miner_snippet": "A paragraph of the page as the miner returned it, long enough to be realistic."}))
        .collect();
    validations::pack_details(Some(&Value::Array(urls))).unwrap_or_default()
}

fn record_task(conn: &Connection, task: &Task, rng: &mut StdRng, last_pass: &mut BTreeMap<(String, &'static str), f64>, detail_from: f64) -> Result<()> {
    let job = json!({
        "round_id": task.round_id, "miner": task.miner, "kind": task.kind, "etag": "\"0f3c\"", "size": 2_400_000,
        "key": format!("uploads/task={}/{}-{}.parquet", task.task_id, task.miner, rng.gen::<u32>()),
        "claimed_at": task.claimed_at, "completed_at": task.completed_at,
    });
    let result = result_of(task, rng);
    let votes: Vec<Value> = if task.validator.is_empty() {
        Vec::new()
    } else {
        VALIDATORS
            .iter()
            .enumerate()
            .map(|(i, validator)| {
                let verdict = if task.reason == "validators_disagree" { ["pass", "fail"][i] } else { task.verdict };
                json!({"validator": validator, "verdict": verdict, "credited": result["credited"], "result": result, "at": task.scored_at - 5.0 + i as f64})
            })
            .collect()
    };
    let report = validations::build_report(&task.task_id, &job, task.validator, result.as_object().unwrap(), &votes, task.scored_at);
    let packed = (!task.validator.is_empty() && task.scored_at >= detail_from).then(|| details(rng));
    validations::record(conn, &report, packed.as_deref())?;
    if !votes.is_empty() {
        let decided = task.reason != "validators_disagree";
        validations::record_votes(conn, &report, &votes, &[], decided)?;
    }
    let key = (task.miner.clone(), task.kind);
    if task.verdict == "fail" {
        validations::withdraw(conn, &task.miner, task.kind, last_pass.get(&key).copied().unwrap_or(task.scored_at - HOUR), false)?;
    } else if task.verdict == "pass" && !task.validator.is_empty() {
        last_pass.insert(key, task.completed_at);
    }
    Ok(())
}

/// A log entry to write: when, what, the task it is about by index, and a refusal or cause.
type Event = (f64, &'static str, usize, Option<Value>, Option<&'static str>);

fn build(dir: &Path, days: i64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(db::FILE);
    if path.exists() {
        bail!("{} exists", path.display());
    }
    let mut conn = Connection::open(&path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "OFF")?;
    create(&conn)?;
    let mut rng = StdRng::seed_from_u64(22);
    let miners: Vec<String> = (0..MINERS).map(hotkey).collect();
    let end = hour_of(now());
    let start = end - days * 24;
    let detail_from = now() - validations::URL_DETAIL_DAYS * 86_400.0;
    let mut last_pass = BTreeMap::new();
    let mut seq = 0;
    let started = Instant::now();
    for hour in start..=end {
        let tx = conn.transaction()?;
        let base = hour as f64 * HOUR;
        let mut tasks = Vec::new();
        let mut rounds: Vec<(String, &'static str)> = Vec::new();
        for (kind, per_hour) in [(budgets::CRAWL, CRAWL_PER_HOUR), (budgets::EMBED, EMBED_PER_HOUR)] {
            for i in 0..per_hour {
                if i % TASKS_PER_ROUND == 0 {
                    rounds.push((hex_id(&mut rng), kind));
                }
                let scored_at = base + (i as f64 + rng.gen::<f64>()) * HOUR / per_hour as f64;
                if scored_at > now() {
                    break;
                }
                let checked = rng.gen_bool(CHECKED_SHARE);
                let wait = if checked { rng.gen_range(100.0..240.0) } else { rng.gen_range(112.0..122.0) };
                let completed_at = scored_at - wait;
                let (verdict, reason) = match (checked, rng.gen::<f64>()) {
                    (false, _) => ("pass", "unchecked"),
                    (true, x) if x < FAIL_SHARE_OF_CHECKED => ("fail", ["content_mismatch", "upload_changed"][rng.gen_range(0..2)]),
                    (true, x) if x < FAIL_SHARE_OF_CHECKED + VOID_SHARE_OF_CHECKED => ("void", "validators_disagree"),
                    (true, _) => ("pass", "ok"),
                };
                tasks.push(Task {
                    task_id: hex_id(&mut rng),
                    kind,
                    round_id: rounds.last().unwrap().0.clone(),
                    miner: miners[rng.gen_range(0..MINERS)].clone(),
                    claimed_at: completed_at - rng.gen_range(1.0..170.0),
                    completed_at,
                    scored_at,
                    validator: if checked { VALIDATORS[rng.gen_range(0..2)] } else { "" },
                    verdict,
                    reason,
                });
            }
        }
        for (round_id, kind) in &rounds {
            tx.execute(
                "INSERT INTO rounds (round_id, kind, manifest_hash, seed_block, opened_at, seed, serve_order, closed_at, batches, filled_at) VALUES (?, ?, ?, 0, ?, 's', '[]', ?, '[]', ?)",
                params![round_id, kind, "ab".repeat(32), base, base + HOUR, base],
            )?;
        }
        tasks.sort_by(|a, b| a.scored_at.total_cmp(&b.scored_at));
        let mut events: Vec<Event> = Vec::new();
        for (i, task) in tasks.iter().enumerate() {
            events.push((task.claimed_at, "issued", i, None, None));
            events.push((task.completed_at, "completed", i, None, None));
        }
        for _ in 0..REFUSED_PER_HOUR {
            let code = match rng.gen::<f64>() {
                x if x < 0.95 => json!({"code": "QUEUE_EMPTY", "inputs": {"depth": 0, "retry_after": 10.0}}),
                x if x < 0.99 => json!({"code": "NO_CAPACITY", "inputs": {"budget": 100, "in_flight": 100, "retry_after": 5.0}}),
                _ => json!({"code": "LOCKED_OUT", "inputs": {"until": base + 7_200.0, "retry_after": 3_600.0}}),
            };
            events.push((base + rng.gen::<f64>() * HOUR, "refused", rng.gen_range(0..tasks.len().max(1)), Some(code), None));
        }
        for _ in 0..RECLAIMED_PER_HOUR {
            events.push((base + rng.gen::<f64>() * HOUR, "reclaimed", rng.gen_range(0..tasks.len().max(1)), None, Some("expired")));
        }
        events.retain(|event| event.0 <= now());
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (at, outcome, i, refusal, cause) in events {
            seq += 1;
            let Some(task) = tasks.get(i) else { continue };
            let hotkey = if refusal.is_some() { &miners[rng.gen_range(0..MINERS)] } else { &task.miner };
            let receipt = Receipt {
                round_id: task.round_id.clone(),
                hotkey: hotkey.clone(),
                requested_at: at,
                outcome,
                seq,
                task_id: refusal.is_none().then(|| task.task_id.clone()),
                refusal,
                cause: cause.map(String::from),
                block: None,
            };
            roundlog::record(&tx, &receipt, &"ab".repeat(64), at)?;
        }
        let mut credits: BTreeMap<(&str, &str), i64> = BTreeMap::new();
        for task in &tasks {
            record_task(&tx, task, &mut rng, &mut last_pass, detail_from)?;
            if task.verdict == "pass" {
                *credits.entry((task.kind, task.miner.as_str())).or_default() += if task.kind == budgets::CRAWL { 980 } else { 3_000_000 };
            }
        }
        for ((pool, miner), amount) in credits {
            tx.execute("INSERT INTO credits VALUES (?, ?, ?, ?)", params![pool, miner, hour, amount])?;
        }
        for miner in &miners {
            tx.execute("INSERT INTO coverage VALUES (?, ?, ?, ?)", params![miner, hour, 80_000, 79_000])?;
        }
        tx.commit()?;
        if (hour - start) % 24 == 0 {
            eprintln!("day {} of {days} written in {:.0}s", (hour - start) / 24, started.elapsed().as_secs_f64());
        }
    }
    let tx = conn.transaction()?;
    for (n, miner) in miners.iter().enumerate() {
        for pool in [budgets::CRAWL, budgets::EMBED] {
            tx.execute("INSERT INTO miners VALUES (?, ?, 100, ?)", params![miner, pool, 1_000_000 + n as i64])?;
        }
        for k in 0..20 {
            tx.execute(
                "INSERT INTO budget_events (hotkey, pool, old_budget, new_budget, cause, task_id, at) VALUES (?, 'crawl', 99, 100, 'verified', 't', ?)",
                params![miner, now() - k as f64 * 3_000.0],
            )?;
        }
    }
    for old in MINERS..MINERS * 3 {
        tx.execute("INSERT INTO miners VALUES (?, 'crawl', 1, 0)", [hotkey(old)])?;
    }
    for miner in miners.iter().take(5) {
        tx.execute("INSERT INTO lockouts VALUES (?, 'crawl', ?, 'content_mismatch')", params![miner, now() + 7_200.0])?;
    }
    for validator in VALIDATORS {
        tx.execute("INSERT INTO validator_audits VALUES (?, 166893, 0)", [validator])?;
    }
    tx.commit()?;
    while validations::roll_up(&conn, 100_000)? > 0 {}
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    eprintln!("built {} in {:.0}s", path.display(), started.elapsed().as_secs_f64());
    Ok(())
}

fn timed<T>(name: &str, mut read: impl FnMut() -> Result<T>, size: impl Fn(&T) -> usize) -> Result<()> {
    let mut times = Vec::with_capacity(RUNS);
    let mut rows = 0;
    for _ in 0..RUNS {
        let started = Instant::now();
        let found = read()?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        rows = size(&found);
    }
    let first = times[0];
    times.sort_by(f64::total_cmp);
    println!("{name:<44} first {first:>9.2} ms   median {:>9.2} ms   rows {rows}", times[RUNS / 2]);
    Ok(())
}

fn time(dir: &Path) -> Result<()> {
    let opened = Connection::open_with_flags(dir.join(db::FILE), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    opened.pragma_update(None, "query_only", "ON")?;
    let conn = &opened;
    let until = now();
    let since = until - SHARE_WINDOW_H as f64 * HOUR;
    let miner = hotkey(7);
    let validator = VALIDATORS[0];
    let checked: String = conn.query_row("SELECT task_id FROM validations WHERE validator != '' ORDER BY id DESC LIMIT 1", [], |row| row.get(0))?;
    let len = |found: &BTreeMap<String, Value>| found.len();
    let (first, last) = (hour_of(until) - SHARE_WINDOW_H + 1, hour_of(until));

    timed("overview: miner totals crawl 24 h", || validations::miner_totals(conn, since, until, budgets::CRAWL, None), len)?;
    timed("overview: miner totals embed 24 h", || validations::miner_totals(conn, since, until, budgets::EMBED, None), len)?;
    timed(
        "overview: budgets, lockouts, coverage, shares",
        || {
            let all = budgets::all(conn, budgets::CRAWL)?;
            budgets::lockouts(conn, budgets::CRAWL, until)?;
            budgets::coverage_report(conn, SHARE_WINDOW_H, until)?;
            budgets::shares(conn, SHARE_WINDOW_H, until)?;
            Ok(all)
        },
        Vec::len,
    )?;
    timed("overview: validator totals 24 h", || validations::validator_totals(conn, since, until), len)?;
    timed("overview: verdicts all time", || validations::verdicts(conn, None), Map::len)?;
    timed(
        "overview: claims, reasons, waits, take-backs",
        || {
            let claims = hours::total_claims(&hours::claims(conn, first, last, budgets::CRAWL, None)?);
            hours::reasons(conn, first, last, budgets::CRAWL)?;
            hours::total_waits(conn, first, last, budgets::CRAWL)?;
            hours::taken_back(conn, since, budgets::CRAWL)?;
            Ok(claims)
        },
        Map::len,
    )?;
    timed(
        "hours: crawl 24 h",
        || {
            validations::hour_series(conn, first as f64 * HOUR, until, budgets::CRAWL, None)?;
            hours::checked(conn, first, last, budgets::CRAWL)?;
            hours::waits(conn, first, last, budgets::CRAWL)?;
            hours::claims(conn, first, last, budgets::CRAWL, None)
        },
        BTreeMap::len,
    )?;
    timed(
        "hours: crawl one miner 24 h",
        || {
            validations::hour_series(conn, first as f64 * HOUR, until, budgets::CRAWL, Some(&miner))?;
            hours::claims(conn, first, last, budgets::CRAWL, Some(&miner))
        },
        BTreeMap::len,
    )?;
    timed(
        "hours: crawl 7 days",
        || {
            let first = last - 7 * 24 + 1;
            validations::hour_series(conn, first as f64 * HOUR, until, budgets::CRAWL, None)?;
            hours::checked(conn, first, last, budgets::CRAWL)?;
            hours::waits(conn, first, last, budgets::CRAWL)?;
            hours::claims(conn, first, last, budgets::CRAWL, None)
        },
        BTreeMap::len,
    )?;
    timed("series: crawl per hour 24 h", || validations::hour_series(conn, since, until, budgets::CRAWL, None), BTreeMap::len)?;
    timed("series: crawl per 5 min 2 h", || validations::task_series(conn, 300, until - 7_200.0, until, budgets::CRAWL, None), BTreeMap::len)?;
    timed("series: one miner per hour 24 h", || validations::hour_series(conn, since, until, budgets::CRAWL, Some(&miner)), BTreeMap::len)?;
    timed("series: one validator per hour 24 h", || validations::vote_series(conn, 3_600, since, until, validator), BTreeMap::len)?;
    timed(
        "miner page reads",
        || {
            for kind in [budgets::CRAWL, budgets::EMBED] {
                validations::miner_totals(conn, since, until, kind, Some(&miner))?;
                hours::claims(conn, first, last, kind, Some(&miner))?;
                budgets::get(conn, &miner, kind)?;
                budgets::locked_until(conn, &miner, kind, until)?;
            }
            budgets::shares(conn, SHARE_WINDOW_H, until)?;
            validations::verdicts(conn, Some(&miner))?;
            budgets::pools_of(conn, &miner)?;
            budgets::coverage_report(conn, SHARE_WINDOW_H, until)?;
            hours::take_backs_of(conn, &miner, 20)?;
            budgets::history(conn, &miner, 100)
        },
        Vec::len,
    )?;
    let votes = |filter: VoteFilter| move || validations::votes(conn, &filter).map(|(found, _)| found);
    timed("votes: newest 50", votes(VoteFilter { limit: 50, ..VoteFilter::default() }), Vec::len)?;
    timed("votes: one validator 50", votes(VoteFilter { validator: Some(validator.into()), limit: 50, ..VoteFilter::default() }), Vec::len)?;
    timed("votes: disagreed 50", votes(VoteFilter { agreed: Some(false), limit: 50, ..VoteFilter::default() }), Vec::len)?;
    timed(
        "votes: one validator disagreed 50",
        votes(VoteFilter { validator: Some(validator.into()), agreed: Some(false), limit: 50, ..VoteFilter::default() }),
        Vec::len,
    )?;
    let events =
        |miner: Option<&'static str>, outcome: Option<&'static str>| move || roundlog::events(conn, miner, None, outcome, None, 50).map(|(found, _)| found);
    let leaked: &'static str = Box::leak(miner.clone().into_boxed_str());
    timed("events: one miner 50", events(Some(leaked), None), Vec::len)?;
    timed("events: refused 50", events(None, Some("refused")), Vec::len)?;
    timed("events: dropped 50", events(None, Some("dropped")), Vec::len)?;
    let tasks = |filter: TaskFilter| move || validations::recent(conn, &filter);
    let filter = |f: fn(&mut TaskFilter)| {
        let mut filter = TaskFilter { before: Some(until), limit: 50, ..TaskFilter::default() };
        f(&mut filter);
        filter
    };
    timed("tasks: newest 50", tasks(filter(|_| {})), Vec::len)?;
    timed("tasks: fail 50", tasks(filter(|f| f.verdict = Some("fail".into()))), Vec::len)?;
    timed("tasks: void 50", tasks(filter(|f| f.verdict = Some("void".into()))), Vec::len)?;
    timed("tasks: embed 50", tasks(filter(|f| f.kind = Some(budgets::EMBED.into()))), Vec::len)?;
    timed("tasks: one validator 50", tasks(filter(|f| f.validator = Some(VALIDATORS[0].into()))), Vec::len)?;
    let mut by_miner = filter(|_| {});
    by_miner.miner = Some(miner.clone());
    timed("tasks: one miner 50", tasks(by_miner), Vec::len)?;
    let mut miner_fails = filter(|f| f.verdict = Some("fail".into()));
    miner_fails.miner = Some(miner.clone());
    timed("tasks: one miner fail 50", tasks(miner_fails), Vec::len)?;
    timed(
        "task page reads",
        || {
            let uploads = validations::uploads(conn, &checked)?;
            let key = uploads[0]["upload_key"].as_str().unwrap_or_default().to_string();
            validations::votes_on(conn, &checked, &key)?;
            validations::urls(conn, &checked)?;
            Ok(uploads)
        },
        Vec::len,
    )?;
    Ok(())
}

/// Counts the log into the tallies as a first start does, pass by pass at the janitor's size (`PASS_ROWS`).
fn tally(dir: &Path) -> Result<()> {
    let mut conn = Connection::open(dir.join(db::FILE))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    let started = Instant::now();
    hours::create(&conn)?;
    println!("create the tally tables: {:.1} ms", started.elapsed().as_secs_f64() * 1000.0);
    let size: i64 = std::env::var("PASS_ROWS").ok().and_then(|n| n.parse().ok()).unwrap_or(5_000);
    for name in ["claims", "checks"] {
        let (mut passes, mut rows, mut first, mut longest, started) = (0, 0, 0.0, 0.0f64, Instant::now());
        loop {
            let pass = Instant::now();
            let tx = conn.transaction()?;
            let counted = if name == "claims" { hours::roll_up_claims(&tx, size, now())? } else { hours::roll_up_checks(&tx, size, now())? };
            tx.commit()?;
            let took = pass.elapsed().as_secs_f64() * 1000.0;
            if counted == 0 {
                break;
            }
            if passes == 0 {
                first = took;
            } else {
                longest = longest.max(took);
            }
            passes += 1;
            rows += counted;
        }
        let total = started.elapsed().as_secs_f64();
        println!(
            "{name}: {rows} rows in {passes} passes of {size}, {total:.1} s, first pass {first:.1} ms, then {:.1} ms on average, longest {longest:.1} ms",
            (total * 1000.0 - first) / (passes - 1).max(1) as f64
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(command), Some(dir)) = (args.next(), args.next()) else { bail!("usage: log_reads build <dir> [days] | tally <dir> | time <dir>") };
    match command.as_str() {
        "build" => build(Path::new(&dir), args.next().map_or(Ok(7), |days| days.parse())?),
        "tally" => tally(Path::new(&dir)),
        "time" => time(Path::new(&dir)),
        other => bail!("unknown command {other}"),
    }
}

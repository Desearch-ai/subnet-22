//! Rounds as SQLite keeps them: each batch's count and hash; the URLs wait in Redis until the round's tasks are queued.

use std::collections::HashMap;

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde_json::{json, Map, Value};

use crate::db::{optional_real, real};
use crate::rounds::{Batch, Round, Url};

const COLUMNS: &str = "round_id, kind, manifest_hash, seed_block, opened_at, seed, serve_order, closed_at, batches";

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS rounds (
            round_id      TEXT PRIMARY KEY,
            kind          TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            seed_block    INTEGER NOT NULL,
            opened_at     REAL NOT NULL,
            seed          TEXT,
            serve_order   TEXT NOT NULL DEFAULT '[]',
            closed_at     REAL,
            batches       TEXT NOT NULL,
            filled_at     REAL
        );
        CREATE INDEX IF NOT EXISTS rounds_opened ON rounds (opened_at);
        CREATE INDEX IF NOT EXISTS rounds_pending ON rounds (seed, closed_at);
        CREATE INDEX IF NOT EXISTS rounds_unfilled ON rounds (opened_at)
            WHERE seed IS NOT NULL AND filled_at IS NULL AND closed_at IS NULL;
        CREATE INDEX IF NOT EXISTS rounds_open_filled ON rounds (round_id)
            WHERE seed IS NOT NULL AND filled_at IS NOT NULL AND closed_at IS NULL;
        CREATE INDEX IF NOT EXISTS rounds_closed ON rounds (closed_at)
            WHERE closed_at IS NOT NULL;
        CREATE TABLE IF NOT EXISTS retention (name TEXT PRIMARY KEY, mark REAL NOT NULL);
        ",
    )?;
    Ok(())
}

/// A batch as the rounds table keeps it.
pub fn stored(batch: &Batch) -> Value {
    let urls: Vec<Value> = batch.urls.iter().map(|u| json!([u.host, u.url])).collect();
    let mut entry = Map::new();
    entry.insert("urls".into(), urls.into());
    entry.insert("extra".into(), Value::Object(batch.extra.clone()));
    if let Some(sealed) = &batch.sealed {
        entry.insert("sealed".into(), sealed.clone());
    }
    Value::Object(entry)
}

fn stored_batches(round: &Round) -> String {
    let batches: Map<String, Value> = round.batches.iter().map(|b| (b.batch_id.clone(), stored(b))).collect();
    Value::Object(batches).to_string()
}

fn batch_of(batch_id: &str, entry: &Value) -> Batch {
    let urls = entry["urls"]
        .as_array()
        .map(|urls| {
            urls.iter().map(|pair| Url { host: pair[0].as_str().unwrap_or_default().into(), url: pair[1].as_str().unwrap_or_default().into() }).collect()
        })
        .unwrap_or_default();
    let extra = entry["extra"].as_object().cloned().unwrap_or_default();
    Batch { batch_id: batch_id.into(), urls, extra, sealed: entry.get("sealed").filter(|s| !s.is_null()).cloned() }
}

fn round_of(row: &Row) -> rusqlite::Result<Round> {
    let order: String = row.get(6)?;
    let batches: String = row.get(8)?;
    let batches: Map<String, Value> = serde_json::from_str(&batches).unwrap_or_default();
    Ok(Round {
        round_id: row.get(0)?,
        kind: row.get(1)?,
        manifest_hash: row.get(2)?,
        seed_block: row.get(3)?,
        opened_at: real(row, 4)?,
        seed: row.get(5)?,
        order: serde_json::from_str(&order).unwrap_or_default(),
        closed_at: optional_real(row, 7)?,
        batches: batches.iter().map(|(id, entry)| batch_of(id, entry)).collect(),
    })
}

pub fn save(conn: &Connection, round: &Round) -> Result<()> {
    conn.execute(
        &format!(
            "INSERT INTO rounds ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (round_id) DO UPDATE SET seed = excluded.seed, serve_order = excluded.serve_order, closed_at = excluded.closed_at"
        ),
        params![
            round.round_id,
            round.kind,
            round.manifest_hash,
            round.seed_block,
            round.opened_at,
            round.seed,
            serde_json::to_string(&round.order)?,
            round.closed_at,
            stored_batches(round)
        ],
    )?;
    Ok(())
}

pub fn get(conn: &Connection, round_id: &str) -> Result<Option<Round>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM rounds WHERE round_id = ?"), [round_id], round_of).optional()?)
}

/// Up to `limit` rounds not revealed yet whose seed block `by_block` has reached, oldest first.
pub fn unrevealed(conn: &Connection, by_block: i64, limit: i64) -> Result<Vec<Round>> {
    let mut statement = conn.prepare(&format!("SELECT {COLUMNS} FROM rounds WHERE seed IS NULL AND seed_block <= ? ORDER BY opened_at LIMIT ?"))?;
    let rounds = statement.query_map([by_block, limit], round_of)?;
    Ok(rounds.collect::<rusqlite::Result<_>>()?)
}

pub fn unrevealed_tasks(conn: &Connection, kind: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT COALESCE(SUM((SELECT COUNT(*) FROM json_each(batches))), 0) FROM rounds WHERE seed IS NULL AND kind = ?", [kind], |row| {
        row.get(0)
    })?)
}

pub fn mark_revealed(conn: &Connection, round: &Round) -> Result<()> {
    conn.execute("UPDATE rounds SET seed = ?, serve_order = ? WHERE round_id = ?", params![round.seed, serde_json::to_string(&round.order)?, round.round_id])?;
    Ok(())
}

/// Revealed rounds whose tasks never reached the queue.
pub fn unfilled(conn: &Connection) -> Result<Vec<Round>> {
    let mut statement = conn.prepare(&format!(
        "SELECT {COLUMNS} FROM rounds INDEXED BY rounds_unfilled WHERE seed IS NOT NULL AND filled_at IS NULL AND closed_at IS NULL ORDER BY opened_at"
    ))?;
    let rounds = statement.query_map([], round_of)?;
    Ok(rounds.collect::<rusqlite::Result<_>>()?)
}

/// The round's batches with their URLs and page lists dropped, keeping each batch's count and hash.
pub fn sealed_batches(round: &Round) -> String {
    let batches: Map<String, Value> = round.batches.iter().map(|b| (b.batch_id.clone(), stored(&b.queued()))).collect();
    Value::Object(batches).to_string()
}

/// Once Redis holds a round's tasks, it keeps only `sealed_batches`.
pub fn mark_filled(conn: &Connection, round_id: &str, sealed_batches: &str, at: f64) -> Result<()> {
    conn.execute("UPDATE rounds SET filled_at = ?, batches = ? WHERE round_id = ?", params![at, sealed_batches, round_id])?;
    Ok(())
}

pub fn open_revealed(conn: &Connection) -> Result<Vec<String>> {
    let mut statement =
        conn.prepare("SELECT round_id FROM rounds INDEXED BY rounds_open_filled WHERE seed IS NOT NULL AND filled_at IS NOT NULL AND closed_at IS NULL")?;
    let ids = statement.query_map([], |row| row.get(0))?;
    Ok(ids.collect::<rusqlite::Result<_>>()?)
}

/// The newest revealed round of each kind that is still open.
pub fn latest_revealed(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut statement = conn.prepare("SELECT kind, round_id FROM rounds WHERE seed IS NOT NULL AND closed_at IS NULL ORDER BY opened_at")?;
    let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// How long a closed round stays in SQLite.
pub const KEPT_S: f64 = 7.0 * 86_400.0;
/// Where the walk dropping closed rounds' URLs and page lists has reached.
const SEAL_MARK: &str = "round_lists";

/// Up to `limit` rounds closed from `from` on and before `to`, oldest first, cut at `bytes` of batches but never under two, so a walk that starts again at its last round moves on.
fn closed_between(conn: &Connection, from: f64, to: f64, limit: i64, bytes: i64) -> Result<Vec<(String, f64)>> {
    let mut statement = conn.prepare(
        "SELECT round_id, closed_at, octet_length(batches) FROM rounds INDEXED BY rounds_closed WHERE closed_at >= ? AND closed_at < ? ORDER BY closed_at LIMIT ?",
    )?;
    let rows = statement.query_map(params![from, to, limit], |row| Ok((row.get::<_, String>(0)?, real(row, 1)?, row.get::<_, i64>(2)?)))?;
    let (mut picked, mut read) = (Vec::new(), 0);
    for row in rows {
        let (round_id, closed_at, size) = row?;
        if picked.len() > 1 && read + size > bytes {
            break;
        }
        read += size;
        picked.push((round_id, closed_at));
    }
    Ok(picked)
}

/// Deletes the oldest rounds closed a week before `now`, at most `limit` of them and `bytes` of their batches; returns how many.
pub fn delete_expired(conn: &Connection, now: f64, limit: i64, bytes: i64) -> Result<usize> {
    let expired = closed_between(conn, 0.0, now - KEPT_S, limit, bytes)?;
    for (round_id, _) in &expired {
        conn.execute("DELETE FROM rounds WHERE round_id = ?", [round_id])?;
    }
    Ok(expired.len())
}

/// Drops the URLs and page lists rounds closed in the last week still keep, from the sealing mark on, at most `limit` rounds and `bytes` of batches; returns how many it rewrote and where `mark_sealed` moves the mark.
pub fn seal_closed(conn: &Connection, now: f64, limit: i64, bytes: i64) -> Result<(usize, Option<f64>)> {
    let mark: f64 = conn.query_row("SELECT COALESCE((SELECT mark FROM retention WHERE name = ?), 0)", [SEAL_MARK], |row| real(row, 0))?;
    let walked = closed_between(conn, mark.max(now - KEPT_S), now, limit, bytes)?;
    let mut sealed = 0;
    for (round_id, _) in &walked {
        let batches: String = conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [round_id], |row| row.get(0))?;
        let batches: Map<String, Value> = match serde_json::from_str(&batches) {
            Ok(batches) => batches,
            Err(error) => {
                eprintln!("round={round_id} keeps batches that do not parse ({error}); left as they are");
                continue;
            }
        };
        let queued: Map<String, Value> = batches.iter().map(|(id, entry)| (id.clone(), stored(&batch_of(id, entry).queued()))).collect();
        if queued != batches {
            conn.execute("UPDATE rounds SET batches = ? WHERE round_id = ?", params![Value::Object(queued).to_string(), round_id])?;
            sealed += 1;
        }
    }
    Ok((sealed, walked.last().map(|(_, closed_at)| *closed_at)))
}

/// Where the next `seal_closed` starts; the round closed at `closed_at` is looked at again, harmlessly.
pub fn mark_sealed(conn: &Connection, closed_at: f64) -> Result<()> {
    conn.execute("INSERT INTO retention VALUES (?, ?) ON CONFLICT (name) DO UPDATE SET mark = excluded.mark", params![SEAL_MARK, closed_at])?;
    Ok(())
}

pub fn close(conn: &Connection, round_id: &str, at: f64) -> Result<()> {
    conn.execute("UPDATE rounds SET closed_at = ? WHERE round_id = ?", params![at, round_id])?;
    Ok(())
}

pub fn recent(conn: &Connection, limit: i64) -> Result<Vec<Value>> {
    let mut statement =
        conn.prepare("SELECT round_id, kind, manifest_hash, seed_block, opened_at, seed IS NOT NULL, closed_at FROM rounds ORDER BY opened_at DESC LIMIT ?")?;
    let rows = statement.query_map([limit], |row| {
        Ok(json!({
            "round_id": row.get::<_, String>(0)?,
            "kind": row.get::<_, String>(1)?,
            "manifest_hash": row.get::<_, String>(2)?,
            "seed_block": row.get::<_, i64>(3)?,
            "opened_at": real(row, 4)?,
            "revealed": row.get::<_, bool>(5)?,
            "closed_at": optional_real(row, 6)?,
        }))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        conn
    }

    fn round(round_id: &str, batches: usize, seed_block: i64, opened_at: f64, seed: Option<&str>) -> Round {
        let batches = (0..batches)
            .map(|n| Batch::new(format!("b{n}"), vec![Url { host: "site.example".into(), url: "https://site.example/".into() }], Map::new()))
            .collect();
        Round {
            round_id: round_id.into(),
            batches,
            manifest_hash: "h".into(),
            seed_block,
            opened_at,
            kind: "crawl".into(),
            seed: seed.map(String::from),
            order: Vec::new(),
            closed_at: None,
        }
    }

    #[test]
    fn open_rounds_are_found_through_their_own_indexes() {
        let conn = store();
        for (n, (seed, filled, closed)) in
            [(None, None, None), (Some("s"), None, None), (Some("s"), Some(1.0), None), (Some("s"), Some(1.0), Some(2.0))].into_iter().enumerate()
        {
            let saved = round(&format!("r{n}"), 1, 1, n as f64, seed);
            save(&conn, &saved).unwrap();
            if let Some(at) = filled {
                mark_filled(&conn, &saved.round_id, &sealed_batches(&saved), at).unwrap();
            }
            if let Some(at) = closed {
                close(&conn, &format!("r{n}"), at).unwrap();
            }
        }
        assert_eq!(unfilled(&conn).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["r1"]);
        assert_eq!(open_revealed(&conn).unwrap(), ["r2"]);
    }

    #[test]
    fn only_rounds_whose_seed_block_has_come_are_loaded_for_reveal() {
        let conn = store();
        save(&conn, &round("due", 3, 10, 1.0, None)).unwrap();
        save(&conn, &round("later", 3, 20, 2.0, None)).unwrap();
        assert_eq!(unrevealed(&conn, 15, 10).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due"]);
        assert_eq!(unrevealed(&conn, i64::MAX, 10).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due", "later"]);
        assert_eq!(unrevealed(&conn, i64::MAX, 1).unwrap().into_iter().map(|r| r.round_id).collect::<Vec<_>>(), ["due"]);
        assert_eq!((unrevealed_tasks(&conn, "crawl").unwrap(), unrevealed_tasks(&conn, "embed").unwrap()), (6, 0));
    }

    #[test]
    fn a_filled_round_keeps_its_manifest_without_its_urls() {
        let conn = store();
        let urls: Vec<Url> = (0..2500).map(|i| Url { host: "example.com".into(), url: format!("https://example.com/{i}") }).collect();
        let mut filled = crate::rounds::open_batches(crate::rounds::pack(urls, 1000), 100, "crawl", 1_800_000_000.0);
        crate::rounds::reveal(&mut filled, &"00".repeat(32));
        save(&conn, &filled).unwrap();
        close(&conn, &filled.round_id, 1_800_000_100.0).unwrap();
        let before = get(&conn, &filled.round_id).unwrap().unwrap().public_view();
        mark_filled(&conn, &filled.round_id, &sealed_batches(&filled), 1_800_000_010.0).unwrap();
        let batches: String = conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [&filled.round_id], |row| row.get(0)).unwrap();
        assert_eq!(batches.matches("https://example.com/").count(), 0);
        assert_eq!(get(&conn, &filled.round_id).unwrap().unwrap().public_view(), before);
    }

    /// An embed round as the janitor opens it: one batch per input, each with its input's pages.
    fn embed_round(round_id: &str, inputs: usize, pages: usize, opened_at: f64) -> Round {
        let batches = (0..inputs)
            .map(|n| {
                let pages: Vec<Value> = (0..pages)
                    .map(|p| json!({"page_key": format!("pages/{n}-{p}.zst"), "content_sha1": format!("{p:040x}"), "url": format!("https://site.example/{n}/{p}"), "host": "site.example"}))
                    .collect();
                let urls = pages.iter().map(|p| Url { host: "site.example".into(), url: p["url"].as_str().unwrap().into() }).collect();
                let mut extra = Map::new();
                extra.insert("input_key".into(), format!("embed-inputs/{n}.parquet").into());
                extra.insert("input_sha256".into(), format!("{n:064x}").into());
                extra.insert("texts".into(), (pages.len() * 6).into());
                extra.insert("model".into(), "desearch-4b-v1".into());
                extra.insert("pages".into(), pages.into());
                Batch::new(format!("b{n}"), urls, extra)
            })
            .collect();
        let mut round = crate::rounds::open_batches(batches, 100, "embed", opened_at);
        round.round_id = round_id.into();
        crate::rounds::reveal(&mut round, &"00".repeat(32));
        round
    }

    /// One sealing pass as the janitor runs it; returns how many rounds were deleted and how many rewritten.
    fn pass(conn: &Connection, now: f64, limit: i64, bytes: i64) -> (usize, usize) {
        let deleted = delete_expired(conn, now, limit, bytes).unwrap();
        let (sealed, walked) = seal_closed(conn, now, limit, bytes).unwrap();
        if let Some(closed_at) = walked {
            mark_sealed(conn, closed_at).unwrap();
        }
        (deleted, sealed)
    }

    #[test]
    fn a_round_whose_batches_do_not_parse_is_passed_over_and_the_rest_are_sealed() {
        let conn = store();
        let now = 1_800_000_000.0;
        for (n, round_id) in ["broken", "after"].iter().enumerate() {
            let round = embed_round(round_id, 1, 50, now);
            save(&conn, &round).unwrap();
            close(&conn, round_id, now - 3_600.0 + n as f64).unwrap();
        }
        conn.execute("UPDATE rounds SET batches = '{not json' WHERE round_id = 'broken'", []).unwrap();
        assert_eq!(pass(&conn, now, 10, 16 << 20), (0, 1));
        assert_eq!(stored_text(&conn, "broken").as_deref(), Some("{not json"));
        assert_eq!(stored_text(&conn, "after").unwrap().matches("page_key").count(), 0);
        assert_eq!(delete_expired(&conn, now + KEPT_S, 10, 16 << 20).unwrap(), 2, "it still goes after a week");
    }

    fn stored_text(conn: &Connection, round_id: &str) -> Option<String> {
        conn.query_row("SELECT batches FROM rounds WHERE round_id = ?", [round_id], |row| row.get(0)).optional().unwrap()
    }

    #[test]
    fn a_queued_embed_round_keeps_its_manifest_and_input_without_its_page_lists() {
        let conn = store();
        let round = embed_round("r", 2, 300, 1_800_000_000.0);
        save(&conn, &round.sealed()).unwrap();
        let unqueued = unfilled(&conn).unwrap();
        assert_eq!(unqueued[0].batches[0].extra["pages"].as_array().unwrap().len(), 300, "an unqueued round keeps the pages its tasks need");
        let before = get(&conn, "r").unwrap().unwrap().public_view();
        mark_filled(&conn, "r", &sealed_batches(&round), 1_800_000_010.0).unwrap();
        let text = stored_text(&conn, "r").unwrap();
        assert_eq!((text.matches("page_key").count(), text.matches("https://site.example/").count()), (0, 0));
        let kept = get(&conn, "r").unwrap().unwrap();
        assert_eq!(kept.public_view(), before);
        let input = &kept.batch("b1").unwrap().extra;
        assert_eq!((input["input_sha256"].clone(), input["input_key"].clone()), (json!(format!("{:064x}", 1)), json!("embed-inputs/1.parquet")));
        assert_eq!(crate::proofs::manifest_hash(&kept.manifest(), kept.seed_block), kept.manifest_hash);
    }

    #[test]
    fn closed_rounds_lose_their_urls_and_page_lists_and_go_a_week_after_closing() {
        let conn = store();
        let now = 1_800_000_000.0;
        let urls: Vec<Url> = (0..2500).map(|i| Url { host: "example.com".into(), url: format!("https://example.com/{i}") }).collect();
        let mut crawl = crate::rounds::open_batches(crate::rounds::pack(urls, 1000), 100, "crawl", now);
        crawl.round_id = "crawl".into();
        crate::rounds::reveal(&mut crawl, &"00".repeat(32));
        let (expired, embed, open) = (embed_round("expired", 2, 50, now), embed_round("embed", 2, 50, now), embed_round("open", 2, 50, now));
        for round in [&crawl, &expired, &embed, &open] {
            save(&conn, round).unwrap();
        }
        close(&conn, "expired", now - KEPT_S - 60.0).unwrap();
        close(&conn, "embed", now - 3.0 * 86_400.0).unwrap();
        close(&conn, "crawl", now - 60.0).unwrap();
        let views: Vec<Map<String, Value>> = ["embed", "crawl"].iter().map(|id| get(&conn, id).unwrap().unwrap().public_view()).collect();

        assert_eq!(pass(&conn, now, 10, 16 << 20), (1, 2));
        assert_eq!(stored_text(&conn, "expired"), None);
        for (n, id) in ["embed", "crawl"].iter().enumerate() {
            let text = stored_text(&conn, id).unwrap();
            assert_eq!((text.matches("page_key").count(), text.matches("https://").count()), (0, 0), "{id}");
            assert_eq!(get(&conn, id).unwrap().unwrap().public_view(), views[n]);
        }
        assert_eq!(stored_text(&conn, "open").unwrap().matches("page_key").count(), 100, "an open round keeps its pages");
        assert_eq!(pass(&conn, now, 10, 16 << 20), (0, 0), "the last round is looked at again, unchanged");
        assert_eq!(pass(&conn, now + 4.0 * 86_400.0 + 120.0, 10, 16 << 20), (1, 0));
        assert_eq!(stored_text(&conn, "embed"), None);
        assert!(stored_text(&conn, "crawl").is_some());
    }

    #[test]
    fn a_sealing_pass_reads_a_bounded_share_of_batches_but_always_moves() {
        let conn = store();
        let now = 1_800_000_000.0;
        for n in 0..5 {
            let round = embed_round(&format!("r{n}"), 1, 400, now);
            save(&conn, &round).unwrap();
            close(&conn, &round.round_id, now - 3_600.0 + n as f64).unwrap();
        }
        assert_eq!(pass(&conn, now, 10, 1).1, 2, "two rounds past the budget still go");
        assert_eq!(pass(&conn, now, 10, 1).1, 1, "starting again on the sealed round the mark is on, it still takes the next");
        assert_eq!(pass(&conn, now, 10, 1 << 30).1, 2);
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM rounds WHERE batches LIKE '%page_key%'", [], |row| row.get::<_, i64>(0)).unwrap(), 0);
    }
}

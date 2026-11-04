//! One hash-chained log of everything that changes the accounts: signed requests, the API's answers and decisions, and the chain blocks it used.
//!
//! Every entry names the hash of the one before it and is written in the same SQLite transaction as the change it records, so the log and
//! the accounts never part. The janitor publishes the entries in numbered signed files to the uploads bucket, and `anchor` commits the head
//! to the chain; anyone can then check that no entry was altered, dropped or forked, and `replay` rebuilds the accounts from the inputs.

use std::collections::BTreeSet;

use anyhow::{anyhow, Result};
use axum::http::HeaderMap;
use bytes::Bytes;
use desearch::hotkey::{public_of, signing_payload, verify};
use desearch::time::now;
use redis::AsyncCommands;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::proofs::sha256;
use crate::py::canonical;
use crate::state::State;
use crate::storage::put_json;

pub const VERSION: i64 = 1;
/// The `prev` of the first entry.
pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// Request bodies up to this size are kept in their entry; larger ones in a blob the entry names by its SHA-256.
pub const INLINE_BYTES: usize = 2048;
pub const LATEST_KEY: &str = "log/journal/latest.json";
/// Hashes of authenticated requests waiting to be written into the log, `<hash> <at>` each.
pub const REQUESTS: &str = "journal:requests";
/// Entries published per file at most.
pub const SEGMENT_ENTRIES: i64 = 5000;
const REQUESTS_AT_ONCE: isize = 20_000;

pub fn segment_key(segment: i64) -> String {
    format!("log/journal/seq/{segment:012}.json")
}

pub fn blob_key(sha256: &str) -> String {
    format!("log/journal/blobs/{sha256}")
}

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS journal (
            seq   INTEGER PRIMARY KEY,
            at    REAL NOT NULL,
            kind  TEXT NOT NULL,
            hash  TEXT NOT NULL,
            entry TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS journal_blobs (
            sha256    TEXT PRIMARY KEY,
            seq       INTEGER NOT NULL,
            body      BLOB NOT NULL,
            published INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS journal_blobs_seq ON journal_blobs (published, seq);
        CREATE TABLE IF NOT EXISTS journal_segments (
            segment    INTEGER PRIMARY KEY,
            first      INTEGER NOT NULL,
            last       INTEGER NOT NULL,
            head       TEXT NOT NULL,
            written_at REAL NOT NULL
        );
        ",
    )?;
    Ok(())
}

/// A body kept beside the log, named by the SHA-256 of its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Blob {
    pub sha256: String,
    pub body: Vec<u8>,
}

impl Blob {
    pub fn of(body: Vec<u8>) -> Blob {
        Blob { sha256: sha256(&body), body }
    }

    pub fn reference(&self) -> Value {
        json!({"blob": self.sha256, "bytes": self.body.len()})
    }
}

/// One entry before it is chained: what came in, what the API made of it, and when.
#[derive(Clone, Debug)]
pub struct Record {
    pub kind: &'static str,
    pub at: f64,
    pub input: Value,
    pub output: Value,
    pub blobs: Vec<Blob>,
}

impl Record {
    pub fn new(kind: &'static str, at: f64, input: Value, output: Value) -> Record {
        Record { kind, at, input, output, blobs: Vec::new() }
    }

    pub fn with_blobs(mut self, blobs: Vec<Blob>) -> Record {
        self.blobs.extend(blobs);
        self
    }
}

/// An entry's hash: SHA-256 of its canonical JSON, the way Python's `json.dumps(sort_keys=True, separators=(",", ":"))` writes it.
pub fn hash_of(entry: &Value) -> String {
    sha256(&canonical(entry))
}

/// The newest entry's number and hash: of this copy's log, else of the head it adopted, else 0 and the genesis hash.
pub fn head(conn: &Connection) -> Result<(i64, String)> {
    if let Some(found) = conn.query_row("SELECT seq, hash FROM journal ORDER BY seq DESC LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?))).optional()? {
        return Ok(found);
    }
    Ok(conn
        .query_row("SELECT last, head FROM journal_segments ORDER BY segment DESC LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?
        .unwrap_or((0, GENESIS.into())))
}

/// A copy appointed without a handover goes on from the head the last holder published (`log/journal/latest.json`): what it logged
/// while it waited, never published, is dropped, and its numbers and links follow on from that head.
pub fn adopt(conn: &Connection, segment: i64, last: i64, head: &str) -> Result<()> {
    conn.execute_batch("DELETE FROM journal; DELETE FROM journal_blobs; DELETE FROM journal_segments;")?;
    conn.execute("INSERT INTO journal_segments (segment, first, last, head, written_at) VALUES (?, ?, ?, ?, ?)", params![segment, last, last, head, now()])?;
    Ok(())
}

/// Chains `record` onto the log; call it inside the transaction that makes the change it records.
pub fn append(conn: &Connection, record: Record) -> Result<(i64, String)> {
    let (last, prev) = head(conn)?;
    let seq = last + 1;
    let entry = json!({
        "v": VERSION,
        "seq": seq,
        "prev": prev,
        "at": record.at,
        "kind": record.kind,
        "input": record.input,
        "output": record.output,
    });
    let text = String::from_utf8(canonical(&entry)).expect("canonical JSON is UTF-8");
    let hash = sha256(text.as_bytes());
    conn.execute("INSERT INTO journal (seq, at, kind, hash, entry) VALUES (?, ?, ?, ?, ?)", params![seq, record.at, record.kind, hash, text])?;
    // A blob published before goes up again with this entry: the bucket lets objects go after a day, and this entry still names it.
    for blob in record.blobs {
        conn.execute(
            "INSERT INTO journal_blobs (sha256, seq, body) VALUES (?, ?, ?)
             ON CONFLICT (sha256) DO UPDATE SET seq = excluded.seq, body = excluded.body, published = 0 WHERE journal_blobs.published = 1",
            params![blob.sha256, seq, blob.body],
        )?;
    }
    Ok((seq, hash))
}

/// Entries after `after`, oldest first, with their hashes.
pub fn entries_after(conn: &Connection, after: i64, limit: i64) -> Result<Vec<(Value, String)>> {
    let mut statement = conn.prepare("SELECT entry, hash FROM journal WHERE seq > ? ORDER BY seq LIMIT ?")?;
    let rows = statement.query_map(params![after, limit], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
    rows.map(|row| {
        let (text, hash) = row?;
        Ok((serde_json::from_str(&text)?, hash))
    })
    .collect()
}

pub fn blob(conn: &Connection, sha256: &str) -> Result<Option<Vec<u8>>> {
    Ok(conn.query_row("SELECT body FROM journal_blobs WHERE sha256 = ?", [sha256], |row| row.get(0)).optional()?)
}

/// A request body as an entry keeps it: inline while it is short UTF-8 text, else a blob.
pub fn body_of(body: &[u8]) -> (Value, Option<Blob>) {
    match std::str::from_utf8(body) {
        Ok(text) if body.len() <= INLINE_BYTES => (text.into(), None),
        _ => {
            let blob = Blob::of(body.to_vec());
            (blob.reference(), Some(blob))
        }
    }
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or_default()
}

/// One signed request's identity, as receipts and the witness relays name it (`auth::request_hash`).
pub fn request_hash(method: &str, path: &str, body: &[u8], timestamp: &str, nonce: &str, hotkey: &str, signature: &str) -> String {
    crate::auth::request_hash(method, path, body, hotkey, timestamp, nonce, signature)
}

/// A signed request as the log keeps it: everything needed to check the caller's signature again, and its hash.
pub fn signed_request(headers: &HeaderMap, method: &str, path: &str, body: &[u8]) -> (Value, Vec<Blob>) {
    let (hotkey, timestamp, nonce, signature) =
        (header(headers, "X-Hotkey"), header(headers, "X-Timestamp"), header(headers, "X-Nonce"), header(headers, "X-Signature"));
    let (kept, blob) = body_of(body);
    let request = json!({
        "method": method,
        "path": path,
        "hotkey": hotkey,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
        "body": kept,
        "request": request_hash(method, path, body, timestamp, nonce, hotkey, signature),
    });
    (request, blob.into_iter().collect())
}

/// The body a logged request was signed over, fetching a blob through `blobs`.
pub fn request_body(request: &Value, blobs: &dyn Fn(&str) -> Option<Vec<u8>>) -> Option<Vec<u8>> {
    match &request["body"] {
        Value::String(text) => Some(text.clone().into_bytes()),
        Value::Object(reference) => {
            let sha = reference.get("blob")?.as_str()?;
            blobs(sha).filter(|body| crate::proofs::sha256(body) == sha)
        }
        _ => None,
    }
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

/// Whether the hotkey a logged request names really signed it; it proves the API did not make the request up.
pub fn request_signed(request: &Value, blobs: &dyn Fn(&str) -> Option<Vec<u8>>) -> bool {
    let field = |name: &str| request[name].as_str().unwrap_or_default();
    let Some(body) = request_body(request, blobs) else { return false };
    let (Some(public), Some(signature)) = (public_of(field("hotkey")), decode_hex(field("signature"))) else { return false };
    let payload = signing_payload(field("method"), field("path"), &body, field("timestamp"), field("nonce"));
    verify(&public, &payload, &signature)
        && field("request") == request_hash(field("method"), field("path"), &body, field("timestamp"), field("nonce"), field("hotkey"), field("signature"))
}

/// What is wrong with a stretch of the log.
#[derive(Clone, Debug, PartialEq)]
pub enum Break {
    /// An entry's number is not the one after the last.
    Gap {
        expected: i64,
        found: i64,
    },
    /// An entry does not name the hash of the one before it: something before it was altered, dropped or forked.
    Link {
        seq: i64,
        expected: String,
        found: String,
    },
    /// A file's head, first or last does not match its entries.
    Head {
        segment: i64,
        detail: String,
    },
    /// A file is not signed by the expected key.
    Signature {
        segment: i64,
    },
    /// The anchored hash at this number is not the log's.
    Anchor {
        seq: i64,
        anchored: String,
        found: String,
    },
    Malformed(String),
}

impl std::fmt::Display for Break {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Break::Gap { expected, found } => write!(f, "entry {found} where {expected} was due: entries are missing"),
            Break::Link { seq, expected, found } => write!(f, "entry {seq} names {found} as the one before it, which hashes to {expected}"),
            Break::Head { segment, detail } => write!(f, "file {segment}: {detail}"),
            Break::Signature { segment } => write!(f, "file {segment} is not signed by the expected key"),
            Break::Anchor { seq, anchored, found } => write!(f, "the chain anchors {anchored} at entry {seq}, the log has {found}"),
            Break::Malformed(why) => write!(f, "{why}"),
        }
    }
}

/// Checks that `entries` follow on from `(last, prev)` link by link; returns the new last number and head.
pub fn verify_chain(last: i64, prev: &str, entries: &[Value]) -> Result<(i64, String), Break> {
    let (mut last, mut prev) = (last, prev.to_string());
    for entry in entries {
        let seq = entry["seq"].as_i64().ok_or_else(|| Break::Malformed("an entry without a number".into()))?;
        if seq != last + 1 {
            return Err(Break::Gap { expected: last + 1, found: seq });
        }
        let named = entry["prev"].as_str().unwrap_or_default();
        if named != prev {
            return Err(Break::Link { seq, expected: prev, found: named.into() });
        }
        last = seq;
        prev = hash_of(entry);
    }
    Ok((last, prev))
}

/// The bytes a published file's signature covers: the file without its signature.
pub fn signed_payload(body: &Value) -> Vec<u8> {
    let mut unsigned = body.as_object().cloned().unwrap_or_default();
    unsigned.remove("signature");
    canonical(&Value::Object(unsigned))
}

pub fn signed_by(body: &Value, signer: &str) -> bool {
    let (Some(public), Some(signature)) = (public_of(signer), body["signature"].as_str().and_then(decode_hex)) else { return false };
    body["signer"] == signer && verify(&public, &signed_payload(body), &signature)
}

/// Checks one published file: signed by `signer`, its entries chained from `(last, prev)`, and its head and range those of its entries.
pub fn verify_segment(segment: &Value, signer: &str, last: i64, prev: &str) -> Result<(i64, String), Break> {
    let number = segment["segment"].as_i64().unwrap_or(-1);
    if !signed_by(segment, signer) {
        return Err(Break::Signature { segment: number });
    }
    let entries = segment["entries"].as_array().ok_or_else(|| Break::Malformed(format!("file {number} has no entries")))?;
    if segment["prev"].as_str() != Some(prev) {
        let found = segment["prev"].as_str().unwrap_or_default().to_string();
        return Err(Break::Link { seq: last + 1, expected: prev.into(), found });
    }
    let (end, head) = verify_chain(last, prev, entries)?;
    let detail = if segment["head"].as_str() != Some(head.as_str()) {
        Some(format!("names head {} but its entries hash to {head}", segment["head"]))
    } else if segment["first"].as_i64() != Some(last + 1) || segment["last"].as_i64() != Some(end) {
        Some(format!("names entries {}..{} but holds {}..{end}", segment["first"], segment["last"], last + 1))
    } else {
        None
    };
    match detail {
        Some(detail) => Err(Break::Head { segment: number, detail }),
        None => Ok((end, head)),
    }
}

/// The blobs entries name, so a reader knows which to fetch.
pub fn blobs_named(entries: &[Value]) -> BTreeSet<String> {
    fn walk(value: &Value, out: &mut BTreeSet<String>) {
        match value {
            Value::Object(map) => {
                if let (Some(Value::String(sha)), Some(Value::Number(_))) = (map.get("blob"), map.get("bytes")) {
                    out.insert(sha.clone());
                }
                map.values().for_each(|v| walk(v, out));
            }
            Value::Array(items) => items.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    entries.iter().for_each(|entry| walk(entry, &mut out));
    out
}

/// Counts an authenticated request; its hash reaches the log with the next `log_requests`.
pub async fn note_request(redis: &crate::state::Redis, hash: &str, at: f64) -> Result<()> {
    let _: i64 = redis.clone().rpush(REQUESTS, format!("{hash} {at}")).await?;
    Ok(())
}

/// Writes the hashes of the requests authenticated since the last call into one entry, with the block they reached, so a relay that
/// kept a signed request can show it was or was not let in.
pub async fn log_requests(state: &State) -> Result<usize> {
    let mut redis = state.redis.clone();
    let waiting: Vec<String> = redis.lrange(REQUESTS, 0, REQUESTS_AT_ONCE - 1).await?;
    if waiting.is_empty() {
        return Ok(0);
    }
    let block = state.seeds.current_block().await.ok();
    let mut rows = Vec::with_capacity(waiting.len());
    for line in &waiting {
        let mut parts = line.splitn(2, ' ');
        let (Some(hash), Some(at)) = (parts.next(), parts.next().and_then(|at| at.parse::<f64>().ok())) else { continue };
        rows.push(json!([hash, at]));
    }
    let count = rows.len();
    let listed = Value::Array(rows);
    let (kept, blob) = body_of(&canonical(&listed));
    let input = json!({"count": count, "requests": if blob.is_some() { kept } else { listed }, "block": block});
    let record = Record::new("requests", now(), input, Value::Null).with_blobs(blob.into_iter().collect());
    state.db.run(move |conn| append(conn, record)).await?;
    // Trimmed once logged, so a crash in between logs them again rather than never.
    let _: () = redis.ltrim(REQUESTS, waiting.len() as isize, -1).await?;
    Ok(count)
}

/// Writes the shares as the API serves them into the log, for a replay to compare with the shares its own accounts give.
pub async fn log_shares(state: &State) -> Result<()> {
    let as_of = now() - state.settings.ledger_delay;
    state
        .db
        .run(move |conn| {
            let pools = crate::budgets::shares(conn, crate::budgets::SHARE_WINDOW_H, as_of)?;
            let input = json!({"as_of": as_of, "window_hours": crate::budgets::SHARE_WINDOW_H});
            append(conn, Record::new("shares", now(), input, json!({"pools": pools})))?;
            Ok(())
        })
        .await
}

/// Publishes the entries not yet published as the next numbered signed file, after the blobs they name; None when there were none.
pub async fn publish(state: &State) -> Result<Option<i64>> {
    let (segment, last, prev) = state
        .db
        .run(|conn| {
            let found: Option<(i64, i64, String)> = conn
                .query_row("SELECT segment, last, head FROM journal_segments ORDER BY segment DESC LIMIT 1", [], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .optional()?;
            Ok(found.map_or((1, 0, GENESIS.to_string()), |(segment, last, head)| (segment + 1, last, head)))
        })
        .await?;
    let (entries, blobs) = state
        .db
        .run(move |conn| {
            let entries = entries_after(conn, last, SEGMENT_ENTRIES)?;
            let until = entries.last().and_then(|(entry, _)| entry["seq"].as_i64()).unwrap_or(last);
            let mut statement = conn.prepare("SELECT sha256, body FROM journal_blobs WHERE published = 0 AND seq <= ?")?;
            let blobs = statement.query_map([until], |row| Ok(Blob { sha256: row.get(0)?, body: row.get(1)? }))?.collect::<rusqlite::Result<Vec<_>>>()?;
            Ok((entries, blobs))
        })
        .await?;
    let Some((last_entry, head)) = entries.last().cloned() else { return Ok(None) };
    for blob in &blobs {
        state.storage.put(&blob_key(&blob.sha256), Bytes::from(blob.body.clone()), "application/octet-stream", None).await?;
    }
    let first = last + 1;
    let until = last_entry["seq"].as_i64().ok_or_else(|| anyhow!("an entry without a number"))?;
    let mut body = Map::new();
    body.insert("segment".into(), segment.into());
    body.insert("first".into(), first.into());
    body.insert("last".into(), until.into());
    body.insert("prev".into(), prev.into());
    body.insert("head".into(), head.clone().into());
    body.insert("written_at".into(), now().into());
    body.insert("entries".into(), entries.into_iter().map(|(entry, _)| entry).collect::<Vec<_>>().into());
    body.insert("signer".into(), state.signer().into());
    let signature = state.sign(&signed_payload(&Value::Object(body.clone())));
    body.insert("signature".into(), signature.into());
    put_json(&state.storage, &segment_key(segment), &Value::Object(body), None).await?;
    let mut latest = Map::new();
    latest.insert("segment".into(), segment.into());
    latest.insert("last".into(), until.into());
    latest.insert("head".into(), head.clone().into());
    latest.insert("signer".into(), state.signer().into());
    let signature = state.sign(&signed_payload(&Value::Object(latest.clone())));
    latest.insert("signature".into(), signature.into());
    put_json(&state.storage, LATEST_KEY, &Value::Object(latest), Some("no-store")).await?;
    let shas: Vec<String> = blobs.into_iter().map(|blob| blob.sha256).collect();
    state
        .db
        .run(move |conn| {
            conn.execute(
                "INSERT INTO journal_segments (segment, first, last, head, written_at) VALUES (?, ?, ?, ?, ?)",
                params![segment, first, until, head, now()],
            )?;
            // A published blob lives on in the bucket; only its name stays here.
            for sha in &shas {
                conn.execute("UPDATE journal_blobs SET published = 1, body = x'' WHERE sha256 = ?", [sha])?;
            }
            Ok(())
        })
        .await?;
    Ok(Some(segment))
}

/// Published entries older than `keep_s` leave SQLite; the newest always stays, so the chain goes on from it.
pub fn prune(conn: &Connection, now: f64, keep_s: f64) -> Result<usize> {
    let published: i64 = conn.query_row("SELECT COALESCE(MAX(last), 0) FROM journal_segments", [], |row| row.get(0))?;
    let (newest, _) = head(conn)?;
    let removed = conn.execute("DELETE FROM journal WHERE at < ? AND seq <= ? AND seq < ?", params![now - keep_s, published, newest])?;
    conn.execute("DELETE FROM journal_blobs WHERE published = 1 AND seq <= ? AND seq NOT IN (SELECT seq FROM journal)", [published])?;
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use desearch::hotkey::{auth_headers, Hotkey};

    fn store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        conn
    }

    fn logged(n: usize) -> (Connection, Vec<Value>) {
        let conn = store();
        for i in 0..n {
            append(&conn, Record::new("test", 1_800_000_000.0 + i as f64 * 0.137, json!({"i": i, "text": "naïve ✓"}), json!({"ok": i % 2 == 0}))).unwrap();
        }
        let entries = entries_after(&conn, 0, 100).unwrap().into_iter().map(|(entry, _)| entry).collect();
        (conn, entries)
    }

    #[test]
    fn each_entry_names_the_hash_of_the_one_before() {
        let (conn, entries) = logged(5);
        assert_eq!(entries[0]["prev"], GENESIS);
        for pair in entries.windows(2) {
            assert_eq!(pair[1]["prev"].as_str().unwrap(), hash_of(&pair[0]));
        }
        let (last, head_hash) = verify_chain(0, GENESIS, &entries).unwrap();
        assert_eq!((last, head_hash), head(&conn).unwrap());
    }

    #[test]
    fn an_entry_read_back_hashes_the_same() {
        let (conn, entries) = logged(3);
        for ((entry, stored), original) in entries_after(&conn, 0, 10).unwrap().iter().zip(&entries) {
            let back: Value = serde_json::from_str(&entry.to_string()).unwrap();
            assert_eq!(&hash_of(&back), stored);
            assert_eq!(hash_of(original), *stored);
        }
    }

    #[test]
    fn an_altered_entry_breaks_the_link_after_it() {
        let (_, mut entries) = logged(5);
        entries[2]["output"]["ok"] = false.into();
        assert!(matches!(verify_chain(0, GENESIS, &entries), Err(Break::Link { seq: 4, .. })));
    }

    #[test]
    fn a_dropped_entry_is_a_gap_and_a_renumbered_one_a_broken_link() {
        let (_, entries) = logged(5);
        let mut dropped = entries.clone();
        dropped.remove(2);
        assert_eq!(verify_chain(0, GENESIS, &dropped), Err(Break::Gap { expected: 3, found: 4 }));
        let mut renumbered = dropped.clone();
        for (i, entry) in renumbered.iter_mut().enumerate() {
            entry["seq"] = (i as i64 + 1).into();
        }
        assert!(matches!(verify_chain(0, GENESIS, &renumbered), Err(Break::Link { seq: 3, .. })));
    }

    #[test]
    fn a_fork_cannot_follow_on_from_the_head_others_saw() {
        let (_, entries) = logged(4);
        let (_, seen) = verify_chain(0, GENESIS, &entries[..2]).unwrap();
        let fork = store();
        for i in 0..4 {
            append(&fork, Record::new("test", 1.0, json!({"i": i, "forked": true}), Value::Null)).unwrap();
        }
        let forked: Vec<Value> = entries_after(&fork, 2, 10).unwrap().into_iter().map(|(e, _)| e).collect();
        assert!(matches!(verify_chain(2, &seen, &forked), Err(Break::Link { seq: 3, .. })));
        assert!(verify_chain(2, &seen, &entries[2..]).is_ok());
    }

    fn signed(mut body: Value, key: &Hotkey) -> Value {
        let signature: String = key.sign(&signed_payload(&body)).iter().map(|b| format!("{b:02x}")).collect();
        body["signature"] = signature.into();
        body
    }

    fn segment(entries: &[Value], key: &Hotkey) -> Value {
        let head = entries.last().map(hash_of).unwrap_or_else(|| GENESIS.into());
        let body = json!({
            "segment": 1, "first": 1, "last": entries.len(), "prev": GENESIS, "head": head,
            "written_at": 1.5, "entries": entries, "signer": key.ss58(),
        });
        signed(body, key)
    }

    #[test]
    fn a_published_file_is_checked_for_its_signer_head_and_links() {
        let key = Hotkey::from_uri("//TaskApi").unwrap();
        let (_, entries) = logged(4);
        let file = segment(&entries, &key);
        assert!(verify_segment(&file, &key.ss58(), 0, GENESIS).is_ok());
        let other = Hotkey::from_uri("//Mallory").unwrap();
        assert_eq!(verify_segment(&file, &other.ss58(), 0, GENESIS), Err(Break::Signature { segment: 1 }));
        let mut altered = file.clone();
        altered["entries"][1]["input"]["i"] = 99.into();
        assert_eq!(verify_segment(&altered, &key.ss58(), 0, GENESIS), Err(Break::Signature { segment: 1 }), "the signature covers the entries");
        let resigned = segment(&altered["entries"].as_array().unwrap().clone(), &key);
        assert!(matches!(verify_segment(&resigned, &key.ss58(), 0, GENESIS), Err(Break::Link { seq: 3, .. })), "and the links still break");
        let mut overstated = file.clone();
        overstated["last"] = 9.into();
        let overstated = signed(overstated, &key);
        assert!(matches!(verify_segment(&overstated, &key.ss58(), 0, GENESIS), Err(Break::Head { segment: 1, .. })));
        let mut wrong_head = file.clone();
        wrong_head["head"] = GENESIS.into();
        let wrong_head = signed(wrong_head, &key);
        assert!(matches!(verify_segment(&wrong_head, &key.ss58(), 0, GENESIS), Err(Break::Head { segment: 1, .. })));
    }

    #[test]
    fn a_logged_request_shows_who_signed_it() {
        let miner = Hotkey::from_uri("//miner").unwrap();
        let body = br#"{"count":2}"#;
        let mut headers = HeaderMap::new();
        for (name, value) in auth_headers(&miner, "POST", "/v1/tasks/claim", body, 1_800_000_000, &"a".repeat(32)) {
            headers.insert(name, value.parse().unwrap());
        }
        let none = |_: &str| None;
        let (request, blobs) = signed_request(&headers, "POST", "/v1/tasks/claim", body);
        assert!(blobs.is_empty() && request_signed(&request, &none));
        let mut forged = request.clone();
        forged["body"] = r#"{"count":3}"#.into();
        assert!(!request_signed(&forged, &none), "the API cannot change what the miner asked");
        let large = vec![b'x'; INLINE_BYTES + 1];
        let mut headers = HeaderMap::new();
        for (name, value) in auth_headers(&miner, "POST", "/v1/admin/enqueue", &large, 1_800_000_000, &"b".repeat(32)) {
            headers.insert(name, value.parse().unwrap());
        }
        let (request, blobs) = signed_request(&headers, "POST", "/v1/admin/enqueue", &large);
        assert_eq!(request["body"]["blob"], blobs[0].sha256.as_str());
        let found = |sha: &str| (sha == blobs[0].sha256).then(|| blobs[0].body.clone());
        assert!(request_signed(&request, &found));
        assert!(!request_signed(&request, &none), "a missing blob proves nothing");
        let swapped = |_: &str| Some(b"other".to_vec());
        assert!(!request_signed(&request, &swapped), "a blob must hash to its name");
        assert_eq!(blobs_named(&[json!({"input": request})]).into_iter().collect::<Vec<_>>(), [blobs[0].sha256.clone()]);
    }

    #[test]
    fn the_request_hash_matches_the_witness_relays() {
        let hash = request_hash("POST", "/v1/tasks/claim", b"{}", "1800000000", &"a".repeat(32), "5Hotkey", "ABCDEF");
        let mut bytes = signing_payload("POST", "/v1/tasks/claim", b"{}", "1800000000", &"a".repeat(32));
        bytes.extend_from_slice(b"\n5Hotkey\nabcdef");
        assert_eq!(hash, sha256(&bytes));
    }

    /// The same vectors are in `tests/validators/test_journal.py`, so a validator's Python hashes what the API's Rust hashed.
    #[test]
    fn python_hashes_the_same_entries_and_requests() {
        let entry: Value = serde_json::from_str(
            r#"{"v":1,"seq":7,"prev":"0000000000000000000000000000000000000000000000000000000000000000","at":1800000000.137,"kind":"test","input":{"text":"naïve ✓","n":3,"x":0.1,"big":1e17,"whole":2.0},"output":null}"#,
        )
        .unwrap();
        assert_eq!(hash_of(&entry), "088e36e726253dfbbde6652955384d607dafbbbe205aed45c06cb205d8a31fe8");
        let hash = request_hash("POST", "/v1/tasks/claim", br#"{"count":2}"#, "1800000000", &"a".repeat(32), "5Hotkey", "ABCDEF");
        assert_eq!(hash, "ecf7fc2f64d46766666b99367b648b73b20644f096744db8db5c190f3d354712");
    }

    #[test]
    fn a_blob_named_again_after_it_was_published_goes_up_again() {
        let conn = store();
        let blob = Blob::of(b"urls".to_vec());
        append(&conn, Record::new("test", 1.0, blob.reference(), Value::Null).with_blobs(vec![blob.clone()])).unwrap();
        conn.execute("UPDATE journal_blobs SET published = 1, body = x''", []).unwrap();
        append(&conn, Record::new("test", 2.0, blob.reference(), Value::Null).with_blobs(vec![blob.clone()])).unwrap();
        let (seq, published, body): (i64, i64, Vec<u8>) =
            conn.query_row("SELECT seq, published, body FROM journal_blobs", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
        assert_eq!((seq, published, body), (2, 0, b"urls".to_vec()), "the bucket may have let the first copy go");
        append(&conn, Record::new("test", 3.0, blob.reference(), Value::Null).with_blobs(vec![blob])).unwrap();
        let seq: i64 = conn.query_row("SELECT seq FROM journal_blobs", [], |row| row.get(0)).unwrap();
        assert_eq!(seq, 2, "while it waits to be published, it goes up with the first entry naming it");
    }

    #[test]
    fn pruning_keeps_unpublished_entries_and_the_head() {
        let (conn, _) = logged(5);
        conn.execute("INSERT INTO journal_segments VALUES (1, 1, 3, 'h', 0)", []).unwrap();
        assert_eq!(prune(&conn, 1_900_000_000.0, 60.0).unwrap(), 3);
        let left: Vec<i64> = entries_after(&conn, 0, 10).unwrap().iter().map(|(e, _)| e["seq"].as_i64().unwrap()).collect();
        assert_eq!(left, [4, 5]);
        conn.execute("INSERT INTO journal_segments VALUES (2, 4, 5, 'h', 0)", []).unwrap();
        prune(&conn, 1_900_000_000.0, 60.0).unwrap();
        assert_eq!(head(&conn).unwrap().0, 5, "the head stays so the next entry links to it");
        append(&conn, Record::new("test", 2.0, Value::Null, Value::Null)).unwrap();
        assert_eq!(head(&conn).unwrap().0, 6);
    }
}

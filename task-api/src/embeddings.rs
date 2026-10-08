//! Which version of each page has vectors, and from which model.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

pub const QUEUED: &str = "queued";
pub const DONE: &str = "done";
pub const DROPPED: &str = "dropped";

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS embeddings (
            page_key     TEXT NOT NULL,
            model        TEXT NOT NULL,
            content_sha1 TEXT NOT NULL,
            url          TEXT NOT NULL,
            state        TEXT NOT NULL,
            batch_id     TEXT NOT NULL,
            vectors_key  TEXT,
            updated_at   REAL NOT NULL,
            PRIMARY KEY (page_key, model)
        );
        CREATE INDEX IF NOT EXISTS embeddings_model_state ON embeddings (model, state);
        ",
    )?;
    Ok(())
}

fn text<'a>(page: &'a Value, name: &str) -> &'a str {
    page[name].as_str().unwrap_or_default()
}

/// The pages whose current text this model has not embedded or queued yet.
pub fn missing(conn: &Connection, pages: &[Value], model: &str) -> Result<Vec<Value>> {
    let mut statement = conn.prepare("SELECT content_sha1, state FROM embeddings WHERE page_key = ? AND model = ?")?;
    let mut wanted = Vec::new();
    for page in pages {
        let found: Option<(String, String)> = statement.query_row(params![text(page, "page_key"), model], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
        if found.is_none_or(|(sha1, state)| sha1 != text(page, "content_sha1") || state == DROPPED) {
            wanted.push(page.clone());
        }
    }
    Ok(wanted)
}

pub fn queue(conn: &Connection, pages: &[Value], model: &str, batch_id: &str, now: f64) -> Result<()> {
    let mut statement = conn.prepare(
        "INSERT INTO embeddings (page_key, model, content_sha1, url, state, batch_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (page_key, model) DO UPDATE SET content_sha1 = excluded.content_sha1, url = excluded.url,
         state = excluded.state, batch_id = excluded.batch_id, vectors_key = NULL, updated_at = excluded.updated_at",
    )?;
    for page in pages {
        statement.execute(params![text(page, "page_key"), model, text(page, "content_sha1"), text(page, "url"), QUEUED, batch_id, now])?;
    }
    Ok(())
}

/// Only rows still queued by this batch change; a newer version keeps its own state.
pub fn finalize(conn: &Connection, batch_id: &str, model: &str, state: &str, vectors_key: Option<&str>, now: f64) -> Result<usize> {
    Ok(conn.execute(
        "UPDATE embeddings SET state = ?, vectors_key = ?, updated_at = ? WHERE batch_id = ? AND model = ? AND state = ?",
        params![state, vectors_key, now, batch_id, model, QUEUED],
    )?)
}

//! Every completed crawl upload, written in numbered signed files to the uploads bucket, so validators count each miner's work themselves.

use anyhow::Result;
use desearch::r2::Bucket;
use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};

use crate::manifest::{log_payload, upload_log_key, UPLOAD_LOG_LATEST};
use crate::state::{Redis, State};
use crate::storage::put_json;

pub const PENDING: &str = "uploadlog:pending";
pub const NEXT: &str = "uploadlog:next";
/// How many pending entries the file being written holds, so a retry writes the same ones.
pub const WRITING: &str = "uploadlog:writing";
pub const BATCH: isize = 1000;

pub fn entry(job: &Value) -> Value {
    let reported = &job["reported"];
    let number = |name: &str| reported.get(name).and_then(Value::as_f64).map_or(0, |n| n as i64);
    json!({
        "task_id": job["task_id"],
        "miner": job["miner"],
        "key": job["key"],
        "completed_at": job["completed_at"],
        "assigned": crate::lifecycle::assigned(job),
        "rows": number("rows"),
        "ok": number("ok"),
        "errors": number("errors"),
    })
}

pub async fn note(redis: &Redis, job: &Value) -> Result<()> {
    let _: i64 = redis.clone().rpush(PENDING, entry(job).to_string()).await?;
    Ok(())
}

/// Writes the pending entries as the next numbered file; on a failed write they wait for the next pass.
pub async fn flush(state: &State, storage: &Bucket) -> Result<Option<i64>> {
    let mut redis = state.redis.clone();
    let writing: Option<isize> = redis.get(WRITING).await?;
    let writing = writing.unwrap_or(0);
    let pending: Vec<String> = redis.lrange(PENDING, 0, if writing > 0 { writing } else { BATCH } - 1).await?;
    if pending.is_empty() {
        return Ok(None);
    }
    if writing == 0 {
        let _: () = redis.set(WRITING, pending.len()).await?;
    }
    let seq: Option<i64> = redis.get(NEXT).await?;
    let seq = seq.unwrap_or(1);
    let entries: Vec<Value> = pending.iter().map(|raw| serde_json::from_str(raw).unwrap_or(Value::Null)).collect();
    let mut body = Map::new();
    body.insert("seq".into(), seq.into());
    body.insert("written_at".into(), now().into());
    body.insert("entries".into(), entries.into());
    body.insert("signer".into(), state.signer().into());
    let signature = state.sign(&log_payload(&body));
    body.insert("signature".into(), signature.into());
    put_json(storage, &upload_log_key(seq as u64), &Value::Object(body), None).await?;
    put_json(storage, UPLOAD_LOG_LATEST, &json!({"seq": seq}), Some("no-store")).await?;
    let _: () = redis::pipe().atomic().ltrim(PENDING, pending.len() as isize, -1).set(NEXT, seq + 1).del(WRITING).query_async(&mut redis).await?;
    Ok(Some(seq))
}

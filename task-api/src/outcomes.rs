//! What became of every URL, in order, for the bot that queued it; the task API writes the dropped ones.

use anyhow::Result;
use bytes::Bytes;
use desearch::canonical::domain_of;
use desearch::feeds::OUTCOMES;
use desearch::outcomes::{encode, Outcome};
use desearch::r2::{Bucket, PARQUET};
use desearch::time::{now, utc_day};

use crate::state::Redis;

pub fn rows_for(urls: &[String], outcome: &'static str, task_id: &str) -> Vec<Outcome> {
    urls.iter().map(|url| Outcome { url: url.clone(), host: domain_of(url).unwrap_or_default(), outcome, task_id: task_id.into() }).collect()
}

/// Writes one outcome file and numbers it in the feed.
pub async fn write(storage: &Bucket, redis: &Redis, rows: Vec<Outcome>) -> Result<Option<u64>> {
    if rows.is_empty() {
        return Ok(None);
    }
    let at = now();
    let key = format!("outcomes/dt={}/{}.parquet", utc_day(at), uuid::Uuid::new_v4().simple());
    let count = rows.len();
    let file = tokio::task::spawn_blocking(move || encode(&rows, (at * 1e6) as i64)).await??;
    storage.put(&key, Bytes::from(file), PARQUET, None).await?;
    Ok(Some(OUTCOMES.number(storage, redis, &key, count).await?))
}

//! Files numbered in the order they were written, as `app.feeds.Feed` numbers them, so a reader follows them one number at a time.

use std::collections::HashMap;

use anyhow::Result;
use bytes::Bytes;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;

use crate::r2::{Bucket, JSON};

pub struct Feed {
    pub name: &'static str,
}

pub const CHANGES: Feed = Feed { name: "changes" };
pub const OUTCOMES: Feed = Feed { name: "outcomes" };

impl Feed {
    pub fn counter(&self) -> String {
        format!("{}:seq", self.name)
    }

    pub fn holes(&self) -> String {
        format!("{}:holes", self.name)
    }

    pub fn latest_key(&self) -> String {
        format!("{}/latest.json", self.name)
    }

    pub fn seq_key(&self, seq: u64) -> String {
        format!("{}/seq/{seq:012}.json", self.name)
    }

    /// Numbers a file already written; a number is only handed out for a file that exists.
    pub async fn number(&self, storage: &Bucket, redis: &ConnectionManager, key: &str, rows: usize) -> Result<u64> {
        let mut redis = redis.clone();
        let at: u64 = redis.incr(self.counter(), 1).await?;
        if let Err(error) = storage.put(&self.seq_key(at), index_json(key, Some(rows)), JSON, None).await {
            eprintln!("could not index {key} as {at}: {error}");
            let _: i64 = redis.hset(self.holes(), at, key).await?;
            return Ok(at);
        }
        if let Err(error) = storage.put(&self.latest_key(), Bytes::from(format!("{{\"seq\": {at}}}")), JSON, Some("no-store")).await {
            eprintln!("could not note {at} as the newest {} file: {error}", self.name);
        }
        Ok(at)
    }

    /// A number whose index write failed still gets one, so readers never wait on it.
    pub async fn fill_holes(&self, storage: &Bucket, redis: &ConnectionManager) -> Result<usize> {
        let mut redis = redis.clone();
        let holes: HashMap<String, String> = redis.hgetall(self.holes()).await?;
        let mut filled = 0;
        for (at, key) in holes {
            let Ok(seq) = at.parse::<u64>() else { continue };
            if storage.put(&self.seq_key(seq), index_json(&key, None), JSON, None).await.is_err() {
                continue;
            }
            let _: i64 = redis.hdel(self.holes(), &at).await?;
            filled += 1;
        }
        Ok(filled)
    }
}

/// Python's `json.dumps({"key": key, "rows": rows}, sort_keys=True)`.
fn index_json(key: &str, rows: Option<usize>) -> Bytes {
    let key = serde_json::to_string(key).unwrap_or_default();
    let rows = rows.map_or("null".to_string(), |r| r.to_string());
    Bytes::from(format!("{{\"key\": {key}, \"rows\": {rows}}}"))
}

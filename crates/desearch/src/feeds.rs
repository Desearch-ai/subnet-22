//! Files numbered in the order they were written, as `app.feeds.Feed` numbers them, so a reader follows them one number at a time.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::Result;
use bytes::Bytes;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};

use crate::r2::{Bucket, JSON};
use crate::time::now;

/// A number taken but not indexed yet; one older than this was lost with its writer and gets indexed by `fill_holes`.
pub const PENDING_GRACE_S: f64 = 60.0;

static TAKE: LazyLock<Script> =
    LazyLock::new(|| Script::new("local at = redis.call('INCR', KEYS[1]) redis.call('HSET', KEYS[2], at, ARGV[1]) return at"));

pub struct Feed {
    pub name: Cow<'static, str>,
}

pub const CHANGES: Feed = Feed { name: Cow::Borrowed("changes") };
pub const OUTCOMES: Feed = Feed { name: Cow::Borrowed("outcomes") };

impl Feed {
    /// The accepted vectors of one embedding model, under `vectors/<model>/`.
    pub fn vectors(model: &str) -> Feed {
        Feed { name: Cow::Owned(format!("vectors/{model}")) }
    }

    pub fn counter(&self) -> String {
        format!("{}:seq", self.name)
    }

    pub fn holes(&self) -> String {
        format!("{}:holes", self.name)
    }

    /// Numbers taken and not indexed yet, as `<unix time>\t<key>`.
    pub fn pending(&self) -> String {
        format!("{}:pending", self.name)
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
        // Taken and noted pending in one step, so a writer that times out or is cancelled here still leaves its number repairable.
        let at: u64 = TAKE.key(self.counter()).key(self.pending()).arg(format!("{}\t{key}", now())).invoke_async(&mut redis).await?;
        if let Err(error) = storage.put(&self.seq_key(at), index_json(key, Some(rows)), JSON, None).await {
            eprintln!("could not index {key} as {at}: {error}");
            let () = redis::pipe().atomic().hset(self.holes(), at, key).ignore().hdel(self.pending(), at).ignore().query_async(&mut redis).await?;
            return Ok(at);
        }
        let _: i64 = redis.hdel(self.pending(), at).await?;
        if let Err(error) = storage.put(&self.latest_key(), Bytes::from(format!("{{\"seq\": {at}}}")), JSON, Some("no-store")).await {
            eprintln!("could not note {at} as the newest {} file: {error}", self.name);
        }
        Ok(at)
    }

    /// A number whose index write failed, or whose writer died before writing it, still gets one, so readers never wait on it.
    pub async fn fill_holes(&self, storage: &Bucket, redis: &ConnectionManager) -> Result<usize> {
        let mut redis = redis.clone();
        let holes: HashMap<String, String> = redis.hgetall(self.holes()).await?;
        let pending: HashMap<String, String> = redis.hgetall(self.pending()).await?;
        let stale = pending.into_iter().filter_map(|(at, noted)| {
            let (since, key) = noted.split_once('\t')?;
            (now() - since.parse::<f64>().ok()? >= PENDING_GRACE_S).then(|| (at, key.to_string(), self.pending()))
        });
        let mut filled = 0;
        for (at, key, from) in holes.into_iter().map(|(at, key)| (at, key, self.holes())).chain(stale) {
            let Ok(seq) = at.parse::<u64>() else { continue };
            if storage.put(&self.seq_key(seq), index_json(&key, None), JSON, None).await.is_err() {
                continue;
            }
            let _: i64 = redis.hdel(from, &at).await?;
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

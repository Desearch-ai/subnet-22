//! The task API's publish queue in Redis, driven with the same keys and the same Lua as `app.queues.PublishQueue`.

use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use base64::Engine;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::planner::EMBED_DROPPED;
use crate::worker::Job;

pub const PUBLISH: &str = "publish:ready";
pub const PCLAIMS: &str = "publish:claims";
pub const PACKED: &str = "z:";
pub const PUBLISHED: &str = "publish:acked";
pub const PPENDING: &str = "publish:pending";
pub const PDEAD: &str = "publish:dead";
pub const LOST: &str = "publish:lost";
pub const LOST_TASKS: &str = "publish:lost:tasks";
/// Tasks taken back, scored by when, so a publish job still queued for one is refused.
pub const WITHDRAWN: &str = "publish:withdrawn";
pub const WITHDRAWN_KEEP_S: f64 = 7.0 * 86_400.0;
/// Followed by a task id: that embed task's published vectors, kept as long as a take-back is remembered.
pub const EMBED_PUBLISHED: &str = "embed:published:";
/// Embed tasks with malformed vectors, for the task API to check their miner's next uploads.
pub const EMBED_MALFORMED: &str = "embed:malformed";
/// Byte for byte the text in `app/queues.py`, so both load the script under one SHA.
pub const PCLAIM: &str = "
local jobs = {}
while #jobs < tonumber(ARGV[1]) do
  local task_id = redis.call('LPOP', KEYS[1])
  if not task_id then break end
  local job = redis.call('GET', 'pjob:' .. task_id)
  if job then
    redis.call('ZADD', KEYS[2], ARGV[2], task_id)
    table.insert(jobs, job)
  end
end
return jobs
";
pub const PACK: &str = "
redis.call('ZREM', KEYS[1], ARGV[1])
if redis.call('ZREM', KEYS[2], ARGV[1]) == 1 then redis.call('INCR', KEYS[3]) end
return redis.call('DEL', 'pjob:' .. ARGV[1], 'ptries:' .. ARGV[1])
";
/// Back on the queue at once and without a try counted, since a failed pass is the publisher's fault, not the job's.
const GIVE_BACK: &str = "
local back = 0
for _, task_id in ipairs(ARGV) do
  if redis.call('ZREM', KEYS[1], task_id) == 1 and redis.call('EXISTS', 'pjob:' .. task_id) == 1 then
    redis.call('RPUSH', KEYS[2], task_id)
    back = back + 1
  end
end
return back
";

pub struct PublishQueue {
    pub redis: ConnectionManager,
    claim_ttl: f64,
    claim: Script,
    ack: Script,
    give_back: Script,
}

/// Where an embed task's vectors were published, so they can be withdrawn.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedVectors {
    pub model: String,
    pub key: String,
    pub miner: String,
    pub input_key: String,
}

/// How far behind publishing is, as the queue shows it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Backlog {
    pub ready: u64,
    pub waiting: u64,
    pub set_aside: u64,
    pub lost: u64,
}

impl PublishQueue {
    pub fn new(redis: ConnectionManager, claim_ttl: f64) -> Self {
        PublishQueue { redis, claim_ttl, claim: Script::new(PCLAIM), ack: Script::new(PACK), give_back: Script::new(GIVE_BACK) }
    }

    /// Up to `count` jobs, each claimed until the claim expires; a job that does not parse stays claimed until the API gives it back.
    pub async fn claim(&self, count: usize) -> Result<Vec<Job>> {
        let raw: Vec<String> = self.claim.key(PUBLISH).key(PCLAIMS).arg(count).arg(now() + self.claim_ttl).invoke_async(&mut self.redis.clone()).await?;
        let mut jobs = Vec::with_capacity(raw.len());
        for job in raw {
            match unpack_job(&job) {
                Ok(job) => jobs.push(job),
                Err(error) => eprintln!("a publish job that does not parse, left to expire: {error:#}"),
            }
        }
        Ok(jobs)
    }

    pub async fn ack(&self, task_id: &str) -> Result<()> {
        let _: i64 = self.ack.key(PCLAIMS).key(PPENDING).key(PUBLISHED).arg(task_id).invoke_async(&mut self.redis.clone()).await?;
        Ok(())
    }

    pub async fn extend_claims(&self, task_ids: &[&str]) -> Result<()> {
        if task_ids.is_empty() {
            return Ok(());
        }
        let until = now() + self.claim_ttl;
        let mut pipe = redis::pipe();
        for task_id in task_ids {
            pipe.cmd("ZADD").arg(PCLAIMS).arg("XX").arg(until).arg(*task_id).ignore();
        }
        let () = pipe.query_async(&mut self.redis.clone()).await?;
        Ok(())
    }

    /// Puts still-claimed jobs back on the queue for the next pass; how many went back.
    pub async fn give_back(&self, task_ids: &[String]) -> Result<usize> {
        if task_ids.is_empty() {
            return Ok(0);
        }
        Ok(self.give_back.key(PCLAIMS).key(PUBLISH).arg(task_ids).invoke_async(&mut self.redis.clone()).await?)
    }

    pub async fn mark_lost(&self, task_id: &str) -> Result<()> {
        let mut redis = self.redis.clone();
        let _: i64 = redis.incr(LOST, 1).await?;
        let _: i64 = redis.sadd(LOST_TASKS, task_id).await?;
        Ok(())
    }

    /// Notes tasks as taken back, for every publisher.
    pub async fn note_withdrawn(&self, task_ids: &[String], at: f64) -> Result<()> {
        if task_ids.is_empty() {
            return Ok(());
        }
        let members: Vec<(f64, &str)> = task_ids.iter().map(|task_id| (at, task_id.as_str())).collect();
        let _: i64 = self.redis.clone().zadd_multiple(WITHDRAWN, &members).await?;
        Ok(())
    }

    /// Which of these tasks were taken back.
    pub async fn withdrawn_among(&self, task_ids: &[&str]) -> Result<std::collections::HashSet<String>> {
        if task_ids.is_empty() {
            return Ok(Default::default());
        }
        let scores: Vec<Option<f64>> = redis::cmd("ZMSCORE").arg(WITHDRAWN).arg(task_ids).query_async(&mut self.redis.clone()).await?;
        Ok(task_ids.iter().zip(scores).filter(|(_, score)| score.is_some()).map(|(task_id, _)| task_id.to_string()).collect())
    }

    /// Forgets tasks taken back more than `WITHDRAWN_KEEP_S` ago, when no job of theirs can still be queued.
    pub async fn forget_withdrawn(&self, now: f64) -> Result<()> {
        let _: i64 = self.redis.clone().zrembyscore(WITHDRAWN, "-inf", now - WITHDRAWN_KEEP_S).await?;
        Ok(())
    }

    pub async fn remember_vectors(&self, task_id: &str, published: &PublishedVectors) -> Result<()> {
        let () = self.redis.clone().set_ex(format!("{EMBED_PUBLISHED}{task_id}"), serde_json::to_string(published)?, WITHDRAWN_KEEP_S as u64).await?;
        Ok(())
    }

    /// The published vectors remembered for these tasks.
    pub async fn published_vectors(&self, task_ids: &[String]) -> Result<Vec<(String, PublishedVectors)>> {
        if task_ids.is_empty() {
            return Ok(Vec::new());
        }
        let keys: Vec<String> = task_ids.iter().map(|task_id| format!("{EMBED_PUBLISHED}{task_id}")).collect();
        let found: Vec<Option<String>> = redis::cmd("MGET").arg(&keys).query_async(&mut self.redis.clone()).await?;
        let mut published = Vec::new();
        for (task_id, raw) in task_ids.iter().zip(found) {
            if let Some(raw) = raw {
                published.push((task_id.clone(), serde_json::from_str(&raw).with_context(|| format!("the published vectors of {task_id}"))?));
            }
        }
        Ok(published)
    }

    pub async fn forget_vectors(&self, task_id: &str) -> Result<()> {
        let _: i64 = self.redis.clone().del(format!("{EMBED_PUBLISHED}{task_id}")).await?;
        Ok(())
    }

    /// Sends pages back to the planner, as the task API sends a dropped task's.
    pub async fn drop_pages<P: Serialize>(&self, input_key: &str, pages: &[P]) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let _: i64 = self.redis.clone().rpush(EMBED_DROPPED, json!({"input_key": input_key, "pages": pages}).to_string()).await?;
        Ok(())
    }

    pub async fn note_malformed(&self, task_id: &str, miner: &str, malformed: usize) -> Result<()> {
        let _: i64 = self.redis.clone().rpush(EMBED_MALFORMED, json!({"task_id": task_id, "miner": miner, "malformed": malformed}).to_string()).await?;
        Ok(())
    }

    pub async fn backlog(&self) -> Result<Backlog> {
        let (ready, waiting, set_aside, lost): (u64, u64, u64, Option<u64>) =
            redis::pipe().llen(PUBLISH).zcard(PPENDING).scard(PDEAD).get(LOST).query_async(&mut self.redis.clone()).await?;
        Ok(Backlog { ready, waiting, set_aside, lost: lost.unwrap_or(0) })
    }
}

/// A job as `pack_job` stored it: `z:` and base64 of zlib-compressed JSON, or plain JSON.
pub fn unpack_job(raw: &str) -> Result<Job> {
    let Some(packed) = raw.strip_prefix(PACKED) else {
        return serde_json::from_str(raw).context("a publish job that is not JSON");
    };
    let compressed = base64::engine::general_purpose::STANDARD.decode(packed).context("a packed publish job that is not base64")?;
    let mut json = Vec::new();
    flate2::read::ZlibDecoder::new(compressed.as_slice()).read_to_end(&mut json).context("a packed publish job that is not zlib")?;
    serde_json::from_slice(&json).context("a packed publish job that is not JSON")
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

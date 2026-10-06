//! The task API's publish queue in Redis, driven with the same keys and the same Lua as `app.queues.PublishQueue`.

use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use base64::Engine;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};

use crate::worker::Job;

pub const PUBLISH: &str = "publish:ready";
pub const PCLAIMS: &str = "publish:claims";
pub const PACKED: &str = "z:";
pub const PUBLISHED: &str = "publish:acked";
pub const PPENDING: &str = "publish:pending";
pub const PDEAD: &str = "publish:dead";
pub const LOST: &str = "publish:lost";
pub const LOST_TASKS: &str = "publish:lost:tasks";
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

pub struct PublishQueue {
    pub redis: ConnectionManager,
    claim_ttl: f64,
    claim: Script,
    ack: Script,
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
        PublishQueue { redis, claim_ttl, claim: Script::new(PCLAIM), ack: Script::new(PACK) }
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

    pub async fn mark_lost(&self, task_id: &str) -> Result<()> {
        let mut redis = self.redis.clone();
        let _: i64 = redis.incr(LOST, 1).await?;
        let _: i64 = redis.sadd(LOST_TASKS, task_id).await?;
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

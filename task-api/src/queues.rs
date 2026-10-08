//! The task, validation and publish queues in Redis, each change one Lua script so it lands whole.

use std::io::Write;
use std::sync::LazyLock;

use anyhow::{anyhow, Result};
use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use redis::{AsyncCommands, Script, Value as Reply};
use serde_json::{json, Map, Value};

use crate::py::round_to;
use crate::state::Redis;

pub const QUEUE: &str = "queue:ready";
pub const CLAIMS: &str = "claims:expiry";
/// Uploads open for every validator to score, by completion time.
pub const VOPEN: &str = "vjobs:open";
/// Uploads waiting for their seed block, which decides whether validators check them.
pub const SEEDING: &str = "vjobs:seeding";
/// Validators by the time they last asked for work or voted.
pub const VACTIVE: &str = "validators:active";
pub const PUBLISH: &str = "publish:ready";
pub const PCLAIMS: &str = "publish:claims";
pub const PACKED: &str = "z:";
/// Every job the publisher finished, for measuring its pace.
pub const PUBLISHED: &str = "publish:acked";
pub const PPENDING: &str = "publish:pending";
pub const PDEAD: &str = "publish:dead";
pub const ROUNDS: &str = "rounds:seq";
/// Published pages waiting to become embed batches, pushed by the publisher.
pub const EMBED_INPUTS: &str = "embed:inputs";
pub const ROUND_SPAN: i64 = 10_000_000;
pub const CLAIM_SCAN: i64 = 50;
pub const OPEN_SCAN: isize = 500;
pub const ACTIVE_S: f64 = 3600.0;
/// A miner that held the whole front of the queue is still served from further back.
pub const CLAIM_SCAN_MAX: i64 = 1000;
pub const HOLDERS_TTL_S: i64 = 7 * 86_400;
pub const KINDS: [&str; 2] = ["crawl", "embed"];

static CLAIM: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/claim.lua")));
static RECLAIM: LazyLock<Script> = LazyLock::new(|| {
    Script::new(concat!(include_str!("lua/inflight.lua"), include_str!("lua/reclaim.lua"), include_str!("lua/requeue.lua"), "return {holder, seq}"))
});
static ABANDON: LazyLock<Script> =
    LazyLock::new(|| Script::new(concat!(include_str!("lua/inflight.lua"), include_str!("lua/abandon.lua"), include_str!("lua/requeue.lua"), "return seq")));
static COMPLETE: LazyLock<Script> = LazyLock::new(|| Script::new(concat!(include_str!("lua/inflight.lua"), include_str!("lua/complete.lua"))));
static START: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/start.lua")));
static RESTORE: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/restore.lua")));
static VOPEN_PICKED: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/vopen_picked.lua")));
static VVOTE: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/vvote.lua")));
static VSETTLE: LazyLock<Script> = LazyLock::new(|| Script::new(concat!(include_str!("lua/inflight.lua"), include_str!("lua/vsettle.lua"))));
static PRETURN: LazyLock<Script> = LazyLock::new(|| Script::new(include_str!("lua/preturn.lua")));

pub fn ready_key(kind: &str) -> String {
    if kind == "crawl" {
        QUEUE.into()
    } else {
        format!("{QUEUE}:{kind}")
    }
}

/// Tasks a miner holds and has not uploaded yet.
pub fn inflight_key(kind: &str, hotkey: &str) -> String {
    if kind == "crawl" {
        format!("inflight:{hotkey}")
    } else {
        format!("inflight:{kind}:{hotkey}")
    }
}

/// A miner's uploads that no verdict has closed yet.
pub fn waiting_key(kind: &str, hotkey: &str) -> String {
    if kind == "crawl" {
        format!("waiting:{hotkey}")
    } else {
        format!("waiting:{kind}:{hotkey}")
    }
}

/// A publish job waits in Redis until published; its URL list compresses several times over.
pub fn pack_job(job: &Value) -> String {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(job.to_string().as_bytes()).expect("writing to memory");
    let compressed = encoder.finish().expect("writing to memory");
    format!("{PACKED}{}", base64::engine::general_purpose::STANDARD.encode(compressed))
}

pub fn text(reply: &Reply) -> Option<String> {
    match reply {
        Reply::BulkString(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
        Reply::SimpleString(text) => Some(text.clone()),
        Reply::VerbatimString { text, .. } => Some(text.clone()),
        Reply::Int(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn int(reply: &Reply) -> Option<i64> {
    match reply {
        Reply::Int(n) => Some(*n),
        other => text(other)?.parse().ok(),
    }
}

fn parsed(raw: Option<String>) -> Option<Value> {
    raw.and_then(|raw| serde_json::from_str(&raw).ok())
}

/// A JSON number Lua and Redis read the same way Python's `str(float)` writes it.
fn number(x: f64) -> String {
    format!("{x}")
}

#[derive(Clone, Debug)]
pub struct Claim {
    pub task_id: String,
    pub payload: Value,
    pub expires_at: f64,
    pub seq: i64,
}

/// Why a claim found no task for this miner.
#[derive(Clone, Debug)]
pub struct Refusal {
    pub code: &'static str,
    pub inputs: Map<String, Value>,
}

impl Refusal {
    pub fn new(code: &'static str, inputs: Value) -> Self {
        Refusal { code, inputs: inputs.as_object().cloned().unwrap_or_default() }
    }

    pub fn as_value(&self) -> Value {
        json!({"code": self.code, "inputs": self.inputs})
    }
}

pub struct TaskQueue {
    pub kind: &'static str,
    pub claim_ttl: f64,
    pub ready: String,
}

impl TaskQueue {
    pub fn new(kind: &'static str, claim_ttl: f64) -> Self {
        TaskQueue { kind, claim_ttl, ready: ready_key(kind) }
    }

    pub async fn fill(&self, redis: &Redis, round_id: &str, order: &[String], payloads: &Map<String, Value>) -> Result<usize> {
        let mut redis = redis.clone();
        let base: i64 = redis.incr(ROUNDS, 1).await?;
        let base = base * ROUND_SPAN;
        let mut pipe = redis::pipe();
        pipe.atomic();
        for (position, task_id) in order.iter().enumerate() {
            let rank = base + position as i64;
            let mut payload = payloads.get(task_id).and_then(Value::as_object).cloned().unwrap_or_default();
            payload.insert("kind".into(), self.kind.into());
            payload.insert("round_id".into(), round_id.into());
            payload.insert("position".into(), position.into());
            payload.insert("rank".into(), rank.into());
            pipe.set(format!("task:{task_id}"), Value::Object(payload).to_string()).ignore();
            pipe.zadd(&self.ready, task_id, rank).ignore();
        }
        let _: () = pipe.query_async(&mut redis).await?;
        Ok(order.len())
    }

    pub async fn depth(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().zcard(&self.ready).await?)
    }

    pub async fn in_flight(&self, redis: &Redis, hotkey: &str) -> Result<i64> {
        Ok(redis.clone().scard(inflight_key(self.kind, hotkey)).await?)
    }

    pub async fn waiting(&self, redis: &Redis, hotkey: &str) -> Result<i64> {
        Ok(redis.clone().scard(waiting_key(self.kind, hotkey)).await?)
    }

    /// Up to `count` tasks, as far as the miner's crawling and waiting limits allow.
    pub async fn claim(&self, redis: &Redis, hotkey: &str, budget: i64, waiting_limit: i64, count: i64, now: f64) -> Result<Result<Vec<Claim>, Refusal>> {
        let mut redis = redis.clone();
        let found: Vec<Reply> = CLAIM
            .key(&self.ready)
            .key(CLAIMS)
            .key(inflight_key(self.kind, hotkey))
            .key(waiting_key(self.kind, hotkey))
            .arg(hotkey)
            .arg(number(self.claim_ttl))
            .arg(number(now))
            .arg(budget)
            .arg(CLAIM_SCAN)
            .arg(HOLDERS_TTL_S)
            .arg(CLAIM_SCAN_MAX)
            .arg(waiting_limit)
            .arg(count)
            .invoke_async(&mut redis)
            .await?;
        let status = found.first().and_then(text).unwrap_or_default();
        let first = || found.get(1).and_then(int).unwrap_or(0);
        Ok(match status.as_str() {
            "full" => Err(Refusal::new("NO_CAPACITY", json!({"budget": budget, "in_flight": first()}))),
            "waiting" => Err(Refusal::new("WAITING_FOR_VERDICTS", json!({"waiting": first(), "limit": waiting_limit}))),
            "held" => Err(Refusal::new("ALREADY_HELD", json!({"depth": self.depth(&redis).await?, "held": first()}))),
            "empty" => Err(Refusal::new("QUEUE_EMPTY", json!({"depth": self.depth(&redis).await?}))),
            _ => Ok(found[1..]
                .chunks(3)
                .map(|claim| Claim {
                    task_id: text(&claim[0]).unwrap_or_default(),
                    payload: parsed(text(&claim[1])).unwrap_or(Value::Null),
                    expires_at: now + self.claim_ttl,
                    seq: int(&claim[2]).unwrap_or(0),
                })
                .collect()),
        })
    }

    /// The claims this hotkey still holds restart their clock now; returns the new expiry.
    pub async fn start_clocks(&self, redis: &Redis, hotkey: &str, task_ids: &[String], now: f64) -> Result<f64> {
        let expiry = now + self.claim_ttl;
        let mut invocation = START.key(CLAIMS);
        invocation.arg(hotkey).arg(number(expiry));
        for task_id in task_ids {
            invocation.arg(task_id);
        }
        let _: i64 = invocation.invoke_async(&mut redis.clone()).await?;
        Ok(expiry)
    }

    pub async fn complete(&self, redis: &Redis, task_id: &str, hotkey: &str, job: &Value, upload_key: &str, arrived: f64) -> Result<Option<i64>> {
        let seq: Option<i64> = COMPLETE
            .key(CLAIMS)
            .key(SEEDING)
            .arg(task_id)
            .arg(hotkey)
            .arg(job.to_string())
            .arg(number(arrived))
            .arg(upload_key)
            .arg(job.get("seed_block").and_then(Value::as_i64).unwrap_or(0))
            .invoke_async(&mut redis.clone())
            .await?;
        Ok(seq)
    }

    pub async fn abandon(&self, redis: &Redis, task_id: &str, hotkey: &str) -> Result<Option<i64>> {
        Ok(ABANDON.key(&self.ready).key(CLAIMS).arg(task_id).arg(hotkey).invoke_async(&mut redis.clone()).await?)
    }

    /// The holder and log number of a claim taken back after its expiry, None if it was not due.
    pub async fn reclaim(&self, redis: &Redis, task_id: &str, now: f64) -> Result<Option<(String, i64)>> {
        let found: Option<Vec<Reply>> = RECLAIM.key(&self.ready).key(CLAIMS).arg(task_id).arg(number(now)).invoke_async(&mut redis.clone()).await?;
        Ok(found.filter(|f| f.len() == 2).map(|f| (text(&f[0]).unwrap_or_default(), int(&f[1]).unwrap_or(0))))
    }

    pub async fn restore(&self, redis: &Redis, task_id: &str, payload: &Value) -> Result<i64> {
        let rank = payload.get("rank").or_else(|| payload.get("position")).and_then(Value::as_f64).unwrap_or(0.0);
        Ok(RESTORE.key(&self.ready).arg(task_id).arg(payload.to_string()).arg(number(rank)).invoke_async(&mut redis.clone()).await?)
    }
}

pub async fn claim_expiry(redis: &Redis, task_id: &str) -> Result<Option<f64>> {
    Ok(redis.clone().zscore(CLAIMS, task_id).await?)
}

pub async fn expired_claims(redis: &Redis, now: f64) -> Result<Vec<String>> {
    Ok(redis.clone().zrangebyscore(CLAIMS, "-inf", number(now)).await?)
}

/// An upload closed by its verdict: the job and every vote it had.
pub struct Finalized {
    pub job: Value,
    pub votes: Vec<Value>,
}

/// Every validator scores every open upload; an upload closes when it is finalized.
pub struct ValidationQueue {
    pub active_s: f64,
}

impl ValidationQueue {
    pub async fn depth(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().zcard(VOPEN).await?)
    }

    pub async fn seeding(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().zcard(SEEDING).await?)
    }

    /// How long the upload with the earliest seed block has waited since it was completed.
    pub async fn oldest_seeding_age(&self, redis: &Redis, now: f64) -> Result<f64> {
        let oldest: Vec<String> = redis.clone().zrange(SEEDING, 0, 0).await?;
        let Some(task_id) = oldest.first() else { return Ok(0.0) };
        Ok(match self.job(redis, task_id).await? {
            Some(job) => round_to(now - job["completed_at"].as_f64().unwrap_or(now), 1),
            None => 0.0,
        })
    }

    /// Uploads whose seed block exists, oldest seed first.
    pub async fn seeded(&self, redis: &Redis, block: i64) -> Result<Vec<String>> {
        Ok(redis.clone().zrangebyscore_limit(SEEDING, "-inf", block, 0, OPEN_SCAN).await?)
    }

    /// A picked upload goes on the open list for validators.
    pub async fn open(&self, redis: &Redis, task_id: &str, job: &Value, now: f64) -> Result<bool> {
        let completed = job.get("completed_at").and_then(Value::as_f64).filter(|c| *c != 0.0).unwrap_or(now);
        let opened: i64 =
            VOPEN_PICKED.key(SEEDING).key(VOPEN).arg(task_id).arg(job.to_string()).arg(number(completed)).invoke_async(&mut redis.clone()).await?;
        Ok(opened == 1)
    }

    pub async fn job(&self, redis: &Redis, task_id: &str) -> Result<Option<Value>> {
        let found: Option<String> = redis.clone().get(format!("vjob:{task_id}")).await?;
        Ok(parsed(found))
    }

    pub async fn votes(&self, redis: &Redis, task_id: &str) -> Result<Vec<Value>> {
        let found: Vec<String> = redis.clone().lrange(format!("votes:{task_id}"), 0, -1).await?;
        Ok(found.into_iter().filter_map(|v| serde_json::from_str(&v).ok()).collect())
    }

    pub async fn voters(&self, redis: &Redis, task_id: &str) -> Result<Vec<String>> {
        let mut found: Vec<String> = redis.clone().smembers(format!("vseen:{task_id}")).await?;
        found.sort();
        Ok(found)
    }

    pub async fn oldest_age(&self, redis: &Redis, now: f64) -> Result<f64> {
        let oldest: Vec<(String, f64)> = redis.clone().zrange_withscores(VOPEN, 0, 0).await?;
        Ok(oldest.first().map_or(0.0, |(_, at)| round_to(now - at, 1)))
    }

    pub async fn open_ids(&self, redis: &Redis, limit: isize) -> Result<Vec<String>> {
        Ok(redis.clone().zrange(VOPEN, 0, limit - 1).await?)
    }

    pub async fn seeding_ids(&self, redis: &Redis, limit: isize) -> Result<Vec<String>> {
        Ok(redis.clone().zrange(SEEDING, 0, limit - 1).await?)
    }

    /// How many votes the upload has now, or zero when this one was not taken.
    pub async fn vote(&self, redis: &Redis, task_id: &str, validator: &str, vote: &Value, now: f64) -> Result<i64> {
        Ok(VVOTE.key(VOPEN).key(VACTIVE).arg(task_id).arg(validator).arg(number(now)).arg(vote.to_string()).invoke_async(&mut redis.clone()).await?)
    }

    /// A validator asking for work is in the electorate, however slow its verdicts.
    pub async fn present(&self, redis: &Redis, validator: &str, now: f64) -> Result<()> {
        let _: i64 = redis.clone().zadd(VACTIVE, validator, number(now)).await?;
        Ok(())
    }

    pub async fn leave(&self, redis: &Redis, validator: &str) -> Result<()> {
        let _: i64 = redis.clone().zrem(VACTIVE, validator).await?;
        Ok(())
    }

    /// Validators that asked for work or voted within the activity window.
    pub async fn active(&self, redis: &Redis, now: f64) -> Result<Vec<String>> {
        Ok(self.last_seen(redis, now).await?.into_iter().map(|(validator, _)| validator).collect())
    }

    /// When each active validator last asked for work or voted.
    pub async fn last_seen(&self, redis: &Redis, now: f64) -> Result<Vec<(String, f64)>> {
        let mut redis = redis.clone();
        let _: i64 = redis.zrembyscore(VACTIVE, "-inf", number(now - self.active_s)).await?;
        Ok(redis.zrange_withscores(VACTIVE, 0, -1).await?)
    }

    pub async fn finalize(&self, redis: &Redis, task_id: &str, publish: Option<&Value>, seeding: bool, now: f64) -> Result<Option<Finalized>> {
        let completed = publish.and_then(|p| p.get("completed_at")).and_then(Value::as_f64).filter(|c| *c != 0.0).unwrap_or(now);
        let found: Option<Vec<Reply>> = VSETTLE
            .key(if seeding { SEEDING } else { VOPEN })
            .key(PUBLISH)
            .key(PPENDING)
            .arg(task_id)
            .arg(publish.map(pack_job).unwrap_or_default())
            .arg(number(completed))
            .invoke_async(&mut redis.clone())
            .await?;
        let Some(found) = found.filter(|f| f.len() == 2) else { return Ok(None) };
        let job = parsed(text(&found[0])).ok_or_else(|| anyhow!("task={task_id} has a job that does not parse"))?;
        let votes = match parsed(text(&found[1])) {
            Some(Value::Array(votes)) => votes.iter().filter_map(|v| v.as_str().and_then(|v| serde_json::from_str(v).ok())).collect(),
            _ => Vec::new(),
        };
        Ok(Some(Finalized { job, votes }))
    }
}

pub struct PublishQueue {
    pub claim_ttl: f64,
    pub max_tries: i64,
}

impl PublishQueue {
    /// A job for the publisher to take these tasks' pages back out of the published set.
    pub async fn withdraw(&self, redis: &Redis, task_ids: &[String], reason: &str, now: f64) -> Result<String> {
        let job_id = format!("withdraw:{}", crate::rounds::new_id());
        let job = json!({"task_id": job_id, "kind": "withdraw", "task_ids": task_ids, "reason": reason});
        let mut redis = redis.clone();
        let _: () = redis.set(format!("pjob:{job_id}"), job.to_string()).await?;
        let _: i64 = redis.rpush(PUBLISH, &job_id).await?;
        let _: i64 = redis.zadd(PPENDING, &job_id, number(now)).await?;
        Ok(job_id)
    }

    pub async fn dead_count(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().scard(PDEAD).await?)
    }

    pub async fn lost_count(&self, redis: &Redis) -> Result<i64> {
        let lost: Option<i64> = redis.clone().get("publish:lost").await?;
        Ok(lost.unwrap_or(0))
    }

    pub async fn expired(&self, redis: &Redis, now: f64) -> Result<Vec<String>> {
        Ok(redis.clone().zrangebyscore(PCLAIMS, "-inf", number(now)).await?)
    }

    /// 1 when the job went back on the queue, -1 when it failed too often and was set aside, 0 when it was gone.
    pub async fn give_back(&self, redis: &Redis, task_id: &str) -> Result<i64> {
        Ok(PRETURN.key(PCLAIMS).key(PUBLISH).key(PPENDING).key(PDEAD).arg(task_id).arg(self.max_tries).invoke_async(&mut redis.clone()).await?)
    }

    pub async fn depth(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().llen(PUBLISH).await?)
    }

    pub async fn waiting(&self, redis: &Redis) -> Result<i64> {
        Ok(redis.clone().zcard(PPENDING).await?)
    }

    pub async fn finished(&self, redis: &Redis) -> Result<i64> {
        let acked: Option<i64> = redis.clone().get(PUBLISHED).await?;
        Ok(acked.unwrap_or(0))
    }

    pub async fn oldest_age(&self, redis: &Redis, now: f64) -> Result<f64> {
        let oldest: Vec<(String, f64)> = redis.clone().zrange_withscores(PPENDING, 0, 0).await?;
        Ok(oldest.first().map_or(0.0, |(_, at)| round_to(now - at, 1)))
    }
}

//! Which uploads validators check: a share of each miner's uploads of each kind, drawn from a block hash nobody knew at upload.

use anyhow::Result;
use redis::AsyncCommands;
use sha2::{Digest, Sha256};

use crate::budgets::CRAWL;
use crate::state::Redis;

pub const SHARE: f64 = 0.05;
/// Drawn checks an hour across all miners, so validators keep up whatever the volume.
pub const CHECKS_PER_HOUR: f64 = 400.0;
pub const EMBED_SHARE: f64 = 0.05;
pub const EMBED_CHECKS_PER_HOUR: f64 = 600.0;
pub const ALL: &str = "all";
pub const NEW_HOTKEY_PASSES: i64 = 10;
pub const RECHECK_UPLOADS: i64 = 10;
pub const NEW: &str = "new";
pub const RECHECK: &str = "recheck";
pub const DRAW: &str = "draw";
const HOUR: f64 = 3600.0;

pub fn draw(seed: &str, task_id: &str, etag: &str) -> f64 {
    let digest = Sha256::digest(format!("{seed}:{task_id}:{etag}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("8 bytes")) as f64 / 2f64.powi(64)
}

/// Every upload of a new hotkey and of one re-checked after a fail; otherwise the share, and about one an hour at least.
pub fn pick_reason(drawn: f64, uploads_last_hour: f64, passes: i64, recheck_left: i64, share: f64) -> Option<&'static str> {
    if passes < NEW_HOTKEY_PASSES {
        return Some(NEW);
    }
    if recheck_left > 0 {
        return Some(RECHECK);
    }
    (drawn < share.max(1.0 / uploads_last_hour.max(1.0))).then_some(DRAW)
}

/// The drawn share, lowered so all miners' draws together stay within the hourly budget; a share of 1 or more checks every upload.
pub fn budget_share(share: f64, per_hour: f64, all_last_hour: f64) -> f64 {
    if per_hour <= 0.0 || share >= 1.0 {
        return share;
    }
    share.min(per_hour / all_last_hour.max(1.0))
}

fn hour_of(at: f64) -> i64 {
    (at / HOUR).floor() as i64
}

/// The hour's count of one hotkey's uploads of a kind, or of everyone's as `ALL`.
pub fn counter_key(kind: &str, counted: &str, hour: i64) -> String {
    if kind == CRAWL {
        format!("uploads:{counted}:{hour}")
    } else {
        format!("uploads:{kind}:{counted}:{hour}")
    }
}

pub async fn note_upload(redis: &Redis, kind: &str, hotkey: &str, at: f64) -> Result<()> {
    let mut redis = redis.clone();
    for counted in [hotkey, ALL] {
        let key = counter_key(kind, counted, hour_of(at));
        let count: i64 = redis.incr(&key, 1).await?;
        if count == 1 {
            let _: bool = redis.expire(&key, 3 * HOUR as i64).await?;
        }
    }
    Ok(())
}

/// This hour's uploads of the kind plus the unexpired part of the last hour's.
pub async fn uploads_last_hour(redis: &Redis, kind: &str, hotkey: &str, now: f64) -> Result<f64> {
    let mut redis = redis.clone();
    let hour = hour_of(now);
    let this_hour: Option<f64> = redis.get(counter_key(kind, hotkey, hour)).await?;
    let last_hour: Option<f64> = redis.get(counter_key(kind, hotkey, hour - 1)).await?;
    Ok(this_hour.unwrap_or(0.0) + last_hour.unwrap_or(0.0) * (1.0 - (now % HOUR) / HOUR))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budgets::EMBED;

    #[test]
    fn draws_match_python() {
        assert_eq!(draw("seed", "task", "\"etag\""), 0.7543348702637427);
        assert_eq!(pick_reason(0.5, 10.0, 3, 0, SHARE), Some(NEW));
        assert_eq!(pick_reason(0.5, 10.0, 10, 2, SHARE), Some(RECHECK));
        assert_eq!(pick_reason(0.09, 10.0, 10, 0, SHARE), Some(DRAW));
        assert_eq!(pick_reason(0.11, 10.0, 10, 0, SHARE), None);
        assert_eq!(budget_share(0.05, 400.0, 20_000.0), 0.02);
    }

    #[test]
    fn a_share_of_one_checks_every_upload_whatever_the_budget() {
        assert_eq!(budget_share(1.0, 600.0, 20_000.0), 1.0);
        assert_eq!(pick_reason(0.999_999, 1e9, 10, 0, budget_share(1.0, 600.0, 1e9)), Some(DRAW));
    }

    #[test]
    fn each_kind_counts_its_uploads_apart_and_crawl_keeps_its_keys() {
        assert_eq!(counter_key(CRAWL, "5Miner", 7), "uploads:5Miner:7");
        assert_eq!(counter_key(CRAWL, ALL, 7), "uploads:all:7");
        assert_eq!(counter_key(EMBED, "5Miner", 7), "uploads:embed:5Miner:7");
        assert_eq!(counter_key(EMBED, ALL, 7), "uploads:embed:all:7");
    }
}

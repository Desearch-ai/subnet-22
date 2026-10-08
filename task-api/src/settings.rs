//! Everything the task API is told through its environment, with the same names and defaults the Python service had.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context, Result};

use crate::flow::{LAG_LIMIT_S, LAG_TARGET_S};
use crate::queues::ACTIVE_S;
use crate::rounds::CLAIM_TTL_S;
use crate::sampling::{CHECKS_PER_HOUR, SHARE};

const DEFAULT_ORIGINS: &str = "http://localhost:5173,http://localhost:8081,http://127.0.0.1:5173,http://127.0.0.1:8081";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryMode {
    Chain { netuid: u16, network: String },
    Local { validator_uris: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SeedsMode {
    Chain { network: String },
    Local { block_seconds: f64, genesis: f64 },
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub listen: String,
    pub data: PathBuf,
    pub redis_url: String,
    pub claim_ttl: f64,
    pub validation_ttl: f64,
    pub poll_rate: f64,
    pub reads_per_minute: i64,
    pub log_reads_per_minute: i64,
    pub failed_writes_per_minute: i64,
    pub max_upload: u64,
    pub max_backlog: f64,
    pub max_attempts: i64,
    pub ledger_delay: f64,
    pub check_share: f64,
    pub checks_per_hour: f64,
    pub queue_target: i64,
    pub task_urls: usize,
    pub publish_lag: f64,
    pub publish_lag_limit: f64,
    pub embed_tasks: bool,
    pub embed_model: String,
    pub active_s: f64,
    pub publish_ttl: f64,
    pub publish_tries: i64,
    pub r2_endpoint: String,
    pub r2_access_key: String,
    pub r2_secret_key: String,
    pub bucket: String,
    pub prefix: String,
    pub pages_bucket: String,
    pub pages_prefix: String,
    pub registry: RegistryMode,
    pub seeds: SeedsMode,
    pub admin_hotkeys: Vec<String>,
    pub admin_uris: String,
    pub key_uri: String,
    pub cors_origins: Vec<String>,
}

fn text(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

fn number<T: FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(name) {
        Ok(value) => value.trim().parse().with_context(|| format!("{name}={value:?} is not a number")),
        Err(_) => Ok(default),
    }
}

fn listed(value: &str) -> Vec<String> {
    value.split(',').map(str::trim).filter(|v| !v.is_empty()).map(String::from).collect()
}

impl Settings {
    pub fn from_env() -> Result<Settings> {
        let registry = match text("TASK_API_REGISTRY", "").as_str() {
            "chain" => RegistryMode::Chain { netuid: number("TASK_API_NETUID", 22)?, network: text("TASK_API_NETWORK", "finney") },
            "local" => RegistryMode::Local { validator_uris: text("TASK_API_VALIDATOR_URIS", "") },
            _ => bail!("TASK_API_REGISTRY must be set to chain or local"),
        };
        let seeds = match text("TASK_API_SEEDS", "local").as_str() {
            "chain" => SeedsMode::Chain { network: text("TASK_API_NETWORK", "finney") },
            _ => SeedsMode::Local { block_seconds: number("TASK_API_BLOCK_SECONDS", 12.0)?, genesis: number("TASK_API_GENESIS", 0.0)? },
        };
        let key_uri = text("TASK_API_KEY_URI", "");
        if key_uri.is_empty() && matches!(registry, RegistryMode::Chain { .. }) {
            bail!("TASK_API_KEY_URI must be set when TASK_API_REGISTRY=chain");
        }
        Ok(Settings {
            listen: text("TASK_API_LISTEN", "0.0.0.0:8080"),
            data: PathBuf::from(text("TASK_API_DATA", ".")),
            redis_url: text("TASK_API_REDIS", "redis://localhost:6379/15"),
            claim_ttl: number("TASK_API_CLAIM_TTL", CLAIM_TTL_S as f64)?,
            validation_ttl: number("TASK_API_VALIDATION_TTL", 900.0)?,
            poll_rate: number("TASK_API_POLL_RATE", 0.5)?,
            reads_per_minute: number("TASK_API_READS_PER_MINUTE", 120)?,
            log_reads_per_minute: number("TASK_API_LOG_READS_PER_MINUTE", 60)?,
            failed_writes_per_minute: number("TASK_API_FAILED_WRITES_PER_MINUTE", 30)?,
            max_upload: number("TASK_API_MAX_UPLOAD_BYTES", 100_000_000)?,
            max_backlog: number("TASK_API_MAX_BACKLOG_S", 43_200.0)?,
            max_attempts: number("TASK_API_MAX_ATTEMPTS", 3)?,
            ledger_delay: number("TASK_API_LEDGER_DELAY_S", 0.0)?,
            check_share: number("TASK_API_CHECK_SHARE", SHARE)?,
            checks_per_hour: number("TASK_API_CHECKS_PER_HOUR", CHECKS_PER_HOUR)?,
            queue_target: number("TASK_API_QUEUE_TARGET", 1_200)?,
            task_urls: number("TASK_API_TASK_URLS", 1_000)?,
            publish_lag: number("TASK_API_PUBLISH_LAG_S", LAG_TARGET_S)?,
            publish_lag_limit: number("TASK_API_PUBLISH_LAG_LIMIT_S", LAG_LIMIT_S)?,
            embed_tasks: text("TASK_API_EMBED_TASKS", "0") == "1",
            embed_model: text("TASK_API_EMBED_MODEL", "qwen3-embedding-8b"),
            active_s: number("TASK_API_ACTIVE_S", ACTIVE_S)?,
            publish_ttl: number("TASK_API_PUBLISH_TTL", 600.0)?,
            publish_tries: number("TASK_API_PUBLISH_TRIES", 5)?,
            r2_endpoint: text("CF_R2_ENDPOINT", ""),
            r2_access_key: text("CF_R2_ACCESS_KEY_ID", ""),
            r2_secret_key: text("CF_R2_SECRET_ACCESS_KEY", ""),
            bucket: text("CF_R2_BUCKET", "subnet-22"),
            prefix: text("TASK_API_R2_PREFIX", ""),
            pages_bucket: text("CF_R2_PAGES_BUCKET", "desearch-pages"),
            pages_prefix: text("CF_R2_PAGES_PREFIX", ""),
            registry,
            seeds,
            admin_hotkeys: listed(&text("TASK_API_ADMIN_HOTKEYS", "")),
            admin_uris: text("TASK_API_ADMIN_URIS", ""),
            key_uri,
            cors_origins: listed(&text("TASK_API_CORS_ORIGINS", DEFAULT_ORIGINS)),
        })
    }
}

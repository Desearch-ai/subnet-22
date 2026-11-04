//! Everything the task API is told through its environment, with the same names and defaults the Python service had.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context, Result};

use crate::budgets::{CRAWL, EMBED};
use crate::rules;
use crate::embedding::{self, DESEARCH_4B_V1};
use crate::queues::ACTIVE_S;
use crate::rounds::{CLAIM_TTL_S, EMBED_CLAIM_TTL_S};
use crate::sampling::{CHECKS_PER_HOUR, EMBED_CHECKS_PER_HOUR, EMBED_SHARE, SHARE};

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

/// The pages bucket's own endpoint and keys, when it is not on the R2 endpoint.
#[derive(Clone, Debug)]
pub struct PagesStore {
    pub endpoint: String,
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
}

impl PagesStore {
    pub fn from(var: impl Fn(&str) -> Option<String>) -> Result<Option<PagesStore>> {
        let Some(endpoint) = var("PAGES_S3_ENDPOINT").filter(|v| !v.is_empty()) else { return Ok(None) };
        let required = |name: &str| var(name).filter(|v| !v.is_empty()).with_context(|| format!("{name} must be set with PAGES_S3_ENDPOINT"));
        Ok(Some(PagesStore {
            endpoint,
            region: required("PAGES_S3_REGION")?,
            access_key: required("PAGES_S3_ACCESS_KEY_ID")?,
            secret_key: required("PAGES_S3_SECRET_ACCESS_KEY")?,
        }))
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub listen: String,
    pub data: PathBuf,
    pub redis_url: String,
    pub claim_ttl: f64,
    pub embed_claim_ttl: f64,
    pub validation_ttl: f64,
    pub poll_rate: f64,
    pub reads_per_minute: i64,
    pub log_reads_per_minute: i64,
    pub failed_writes_per_minute: i64,
    pub max_upload: u64,
    pub max_backlog: f64,
    pub max_attempts: i64,
    /// Times an embed task may be handed back before it is dropped at its next claim.
    pub max_handbacks: i64,
    pub ledger_delay: f64,
    pub check_share: f64,
    pub checks_per_hour: f64,
    pub embed_check_share: f64,
    pub embed_checks_per_hour: f64,
    pub queue_target: i64,
    pub embed_queue_target: i64,
    /// Seconds after the planner wrote an embed input that its task is dropped at its next claim; inputs are kept a day.
    pub embed_input_max_age: f64,
    /// Seconds a revealed round's tasks are released over, one by one, so miners see a steady supply rather than whole rounds at once.
    pub release_spread: f64,
    pub task_urls: usize,
    pub embed_tasks: bool,
    pub embed_model: String,
    pub active_s: f64,
    pub publish_ttl: f64,
    pub publish_tries: i64,
    pub r2_endpoint: String,
    pub r2_access_key: String,
    pub r2_secret_key: String,
    pub r2_region: String,
    /// The uploads bucket is private and validators read objects the API opens one by one.
    pub public_objects: bool,
    pub bucket: String,
    pub prefix: String,
    pub pages_bucket: String,
    pub pages_prefix: String,
    pub pages_store: Option<PagesStore>,
    pub registry: RegistryMode,
    pub seeds: SeedsMode,
    pub admin_hotkeys: Vec<String>,
    pub admin_uris: String,
    pub key_uri: String,
    pub cors_origins: Vec<String>,
    /// Seconds between storage probes; a failed one opens an outage that forgives the claims alive in it.
    pub storage_probe_s: f64,
    /// Blocks finality may fall behind the best block before the claims alive then are forgiven.
    pub finality_stall_blocks: i64,
    /// Matching reports an objective fail needs before it charges a miner anything.
    pub objective_quorum: usize,
    /// Validators of distinct coldkeys whose own models must find the same mismatch before an embed upload fails on it.
    pub embed_references: usize,
    /// Validator samples a round needs before its loadability lowers the coverage gate.
    pub loadability_samples: i64,
    /// Checked uploads with rows of a host a round needs before the host can be neutral.
    pub neutral_uploads: i64,
    /// Share of those uploads whose rows of the host are mostly errors past which the host is neutral for the round.
    pub neutral_share: f64,
    /// Share of the incentive its counted work implies at which a miner's budget is no longer capped; 0 turns the cap off.
    pub incentive_full: f64,
    /// Hours of counted work before a hotkey's incentive caps its budget.
    pub incentive_grace_h: i64,
    /// Each pool's share of the emission, as validators split it (`CRAWL_PERC`, `EMBED_PERC` in the validator).
    pub crawl_emission: f64,
    pub embed_emission: f64,
    /// Reports a validator may send that the task could not have produced, within the window, before its reports are refused.
    pub wrong_reports: i64,
    pub wrong_reports_window_s: f64,
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

pub(crate) fn listed(value: &str) -> Vec<String> {
    value.split(',').map(str::trim).filter(|v| !v.is_empty()).map(String::from).collect()
}

impl Settings {
    /// Seconds a miner has to upload a task of this kind after claiming it.
    pub fn claim_ttl_for(&self, kind: &str) -> f64 {
        if kind == EMBED {
            self.embed_claim_ttl
        } else {
            self.claim_ttl
        }
    }

    /// The share of an established hotkey's uploads of this kind drawn for a check, and the hourly budget of those draws.
    pub fn checks_for(&self, kind: &str) -> (f64, f64) {
        if kind == EMBED {
            (self.embed_check_share, self.embed_checks_per_hour)
        } else {
            (self.check_share, self.checks_per_hour)
        }
    }

    /// The knobs of the rules behind budgets, strikes and the API's count of work.
    pub fn rules(&self) -> rules::Params {
        rules::Params {
            coverage_gate: crate::credit::COVERAGE_GATE,
            objective_quorum: self.objective_quorum,
            embed_references: self.embed_references,
            loadability_samples: self.loadability_samples,
            neutral_uploads: self.neutral_uploads,
            neutral_share: self.neutral_share,
            incentive_full: self.incentive_full,
            incentive_grace_h: self.incentive_grace_h,
            emission: [(CRAWL.to_string(), self.crawl_emission), (EMBED.to_string(), self.embed_emission)].into_iter().collect(),
        }
    }

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
        let pages_store = PagesStore::from(|name| std::env::var(name).ok())?;
        let pages_bucket = text("CF_R2_PAGES_BUCKET", "desearch-pages");
        let embed_tasks = text("TASK_API_EMBED_TASKS", "0") == "1";
        let embed_model = text("TASK_API_EMBED_MODEL", DESEARCH_4B_V1.name);
        if embed_tasks && embedding::model(&embed_model).is_none() {
            bail!("TASK_API_EMBED_MODEL={embed_model:?} is not a model whose vectors the task API knows");
        }
        if key_uri.is_empty() && matches!(registry, RegistryMode::Chain { .. }) {
            bail!("TASK_API_KEY_URI must be set when TASK_API_REGISTRY=chain");
        }
        Ok(Settings {
            listen: text("TASK_API_LISTEN", "0.0.0.0:8080"),
            data: PathBuf::from(text("TASK_API_DATA", ".")),
            redis_url: text("TASK_API_REDIS", "redis://localhost:6379/15"),
            claim_ttl: number("TASK_API_CLAIM_TTL", CLAIM_TTL_S as f64)?,
            embed_claim_ttl: number("TASK_API_EMBED_CLAIM_TTL", EMBED_CLAIM_TTL_S as f64)?,
            validation_ttl: number("TASK_API_VALIDATION_TTL", 900.0)?,
            poll_rate: number("TASK_API_POLL_RATE", 0.5)?,
            reads_per_minute: number("TASK_API_READS_PER_MINUTE", 120)?,
            log_reads_per_minute: number("TASK_API_LOG_READS_PER_MINUTE", 60)?,
            failed_writes_per_minute: number("TASK_API_FAILED_WRITES_PER_MINUTE", 30)?,
            max_upload: number("TASK_API_MAX_UPLOAD_BYTES", 100_000_000)?,
            max_backlog: number("TASK_API_MAX_BACKLOG_S", 21_600.0)?,
            max_attempts: number("TASK_API_MAX_ATTEMPTS", 3)?,
            max_handbacks: number("TASK_API_MAX_HANDBACKS", 5)?,
            ledger_delay: number("TASK_API_LEDGER_DELAY_S", 0.0)?,
            check_share: number("TASK_API_CHECK_SHARE", SHARE)?,
            checks_per_hour: number("TASK_API_CHECKS_PER_HOUR", CHECKS_PER_HOUR)?,
            embed_check_share: number("TASK_API_EMBED_CHECK_SHARE", EMBED_SHARE)?,
            embed_checks_per_hour: number("TASK_API_EMBED_CHECKS_PER_HOUR", EMBED_CHECKS_PER_HOUR)?,
            queue_target: number("TASK_API_QUEUE_TARGET", 1_200)?,
            embed_queue_target: number("TASK_API_EMBED_QUEUE_TARGET", 100)?,
            embed_input_max_age: number("TASK_API_EMBED_INPUT_MAX_AGE_S", 72_000.0)?,
            release_spread: number("TASK_API_RELEASE_SPREAD_S", 60.0)?,
            task_urls: number("TASK_API_TASK_URLS", 1_000)?,
            embed_tasks,
            embed_model,
            active_s: number("TASK_API_ACTIVE_S", ACTIVE_S)?,
            publish_ttl: number("TASK_API_PUBLISH_TTL", 600.0)?,
            publish_tries: number("TASK_API_PUBLISH_TRIES", 5)?,
            r2_endpoint: text("CF_R2_ENDPOINT", ""),
            r2_access_key: text("CF_R2_ACCESS_KEY_ID", ""),
            r2_secret_key: text("CF_R2_SECRET_ACCESS_KEY", ""),
            r2_region: text("CF_R2_REGION", "auto"),
            public_objects: text("CF_R2_PUBLIC_OBJECTS", "0") == "1",
            bucket: text("CF_R2_BUCKET", "subnet-22"),
            prefix: text("TASK_API_R2_PREFIX", ""),
            pages_bucket: match pages_store {
                Some(_) => std::env::var("PAGES_S3_BUCKET").ok().filter(|v| !v.is_empty()).unwrap_or(pages_bucket),
                None => pages_bucket,
            },
            pages_prefix: text("CF_R2_PAGES_PREFIX", ""),
            pages_store,
            registry,
            seeds,
            admin_hotkeys: listed(&text("TASK_API_ADMIN_HOTKEYS", "")),
            admin_uris: text("TASK_API_ADMIN_URIS", ""),
            key_uri,
            cors_origins: listed(&text("TASK_API_CORS_ORIGINS", DEFAULT_ORIGINS)),
            storage_probe_s: number("TASK_API_STORAGE_PROBE_S", 30.0)?,
            finality_stall_blocks: number("TASK_API_FINALITY_STALL_BLOCKS", 10)?,
            objective_quorum: number("TASK_API_OBJECTIVE_QUORUM", 1)?,
            embed_references: number("TASK_API_EMBED_REFERENCES", 2)?,
            loadability_samples: number("TASK_API_LOADABILITY_SAMPLES", 30)?,
            neutral_uploads: number("TASK_API_NEUTRAL_UPLOADS", 3)?,
            neutral_share: number("TASK_API_NEUTRAL_SHARE", 0.5)?,
            incentive_full: number("TASK_API_INCENTIVE_FULL", 0.5)?,
            incentive_grace_h: number("TASK_API_INCENTIVE_GRACE_H", 24)?,
            crawl_emission: number("TASK_API_CRAWL_EMISSION", 0.2)?,
            embed_emission: number("TASK_API_EMBED_EMISSION", 0.5)?,
            wrong_reports: number("TASK_API_WRONG_REPORTS", 5)?,
            wrong_reports_window_s: number("TASK_API_WRONG_REPORTS_WINDOW_S", 86_400.0)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn the_pages_bucket_gets_its_own_endpoint_only_with_all_its_keys() {
        let lookup = |pairs: &[(&str, &str)]| -> HashMap<String, String> { pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() };
        let none = lookup(&[("CF_R2_ENDPOINT", "https://r2.example")]);
        assert!(PagesStore::from(|name| none.get(name).cloned()).unwrap().is_none());

        let keyless = lookup(&[("PAGES_S3_ENDPOINT", "https://s3.example"), ("PAGES_S3_REGION", "us-east-va")]);
        assert!(PagesStore::from(|name| keyless.get(name).cloned()).is_err());

        let own = lookup(&[
            ("PAGES_S3_ENDPOINT", "https://s3.example"),
            ("PAGES_S3_REGION", "us-east-va"),
            ("PAGES_S3_ACCESS_KEY_ID", "s3key"),
            ("PAGES_S3_SECRET_ACCESS_KEY", "s3secret"),
        ]);
        let store = PagesStore::from(|name| own.get(name).cloned()).unwrap().unwrap();
        assert_eq!((store.endpoint.as_str(), store.region.as_str(), store.access_key.as_str()), ("https://s3.example", "us-east-va", "s3key"));
    }
}

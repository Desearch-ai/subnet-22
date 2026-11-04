//! Everything the task API is told through its environment, with the same names and defaults the Python service had.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::budgets::{CRAWL, EMBED};
use crate::embedding::{self, DESEARCH_4B_V1};
use crate::queues::ACTIVE_S;
use crate::rounds::{CLAIM_TTL_S, EMBED_CLAIM_TTL_S};
use crate::rules;
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

/// Where and how often the log's head is committed on chain; off unless `TASK_API_ANCHOR_NETWORK` is set.
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorSettings {
    pub network: String,
    pub netuid: u16,
    /// The hotkey that commits; the log's signing key when empty. It must be registered on the subnet.
    pub key_uri: String,
    pub every_blocks: i64,
    /// Blocks between anchors at least, when a round opened since the last.
    pub min_blocks: i64,
    pub per_epoch: i64,
}

/// Where the receipt key comes from: a secret URI handed in, or a key the CVM makes for itself and never lets out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeySource {
    Uri,
    Cvm,
}

/// How the chain is read: how many endpoints must agree, which head, and whether blocks are counted on the clock in between.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainOptions {
    pub quorum: usize,
    pub finalized: bool,
    pub count_on: bool,
    pub refresh_s: f64,
}

impl Default for ChainOptions {
    fn default() -> Self {
        ChainOptions { quorum: 1, finalized: false, count_on: true, refresh_s: 30.0 }
    }
}

/// What only matters inside a TEE: the key, the attestation, custody of the log and the vote that moves it.
#[derive(Clone, Debug, PartialEq)]
pub struct TeeOptions {
    pub key_source: KeySource,
    pub dstack_socket: String,
    /// The PEM certificate the TLS proxy inside the CVM serves, whose public key the attestation binds.
    pub tls_cert: String,
    /// Off: this copy signs from the start, as before. On: it waits to be appointed or handed the log.
    pub custody: bool,
    pub custody_poll_s: f64,
    /// Share of the independent validators' stake that must approve a version, strictly more than.
    pub gov_threshold: f64,
    pub gov_timelock_blocks: i64,
    /// Measurements (`mrtd:rtmr0:rtmr1:rtmr2`) a successor may boot besides this CVM's own.
    pub successor_os: Vec<String>,
    pub pccs_url: String,
    /// The signer before custody records: the first record may be its handover.
    pub legacy_signer: String,
    /// Takes a successor's quote without Intel's signature, for the dstack simulator on a testnet; validators refuse such a compose.
    pub simulated: bool,
}

impl Default for TeeOptions {
    fn default() -> Self {
        TeeOptions {
            key_source: KeySource::Uri,
            dstack_socket: DSTACK_SOCKET.into(),
            tls_cert: String::new(),
            custody: false,
            custody_poll_s: 60.0,
            gov_threshold: 0.5,
            gov_timelock_blocks: 50_400,
            successor_os: Vec::new(),
            pccs_url: PCCS_URL.into(),
            legacy_signer: String::new(),
            simulated: false,
        }
    }
}

pub const DSTACK_SOCKET: &str = "/var/run/dstack.sock";
pub const PCCS_URL: &str = "https://pccs.phala.network";

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
    /// Seconds between storage canary probes; 0 turns the canary off.
    pub canary_every: f64,
    /// Seconds each pinned upload version is put under Object Lock (compliance mode) at its completion; 0 locks
    /// none. Needs a versioned bucket with Object Lock on.
    pub upload_lock_s: f64,
    /// Seconds between published files of the log.
    pub journal_every: f64,
    /// Seconds between the shares written into the log, for a replay to compare with its own.
    pub journal_shares_every: f64,
    /// Seconds between snapshots of the accounts in the log, which a replay may start from.
    pub snapshot_every: f64,
    /// Seconds published log entries stay in SQLite.
    pub journal_keep: f64,
    pub anchor: Option<AnchorSettings>,
    /// The header the proxy in front writes the caller's address into.
    pub client_address_header: String,
    pub chain: ChainOptions,
    pub tee: TeeOptions,
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
    /// The knobs that shape decisions, written into the log at every start so a replay and every miner see what the API runs with.
    pub fn journaled(&self) -> serde_json::Value {
        let registry = match &self.registry {
            RegistryMode::Chain { netuid, .. } => serde_json::json!({"mode": "chain", "netuid": netuid}),
            RegistryMode::Local { .. } => serde_json::json!({"mode": "local"}),
        };
        serde_json::json!({
            "claim_ttl": self.claim_ttl,
            "embed_claim_ttl": self.embed_claim_ttl,
            "validation_ttl": self.validation_ttl,
            "poll_rate": self.poll_rate,
            "max_backlog": self.max_backlog,
            "max_attempts": self.max_attempts,
            "max_handbacks": self.max_handbacks,
            "ledger_delay": self.ledger_delay,
            "check_share": self.check_share,
            "checks_per_hour": self.checks_per_hour,
            "embed_check_share": self.embed_check_share,
            "embed_checks_per_hour": self.embed_checks_per_hour,
            "queue_target": self.queue_target,
            "embed_queue_target": self.embed_queue_target,
            "embed_input_max_age": self.embed_input_max_age,
            "release_spread": self.release_spread,
            "task_urls": self.task_urls,
            "embed_tasks": self.embed_tasks,
            "embed_model": self.embed_model,
            "active_s": self.active_s,
            "max_upload": self.max_upload,
            "canary_every": self.canary_every,
            "upload_lock_s": self.upload_lock_s,
            "registry": registry,
            "seeds": match self.seeds { SeedsMode::Chain { .. } => "chain", SeedsMode::Local { .. } => "local" },
            "block_seconds": match self.seeds { SeedsMode::Chain { .. } => crate::seeds::BLOCK_SECONDS, SeedsMode::Local { block_seconds, .. } => block_seconds },
            "anchored": self.anchor.is_some(),
            "rules": self.rules(),
            "wrong_reports": self.wrong_reports,
            "wrong_reports_window_s": self.wrong_reports_window_s,
        })
    }

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
        let key_source = match text("TASK_API_KEY_SOURCE", "uri").as_str() {
            "uri" => KeySource::Uri,
            "cvm" => KeySource::Cvm,
            other => bail!("TASK_API_KEY_SOURCE={other:?} must be uri or cvm"),
        };
        let custody = text("TASK_API_CUSTODY", "0") == "1";
        if key_source == KeySource::Cvm && !key_uri.is_empty() {
            bail!("TASK_API_KEY_URI must not be set when TASK_API_KEY_SOURCE=cvm: the key is made inside the CVM");
        }
        if custody && key_source != KeySource::Cvm {
            bail!("TASK_API_CUSTODY=1 needs TASK_API_KEY_SOURCE=cvm");
        }
        let chain = ChainOptions {
            quorum: number("TASK_API_CHAIN_QUORUM", 1)?,
            finalized: text("TASK_API_CHAIN_FINALIZED", "0") == "1",
            count_on: text("TASK_API_CHAIN_COUNT_ON", "1") == "1",
            refresh_s: number("TASK_API_CHAIN_REFRESH_S", 30.0)?,
        };
        if chain.quorum == 0 {
            bail!("TASK_API_CHAIN_QUORUM must be at least 1");
        }
        let gov_threshold: f64 = number("TASK_API_GOV_THRESHOLD", 0.5)?;
        if !(0.0..1.0).contains(&gov_threshold) {
            bail!("TASK_API_GOV_THRESHOLD={gov_threshold} must be at least 0 and below 1");
        }
        let tee = TeeOptions {
            key_source,
            dstack_socket: text("TASK_API_DSTACK_SOCKET", DSTACK_SOCKET),
            tls_cert: text("TASK_API_TLS_CERT", ""),
            custody,
            custody_poll_s: number("TASK_API_CUSTODY_POLL_S", 60.0)?,
            gov_threshold,
            gov_timelock_blocks: number("TASK_API_GOV_TIMELOCK_BLOCKS", 50_400)?,
            successor_os: listed(&text("TASK_API_SUCCESSOR_OS", "")),
            pccs_url: text("TASK_API_PCCS_URL", PCCS_URL),
            legacy_signer: text("TASK_API_LEGACY_SIGNER", ""),
            simulated: text("TASK_API_TEE_SIMULATED", "0") == "1",
        };
        let pages_store = PagesStore::from(|name| std::env::var(name).ok())?;
        let pages_bucket = text("CF_R2_PAGES_BUCKET", "desearch-pages");
        let embed_tasks = text("TASK_API_EMBED_TASKS", "0") == "1";
        let embed_model = text("TASK_API_EMBED_MODEL", DESEARCH_4B_V1.name);
        if embed_tasks && embedding::model(&embed_model).is_none() {
            bail!("TASK_API_EMBED_MODEL={embed_model:?} is not a model whose vectors the task API knows");
        }
        if key_uri.is_empty() && key_source == KeySource::Uri && matches!(registry, RegistryMode::Chain { .. }) {
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
            canary_every: number("TASK_API_CANARY_EVERY_S", 60.0)?,
            upload_lock_s: number("TASK_API_UPLOAD_LOCK_S", 0.0)?,
            journal_every: number("TASK_API_JOURNAL_EVERY_S", 30.0)?,
            journal_shares_every: number("TASK_API_JOURNAL_SHARES_EVERY_S", 600.0)?,
            snapshot_every: number("TASK_API_SNAPSHOT_EVERY_S", 21_600.0)?,
            journal_keep: number("TASK_API_JOURNAL_KEEP_S", 8.0 * 86_400.0)?,
            anchor: match text("TASK_API_ANCHOR_NETWORK", "").as_str() {
                "" => None,
                network => Some(AnchorSettings {
                    network: network.into(),
                    netuid: number("TASK_API_ANCHOR_NETUID", number("TASK_API_NETUID", 22)?)?,
                    key_uri: text("TASK_API_ANCHOR_KEY_URI", ""),
                    every_blocks: number("TASK_API_ANCHOR_EVERY_BLOCKS", 12)?,
                    min_blocks: number("TASK_API_ANCHOR_MIN_BLOCKS", 3)?,
                    per_epoch: number("TASK_API_ANCHOR_PER_EPOCH", crate::anchor::PER_EPOCH - 1)?,
                }),
            },
            client_address_header: text("TASK_API_CLIENT_ADDRESS_HEADER", crate::auth::CLIENT_ADDRESS_HEADER),
            chain,
            tee,
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

    /// Every setting this copy runs with, by the variable that sets it, secrets left out: what `/v1/config` serves.
    pub fn public(&self) -> Value {
        let (registry, netuid, validator_uris) = match &self.registry {
            RegistryMode::Chain { netuid, .. } => ("chain", Some(*netuid), None),
            RegistryMode::Local { validator_uris } => ("local", None, Some(validator_uris.clone())),
        };
        let (seeds, block_seconds, genesis) = match &self.seeds {
            SeedsMode::Chain { .. } => ("chain", None, None),
            SeedsMode::Local { block_seconds, genesis } => ("local", Some(*block_seconds), Some(*genesis)),
        };
        let network = match (&self.registry, &self.seeds) {
            (RegistryMode::Chain { network, .. }, _) | (_, SeedsMode::Chain { network }) => Some(network.clone()),
            _ => None,
        };
        let pages = self.pages_store.as_ref();
        let settings: Map<String, Value> = [
            ("TASK_API_LISTEN", json!(self.listen)),
            ("TASK_API_DATA", json!(self.data.display().to_string())),
            ("TASK_API_REDIS", json!(without_password(&self.redis_url))),
            ("TASK_API_CLAIM_TTL", json!(self.claim_ttl)),
            ("TASK_API_EMBED_CLAIM_TTL", json!(self.embed_claim_ttl)),
            ("TASK_API_VALIDATION_TTL", json!(self.validation_ttl)),
            ("TASK_API_POLL_RATE", json!(self.poll_rate)),
            ("TASK_API_READS_PER_MINUTE", json!(self.reads_per_minute)),
            ("TASK_API_LOG_READS_PER_MINUTE", json!(self.log_reads_per_minute)),
            ("TASK_API_FAILED_WRITES_PER_MINUTE", json!(self.failed_writes_per_minute)),
            ("TASK_API_MAX_UPLOAD_BYTES", json!(self.max_upload)),
            ("TASK_API_MAX_BACKLOG_S", json!(self.max_backlog)),
            ("TASK_API_MAX_ATTEMPTS", json!(self.max_attempts)),
            ("TASK_API_MAX_HANDBACKS", json!(self.max_handbacks)),
            ("TASK_API_LEDGER_DELAY_S", json!(self.ledger_delay)),
            ("TASK_API_CHECK_SHARE", json!(self.check_share)),
            ("TASK_API_CHECKS_PER_HOUR", json!(self.checks_per_hour)),
            ("TASK_API_EMBED_CHECK_SHARE", json!(self.embed_check_share)),
            ("TASK_API_EMBED_CHECKS_PER_HOUR", json!(self.embed_checks_per_hour)),
            ("TASK_API_QUEUE_TARGET", json!(self.queue_target)),
            ("TASK_API_EMBED_QUEUE_TARGET", json!(self.embed_queue_target)),
            ("TASK_API_EMBED_INPUT_MAX_AGE_S", json!(self.embed_input_max_age)),
            ("TASK_API_RELEASE_SPREAD_S", json!(self.release_spread)),
            ("TASK_API_TASK_URLS", json!(self.task_urls)),
            ("TASK_API_EMBED_TASKS", json!(self.embed_tasks)),
            ("TASK_API_EMBED_MODEL", json!(self.embed_model)),
            ("TASK_API_ACTIVE_S", json!(self.active_s)),
            ("TASK_API_PUBLISH_TTL", json!(self.publish_ttl)),
            ("TASK_API_PUBLISH_TRIES", json!(self.publish_tries)),
            ("CF_R2_ENDPOINT", json!(self.r2_endpoint)),
            ("CF_R2_REGION", json!(self.r2_region)),
            ("CF_R2_PUBLIC_OBJECTS", json!(self.public_objects)),
            ("CF_R2_BUCKET", json!(self.bucket)),
            ("TASK_API_R2_PREFIX", json!(self.prefix)),
            ("CF_R2_PAGES_BUCKET", json!(self.pages_bucket)),
            ("CF_R2_PAGES_PREFIX", json!(self.pages_prefix)),
            ("PAGES_S3_ENDPOINT", json!(pages.map(|p| p.endpoint.clone()))),
            ("PAGES_S3_REGION", json!(pages.map(|p| p.region.clone()))),
            ("TASK_API_REGISTRY", json!(registry)),
            ("TASK_API_NETUID", json!(netuid)),
            ("TASK_API_VALIDATOR_URIS", json!(validator_uris)),
            ("TASK_API_SEEDS", json!(seeds)),
            ("TASK_API_BLOCK_SECONDS", json!(block_seconds)),
            ("TASK_API_GENESIS", json!(genesis)),
            ("TASK_API_NETWORK", json!(network)),
            ("TASK_API_ADMIN_HOTKEYS", json!(self.admin_hotkeys)),
            ("TASK_API_ADMIN_URIS", json!(self.admin_uris)),
            ("TASK_API_CORS_ORIGINS", json!(self.cors_origins)),
            ("TASK_API_CLIENT_ADDRESS_HEADER", json!(self.client_address_header)),
            ("TASK_API_CHAIN_QUORUM", json!(self.chain.quorum)),
            ("TASK_API_CHAIN_FINALIZED", json!(self.chain.finalized)),
            ("TASK_API_CHAIN_COUNT_ON", json!(self.chain.count_on)),
            ("TASK_API_CHAIN_REFRESH_S", json!(self.chain.refresh_s)),
            (
                "TASK_API_KEY_SOURCE",
                json!(match self.tee.key_source {
                    KeySource::Uri => "uri",
                    KeySource::Cvm => "cvm",
                }),
            ),
            ("TASK_API_DSTACK_SOCKET", json!(self.tee.dstack_socket)),
            ("TASK_API_TLS_CERT", json!(self.tee.tls_cert)),
            ("TASK_API_CUSTODY", json!(self.tee.custody)),
            ("TASK_API_CUSTODY_POLL_S", json!(self.tee.custody_poll_s)),
            ("TASK_API_GOV_THRESHOLD", json!(self.tee.gov_threshold)),
            ("TASK_API_GOV_TIMELOCK_BLOCKS", json!(self.tee.gov_timelock_blocks)),
            ("TASK_API_SUCCESSOR_OS", json!(self.tee.successor_os)),
            ("TASK_API_PCCS_URL", json!(self.tee.pccs_url)),
            ("TASK_API_LEGACY_SIGNER", json!(self.tee.legacy_signer)),
            ("TASK_API_TEE_SIMULATED", json!(self.tee.simulated)),
            ("TASK_API_CANARY_EVERY_S", json!(self.canary_every)),
            ("TASK_API_STORAGE_PROBE_S", json!(self.storage_probe_s)),
            ("TASK_API_FINALITY_STALL_BLOCKS", json!(self.finality_stall_blocks)),
            ("TASK_API_OBJECTIVE_QUORUM", json!(self.objective_quorum)),
            ("TASK_API_EMBED_REFERENCES", json!(self.embed_references)),
            ("TASK_API_LOADABILITY_SAMPLES", json!(self.loadability_samples)),
            ("TASK_API_NEUTRAL_UPLOADS", json!(self.neutral_uploads)),
            ("TASK_API_NEUTRAL_SHARE", json!(self.neutral_share)),
            ("TASK_API_INCENTIVE_FULL", json!(self.incentive_full)),
            ("TASK_API_INCENTIVE_GRACE_H", json!(self.incentive_grace_h)),
            ("TASK_API_CRAWL_EMISSION", json!(self.crawl_emission)),
            ("TASK_API_EMBED_EMISSION", json!(self.embed_emission)),
            ("TASK_API_WRONG_REPORTS", json!(self.wrong_reports)),
            ("TASK_API_WRONG_REPORTS_WINDOW_S", json!(self.wrong_reports_window_s)),
            ("TASK_API_UPLOAD_LOCK_S", json!(self.upload_lock_s)),
            ("TASK_API_JOURNAL_EVERY_S", json!(self.journal_every)),
            ("TASK_API_JOURNAL_SHARES_EVERY_S", json!(self.journal_shares_every)),
            ("TASK_API_SNAPSHOT_EVERY_S", json!(self.snapshot_every)),
            ("TASK_API_JOURNAL_KEEP_S", json!(self.journal_keep)),
            ("TASK_API_ANCHOR_NETWORK", json!(self.anchor.as_ref().map(|a| a.network.clone()))),
            ("TASK_API_ANCHOR_NETUID", json!(self.anchor.as_ref().map(|a| a.netuid))),
            ("TASK_API_ANCHOR_EVERY_BLOCKS", json!(self.anchor.as_ref().map(|a| a.every_blocks))),
            ("TASK_API_ANCHOR_MIN_BLOCKS", json!(self.anchor.as_ref().map(|a| a.min_blocks))),
            ("TASK_API_ANCHOR_PER_EPOCH", json!(self.anchor.as_ref().map(|a| a.per_epoch))),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect();
        let secrets_set: Vec<&str> = [
            ("CF_R2_ACCESS_KEY_ID", &self.r2_access_key),
            ("CF_R2_SECRET_ACCESS_KEY", &self.r2_secret_key),
            ("PAGES_S3_ACCESS_KEY_ID", &pages.map(|p| p.access_key.clone()).unwrap_or_default()),
            ("PAGES_S3_SECRET_ACCESS_KEY", &pages.map(|p| p.secret_key.clone()).unwrap_or_default()),
            ("TASK_API_KEY_URI", &self.key_uri),
            ("TASK_API_ANCHOR_KEY_URI", &self.anchor.as_ref().map(|a| a.key_uri.clone()).unwrap_or_default()),
        ]
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(name, _)| name)
        .collect();
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "commit": option_env!("TASK_API_COMMIT"),
            "settings": settings,
            "secrets_set": secrets_set,
        })
    }
}

/// A Redis URL with any password replaced, so `/v1/config` can show where Redis is without how to get in.
fn without_password(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else { return url.to_string() };
    match rest.rsplit_once('@') {
        Some((credentials, host)) => {
            let user = credentials.split(':').next().unwrap_or_default();
            format!("{scheme}://{user}:***@{host}")
        }
        None => url.to_string(),
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

    #[test]
    fn a_redis_password_is_kept_out_of_the_public_settings() {
        assert_eq!(without_password("redis://:hunter2@redis:6379/0"), "redis://:***@redis:6379/0");
        assert_eq!(without_password("rediss://api:hunter2@redis.example:6380/1"), "rediss://api:***@redis.example:6380/1");
        assert_eq!(without_password("redis://redis:6379/0"), "redis://redis:6379/0");
        assert_eq!(without_password("unix:///run/redis/redis.sock?db=0"), "unix:///run/redis/redis.sock?db=0");
    }
}

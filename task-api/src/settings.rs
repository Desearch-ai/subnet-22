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
    /// How many of the anchor network's endpoints must agree on what an anchor is signed over and what the chain holds.
    pub quorum: usize,
    pub netuid: u16,
    /// The hotkey that commits, never the log's signing key or the legacy signer. It must be registered on the subnet.
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
    /// Distinct coldkeys among the approving validators a version needs, besides their stake.
    pub gov_min_coldkeys: usize,
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
            gov_min_coldkeys: GOV_MIN_COLDKEYS,
            successor_os: Vec::new(),
            pccs_url: PCCS_URL.into(),
            legacy_signer: String::new(),
            simulated: false,
        }
    }
}

pub const DSTACK_SOCKET: &str = "/var/run/dstack.sock";
pub const GOV_MIN_COLDKEYS: usize = 3;
/// Seconds after its completion an upload can still be checked or taken back: its pinned version is locked at least this long in a CVM.
pub const CHECK_WINDOW_S: f64 = crate::credit::PENALTY_WINDOW_S;
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
    /// Distinct validators whose failed storage witnesses within `storage_witness_window` seconds of each other make a storage
    /// outage that forgives claims, whatever the API's own probes saw; 0 turns it off.
    pub storage_witness_quorum: i64,
    pub storage_witness_window: f64,
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
    /// Seconds the host clock may be off the chain's before signed requests are timed by the chain and claims are forgiven; 0 is off.
    pub clock_skew_s: f64,
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
    /// The knobs that shape decisions, written into the log at every start so a replay and every miner see what the API runs with:
    /// those a replay reads by name, and under `env` every setting by its variable (`knobs`), so none is left out.
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
            "storage_witness_quorum": self.storage_witness_quorum,
            "storage_witness_window": self.storage_witness_window,
            "registry": registry,
            "seeds": match self.seeds { SeedsMode::Chain { .. } => "chain", SeedsMode::Local { .. } => "local" },
            "block_seconds": match self.seeds { SeedsMode::Chain { .. } => crate::seeds::BLOCK_SECONDS, SeedsMode::Local { block_seconds, .. } => block_seconds },
            "anchored": self.anchor.is_some(),
            "rules": self.rules(),
            "wrong_reports": self.wrong_reports,
            "wrong_reports_window_s": self.wrong_reports_window_s,
            "anchor_hotkey": self.anchor.as_ref().filter(|a| !a.key_uri.is_empty()).map(|a| addresses_of(&a.key_uri)),
            "env": self.knobs(),
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
            gov_min_coldkeys: number("TASK_API_GOV_MIN_COLDKEYS", GOV_MIN_COLDKEYS)?,
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
        let settings = Settings {
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
            storage_witness_quorum: number("TASK_API_STORAGE_WITNESS_QUORUM", 3i64)?,
            storage_witness_window: number("TASK_API_STORAGE_WITNESS_WINDOW_S", 300.0)?,
            journal_every: number("TASK_API_JOURNAL_EVERY_S", 30.0)?,
            journal_shares_every: number("TASK_API_JOURNAL_SHARES_EVERY_S", 600.0)?,
            snapshot_every: number("TASK_API_SNAPSHOT_EVERY_S", 21_600.0)?,
            journal_keep: number("TASK_API_JOURNAL_KEEP_S", 8.0 * 86_400.0)?,
            anchor: match text("TASK_API_ANCHOR_NETWORK", "").as_str() {
                "" => None,
                network => Some(AnchorSettings {
                    network: network.into(),
                    quorum: number("TASK_API_ANCHOR_QUORUM", chain.quorum)?,
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
            clock_skew_s: number("TASK_API_CLOCK_SKEW_S", 30.0)?,
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
        };
        settings.check()?;
        Ok(settings)
    }

    /// Settings that each pass alone but not together. In a CVM, whose compose validators approve, every protection is on: the
    /// chain read through independent endpoints that must agree, finalized heads only, the log anchored by a key of its own, and
    /// each upload locked for as long as it can still be checked.
    pub fn check(&self) -> Result<()> {
        if let Some(anchor) = &self.anchor {
            let endpoints = crate::chain::endpoints(&anchor.network);
            if anchor.quorum == 0 || anchor.quorum > endpoints.len() {
                bail!("TASK_API_ANCHOR_QUORUM={} needs that many endpoints in TASK_API_ANCHOR_NETWORK, which names {}", anchor.quorum, endpoints.len());
            }
            if anchor.key_uri.is_empty() {
                bail!("TASK_API_ANCHOR_KEY_URI must be set with TASK_API_ANCHOR_NETWORK: anchors are committed by a hotkey of their own");
            }
            let anchorer = desearch::hotkey::Hotkey::from_uri(&anchor.key_uri).context("TASK_API_ANCHOR_KEY_URI")?.ss58();
            let signer = (!self.key_uri.is_empty()).then(|| desearch::hotkey::Hotkey::from_uri(&self.key_uri).map(|k| k.ss58())).transpose()?;
            if signer.as_deref() == Some(anchorer.as_str()) || self.tee.legacy_signer == anchorer {
                bail!("TASK_API_ANCHOR_KEY_URI is the log's signing key or the legacy signer; anchors need a key of their own");
            }
        }
        if self.tee.key_source != KeySource::Cvm {
            return Ok(());
        }
        let network = match (&self.registry, &self.seeds) {
            (RegistryMode::Chain { network, .. }, _) | (_, SeedsMode::Chain { network }) => network.as_str(),
            _ => bail!("TASK_API_KEY_SOURCE=cvm reads the chain: TASK_API_REGISTRY=chain and TASK_API_SEEDS=chain"),
        };
        independent("TASK_API_NETWORK", network, self.chain.quorum, "TASK_API_CHAIN_QUORUM")?;
        if !self.chain.finalized || self.chain.count_on {
            bail!(
                "TASK_API_KEY_SOURCE=cvm reads finalized blocks only and never counts them on the clock: TASK_API_CHAIN_FINALIZED=1, TASK_API_CHAIN_COUNT_ON=0"
            );
        }
        let Some(anchor) = &self.anchor else { bail!("TASK_API_KEY_SOURCE=cvm anchors its log: TASK_API_ANCHOR_NETWORK must be set") };
        independent("TASK_API_ANCHOR_NETWORK", &anchor.network, anchor.quorum, "TASK_API_ANCHOR_QUORUM")?;
        if self.upload_lock_s < CHECK_WINDOW_S {
            bail!("TASK_API_UPLOAD_LOCK_S={} is shorter than the {CHECK_WINDOW_S} s an upload can still be checked in", self.upload_lock_s);
        }
        if self.tee.gov_min_coldkeys == 0 {
            bail!("TASK_API_GOV_MIN_COLDKEYS must be at least 1");
        }
        Ok(())
    }

    /// Every setting this copy runs with, by the variable that sets it, secrets left out: what `/v1/config` serves and every `start`
    /// entry of the log holds. Each field is taken apart by name, so a setting added to `Settings` does not build until it is listed.
    pub fn knobs(&self) -> Map<String, Value> {
        let Settings {
            listen,
            data,
            redis_url,
            claim_ttl,
            embed_claim_ttl,
            validation_ttl,
            poll_rate,
            reads_per_minute,
            log_reads_per_minute,
            failed_writes_per_minute,
            max_upload,
            max_backlog,
            max_attempts,
            max_handbacks,
            ledger_delay,
            check_share,
            checks_per_hour,
            embed_check_share,
            embed_checks_per_hour,
            queue_target,
            embed_queue_target,
            embed_input_max_age,
            release_spread,
            task_urls,
            embed_tasks,
            embed_model,
            active_s,
            publish_ttl,
            publish_tries,
            r2_endpoint,
            r2_access_key: _,
            r2_secret_key: _,
            r2_region,
            public_objects,
            bucket,
            prefix,
            pages_bucket,
            pages_prefix,
            pages_store,
            registry,
            seeds,
            admin_hotkeys,
            admin_uris,
            key_uri: _,
            cors_origins,
            canary_every,
            upload_lock_s,
            storage_witness_quorum,
            storage_witness_window,
            journal_every,
            journal_shares_every,
            snapshot_every,
            journal_keep,
            anchor,
            client_address_header,
            chain,
            tee,
            storage_probe_s,
            finality_stall_blocks,
            clock_skew_s,
            objective_quorum,
            embed_references,
            loadability_samples,
            neutral_uploads,
            neutral_share,
            incentive_full,
            incentive_grace_h,
            crawl_emission,
            embed_emission,
            wrong_reports,
            wrong_reports_window_s,
        } = self;
        let ChainOptions { quorum, finalized, count_on, refresh_s } = chain;
        let TeeOptions {
            key_source,
            dstack_socket,
            tls_cert,
            custody,
            custody_poll_s,
            gov_threshold,
            gov_timelock_blocks,
            gov_min_coldkeys,
            successor_os,
            pccs_url,
            legacy_signer,
            simulated,
        } = tee;
        let (registry_mode, netuid, validator_uris) = match registry {
            RegistryMode::Chain { netuid, .. } => ("chain", Some(*netuid), None),
            RegistryMode::Local { validator_uris } => ("local", None, Some(addresses_of(validator_uris))),
        };
        let (seeds_mode, block_seconds, genesis) = match seeds {
            SeedsMode::Chain { .. } => ("chain", None, None),
            SeedsMode::Local { block_seconds, genesis } => ("local", Some(*block_seconds), Some(*genesis)),
        };
        let network = match (registry, seeds) {
            (RegistryMode::Chain { network, .. }, _) | (_, SeedsMode::Chain { network }) => Some(without_credentials(network)),
            _ => None,
        };
        let (pages_endpoint, pages_region) = match pages_store {
            Some(PagesStore { endpoint, region, access_key: _, secret_key: _ }) => (Some(endpoint.clone()), Some(region.clone())),
            None => (None, None),
        };
        // The anchoring key is a secret; `journaled` names its address.
        let anchored = anchor.as_ref().map(|AnchorSettings { network, quorum, netuid, key_uri: _, every_blocks, min_blocks, per_epoch }| {
            (without_credentials(network), *quorum, *netuid, *every_blocks, *min_blocks, *per_epoch)
        });
        let anchor_field =
            |pick: fn(&(String, usize, u16, i64, i64, i64)) -> Value| anchored.as_ref().map(pick).unwrap_or(Value::Null);
        [
            ("TASK_API_LISTEN", json!(listen)),
            ("TASK_API_DATA", json!(data.display().to_string())),
            ("TASK_API_REDIS", json!(without_password(redis_url))),
            ("TASK_API_CLAIM_TTL", json!(claim_ttl)),
            ("TASK_API_EMBED_CLAIM_TTL", json!(embed_claim_ttl)),
            ("TASK_API_VALIDATION_TTL", json!(validation_ttl)),
            ("TASK_API_POLL_RATE", json!(poll_rate)),
            ("TASK_API_READS_PER_MINUTE", json!(reads_per_minute)),
            ("TASK_API_LOG_READS_PER_MINUTE", json!(log_reads_per_minute)),
            ("TASK_API_FAILED_WRITES_PER_MINUTE", json!(failed_writes_per_minute)),
            ("TASK_API_MAX_UPLOAD_BYTES", json!(max_upload)),
            ("TASK_API_MAX_BACKLOG_S", json!(max_backlog)),
            ("TASK_API_MAX_ATTEMPTS", json!(max_attempts)),
            ("TASK_API_MAX_HANDBACKS", json!(max_handbacks)),
            ("TASK_API_LEDGER_DELAY_S", json!(ledger_delay)),
            ("TASK_API_CHECK_SHARE", json!(check_share)),
            ("TASK_API_CHECKS_PER_HOUR", json!(checks_per_hour)),
            ("TASK_API_EMBED_CHECK_SHARE", json!(embed_check_share)),
            ("TASK_API_EMBED_CHECKS_PER_HOUR", json!(embed_checks_per_hour)),
            ("TASK_API_QUEUE_TARGET", json!(queue_target)),
            ("TASK_API_EMBED_QUEUE_TARGET", json!(embed_queue_target)),
            ("TASK_API_EMBED_INPUT_MAX_AGE_S", json!(embed_input_max_age)),
            ("TASK_API_RELEASE_SPREAD_S", json!(release_spread)),
            ("TASK_API_TASK_URLS", json!(task_urls)),
            ("TASK_API_EMBED_TASKS", json!(embed_tasks)),
            ("TASK_API_EMBED_MODEL", json!(embed_model)),
            ("TASK_API_ACTIVE_S", json!(active_s)),
            ("TASK_API_PUBLISH_TTL", json!(publish_ttl)),
            ("TASK_API_PUBLISH_TRIES", json!(publish_tries)),
            ("CF_R2_ENDPOINT", json!(without_credentials(r2_endpoint))),
            ("CF_R2_REGION", json!(r2_region)),
            ("CF_R2_PUBLIC_OBJECTS", json!(public_objects)),
            ("CF_R2_BUCKET", json!(bucket)),
            ("TASK_API_R2_PREFIX", json!(prefix)),
            ("CF_R2_PAGES_BUCKET", json!(pages_bucket)),
            ("CF_R2_PAGES_PREFIX", json!(pages_prefix)),
            ("PAGES_S3_ENDPOINT", json!(pages_endpoint.map(|e| without_credentials(&e)))),
            ("PAGES_S3_REGION", json!(pages_region)),
            ("TASK_API_REGISTRY", json!(registry_mode)),
            ("TASK_API_NETUID", json!(netuid)),
            // Secret URIs: the addresses they make, never the URIs.
            ("TASK_API_VALIDATOR_URIS", json!(validator_uris)),
            ("TASK_API_SEEDS", json!(seeds_mode)),
            ("TASK_API_BLOCK_SECONDS", json!(block_seconds)),
            ("TASK_API_GENESIS", json!(genesis)),
            ("TASK_API_NETWORK", json!(network)),
            ("TASK_API_ADMIN_HOTKEYS", json!(admin_hotkeys)),
            ("TASK_API_ADMIN_URIS", json!(addresses_of(admin_uris))),
            ("TASK_API_CORS_ORIGINS", json!(cors_origins)),
            ("TASK_API_CLIENT_ADDRESS_HEADER", json!(client_address_header)),
            ("TASK_API_CHAIN_QUORUM", json!(quorum)),
            ("TASK_API_CHAIN_FINALIZED", json!(finalized)),
            ("TASK_API_CHAIN_COUNT_ON", json!(count_on)),
            ("TASK_API_CHAIN_REFRESH_S", json!(refresh_s)),
            (
                "TASK_API_KEY_SOURCE",
                json!(match key_source {
                    KeySource::Uri => "uri",
                    KeySource::Cvm => "cvm",
                }),
            ),
            ("TASK_API_DSTACK_SOCKET", json!(dstack_socket)),
            ("TASK_API_TLS_CERT", json!(tls_cert)),
            ("TASK_API_CUSTODY", json!(custody)),
            ("TASK_API_CUSTODY_POLL_S", json!(custody_poll_s)),
            ("TASK_API_GOV_THRESHOLD", json!(gov_threshold)),
            ("TASK_API_GOV_TIMELOCK_BLOCKS", json!(gov_timelock_blocks)),
            ("TASK_API_GOV_MIN_COLDKEYS", json!(gov_min_coldkeys)),
            ("TASK_API_SUCCESSOR_OS", json!(successor_os)),
            ("TASK_API_PCCS_URL", json!(pccs_url)),
            ("TASK_API_LEGACY_SIGNER", json!(legacy_signer)),
            ("TASK_API_TEE_SIMULATED", json!(simulated)),
            ("TASK_API_CANARY_EVERY_S", json!(canary_every)),
            ("TASK_API_STORAGE_PROBE_S", json!(storage_probe_s)),
            ("TASK_API_FINALITY_STALL_BLOCKS", json!(finality_stall_blocks)),
            ("TASK_API_CLOCK_SKEW_S", json!(clock_skew_s)),
            ("TASK_API_OBJECTIVE_QUORUM", json!(objective_quorum)),
            ("TASK_API_EMBED_REFERENCES", json!(embed_references)),
            ("TASK_API_LOADABILITY_SAMPLES", json!(loadability_samples)),
            ("TASK_API_NEUTRAL_UPLOADS", json!(neutral_uploads)),
            ("TASK_API_NEUTRAL_SHARE", json!(neutral_share)),
            ("TASK_API_INCENTIVE_FULL", json!(incentive_full)),
            ("TASK_API_INCENTIVE_GRACE_H", json!(incentive_grace_h)),
            ("TASK_API_CRAWL_EMISSION", json!(crawl_emission)),
            ("TASK_API_EMBED_EMISSION", json!(embed_emission)),
            ("TASK_API_WRONG_REPORTS", json!(wrong_reports)),
            ("TASK_API_WRONG_REPORTS_WINDOW_S", json!(wrong_reports_window_s)),
            ("TASK_API_UPLOAD_LOCK_S", json!(upload_lock_s)),
            ("TASK_API_STORAGE_WITNESS_QUORUM", json!(storage_witness_quorum)),
            ("TASK_API_STORAGE_WITNESS_WINDOW_S", json!(storage_witness_window)),
            ("TASK_API_JOURNAL_EVERY_S", json!(journal_every)),
            ("TASK_API_JOURNAL_SHARES_EVERY_S", json!(journal_shares_every)),
            ("TASK_API_SNAPSHOT_EVERY_S", json!(snapshot_every)),
            ("TASK_API_JOURNAL_KEEP_S", json!(journal_keep)),
            ("TASK_API_ANCHOR_NETWORK", anchor_field(|a| json!(a.0))),
            ("TASK_API_ANCHOR_QUORUM", anchor_field(|a| json!(a.1))),
            ("TASK_API_ANCHOR_NETUID", anchor_field(|a| json!(a.2))),
            ("TASK_API_ANCHOR_EVERY_BLOCKS", anchor_field(|a| json!(a.3))),
            ("TASK_API_ANCHOR_MIN_BLOCKS", anchor_field(|a| json!(a.4))),
            ("TASK_API_ANCHOR_PER_EPOCH", anchor_field(|a| json!(a.5))),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
    }

    /// Every setting this copy runs with, by the variable that sets it, secrets left out: what `/v1/config` serves.
    pub fn public(&self) -> Value {
        let pages = self.pages_store.as_ref();
        let secrets_set: Vec<&str> = [
            ("CF_R2_ACCESS_KEY_ID", &self.r2_access_key),
            ("CF_R2_SECRET_ACCESS_KEY", &self.r2_secret_key),
            ("PAGES_S3_ACCESS_KEY_ID", &pages.map(|p| p.access_key.clone()).unwrap_or_default()),
            ("PAGES_S3_SECRET_ACCESS_KEY", &pages.map(|p| p.secret_key.clone()).unwrap_or_default()),
            ("TASK_API_KEY_URI", &self.key_uri),
            ("TASK_API_ANCHOR_KEY_URI", &self.anchor.as_ref().map(|a| a.key_uri.clone()).unwrap_or_default()),
            ("TASK_API_ADMIN_URIS", &self.admin_uris),
        ]
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(name, _)| name)
        .collect();
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "commit": option_env!("TASK_API_COMMIT"),
            "settings": self.knobs(),
            "secrets_set": secrets_set,
        })
    }
}

/// The addresses comma-separated secret URIs make; one that does not parse shows as such, never as itself.
fn addresses_of(uris: &str) -> Vec<String> {
    let mut found: Vec<String> =
        listed(uris).iter().map(|uri| desearch::hotkey::Hotkey::from_uri(uri).map(|key| key.ss58()).unwrap_or_else(|_| "(not a key URI)".into())).collect();
    found.sort();
    found
}

/// Endpoint URLs, comma-separated, with any user, password or query left out: an RPC provider's key often rides in one.
fn without_credentials(urls: &str) -> String {
    urls.split(',')
        .map(|url| {
            let url = url.trim();
            let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
            let rest = rest.rsplit_once('@').map_or(rest, |(_, host)| host);
            let rest = rest.split_once('?').map_or(rest, |(path, _)| path);
            if scheme.is_empty() {
                rest.to_string()
            } else {
                format!("{scheme}://{rest}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// At least `quorum` endpoints in `network`, and no one operator holding as many: a name's operator is its registrable domain
/// (`opentensor.ai`), an address's is the address and port.
fn independent(name: &str, network: &str, quorum: usize, quorum_name: &str) -> Result<()> {
    if quorum < 2 {
        bail!("{quorum_name}={quorum}: TASK_API_KEY_SOURCE=cvm needs at least 2 chain endpoints to agree");
    }
    let operators: Vec<String> = crate::chain::endpoints(network).iter().map(|url| crate::chain::operator(url)).collect();
    if operators.len() < quorum {
        bail!("{quorum_name}={quorum} needs that many endpoints in {name}, which names {}", operators.len());
    }
    if let Some(one) = operators.iter().find(|op| operators.iter().filter(|other| other == op).count() >= quorum) {
        bail!("{name}: {one} runs {quorum_name}={quorum} or more of its endpoints, so it alone would make the quorum");
    }
    Ok(())
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
    fn an_endpoint_key_is_kept_out_of_the_public_settings() {
        assert_eq!(
            without_credentials("wss://user:pw@rpc.example/ws?apikey=s3cret, https://lite.chain.opentensor.ai:443"),
            "wss://rpc.example/ws,https://lite.chain.opentensor.ai:443"
        );
        assert_eq!(without_credentials("finney"), "finney");
        assert_eq!(addresses_of("//Alice, nonsense uri"), vec!["(not a key URI)".to_string(), desearch::hotkey::Hotkey::from_uri("//Alice").unwrap().ss58()]);
    }

    #[test]
    fn a_redis_password_is_kept_out_of_the_public_settings() {
        assert_eq!(without_password("redis://:hunter2@redis:6379/0"), "redis://:***@redis:6379/0");
        assert_eq!(without_password("rediss://api:hunter2@redis.example:6380/1"), "rediss://api:***@redis.example:6380/1");
        assert_eq!(without_password("redis://redis:6379/0"), "redis://redis:6379/0");
        assert_eq!(without_password("unix:///run/redis/redis.sock?db=0"), "unix:///run/redis/redis.sock?db=0");
    }
}

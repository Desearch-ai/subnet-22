//! Custody of the log: which key signs it, how a CVM is appointed by validators' stake or handed the log by the one before,
//! and the pause, the only emergency power. See docs/tee.md.
//!
//! A copy that does not hold the log (in standby, handing over, or retired) signs nothing: it serves reads, its attestation and
//! its custody, and turns every write away.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use bytes::Bytes;
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use desearch::canonical::hex;
use desearch::hotkey::{public_of, verify};
use desearch::r2::Error as StorageError;
use desearch::time::now;
use futures::future::BoxFuture;
use hkdf::Hkdf;
use rand::RngCore;
use redis::AsyncCommands;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::attestation::{decode_hex, sha256_hex};
use crate::governance::{self, Rule, Snapshot};
use crate::http::ApiError;
use crate::manifest::UPLOAD_LOG_LATEST;
use crate::py::canonical;
use crate::settings::RegistryMode;
use crate::state::State;
use crate::storage::put_json;
use crate::{uploadlog, verify as attest};

pub const LATEST: &str = "log/custody/latest.json";
pub const PAUSE_KEY: &str = "custody:pause";
pub const APPOINT: &str = "appoint";
pub const HANDOVER: &str = "handover";
pub const RECEIVE_PATH: &str = "/v1/custody/receive";
pub const HANDOVER_PATH: &str = "/v1/admin/handover";
/// A handed-over state is sent in one request: the compacted SQLite file and every Redis key.
pub const HANDOVER_BYTES: usize = 4 << 30;
const STATUS_FILE: &str = "custody.json";
const SEAL_INFO: &[u8] = b"sn22-task-api-handover-v1";
/// Requests already inside finish before the state is read.
const DRAIN: Duration = Duration::from_secs(5);
const PEER_TIMEOUT: Duration = Duration::from_secs(600);
const SCAN_COUNT: usize = 1000;

pub fn record_key(seq: i64) -> String {
    format!("log/custody/seq/{seq:012}.json")
}

pub fn attestation_key(signer: &str) -> String {
    format!("log/custody/attestations/{signer}.json")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Waits to be appointed or handed the log; signs nothing.
    Standby,
    Active,
    /// Writes are turned away while the state goes to the successor.
    HandingOver,
    /// Handed the log on, or replaced: signs nothing ever again.
    Retired,
}

#[derive(Default, Serialize, Deserialize)]
struct Kept {
    status: Option<Status>,
    /// The handover record a standby copy was sent, waiting for its predecessor to publish it.
    received: Option<Value>,
}

pub type SnapshotSource = Arc<dyn Fn() -> BoxFuture<'static, Result<Snapshot>> + Send + Sync>;
pub type QuoteCheck = Arc<dyn Fn(Vec<u8>) -> BoxFuture<'static, Result<Option<Vec<u8>>>> + Send + Sync>;

pub struct Custody {
    /// Off, a copy holds the log from the start as before, until it hands it over.
    pub enabled: bool,
    path: PathBuf,
    kept: Mutex<Kept>,
    paused: Mutex<Option<Value>>,
    /// One appointment or handover at a time.
    busy: tokio::sync::Mutex<()>,
    /// Where approvals come from; the chain unless a test says otherwise.
    pub snapshots: Mutex<Option<SnapshotSource>>,
    /// How a successor's quote signature is checked; Intel's DCAP chain unless a test says otherwise.
    pub quotes: Mutex<Option<QuoteCheck>>,
}

impl Custody {
    pub fn open(data: &std::path::Path, enabled: bool) -> Result<Custody> {
        let path = data.join("cvm").join(STATUS_FILE);
        let mut kept: Kept = match std::fs::read(&path) {
            Ok(raw) => serde_json::from_slice(&raw).with_context(|| format!("{path:?}"))?,
            Err(_) => Kept::default(),
        };
        kept.status.get_or_insert(if enabled { Status::Standby } else { Status::Active });
        Ok(Custody {
            enabled,
            path,
            kept: Mutex::new(kept),
            paused: Mutex::default(),
            busy: tokio::sync::Mutex::default(),
            snapshots: Mutex::default(),
            quotes: Mutex::default(),
        })
    }

    pub fn status(&self) -> Status {
        self.kept.lock().expect("custody").status.unwrap_or(Status::Standby)
    }

    /// Whether this copy may sign: it holds the log.
    pub fn signing(&self) -> bool {
        self.status() == Status::Active
    }

    pub fn received(&self) -> Option<Value> {
        self.kept.lock().expect("custody").received.clone()
    }

    fn keep(&self, status: Status, received: Option<Value>) -> Result<()> {
        let mut kept = self.kept.lock().expect("custody");
        *kept = Kept { status: Some(status), received };
        std::fs::create_dir_all(self.path.parent().expect("a directory"))?;
        let partial = self.path.with_extension("partial");
        std::fs::write(&partial, serde_json::to_vec(&*kept)?)?;
        std::fs::File::open(&partial)?.sync_all()?;
        std::fs::rename(&partial, &self.path)?;
        Ok(())
    }

    fn set(&self, status: Status) -> Result<()> {
        let received = self.received();
        self.keep(status, received)
    }

    pub fn paused(&self) -> Option<Value> {
        self.paused.lock().expect("pause").clone()
    }

    pub fn note_pause(&self, pause: Option<Value>) {
        *self.paused.lock().expect("pause") = pause;
    }
}

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS custody (seq INTEGER PRIMARY KEY, record TEXT NOT NULL)")?;
    Ok(())
}

pub fn records(conn: &Connection) -> Result<Vec<Value>> {
    let mut query = conn.prepare("SELECT record FROM custody ORDER BY seq")?;
    let rows = query.query_map([], |row| row.get::<_, String>(0))?;
    rows.map(|raw| Ok(serde_json::from_str(&raw?)?)).collect()
}

fn last(conn: &Connection) -> Result<Option<Value>> {
    let raw: Option<String> = conn.query_row("SELECT record FROM custody ORDER BY seq DESC LIMIT 1", [], |row| row.get(0)).optional()?;
    raw.map(|raw| Ok(serde_json::from_str(&raw)?)).transpose()
}

fn append(conn: &Connection, record: &Value) -> Result<()> {
    conn.execute("INSERT OR REPLACE INTO custody (seq, record) VALUES (?, ?)", params![record["seq"].as_i64(), record.to_string()])?;
    Ok(())
}

pub fn record_hash(record: &Value) -> String {
    sha256_hex(&canonical(record))
}

fn signed(state: &State, record: Map<String, Value>) -> Value {
    let mut record = record;
    let signature = state.sign(&canonical(&Value::Object(record.clone())));
    record.insert("signature".into(), signature.into());
    Value::Object(record)
}

/// Whoever must have signed a record did: an appointed CVM itself, or the signer handing over.
pub fn signature_holds(record: &Value) -> bool {
    let by = if record["kind"] == HANDOVER { &record["from"] } else { &record["signer"] };
    let mut unsigned = record.clone();
    let Some(fields) = unsigned.as_object_mut() else { return false };
    let signature = fields.remove("signature").and_then(|s| s.as_str().and_then(decode_hex));
    match (by.as_str().and_then(public_of), signature) {
        (Some(public), Some(signature)) => verify(&public, &canonical(&unsigned), &signature),
        _ => false,
    }
}

/// Why `record` does not follow `before` in the custody chain, None when it does.
pub fn link_problem(record: &Value, before: Option<&Value>, legacy_signer: &str) -> Option<String> {
    let seq = before.map_or(0, |b| b["seq"].as_i64().unwrap_or(-1) + 1);
    if record["seq"].as_i64() != Some(seq) {
        return Some(format!("seq {} where {seq} comes next", record["seq"]));
    }
    if record["prev"] != before.map_or(Value::Null, |b| Value::from(record_hash(b))) {
        return Some("prev does not name the last record".into());
    }
    if record["kind"] == HANDOVER {
        let holder = before.and_then(|b| b["signer"].as_str()).unwrap_or(legacy_signer);
        if holder.is_empty() || record["from"] != holder {
            return Some(format!("handed over by {}, not by the holder {holder:?}", record["from"]));
        }
    } else if record["kind"] != APPOINT {
        return Some(format!("kind {}", record["kind"]));
    }
    (!signature_holds(record)).then(|| "the signature does not hold".into())
}

async fn read_json(state: &State, key: &str) -> Result<Option<Value>> {
    match state.storage.get(key).await {
        Ok(body) => Ok(Some(serde_json::from_slice(&body).with_context(|| format!("{key} is not JSON"))?)),
        Err(StorageError::Missing) => Ok(None),
        Err(error) => Err(anyhow!("reading {key}: {error}")),
    }
}

/// The newest record anyone published, whoever wrote it.
pub async fn published(state: &State) -> Result<Option<Value>> {
    read_json(state, LATEST).await
}

async fn publish(state: &State, record: &Value, attestation: &Value) -> Result<()> {
    let seq = record["seq"].as_i64().context("a record without a seq")?;
    let signer = record["signer"].as_str().context("a record without a signer")?;
    put_json(&state.storage, &attestation_key(signer), attestation, None).await.map_err(|e| anyhow!("{e}"))?;
    put_json(&state.storage, &record_key(seq), record, None).await.map_err(|e| anyhow!("{e}"))?;
    put_json(&state.storage, LATEST, record, Some("no-store")).await.map_err(|e| anyhow!("{e}"))?;
    Ok(())
}

fn rule(state: &State) -> Rule {
    Rule::new(state.settings.tee.gov_threshold, state.settings.tee.gov_timelock_blocks)
}

async fn snapshot(state: &State) -> Result<Snapshot> {
    let source = state.custody.snapshots.lock().expect("snapshots").clone();
    if let Some(source) = source {
        return source().await;
    }
    let chain = state.chain.as_ref().context("approvals are read from the chain, and this copy has none")?;
    let netuid = match &state.settings.registry {
        RegistryMode::Chain { netuid, .. } => *netuid,
        RegistryMode::Local { .. } => 22,
    };
    governance::read(chain, netuid).await
}

async fn quote_check(state: &State, quote: Vec<u8>) -> Result<Option<Vec<u8>>> {
    let check = state.custody.quotes.lock().expect("quotes").clone();
    if let Some(check) = check {
        return check(quote).await;
    }
    if state.settings.tee.simulated {
        return Ok(None);
    }
    attest::dcap(state.settings.tee.pccs_url.clone(), quote).await
}

/// The last upload log file signed, so the next holder's turn starts after it.
async fn upload_log_seq(state: &State) -> Result<i64> {
    let next: Option<i64> = state.redis.clone().get(uploadlog::NEXT).await?;
    Ok(next.unwrap_or(1) - 1)
}

async fn compose_hash(state: &State) -> Result<String> {
    let tee = state.tee.as_ref().context("not in a TEE")?;
    tee.info().await?["compose_hash"].as_str().map(str::to_string).context("dstack gave no compose hash")
}

/// One round of custody upkeep: a standby copy takes the log when appointed or handed it; a holder stands down when replaced.
pub async fn poll(state: &State) -> Result<()> {
    if !state.custody.enabled {
        return Ok(());
    }
    let _busy = state.custody.busy.lock().await;
    match state.custody.status() {
        Status::Standby => match state.custody.received() {
            Some(record) => activate_received(state, record).await,
            None => appoint(state).await,
        },
        Status::Active => stand_down_if_replaced(state).await,
        Status::HandingOver | Status::Retired => Ok(()),
    }
}

/// Takes the log once the predecessor published the handover this copy received.
async fn activate_received(state: &State, record: Value) -> Result<()> {
    let Some(latest) = published(state).await? else { return Ok(()) };
    if record_hash(&latest) != record_hash(&record) {
        return Ok(());
    }
    state.db.run(move |conn| append(conn, &record)).await?;
    state.custody.keep(Status::Active, None)?;
    eprintln!("custody: handed the log, signing as {}", state.signer());
    Ok(())
}

/// Takes the log without a handover once validators' stake appointed this CVM: genesis, or recovery after a lost CVM.
async fn appoint(state: &State) -> Result<()> {
    let tee = state.tee.as_ref().context("custody needs the CVM's key")?;
    let version = compose_hash(state).await?;
    let signer = state.signer();
    let found = snapshot(state).await?;
    if !found.appointed(rule(state), &version, &signer) {
        return Ok(());
    }
    let head = published(state).await?;
    if let Some(head) = &head {
        if head["signer"] == signer.as_str() {
            bail!("the published record already names this CVM, but its own disk does not hold the log");
        }
        if !signature_holds(head) {
            bail!("the published custody record {} is not signed by who it says", head["seq"]);
        }
    }
    // A CVM starting afresh continues the upload log's numbering rather than writing over it.
    let written = read_json(state, UPLOAD_LOG_LATEST).await?.and_then(|l| l["seq"].as_i64()).unwrap_or(0);
    let local = upload_log_seq(state).await?;
    if written > local {
        let _: () = state.redis.clone().set(uploadlog::NEXT, written + 1).await?;
    }
    let attestation = tee.attestation(&signer, &state.settings.public(), None).await?;
    let mut record = Map::new();
    record.insert("kind".into(), APPOINT.into());
    record.insert("seq".into(), head.as_ref().map_or(0, |h| h["seq"].as_i64().unwrap_or(-1) + 1).into());
    record.insert("prev".into(), head.as_ref().map_or(Value::Null, |h| record_hash(h).into()));
    record.insert("signer".into(), signer.clone().into());
    record.insert("compose_hash".into(), version.into());
    record.insert("attestation_sha256".into(), sha256_hex(&canonical(&attestation["document"])).into());
    record.insert("upload_log_seq".into(), written.max(local).into());
    record.insert("block".into(), found.block.into());
    record.insert("at".into(), now().into());
    let record = signed(state, record);
    publish(state, &record, &attestation).await?;
    let kept = record.clone();
    state.db.run(move |conn| append(conn, &kept)).await?;
    state.custody.set(Status::Active)?;
    eprintln!("custody: appointed by validators' stake, signing as {signer} from record {}", record["seq"]);
    Ok(())
}

/// A holder whose log another CVM was appointed to, or whose own handover is published while its disk says otherwise, stops.
async fn stand_down_if_replaced(state: &State) -> Result<()> {
    let Some(latest) = published(state).await? else { return Ok(()) };
    let own = state.db.run(last).await?;
    let ahead = latest["seq"].as_i64() > own.as_ref().and_then(|o| o["seq"].as_i64()).or(Some(-1));
    if !ahead || latest["signer"] == state.signer().as_str() {
        return Ok(());
    }
    let replaced = if latest["kind"] == HANDOVER {
        // Only this CVM could have signed it: its disk was rolled back to before it handed over.
        latest["from"] == state.signer().as_str() && signature_holds(&latest)
    } else {
        let version = latest["compose_hash"].as_str().unwrap_or_default();
        signature_holds(&latest) && snapshot(state).await?.appointed(rule(state), version, latest["signer"].as_str().unwrap_or_default())
    };
    if replaced {
        state.custody.set(Status::Retired)?;
        eprintln!("custody: record {} names {}; this CVM signs no more", latest["seq"], latest["signer"]);
    }
    Ok(())
}

/// Runs the custody upkeep every `TASK_API_CUSTODY_POLL_S` until `stop`.
pub async fn run(state: Arc<State>, mut stop: tokio::sync::watch::Receiver<bool>) {
    let every = Duration::from_secs_f64(state.settings.tee.custody_poll_s.max(1.0));
    while !*stop.borrow() {
        if let Err(error) = poll(&state).await {
            eprintln!("custody upkeep failed: {error:#}");
        }
        tokio::select! {
            _ = tokio::time::sleep(every) => {}
            _ = stop.changed() => {}
        }
    }
}

/// Every Redis key with its time to live and dumped value, and the SQLite file compacted, framed as one blob.
async fn read_state(state: &State) -> Result<Vec<u8>> {
    let path = state.settings.data.join(format!("handover-{}.db", uuid::Uuid::new_v4().simple()));
    let target = path.display().to_string();
    state.db.whole(move |conn| Ok(conn.execute("VACUUM INTO ?1", [target])?)).await?;
    let sqlite = std::fs::read(&path);
    let _ = std::fs::remove_file(&path);
    let sqlite = sqlite?;
    let mut redis = state.redis.clone();
    let mut keys = Vec::new();
    let mut cursor: u64 = 0;
    loop {
        let (next, page): (u64, Vec<Vec<u8>>) = redis::cmd("SCAN").arg(cursor).arg("COUNT").arg(SCAN_COUNT).query_async(&mut redis).await?;
        keys.extend(page);
        if next == 0 {
            break;
        }
        cursor = next;
    }
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let ttl: i64 = redis::cmd("PTTL").arg(&key).query_async(&mut redis).await?;
        let dumped: Option<Vec<u8>> = redis::cmd("DUMP").arg(&key).query_async(&mut redis).await?;
        if let Some(dumped) = dumped {
            entries.push(json!([b64.encode(&key), ttl.max(0), b64.encode(dumped)]));
        }
    }
    let mut blob = (sqlite.len() as u64).to_be_bytes().to_vec();
    blob.extend(sqlite);
    blob.extend(Value::Array(entries).to_string().into_bytes());
    Ok(blob)
}

/// Puts a predecessor's state in place of this copy's own, which in standby holds nothing that matters.
async fn write_state(state: &State, blob: &[u8]) -> Result<()> {
    let length = u64::from_be_bytes(blob.get(..8).context("a state too short")?.try_into().expect("eight bytes")) as usize;
    let sqlite = blob.get(8..8 + length).context("a state cut short")?;
    let entries: Vec<(String, i64, String)> = serde_json::from_slice(&blob[8 + length..]).context("the Redis keys")?;
    let path = state.settings.data.join(format!("handed-{}.db", uuid::Uuid::new_v4().simple()));
    std::fs::write(&path, sqlite)?;
    let source = path.clone();
    let restored = state.db.whole(move |conn| Ok(conn.restore(rusqlite::DatabaseName::Main, &source, None::<fn(rusqlite::backup::Progress)>)?)).await;
    let _ = std::fs::remove_file(&path);
    restored?;
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut redis = state.redis.clone();
    let _: () = redis::cmd("FLUSHDB").query_async(&mut redis).await?;
    for (key, ttl, dumped) in entries {
        let _: () = redis::cmd("RESTORE").arg(b64.decode(key)?).arg(ttl).arg(b64.decode(dumped)?).arg("REPLACE").query_async(&mut redis).await?;
    }
    Ok(())
}

fn seal_key(shared: &[u8], ephemeral: &[u8; 32], recipient: &[u8; 32]) -> Key {
    let mut key = [0u8; 32];
    Hkdf::<Sha256>::new(Some(&[ephemeral.as_slice(), recipient.as_slice()].concat()), shared).expand(SEAL_INFO, &mut key).expect("32 bytes");
    Key::from(key)
}

/// Encrypts to an X25519 key: an ephemeral key's shared secret through HKDF-SHA256, then ChaCha20-Poly1305 bound to `aad`.
pub fn seal(plain: &[u8], recipient: &[u8; 32], aad: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let ephemeral = StaticSecret::random_from_rng(rand::rngs::OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral);
    let shared = ephemeral.diffie_hellman(&PublicKey::from(*recipient));
    let mut nonce = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let cipher = ChaCha20Poly1305::new(&seal_key(shared.as_bytes(), ephemeral_public.as_bytes(), recipient));
    let sealed = cipher.encrypt(Nonce::from_slice(&nonce), Payload { msg: plain, aad }).map_err(|_| anyhow!("sealing the state failed"))?;
    Ok(([ephemeral_public.as_bytes().as_slice(), &nonce].concat(), sealed))
}

pub fn unseal(secret: &StaticSecret, header: &[u8], sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    let ephemeral: [u8; 32] = header.get(..32).context("a seal without its key")?.try_into().expect("32 bytes");
    let nonce = header.get(32..44).context("a seal without its nonce")?;
    let recipient = PublicKey::from(secret);
    let shared = secret.diffie_hellman(&PublicKey::from(ephemeral));
    let cipher = ChaCha20Poly1305::new(&seal_key(shared.as_bytes(), &ephemeral, recipient.as_bytes()));
    cipher.decrypt(Nonce::from_slice(nonce), Payload { msg: sealed, aad }).map_err(|_| anyhow!("the state does not open with this CVM's key"))
}

/// The receive request: four bytes of length, the JSON of the record and the seal, then the sealed state.
fn frame(meta: &Value, sealed: &[u8]) -> Vec<u8> {
    let meta = meta.to_string().into_bytes();
    let mut out = (meta.len() as u32).to_be_bytes().to_vec();
    out.extend(meta);
    out.extend_from_slice(sealed);
    out
}

fn unframe(body: &[u8]) -> Result<(Value, &[u8])> {
    let length = u32::from_be_bytes(body.get(..4).context("an empty handover")?.try_into().expect("four bytes")) as usize;
    let meta = serde_json::from_slice(body.get(4..4 + length).context("a handover cut short")?)?;
    Ok((meta, &body[4 + length..]))
}

fn conflict(detail: impl Into<String>) -> ApiError {
    ApiError::status(409, detail)
}

/// The OS measurements a successor may boot: this CVM's own and those its compose allows.
async fn allowed_os(state: &State) -> Result<Vec<String>> {
    let mut allowed = state.settings.tee.successor_os.clone();
    if let Some(tee) = &state.tee {
        let info = tee.info().await?;
        let tcb: Value = info["tcb_info"].as_str().and_then(|t| serde_json::from_str(t).ok()).unwrap_or_default();
        let register = |name: &str| tcb[name].as_str().unwrap_or_default().to_string();
        allowed.push(format!("{}:{}:{}:{}", register("mrtd"), register("rtmr0"), register("rtmr1"), register("rtmr2")));
    }
    Ok(allowed)
}

/// Hands the log to the CVM at `successor` once its attestation holds and its version is approved; this copy then retires.
pub async fn hand_over(state: &State, successor: &str) -> Result<Value, ApiError> {
    let Ok(_busy) = state.custody.busy.try_lock() else { return Err(conflict("a handover or appointment is already running")) };
    if state.custody.status() != Status::Active {
        return Err(conflict(format!("this copy does not hold the log ({:?})", state.custody.status())));
    }
    let http = reqwest::Client::builder().timeout(PEER_TIMEOUT).build().map_err(anyhow::Error::from)?;
    let base = successor.trim_end_matches('/');
    let attestation: Value = http
        .get(format!("{base}/v1/attestation"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| conflict(format!("the successor's attestation: {e}")))?
        .json()
        .await
        .map_err(|e| conflict(format!("the successor's attestation: {e}")))?;
    let allowed = allowed_os(state).await?;
    let attested = attest::check(&attestation, &allowed, |quote| quote_check(state, quote))
        .await
        .map_err(|e| conflict(format!("the successor's attestation does not hold: {e:#}")))?;
    let found = snapshot(state).await?;
    if !found.approved(rule(state), &attested.compose_hash) {
        return Err(conflict(format!("version {} is not approved by validators' stake at block {}", attested.compose_hash, found.block)));
    }
    let to = attested.document["signer"].as_str().unwrap_or_default().to_string();
    let recipient: [u8; 32] = attested.document["handover_key"]
        .as_str()
        .and_then(decode_hex)
        .and_then(|k| k.try_into().ok())
        .ok_or_else(|| conflict("the successor has no handover key"))?;
    state.custody.set(Status::HandingOver)?;
    let sent = send(state, &http, base, &attested, &attestation, &to, &recipient, found.block).await;
    match sent {
        Ok(record) => {
            state.custody.set(Status::Retired)?;
            eprintln!("custody: handed the log to {to} in record {}; this copy signs no more", record["seq"]);
            Ok(json!({"record": record}))
        }
        Err(error) => {
            state.custody.set(Status::Active)?;
            Err(conflict(format!("the handover did not finish, this copy holds the log again: {error:#}")))
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn send(
    state: &State,
    http: &reqwest::Client,
    base: &str,
    attested: &attest::Attested,
    attestation: &Value,
    to: &str,
    recipient: &[u8; 32],
    block: i64,
) -> Result<Value> {
    tokio::time::sleep(DRAIN).await;
    while uploadlog::flush(state, &state.storage).await?.is_some() {}
    let blob = read_state(state).await?;
    let chain = state.db.run(records).await?;
    let mut record = Map::new();
    record.insert("kind".into(), HANDOVER.into());
    record.insert("seq".into(), chain.last().map_or(0, |h| h["seq"].as_i64().unwrap_or(-1) + 1).into());
    record.insert("prev".into(), chain.last().map_or(Value::Null, |h| record_hash(h).into()));
    record.insert("from".into(), state.signer().into());
    record.insert("signer".into(), to.into());
    record.insert("compose_hash".into(), attested.compose_hash.clone().into());
    record.insert("attestation_sha256".into(), sha256_hex(&canonical(&attested.document)).into());
    record.insert("state_sha256".into(), sha256_hex(&blob).into());
    record.insert("upload_log_seq".into(), upload_log_seq(state).await?.into());
    record.insert("block".into(), block.into());
    record.insert("at".into(), now().into());
    let record = signed(state, record);
    let hash = record_hash(&record);
    let (seal, sealed) = seal(&blob, recipient, hash.as_bytes())?;
    let meta = json!({"record": record, "chain": chain, "seal": hex(&seal)});
    let answer: Value = http.post(format!("{base}{RECEIVE_PATH}")).body(frame(&meta, &sealed)).send().await?.error_for_status()?.json().await?;
    let ack = answer["ack"].as_str().and_then(decode_hex).context("the successor did not confirm")?;
    let public = public_of(to).context("the successor's signer is not an address")?;
    if !verify(&public, &canonical(&json!({"ack": hash})), &ack) {
        bail!("the successor's confirmation is not signed by {to}");
    }
    let kept = record.clone();
    state.db.run(move |conn| append(conn, &kept)).await?;
    publish(state, &record, attestation).await?;
    Ok(record)
}

/// Takes a predecessor's state while in standby; this copy holds the log once the predecessor publishes the record.
pub async fn receive(state: &State, body: Bytes) -> Result<Value, ApiError> {
    let Ok(_busy) = state.custody.busy.try_lock() else { return Err(conflict("a handover or appointment is already running")) };
    let tee = state.tee.as_ref().ok_or_else(|| conflict("this copy does not run in a TEE"))?;
    if !state.custody.enabled || state.custody.status() != Status::Standby {
        return Err(conflict(format!("this copy takes a handover only in standby ({:?})", state.custody.status())));
    }
    let (meta, sealed) = unframe(&body).map_err(|e| ApiError::status(422, e.to_string()))?;
    let record = &meta["record"];
    let signer = state.signer();
    if record["kind"] != HANDOVER || record["signer"] != signer.as_str() {
        return Err(conflict("the handover is not to this CVM"));
    }
    let chain = meta["chain"].as_array().cloned().unwrap_or_default();
    let legacy = &state.settings.tee.legacy_signer;
    for (at, link) in chain.iter().enumerate() {
        if let Some(problem) = link_problem(link, at.checked_sub(1).map(|b| &chain[b]), legacy) {
            return Err(conflict(format!("custody record {at}: {problem}")));
        }
    }
    if let Some(problem) = link_problem(record, chain.last(), legacy) {
        return Err(conflict(format!("the handover: {problem}")));
    }
    if let Some(latest) = published(state).await? {
        if latest["seq"].as_i64() >= record["seq"].as_i64() {
            return Err(conflict("a later custody record is already published"));
        }
    }
    let own = tee.attestation(&signer, &state.settings.public(), None).await?;
    if record["attestation_sha256"] != sha256_hex(&canonical(&own["document"])).as_str() {
        return Err(conflict("the handover names another attestation of this CVM"));
    }
    let version = compose_hash(state).await?;
    if record["compose_hash"] != version.as_str() || !snapshot(state).await?.approved(rule(state), &version) {
        return Err(conflict("this CVM's version is not approved by validators' stake"));
    }
    let hash = record_hash(record);
    let seal = meta["seal"].as_str().and_then(decode_hex).ok_or_else(|| ApiError::status(422, "no seal"))?;
    let blob = unseal(&tee.handover, &seal, sealed, hash.as_bytes()).map_err(|e| conflict(e.to_string()))?;
    if record["state_sha256"] != sha256_hex(&blob).as_str() {
        return Err(conflict("the state does not match the record"));
    }
    write_state(state, &blob).await?;
    state.custody.keep(Status::Standby, Some(record.clone()))?;
    eprintln!("custody: state of {} received in record {}, waiting for it to be published", record["from"], record["seq"]);
    Ok(json!({"ack": state.sign(&canonical(&json!({"ack": hash})))}))
}

/// Stops or resumes handing out tasks; nothing else.
pub async fn pause(state: &State, paused: bool, reason: &str, by: &str) -> Result<Value> {
    let mut redis = state.redis.clone();
    let pause = paused.then(|| json!({"reason": reason, "since": now(), "by": by}));
    match &pause {
        Some(pause) => redis.set::<_, _, ()>(PAUSE_KEY, pause.to_string()).await?,
        None => redis.del::<_, ()>(PAUSE_KEY).await?,
    }
    state.custody.note_pause(pause.clone());
    eprintln!("custody: {} by {by}: {reason}", if paused { "paused" } else { "resumed" });
    Ok(json!({"paused": pause}))
}

/// The pause as Redis holds it, so every copy of the API sees the same.
pub async fn refresh_pause(state: &State) -> Result<()> {
    let raw: Option<String> = state.redis.clone().get(PAUSE_KEY).await?;
    state.custody.note_pause(raw.and_then(|raw| serde_json::from_str(&raw).ok()));
    Ok(())
}

/// What `/v1/custody` serves.
pub async fn view(state: &State) -> Result<Value> {
    let chain = state.lookups.read(records).await?;
    Ok(json!({
        "status": state.custody.status(),
        "signer": state.signer(),
        "paused": state.custody.paused(),
        "records": chain,
        "received": state.custody.received(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sealed_state_opens_only_with_the_recipients_key_and_its_record() {
        let secret = StaticSecret::random_from_rng(rand::rngs::OsRng);
        let recipient = *PublicKey::from(&secret).as_bytes();
        let (header, sealed) = seal(b"the state", &recipient, b"record").unwrap();
        assert_eq!(unseal(&secret, &header, &sealed, b"record").unwrap(), b"the state");
        assert!(unseal(&secret, &header, &sealed, b"another record").is_err());
        let other = StaticSecret::random_from_rng(rand::rngs::OsRng);
        assert!(unseal(&other, &header, &sealed, b"record").is_err());
    }

    #[test]
    fn a_frame_carries_the_record_then_the_seal() {
        let meta = json!({"record": {"seq": 1}});
        let framed = frame(&meta, b"sealed");
        let (found, rest) = unframe(&framed).unwrap();
        assert_eq!((found, rest), (meta, b"sealed".as_slice()));
    }
}

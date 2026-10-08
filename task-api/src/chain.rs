//! The subtensor chain over JSON-RPC: the best block, block hashes, and a subnet's metagraph decoded with the chain's own metadata.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use desearch::canonical::hex;
use frame_metadata::{RuntimeMetadata, RuntimeMetadataPrefixed};
use parity_scale_codec::Decode;
use scale_value::{Composite, Primitive, Value, ValueDef};
use serde_json::json;
use tokio::sync::Mutex;

use crate::seeds::BLOCK_SECONDS;

/// The best block is read this often; in between it is counted on the clock, at most a block behind.
const BLOCK_REFRESH: Duration = Duration::from_secs(30);
/// After a failed call every call waits this long, doubling up to `PAUSE_MAX`, so an unwell endpoint is not hammered.
const PAUSE_FIRST: Duration = Duration::from_secs(5);
const PAUSE_MAX: Duration = Duration::from_secs(30);
const RPC_TIMEOUT: Duration = Duration::from_secs(8);
const HASHES_KEPT: usize = 4096;
/// A block found not made yet is taken as still missing this long, so rounds waiting on it do not each ask.
const ABSENT_FOR: Duration = Duration::from_secs(2);
const METADATA_VERSION: u32 = 15;

/// One hotkey of the subnet, as the metagraph lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    pub uid: i64,
    pub hotkey: String,
    pub coldkey: String,
    pub validator_permit: bool,
    pub total_stake_rao: u128,
    pub alpha_stake_rao: u128,
}

pub struct Chain {
    http: reqwest::Client,
    url: String,
    best: std::sync::Mutex<Option<(i64, Instant)>>,
    reading: AtomicBool,
    first: Mutex<()>,
    pause: std::sync::Mutex<Pause>,
    hashes: Mutex<HashMap<i64, String>>,
    absent: std::sync::Mutex<Option<(i64, Instant)>>,
}

#[derive(Default)]
struct Pause {
    until: Option<Instant>,
    length: Duration,
}

/// The JSON-RPC endpoint of a bittensor network name, or the URL itself.
pub fn endpoint(network: &str) -> String {
    match network {
        "finney" => "https://entrypoint-finney.opentensor.ai:443".into(),
        "test" => "https://test.finney.opentensor.ai:443".into(),
        "archive" => "https://archive.chain.opentensor.ai:443".into(),
        "local" => "http://127.0.0.1:9944".into(),
        url => url.replacen("wss://", "https://", 1).replacen("ws://", "http://", 1),
    }
}

impl Chain {
    pub fn new(network: &str) -> Result<Chain> {
        let http = reqwest::Client::builder().timeout(RPC_TIMEOUT).build()?;
        Ok(Chain {
            http,
            url: endpoint(network),
            best: std::sync::Mutex::default(),
            reading: AtomicBool::new(false),
            first: Mutex::default(),
            pause: std::sync::Mutex::default(),
            hashes: Mutex::default(),
            absent: std::sync::Mutex::default(),
        })
    }

    async fn rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        if let Some(left) = self.paused_for() {
            bail!("{method}: the chain is not called for {:.0}s after failing", left.as_secs_f64());
        }
        let reply = self.call(method, params).await;
        let mut pause = self.pause.lock().expect("chain pause");
        match &reply {
            Ok(_) => *pause = Pause::default(),
            Err(_) => {
                pause.length = (pause.length * 2).clamp(PAUSE_FIRST, PAUSE_MAX);
                pause.until = Some(Instant::now() + pause.length);
            }
        }
        reply
    }

    async fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = json!({"id": 1, "jsonrpc": "2.0", "method": method, "params": params});
        let reply: serde_json::Value = self.http.post(&self.url).json(&request).send().await?.error_for_status()?.json().await?;
        if let Some(error) = reply.get("error") {
            bail!("{method}: {error}");
        }
        reply.get("result").cloned().ok_or_else(|| anyhow!("{method} answered without a result"))
    }

    fn paused_for(&self) -> Option<Duration> {
        let until = self.pause.lock().expect("chain pause").until?;
        until.checked_duration_since(Instant::now())
    }

    /// The best block, counted on from the last one read; only the very first call waits on the chain.
    pub async fn current_block(self: &Arc<Self>) -> Result<i64> {
        if let Some(block) = self.counted() {
            return Ok(block);
        }
        let _first = self.first.lock().await;
        if let Some(block) = self.counted() {
            return Ok(block);
        }
        self.read_best().await
    }

    fn counted(self: &Arc<Self>) -> Option<i64> {
        let (block, at) = (*self.best.lock().expect("best block"))?;
        if at.elapsed() >= BLOCK_REFRESH && self.paused_for().is_none() && !self.reading.swap(true, Ordering::AcqRel) {
            let chain = self.clone();
            tokio::spawn(async move {
                if let Err(error) = chain.read_best().await {
                    eprintln!("reading the best block failed: {error:#}");
                }
                chain.reading.store(false, Ordering::Release);
            });
        }
        Some(block + (at.elapsed().as_secs_f64() / BLOCK_SECONDS).floor() as i64)
    }

    /// Counts on from a block read earlier by this or another copy, so a restart does not wait on the chain.
    pub fn assume(&self, block: i64, read_at: f64) {
        let age = Duration::from_secs_f64((desearch::time::now() - read_at).max(0.0));
        let Some(at) = Instant::now().checked_sub(age) else { return };
        let mut best = self.best.lock().expect("best block");
        if best.is_none() {
            *best = Some((block, at));
        }
    }

    /// The last best block read and when, as a Unix time.
    pub fn last_read(&self) -> Option<(i64, f64)> {
        let (block, at) = (*self.best.lock().expect("best block"))?;
        Some((block, desearch::time::now() - at.elapsed().as_secs_f64()))
    }

    async fn read_best(&self) -> Result<i64> {
        let header = self.rpc("chain_getHeader", json!([])).await?;
        let number = header["number"].as_str().ok_or_else(|| anyhow!("a header without a number"))?;
        let block = i64::from_str_radix(number.trim_start_matches("0x"), 16).context("a header's number")?;
        *self.best.lock().expect("best block") = Some((block, Instant::now()));
        Ok(block)
    }

    /// The block's hash as `0x` and lowercase hex, None before it exists.
    pub async fn block_hash(&self, block: i64) -> Result<Option<String>> {
        if let Some(found) = self.hashes.lock().await.get(&block) {
            return Ok(Some(found.clone()));
        }
        if self.absent.lock().expect("absent block").is_some_and(|(from, at)| block >= from && at.elapsed() < ABSENT_FOR) {
            return Ok(None);
        }
        let found = self.rpc("chain_getBlockHash", json!([block])).await?;
        let Some(hash) = found.as_str().map(str::to_lowercase) else {
            *self.absent.lock().expect("absent block") = Some((block, Instant::now()));
            return Ok(None);
        };
        let mut hashes = self.hashes.lock().await;
        if hashes.len() >= HASHES_KEPT {
            hashes.clear();
        }
        hashes.insert(block, hash.clone());
        Ok(Some(hash))
    }

    async fn state_call(&self, method: &str, input: &[u8]) -> Result<Vec<u8>> {
        let found = self.rpc("state_call", json!([method, format!("0x{}", hex(input))])).await?;
        let text = found.as_str().ok_or_else(|| anyhow!("{method} returned no bytes"))?;
        decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("{method} returned bytes that are not hex"))
    }

    /// Every neuron of the subnet, from `SubnetInfoRuntimeApi.get_metagraph` as bittensor's metagraph reads it.
    pub async fn metagraph(&self, netuid: u16) -> Result<Vec<Neuron>> {
        let raw = self.state_call("Metadata_metadata_at_version", &METADATA_VERSION.to_le_bytes()).await?;
        let opaque = Option::<Vec<u8>>::decode(&mut raw.as_slice())?.ok_or_else(|| anyhow!("the chain has no v{METADATA_VERSION} metadata"))?;
        let RuntimeMetadata::V15(metadata) = RuntimeMetadataPrefixed::decode(&mut opaque.as_slice())?.1 else {
            bail!("the chain's metadata is not v{METADATA_VERSION}");
        };
        let method = metadata
            .apis
            .iter()
            .filter(|api| api.name == "SubnetInfoRuntimeApi")
            .flat_map(|api| &api.methods)
            .find(|method| method.name == "get_metagraph")
            .ok_or_else(|| anyhow!("the runtime has no SubnetInfoRuntimeApi.get_metagraph"))?;
        let input = method.inputs.first().ok_or_else(|| anyhow!("get_metagraph takes no netuid"))?;
        let mut encoded = Vec::new();
        scale_value::scale::encode_as_type(&Value::u128(netuid.into()), input.ty.id, &metadata.types, &mut encoded)
            .map_err(|e| anyhow!("encoding the netuid: {e}"))?;
        let output = self.state_call("SubnetInfoRuntimeApi_get_metagraph", &encoded).await?;
        let graph = scale_value::scale::decode_as_type(&mut output.as_slice(), method.output.id, &metadata.types)
            .map_err(|e| anyhow!("decoding the metagraph: {e}"))?;
        neurons(&graph)
    }
}

fn neurons(graph: &Value<u32>) -> Result<Vec<Neuron>> {
    let graph = unwrap_some(graph).ok_or_else(|| anyhow!("the subnet does not exist"))?;
    let column = |name: &str| -> Vec<&Value<u32>> { field(graph, name).map(items).unwrap_or_default() };
    let (hotkeys, coldkeys, permits, total, alpha) =
        (column("hotkeys"), column("coldkeys"), column("validator_permit"), column("total_stake"), column("alpha_stake"));
    hotkeys
        .iter()
        .enumerate()
        .map(|(uid, hotkey)| {
            let account = |value: Option<&&Value<u32>>| value.and_then(|v| account(v)).map(|key| desearch::hotkey::ss58(&key));
            Ok(Neuron {
                uid: uid as i64,
                hotkey: account(Some(hotkey)).ok_or_else(|| anyhow!("uid {uid} has no hotkey"))?,
                coldkey: account(coldkeys.get(uid)).unwrap_or_default(),
                validator_permit: permits.get(uid).and_then(|v| boolean(v)).unwrap_or(false),
                total_stake_rao: total.get(uid).and_then(|v| number(v)).unwrap_or(0),
                alpha_stake_rao: alpha.get(uid).and_then(|v| number(v)).unwrap_or(0),
            })
        })
        .collect()
}

fn unwrap_some(value: &Value<u32>) -> Option<&Value<u32>> {
    match &value.value {
        ValueDef::Variant(variant) if variant.name == "Some" => variant.values.values().next(),
        ValueDef::Variant(_) => None,
        _ => Some(value),
    }
}

fn field<'a>(value: &'a Value<u32>, name: &str) -> Option<&'a Value<u32>> {
    match &value.value {
        ValueDef::Composite(Composite::Named(fields)) => fields.iter().find(|(n, _)| n == name).map(|(_, v)| v),
        _ => None,
    }
}

fn items(value: &Value<u32>) -> Vec<&Value<u32>> {
    match &value.value {
        ValueDef::Composite(composite) => composite.values().collect(),
        _ => Vec::new(),
    }
}

/// The only value inside newtypes and single-field wrappers.
fn inner(value: &Value<u32>) -> &Value<u32> {
    match &value.value {
        ValueDef::Composite(composite) if composite.len() == 1 => inner(composite.values().next().expect("one value")),
        _ => value,
    }
}

fn number(value: &Value<u32>) -> Option<u128> {
    match &inner(value).value {
        ValueDef::Primitive(Primitive::U128(n)) => Some(*n),
        _ => None,
    }
}

fn boolean(value: &Value<u32>) -> Option<bool> {
    match &inner(value).value {
        ValueDef::Primitive(Primitive::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// A 32-byte account id, whatever wrappers it comes in.
fn account(value: &Value<u32>) -> Option<[u8; 32]> {
    fn bytes(value: &Value<u32>, out: &mut Vec<u8>) -> Option<()> {
        match &value.value {
            ValueDef::Primitive(Primitive::U128(n)) => out.push(u8::try_from(*n).ok()?),
            ValueDef::Composite(composite) => {
                for value in composite.values() {
                    bytes(value, out)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut out = Vec::with_capacity(32);
    bytes(value, &mut out)?;
    out.try_into().ok()
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

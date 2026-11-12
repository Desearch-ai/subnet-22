//! The subtensor chain over JSON-RPC: blocks and hashes, storage, runtime calls and a subnet's metagraph, decoded with the chain's own metadata.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use frame_metadata::v15::{RuntimeMetadataV15, StorageEntryMetadata, StorageEntryModifier, StorageEntryType, StorageHasher};
use frame_metadata::{RuntimeMetadata, RuntimeMetadataPrefixed};
use parity_scale_codec::Decode;
use scale_info::form::PortableForm;
use scale_info::{PortableRegistry, TypeDef, TypeDefPrimitive};
use scale_value::{Composite, Primitive, Value, ValueDef};
use serde_json::json;
use tokio::sync::Mutex;

use crate::canonical::hex;

pub const BLOCK_SECONDS: f64 = 12.0;
/// The best block is read this often; in between it is counted on the clock, at most a block behind.
const BLOCK_REFRESH: Duration = Duration::from_secs(30);
/// After a failed call an endpoint is skipped this long, doubling up to `PAUSE_MAX`, so an unwell one is not hammered while the next is asked.
const PAUSE_FIRST: Duration = Duration::from_secs(5);
const PAUSE_MAX: Duration = Duration::from_secs(30);
const RPC_TIMEOUT: Duration = Duration::from_secs(8);
const HASHES_KEPT: usize = 4096;
/// A block found not made yet is taken as still missing this long, so rounds waiting on it do not each ask.
const ABSENT_FOR: Duration = Duration::from_secs(2);
const METADATA_VERSION: u32 = 15;
const KEYS_PAGE: usize = 1000;
/// Account ids print with the generic substrate prefix, as bittensor prints them.
const ACCOUNT_PATH: [&str; 3] = ["sp_core", "crypto", "AccountId32"];

/// A value decoded with the metadata, each part tagged with its type id.
pub type Decoded = Value<u32>;

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

/// A JSON-RPC call the chain answered without a result: an `error` object, or neither.
#[derive(Debug)]
pub enum Refusal {
    Error { method: String, error: serde_json::Value },
    NoResult { method: String },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Error { method, error } => write!(f, "{method}: {error}"),
            Refusal::NoResult { method } => write!(f, "{method} answered without a result"),
        }
    }
}

impl std::error::Error for Refusal {}

pub struct Chain {
    http: reqwest::Client,
    /// Asked in this order; one failing is skipped for a while and the next asked.
    urls: Vec<String>,
    pausing: bool,
    best: std::sync::Mutex<Option<(i64, Instant)>>,
    reading: AtomicBool,
    first: Mutex<()>,
    pauses: std::sync::Mutex<Vec<Pause>>,
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
    /// One network, or several separated by commas to fall back on in order.
    pub fn new(network: &str) -> Result<Chain> {
        Chain::with_client(network, reqwest::Client::builder().timeout(RPC_TIMEOUT).build()?, true)
    }

    /// The same over a caller's HTTP client; without `pausing`, every call is sent even right after one failed.
    pub fn with_client(network: &str, http: reqwest::Client, pausing: bool) -> Result<Chain> {
        let urls: Vec<String> = network.split(',').map(str::trim).filter(|name| !name.is_empty()).map(endpoint).collect();
        if urls.is_empty() {
            bail!("no chain endpoint in {network:?}");
        }
        Ok(Chain {
            http,
            pausing,
            best: std::sync::Mutex::default(),
            reading: AtomicBool::new(false),
            first: Mutex::default(),
            pauses: std::sync::Mutex::new(urls.iter().map(|_| Pause::default()).collect()),
            urls,
            hashes: Mutex::default(),
            absent: std::sync::Mutex::default(),
        })
    }

    /// One JSON-RPC call's result, from the first endpoint that answers.
    pub async fn request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let mut failed = None;
        for (at, url) in self.urls.iter().enumerate() {
            if let Some(left) = self.paused_for(at) {
                failed.get_or_insert_with(|| anyhow!("{method}: the chain is not called for {:.0}s after failing", left.as_secs_f64()));
                continue;
            }
            let reply = self.call(url, method, params.clone()).await;
            let mut pauses = self.pauses.lock().expect("chain pause");
            match reply {
                Ok(found) => {
                    pauses[at] = Pause::default();
                    return Ok(found);
                }
                Err(error) => {
                    if self.pausing {
                        let pause = &mut pauses[at];
                        pause.length = (pause.length * 2).clamp(PAUSE_FIRST, PAUSE_MAX);
                        pause.until = Some(Instant::now() + pause.length);
                    }
                    failed = Some(error);
                }
            }
        }
        Err(failed.unwrap_or_else(|| anyhow!("{method}: no chain endpoint")))
    }

    async fn call(&self, url: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = json!({"id": 1, "jsonrpc": "2.0", "method": method, "params": params});
        let mut reply: serde_json::Value = self.http.post(url).json(&request).send().await?.error_for_status()?.json().await?;
        if let Some(error) = reply.get("error") {
            return Err(Refusal::Error { method: method.into(), error: error.clone() }.into());
        }
        reply.get_mut("result").map(serde_json::Value::take).ok_or_else(|| Refusal::NoResult { method: method.into() }.into())
    }

    fn paused_for(&self, at: usize) -> Option<Duration> {
        let until = self.pauses.lock().expect("chain pause")[at].until?;
        until.checked_duration_since(Instant::now())
    }

    fn all_paused(&self) -> bool {
        (0..self.urls.len()).all(|at| self.paused_for(at).is_some())
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
        if at.elapsed() >= BLOCK_REFRESH && !self.all_paused() && !self.reading.swap(true, Ordering::AcqRel) {
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
        let age = Duration::from_secs_f64((crate::time::now() - read_at).max(0.0));
        let Some(at) = Instant::now().checked_sub(age) else { return };
        let mut best = self.best.lock().expect("best block");
        if best.is_none() {
            *best = Some((block, at));
        }
    }

    /// The last best block read and when, as a Unix time.
    pub fn last_read(&self) -> Option<(i64, f64)> {
        let (block, at) = (*self.best.lock().expect("best block"))?;
        Some((block, crate::time::now() - at.elapsed().as_secs_f64()))
    }

    /// The best block as the chain reports it now.
    pub async fn read_best(&self) -> Result<i64> {
        let header = self.request("chain_getHeader", json!([])).await?;
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
        let found = self.request("chain_getBlockHash", json!([block])).await?;
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

    async fn state_call(&self, method: &str, input: &[u8], at: Option<&str>) -> Result<Vec<u8>> {
        let data = format!("0x{}", hex(input));
        let params = match at {
            Some(hash) => json!([method, data, hash]),
            None => json!([method, data]),
        };
        let found = self.request("state_call", params).await?;
        let text = found.as_str().ok_or_else(|| anyhow!("{method} returned no bytes"))?;
        decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("{method} returned bytes that are not hex"))
    }

    /// The runtime's v15 metadata at a block, the best block when `at` is None.
    pub async fn metadata(&self, at: Option<&str>) -> Result<RuntimeMetadataV15> {
        let raw = self.state_call("Metadata_metadata_at_version", &METADATA_VERSION.to_le_bytes(), at).await?;
        let opaque = Option::<Vec<u8>>::decode(&mut raw.as_slice())?.ok_or_else(|| anyhow!("the chain has no v{METADATA_VERSION} metadata"))?;
        let RuntimeMetadata::V15(metadata) = RuntimeMetadataPrefixed::decode(&mut opaque.as_slice())?.1 else {
            bail!("the chain's metadata is not v{METADATA_VERSION}");
        };
        Ok(metadata)
    }

    /// A runtime API call's answer, its arguments encoded and its output decoded by the metadata's types.
    pub async fn runtime_call(&self, metadata: &RuntimeMetadataV15, api: &str, method: &str, args: &[Value], at: Option<&str>) -> Result<Value<u32>> {
        let found = metadata
            .apis
            .iter()
            .filter(|found| found.name == api)
            .flat_map(|found| &found.methods)
            .find(|found| found.name == method)
            .ok_or_else(|| anyhow!("the runtime has no {api}.{method}"))?;
        if found.inputs.len() != args.len() {
            bail!("{api}.{method} takes {} arguments, not {}", found.inputs.len(), args.len());
        }
        let mut encoded = Vec::new();
        for (input, arg) in found.inputs.iter().zip(args) {
            scale_value::scale::encode_as_type(arg, input.ty.id, &metadata.types, &mut encoded)
                .map_err(|e| anyhow!("encoding {api}.{method}'s {}: {e}", input.name))?;
        }
        let output = self.state_call(&format!("{api}_{method}"), &encoded, at).await?;
        scale_value::scale::decode_as_type(&mut output.as_slice(), found.output.id, &metadata.types).map_err(|e| anyhow!("decoding {api}.{method}: {e}"))
    }

    /// A plain storage value at a block; a missing one is the item's default, or None when the item is optional.
    pub async fn storage(&self, metadata: &RuntimeMetadataV15, pallet: &str, item: &str, at: Option<&str>) -> Result<Option<Value<u32>>> {
        let (prefix, entry) = storage_entry(metadata, pallet, item)?;
        let StorageEntryType::Plain(ty) = &entry.ty else {
            bail!("{pallet}.{item} is a map");
        };
        let key = storage_prefix(prefix, &entry.name);
        let found = self.request("state_getStorage", json!([format!("0x{}", hex(&key)), at])).await?;
        let raw = match found.as_str() {
            Some(text) => decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("{pallet}.{item} is not hex"))?,
            None if matches!(entry.modifier, StorageEntryModifier::Default) => entry.default.clone(),
            None => return Ok(None),
        };
        let value = scale_value::scale::decode_as_type(&mut raw.as_slice(), ty.id, &metadata.types).map_err(|e| anyhow!("decoding {pallet}.{item}: {e}"))?;
        Ok(Some(value))
    }

    /// Every key and value of a one-key storage map at a block, in key order.
    pub async fn storage_map(&self, metadata: &RuntimeMetadataV15, pallet: &str, item: &str, at: Option<&str>) -> Result<Vec<(Value<u32>, Value<u32>)>> {
        let (prefix, entry) = storage_entry(metadata, pallet, item)?;
        let StorageEntryType::Map { hashers, key, value } = &entry.ty else {
            bail!("{pallet}.{item} is not a map");
        };
        let [hasher] = hashers.as_slice() else {
            bail!("{pallet}.{item} has {} keys", hashers.len());
        };
        let prefix = storage_prefix(prefix, &entry.name);
        let prefix_hex = format!("0x{}", hex(&prefix));
        let mut keys: Vec<String> = Vec::new();
        loop {
            let start = keys.last().cloned().unwrap_or_else(|| prefix_hex.clone());
            let page = self.request("state_getKeysPaged", json!([prefix_hex, KEYS_PAGE, start, at])).await?;
            let page: Vec<String> = serde_json::from_value(page).context("state_getKeysPaged keys")?;
            let done = page.len() < KEYS_PAGE;
            keys.extend(page);
            if done {
                break;
            }
        }
        let mut pairs = Vec::with_capacity(keys.len());
        for chunk in keys.chunks(KEYS_PAGE) {
            let found = self.request("state_queryStorageAt", json!([chunk, at])).await?;
            let mut values: HashMap<String, Option<String>> = HashMap::new();
            for set in found.as_array().into_iter().flatten() {
                for change in set["changes"].as_array().into_iter().flatten() {
                    if let Some(name) = change[0].as_str() {
                        values.insert(name.to_lowercase(), change[1].as_str().map(str::to_string));
                    }
                }
            }
            for name in chunk {
                let Some(Some(text)) = values.get(&name.to_lowercase()) else { continue };
                let raw_key = decode_hex(name.trim_start_matches("0x")).ok_or_else(|| anyhow!("a {pallet}.{item} key is not hex"))?;
                let mut rest = raw_key.get(prefix.len() + hash_len(hasher)..).ok_or_else(|| anyhow!("a short {pallet}.{item} key"))?;
                let decoded_key =
                    scale_value::scale::decode_as_type(&mut rest, key.id, &metadata.types).map_err(|e| anyhow!("decoding a {pallet}.{item} key: {e}"))?;
                let raw = decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("a {pallet}.{item} value is not hex"))?;
                let decoded = scale_value::scale::decode_as_type(&mut raw.as_slice(), value.id, &metadata.types)
                    .map_err(|e| anyhow!("decoding a {pallet}.{item} value: {e}"))?;
                pairs.push((decoded_key, decoded));
            }
        }
        Ok(pairs)
    }

    /// Every neuron of the subnet, from `SubnetInfoRuntimeApi.get_metagraph` as bittensor's metagraph reads it.
    pub async fn metagraph(&self, netuid: u16) -> Result<Vec<Neuron>> {
        let metadata = self.metadata(None).await?;
        let graph = self.runtime_call(&metadata, "SubnetInfoRuntimeApi", "get_metagraph", &[Value::u128(netuid.into())], None).await?;
        neurons(&graph)
    }
}

fn storage_entry<'a>(metadata: &'a RuntimeMetadataV15, pallet: &str, item: &str) -> Result<(&'a str, &'a StorageEntryMetadata<PortableForm>)> {
    let storage = metadata
        .pallets
        .iter()
        .find(|found| found.name == pallet)
        .and_then(|found| found.storage.as_ref())
        .ok_or_else(|| anyhow!("the runtime has no {pallet} storage"))?;
    let entry = storage.entries.iter().find(|entry| entry.name == item).ok_or_else(|| anyhow!("the runtime has no {pallet}.{item}"))?;
    Ok((&storage.prefix, entry))
}

fn hash_len(hasher: &StorageHasher) -> usize {
    match hasher {
        StorageHasher::Blake2_128Concat => 16,
        StorageHasher::Twox64Concat => 8,
        _ => 0,
    }
}

/// `xxhash64` under seeds 0 and 1, little-endian: substrate's `twox_128`.
pub fn twox128(data: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&twox_hash::XxHash64::oneshot(0, data).to_le_bytes());
    out[8..].copy_from_slice(&twox_hash::XxHash64::oneshot(1, data).to_le_bytes());
    out
}

fn storage_prefix(pallet: &str, item: &str) -> Vec<u8> {
    [twox128(pallet.as_bytes()), twox128(item.as_bytes())].concat()
}

/// Why a block's first extrinsic gives no moment.
#[derive(Debug, PartialEq, Eq)]
pub enum InherentError {
    /// Not hex, or not an unsigned `Timestamp.set` inherent.
    Invalid(String),
    /// The bytes end before a length or the moment.
    Short,
}

impl fmt::Display for InherentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InherentError::Invalid(why) => f.write_str(why),
            InherentError::Short => f.write_str("index out of range"),
        }
    }
}

impl std::error::Error for InherentError {}

fn compact(data: &[u8], at: usize) -> Result<(u128, usize), InherentError> {
    let first = *data.get(at).ok_or(InherentError::Short)?;
    let little = |from: usize, to: usize| -> u128 {
        let bytes = data.get(from.min(data.len())..to.min(data.len())).unwrap_or_default();
        bytes.iter().rev().take(16).fold(0u128, |n, b| (n << 8) | u128::from(*b))
    };
    Ok(match first & 3 {
        0 => (u128::from(first >> 2), at + 1),
        1 => (little(at, at + 2) >> 2, at + 2),
        2 => (little(at, at + 4) >> 2, at + 4),
        _ => {
            let size = usize::from(first >> 2) + 4;
            (little(at + 1, at + 1 + size), at + 1 + size)
        }
    })
}

/// The milliseconds in a block's first extrinsic, the unsigned `Timestamp.set` inherent.
pub fn inherent_moment(extrinsic_hex: &str) -> Result<u128, InherentError> {
    let text = extrinsic_hex.strip_prefix("0x").unwrap_or(extrinsic_hex);
    let data = decode_hex_spaced(text).ok_or_else(|| InherentError::Invalid("non-hexadecimal number found in fromhex() arg".into()))?;
    let (_, at) = compact(&data, 0)?;
    let version = *data.get(at).ok_or(InherentError::Short)?;
    if version & 0x80 != 0 || !matches!(version & 0x7F, 4 | 5) {
        return Err(InherentError::Invalid("the first extrinsic is not an unsigned inherent".into()));
    }
    Ok(compact(&data, at + 3)?.0)
}

/// `bytes.fromhex`: pairs of hex digits, whitespace allowed between pairs.
fn decode_hex_spaced(text: &str) -> Option<Vec<u8>> {
    let digits: Vec<u8> = text.bytes().collect();
    let mut out = Vec::with_capacity(digits.len() / 2);
    let mut i = 0;
    while i < digits.len() {
        if digits[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let pair = digits.get(i..i + 2)?;
        out.push(u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?);
        i += 2;
    }
    Some(out)
}

fn is_u8(types: &PortableRegistry, id: u32) -> bool {
    types.resolve(id).is_some_and(|ty| matches!(ty.type_def, TypeDef::Primitive(TypeDefPrimitive::U8)))
}

fn byte_values(composite: &Composite<u32>) -> Option<Vec<u8>> {
    composite
        .values()
        .map(|value| match &value.value {
            ValueDef::Primitive(Primitive::U128(n)) => u8::try_from(*n).ok(),
            _ => None,
        })
        .collect()
}

fn number(n: u128) -> serde_json::Value {
    match u64::try_from(n) {
        Ok(small) => json!(small),
        Err(_) => json!(n as f64),
    }
}

/// A decoded value as bittensor returns it: structs as dicts, byte vectors as text (hex when not UTF-8), accounts as SS58.
pub fn plain(value: &Value<u32>, types: &PortableRegistry) -> serde_json::Value {
    let ty = types.resolve(value.context);
    match &value.value {
        ValueDef::Primitive(primitive) => match primitive {
            Primitive::Bool(b) => json!(b),
            Primitive::Char(c) => json!(c.to_string()),
            Primitive::String(s) => json!(s),
            Primitive::U128(n) => number(*n),
            Primitive::I128(n) => i64::try_from(*n).map_or_else(|_| json!(*n as f64), |small| json!(small)),
            Primitive::U256(bytes) | Primitive::I256(bytes) => json!(format!("0x{}", hex(bytes))),
        },
        ValueDef::Composite(composite) => {
            let path: Vec<&str> = ty.map(|ty| ty.path.segments.iter().map(String::as_str).collect()).unwrap_or_default();
            if path == ACCOUNT_PATH {
                if let Some(key) = account(value) {
                    return json!(crate::hotkey::ss58(&key));
                }
            }
            match ty.map(|ty| &ty.type_def) {
                Some(TypeDef::Sequence(seq)) if is_u8(types, seq.type_param.id) => {
                    if let Some(bytes) = byte_values(composite) {
                        return match String::from_utf8(bytes) {
                            Ok(text) => json!(text),
                            Err(error) => json!(format!("0x{}", hex(error.as_bytes()))),
                        };
                    }
                }
                Some(TypeDef::Array(array)) if is_u8(types, array.type_param.id) => {
                    if let Some(bytes) = byte_values(composite) {
                        return json!(format!("0x{}", hex(&bytes)));
                    }
                }
                _ => {}
            }
            match composite {
                Composite::Named(fields) => serde_json::Value::Object(fields.iter().map(|(name, value)| (name.clone(), plain(value, types))).collect()),
                Composite::Unnamed(values) if values.len() == 1 && !matches!(ty.map(|ty| &ty.type_def), Some(TypeDef::Sequence(_) | TypeDef::Array(_))) => {
                    plain(&values[0], types)
                }
                Composite::Unnamed(values) => serde_json::Value::Array(values.iter().map(|value| plain(value, types)).collect()),
            }
        }
        ValueDef::Variant(variant) => match (variant.name.as_str(), variant.values.len()) {
            ("None", 0) => serde_json::Value::Null,
            ("Some", 1) => plain(variant.values.values().next().expect("one value"), types),
            (name, 0) => json!(name),
            (name, _) => {
                let inner = plain(&Value { value: ValueDef::Composite(variant.values.clone()), context: value.context }, types);
                serde_json::Value::Object([(name.to_string(), inner)].into_iter().collect())
            }
        },
        ValueDef::BitSequence(bits) => serde_json::Value::Array(bits.iter().map(|bit| json!(bit)).collect()),
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
            let account = |value: Option<&&Value<u32>>| value.and_then(|v| account(v)).map(|key| crate::hotkey::ss58(&key));
            Ok(Neuron {
                uid: uid as i64,
                hotkey: account(Some(hotkey)).ok_or_else(|| anyhow!("uid {uid} has no hotkey"))?,
                coldkey: account(coldkeys.get(uid)).unwrap_or_default(),
                validator_permit: permits.get(uid).and_then(|v| boolean(v)).unwrap_or(false),
                total_stake_rao: total.get(uid).and_then(|v| uint(v)).unwrap_or(0),
                alpha_stake_rao: alpha.get(uid).and_then(|v| uint(v)).unwrap_or(0),
            })
        })
        .collect()
}

pub fn unwrap_some(value: &Value<u32>) -> Option<&Value<u32>> {
    match &value.value {
        ValueDef::Variant(variant) if variant.name == "Some" => variant.values.values().next(),
        ValueDef::Variant(_) => None,
        _ => Some(value),
    }
}

pub fn field<'a>(value: &'a Value<u32>, name: &str) -> Option<&'a Value<u32>> {
    match &value.value {
        ValueDef::Composite(Composite::Named(fields)) => fields.iter().find(|(n, _)| n == name).map(|(_, v)| v),
        _ => None,
    }
}

pub fn items(value: &Value<u32>) -> Vec<&Value<u32>> {
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

/// An unsigned number, whatever newtypes it comes in.
pub fn uint(value: &Value<u32>) -> Option<u128> {
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
pub fn account(value: &Value<u32>) -> Option<[u8; 32]> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_prefixes_are_substrate_twox() {
        assert_eq!(hex(&storage_prefix("Timestamp", "Now")), "f0c365c3cf59d671eb72da0e7a4113c49f1f0515f462cdcf84e0f1d6045dfcbb");
        assert_eq!(hex(&twox128(b"System")), "26aa394eea5630e07c48ae0c9558cef7");
    }

    #[test]
    fn inherents_give_their_moment() {
        let moment: u64 = 1_703_549_172_000;
        let body = [&[0x04u8, 0x02, 0x00, 0x13][..], &moment.to_le_bytes()].concat();
        let extrinsic = [vec![(body.len() as u8) << 2], body].concat();
        assert_eq!(inherent_moment(&format!("0x{}", hex(&extrinsic))), Ok(u128::from(moment)));
        assert!(matches!(inherent_moment("0x1084020000"), Err(InherentError::Invalid(_))));
        assert_eq!(inherent_moment("0x"), Err(InherentError::Short));
        assert!(matches!(inherent_moment("0xzz"), Err(InherentError::Invalid(_))));
    }
}

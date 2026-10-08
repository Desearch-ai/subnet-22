//! The subtensor chain over JSON-RPC: the best block, block hashes, and a subnet's metagraph decoded with the chain's own metadata.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use desearch::canonical::hex;
use frame_metadata::{RuntimeMetadata, RuntimeMetadataPrefixed};
use parity_scale_codec::Decode;
use scale_value::{Composite, Primitive, Value, ValueDef};
use serde_json::json;
use tokio::sync::Mutex;

/// A block is about 12 seconds, so a best block read this recently is still the best block or one behind.
const BLOCK_FRESH: Duration = Duration::from_secs(1);
const RPC_TIMEOUT: Duration = Duration::from_secs(20);
const HASHES_KEPT: usize = 4096;
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
    block: Mutex<Option<(Instant, i64)>>,
    hashes: Mutex<HashMap<i64, String>>,
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
        Ok(Chain { http, url: endpoint(network), block: Mutex::default(), hashes: Mutex::default() })
    }

    async fn rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = json!({"id": 1, "jsonrpc": "2.0", "method": method, "params": params});
        let reply: serde_json::Value = self.http.post(&self.url).json(&request).send().await?.error_for_status()?.json().await?;
        if let Some(error) = reply.get("error") {
            bail!("{method}: {error}");
        }
        reply.get("result").cloned().ok_or_else(|| anyhow!("{method} answered without a result"))
    }

    /// The best block's number, as bittensor's `Subtensor.block` reads it.
    pub async fn current_block(&self) -> Result<i64> {
        let mut cached = self.block.lock().await;
        if let Some((at, block)) = *cached {
            if at.elapsed() < BLOCK_FRESH {
                return Ok(block);
            }
        }
        let header = self.rpc("chain_getHeader", json!([])).await?;
        let number = header["number"].as_str().ok_or_else(|| anyhow!("a header without a number"))?;
        let block = i64::from_str_radix(number.trim_start_matches("0x"), 16).context("a header's number")?;
        *cached = Some((Instant::now(), block));
        Ok(block)
    }

    /// The block's hash as `0x` and lowercase hex, None before it exists.
    pub async fn block_hash(&self, block: i64) -> Result<Option<String>> {
        if let Some(found) = self.hashes.lock().await.get(&block) {
            return Ok(Some(found.clone()));
        }
        let found = self.rpc("chain_getBlockHash", json!([block])).await?;
        let Some(hash) = found.as_str().map(str::to_lowercase) else { return Ok(None) };
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

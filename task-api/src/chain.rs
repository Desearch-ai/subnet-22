//! The subtensor chain over JSON-RPC: the best or finalized block, block hashes, storage and a subnet's metagraph decoded with the chain's own metadata.
//!
//! With a quorum above one, every answer is asked of all the endpoints and taken only when that many agree, so no single
//! endpoint can feed the API a seed, a height or a metagraph of its own.

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
use crate::settings::ChainOptions;

/// The finalized head, which every deadline is measured on, is read this often and never counted on between reads.
const FINALIZED_REFRESH: Duration = Duration::from_secs(6);
/// After a failed call an endpoint is skipped this long, doubling up to `PAUSE_MAX`, so an unwell one is not hammered while the next is asked.
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
    /// Its share of the subnet's incentive, as Yuma consensus last set it from validators' weights; None when the
    /// metagraph gives no incentive, or none to anyone yet.
    pub incentive: Option<f64>,
}

pub struct Chain {
    http: reqwest::Client,
    /// Asked in this order; one failing is skipped for a while and the next asked.
    urls: Vec<String>,
    /// How many endpoints must give the same answer; one keeps the first that answers.
    quorum: usize,
    /// The finalized head instead of the best block, so a seed never comes from a block that can still change.
    finalized: bool,
    /// Off, the current block is the last one read and never counted on the local clock, so time can only run slow.
    count_on: bool,
    /// The head is read this often; in between it is counted on the clock, at most a block behind, when `count_on`.
    refresh: Duration,
    best: std::sync::Mutex<Option<(i64, Instant)>>,
    reading: AtomicBool,
    first: Mutex<()>,
    finalized_head: std::sync::Mutex<Option<(i64, Instant)>>,
    finalizing: AtomicBool,
    first_finalized: Mutex<()>,
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

/// The endpoints of a comma-separated list of networks, in order.
pub fn endpoints(network: &str) -> Vec<String> {
    network.split(',').map(str::trim).filter(|name| !name.is_empty()).map(endpoint).collect()
}

/// Who runs an endpoint, as far as its URL says: the last two labels of a name, or an address (or a one-label name) with its port.
pub fn operator(url: &str) -> String {
    let Ok(parsed) = reqwest::Url::parse(url) else { return url.to_string() };
    match (parsed.domain(), parsed.host_str()) {
        (Some(name), _) if name.trim_end_matches('.').contains('.') => {
            let labels: Vec<&str> = name.trim_end_matches('.').rsplit('.').take(2).collect();
            labels.into_iter().rev().collect::<Vec<_>>().join(".").to_lowercase()
        }
        (_, Some(address)) => format!("{address}:{}", parsed.port_or_known_default().unwrap_or_default()),
        (_, None) => url.to_string(),
    }
}

impl Chain {
    /// One network, or several separated by commas to fall back on in order.
    pub fn new(network: &str) -> Result<Chain> {
        Chain::with(network, &ChainOptions::default())
    }

    /// Several networks, `quorum` of which must agree on every answer.
    pub fn with(network: &str, options: &ChainOptions) -> Result<Chain> {
        let http = reqwest::Client::builder().timeout(RPC_TIMEOUT).build()?;
        let urls = endpoints(network);
        if urls.is_empty() {
            bail!("no chain endpoint in {network:?}");
        }
        if options.quorum > urls.len() {
            bail!("a quorum of {} needs as many chain endpoints, {network:?} names {}", options.quorum, urls.len());
        }
        Ok(Chain {
            http,
            quorum: options.quorum.max(1),
            finalized: options.finalized,
            count_on: options.count_on,
            refresh: Duration::from_secs_f64(options.refresh_s.max(1.0)),
            best: std::sync::Mutex::default(),
            reading: AtomicBool::new(false),
            first: Mutex::default(),
            finalized_head: std::sync::Mutex::default(),
            finalizing: AtomicBool::new(false),
            first_finalized: Mutex::default(),
            pauses: std::sync::Mutex::new(urls.iter().map(|_| Pause::default()).collect()),
            urls,
            hashes: Mutex::default(),
            absent: std::sync::Mutex::default(),
        })
    }

    /// One JSON-RPC call, to the first endpoint that answers.
    pub async fn rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
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
                    let pause = &mut pauses[at];
                    pause.length = (pause.length * 2).clamp(PAUSE_FIRST, PAUSE_MAX);
                    pause.until = Some(Instant::now() + pause.length);
                    failed = Some(error);
                }
            }
        }
        Err(failed.unwrap_or_else(|| anyhow!("{method}: no chain endpoint")))
    }

    /// The answer `quorum` endpoints give alike; with a quorum of one, the first endpoint's that answers.
    pub async fn agreed(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        if self.quorum <= 1 {
            return self.rpc(method, params).await;
        }
        let answers = self.ask_all(method, params).await;
        let mut tally: Vec<(serde_json::Value, usize)> = Vec::new();
        for answer in answers.iter().flatten() {
            match tally.iter_mut().find(|(value, _)| value == answer) {
                Some((_, votes)) => *votes += 1,
                None => tally.push((answer.clone(), 1)),
            }
        }
        if let Some((value, _)) = tally.iter().find(|(_, votes)| *votes >= self.quorum) {
            return Ok(value.clone());
        }
        let failed = answers.iter().filter(|answer| answer.is_err()).count();
        bail!("{method}: {} of {} chain endpoints must agree; {} answers differed and {failed} failed", self.quorum, self.urls.len(), tally.len())
    }

    /// Sends to every endpoint at once, so no single one can drop it; the first answer of any that took it.
    pub async fn broadcast(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let answers = self.ask_all(method, params).await;
        let failed: Vec<String> = answers.iter().filter_map(|answer| answer.as_ref().err().map(|e| format!("{e:#}"))).collect();
        answers.into_iter().flatten().next().ok_or_else(|| anyhow!("{method}: no chain endpoint took it: {}", failed.join("; ")))
    }

    /// How many endpoints there are.
    pub fn endpoints(&self) -> usize {
        self.urls.len()
    }

    /// Every endpoint not paused, at once; a paused one counts as failed.
    async fn ask_all(&self, method: &str, params: serde_json::Value) -> Vec<Result<serde_json::Value>> {
        let calls = self.urls.iter().enumerate().map(|(at, url)| {
            let params = params.clone();
            async move {
                if let Some(left) = self.paused_for(at) {
                    bail!("{method}: {url} is not called for {:.0}s after failing", left.as_secs_f64());
                }
                let reply = self.call(url, method, params).await;
                let mut pauses = self.pauses.lock().expect("chain pause");
                match &reply {
                    Ok(_) => pauses[at] = Pause::default(),
                    Err(_) => {
                        let pause = &mut pauses[at];
                        pause.length = (pause.length * 2).clamp(PAUSE_FIRST, PAUSE_MAX);
                        pause.until = Some(Instant::now() + pause.length);
                    }
                }
                reply
            }
        });
        futures::future::join_all(calls).await
    }

    async fn call(&self, url: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = json!({"id": 1, "jsonrpc": "2.0", "method": method, "params": params});
        let reply: serde_json::Value = self.http.post(url).json(&request).send().await?.error_for_status()?.json().await?;
        if let Some(error) = reply.get("error") {
            bail!("{method}: {error}");
        }
        reply.get("result").cloned().ok_or_else(|| anyhow!("{method} answered without a result"))
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
        if at.elapsed() >= self.refresh && !self.all_paused() && !self.reading.swap(true, Ordering::AcqRel) {
            let chain = self.clone();
            tokio::spawn(async move {
                if let Err(error) = chain.read_best().await {
                    eprintln!("reading the best block failed: {error:#}");
                }
                chain.reading.store(false, Ordering::Release);
            });
        }
        if !self.count_on {
            return Some(block);
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

    /// The newest finalized block read from the chain, read again in the background once it is `FINALIZED_REFRESH` old.
    /// Unlike `current_block` it is never counted on the clock: a chain or an endpoint that stalls stalls every deadline with it.
    pub async fn finalized_block(self: &Arc<Self>) -> Result<i64> {
        if let Some(block) = self.last_finalized() {
            return Ok(block);
        }
        let _first = self.first_finalized.lock().await;
        if let Some(block) = self.last_finalized() {
            return Ok(block);
        }
        self.read_finalized().await
    }

    fn last_finalized(self: &Arc<Self>) -> Option<i64> {
        let (block, at) = (*self.finalized_head.lock().expect("finalized block"))?;
        if at.elapsed() >= FINALIZED_REFRESH && !self.all_paused() && !self.finalizing.swap(true, Ordering::AcqRel) {
            let chain = self.clone();
            tokio::spawn(async move {
                if let Err(error) = chain.read_finalized().await {
                    eprintln!("reading the finalized block failed: {error:#}");
                }
                chain.finalizing.store(false, Ordering::Release);
            });
        }
        Some(block)
    }

    async fn read_finalized(&self) -> Result<i64> {
        // Through the quorum, as every other height: no single endpoint sets the clock.
        let block = if self.quorum <= 1 { self.head_from(None, true).await? } else { self.agreed_head(true).await? };
        let mut kept = self.finalized_head.lock().expect("finalized block");
        // Endpoints that lag one another must not move the clock back.
        let block = kept.map_or(block, |(seen, _)| seen.max(block));
        *kept = Some((block, Instant::now()));
        Ok(block)
    }

    async fn read_best(&self) -> Result<i64> {
        let block = if self.quorum <= 1 { self.head_from(None, self.finalized).await? } else { self.agreed_head(self.finalized).await? };
        let mut best = self.best.lock().expect("best block");
        // A slower quorum never moves the clock back.
        let block = best.map_or(block, |(known, _)| known.max(block));
        *best = Some((block, Instant::now()));
        Ok(block)
    }

    /// The head of one endpoint, or of the first that answers: the finalized one when asked for, else the best.
    async fn head_from(&self, url: Option<&str>, finalized: bool) -> Result<i64> {
        let ask = |method: &'static str, params: serde_json::Value| async move {
            match url {
                Some(url) => self.call(url, method, params).await,
                None => self.rpc(method, params).await,
            }
        };
        let header = if finalized {
            let hash = ask("chain_getFinalizedHead", json!([])).await?;
            ask("chain_getHeader", json!([hash])).await?
        } else {
            ask("chain_getHeader", json!([])).await?
        };
        let number = header["number"].as_str().ok_or_else(|| anyhow!("a header without a number"))?;
        i64::from_str_radix(number.trim_start_matches("0x"), 16).context("a header's number")
    }

    /// The highest block `quorum` endpoints have all reached: one endpoint running ahead cannot make time pass faster.
    async fn agreed_head(&self, finalized: bool) -> Result<i64> {
        let heads = futures::future::join_all(self.urls.iter().enumerate().map(|(at, url)| async move {
            if self.paused_for(at).is_some() {
                return None;
            }
            match self.head_from(Some(url), finalized).await {
                Ok(block) => Some(block),
                Err(error) => {
                    let mut pauses = self.pauses.lock().expect("chain pause");
                    let pause = &mut pauses[at];
                    pause.length = (pause.length * 2).clamp(PAUSE_FIRST, PAUSE_MAX);
                    pause.until = Some(Instant::now() + pause.length);
                    eprintln!("chain endpoint {url}: {error:#}");
                    None
                }
            }
        }))
        .await;
        let mut heads: Vec<i64> = heads.into_iter().flatten().collect();
        heads.sort_unstable_by(|a, b| b.cmp(a));
        heads
            .get(self.quorum - 1)
            .copied()
            .ok_or_else(|| anyhow!("only {} of {} chain endpoints gave a head, {} must", heads.len(), self.urls.len(), self.quorum))
    }

    /// The block's hash as `0x` and lowercase hex, None before it exists.
    pub async fn block_hash(&self, block: i64) -> Result<Option<String>> {
        if let Some(found) = self.hashes.lock().await.get(&block) {
            return Ok(Some(found.clone()));
        }
        if self.absent.lock().expect("absent block").is_some_and(|(from, at)| block >= from && at.elapsed() < ABSENT_FOR) {
            return Ok(None);
        }
        if self.finalized && self.counted_or_read().await? < block {
            *self.absent.lock().expect("absent block") = Some((block, Instant::now()));
            return Ok(None);
        }
        let found = match self.agreed("chain_getBlockHash", json!([block])).await {
            Ok(found) => found,
            // Endpoints a block apart disagree about the newest block for a moment; it is taken as not made yet.
            Err(error) if self.quorum > 1 => {
                eprintln!("block {block}: {error:#}");
                serde_json::Value::Null
            }
            Err(error) => return Err(error),
        };
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

    /// The current block as `current_block` gives it, for callers that hold no `Arc`.
    async fn counted_or_read(&self) -> Result<i64> {
        if let Some((block, at)) = *self.best.lock().expect("best block") {
            if !self.count_on {
                return Ok(block);
            }
            return Ok(block + (at.elapsed().as_secs_f64() / BLOCK_SECONDS).floor() as i64);
        }
        self.read_best().await
    }

    /// The hash the endpoints agree the current head has: what a metagraph or storage read is pinned to.
    pub async fn head_hash(&self) -> Result<(i64, String)> {
        let block = self.counted_or_read().await?;
        let hash = self.block_hash(block).await?.ok_or_else(|| anyhow!("block {block} has no hash yet"))?;
        Ok((block, hash))
    }

    /// The raw value at a storage key, at a block, None when unset.
    pub async fn storage(&self, key: &[u8], at: &str) -> Result<Option<Vec<u8>>> {
        let found = self.agreed("state_getStorage", json!([format!("0x{}", hex(key)), at])).await?;
        found.as_str().map(|text| decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("storage that is not hex"))).transpose()
    }

    /// The values at many storage keys, at a block, in one call per batch: missing ones are left out.
    pub async fn storage_many(&self, keys: &[Vec<u8>], at: &str) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
        const BATCH: usize = 256;
        let mut values = HashMap::new();
        for batch in keys.chunks(BATCH) {
            let asked: Vec<String> = batch.iter().map(|key| format!("0x{}", hex(key))).collect();
            let found = self.agreed("state_queryStorageAt", json!([asked, at])).await?;
            for set in found.as_array().into_iter().flatten() {
                for change in set["changes"].as_array().into_iter().flatten() {
                    let (Some(key), Some(value)) = (change[0].as_str(), change[1].as_str()) else { continue };
                    let decoded = |text: &str| decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("storage that is not hex"));
                    values.insert(decoded(key)?, decoded(value)?);
                }
            }
        }
        Ok(values)
    }

    /// Every storage key under a prefix, at a block.
    pub async fn keys(&self, prefix: &[u8], at: &str) -> Result<Vec<Vec<u8>>> {
        const PAGE: usize = 1000;
        let mut keys = Vec::new();
        let mut start: Option<String> = None;
        loop {
            let found = self.agreed("state_getKeysPaged", json!([format!("0x{}", hex(prefix)), PAGE, start, at])).await?;
            let page: Vec<String> = serde_json::from_value(found).context("a page of storage keys")?;
            let done = page.len() < PAGE;
            start = page.last().cloned();
            for key in page {
                keys.push(decode_hex(key.trim_start_matches("0x")).ok_or_else(|| anyhow!("a storage key that is not hex"))?);
            }
            if done {
                return Ok(keys);
            }
        }
    }

    /// The runtime's v15 metadata at a block, for decoding storage and runtime calls.
    pub async fn metadata(&self, at: Option<&str>) -> Result<frame_metadata::v15::RuntimeMetadataV15> {
        let raw = self.state_call_at("Metadata_metadata_at_version", &METADATA_VERSION.to_le_bytes(), at).await?;
        let opaque = Option::<Vec<u8>>::decode(&mut raw.as_slice())?.ok_or_else(|| anyhow!("the chain has no v{METADATA_VERSION} metadata"))?;
        let RuntimeMetadata::V15(metadata) = RuntimeMetadataPrefixed::decode(&mut opaque.as_slice())?.1 else {
            bail!("the chain's metadata is not v{METADATA_VERSION}");
        };
        Ok(metadata)
    }

    /// A runtime call at the head, as the endpoints agree on it.
    pub async fn state_call(&self, method: &str, input: &[u8]) -> Result<Vec<u8>> {
        self.state_call_at(method, input, None).await
    }

    pub(crate) async fn state_call_at(&self, method: &str, input: &[u8], at: Option<&str>) -> Result<Vec<u8>> {
        let mut params = vec![json!(method), json!(format!("0x{}", hex(input)))];
        if let Some(at) = at {
            params.push(json!(at));
        }
        let found = self.agreed("state_call", serde_json::Value::Array(params)).await?;
        let text = found.as_str().ok_or_else(|| anyhow!("{method} returned no bytes"))?;
        decode_hex(text.trim_start_matches("0x")).ok_or_else(|| anyhow!("{method} returned bytes that are not hex"))
    }

    /// Every neuron of the subnet, from `SubnetInfoRuntimeApi.get_metagraph` as bittensor's metagraph reads it.
    pub async fn metagraph(&self, netuid: u16) -> Result<Vec<Neuron>> {
        // Endpoints a block apart answer differently, so a quorum reads at a block they agree on.
        let at = if self.quorum > 1 { Some(self.head_hash().await?.1) } else { None };
        self.metagraph_at(netuid, at.as_deref()).await
    }

    /// Every neuron of the subnet at a block, or at the endpoint's own head.
    pub async fn metagraph_at(&self, netuid: u16, at: Option<&str>) -> Result<Vec<Neuron>> {
        let metadata = self.metadata(at).await?;
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
        let output = self.state_call_at("SubnetInfoRuntimeApi_get_metagraph", &encoded, at).await?;
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
    let incentives: Vec<Option<u128>> = column("incentives").iter().map(|v| number(v)).collect();
    // A metagraph without incentives, or before any weights, says nothing about any miner.
    let known = incentives.iter().any(|n| n.is_some_and(|n| n > 0));
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
                incentive: incentives.get(uid).copied().flatten().filter(|_| known).map(|n| n as f64 / u16::MAX as f64),
            })
        })
        .collect()
}

pub(crate) fn unwrap_some(value: &Value<u32>) -> Option<&Value<u32>> {
    match &value.value {
        ValueDef::Variant(variant) if variant.name == "Some" => variant.values.values().next(),
        ValueDef::Variant(_) => None,
        _ => Some(value),
    }
}

pub(crate) fn field<'a>(value: &'a Value<u32>, name: &str) -> Option<&'a Value<u32>> {
    match &value.value {
        ValueDef::Composite(Composite::Named(fields)) => fields.iter().find(|(n, _)| n == name).map(|(_, v)| v),
        _ => None,
    }
}

pub(crate) fn items(value: &Value<u32>) -> Vec<&Value<u32>> {
    match &value.value {
        ValueDef::Composite(composite) => composite.values().collect(),
        _ => Vec::new(),
    }
}

/// The only value inside newtypes and single-field wrappers.
pub(crate) fn inner(value: &Value<u32>) -> &Value<u32> {
    match &value.value {
        ValueDef::Composite(composite) if composite.len() == 1 => inner(composite.values().next().expect("one value")),
        _ => value,
    }
}

pub(crate) fn number(value: &Value<u32>) -> Option<u128> {
    match &inner(value).value {
        ValueDef::Primitive(Primitive::U128(n)) => Some(*n),
        _ => None,
    }
}

pub(crate) fn boolean(value: &Value<u32>) -> Option<bool> {
    match &inner(value).value {
        ValueDef::Primitive(Primitive::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// A 32-byte account id, whatever wrappers it comes in.
pub(crate) fn account(value: &Value<u32>) -> Option<[u8; 32]> {
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

pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(incentives: Option<Vec<u128>>) -> Value<u32> {
        let account = |b: u128| Value::unnamed_composite([Value::unnamed_composite((0..32).map(|_| Value::u128(b)))]);
        let mut fields = vec![
            ("hotkeys".to_string(), Value::unnamed_composite([account(1), account(2)])),
            ("coldkeys".to_string(), Value::unnamed_composite([account(3), account(4)])),
        ];
        if let Some(incentives) = incentives {
            fields.push(("incentives".to_string(), Value::unnamed_composite(incentives.into_iter().map(Value::u128))));
        }
        Value::named_composite(fields).map_context(|_| 0u32)
    }

    #[test]
    fn an_endpoints_operator_is_its_domain_or_its_address() {
        assert_eq!(operator(&endpoint("finney")), "opentensor.ai");
        assert_eq!(operator("https://lite.chain.opentensor.ai:443"), "opentensor.ai");
        assert_eq!(operator("https://bittensor-finney.api.onfinality.io/public"), "onfinality.io");
        assert_eq!(operator("http://127.0.0.1:9944"), "127.0.0.1:9944");
        assert_ne!(operator("http://127.0.0.1:9944"), operator("http://127.0.0.1:9945"));
        assert_eq!(operator("http://localhost:9944"), "localhost:9944");
    }

    #[test]
    fn incentives_are_read_and_a_metagraph_without_any_says_nothing() {
        let read = neurons(&graph(Some(vec![65_535, 0]))).unwrap();
        assert_eq!(read.iter().map(|n| n.incentive).collect::<Vec<_>>(), [Some(1.0), Some(0.0)]);
        assert!(neurons(&graph(None)).unwrap().iter().all(|n| n.incentive.is_none()), "no column: unknown, not zero");
        assert!(neurons(&graph(Some(vec![0, 0]))).unwrap().iter().all(|n| n.incentive.is_none()), "no weights yet: unknown, not zero");
    }
}

//! Which version of the API validators' stake approves: their commitments on chain, weighed by stake, the subnet owner's left out.
//!
//! A validator commits `sn22tee:<compose hash>`, `sn22tee:<hash>,<hash>` or `sn22tee:<hash>@<signer>`. A version is approved
//! once the independent validators whose commitment names it, made at least the timelock ago, hold more than the threshold of
//! the independent validators' stake. See docs/tee.md.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use blake2::digest::consts::{U16, U32};
use blake2::{Blake2b, Digest};
use frame_metadata::v14::{StorageEntryType, StorageHasher};
use frame_metadata::v15::RuntimeMetadataV15;
use scale_value::{Value, ValueDef};

use crate::chain::{self, Chain, Neuron};

pub const PREFIX: &str = "sn22tee:";
const PPM: u128 = 1_000_000;

/// What one validator's commitment says: the versions it approves, and the signer it appoints for the first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub versions: Vec<String>,
    pub appoint: Option<String>,
}

fn is_hash(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A commitment in one of the three forms, None for anything else a hotkey may have committed.
pub fn parse(text: &str) -> Option<Commitment> {
    let body = text.trim().strip_prefix(PREFIX)?;
    let (versions, appoint) = match body.split_once('@') {
        Some((version, signer)) => {
            desearch::hotkey::public_of(signer)?;
            (vec![version.to_string()], Some(signer.to_string()))
        }
        None => (body.split(',').map(str::to_string).collect::<Vec<_>>(), None),
    };
    if versions.is_empty() || versions.len() > 2 || !versions.iter().all(|v| is_hash(v)) {
        return None;
    }
    Some(Commitment { versions, appoint })
}

/// The share of stake and the timelock a version needs.
#[derive(Clone, Copy, Debug)]
pub struct Rule {
    threshold_ppm: u128,
    pub timelock: i64,
}

impl Rule {
    pub fn new(threshold: f64, timelock: i64) -> Rule {
        Rule { threshold_ppm: (threshold * PPM as f64).round() as u128, timelock }
    }
}

/// What the chain says at one block about who approves what.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub block: i64,
    pub neurons: Vec<Neuron>,
    pub owner_coldkey: String,
    pub owner_hotkey: String,
    /// Each hotkey's latest commitment: the block it was made at and its text.
    pub commitments: HashMap<String, (i64, String)>,
}

impl Snapshot {
    /// Validators with a permit that the subnet owner does not hold.
    fn independent(&self) -> impl Iterator<Item = &Neuron> {
        self.neurons.iter().filter(|n| n.validator_permit && n.coldkey != self.owner_coldkey && n.hotkey != self.owner_hotkey)
    }

    pub fn total(&self) -> u128 {
        self.independent().map(|n| n.total_stake_rao).sum()
    }

    /// The independent stake whose commitment names `version` (appointing `signer`, when given) and is at least `age` blocks old.
    pub fn support(&self, version: &str, age: i64, signer: Option<&str>) -> u128 {
        self.independent()
            .filter(|n| {
                let Some((made, text)) = self.commitments.get(&n.hotkey) else { return false };
                let Some(commitment) = parse(text) else { return false };
                let named = match signer {
                    Some(signer) => commitment.appoint.as_deref() == Some(signer) && commitment.versions[0] == version,
                    None => commitment.versions.iter().any(|v| v == version),
                };
                named && self.block - made >= age
            })
            .map(|n| n.total_stake_rao)
            .sum()
    }

    fn passes(&self, rule: Rule, support: u128) -> bool {
        let total = self.total();
        total > 0 && support.saturating_mul(PPM) > rule.threshold_ppm.saturating_mul(total)
    }

    /// The version may take over: approved for at least the timelock.
    pub fn approved(&self, rule: Rule, version: &str) -> bool {
        self.passes(rule, self.support(version, rule.timelock, None))
    }

    /// This CVM of the version may take the log without a handover: appointed for at least the timelock.
    pub fn appointed(&self, rule: Rule, version: &str, signer: &str) -> bool {
        self.passes(rule, self.support(version, rule.timelock, Some(signer)))
    }

    /// The running version is still wanted: approved now, however recently, so adding a successor keeps it endorsed.
    pub fn endorsed(&self, rule: Rule, version: &str) -> bool {
        self.passes(rule, self.support(version, 0, None))
    }
}

fn twox64(data: &[u8]) -> [u8; 8] {
    twox_hash::XxHash64::oneshot(0, data).to_le_bytes()
}

fn twox128(data: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&twox_hash::XxHash64::oneshot(0, data).to_le_bytes());
    out[8..].copy_from_slice(&twox_hash::XxHash64::oneshot(1, data).to_le_bytes());
    out
}

fn hashed(hasher: &StorageHasher, data: &[u8]) -> Vec<u8> {
    match hasher {
        StorageHasher::Blake2_128 => Blake2b::<U16>::digest(data).to_vec(),
        StorageHasher::Blake2_256 => Blake2b::<U32>::digest(data).to_vec(),
        StorageHasher::Blake2_128Concat => [Blake2b::<U16>::digest(data).as_slice(), data].concat(),
        StorageHasher::Twox128 => twox128(data).to_vec(),
        StorageHasher::Twox256 => {
            let mut out = Vec::with_capacity(32);
            for seed in 0..4 {
                out.extend(twox_hash::XxHash64::oneshot(seed, data).to_le_bytes());
            }
            out
        }
        StorageHasher::Twox64Concat => [twox64(data).as_slice(), data].concat(),
        StorageHasher::Identity => data.to_vec(),
    }
}

/// The bytes a hasher leaves of its input at the end of a key, for the hashers that keep it.
fn kept_len(hasher: &StorageHasher) -> Option<usize> {
    match hasher {
        StorageHasher::Blake2_128Concat => Some(16),
        StorageHasher::Twox64Concat => Some(8),
        StorageHasher::Identity => Some(0),
        _ => None,
    }
}

/// A storage entry of the metadata: its key prefix, hashers and value type.
struct Entry {
    prefix: Vec<u8>,
    hashers: Vec<StorageHasher>,
    value: u32,
}

fn entry(metadata: &RuntimeMetadataV15, pallet: &str, name: &str) -> Result<Entry> {
    let storage = metadata.pallets.iter().find(|p| p.name == pallet).and_then(|p| p.storage.as_ref()).ok_or_else(|| anyhow!("no {pallet} storage"))?;
    let found = storage.entries.iter().find(|e| e.name == name).ok_or_else(|| anyhow!("no {pallet}.{name}"))?;
    let prefix = [twox128(storage.prefix.as_bytes()), twox128(name.as_bytes())].concat();
    Ok(match &found.ty {
        StorageEntryType::Plain(ty) => Entry { prefix, hashers: Vec::new(), value: ty.id },
        StorageEntryType::Map { hashers, value, .. } => Entry { prefix, hashers: hashers.clone(), value: value.id },
    })
}

fn decode(metadata: &RuntimeMetadataV15, ty: u32, bytes: &[u8]) -> Result<Value<u32>> {
    scale_value::scale::decode_as_type(&mut &bytes[..], ty, &metadata.types).map_err(|e| anyhow!("decoding storage: {e}"))
}

/// Every byte inside a value, in order: a `Raw` field's array, however wrapped.
fn bytes_in(value: &Value<u32>, out: &mut Vec<u8>) {
    match &value.value {
        ValueDef::Primitive(scale_value::Primitive::U128(n)) => out.push(*n as u8),
        ValueDef::Composite(composite) => composite.values().for_each(|v| bytes_in(v, out)),
        ValueDef::Variant(variant) => variant.values.values().for_each(|v| bytes_in(v, out)),
        _ => {}
    }
}

/// A commitment's block and its `Raw`/`BigRaw` fields joined, as bittensor reads them back.
pub fn registration(value: &Value<u32>) -> Option<(i64, String)> {
    let block = chain::number(chain::field(value, "block")?)? as i64;
    let mut data = Vec::new();
    raw_fields(chain::field(chain::field(value, "info")?, "fields")?, &mut data);
    Some((block, String::from_utf8(data).ok()?))
}

/// The bytes of every `Raw` and `BigRaw` field, in order, through the bounded vector's wrappers.
fn raw_fields(value: &Value<u32>, out: &mut Vec<u8>) {
    match &value.value {
        ValueDef::Variant(variant) if variant.name.starts_with("Raw") || variant.name == "BigRaw" => bytes_in(value, out),
        ValueDef::Composite(composite) => composite.values().for_each(|v| raw_fields(v, out)),
        _ => {}
    }
}

/// The chain's view of approvals for the subnet at a block the endpoints agree on.
pub async fn read(chain: &Chain, netuid: u16) -> Result<Snapshot> {
    let (block, at) = chain.head_hash().await?;
    let metadata = chain.metadata(Some(&at)).await?;
    let netuid_key = netuid.to_le_bytes();
    let account_at = |pallet: &str, name: &str| -> Result<(Vec<u8>, u32)> {
        let entry = entry(&metadata, pallet, name)?;
        let hasher = entry.hashers.first().context("a map without a hasher")?;
        Ok(([entry.prefix, hashed(hasher, &netuid_key)].concat(), entry.value))
    };
    let mut owners = Vec::new();
    for name in ["SubnetOwner", "SubnetOwnerHotkey"] {
        let (key, ty) = account_at("SubtensorModule", name)?;
        let owner = match chain.storage(&key, &at).await? {
            Some(raw) => chain::account(&decode(&metadata, ty, &raw)?).map(|key| desearch::hotkey::ss58(&key)).unwrap_or_default(),
            None => String::new(),
        };
        owners.push(owner);
    }
    let commitments_of = entry(&metadata, "Commitments", "CommitmentOf")?;
    let [first, second] = commitments_of.hashers.as_slice() else { bail!("Commitments.CommitmentOf is not a double map") };
    let prefix = [commitments_of.prefix.clone(), hashed(first, &netuid_key)].concat();
    let kept = kept_len(second).context("the commitment's account cannot be read back from its key")?;
    let mut commitments = HashMap::new();
    let keys = chain.keys(&prefix, &at).await?;
    let values = chain.storage_many(&keys, &at).await?;
    for (key, raw) in &values {
        let Some(account) = key.get(prefix.len() + kept..).and_then(|rest| <[u8; 32]>::try_from(rest).ok()) else { continue };
        if let Some(found) = registration(&decode(&metadata, commitments_of.value, raw)?) {
            commitments.insert(desearch::hotkey::ss58(&account), found);
        }
    }
    let neurons = chain.metagraph_at(netuid, Some(&at)).await?;
    let [owner_coldkey, owner_hotkey]: [String; 2] = owners.try_into().map_err(|_| anyhow!("two owners"))?;
    Ok(Snapshot { block, neurons, owner_coldkey, owner_hotkey, commitments })
}

#[cfg(test)]
mod tests {
    use super::*;

    const H1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const H2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn neuron(uid: i64, coldkey: &str, stake: u128) -> Neuron {
        let hotkey = desearch::hotkey::Hotkey::from_uri(&format!("//gov{uid}")).unwrap().ss58();
        Neuron { uid, hotkey, coldkey: coldkey.into(), validator_permit: true, total_stake_rao: stake, alpha_stake_rao: 0, incentive: None }
    }

    fn snapshot(at: i64, commitments: &[(usize, i64, String)]) -> Snapshot {
        let neurons = vec![
            neuron(0, "owner", 1_000),
            neuron(1, "a", 30),
            neuron(2, "b", 30),
            neuron(3, "c", 40),
            Neuron { validator_permit: false, ..neuron(4, "d", 500) },
        ];
        let commitments = commitments.iter().map(|(uid, block, text)| (neurons[*uid].hotkey.clone(), (*block, text.clone()))).collect();
        Snapshot { block: at, neurons, owner_coldkey: "owner".into(), owner_hotkey: String::new(), commitments }
    }

    #[test]
    fn a_commitment_is_one_of_three_forms() {
        let signer = desearch::hotkey::Hotkey::from_uri("//Alice").unwrap().ss58();
        assert_eq!(parse(&format!("sn22tee:{H1}")), Some(Commitment { versions: vec![H1.into()], appoint: None }));
        assert_eq!(parse(&format!("sn22tee:{H1},{H2}")).unwrap().versions, [H1, H2]);
        assert_eq!(parse(&format!("sn22tee:{H1}@{signer}")).unwrap().appoint.as_deref(), Some(signer.as_str()));
        for other in
            [format!("sn22tee:{}", &H1[1..]), format!("sn22tee:{H1},{H2},{H1}"), format!("sn22tee:{H1}@nobody"), format!("other:{H1}"), H1.to_uppercase()]
        {
            assert_eq!(parse(&other), None, "{other}");
        }
    }

    #[test]
    fn approval_needs_more_than_half_the_independent_stake_for_the_timelock() {
        let rule = Rule::new(0.5, 100);
        let one = format!("sn22tee:{H1}");
        // The owner's 1000 never counts: 30 + 30 of 100 independent is a majority.
        let s = snapshot(1_000, &[(0, 0, one.clone()), (1, 850, one.clone()), (2, 900, one.clone())]);
        assert_eq!((s.total(), s.support(H1, 100, None)), (100, 60));
        assert!(s.approved(rule, H1));
        let young = snapshot(1_000, &[(1, 850, one.clone()), (2, 901, one.clone())]);
        assert!(!young.approved(rule, H1), "one commitment is not old enough yet");
        assert!(young.endorsed(rule, H1), "but the version is endorsed without the timelock");
        let half = snapshot(1_000, &[(1, 0, one.clone()), (3, 0, format!("sn22tee:{H2}")), (2, 0, format!("sn22tee:{H2}"))]);
        assert!(!half.approved(rule, H1) && half.approved(rule, H2), "40 + 30 of 100 beats 30");
        let owner_only = snapshot(1_000, &[(0, 0, one.clone())]);
        assert!(!owner_only.approved(rule, H1) && !owner_only.endorsed(rule, H1), "the owner's stake alone approves nothing");
        let exactly = snapshot(1_000, &[(1, 0, one.clone()), (2, 0, format!("sn22tee:{H1},{H2}")), (3, 0, format!("sn22tee:{H2}"))]);
        assert!(exactly.approved(rule, H1), "60 > 50");
        assert!(!Snapshot { neurons: vec![], ..exactly.clone() }.approved(rule, H1), "no stake approves nothing");
    }

    #[test]
    fn an_appointment_names_the_signer_too() {
        let rule = Rule::new(0.5, 10);
        let alice = desearch::hotkey::Hotkey::from_uri("//Alice").unwrap().ss58();
        let bob = desearch::hotkey::Hotkey::from_uri("//Bob").unwrap().ss58();
        let s = snapshot(100, &[(1, 0, format!("sn22tee:{H1}@{alice}")), (3, 0, format!("sn22tee:{H1}@{alice}")), (2, 0, format!("sn22tee:{H1}@{bob}"))]);
        assert!(s.appointed(rule, H1, &alice));
        assert!(!s.appointed(rule, H1, &bob));
        assert!(s.approved(rule, H1), "an appointment approves its version too");
    }

    /// Reads subnet 22 on finney, read-only: `cargo test --lib governance -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn reads_the_live_chain() {
        let options = crate::settings::ChainOptions { quorum: 1, finalized: true, count_on: false, refresh_s: 6.0 };
        let chain = std::sync::Arc::new(Chain::with("finney", &options).unwrap());
        chain.current_block().await.unwrap();
        let netuid = std::env::var("NETUID").ok().and_then(|n| n.parse().ok()).unwrap_or(22);
        let s = read(&chain, netuid).await.unwrap();
        let permits = s.neurons.iter().filter(|n| n.validator_permit).count();
        eprintln!(
            "block {} owner {} / {}: {} neurons, {permits} permits, independent stake {}, {} commitments",
            s.block,
            s.owner_coldkey,
            s.owner_hotkey,
            s.neurons.len(),
            s.total(),
            s.commitments.len()
        );
        let mut kept: Vec<_> = s.commitments.iter().collect();
        kept.sort_by_key(|(_, (block, _))| std::cmp::Reverse(*block));
        for (hotkey, (block, text)) in kept.iter().take(5) {
            eprintln!("  {hotkey} at {block}: {:?}", text.chars().take(80).collect::<String>());
        }
        assert!(!s.owner_coldkey.is_empty() && !s.neurons.is_empty());
    }

    #[test]
    fn storage_keys_hash_as_substrate_does() {
        // twox128("System") ++ twox128("Number"), the best known storage key.
        let key = [twox128(b"System"), twox128(b"Number")].concat();
        assert_eq!(desearch::canonical::hex(&key), "26aa394eea5630e07c48ae0c9558cef702a5c1b19ab7a04f536c519aca4983ac");
        assert_eq!(hashed(&StorageHasher::Identity, &22u16.to_le_bytes()), [22, 0]);
        assert_eq!(hashed(&StorageHasher::Twox64Concat, b"ab").len(), 10);
    }
}

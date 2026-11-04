//! The log's head committed to the chain, so the API cannot rewrite, roll back or fork what it published without the chain showing it.
//!
//! The API's key commits `sn22j1:<seq>` and the hash of entry `seq` with `Commitments.set_commitment` on the subnet: free of fees, held
//! in chain state where any node answers `CommitmentOf(netuid, hotkey)`, and limited to about 31 commitments per hotkey per epoch
//! (3,100 bytes, 100 at least per call). The key must be a hotkey registered on the subnet. Entries are anchored every
//! `TASK_API_ANCHOR_EVERY_BLOCKS`, and sooner once a round opens, so its commitment is on chain before its seed block when the budget allows.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use desearch::canonical::hex;
use desearch::hotkey::Hotkey;
use frame_metadata::v14::{StorageEntryType, StorageHasher};
use frame_metadata::v15::RuntimeMetadataV15;
use parity_scale_codec::{Compact, Encode};
use rusqlite::{params, Connection, OptionalExtension};
use scale_info::form::PortableForm;
use scale_info::{TypeDef, TypeDefPrimitive};
use scale_value::{Composite, Primitive, Value, ValueDef};
use serde_json::json;

use crate::chain::{decode_hex, Chain};
use crate::journal;
use crate::state::State;

pub const TAG: &str = "sn22j1:";
/// Commitments a hotkey may make per epoch: 3,100 bytes, each call counted as 100 at least.
pub const PER_EPOCH: i64 = 31;
const EPOCH_BLOCKS: i64 = 361;
/// Blocks an anchor is waited for before it is sent again.
const INCLUSION_BLOCKS: i64 = 20;

/// An entry's hash as the chain holds it, and the block that committed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Anchored {
    pub seq: i64,
    pub head: String,
    pub block: i64,
}

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS anchors (
            seq       INTEGER PRIMARY KEY,
            head      TEXT NOT NULL,
            block     INTEGER,
            sent_at   INTEGER NOT NULL,
            extrinsic TEXT NOT NULL
        );
        ",
    )?;
    Ok(())
}

/// The commitment's fields: the entry number as `Raw`, the entry's hash as `Sha256`.
pub fn fields(seq: i64, head: &str) -> Result<Value<()>> {
    let tag = format!("{TAG}{seq}").into_bytes();
    let hash = decode_hex(head).filter(|h| h.len() == 32).ok_or_else(|| anyhow!("{head} is not a 32-byte hash"))?;
    Ok(Value::unnamed_composite([
        Value::unnamed_variant(format!("Raw{}", tag.len()), [Value::from_bytes(&tag)]),
        Value::unnamed_variant("Sha256", [Value::from_bytes(&hash)]),
    ]))
}

/// Every byte inside a value, through the wrappers it comes in.
fn bytes_of<T>(value: &Value<T>) -> Option<Vec<u8>> {
    fn walk<T>(value: &Value<T>, out: &mut Vec<u8>) -> Option<()> {
        match &value.value {
            ValueDef::Primitive(Primitive::U128(n)) => out.push(u8::try_from(*n).ok()?),
            ValueDef::Composite(composite) => {
                for item in composite.values() {
                    walk(item, out)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut out = Vec::new();
    walk(value, &mut out)?;
    Some(out)
}

fn field<'a, T>(value: &'a Value<T>, name: &str) -> Option<&'a Value<T>> {
    match &value.value {
        ValueDef::Composite(Composite::Named(fields)) => fields.iter().find(|(n, _)| n == name).map(|(_, v)| v),
        _ => None,
    }
}

/// The variants inside the commitment's field list, through its bounded-vec wrapper.
fn variants<T>(value: &Value<T>) -> Vec<&Value<T>> {
    match &value.value {
        ValueDef::Variant(_) => vec![value],
        ValueDef::Composite(composite) => composite.values().flat_map(variants).collect(),
        _ => Vec::new(),
    }
}

/// The anchor a decoded `CommitmentOf` registration holds, or None when it holds another commitment.
pub fn parse<T>(registration: &Value<T>) -> Option<Anchored> {
    let block = match &field(registration, "block")?.value {
        ValueDef::Primitive(Primitive::U128(n)) => *n as i64,
        _ => 0,
    };
    let (mut seq, mut head) = (None, None);
    for found in variants(field(registration, "info")?) {
        let ValueDef::Variant(variant) = &found.value else { continue };
        let Some(bytes) = variant.values.values().map(bytes_of).collect::<Option<Vec<_>>>().map(|parts| parts.concat()) else { continue };
        if variant.name.starts_with("Raw") {
            seq = String::from_utf8(bytes).ok().and_then(|text| text.strip_prefix(TAG).and_then(|n| n.parse::<i64>().ok())).or(seq);
        } else if variant.name == "Sha256" && bytes.len() == 32 {
            head = Some(hex(&bytes));
        }
    }
    Some(Anchored { seq: seq?, head: head?, block })
}

pub fn twox(data: &[u8], words: u64) -> Vec<u8> {
    (0..words).flat_map(|seed| twox_hash::XxHash64::oneshot(seed, data).to_le_bytes()).collect()
}

pub fn blake2_256(data: &[u8]) -> [u8; 32] {
    Blake2b::<U32>::digest(data).into()
}

fn blake2_128(data: &[u8]) -> Vec<u8> {
    Blake2b::<blake2::digest::consts::U16>::digest(data).to_vec()
}

/// A value of the type that encodes as its zero: no tip, an immortal era, no metadata hash, the first variant of any enum.
fn zero(types: &scale_info::PortableRegistry, id: u32) -> Result<Value<()>> {
    let ty = types.resolve(id).ok_or_else(|| anyhow!("type {id} is not in the metadata"))?;
    Ok(match &ty.type_def {
        TypeDef::Composite(composite) => {
            if composite.fields.iter().all(|f| f.name.is_some()) && !composite.fields.is_empty() {
                Value::named_composite(
                    composite.fields.iter().map(|f| Ok((f.name.clone().unwrap_or_default(), zero(types, f.ty.id)?))).collect::<Result<Vec<_>>>()?,
                )
            } else {
                Value::unnamed_composite(composite.fields.iter().map(|f| zero(types, f.ty.id)).collect::<Result<Vec<_>>>()?)
            }
        }
        TypeDef::Variant(variant) => {
            let first = variant.variants.iter().min_by_key(|v| v.index).ok_or_else(|| anyhow!("type {id} has no variants"))?;
            if first.fields.iter().all(|f| f.name.is_some()) && !first.fields.is_empty() {
                Value::named_variant(
                    first.name.clone(),
                    first.fields.iter().map(|f| Ok((f.name.clone().unwrap_or_default(), zero(types, f.ty.id)?))).collect::<Result<Vec<_>>>()?,
                )
            } else {
                Value::unnamed_variant(first.name.clone(), first.fields.iter().map(|f| zero(types, f.ty.id)).collect::<Result<Vec<_>>>()?)
            }
        }
        TypeDef::Sequence(_) => Value::unnamed_composite([]),
        TypeDef::Array(array) => Value::unnamed_composite((0..array.len).map(|_| zero(types, array.type_param.id)).collect::<Result<Vec<_>>>()?),
        TypeDef::Tuple(tuple) => Value::unnamed_composite(tuple.fields.iter().map(|f| zero(types, f.id)).collect::<Result<Vec<_>>>()?),
        TypeDef::Primitive(TypeDefPrimitive::Bool) => Value::bool(false),
        TypeDef::Primitive(TypeDefPrimitive::Str) => Value::string(""),
        TypeDef::Primitive(TypeDefPrimitive::Char) => Value::char('\0'),
        TypeDef::Primitive(TypeDefPrimitive::I8 | TypeDefPrimitive::I16 | TypeDefPrimitive::I32 | TypeDefPrimitive::I64 | TypeDefPrimitive::I128) => {
            Value::i128(0)
        }
        TypeDef::Primitive(_) => Value::u128(0),
        TypeDef::Compact(compact) => zero(types, compact.type_param.id)?,
        TypeDef::BitSequence(_) => bail!("type {id} is a bit sequence"),
    })
}

fn encode(types: &scale_info::PortableRegistry, id: u32, value: &Value<()>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    scale_value::scale::encode_as_type(value, id, types, &mut out).map_err(|e| anyhow!("encoding as type {id}: {e}"))?;
    Ok(out)
}

/// Signs and sends calls as the runtime's own metadata describes them, with every transaction extension at its zero but the ones
/// that carry the nonce, the runtime's versions and the genesis hash.
pub struct Extrinsics {
    pub chain: Arc<Chain>,
    pub metadata: RuntimeMetadataV15,
    genesis: Vec<u8>,
    spec_version: u32,
    transaction_version: u32,
}

impl Extrinsics {
    pub async fn load(chain: Arc<Chain>) -> Result<Extrinsics> {
        let metadata = chain.metadata(None).await?;
        let genesis = chain.rpc("chain_getBlockHash", json!([0])).await?;
        let genesis = genesis.as_str().and_then(|h| decode_hex(h.trim_start_matches("0x"))).ok_or_else(|| anyhow!("no genesis hash"))?;
        let version = chain.rpc("state_getRuntimeVersion", json!([])).await?;
        let number = |name: &str| version[name].as_u64().map(|n| n as u32).ok_or_else(|| anyhow!("the runtime version has no {name}"));
        Ok(Extrinsics { spec_version: number("specVersion")?, transaction_version: number("transactionVersion")?, chain, metadata, genesis })
    }

    pub fn types(&self) -> &scale_info::PortableRegistry {
        &self.metadata.types
    }

    /// A call to `pallet` as the runtime encodes it.
    pub fn call(&self, pallet: &str, call: &str, args: Vec<(&str, Value<()>)>) -> Result<Vec<u8>> {
        let value = Value::unnamed_variant(pallet, [Value::named_variant(call, args.into_iter().map(|(name, value)| (name.to_string(), value)))]);
        encode(self.types(), self.metadata.extrinsic.call_ty.id, &value)
    }

    /// The signed extrinsic, as `author_submitExtrinsic` takes it.
    pub fn signed(&self, call: &[u8], key: &Hotkey, nonce: u64) -> Result<Vec<u8>> {
        let (mut extra, mut implied) = (Vec::new(), Vec::new());
        for extension in &self.metadata.extrinsic.signed_extensions {
            let types = self.types();
            let extra_value = match extension.identifier.as_str() {
                "CheckNonce" => Value::u128(nonce.into()),
                _ => zero(types, extension.ty.id)?,
            };
            let implied_value = match extension.identifier.as_str() {
                "CheckSpecVersion" => Value::u128(self.spec_version.into()),
                "CheckTxVersion" => Value::u128(self.transaction_version.into()),
                // An immortal transaction names the genesis block.
                "CheckGenesis" | "CheckMortality" | "CheckEra" => Value::from_bytes(&self.genesis),
                _ => zero(types, extension.additional_signed.id)?,
            };
            extra.extend(encode(types, extension.ty.id, &extra_value).with_context(|| format!("extension {}", extension.identifier))?);
            implied.extend(encode(types, extension.additional_signed.id, &implied_value).with_context(|| format!("extension {}", extension.identifier))?);
        }
        let mut payload = call.to_vec();
        payload.extend(&extra);
        payload.extend(&implied);
        let signature = if payload.len() > 256 { key.sign(&blake2_256(&payload)) } else { key.sign(&payload) };
        let address = Value::unnamed_variant("Id", [Value::from_bytes(key.public())]);
        let signature = Value::unnamed_variant("Sr25519", [Value::from_bytes(signature)]);
        let mut body = vec![0x84];
        body.extend(encode(self.types(), self.metadata.extrinsic.address_ty.id, &address)?);
        body.extend(encode(self.types(), self.metadata.extrinsic.signature_ty.id, &signature)?);
        body.extend(extra);
        body.extend(call);
        let mut extrinsic = Compact(body.len() as u32).encode();
        extrinsic.extend(body);
        Ok(extrinsic)
    }

    pub async fn nonce(&self, key: &Hotkey) -> Result<u64> {
        let found = self.chain.rpc("system_accountNextIndex", json!([key.ss58()])).await?;
        found.as_u64().ok_or_else(|| anyhow!("no nonce for {}", key.ss58()))
    }

    /// Sends a signed extrinsic; returns its hash.
    pub async fn submit(&self, extrinsic: &[u8]) -> Result<String> {
        let found = self.chain.rpc("author_submitExtrinsic", json!([format!("0x{}", hex(extrinsic))])).await?;
        Ok(found.as_str().unwrap_or_default().to_string())
    }

    /// What the runtime makes of a signed extrinsic without taking it: `Ok` with its validity, or the reason it would refuse it.
    pub async fn validate(&self, extrinsic: &[u8], at: &str) -> Result<Vec<u8>> {
        let mut input = vec![2u8]; // TransactionSource::External
        input.extend(extrinsic);
        input.extend(decode_hex(at.trim_start_matches("0x")).ok_or_else(|| anyhow!("{at} is not a block hash"))?);
        self.chain.state_call("TaggedTransactionQueue_validate_transaction", &input).await
    }

    fn storage_entry(&self, pallet: &str, entry: &str) -> Result<&frame_metadata::v14::StorageEntryMetadata<PortableForm>> {
        self.metadata
            .pallets
            .iter()
            .find(|p| p.name == pallet)
            .and_then(|p| p.storage.as_ref())
            .and_then(|s| s.entries.iter().find(|e| e.name == entry))
            .ok_or_else(|| anyhow!("the runtime has no storage {pallet}.{entry}"))
    }

    /// The key of a map's entry: the pallet and entry names hashed, then each key encoded and hashed as the metadata says.
    pub fn storage_key(&self, pallet: &str, entry: &str, keys: &[Value<()>]) -> Result<Vec<u8>> {
        let StorageEntryType::Map { hashers, key, .. } = &self.storage_entry(pallet, entry)?.ty else { bail!("{pallet}.{entry} is not a map") };
        let key_types: Vec<u32> = match &self.types().resolve(key.id).ok_or_else(|| anyhow!("no key type"))?.type_def {
            TypeDef::Tuple(tuple) if hashers.len() > 1 => tuple.fields.iter().map(|f| f.id).collect(),
            _ => vec![key.id],
        };
        let mut out = twox(pallet.as_bytes(), 2);
        out.extend(twox(entry.as_bytes(), 2));
        for ((value, ty), hasher) in keys.iter().zip(key_types).zip(hashers) {
            let encoded = encode(self.types(), ty, value)?;
            out.extend(match hasher {
                StorageHasher::Identity => encoded,
                StorageHasher::Twox64Concat => [twox(&encoded, 1), encoded].concat(),
                StorageHasher::Blake2_128Concat => [blake2_128(&encoded), encoded].concat(),
                other => bail!("{pallet}.{entry} hashes its key with {other:?}"),
            });
        }
        Ok(out)
    }

    /// A map entry's value, decoded with the runtime's types, as of block `at` or the best block.
    pub async fn storage(&self, pallet: &str, entry: &str, keys: &[Value<()>], at: Option<&str>) -> Result<Option<Value<u32>>> {
        let key = format!("0x{}", hex(&self.storage_key(pallet, entry, keys)?));
        let found = self.chain.rpc("state_getStorage", if let Some(at) = at { json!([key, at]) } else { json!([key]) }).await?;
        let Some(bytes) = found.as_str().and_then(|h| decode_hex(h.trim_start_matches("0x"))) else { return Ok(None) };
        let StorageEntryType::Map { value, .. } = &self.storage_entry(pallet, entry)?.ty else { bail!("{pallet}.{entry} is not a map") };
        let decoded =
            scale_value::scale::decode_as_type(&mut bytes.as_slice(), value.id, self.types()).map_err(|e| anyhow!("decoding {pallet}.{entry}: {e}"))?;
        Ok(Some(decoded))
    }

    /// The `set_commitment` naming entry `seq` and its hash, signed by `key`.
    pub async fn commitment(&self, key: &Hotkey, netuid: u16, seq: i64, head: &str) -> Result<Vec<u8>> {
        let info = Value::named_composite([("fields", fields(seq, head)?)]);
        let call = self.call("Commitments", "set_commitment", vec![("netuid", Value::u128(netuid.into())), ("info", info)])?;
        self.signed(&call, key, self.nonce(key).await?)
    }

    /// What the chain holds as `account`'s anchor on the subnet, as of block `at` or the best block.
    pub async fn anchored(&self, netuid: u16, account: &str, at: Option<&str>) -> Result<Option<Anchored>> {
        let public = desearch::hotkey::public_of(account).ok_or_else(|| anyhow!("{account} is not an ss58 address"))?;
        let found = self.storage("Commitments", "CommitmentOf", &[Value::u128(netuid.into()), Value::from_bytes(public)], at).await?;
        Ok(found.as_ref().and_then(parse))
    }
}

/// The anchoring loop's memory between passes.
#[derive(Default)]
pub struct Anchoring {
    extrinsics: Option<Extrinsics>,
    key: Option<Hotkey>,
}

/// One pass: confirms an anchor sent before, and sends the head when it is due. Runs only on the copy holding the janitor lease.
pub async fn pass(state: &State, anchoring: &mut Anchoring) -> Result<()> {
    let Some(settings) = state.settings.anchor.clone() else { return Ok(()) };
    if anchoring.extrinsics.is_none() {
        let chain = Arc::new(Chain::with(&settings.network, &state.settings.chain)?);
        anchoring.extrinsics = Some(Extrinsics::load(chain).await?);
        anchoring.key = Some(Hotkey::from_uri(if settings.key_uri.is_empty() { &state.settings.key_uri } else { &settings.key_uri })?);
    }
    let (Some(extrinsics), Some(key)) = (&anchoring.extrinsics, &anchoring.key) else { return Ok(()) };
    let block = extrinsics.chain.current_block().await?;
    let account = key.ss58();
    if let Some(on_chain) = extrinsics.anchored(settings.netuid, &account, None).await? {
        let confirmed = on_chain.clone();
        state
            .db
            .run(move |conn| {
                let pending: Option<Option<i64>> = conn
                    .query_row("SELECT block FROM anchors WHERE seq = ? AND head = ?", params![confirmed.seq, confirmed.head], |row| row.get(0))
                    .optional()?;
                if pending == Some(None) {
                    conn.execute("UPDATE anchors SET block = ? WHERE seq = ?", params![confirmed.block, confirmed.seq])?;
                    let input = json!({"netuid": settings.netuid, "hotkey": account, "seq": confirmed.seq, "head": confirmed.head, "block": confirmed.block});
                    journal::append(conn, journal::Record::new("anchor", desearch::time::now(), input, serde_json::Value::Null))?;
                }
                Ok(())
            })
            .await?;
    }
    let due = state
        .db
        .run(move |conn| {
            let (seq, head) = journal::head(conn)?;
            let last: Option<(i64, Option<i64>, i64)> = conn
                .query_row("SELECT seq, block, sent_at FROM anchors ORDER BY seq DESC LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .optional()?;
            let sent: i64 = conn.query_row("SELECT COUNT(*) FROM anchors WHERE sent_at > ?", [block - EPOCH_BLOCKS], |row| row.get(0))?;
            let (last_seq, last_block, last_sent) = last.unwrap_or((0, Some(0), 0));
            // A round's commitment should reach the chain before its seed block, so a round opening is anchored soon.
            let opened: bool = conn
                .query_row("SELECT 1 FROM journal WHERE seq > ? AND kind IN ('enqueue', 'embed_round') LIMIT 1", [last_seq], |_| Ok(()))
                .optional()?
                .is_some();
            let waiting = last_block.is_none() && block < last_sent + INCLUSION_BLOCKS;
            let wait = if opened { settings.min_blocks } else { settings.every_blocks };
            let due = seq > last_seq && !waiting && block >= last_sent + wait && sent < settings.per_epoch;
            Ok(due.then_some((seq, head)))
        })
        .await?;
    let Some((seq, head)) = due else { return Ok(()) };
    let extrinsic = extrinsics.commitment(key, settings.netuid, seq, &head).await?;
    let sent = extrinsics.submit(&extrinsic).await?;
    state
        .db
        .run(move |conn| {
            conn.execute("INSERT OR REPLACE INTO anchors (seq, head, block, sent_at, extrinsic) VALUES (?, ?, NULL, ?, ?)", params![seq, head, block, sent])?;
            Ok(())
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_prefixes_hash_as_substrate_does() {
        assert_eq!(hex(&twox(b"System", 2)), "26aa394eea5630e07c48ae0c9558cef7");
        assert_eq!(hex(&twox(b"Account", 2)), "b99d880ec681799c0cf30e8886371da9");
    }

    #[test]
    fn an_anchor_reads_back_from_the_registration_the_chain_keeps() {
        let head = "ab".repeat(32);
        let registration = Value::named_composite([
            ("deposit", Value::u128(0)),
            ("block", Value::u128(4_321)),
            ("info", Value::named_composite([("fields", Value::unnamed_composite([fields(1234, &head).unwrap()]))])),
        ]);
        assert_eq!(parse(&registration), Some(Anchored { seq: 1234, head: head.clone(), block: 4_321 }));
        let other = Value::named_composite([
            ("block", Value::u128(1)),
            ("info", Value::named_composite([("fields", Value::unnamed_composite([Value::unnamed_variant("Raw5", [Value::from_bytes(b"hello")])]))])),
        ]);
        assert_eq!(parse(&other), None, "a commitment that is not an anchor");
        assert!(fields(1, "abc").is_err());
    }
}

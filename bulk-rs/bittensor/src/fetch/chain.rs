//! `chain_snapshot` and `registration_dates`: every subnet at one finney block, then each registration block's time.

use anyhow::{anyhow, bail, Result};
use bulk_core::py::truthy;
use bulk_core::Counts;
use desearch::chain::{plain, Chain, Decoded, InherentError, Refusal};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub const NETWORK: &str = "finney";
pub const DATE_ATTEMPTS: u32 = 6;
const PALLET: &str = "SubtensorModule";

/// The chain read gave fewer subnets than it counts: `python -m bulk` exits 2 on this.
#[derive(Debug)]
pub struct ChainError(pub String);

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ChainError {}

/// `datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()`.
pub fn iso_from_ms(ms: u128) -> Result<String> {
    let t = ratio(ms, 1000);
    let whole = t.trunc();
    let mut micros = ((t - whole) * 1e6).round_ties_even();
    let mut seconds = whole;
    if micros >= 1e6 {
        micros -= 1e6;
        seconds += 1.0;
    }
    if !(-62_135_596_800.0..253_402_300_800.0).contains(&seconds) {
        bail!("year is out of range");
    }
    let [year, month, day, hour, minute, second] = desearch::time::utc(seconds as i64);
    let fraction = if micros > 0.0 { format!(".{:06}", micros as u32) } else { String::new() };
    Ok(format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{fraction}+00:00"))
}

/// `a / b` of Python ints, correctly rounded.
pub fn ratio(a: u128, b: u128) -> f64 {
    const EXACT: u128 = 1 << 53;
    if a < EXACT && b < EXACT {
        return a as f64 / b as f64;
    }
    let bits = |n: u128| 128 - n.leading_zeros() as i32;
    let shift = 55 - (bits(a) - bits(b));
    let (num, den) = if shift >= 0 {
        match a.checked_shl(shift as u32).filter(|n| n >> shift == a) {
            Some(n) => (n, b),
            None => return a as f64 / b as f64,
        }
    } else {
        (a, b.checked_shl((-shift) as u32).unwrap_or(u128::MAX))
    };
    let (q, r) = (num / den, num % den);
    let q = q | u128::from(r != 0);
    (q as f64) * 2f64.powi(-shift)
}

fn int_of(value: &Value, what: &str) -> Result<u128> {
    match value {
        Value::Number(n) => n.as_u64().map(u128::from).ok_or_else(|| anyhow!("{what} {n} is not an unsigned integer")),
        Value::Bool(b) => Ok(u128::from(*b)),
        other => bail!("{what} is {other}, not an integer"),
    }
}

/// `text_of`: bytes as lists come out UTF-8 decoded with replacement.
fn text_of(value: Option<&Value>) -> Result<Value> {
    match value {
        Some(Value::Array(items)) => {
            let bytes: Vec<u8> =
                items.iter().map(|b| int_of(b, "a name byte").and_then(|n| u8::try_from(n).map_err(|e| anyhow!("{e}")))).collect::<Result<_>>()?;
            Ok(json!(String::from_utf8_lossy(&bytes)))
        }
        Some(v) if truthy(Some(v)) => Ok(v.clone()),
        _ => Ok(json!("")),
    }
}

/// `owner_identity`: the owner's chain identity under the names the subnet pages read.
pub fn owner_identity(found: Option<&Value>) -> Value {
    let Some(Value::Object(found)) = found.filter(|f| truthy(Some(f))) else {
        return Value::Null;
    };
    let get = |key: &str| found.get(key).cloned().unwrap_or_else(|| json!(""));
    let github = found.get("github_repo").cloned().unwrap_or_else(|| get("github"));
    json!({
        "name": get("name"),
        "url": get("url"),
        "github": github,
        "image": get("image"),
        "discord": get("discord"),
        "description": get("description"),
        "additional": get("additional"),
    })
}

/// `chain_snapshot`: every subnet at the block (the best one by default), failing unless all subnets read.
pub async fn snapshot(chain: &Chain, block: Option<i64>) -> Result<Map<String, Value>> {
    let block = match block {
        Some(block) => block,
        None => chain.read_best().await?,
    };
    let hash = chain.block_hash(block).await?.ok_or_else(|| anyhow!("block {block} is not made yet"))?;
    let at = Some(hash.as_str());
    let metadata = chain.metadata(at).await?;
    let types = &metadata.types;
    let one = |value: Option<Decoded>| value.map_or(Value::Null, |v| plain(&v, types));
    let taken = one(chain.storage(&metadata, "Timestamp", "Now", at).await?);
    let total = one(chain.storage(&metadata, PALLET, "TotalNetworks", at).await?);
    let dynamic = plain(&chain.runtime_call(&metadata, "SubnetInfoRuntimeApi", "get_all_dynamic_info", &[], at).await?, types);
    let dynamic: Vec<Value> = dynamic.as_array().cloned().unwrap_or_default().into_iter().filter(|d| truthy(Some(d))).collect();
    let map =
        |pairs: Vec<(Decoded, Decoded)>| -> HashMap<String, Value> { pairs.into_iter().map(|(k, v)| (key_of(&plain(&k, types)), plain(&v, types))).collect() };
    let identities = map(chain.storage_map(&metadata, PALLET, "SubnetIdentitiesV3", at).await?);
    let neurons = map(chain.storage_map(&metadata, PALLET, "SubnetworkN", at).await?);
    let max_neurons = map(chain.storage_map(&metadata, PALLET, "MaxAllowedUids", at).await?);
    let owners = map(chain.storage_map(&metadata, PALLET, "IdentitiesV2", at).await?);
    let total_n = int_of(&total, "TotalNetworks")?;
    if total_n == 0 || (dynamic.len() as u128) < total_n {
        return Err(ChainError(format!("read {} subnets of the chain's {}", dynamic.len(), total)).into());
    }
    let field = |d: &Value, key: &str| d.get(key).cloned().ok_or_else(|| anyhow!("KeyError: {key:?}"));
    let mut emitted: u128 = 0;
    for d in &dynamic {
        emitted += int_of(&field(d, "tao_in_emission")?, "tao_in_emission")?;
    }
    let mut ordered: Vec<(u128, &Value)> = Vec::with_capacity(dynamic.len());
    for d in &dynamic {
        ordered.push((int_of(&field(d, "netuid")?, "netuid")?, d));
    }
    ordered.sort_by_key(|(netuid, _)| *netuid);
    let mut subnets = Vec::with_capacity(ordered.len());
    for (netuid, d) in ordered {
        let tao_in = int_of(&field(d, "tao_in_emission")?, "tao_in_emission")?;
        let coldkey = field(d, "owner_coldkey")?;
        let key = netuid.to_string();
        subnets.push(json!({
            "netuid": netuid as u64,
            "name": text_of(d.get("subnet_name"))?,
            "symbol": text_of(d.get("token_symbol"))?,
            "owner_hotkey": field(d, "owner_hotkey")?,
            "owner_coldkey": coldkey,
            "owner_identity": owner_identity(owners.get(&key_of(&coldkey))),
            "identity": identities.get(&key).cloned().unwrap_or(Value::Null),
            "tempo": int_of(&field(d, "tempo")?, "tempo")? as u64,
            "registered_block": int_of(&field(d, "network_registered_at")?, "network_registered_at")? as u64,
            "registered_at": null,
            "neurons": neurons.get(&key).cloned().unwrap_or(Value::Null),
            "max_neurons": max_neurons.get(&key).cloned().unwrap_or(Value::Null),
            "tao_in_emission": ratio(tao_in, 1) / 1e9,
            "emission_share": if emitted > 0 { json!(ratio(tao_in, emitted)) } else { Value::Null },
        }));
    }
    let taken_ms = int_of(&taken, "Timestamp.Now")?;
    let mut found = Map::new();
    found.insert("block".into(), json!(block));
    found.insert("taken_at".into(), json!(iso_from_ms(taken_ms)?));
    found.insert("network".into(), json!(NETWORK));
    found.insert("total_networks".into(), total);
    found.insert("subnets".into(), Value::Array(subnets));
    Ok(found)
}

fn key_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Why one registration date read failed, by the Python exception it stands for; `Fatal` ends the fetch as an uncaught one would.
enum DateError {
    Caught(&'static str),
    Fatal(anyhow::Error),
}

fn caught(error: anyhow::Error) -> DateError {
    if let Some(e) = error.downcast_ref::<reqwest::Error>() {
        return DateError::Caught(if e.is_timeout() {
            "TimeoutError"
        } else if e.is_status() {
            "ClientResponseError"
        } else if e.is_decode() {
            "ContentTypeError"
        } else if e.is_connect() {
            "ClientConnectorError"
        } else {
            "ClientOSError"
        });
    }
    match error.downcast_ref::<Refusal>() {
        Some(Refusal::Error { .. }) => DateError::Caught("ChainError"),
        Some(Refusal::NoResult { .. }) => DateError::Caught("KeyError"),
        None => DateError::Fatal(error),
    }
}

async fn block_time(chain: &Chain, number: u128) -> std::result::Result<String, DateError> {
    let hash = chain.request("chain_getBlockHash", json!([number as u64])).await.map_err(caught)?;
    let block = chain.request("chain_getBlock", json!([hash])).await.map_err(caught)?;
    let inner = match &block {
        Value::Object(map) => map.get("block").ok_or(DateError::Caught("KeyError"))?,
        _ => return Err(DateError::Caught("TypeError")),
    };
    let extrinsics = match inner {
        Value::Object(map) => map.get("extrinsics").ok_or(DateError::Caught("KeyError"))?,
        _ => return Err(DateError::Caught("TypeError")),
    };
    let first = match extrinsics {
        Value::Array(items) => items.first().ok_or_else(|| DateError::Fatal(anyhow!("IndexError: list index out of range")))?,
        _ => return Err(DateError::Caught("TypeError")),
    };
    let text = first.as_str().ok_or_else(|| DateError::Fatal(anyhow!("AttributeError: the first extrinsic is not a str")))?;
    let moment = match desearch::chain::inherent_moment(text) {
        Ok(moment) => moment,
        Err(InherentError::Invalid(_)) => return Err(DateError::Caught("ValueError")),
        Err(InherentError::Short) => return Err(DateError::Fatal(anyhow!("IndexError: index out of range"))),
    };
    iso_from_ms(moment).map_err(|_| DateError::Caught("ValueError"))
}

/// `registration_dates`: each registration block's time from the block itself, which pruned nodes keep after its state is gone.
pub async fn registration_dates(chain: &Chain, found: &mut Map<String, Value>, counts: &mut Counts, pace: super::Pace) -> Result<()> {
    let Some(Value::Array(subnets)) = found.get_mut("subnets") else { return Ok(()) };
    for subnet in subnets.iter_mut() {
        let registered = subnet.get("registered_block").cloned().unwrap_or(Value::Null);
        if !truthy(Some(&registered)) {
            continue;
        }
        let number = int_of(&registered, "registered_block")?;
        for attempt in 0..DATE_ATTEMPTS {
            match block_time(chain, number).await {
                Ok(moment) => {
                    subnet["registered_at"] = json!(moment);
                    counts.add("registration_dates", 1);
                    break;
                }
                Err(DateError::Fatal(error)) => return Err(error),
                Err(DateError::Caught(name)) => {
                    if attempt == DATE_ATTEMPTS - 1 {
                        counts.add("registration_date_failed", 1);
                        counts.add(&format!("registration_date_error {name}"), 1);
                    } else {
                        pace.sleep(f64::from(1u32 << attempt)).await;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_print_as_isoformat() {
        assert_eq!(iso_from_ms(1_703_549_172_001).unwrap(), "2023-12-26T00:06:12.001000+00:00");
        assert_eq!(iso_from_ms(1_762_850_700_000).unwrap(), "2025-11-11T08:45:00+00:00");
        assert_eq!(iso_from_ms(0).unwrap(), "1970-01-01T00:00:00+00:00");
    }

    #[test]
    fn int_division_rounds_as_python() {
        assert_eq!(ratio(1, 3), 1.0 / 3.0);
        assert_eq!(ratio(30, 30), 1.0);
        let big = (1u128 << 60) + 1;
        assert_eq!(ratio(big, 3), 384307168202282325.33333333);
        assert_eq!(ratio(2, big), 1.734723475976807e-18);
    }
}

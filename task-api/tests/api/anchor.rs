//! Anchoring against a real chain: only with `TASK_API_TEST_CHAIN` set (`test`, or a local node's URL). Nothing is ever sent to finney.
//!
//! On any chain the extrinsic is checked by the runtime's own `validate_transaction`, which rejects a wrong encoding or signature as
//! `BadProof` without taking it in. With `TASK_API_TEST_ANCHOR_KEY` (a hotkey registered on `TASK_API_TEST_NETUID`, holding a little
//! balance) on a local chain, the anchor is also sent and read back from chain state.

use std::sync::Arc;

use desearch::hotkey::Hotkey;
use serde_json::json;
use task_api::anchor::{parse, Extrinsics};
use task_api::chain::{decode_hex, Chain};

fn chain() -> Option<String> {
    let network = std::env::var("TASK_API_TEST_CHAIN").ok().filter(|n| !n.is_empty())?;
    assert!(!network.contains("finney") && !network.contains("entrypoint"), "anchor tests never run against mainnet");
    Some(network)
}

/// `Result<ValidTransaction, TransactionValidityError>`: the error's two leading bytes, or None when the runtime would take it.
fn refusal(validity: &[u8]) -> Option<(u8, u8)> {
    (validity.first() == Some(&1)).then(|| (validity[1], validity.get(2).copied().unwrap_or(0)))
}

#[tokio::test]
async fn the_runtime_takes_an_anchor_as_signed() {
    let Some(network) = chain() else { return };
    let chain = Arc::new(Chain::new(&network).unwrap());
    let extrinsics = Extrinsics::load(chain.clone()).await.unwrap();
    let key = Hotkey::from_uri("//sn22-anchor-test").unwrap();
    let head = "cd".repeat(32);
    let extrinsic = extrinsics.commitment(&key, 22, 42, &head).await.unwrap();
    let best = chain.rpc("chain_getBlockHash", json!([])).await.unwrap();
    let validity = extrinsics.validate(&extrinsic, best.as_str().unwrap()).await.unwrap();
    // 1 = Err, 0 = Invalid, 4 = BadProof: the signature does not cover what the runtime rebuilt from the extrinsic.
    assert_ne!(refusal(&validity), Some((0, 4)), "the runtime rebuilt another payload than the one signed: {validity:?}");
    assert_ne!(refusal(&validity), Some((0, 0)), "the runtime could not take the call: {validity:?}");

    let mut forged = extrinsic.clone();
    let at = forged.len() - 1;
    forged[at] ^= 1;
    let validity = extrinsics.validate(&forged, best.as_str().unwrap()).await.unwrap();
    assert_eq!(refusal(&validity), Some((0, 4)), "a byte changed after signing is refused as a bad proof, so the check above means something");

    // Reading anchors back: any commitment on the chain decodes through the same keys and types.
    let prefix = format!(
        "0x{}{}",
        desearch::canonical::hex(&task_api::anchor::twox(b"Commitments", 2)),
        desearch::canonical::hex(&task_api::anchor::twox(b"CommitmentOf", 2))
    );
    let keys = chain.rpc("state_getKeysPaged", json!([prefix, 5])).await.unwrap();
    for key in keys.as_array().unwrap() {
        let raw = decode_hex(key.as_str().unwrap().trim_start_matches("0x")).unwrap();
        let netuid = u16::from_le_bytes([raw[32], raw[33]]);
        let account: [u8; 32] = raw[raw.len() - 32..].try_into().unwrap();
        let found = extrinsics
            .storage("Commitments", "CommitmentOf", &[scale_value::Value::u128(netuid.into()), scale_value::Value::from_bytes(account)], None)
            .await
            .unwrap();
        assert!(found.is_some(), "the key built for netuid {netuid} is the chain's own");
        let _ = found.as_ref().and_then(parse);
    }

    let Ok(uri) = std::env::var("TASK_API_TEST_ANCHOR_KEY") else { return };
    let key = Hotkey::from_uri(&uri).unwrap();
    let netuid: u16 = std::env::var("TASK_API_TEST_NETUID").ok().and_then(|n| n.parse().ok()).unwrap_or(1);
    let head = desearch::canonical::hex(&task_api::anchor::blake2_256(desearch::time::now().to_string().as_bytes()));
    let extrinsic = extrinsics.commitment(&key, netuid, 7, &head).await.unwrap();
    extrinsics.submit(&extrinsic).await.unwrap();
    for _ in 0..120 {
        if let Some(found) = extrinsics.anchored(netuid, &key.ss58(), None).await.unwrap().filter(|a| a.head == head) {
            assert_eq!(found.seq, 7);
            assert!(found.block > 0);
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    panic!("the anchor never reached chain state");
}

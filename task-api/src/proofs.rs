//! The commitments anyone can check: a round's manifest hash, its serve order and the Merkle root of its log.

use desearch::canonical::hex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::py::canonical;

pub const ALGORITHM: &str = "desearch-serve-order-2";

pub fn sha256(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

pub fn manifest_hash(entries: &[Value], seed_block: i64) -> String {
    let mut ordered = entries.to_vec();
    ordered.sort_by(|a, b| a["batch_id"].as_str().cmp(&b["batch_id"].as_str()));
    sha256(&canonical(&json!({"algorithm": ALGORITHM, "seed_block": seed_block, "batches": ordered})))
}

pub fn position_key(seed: &str, batch_id: &str) -> String {
    sha256(format!("{seed}:{batch_id}").as_bytes())
}

pub fn serve_order(seed: &str, batch_ids: &[String]) -> Vec<String> {
    let mut keyed: Vec<(String, &String)> = batch_ids.iter().map(|id| (position_key(seed, id), id)).collect();
    keyed.sort();
    keyed.into_iter().map(|(_, id)| id.clone()).collect()
}

pub fn merkle_root(leaves: &[Vec<u8>]) -> String {
    if leaves.is_empty() {
        return sha256(b"");
    }
    let mut level: Vec<[u8; 32]> = leaves.iter().map(|leaf| Sha256::digest(leaf).into()).collect();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push(*level.last().expect("a non-empty level"));
        }
        level = level.chunks(2).map(|pair| Sha256::new().chain_update(pair[0]).chain_update(pair[1]).finalize().into()).collect();
    }
    hex(&level[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_match_python() {
        let entries = [json!({"batch_id": "b2", "url_count": 2, "urls_hash": "x"}), json!({"batch_id": "a1", "url_count": 1, "urls_hash": "y"})];
        assert_eq!(manifest_hash(&entries, 77), "a62ee5b5e899719bcd39a65767ca868493c0b087ff905714f160dabe4cb79493");
        assert_eq!(serve_order("seed", &["a".into(), "b".into(), "c".into()]), ["a", "c", "b"]);
        assert_eq!(merkle_root(&[]), sha256(b""));
        assert_eq!(merkle_root(&[b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]), "d31a37ef6ac14a2db1470c4316beb5592e6afd4465022339adafda76a18ffabe");
    }

    fn manifest(n: usize) -> Vec<Value> {
        (0..n).map(|i| json!({"batch_id": format!("b{i:04}"), "url_count": 10 + i, "urls_hash": format!("{i:064x}")})).collect()
    }

    #[test]
    fn the_manifest_hash_ignores_packing_order_and_notices_any_change() {
        let entries = manifest(50);
        let mut reversed = entries.clone();
        reversed.reverse();
        assert_eq!(manifest_hash(&entries, 100), manifest_hash(&reversed, 100));
        let mut tampered = entries.clone();
        tampered[0]["url_count"] = 999.into();
        assert_ne!(manifest_hash(&entries, 100), manifest_hash(&tampered, 100));
        assert_ne!(manifest_hash(&entries, 100), manifest_hash(&entries, 101), "else a block with a favourable hash could be shopped for");
    }

    #[test]
    fn the_serve_order_is_a_seeded_permutation() {
        let ids: Vec<String> = (0..200).map(|i| format!("b{i:04}")).collect();
        let order = serve_order("seed", &ids);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, ids);
        assert_ne!(order, ids);
        let reversed: Vec<String> = ids.iter().rev().cloned().collect();
        assert_eq!(serve_order("s", &ids), serve_order("s", &reversed));
        assert_ne!(serve_order("seed-a", &ids), serve_order("seed-b", &ids));
    }

    #[test]
    fn the_merkle_root_is_sensitive_to_every_leaf() {
        let leaves: Vec<Vec<u8>> = (0..9).map(|i| format!("line-{i}").into_bytes()).collect();
        let root = merkle_root(&leaves);
        for i in 0..leaves.len() {
            let mut altered = leaves.clone();
            altered[i] = b"tampered".to_vec();
            assert_ne!(merkle_root(&altered), root, "leaf {i} did not affect the root");
        }
    }
}

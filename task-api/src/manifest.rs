//! The signed notes left in the uploads bucket, as `desearch.manifest` writes and verifies them.

use serde_json::{Map, Value};

use crate::proofs::sha256;
use crate::py::canonical;

pub const OPEN_LIST_KEY: &str = "validation/open.json";
pub const UPLOAD_LOG_LATEST: &str = "log/uploads/latest.json";
/// Far enough ahead to anchor the commitment before the seed block.
pub const REVEAL_AFTER_BLOCKS: i64 = 10;
pub const FIELDS: [&str; 17] = [
    "task_id",
    "kind",
    "round_id",
    "miner",
    "key",
    "urls",
    "size",
    "etag",
    "frozen_block",
    "seed_block",
    "completed_at",
    "deadline",
    "model",
    "texts",
    "chars",
    "input_key",
    "input_sha256",
];

/// The manifest fields that are set, as the bytes its signature covers.
pub fn payload(manifest: &Map<String, Value>) -> Vec<u8> {
    let body: Map<String, Value> =
        FIELDS.iter().filter_map(|name| manifest.get(*name).filter(|v| !v.is_null()).map(|v| (name.to_string(), v.clone()))).collect();
    canonical(&Value::Object(body))
}

pub fn upload_log_key(seq: u64) -> String {
    format!("log/uploads/seq/{seq:012}.json")
}

pub fn log_payload(body: &Map<String, Value>) -> Vec<u8> {
    let mut unsigned = body.clone();
    unsigned.remove("signature");
    canonical(&Value::Object(unsigned))
}

pub fn seed_from_hash(block_hash: &str) -> String {
    sha256(block_hash.as_bytes())
}

pub fn local_block_hash(block: i64) -> String {
    format!("local:{block}")
}

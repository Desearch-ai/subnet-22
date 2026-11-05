//! A JSON-RPC stand-in for subtensor: blocks every `block_seconds` on the clock, calls counted, and refusals on demand.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;

pub struct FakeChain {
    pub url: String,
    pub refusing: AtomicBool,
    /// Answers heads this far ahead and hashes of its own, as a lying endpoint would.
    pub lying: AtomicBool,
    /// The finalized head trails the best block by this many blocks.
    pub finality_lag: i64,
    /// Milliseconds the time block authors put in blocks runs behind this machine's clock: the API's host clock seen ahead.
    behind_ms: AtomicI64,
    calls: Mutex<HashMap<String, usize>>,
    started: Instant,
    first_block: i64,
    block_seconds: f64,
}

impl FakeChain {
    pub async fn start(first_block: i64, block_seconds: f64) -> Arc<FakeChain> {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        let chain = Arc::new(FakeChain {
            url,
            refusing: AtomicBool::new(false),
            lying: AtomicBool::new(false),
            finality_lag: 2,
            behind_ms: AtomicI64::new(0),
            calls: Mutex::default(),
            started: Instant::now(),
            first_block,
            block_seconds,
        });
        let router = Router::new().route("/", post(answer)).with_state(chain.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.expect("the fake chain serves") });
        chain
    }

    pub fn best(&self) -> i64 {
        self.first_block + (self.started.elapsed().as_secs_f64() / self.block_seconds).floor() as i64
    }

    pub fn calls(&self, method: &str) -> usize {
        self.calls.lock().expect("calls").get(method).copied().unwrap_or(0)
    }

    pub fn refuse(&self, refusing: bool) {
        self.refusing.store(refusing, Ordering::SeqCst);
    }

    pub fn lie(&self, lying: bool) {
        self.lying.store(lying, Ordering::SeqCst);
    }

    /// Block authors' clocks this many seconds behind this machine's, as if the API's host clock had jumped ahead.
    pub fn set_behind(&self, seconds: f64) {
        self.behind_ms.store((seconds * 1000.0) as i64, Ordering::SeqCst);
    }

    /// `Timestamp::Now` of a block, in milliseconds: when it was made, by its author's clock, on a chain of 12-second blocks
    /// whose newest block was made just now, however fast this one counts.
    fn block_time_ms(&self, block: i64) -> u64 {
        let made = desearch::time::now() - (self.best() - block) as f64 * task_api::seeds::BLOCK_SECONDS;
        (made * 1000.0) as u64 - self.behind_ms.load(Ordering::SeqCst) as u64
    }

    pub fn finalized(&self) -> i64 {
        self.best() - self.finality_lag
    }

    /// The block a hash this chain gave belongs to.
    fn block_of(&self, hash: &str) -> Option<i64> {
        (0..=self.best()).rev().take(10_000).find(|block| hash_of(*block) == hash)
    }
}

/// What a lying endpoint answers for a block instead of its hash.
pub fn forged_hash_of(block: i64) -> String {
    hash_of(block + 1_000_000)
}

/// `twox128("Timestamp") ++ twox128("Now")`, the key every Substrate chain keeps a block's time at.
pub const TIMESTAMP_NOW: &str = "0xf0c365c3cf59d671eb72da0e7a4113c49f1f0515f462cdcf84e0f1d6045dfcbb";

pub fn hash_of(block: i64) -> String {
    format!("0x{}", desearch::canonical::hex(&Sha256::digest(block.to_string().as_bytes())))
}

async fn answer(State(chain): State<Arc<FakeChain>>, Json(request): Json<Value>) -> Response {
    let method = request["method"].as_str().unwrap_or_default().to_string();
    *chain.calls.lock().expect("calls").entry(method.clone()).or_default() += 1;
    if chain.refusing.load(Ordering::SeqCst) {
        return (StatusCode::TOO_MANY_REQUESTS, "Too many requests from this source.").into_response();
    }
    let lying = chain.lying.load(Ordering::SeqCst);
    let ahead = if lying { 1_000 } else { 0 };
    let result = match method.as_str() {
        "chain_getHeader" => match request["params"][0].as_str() {
            Some(hash) if lying => json!({"number": format!("0x{:x}", chain.best() + ahead), "hash": hash}),
            Some(hash) => match chain.block_of(hash) {
                Some(block) => json!({"number": format!("0x{:x}", block)}),
                None => Value::Null,
            },
            None => json!({"number": format!("0x{:x}", chain.best() + ahead)}),
        },
        "chain_getFinalizedHead" => json!(if lying { forged_hash_of(chain.finalized()) } else { hash_of(chain.finalized()) }),
        "chain_getBlockHash" => {
            let block = request["params"][0].as_i64().unwrap_or(0);
            if lying {
                json!(forged_hash_of(block))
            } else if block <= chain.best() {
                json!(hash_of(block))
            } else {
                Value::Null
            }
        }
        "state_getStorage" if request["params"][0] == TIMESTAMP_NOW => match request["params"][1].as_str().and_then(|hash| chain.block_of(hash)) {
            Some(block) => json!(format!("0x{}", desearch::canonical::hex(&chain.block_time_ms(block).to_le_bytes()))),
            None => Value::Null,
        },
        _ => return Json(json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "not here"}})).into_response(),
    };
    Json(json!({"jsonrpc": "2.0", "id": 1, "result": result})).into_response()
}

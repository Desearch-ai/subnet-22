//! A JSON-RPC stand-in for subtensor: blocks every `block_seconds` on the clock, calls counted, and refusals on demand.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
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
    calls: Mutex<HashMap<String, usize>>,
    started: Instant,
    first_block: i64,
    block_seconds: f64,
}

impl FakeChain {
    pub async fn start(first_block: i64, block_seconds: f64) -> Arc<FakeChain> {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        let chain = Arc::new(FakeChain { url, refusing: AtomicBool::new(false), calls: Mutex::default(), started: Instant::now(), first_block, block_seconds });
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
}

pub fn hash_of(block: i64) -> String {
    format!("0x{}", desearch::canonical::hex(&Sha256::digest(block.to_string().as_bytes())))
}

async fn answer(State(chain): State<Arc<FakeChain>>, Json(request): Json<Value>) -> Response {
    let method = request["method"].as_str().unwrap_or_default().to_string();
    *chain.calls.lock().expect("calls").entry(method.clone()).or_default() += 1;
    if chain.refusing.load(Ordering::SeqCst) {
        return (StatusCode::TOO_MANY_REQUESTS, "Too many requests from this source.").into_response();
    }
    let result = match method.as_str() {
        "chain_getHeader" => json!({"number": format!("0x{:x}", chain.best())}),
        "chain_getBlockHash" => {
            let block = request["params"][0].as_i64().unwrap_or(0);
            if block <= chain.best() {
                json!(hash_of(block))
            } else {
                Value::Null
            }
        }
        _ => return Json(json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "not here"}})).into_response(),
    };
    Json(json!({"jsonrpc": "2.0", "id": 1, "result": result})).into_response()
}

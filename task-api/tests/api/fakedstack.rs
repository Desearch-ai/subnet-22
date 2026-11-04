//! A stand-in for dstack's guest agent on a Unix socket: quotes laid out as TDX quotes v4 with the asked report data, and calls counted.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use desearch::canonical::hex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256, Sha384};

pub const MRTD: [u8; 48] = [0x11; 48];
pub const APP_COMPOSE: &str = r#"{"allowed_envs":["CF_R2_ACCESS_KEY_ID","CF_R2_SECRET_ACCESS_KEY"],"docker_compose_file":"services: {}","kms_enabled":true,"manifest_version":2,"name":"sn22-task-api","runner":"docker-compose"}"#;
const RUNTIME_EVENT: u32 = 0x0800_0001;

pub struct FakeDstack {
    pub socket: PathBuf,
    calls: Mutex<HashMap<String, usize>>,
}

impl FakeDstack {
    pub async fn start(dir: &std::path::Path) -> Arc<FakeDstack> {
        std::fs::create_dir_all(dir).expect("a socket directory");
        let socket = dir.join("dstack.sock");
        let _ = std::fs::remove_file(&socket);
        let listener = tokio::net::UnixListener::bind(&socket).expect("the socket binds");
        let agent = Arc::new(FakeDstack { socket, calls: Mutex::default() });
        let router = Router::new().route("/Info", get(info)).route("/GetQuote", post(quote)).with_state(agent.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.expect("the fake agent serves") });
        agent
    }

    pub fn calls(&self, path: &str) -> usize {
        self.calls.lock().expect("calls").get(path).copied().unwrap_or(0)
    }

    fn count(&self, path: &str) {
        *self.calls.lock().expect("calls").entry(path.into()).or_default() += 1;
    }
}

pub fn compose_hash() -> String {
    hex(&Sha256::digest(APP_COMPOSE.as_bytes()))
}

/// The RTMR3 events dstack logs at boot, digested as its V1 runtime events are.
pub fn event_log() -> Vec<Value> {
    let events = [
        ("app-id", vec![0x22; 20]),
        ("compose-hash", Sha256::digest(APP_COMPOSE.as_bytes()).to_vec()),
        ("instance-id", vec![0x33; 20]),
        ("boot-mr-done", vec![]),
    ];
    events
        .iter()
        .map(|(name, payload)| {
            let mut digest = Sha384::new();
            digest.update(RUNTIME_EVENT.to_le_bytes());
            digest.update(b":");
            digest.update(name.as_bytes());
            digest.update(b":");
            digest.update(payload);
            json!({"imr": 3, "event_type": RUNTIME_EVENT, "digest": hex(&digest.finalize()), "event": name, "event_payload": hex(payload)})
        })
        .collect()
}

fn rtmr3() -> [u8; 48] {
    let mut rtmr = [0u8; 48];
    for event in event_log() {
        let digest = task_api::attestation::decode_hex(event["digest"].as_str().unwrap()).unwrap();
        let mut next = Sha384::new();
        next.update(rtmr);
        next.update(&digest);
        rtmr = next.finalize().into();
    }
    rtmr
}

/// A quote v4 for a TD: the 48-byte header, the 584-byte report body, and an empty signature section.
pub fn quote_bytes(report_data: &[u8]) -> Vec<u8> {
    let mut quote = Vec::with_capacity(48 + 584 + 4);
    quote.extend(4u16.to_le_bytes());
    quote.extend(2u16.to_le_bytes());
    quote.extend(0x81u32.to_le_bytes());
    quote.extend([0u8; 40]);
    quote.extend([0u8; 16 + 48 + 48 + 8 + 8 + 8]);
    quote.extend(MRTD);
    // mrconfigid, mrowner, mrownerconfig, then RTMR0 to RTMR2.
    quote.extend([0u8; 48 * 6]);
    quote.extend(rtmr3());
    let mut data = [0u8; 64];
    data[..report_data.len()].copy_from_slice(report_data);
    quote.extend(data);
    quote.extend(0u32.to_le_bytes());
    quote
}

async fn info(State(agent): State<Arc<FakeDstack>>) -> Json<Value> {
    agent.count("/Info");
    let zero = hex(&[0u8; 48]);
    let tcb = json!({"mrtd": hex(&MRTD), "rtmr0": zero, "rtmr1": zero, "rtmr2": zero, "rtmr3": hex(&rtmr3()), "event_log": event_log(), "app_compose": APP_COMPOSE, "compose_hash": compose_hash()});
    Json(json!({
        "app_id": hex(&[0x22; 20]),
        "instance_id": hex(&[0x33; 20]),
        "app_name": "sn22-task-api",
        "compose_hash": compose_hash(),
        "device_id": hex(&[0x44; 32]),
        "os_image_hash": hex(&[0x55; 32]),
        "tcb_info": serde_json::to_string_pretty(&tcb).unwrap(),
        "vm_config": "{\"cpu_count\":4}",
    }))
}

async fn quote(State(agent): State<Arc<FakeDstack>>, Json(request): Json<Value>) -> Json<Value> {
    agent.count("/GetQuote");
    let data = task_api::attestation::decode_hex(request["report_data"].as_str().unwrap_or_default()).unwrap_or_default();
    let mut padded = [0u8; 64];
    padded[..data.len().min(64)].copy_from_slice(&data[..data.len().min(64)]);
    Json(json!({
        "quote": hex(&quote_bytes(&padded)),
        "event_log": serde_json::to_string(&event_log()).unwrap(),
        "report_data": hex(&padded),
        "vm_config": "{\"cpu_count\":4}",
    }))
}

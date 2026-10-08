//! Who is calling: four signed headers, checked against the subnet's hotkeys and a nonce used once.

use axum::http::HeaderMap;
use desearch::hotkey::{public_of, signing_payload, verify};
use desearch::time::now;

use crate::http::ApiError;
use crate::state::State;

const TOLERANCE_S: f64 = 60.0;
const NONCE_TTL_S: i64 = 120;
const MAX_HOTKEY_CHARS: usize = 64;
pub const CLIENT_ADDRESS_HEADER: &str = "CF-Connecting-IP";

#[derive(Clone, Debug)]
pub struct Caller {
    pub hotkey: String,
    pub uid: Option<i64>,
    pub is_validator: bool,
    pub requested_at: f64,
    pub is_admin: bool,
}

pub fn client_address(headers: &HeaderMap, peer: &str) -> String {
    headers.get(CLIENT_ADDRESS_HEADER).and_then(|v| v.to_str().ok()).filter(|v| !v.is_empty()).unwrap_or(peer).to_string()
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or_default()
}

fn signature_verifies(hotkey: &str, payload: &[u8], signature: &str) -> bool {
    let (Some(public), Some(signature)) = (public_of(hotkey), decode_hex(signature)) else { return false };
    verify(&public, payload, &signature)
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

/// The caller of a signed request; a failure counts against the address it came from.
pub async fn caller(state: &State, headers: &HeaderMap, method: &str, path: &str, body: &[u8], address: &str) -> Result<Caller, ApiError> {
    match authenticate(state, headers, method, path, body).await {
        Err(ApiError::Status { status, detail, retry_after }) => {
            state.write_denied(address).await?;
            Err(ApiError::Status { status, detail, retry_after })
        }
        other => other,
    }
}

async fn authenticate(state: &State, headers: &HeaderMap, method: &str, path: &str, body: &[u8]) -> Result<Caller, ApiError> {
    let (hotkey, timestamp, nonce, signature) =
        (header(headers, "X-Hotkey"), header(headers, "X-Timestamp"), header(headers, "X-Nonce"), header(headers, "X-Signature"));
    if [hotkey, timestamp, nonce, signature].iter().any(|h| h.is_empty()) {
        return Err(ApiError::status(401, "missing auth headers"));
    }
    let nonce_ok = nonce.len() == 32 && nonce.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if hotkey.chars().count() > MAX_HOTKEY_CHARS || !nonce_ok || !timestamp.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ApiError::status(401, "malformed auth headers"));
    }
    let sent_at: f64 = timestamp.parse::<u64>().map_err(|_| ApiError::status(401, "malformed auth headers"))? as f64;
    if (now() - sent_at).abs() > TOLERANCE_S {
        return Err(ApiError::status(401, "timestamp outside tolerance"));
    }
    let admin = state.admins.contains(hotkey);
    let entry = if admin { None } else { state.registry.lookup(hotkey).await };
    if entry.is_none() && !admin {
        return Err(ApiError::status(403, "hotkey is not registered on this subnet"));
    }
    if !signature_verifies(hotkey, &signing_payload(method, path, body, timestamp, nonce), signature) {
        return Err(ApiError::status(401, "bad signature"));
    }
    // Claimed last so unauthenticated requests cannot fill the nonce cache.
    if !state.set_once(&format!("nonce:{hotkey}:{nonce}"), "1", NONCE_TTL_S).await? {
        return Err(ApiError::status(401, "nonce already used"));
    }
    Ok(Caller {
        hotkey: hotkey.into(),
        uid: entry.as_ref().and_then(|e| e.uid),
        is_validator: entry.as_ref().is_some_and(|e| e.is_validator),
        requested_at: sent_at,
        is_admin: admin,
    })
}

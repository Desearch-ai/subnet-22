//! A local server that answers every recorded host over HTTPS or plain HTTP on one port, for bittensor-fetch replays.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use base64::Engine;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_rustls::rustls;

pub struct Replay {
    pub addr: SocketAddr,
    pub ca_pem: String,
    pub misses: Arc<Mutex<Vec<String>>>,
}

/// The key a request is answered by: host, method and target, and for a JSON-RPC post its method and params.
pub fn key(host: &str, method: &str, target: &str, body: &[u8]) -> String {
    let mut key = format!("{host} {method} {target}");
    if method == "POST" {
        if let Ok(request) = serde_json::from_slice::<Value>(body) {
            key.push_str(&format!(" {} {}", request["method"].as_str().unwrap_or(""), request["params"]));
        }
    }
    key
}

fn answer(record: &Value) -> Option<Vec<u8>> {
    if record.get("close").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let status = record["status"].as_u64().unwrap_or(200);
    let body = record.get("b64").and_then(Value::as_str).map(|b| base64::engine::general_purpose::STANDARD.decode(b).expect("base64")).unwrap_or_default();
    let mut head = format!("HTTP/1.1 {status} X\r\n");
    for pair in record.get("headers").and_then(Value::as_array).into_iter().flatten() {
        head.push_str(&format!("{}: {}\r\n", pair[0].as_str().unwrap_or(""), pair[1].as_str().unwrap_or("")));
    }
    head.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", body.len()));
    Some([head.into_bytes(), body].concat())
}

struct State {
    answers: HashMap<String, Vec<Value>>,
    served: Mutex<HashMap<String, usize>>,
    misses: Arc<Mutex<Vec<String>>>,
}

async fn serve<S: AsyncRead + AsyncWrite + Unpin>(mut stream: S, state: &State) {
    let mut request = Vec::new();
    let mut buf = [0u8; 8192];
    let head_end = loop {
        if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    };
    let head = String::from_utf8_lossy(&request[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
    let (mut host, mut length) = (String::new(), 0usize);
    for line in lines {
        let Some((name, value)) = line.split_once(':') else { continue };
        match name.trim().to_ascii_lowercase().as_str() {
            "host" => {
                host = value.trim().rsplit_once(':').map_or(value.trim(), |(h, p)| if p.parse::<u16>().is_ok() { h } else { value.trim() }).to_lowercase()
            }
            "content-length" => length = value.trim().parse().unwrap_or(0),
            _ => {}
        }
    }
    let mut body = request[head_end..].to_vec();
    while body.len() < length {
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&buf[..n]),
        }
    }
    let key = key(&host, &method, &target, &body);
    let record = match state.answers.get(&key) {
        Some(records) => {
            let mut served = state.served.lock().expect("served");
            let at = served.entry(key.clone()).or_default();
            let record = records[(*at).min(records.len() - 1)].clone();
            *at += 1;
            record
        }
        None => {
            state.misses.lock().expect("misses").push(key);
            serde_json::json!({"status": 404, "headers": [["Content-Type", "text/plain"]]})
        }
    };
    if let Some(bytes) = answer(&record) {
        let _ = stream.write_all(&bytes).await;
        let _ = stream.flush().await;
    }
    let _ = stream.shutdown().await;
}

/// Serve `exchanges` (`{"<key>": [answers in order]}`) on 127.0.0.1 under a certificate made for `hosts`.
pub async fn start(exchanges: &Value, hosts: Vec<String>, port: u16) -> Replay {
    let made = rcgen::generate_simple_self_signed(hosts).expect("a certificate");
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(made.signing_key.serialize_der().into());
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .expect("TLS versions")
        .with_no_client_auth()
        .with_single_cert(vec![made.cert.der().clone()], key)
        .expect("a TLS config");
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind(("127.0.0.1", port)).await.expect("a port");
    let addr = listener.local_addr().expect("an address");
    let answers: HashMap<String, Vec<Value>> =
        exchanges.as_object().expect("exchanges").iter().map(|(k, v)| (k.clone(), v.as_array().cloned().unwrap_or_default())).collect();
    let misses = Arc::new(Mutex::new(Vec::new()));
    let state = Arc::new(State { answers, served: Mutex::new(HashMap::new()), misses: misses.clone() });
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            let (acceptor, state) = (acceptor.clone(), state.clone());
            tokio::spawn(async move {
                let mut first = [0u8; 1];
                if socket.peek(&mut first).await.is_err() {
                    return;
                }
                if first[0] == 0x16 {
                    if let Ok(stream) = acceptor.accept(socket).await {
                        serve(stream, &state).await;
                    }
                } else {
                    serve(socket, &state).await;
                }
            });
        }
    });
    Replay { addr, ca_pem: made.cert.pem(), misses }
}

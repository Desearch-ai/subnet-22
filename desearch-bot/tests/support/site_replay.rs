//! A local HTTPS server that answers every fixture host's recorded responses and logs each request, for the crawl-sites parity runs.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use base64::Engine;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_rustls::rustls;

#[derive(Clone, Debug)]
pub struct Hit {
    pub t: f64,
    pub host: String,
    pub target: String,
    pub cookies: Vec<String>,
    pub agent: String,
}

impl Hit {
    pub fn json(&self) -> Value {
        json!({"t": self.t, "host": self.host, "target": self.target, "cookies": self.cookies, "agent": self.agent})
    }
}

pub struct Replay {
    pub addr: SocketAddr,
    pub ca_pem: String,
    pub hits: Arc<Mutex<Vec<Hit>>>,
}

fn answer(hosts: &Value, host: &str, target: &str, cookies: &[String]) -> Option<Vec<u8>> {
    let mut resp = hosts.get(host).and_then(|h| h.get(target)).cloned().unwrap_or_else(|| json!({"status": 404}));
    while let Some(name) = resp.get("need_cookie").and_then(Value::as_str) {
        if cookies.iter().any(|c| c == name) {
            resp.as_object_mut().unwrap().remove("need_cookie");
        } else {
            resp = resp["else"].clone();
        }
    }
    if resp.get("close").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let status = resp["status"].as_u64().unwrap_or(200);
    let body: Vec<u8> = match (resp.get("b64").and_then(Value::as_str), resp.get("body").and_then(Value::as_str)) {
        (Some(b64), _) => base64::engine::general_purpose::STANDARD.decode(b64).unwrap(),
        (None, Some(text)) => text.as_bytes().to_vec(),
        _ => Vec::new(),
    };
    let mut head = format!("HTTP/1.1 {status} X\r\n");
    if let Some(kind) = resp.get("type").and_then(Value::as_str) {
        head.push_str(&format!("Content-Type: {kind}\r\n"));
    }
    for pair in resp.get("headers").and_then(Value::as_array).into_iter().flatten() {
        head.push_str(&format!("{}: {}\r\n", pair[0].as_str().unwrap(), pair[1].as_str().unwrap()));
    }
    head.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", body.len()));
    Some([head.into_bytes(), body].concat())
}

async fn serve<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    mut stream: S,
    hosts: &Value,
    hits: &Mutex<Vec<Hit>>,
    started: Instant,
    log: Option<&Mutex<std::fs::File>>,
) {
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") && request.len() < 65_536 {
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
    let text = String::from_utf8_lossy(&request).into_owned();
    let mut lines = text.split("\r\n");
    let target = lines.next().unwrap_or("").split(' ').nth(1).unwrap_or("").to_string();
    let (mut host, mut cookies, mut agent) = (String::new(), Vec::new(), String::new());
    for line in lines {
        let Some((name, value)) = line.split_once(':') else { continue };
        match name.trim().to_ascii_lowercase().as_str() {
            "host" => host = value.trim().split(':').next().unwrap_or("").to_lowercase(),
            "cookie" => cookies.extend(value.split(';').filter_map(|c| c.split_once('=').map(|(n, _)| n.trim().to_string()))),
            "user-agent" => agent = value.trim().to_string(),
            _ => {}
        }
    }
    let hit = Hit { t: started.elapsed().as_secs_f64(), host: host.clone(), target: target.clone(), cookies: cookies.clone(), agent };
    if let Some(log) = log {
        use std::io::Write;
        let mut file = log.lock().unwrap();
        let _ = writeln!(file, "{}", hit.json());
        let _ = file.flush();
    }
    hits.lock().unwrap().push(hit);
    if let Some(bytes) = answer(hosts, &host, &target, &cookies) {
        let _ = stream.write_all(&bytes).await;
        let _ = stream.flush().await;
    }
    let _ = stream.shutdown().await;
}

/// Serve the fixture's hosts on 127.0.0.1 (port 0 for any), under a certificate made for their names.
pub async fn start(fixture: &Value, port: u16, log: Option<std::path::PathBuf>) -> Replay {
    let hosts = Arc::new(fixture["hosts"].clone());
    let names: Vec<String> = hosts.as_object().unwrap().keys().cloned().collect();
    let made = rcgen::generate_simple_self_signed(names).unwrap();
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(made.signing_key.serialize_der().into());
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![made.cert.der().clone()], key)
        .unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind(("127.0.0.1", port)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let log = log.map(|path| Arc::new(Mutex::new(std::fs::File::create(path).unwrap())));
    let (served, started) = (hits.clone(), Instant::now());
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            let (acceptor, hosts, hits, log) = (acceptor.clone(), hosts.clone(), served.clone(), log.clone());
            tokio::spawn(async move {
                if let Ok(stream) = acceptor.accept(socket).await {
                    serve(stream, &hosts, &hits, started, log.as_deref()).await;
                }
            });
        }
    });
    Replay { addr, ca_pem: made.cert.pem(), hits }
}

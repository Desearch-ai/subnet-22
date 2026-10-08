//! A stand-in for R2 that keeps objects as files under one folder, for tests and local runs; it checks no signatures.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::time::http_date;

#[derive(Default)]
pub struct State {
    pub root: PathBuf,
    meta: Mutex<HashMap<String, (String, String)>>,
    uploads: Mutex<HashMap<String, BTreeMap<u32, Vec<u8>>>>,
    failures: Mutex<Vec<(String, String, u32)>>,
    delays: Mutex<Vec<(String, String, u32, std::time::Duration)>>,
    next_upload: AtomicU64,
    counts: Mutex<HashMap<String, u64>>,
}

pub struct Stub {
    pub addr: SocketAddr,
    pub state: Arc<State>,
}

struct Request {
    method: String,
    path: String,
    query: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl State {
    /// The next `times` requests of this method for objects under `prefix` (`bucket/key...`) get a 503.
    pub fn fail(&self, method: &str, prefix: &str, times: u32) {
        self.failures.lock().unwrap().push((method.into(), prefix.into(), times));
    }

    /// The next `times` requests of this method for objects under `prefix` are answered after `delay`.
    pub fn slow(&self, method: &str, prefix: &str, times: u32, delay: std::time::Duration) {
        self.delays.lock().unwrap().push((method.into(), prefix.into(), times, delay));
    }

    fn delay(&self, method: &str, path: &str) -> Option<std::time::Duration> {
        let target = path.trim_start_matches('/');
        let mut delays = self.delays.lock().unwrap();
        let rule = delays.iter_mut().find(|(m, prefix, times, _)| m == method && target.starts_with(prefix.as_str()) && *times > 0)?;
        rule.2 -= 1;
        Some(rule.3)
    }

    pub fn object(&self, bucket: &str, key: &str) -> Option<Vec<u8>> {
        std::fs::read(self.root.join(bucket).join(key)).ok()
    }

    /// Content type and cache control the object was written with.
    pub fn meta(&self, bucket: &str, key: &str) -> Option<(String, String)> {
        self.meta.lock().unwrap().get(&format!("{bucket}/{key}")).cloned()
    }

    pub fn keys(&self, bucket: &str, prefix: &str) -> Vec<String> {
        let base = self.root.join(bucket);
        let mut found = Vec::new();
        walk(&base, &base, &mut found);
        found.retain(|key| key.starts_with(prefix));
        found.sort();
        found
    }

    /// Requests served by method, failures included.
    pub fn count(&self, method: &str) -> u64 {
        self.counts.lock().unwrap().get(method).copied().unwrap_or(0)
    }
}

fn walk(base: &Path, dir: &Path, found: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(base, &path, found);
        } else if let Ok(relative) = path.strip_prefix(base) {
            found.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Serves on `listen` until the process exits.
pub async fn start(root: PathBuf, listen: &str) -> Result<Stub> {
    std::fs::create_dir_all(&root)?;
    let listener = TcpListener::bind(listen).await?;
    let addr = listener.local_addr()?;
    let state = Arc::new(State { root, ..State::default() });
    let serving = state.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let state = serving.clone();
            tokio::spawn(async move {
                let _ = connection(stream, state).await;
            });
        }
    });
    Ok(Stub { addr, state })
}

async fn connection(stream: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        let mut parts = line.split_whitespace();
        let (method, target) = (parts.next().unwrap_or_default().to_string(), parts.next().unwrap_or_default().to_string());
        let mut headers = HashMap::new();
        loop {
            let mut header = String::new();
            reader.read_line(&mut header).await?;
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let length: usize = headers.get("content-length").and_then(|l| l.parse().ok()).unwrap_or(0);
        let mut body = vec![0; length];
        reader.read_exact(&mut body).await?;
        let (path, query) = target.split_once('?').unwrap_or((&target, ""));
        let query = query.split('&').filter(|p| !p.is_empty()).map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k), decode(v))
        });
        let request = Request { method, path: decode(path), query: query.collect(), headers, body };
        if let Some(delay) = state.delay(&request.method, &request.path) {
            tokio::time::sleep(delay).await;
        }
        let head = request.method == "HEAD";
        let (status, headers, body) = respond(&state, request);
        let mut out =
            format!("HTTP/1.1 {status} {}\r\nContent-Length: {}\r\n", reason(status), headers.get("length").cloned().unwrap_or(body.len().to_string()));
        for (name, value) in headers.iter().filter(|(name, _)| **name != "length") {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
        out.push_str("\r\n");
        writer.write_all(out.as_bytes()).await?;
        if !head {
            writer.write_all(&body).await?;
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        206 => "Partial Content",
        404 => "Not Found",
        412 => "Precondition Failed",
        503 => "Service Unavailable",
        _ => "Bad Request",
    }
}

fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let pair = bytes.get(i + 1..i + 3).and_then(|pair| std::str::from_utf8(pair).ok()).and_then(|pair| u8::from_str_radix(pair, 16).ok());
        match (bytes[i], pair) {
            (b'%', Some(byte)) => {
                out.push(byte);
                i += 3;
            }
            (byte, _) => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn etag(path: &Path) -> Option<String> {
    let found = std::fs::metadata(path).ok()?;
    let modified = found.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("\"{:x}{modified:x}\"", found.len()))
}

fn last_modified(path: &Path) -> Option<String> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(http_date(modified.as_secs() as i64))
}

type Response = (u16, BTreeMap<&'static str, String>, Vec<u8>);

fn respond(state: &State, request: Request) -> Response {
    *state.counts.lock().unwrap().entry(request.method.clone()).or_default() += 1;
    let target = request.path.trim_start_matches('/').to_string();
    let (bucket, key) = target.split_once('/').unwrap_or((&target, ""));
    {
        let mut failures = state.failures.lock().unwrap();
        if let Some(rule) = failures.iter_mut().find(|(method, prefix, times)| *method == request.method && target.starts_with(prefix.as_str()) && *times > 0) {
            rule.2 -= 1;
            return (503, BTreeMap::new(), b"<Error><Code>SlowDown</Code></Error>".to_vec());
        }
    }
    let empty = BTreeMap::new;
    if key.is_empty() || key.split('/').any(|part| part == "..") {
        return (if request.method == "HEAD" { 200 } else { 400 }, empty(), Vec::new());
    }
    let path = state.root.join(bucket).join(key);
    let name = format!("{bucket}/{key}");
    match (request.method.as_str(), request.query.get("uploadId"), request.query.contains_key("uploads")) {
        ("POST", None, true) => {
            let id = format!("upload-{}", state.next_upload.fetch_add(1, Ordering::Relaxed));
            state.uploads.lock().unwrap().insert(id.clone(), BTreeMap::new());
            let body =
                format!("<InitiateMultipartUploadResult><Bucket>{bucket}</Bucket><Key>{key}</Key><UploadId>{id}</UploadId></InitiateMultipartUploadResult>");
            (200, empty(), body.into_bytes())
        }
        ("PUT", Some(id), _) => {
            let number: u32 = request.query.get("partNumber").and_then(|n| n.parse().ok()).unwrap_or(0);
            match state.uploads.lock().unwrap().get_mut(id) {
                Some(parts) => {
                    parts.insert(number, request.body);
                    (200, BTreeMap::from([("ETag", format!("\"part{number}\""))]), Vec::new())
                }
                None => (404, empty(), Vec::new()),
            }
        }
        ("POST", Some(id), _) => {
            let Some(parts) = state.uploads.lock().unwrap().remove(id) else {
                return (404, empty(), Vec::new());
            };
            let whole: Vec<u8> = parts.into_values().flatten().collect();
            if write(&path, &whole).is_err() {
                return (500, empty(), Vec::new());
            }
            let tag = etag(&path).unwrap_or_default();
            (200, empty(), format!("<CompleteMultipartUploadResult><ETag>{tag}</ETag></CompleteMultipartUploadResult>").into_bytes())
        }
        ("DELETE", Some(id), _) => {
            state.uploads.lock().unwrap().remove(id);
            (204, empty(), Vec::new())
        }
        ("PUT", None, _) if request.headers.contains_key("x-amz-copy-source") => {
            let source = decode(request.headers["x-amz-copy-source"].trim_start_matches('/'));
            let from = state.root.join(&source);
            let (Some(tag), Ok(data)) = (etag(&from), std::fs::read(&from)) else {
                return (404, empty(), Vec::new());
            };
            if request.headers.get("x-amz-copy-source-if-match").is_some_and(|wanted| *wanted != tag) {
                return (412, empty(), Vec::new());
            }
            if write(&path, &data).is_err() {
                return (500, empty(), Vec::new());
            }
            let copied = state.meta.lock().unwrap().get(&source).cloned();
            if let Some(meta) = copied {
                state.meta.lock().unwrap().insert(name, meta);
            }
            let tag = etag(&path).unwrap_or_default();
            (200, empty(), format!("<CopyObjectResult><ETag>{tag}</ETag></CopyObjectResult>").into_bytes())
        }
        ("PUT", None, _) => {
            if write(&path, &request.body).is_err() {
                return (500, empty(), Vec::new());
            }
            let content_type = request.headers.get("content-type").cloned().unwrap_or_default();
            let cache_control = request.headers.get("cache-control").cloned().unwrap_or_default();
            state.meta.lock().unwrap().insert(name, (content_type, cache_control));
            (200, BTreeMap::from([("ETag", etag(&path).unwrap_or_default())]), Vec::new())
        }
        ("DELETE", None, _) => {
            let _ = std::fs::remove_file(&path);
            (204, empty(), Vec::new())
        }
        ("HEAD" | "GET", None, _) => {
            let (Some(tag), Ok(data)) = (etag(&path), std::fs::read(&path)) else {
                return (404, empty(), Vec::new());
            };
            if request.headers.get("if-match").is_some_and(|wanted| *wanted != tag) {
                return (412, empty(), Vec::new());
            }
            let mut headers = BTreeMap::from([("ETag", tag), ("Last-Modified", last_modified(&path).unwrap_or_default())]);
            if request.method == "HEAD" {
                headers.insert("length", data.len().to_string());
                return (200, headers, Vec::new());
            }
            match request.headers.get("range").and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.split_once('-')) {
                Some(("", tail)) => {
                    let tail: usize = tail.parse().unwrap_or(0);
                    (206, headers, data[data.len().saturating_sub(tail)..].to_vec())
                }
                Some((start, end)) => {
                    let start: usize = start.parse().unwrap_or(0);
                    let end: usize = end.parse::<usize>().map_or(data.len(), |e| (e + 1).min(data.len()));
                    (206, headers, data.get(start..end).unwrap_or_default().to_vec())
                }
                None => (200, headers, data),
            }
        }
        _ => (400, empty(), Vec::new()),
    }
}

fn write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, data)
}

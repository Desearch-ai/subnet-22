//! One GET without following redirects: our user agent, size and time limits, the run's cookies.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use super::cookies::Jar;
use crate::blocklist;
use crate::net::{self, SystemResolver};
use crate::signing::USER_AGENT;
use crate::urls;

pub const MAX_BYTES: usize = 8 << 20;
pub const TIMEOUT: Duration = Duration::from_secs(30);
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const RETRY_TIMEOUT: Duration = Duration::from_secs(60);
pub const RETRY_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

static CHARSET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)charset=["']?([\w.:-]+)"#).unwrap());

#[derive(Clone, Debug)]
pub struct Response {
    pub url: String,
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub fetched_at: DateTime<Utc>,
    pub error: Option<String>,
}

impl Response {
    pub fn failed(url: &str, error: &str) -> Self {
        Response { url: url.to_string(), status: 0, headers: HashMap::new(), body: Vec::new(), fetched_at: Utc::now(), error: Some(error.to_string()) }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }

    pub fn content_type(&self) -> String {
        let value = self.header("content-type").unwrap_or("");
        crate::pystr::strip(value.split(';').next().unwrap_or("")).to_lowercase()
    }

    /// The body in the charset its content type names, UTF-8 when it names none Python knows, bad bytes replaced.
    pub fn text(&self) -> String {
        let label = self.header("content-type").and_then(|value| CHARSET.captures(value)).map(|c| c[1].to_string());
        decode(&self.body, label.as_deref())
    }
}

/// Python's `bytes.decode(label, errors="replace")`, with UTF-8 for a label Python has no codec for.
pub fn decode(body: &[u8], label: Option<&str>) -> String {
    let name = label.map(|l| l.to_lowercase().replace(['-', ' ', '.', ':'], "_")).unwrap_or_default();
    match name.as_str() {
        "" | "utf_8" | "utf8" | "u8" | "utf" | "cp65001" | "utf8_ucs2" | "utf8_ucs4" => String::from_utf8_lossy(body).into_owned(),
        "ascii" | "us_ascii" | "646" | "us" | "ansi_x3_4_1968" | "ansi_x3_4_1986" | "cp367" | "csascii" | "ibm367" | "iso646_us" | "iso_646_irv_1991"
        | "iso_ir_6" => body.iter().map(|&b| if b < 0x80 { b as char } else { '\u{fffd}' }).collect(),
        "latin_1" | "latin1" | "latin" | "l1" | "iso8859_1" | "iso_8859_1" | "iso_8859_1_1987" | "8859" | "cp819" | "csisolatin1" | "ibm819" | "iso_ir_100" => {
            body.iter().map(|&b| b as char).collect()
        }
        _ => {
            let encoding = encoding_rs::Encoding::for_label(name.replace('_', "-").as_bytes())
                .or_else(|| windows_codepage(&name))
                .filter(|e| *e != encoding_rs::REPLACEMENT && *e != encoding_rs::X_USER_DEFINED);
            match encoding {
                Some(e) if e == encoding_rs::UTF_8 => String::from_utf8_lossy(body).into_owned(),
                Some(e) if is_windows_single_byte(e) => body
                    .iter()
                    .map(|&b| {
                        let c = e.decode_without_bom_handling(&[b]).0.chars().next().unwrap_or('\u{fffd}');
                        if ('\u{80}'..='\u{9f}').contains(&c) {
                            '\u{fffd}'
                        } else {
                            c
                        }
                    })
                    .collect(),
                Some(e) => e.decode_without_bom_handling(body).0.into_owned(),
                None => String::from_utf8_lossy(body).into_owned(),
            }
        }
    }
}

fn windows_codepage(name: &str) -> Option<&'static encoding_rs::Encoding> {
    let page = name.strip_prefix("cp").or_else(|| name.strip_prefix("windows_"))?;
    encoding_rs::Encoding::for_label(format!("windows-{page}").as_bytes())
}

/// Python's cp125x codecs leave undefined the bytes WHATWG maps to C1 controls.
fn is_windows_single_byte(e: &'static encoding_rs::Encoding) -> bool {
    e.name().starts_with("windows-125")
}

const ALLOWED: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$'()*,";

/// yarl's `_Quoter` with requoting: what aiohttp puts on the wire for a path or query it is handed.
fn requote(value: &str, safe: &[u8], protected: &[u8], qs: bool) -> String {
    let mut allowed: Vec<u8> = [safe, ALLOWED, protected].concat();
    if !qs {
        allowed.extend_from_slice(b"+&=;");
    }
    let b = value.as_bytes();
    let mut out = String::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let ch = b[i];
        if ch == b'%' {
            match (b.get(i + 1), b.get(i + 2)) {
                (Some(&h), Some(&l)) if h.is_ascii_hexdigit() && l.is_ascii_hexdigit() => {
                    let code = u8::from_str_radix(std::str::from_utf8(&[h, l]).unwrap(), 16).unwrap();
                    if code < 0x80 && !protected.contains(&code) && allowed.contains(&code) {
                        out.push(code as char);
                    } else {
                        out.push_str(&format!("%{code:02X}"));
                    }
                    i += 3;
                }
                _ => {
                    out.push_str("%25");
                    i += 1;
                }
            }
            continue;
        }
        if qs && ch == b' ' {
            out.push('+');
        } else if allowed.contains(&ch) {
            out.push(ch as char);
        } else {
            out.push_str(&format!("%{ch:02X}"));
        }
        i += 1;
    }
    out
}

/// yarl's `normalize_path`: `.` and `..` segments resolved, the root kept.
fn dot_segments(path: &str) -> String {
    let (prefix, rest) = path.strip_prefix('/').map_or(("", path), |rest| ("/", rest));
    let segments: Vec<&str> = rest.split('/').collect();
    let mut resolved: Vec<&str> = Vec::new();
    for seg in &segments {
        match *seg {
            ".." => {
                resolved.pop();
            }
            "." => {}
            other => resolved.push(other),
        }
    }
    if matches!(segments.last(), Some(&".") | Some(&"..")) {
        resolved.push("");
    }
    format!("{prefix}{}", resolved.join("/"))
}

/// The URL as aiohttp sends it: yarl requotes the path and query and resolves dot segments.
pub fn wire_url(parts: &urls::Parts) -> String {
    let scheme = parts.scheme.as_deref().unwrap_or("");
    let netloc = parts.netloc();
    let mut path = requote(&parts.path, b"@:", b"/+", false);
    if !netloc.is_empty() && path.contains('.') {
        path = dot_segments(&path);
    }
    if path.is_empty() {
        path.push('/');
    }
    let mut url = format!("{scheme}://{netloc}{path}");
    if let Some(query) = parts.query.as_deref().filter(|q| !q.is_empty()) {
        url.push('?');
        url.push_str(&requote(query, b"?/:@", b"=+&;", true));
    }
    url
}

/// Every name to one address, for tests against a local replay server.
struct Fixed(SocketAddr);

impl Resolve for Fixed {
    fn resolve(&self, _: Name) -> Resolving {
        let addr = self.0;
        Box::pin(async move { Ok(Box::new(std::iter::once(addr)) as Addrs) })
    }
}

/// Where requests go: the system resolver with private answers dropped, or one address for every name in a test.
#[derive(Clone, Default)]
pub struct Connect {
    pub resolve_all: Option<SocketAddr>,
    pub root_certificate: Option<Vec<u8>>,
}

impl Connect {
    pub fn with_certificate_file(mut self, path: &Path) -> Result<Self> {
        self.root_certificate = Some(std::fs::read(path).with_context(|| format!("reading {}", path.display()))?);
        Ok(self)
    }
}

/// The run's two clients, a normal one and the longer one robots.txt is asked again with, and their cookie jar.
pub struct Http {
    normal: reqwest::Client,
    patient: reqwest::Client,
    pub jar: Arc<Jar>,
}

impl Http {
    pub fn new(connect: &Connect) -> Result<Self> {
        Self::with_headers(connect, &[])
    }

    /// A session whose headers replace ours, as `net.session(headers)` makes one.
    pub fn with_headers(connect: &Connect, headers: &[(&str, &str)]) -> Result<Self> {
        let mut session = reqwest::header::HeaderMap::new();
        for (name, value) in headers {
            session.insert(reqwest::header::HeaderName::from_bytes(name.as_bytes())?, reqwest::header::HeaderValue::from_str(value)?);
        }
        let build = |timeout: Duration, connect_timeout: Duration| -> Result<reqwest::Client> {
            let mut builder = reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .default_headers(session.clone())
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .timeout(timeout)
                .connect_timeout(connect_timeout)
                .pool_idle_timeout(Duration::from_secs(15))
                .http1_only()
                .tcp_nodelay(true);
            builder = match connect.resolve_all {
                Some(addr) => builder.dns_resolver(Arc::new(Fixed(addr))),
                None => builder.dns_resolver(Arc::new(SystemResolver)),
            };
            if let Some(pem) = &connect.root_certificate {
                builder = builder.add_root_certificate(reqwest::Certificate::from_pem(pem)?);
            }
            Ok(builder.build()?)
        };
        Ok(Http { normal: build(TIMEOUT, CONNECT_TIMEOUT)?, patient: build(RETRY_TIMEOUT, RETRY_CONNECT_TIMEOUT)?, jar: Arc::new(Jar::default()) })
    }

    /// One GET; status 0 with `error` named as aiohttp names it when no whole answer came.
    pub async fn get(&self, url: &str, max_bytes: usize, patient: bool) -> Response {
        let parts = urls::split(url).unwrap_or_default();
        let scheme = parts.scheme.clone().unwrap_or_default();
        if scheme != "http" && scheme != "https" {
            return Response::failed(url, "NonHttpUrlClientError");
        }
        let Some(host) = parts.hostname() else {
            return Response::failed(url, "InvalidUrlClientError");
        };
        if blocklist::blocked_host(&host) {
            return Response::failed(url, net::BLOCKED);
        }
        let path = if parts.path.is_empty() { "/".to_string() } else { parts.path.clone() };
        let client = if patient { &self.patient } else { &self.normal };
        let connecting = if patient { RETRY_CONNECT_TIMEOUT } else { CONNECT_TIMEOUT };
        let wire = wire_url(&parts);
        let mut again = true;
        let (mut response, started) = loop {
            let mut request = client.get(&wire);
            if let Some(cookie) = self.jar.header(&scheme, &host, &path) {
                request = request.header(reqwest::header::COOKIE, cookie);
            }
            let started = Instant::now();
            match request.send().await {
                Ok(r) => break (r, started),
                Err(e) => {
                    let name = failure(&e, started.elapsed(), Some(connecting));
                    // aiohttp sends a GET once more when the server drops it before answering.
                    if again && matches!(name, "ServerDisconnectedError" | "ClientOSError") {
                        again = false;
                        continue;
                    }
                    return Response::failed(url, name);
                }
            }
        };
        let status = response.status().as_u16();
        let mut headers = HashMap::new();
        let mut set_cookie = Vec::new();
        for (name, value) in response.headers() {
            let value = String::from_utf8_lossy(value.as_bytes()).into_owned();
            if name == reqwest::header::SET_COOKIE {
                set_cookie.push(value.clone());
            }
            headers.insert(name.as_str().to_string(), value);
        }
        self.jar.store(&host, &path, &set_cookie);
        let mut body = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    body.extend_from_slice(&chunk);
                    if body.len() > max_bytes {
                        return Response { url: url.to_string(), status, headers, body: Vec::new(), fetched_at: Utc::now(), error: Some("too_large".into()) };
                    }
                }
                Ok(None) => break,
                Err(e) => return Response::failed(url, failure(&e, started.elapsed(), None)),
            }
        }
        Response { url: url.to_string(), status, headers, body, fetched_at: Utc::now(), error: None }
    }
}

/// aiohttp's names: a total timeout is `TimeoutError`, a connect timeout `ConnectionTimeoutError`.
fn failure(error: &reqwest::Error, elapsed: Duration, connecting: Option<Duration>) -> &'static str {
    if error.is_timeout() {
        return if error.is_connect() { "ConnectionTimeoutError" } else { "TimeoutError" };
    }
    net::failure(error, elapsed, connecting)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_as_python_codecs_do() {
        assert_eq!(decode(b"caf\xe9", Some("ISO-8859-1")), "café");
        assert_eq!(decode(b"\x80\x81", Some("windows-1252")), "€\u{fffd}");
        assert_eq!(decode(b"\x80\x81", Some("cp1252")), "€\u{fffd}");
        assert_eq!(decode(b"a\xe9", Some("us-ascii")), "a\u{fffd}");
        assert_eq!(decode(b"caf\xc3\xa9", Some("no-such-codec")), "café");
        assert_eq!(decode(b"caf\xc3\xa9\xff", None), "café\u{fffd}");
        assert_eq!(decode("日本".as_bytes(), Some("UTF8")), "日本");
    }

    #[test]
    fn wire_urls_as_yarl_requotes_them() {
        let wire = |url: &str| wire_url(&urls::split(url).unwrap());
        assert_eq!(wire("https://a.test/authorize?back=%2Fart"), "https://a.test/authorize?back=/art");
        assert_eq!(
            wire("https://a.test/A%20b/%41%2f%2B%3a/caf\u{e9}/x y?q=a%2Bb%3D&s=%7e+1&t=x y"),
            "https://a.test/A%20b/A%2F%2B:/caf%C3%A9/x%20y?q=a%2Bb%3D&s=~+1&t=x+y"
        );
        assert_eq!(wire("https://a.test/a/./b/../c%zz%4"), "https://a.test/a/c%25zz%254");
        assert_eq!(wire("https://a.test"), "https://a.test/");
    }
}

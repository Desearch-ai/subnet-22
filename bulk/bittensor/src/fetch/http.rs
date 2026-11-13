//! The fetch's HTTP session as aiohttp runs it: one cookie jar, gzip and deflate, a 30 s total per request, redirects followed by yarl's rules.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, LOCATION, USER_AGENT};

use super::yarl::Url;

pub const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REDIRECTS: usize = 10;
const REDIRECTS: [u16; 5] = [301, 302, 303, 307, 308];

/// A response as `get` returns it: status, final URL, content type and the decoded body.
#[derive(Clone, Debug)]
pub struct Got {
    pub status: u16,
    pub final_url: String,
    pub content_type: String,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

/// A failed request, named as the aiohttp exception it stands for; `repr` is what Python's `repr(error)` would begin with.
#[derive(Clone, Debug)]
pub struct Failure {
    pub class: &'static str,
    pub repr: String,
}

impl Failure {
    fn new(class: &'static str, repr: String) -> Failure {
        Failure { class, repr }
    }

    pub fn timeout() -> Failure {
        Failure::new("TimeoutError", "TimeoutError()".into())
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.repr)
    }
}

impl std::error::Error for Failure {}

struct ResolveAll(SocketAddr);

impl reqwest::dns::Resolve for ResolveAll {
    fn resolve(&self, _: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let addr = self.0;
        Box::pin(async move { Ok(Box::new(std::iter::once(addr)) as reqwest::dns::Addrs) })
    }
}

/// Where the session's connections go in tests: every host to one address, trusting one more CA.
#[derive(Clone, Debug, Default)]
pub struct TestRoute {
    pub resolve_all: Option<SocketAddr>,
    pub ca_cert: Option<std::path::PathBuf>,
}

pub struct Http {
    client: reqwest::Client,
}

fn header_text(value: &HeaderValue) -> String {
    match value.to_str() {
        Ok(text) => text.to_string(),
        Err(_) => String::from_utf8_lossy(value.as_bytes()).into_owned(),
    }
}

fn class_of(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "TimeoutError"
    } else if error.is_connect() {
        "ClientConnectorError"
    } else if error.is_body() || error.is_decode() {
        "ClientPayloadError"
    } else {
        "ClientOSError"
    }
}

fn failure(error: reqwest::Error, url: &Url) -> Failure {
    let mut source: Option<&dyn std::error::Error> = Some(&error);
    while let Some(cause) = source {
        if cause.to_string().contains("connection closed before message completed") {
            return Failure::new("ServerDisconnectedError", "ServerDisconnectedError('Server disconnected')".into());
        }
        source = cause.source();
    }
    let class = class_of(&error);
    if class == "TimeoutError" {
        return Failure::timeout();
    }
    if class == "ClientConnectorError" {
        let ssl = url.scheme == "https";
        let port = url.netloc.rsplit_once(':').and_then(|(_, p)| p.parse::<u16>().ok()).unwrap_or(if ssl { 443 } else { 80 });
        let ssl_text = if ssl { "True" } else { "False" };
        return Failure::new(
            class,
            format!(
                "ClientConnectorError(ConnectionKey(host='{}', port={port}, is_ssl={ssl_text}, ssl=True, proxy=None, proxy_auth=None, proxy_headers_hash=None), {})",
                url.raw_host(),
                std::error::Error::source(&error).map_or_else(|| error.to_string(), |s| s.to_string())
            ),
        );
    }
    Failure::new(class, format!("{class}({:?})", error.to_string()))
}

impl Http {
    pub fn new(user_agent: &str, route: &TestRoute) -> Result<Http> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_str(user_agent)?);
        let mut builder = reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .cookie_store(true)
            .gzip(true)
            .deflate(true)
            .timeout(TIMEOUT);
        if let Some(addr) = route.resolve_all {
            builder = builder.dns_resolver(Arc::new(ResolveAll(addr)));
        }
        if let Some(path) = &route.ca_cert {
            let pem = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            builder = builder.add_root_certificate(reqwest::Certificate::from_pem(&pem)?);
        }
        Ok(Http { client: builder.build()? })
    }

    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    /// `session.get(url, allow_redirects=True)` and `response.read()`, all within the total timeout.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> std::result::Result<Got, Failure> {
        match tokio::time::timeout(TIMEOUT, self.follow(url, headers)).await {
            Ok(done) => done,
            Err(_) => Err(Failure::timeout()),
        }
    }

    async fn follow(&self, text: &str, headers: &[(&str, &str)]) -> std::result::Result<Got, Failure> {
        let mut url = Url::parse(text).map_err(|_| Failure::new("InvalidUrlClientError", format!("<InvalidUrlClientError {text}>")))?;
        if url.scheme != "http" && url.scheme != "https" {
            return Err(Failure::new("NonHttpUrlClientError", format!("<NonHttpUrlClientError {url}>")));
        }
        let mut redirects = 0;
        loop {
            if url.raw_host().is_empty() {
                let class = if redirects > 0 { "InvalidUrlRedirectClientError" } else { "InvalidUrlClientError" };
                return Err(Failure::new(class, format!("<{class} {url}>")));
            }
            let target = url.without_fragment().to_string();
            let mut request = self.client.get(&target);
            for (name, value) in headers {
                request = request.header(*name, *value);
            }
            let response = request.send().await.map_err(|e| failure(e, &url))?;
            let status = response.status().as_u16();
            if REDIRECTS.contains(&status) {
                redirects += 1;
                if redirects >= MAX_REDIRECTS {
                    return Err(Failure::new("TooManyRedirects", format!("TooManyRedirects(RequestInfo(url=URL('{text}'), method='GET'")));
                }
                let location = response.headers().get(LOCATION).or_else(|| response.headers().get("uri")).map(header_text);
                if let Some(location) = location {
                    let mut next = Url::parse(&location).map_err(|_| {
                        Failure::new(
                            "InvalidUrlRedirectClientError",
                            format!("<InvalidUrlRedirectClientError {location} - Server attempted redirecting to a location that does not look like a URL>"),
                        )
                    })?;
                    if !next.scheme.is_empty() && next.scheme != "http" && next.scheme != "https" {
                        return Err(Failure::new("NonHttpUrlRedirectClientError", format!("<NonHttpUrlRedirectClientError {location}>")));
                    }
                    if next.scheme.is_empty() {
                        next = url.join(&next);
                    }
                    if next.raw_host().is_empty() {
                        return Err(Failure::new(
                            "InvalidUrlRedirectClientError",
                            format!("<InvalidUrlRedirectClientError {next} - Invalid redirect URL origin>"),
                        ));
                    }
                    url = next;
                    continue;
                }
            }
            let content_type = response.headers().get(CONTENT_TYPE).map(header_text).unwrap_or_default();
            let headers = response.headers().clone();
            let body = response.bytes().await.map_err(|e| failure(e, &url))?.to_vec();
            return Ok(Got { status, final_url: url.without_fragment().to_string(), content_type, headers, body });
        }
    }
}

/// `response.text()`: the body decoded by its charset, UTF-8 when it names none; anything else fails as aiohttp's strict decode does.
pub fn text_of(got: &Got) -> Result<String> {
    let charset = got
        .content_type
        .split(';')
        .skip(1)
        .filter_map(|p| p.split_once('='))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("charset"))
        .map(|(_, v)| v.trim().trim_matches('"').to_ascii_lowercase());
    match charset.as_deref() {
        None | Some("utf-8") | Some("utf8") => String::from_utf8(got.body.clone()).context("UnicodeDecodeError: the body is not UTF-8"),
        Some(other) => anyhow::bail!("a body in {other}, which this fetch does not decode"),
    }
}

//! Cloudflare R2 over the S3 API: SigV4-signed HEAD, ranged GET, PUT, DELETE and multipart uploads, retried with backoff.

use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Method, StatusCode};
use sha2::{Digest, Sha256};
use tokio::runtime::Handle;

use crate::canonical::hex;
use crate::reading::RangeRead;
use crate::records::iso;

pub const PARQUET: &str = "application/vnd.apache.parquet";
pub const JSON: &str = "application/json";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const UNSIGNED: &str = "UNSIGNED-PAYLOAD";
const ATTEMPTS: u32 = 5;
const FIRST_BACKOFF: Duration = Duration::from_millis(250);
const SHORT: Duration = Duration::from_secs(30);
/// A ranged read or HEAD answers in well under a second; one that hangs is retried instead of waited out.
const READ: Duration = Duration::from_secs(10);
const LONG: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub region: String,
}

/// Why a storage call failed.
#[derive(Debug)]
pub enum Error {
    Missing,
    /// The object is not the version asked for.
    Changed,
    Status(u16, String),
    Transport(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing => f.write_str("no such key"),
            Error::Changed => f.write_str("the object changed"),
            Error::Status(status, body) => write!(f, "HTTP {status}: {}", body.chars().take(300).collect::<String>()),
            Error::Transport(error) => f.write_str(error),
        }
    }
}

impl std::error::Error for Error {}

/// A stored object's size and entity tag, quotes included as R2 sends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub size: u64,
    pub etag: String,
}

/// One bucket under one key prefix.
#[derive(Clone)]
pub struct Bucket {
    http: reqwest::Client,
    endpoint: String,
    host: String,
    pub bucket: String,
    pub prefix: String,
    credentials: Arc<Credentials>,
    /// Failed attempts, retried or not, for the minute summary.
    pub errors: Arc<AtomicU64>,
}

pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(512)
        .pool_idle_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(15))
        .build()?)
}

impl Bucket {
    pub fn new(http: reqwest::Client, endpoint: &str, bucket: &str, prefix: &str, credentials: Credentials, errors: Arc<AtomicU64>) -> Result<Self> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let url = reqwest::Url::parse(&endpoint).with_context(|| format!("CF_R2_ENDPOINT {endpoint:?} is not a URL"))?;
        let host = match (url.host_str(), url.port()) {
            (Some(host), Some(port)) => format!("{host}:{port}"),
            (Some(host), None) => host.to_string(),
            (None, _) => return Err(anyhow!("CF_R2_ENDPOINT {endpoint:?} has no host")),
        };
        Ok(Bucket { http, endpoint, host, bucket: bucket.into(), prefix: prefix.into(), credentials: Arc::new(credentials), errors })
    }

    pub fn path(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    /// The bucket answers and these credentials may use it.
    pub async fn check(&self) -> Result<()> {
        self.call(Method::HEAD, None, &[], HeaderMap::new(), Bytes::new(), SHORT).await.with_context(|| format!("bucket {}", self.bucket))?;
        Ok(())
    }

    pub async fn head(&self, key: &str) -> Result<Option<Head>, Error> {
        match self.call(Method::HEAD, Some(key), &[], HeaderMap::new(), Bytes::new(), READ).await {
            Ok((headers, _)) => {
                let text = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
                let size = text("content-length").parse().map_err(|_| Error::Transport("a HEAD without a length".into()))?;
                Ok(Some(Head { size, etag: text("etag") }))
            }
            Err(Error::Missing) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub async fn exists(&self, key: &str) -> Result<bool, Error> {
        Ok(self.head(key).await?.is_some())
    }

    /// Bytes `range` of an object, refused as `Changed` when it is no longer the version `if_match` names.
    pub async fn get_range(&self, key: &str, range: Range<u64>, if_match: Option<&str>) -> Result<Bytes, Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::RANGE, value(&format!("bytes={}-{}", range.start, range.end - 1)));
        if let Some(etag) = if_match {
            headers.insert(reqwest::header::IF_MATCH, value(etag));
        }
        let (_, body) = self.call(Method::GET, Some(key), &[], headers, Bytes::new(), READ).await?;
        if body.len() as u64 != range.end - range.start {
            return Err(Error::Transport(format!("asked for {} bytes of {key}, got {}", range.end - range.start, body.len())));
        }
        Ok(body)
    }

    pub async fn get(&self, key: &str) -> Result<Bytes, Error> {
        Ok(self.call(Method::GET, Some(key), &[], HeaderMap::new(), Bytes::new(), LONG).await?.1)
    }

    pub async fn put(&self, key: &str, body: Bytes, content_type: &str, cache_control: Option<&str>) -> Result<(), Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, value(content_type));
        if let Some(cache_control) = cache_control {
            headers.insert(reqwest::header::CACHE_CONTROL, value(cache_control));
        }
        self.call(Method::PUT, Some(key), &[], headers, body, LONG).await?;
        Ok(())
    }

    /// A large object as parts sent `at_once`, since one stream to R2 tops out far below the link.
    pub async fn put_in_parts(&self, key: &str, body: Bytes, content_type: &str, part_bytes: usize, at_once: usize) -> Result<(), Error> {
        if body.len() <= part_bytes {
            return self.put(key, body, content_type, None).await;
        }
        let upload_id = self.start_multipart(key, content_type).await?;
        let parts = (0..body.len()).step_by(part_bytes).zip(1u32..).map(|(start, number)| (number, body.slice(start..(start + part_bytes).min(body.len()))));
        let sent: Result<Vec<(u32, String)>, Error> = futures::stream::iter(parts)
            .map(|(number, part)| {
                let upload_id = &upload_id;
                async move { Ok((number, self.upload_part(key, upload_id, number, part).await?)) }
            })
            .buffered(at_once.max(1))
            .try_collect()
            .await;
        let finished = match sent {
            Ok(parts) => self.complete_multipart(key, &upload_id, &parts).await,
            Err(error) => Err(error),
        };
        if finished.is_err() {
            let _ = self.abort_multipart(key, &upload_id).await;
        }
        finished
    }

    pub async fn delete(&self, key: &str) -> Result<(), Error> {
        match self.call(Method::DELETE, Some(key), &[], HeaderMap::new(), Bytes::new(), SHORT).await {
            Ok(_) | Err(Error::Missing) => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub async fn start_multipart(&self, key: &str, content_type: &str) -> Result<String, Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, value(content_type));
        let (_, body) = self.call(Method::POST, Some(key), &[("uploads", "")], headers, Bytes::new(), SHORT).await?;
        xml_field(&body, "UploadId").ok_or_else(|| Error::Transport("a multipart upload without an UploadId".into()))
    }

    /// One part of a multipart upload; returns its entity tag.
    pub async fn upload_part(&self, key: &str, upload_id: &str, number: u32, body: Bytes) -> Result<String, Error> {
        let number = number.to_string();
        let (headers, _) = self.call(Method::PUT, Some(key), &[("partNumber", &number), ("uploadId", upload_id)], HeaderMap::new(), body, LONG).await?;
        Ok(headers.get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_string())
    }

    pub async fn complete_multipart(&self, key: &str, upload_id: &str, parts: &[(u32, String)]) -> Result<(), Error> {
        let mut xml = String::from("<CompleteMultipartUpload>");
        for (number, etag) in parts {
            xml.push_str(&format!("<Part><PartNumber>{number}</PartNumber><ETag>{etag}</ETag></Part>"));
        }
        xml.push_str("</CompleteMultipartUpload>");
        let (_, body) = self.call(Method::POST, Some(key), &[("uploadId", upload_id)], HeaderMap::new(), Bytes::from(xml), LONG).await?;
        // S3 can answer 200 and still report the failure in the body.
        if xml_field(&body, "Code").is_some() {
            return Err(Error::Status(200, String::from_utf8_lossy(&body).into_owned()));
        }
        Ok(())
    }

    pub async fn abort_multipart(&self, key: &str, upload_id: &str) -> Result<(), Error> {
        self.call(Method::DELETE, Some(key), &[("uploadId", upload_id)], HeaderMap::new(), Bytes::new(), SHORT).await?;
        Ok(())
    }

    async fn call(&self, method: Method, key: Option<&str>, query: &[(&str, &str)], headers: HeaderMap, body: Bytes, timeout: Duration) -> Result<(HeaderMap, Bytes), Error> {
        let path = match key {
            Some(key) => format!("/{}/{}", self.bucket, encode(&self.path(key), false)),
            None => format!("/{}", self.bucket),
        };
        let mut pairs: Vec<(String, String)> = query.iter().map(|(k, v)| (encode(k, true), encode(v, true))).collect();
        pairs.sort();
        let query = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
        let url = if query.is_empty() { format!("{}{path}", self.endpoint) } else { format!("{}{path}?{query}", self.endpoint) };
        let payload = if body.is_empty() { EMPTY_SHA256 } else { UNSIGNED };
        let mut attempt = 0;
        loop {
            let amz_date = amz_date(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default());
            let authorization = sign(method.as_str(), &self.host, &path, &query, payload, &amz_date, &self.credentials);
            let request = self
                .http
                .request(method.clone(), &url)
                .headers(headers.clone())
                .header("x-amz-date", &amz_date)
                .header("x-amz-content-sha256", payload)
                .header(reqwest::header::AUTHORIZATION, authorization)
                .timeout(timeout)
                .body(body.clone());
            let failure = match request.send().await {
                Ok(response) => {
                    let status = response.status();
                    let headers = response.headers().clone();
                    match response.bytes().await {
                        Ok(bytes) if status.is_success() => return Ok((headers, bytes)),
                        Ok(_) if status == StatusCode::NOT_FOUND => return Err(Error::Missing),
                        Ok(_) if status == StatusCode::PRECONDITION_FAILED => return Err(Error::Changed),
                        Ok(bytes) if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS => {
                            Error::Status(status.as_u16(), String::from_utf8_lossy(&bytes).into_owned())
                        }
                        Ok(bytes) => {
                            self.errors.fetch_add(1, Ordering::Relaxed);
                            return Err(Error::Status(status.as_u16(), String::from_utf8_lossy(&bytes).into_owned()));
                        }
                        Err(error) => Error::Transport(format!("{method} {path}: {error}")),
                    }
                }
                Err(error) => Error::Transport(format!("{method} {path}: {error}")),
            };
            self.errors.fetch_add(1, Ordering::Relaxed);
            attempt += 1;
            if attempt >= ATTEMPTS {
                return Err(failure);
            }
            tokio::time::sleep(backoff(attempt)).await;
        }
    }
}

fn backoff(attempt: u32) -> Duration {
    let base = FIRST_BACKOFF * 2u32.pow(attempt - 1);
    let jitter = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().subsec_nanos() % 1000;
    base + base * jitter / 2000
}

fn value(text: &str) -> HeaderValue {
    HeaderValue::from_str(text).unwrap_or_else(|_| HeaderValue::from_static(""))
}

fn xml_field(body: &[u8], name: &str) -> Option<String> {
    let text = std::str::from_utf8(body).ok()?;
    let start = text.find(&format!("<{name}>"))? + name.len() + 2;
    let end = text[start..].find(&format!("</{name}>"))? + start;
    Some(text[start..end].to_string())
}

/// SigV4's URI encoding: everything but unreserved characters, and slashes too unless `slash` is false.
pub fn encode(text: &str, slash: bool) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(text.len() + 8);
    for &b in text.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') || (b == b'/' && !slash) {
            out.push(b as char);
        } else {
            out.extend(['%', HEX[(b >> 4) as usize] as char, HEX[(b & 15) as usize] as char]);
        }
    }
    out
}

/// The `x-amz-date` for a moment since the epoch, as `20261006T221000Z`.
pub fn amz_date(since_epoch: Duration) -> String {
    let t = iso(since_epoch.as_micros() as i64);
    format!("{}{}{}T{}{}{}Z", &t[0..4], &t[5..7], &t[8..10], &t[11..13], &t[14..16], &t[17..19])
}

/// The `Authorization` header for a request signed over its host, payload hash and date.
pub fn sign(method: &str, host: &str, path: &str, query: &str, payload: &str, amz_date: &str, credentials: &Credentials) -> String {
    const SIGNED: &str = "host;x-amz-content-sha256;x-amz-date";
    let canonical = format!("{method}\n{path}\n{query}\nhost:{host}\nx-amz-content-sha256:{payload}\nx-amz-date:{amz_date}\n\n{SIGNED}\n{payload}");
    let day = &amz_date[..8];
    let scope = format!("{day}/{}/s3/aws4_request", credentials.region);
    let to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", hex(&Sha256::digest(canonical.as_bytes())));
    let mut key = hmac(format!("AWS4{}", credentials.secret_key).as_bytes(), day.as_bytes());
    for part in [credentials.region.as_str(), "s3", "aws4_request"] {
        key = hmac(&key, part.as_bytes());
    }
    let signature = hex(&hmac(&key, to_sign.as_bytes()));
    format!("AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={SIGNED}, Signature={signature}", credentials.access_key)
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// An upload read in ranges from R2, several ranges at once, each refused if the object changed since its HEAD.
pub struct Remote {
    pub bucket: Bucket,
    pub key: String,
    pub head: Head,
    pub handle: Handle,
    pub ranges_at_once: usize,
    pub traffic: Arc<crate::reading::Traffic>,
}

impl Remote {
    fn count(&self, ranges: &[Range<u64>]) {
        self.traffic.requests.fetch_add(ranges.len() as u64, Ordering::Relaxed);
        self.traffic.bytes.fetch_add(ranges.iter().map(|r| r.end - r.start).sum(), Ordering::Relaxed);
    }
}

impl RangeRead for Remote {
    fn size(&self) -> u64 {
        self.head.size
    }

    fn read(&self, range: Range<u64>) -> io::Result<Bytes> {
        self.count(std::slice::from_ref(&range));
        let etag = Some(self.head.etag.as_str()).filter(|e| !e.is_empty());
        self.handle.block_on(self.bucket.get_range(&self.key, range, etag)).map_err(io::Error::other)
    }

    fn read_many(&self, ranges: &[Range<u64>]) -> io::Result<Vec<Bytes>> {
        self.count(ranges);
        let etag = Some(self.head.etag.as_str()).filter(|e| !e.is_empty());
        let reads = futures::stream::iter(ranges.iter().cloned()).map(|range| self.bucket.get_range(&self.key, range, etag)).buffered(self.ranges_at_once.max(1));
        self.handle.block_on(reads.try_collect()).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_match_botocore() {
        let credentials = Credentials { access_key: "AKIDEXAMPLE".into(), secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(), region: "auto".into() };
        let host = "acct.r2.cloudflarestorage.com";
        let cases = [
            ("GET", "/subnet-22/submitted/dt%3D2026-10-06/task%3Dabc/5C-1-x.parquet", "", "87280451a346b94fe40653477628deb526bd967a022c8a9e2c3b256883ba70d1"),
            ("POST", "/desearch-pages/index/snapshots/2026-10-06.parquet", "uploads=", "8b4dabbb5d023ba72e5255cc47fecca67d5c5839757df107b7ff9e5759b0dc9d"),
            ("PUT", "/desearch-pages/index/snapshots/2026-10-06.parquet", "partNumber=2&uploadId=a%2Fb%2Bc", "898408ff584294933f157dd8f6aee783f67851ae6d761b168e588310076dd6a3"),
            ("HEAD", "/subnet-22/a%20b/%C3%A9~_.-", "", "ee7189b1f9615bc4c1c43d68e4f3f79a6c71acf1ad6b54d69866fb860f2332b0"),
        ];
        let date = amz_date(Duration::from_secs(1_791_324_600));
        assert_eq!(date, "20261006T221000Z");
        for (method, path, query, signature) in cases {
            let header = sign(method, host, path, query, EMPTY_SHA256, &date, &credentials);
            assert!(header.ends_with(&format!("Signature={signature}")), "{method} {path}: {header}");
        }
        assert_eq!(encode("submitted/dt=2026-10-06/a b/\u{e9}~", false), "submitted/dt%3D2026-10-06/a%20b/%C3%A9~");
        assert_eq!(encode("a/b+c", true), "a%2Fb%2Bc");
    }

    #[test]
    fn a_client_builds() {
        client().unwrap();
    }
}

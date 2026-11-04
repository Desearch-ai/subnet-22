//! What the API proves about itself inside a dstack CVM: a key made inside it, and a TDX quote binding that key, its settings and its TLS certificate.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use desearch::canonical::hex;
use desearch::hotkey::Hotkey;
use rand::RngCore;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256, Sha512};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::py::canonical;

pub const DOCUMENT_KIND: &str = "sn22-task-api-attestation";
pub const DOCUMENT_VERSION: i64 = 1;
/// Prefixed to the document before hashing, so a quote made for this cannot be taken for another protocol's.
pub const REPORT_DATA_PREFIX: &[u8] = b"sn22-task-api-attestation-v1:";
/// The keys live on the CVM's encrypted disk, which only this CVM can open.
const KEY_DIR: &str = "cvm";
const SIGNER_FILE: &str = "signer.seed";
const HANDOVER_FILE: &str = "handover.key";
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_NONCE_BYTES: usize = 32;

/// The dstack guest agent on its Unix socket, spoken to over plain HTTP/1.1.
#[derive(Clone, Debug)]
pub struct Dstack {
    socket: PathBuf,
}

impl Dstack {
    pub fn new(socket: impl Into<PathBuf>) -> Dstack {
        Dstack { socket: socket.into() }
    }

    /// The app's identity and the measurements dstack reports for it; `tcb_info` holds the booted app compose.
    pub async fn info(&self) -> Result<Value> {
        self.call("/Info", None).await
    }

    /// A TDX quote over these 64 bytes of report data, with the event log that replays RTMR3.
    pub async fn quote(&self, report_data: &[u8; 64]) -> Result<Value> {
        self.call("/GetQuote", Some(&json!({"report_data": hex(report_data)}))).await
    }

    async fn call(&self, path: &str, body: Option<&Value>) -> Result<Value> {
        tokio::time::timeout(SOCKET_TIMEOUT, self.exchange(path, body))
            .await
            .map_err(|_| anyhow!("the dstack agent did not answer {path} within {}s", SOCKET_TIMEOUT.as_secs()))?
    }

    async fn exchange(&self, path: &str, body: Option<&Value>) -> Result<Value> {
        let mut stream = tokio::net::UnixStream::connect(&self.socket).await.with_context(|| format!("the dstack socket {:?}", self.socket))?;
        let payload = body.map(|b| b.to_string()).unwrap_or_default();
        let method = if body.is_some() { "POST" } else { "GET" };
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: dstack\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        stream.write_all(request.as_bytes()).await?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;
        let (status, body) = http_body(&raw)?;
        if status != 200 {
            bail!("the dstack agent answered {path} with {status}: {}", String::from_utf8_lossy(&body));
        }
        serde_json::from_slice(&body).with_context(|| format!("the dstack agent's {path} answer is not JSON"))
    }
}

/// The status and body of a whole HTTP/1.1 response, chunked or not.
fn http_body(raw: &[u8]) -> Result<(u16, Vec<u8>)> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").context("an HTTP response without a header end")?;
    let head = std::str::from_utf8(&raw[..split]).context("HTTP headers that are not text")?;
    let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).context("an HTTP response without a status")?;
    let mut body = raw[split + 4..].to_vec();
    if head.lines().any(|line| line.to_ascii_lowercase().replace(' ', "") == "transfer-encoding:chunked") {
        body = dechunk(&body)?;
    }
    Ok((status, body))
}

fn dechunk(mut raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let end = raw.windows(2).position(|w| w == b"\r\n").context("a chunk without a size line")?;
        let size = usize::from_str_radix(std::str::from_utf8(&raw[..end])?.split(';').next().unwrap_or("").trim(), 16).context("a chunk size")?;
        raw = &raw[end + 2..];
        if size == 0 {
            return Ok(out);
        }
        out.extend_from_slice(raw.get(..size).context("a chunk shorter than its size")?);
        raw = raw.get(size + 2..).unwrap_or_default();
    }
}

/// Reads a 32-byte secret kept under the data directory, or makes one from the OS's randomness the first time.
fn kept_secret(data: &Path, name: &str) -> Result<[u8; 32]> {
    let dir = data.join(KEY_DIR);
    let path = dir.join(name);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let bytes = decode_hex(text.trim()).with_context(|| format!("{path:?} is not hex"))?;
        return bytes.try_into().map_err(|_| anyhow!("{path:?} is not 32 bytes"));
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("{dir:?}"))?;
    let mut secret = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut secret);
    let partial = dir.join(format!("{name}.partial"));
    {
        use std::io::Write;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let _ = std::fs::remove_file(&partial);
        let mut file = options.open(&partial).with_context(|| format!("{partial:?}"))?;
        file.write_all(hex(&secret).as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(&partial, &path)?;
    Ok(secret)
}

/// The receipt key of a CVM: random, made at its first boot, never derived from the app's KMS key every copy of the app shares.
pub fn cvm_key(data: &Path) -> Result<Hotkey> {
    Hotkey::from_uri(&format!("0x{}", hex(&kept_secret(data, SIGNER_FILE)?)))
}

/// The X25519 key a predecessor encrypts its state to; its public half is in the attestation.
pub fn handover_key(data: &Path) -> Result<StaticSecret> {
    Ok(StaticSecret::from(kept_secret(data, HANDOVER_FILE)?))
}

pub fn report_data(document: &Value) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(REPORT_DATA_PREFIX);
    hasher.update(canonical(document));
    hasher.finalize().into()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The SHA-256 of the DER SubjectPublicKeyInfo of the first certificate in a PEM file.
pub fn spki_sha256(pem: &str) -> Result<String> {
    let start = pem.find("-----BEGIN CERTIFICATE-----").context("no certificate in the PEM")? + "-----BEGIN CERTIFICATE-----".len();
    let end = pem[start..].find("-----END CERTIFICATE-----").context("an unfinished certificate")? + start;
    let body: String = pem[start..end].chars().filter(|c| !c.is_whitespace()).collect();
    let der = base64::engine::general_purpose::STANDARD.decode(body).context("a certificate that is not base64")?;
    let (certificate, _) = der_item(&der)?;
    let (tbs, _) = der_item(der_content(certificate)?)?;
    let mut fields = der_content(tbs)?;
    let mut items = Vec::new();
    while !fields.is_empty() {
        let (item, rest) = der_item(fields)?;
        items.push(item);
        fields = rest;
    }
    // version [0] when present, serial, signature algorithm, issuer, validity, subject, then the key.
    let skip = usize::from(items.first().is_some_and(|item| item[0] == 0xa0));
    let spki = items.get(skip + 5).context("a certificate without a public key")?;
    Ok(sha256_hex(spki))
}

/// One DER item (tag, length and content) and what follows it.
fn der_item(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let first = *bytes.get(1).context("a DER item cut short")?;
    let (length, header) = if first < 0x80 {
        (first as usize, 2)
    } else {
        let count = (first & 0x7f) as usize;
        if count == 0 || count > 4 {
            bail!("a DER length of {count} bytes");
        }
        let digits = bytes.get(2..2 + count).context("a DER length cut short")?;
        (digits.iter().fold(0usize, |n, d| (n << 8) | *d as usize), 2 + count)
    };
    let end = header.checked_add(length).filter(|end| *end <= bytes.len()).context("a DER item longer than its bytes")?;
    Ok((&bytes[..end], &bytes[end..]))
}

fn der_content(item: &[u8]) -> Result<&[u8]> {
    let first = item[1];
    let header = if first < 0x80 { 2 } else { 2 + (first & 0x7f) as usize };
    item.get(header..).context("a DER item without content")
}

pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    let text = text.strip_prefix("0x").unwrap_or(text);
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

/// A nonce the caller may have quoted afresh: at most 32 bytes of hex.
pub fn nonce(text: &str) -> Result<String> {
    let bytes = decode_hex(text).filter(|b| !b.is_empty() && b.len() <= MAX_NONCE_BYTES).context("the nonce must be 1 to 32 bytes of hex")?;
    Ok(hex(&bytes))
}

/// What the CVM's attestation binds: its signer, the key it receives state with, its settings and its TLS key.
pub fn document(signer: &str, handover: &PublicKey, config: &Value, tls_spki_sha256: Option<String>, nonce: Option<&str>) -> Value {
    let mut document = Map::new();
    document.insert("kind".into(), DOCUMENT_KIND.into());
    document.insert("version".into(), DOCUMENT_VERSION.into());
    document.insert("signer".into(), signer.into());
    document.insert("handover_key".into(), hex(handover.as_bytes()).into());
    document.insert("config_sha256".into(), sha256_hex(&canonical(config)).into());
    document.insert("tls_spki_sha256".into(), tls_spki_sha256.map_or(Value::Null, Value::from));
    if let Some(nonce) = nonce {
        document.insert("nonce".into(), nonce.into());
    }
    Value::Object(document)
}

/// The answer of `/v1/attestation` from dstack's quote and info: the event log and app compose parsed out of the strings dstack wraps them in.
pub fn assemble(document: Value, report_data: &[u8; 64], quote: &Value, info: &Value) -> Result<Value> {
    let quoted = quote["report_data"].as_str().and_then(decode_hex).unwrap_or_default();
    if quoted.get(..64) != Some(&report_data[..]) {
        bail!("dstack quoted other report data than asked");
    }
    let parse = |value: &Value| match value {
        Value::String(text) => serde_json::from_str(text).unwrap_or(Value::Null),
        other => other.clone(),
    };
    let tcb = parse(&info["tcb_info"]);
    let field = |name: &str| info.get(name).cloned().unwrap_or(Value::Null);
    Ok(json!({
        "document": document,
        "report_data": hex(report_data),
        "quote": quote["quote"].as_str().context("dstack gave no quote")?,
        "event_log": parse(&quote["event_log"]),
        "vm_config": quote.get("vm_config").cloned().unwrap_or(Value::Null),
        "app_compose": tcb.get("app_compose").cloned().unwrap_or(Value::Null),
        "info": {
            "app_id": field("app_id"),
            "instance_id": field("instance_id"),
            "app_name": field("app_name"),
            "compose_hash": field("compose_hash"),
            "os_image_hash": field("os_image_hash"),
            "device_id": field("device_id"),
            "mrtd": tcb.get("mrtd").cloned().unwrap_or(Value::Null),
            "rtmr0": tcb.get("rtmr0").cloned().unwrap_or(Value::Null),
            "rtmr1": tcb.get("rtmr1").cloned().unwrap_or(Value::Null),
            "rtmr2": tcb.get("rtmr2").cloned().unwrap_or(Value::Null),
        },
    }))
}

/// A CVM's attestation side: the agent, the handover key, and the last quote made, as one takes a second or two.
pub struct Tee {
    pub dstack: Dstack,
    pub handover: StaticSecret,
    pub handover_public: PublicKey,
    tls_cert: String,
    info: tokio::sync::OnceCell<Value>,
    /// Held while a quote is made, so callers asking at once wait for one quote instead of making many.
    last: tokio::sync::Mutex<Option<([u8; 64], Value)>>,
}

impl Tee {
    pub fn open(data: &Path, dstack_socket: &str, tls_cert: &str) -> Result<Tee> {
        let handover = handover_key(data)?;
        Ok(Tee {
            dstack: Dstack::new(dstack_socket),
            handover_public: PublicKey::from(&handover),
            handover,
            tls_cert: tls_cert.to_string(),
            info: tokio::sync::OnceCell::new(),
            last: tokio::sync::Mutex::new(None),
        })
    }

    /// The served TLS certificate's key hash, None before the proxy has one.
    pub fn tls_spki_sha256(&self) -> Option<String> {
        if self.tls_cert.is_empty() {
            return None;
        }
        let pem = std::fs::read_to_string(&self.tls_cert).ok()?;
        spki_sha256(&pem).map_err(|error| eprintln!("the TLS certificate {:?} is unreadable: {error:#}", self.tls_cert)).ok()
    }

    pub async fn info(&self) -> Result<Value> {
        self.info.get_or_try_init(|| self.dstack.info()).await.cloned()
    }

    /// This CVM's attestation now: quoted again only when what it binds changed, or for a nonce.
    pub async fn attestation(&self, signer: &str, config: &Value, nonce: Option<&str>) -> Result<Value> {
        let document = document(signer, &self.handover_public, config, self.tls_spki_sha256(), nonce);
        let data = report_data(&document);
        let mut last = self.last.lock().await;
        if let Some((bound, answer)) = last.as_ref() {
            if *bound == data {
                return Ok(answer.clone());
            }
        }
        let quote = self.dstack.quote(&data).await?;
        let answer = assemble(document, &data, &quote, &self.info().await?)?;
        if nonce.is_none() {
            *last = Some((data, answer.clone()));
        }
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunked_answer_is_put_back_together() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"a\":\r\n3\r\n12}\r\n0\r\n\r\n";
        let (status, body) = http_body(raw).unwrap();
        assert_eq!((status, body.as_slice()), (200, b"{\"a\":12}".as_slice()));
        let plain = b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\n\r\nno";
        assert_eq!(http_body(plain).unwrap(), (404, b"no".to_vec()));
    }

    #[test]
    fn the_report_data_is_a_prefixed_sha512_of_the_canonical_document() {
        let document = json!({"version": 1, "kind": DOCUMENT_KIND, "signer": "5Grw"});
        let mut hasher = Sha512::new();
        hasher.update(b"sn22-task-api-attestation-v1:{\"kind\":\"sn22-task-api-attestation\",\"signer\":\"5Grw\",\"version\":1}");
        let expected: [u8; 64] = hasher.finalize().into();
        assert_eq!(report_data(&document), expected);
    }

    #[test]
    fn a_kept_key_is_made_once_and_read_back() {
        let dir = std::env::temp_dir().join(format!("task-api-cvm-{}", uuid::Uuid::new_v4().simple()));
        let first = cvm_key(&dir).unwrap().ss58();
        assert_eq!(cvm_key(&dir).unwrap().ss58(), first, "the same key after a restart");
        let other = std::env::temp_dir().join(format!("task-api-cvm-{}", uuid::Uuid::new_v4().simple()));
        assert_ne!(cvm_key(&other).unwrap().ss58(), first, "another CVM makes its own");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join(KEY_DIR).join(SIGNER_FILE)).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "only the API's user may read it");
        }
        for dir in [dir, other] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_nonce_is_short_hex() {
        assert_eq!(nonce("0xAB01").unwrap(), "ab01");
        assert!(nonce("").is_err());
        assert!(nonce("zz").is_err());
        assert!(nonce(&"00".repeat(33)).is_err());
    }

    /// Made with `openssl req -x509`: a v3 certificate on a named curve, and a v1 one (no version field) with explicit curve parameters.
    const V3: &str = "-----BEGIN CERTIFICATE-----
MIIBNzCB3aADAgECAgkA803wsqppjlkwCgYIKoZIzj0EAwIwFDESMBAGA1UEAwwJ
c24yMi50ZXN0MB4XDTI2MTEwNDA4MjUxNloXDTI3MTEwNDA4MjUxNlowFDESMBAG
A1UEAwwJc24yMi50ZXN0MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEhmquD5N9
7QwVwq/kZ5BDcXIk/mVZxNnDlX379wEDT+zoTLKwWjNrMk3EKvl9dzbAOv2xt7Oa
jOGTGfGG6dyl56MYMBYwFAYDVR0RBA0wC4IJc24yMi50ZXN0MAoGCCqGSM49BAMC
A0kAMEYCIQDG6h2Zx7V4Nji3B8pl31g2t5VQZuWuQl66xdgjskniXQIhAKtdIpX/
xiN5vTqzlQ5sWhTp7zT+lUaFlrkauW3gM5tk
-----END CERTIFICATE-----";
    const V1: &str = "-----BEGIN CERTIFICATE-----
MIICDDCCAbICCQCd3HIrcyN5zzAKBggqhkjOPQQDAjAUMRIwEAYDVQQDDAlzbjIy
LnRlc3QwHhcNMjYxMTA0MDgyNTA2WhcNMjcxMTA0MDgyNTA2WjAUMRIwEAYDVQQD
DAlzbjIyLnRlc3QwggFLMIIBAwYHKoZIzj0CATCB9wIBATAsBgcqhkjOPQEBAiEA
/////wAAAAEAAAAAAAAAAAAAAAD///////////////8wWwQg/////wAAAAEAAAAA
AAAAAAAAAAD///////////////wEIFrGNdiqOpPns+u9VXaYhrxlHQawzFOw9jvO
PD4n0mBLAxUAxJ02CIbnBJNqZnjhE50mt4GffpAEQQRrF9Hy4SxCR/i85uVjpEDy
dwN9gS3rM6D0oTlF2JjClk/jQuL+Gn+bjufrSnwPnhYrzjNXazFezsu2QGg3v1H1
AiEA/////wAAAAD//////////7zm+q2nF56E87nKwvxjJVECAQEDQgAEFKyxWgRU
Yq/3KdJ26UGYYGueV9CsH8KoEnyv/4OrX7xRXAhpTUpfg0lxjLocyjITO3n9ICik
Po3Snv7koIBUzjAKBggqhkjOPQQDAgNIADBFAiEA+Ek2AyYdvGRMhasbJiRC8w8x
1lU+3hvrNcklmH7QvqUCICjQEdfMHUKktu95NWFEl0keTuysS0QHwSJV0kPPhjF6
-----END CERTIFICATE-----";

    #[test]
    fn the_tls_key_hash_covers_the_certificates_public_key_only() {
        // `openssl x509 -pubkey -noout | openssl pkey -pubin -outform DER | shasum -a 256`
        assert_eq!(spki_sha256(V3).unwrap(), "438f9d36ebfc1cb619214f82b156daefac58ecb7dbc78ec931e79910270d9d88");
        assert_eq!(spki_sha256(V1).unwrap(), "19d72bbaffcb2bc4d10d6b2946261efe5b92f0d9a7c2858da819542a937b307a");
        assert!(spki_sha256("no certificate").is_err());
    }
}

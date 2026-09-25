//! Web Bot Auth: sign requests with Ed25519 so a host can prove the traffic is ours.

use anyhow::Context;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use ed25519_dalek::pkcs8::DecodePrivateKey;
use ed25519_dalek::{Signer as _, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::urls;

pub const USER_AGENT: &str = "Mozilla/5.0 (compatible; DesearchBot/1.0; +https://www.desearch.ai/crawler)";
pub const SIGNATURE_AGENT: &str = "https://www.desearch.ai/crawler";
const LABEL: &str = "sig1";
const TAG: &str = "web-bot-auth";
const LIFETIME: i64 = 300;

/// The @authority component: host, lowercased, without userinfo or a default port.
pub fn authority(url: &str) -> String {
    let parts = urls::split(url).unwrap_or_default();
    let netloc = parts.netloc();
    let host = netloc.rsplit_once('@').map_or(netloc, |(_, host)| host).to_lowercase();
    let port = match parts.scheme.as_deref() {
        Some("https") => ":443",
        Some("http") => ":80",
        _ => return host,
    };
    host.strip_suffix(port).map(str::to_string).unwrap_or(host)
}

/// RFC 7638 JWK thumbprint, the keyid a verifier looks up in our directory.
pub fn thumbprint(key: &VerifyingKey) -> String {
    let x = URL_SAFE_NO_PAD.encode(key.as_bytes());
    let canonical = format!(r#"{{"crv":"Ed25519","kty":"OKP","x":"{x}"}}"#);
    URL_SAFE_NO_PAD.encode(Sha256::digest(canonical.as_bytes()))
}

pub struct Signer {
    key: SigningKey,
    agent: String,
    keyid: String,
}

impl Signer {
    pub fn new(key: SigningKey) -> Self {
        let keyid = thumbprint(&key.verifying_key());
        Signer { key, agent: format!("\"{SIGNATURE_AGENT}\""), keyid }
    }

    pub fn load(pem: &str) -> anyhow::Result<Self> {
        let key = SigningKey::from_pkcs8_pem(pem).map_err(|e| anyhow::anyhow!("signing key must be Ed25519 PKCS#8: {e}"))?;
        Ok(Signer::new(key))
    }

    /// Read the key from DESEARCH_SIGNING_KEY_FILE, or inline from DESEARCH_SIGNING_KEY.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        if let Some(path) = std::env::var_os("DESEARCH_SIGNING_KEY_FILE").filter(|p| !p.is_empty()) {
            let pem = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.to_string_lossy()))?;
            return Signer::load(&pem).map(Some);
        }
        match std::env::var("DESEARCH_SIGNING_KEY") {
            Ok(pem) if !pem.is_empty() => Signer::load(&pem.replace("\\n", "\n")).map(Some),
            _ => Ok(None),
        }
    }

    /// The three headers that sign a request to this URL, created at this Unix time.
    pub fn headers(&self, url: &str, created: i64) -> [(&'static str, String); 3] {
        let params = format!(
            "(\"@authority\" \"signature-agent\");created={created};keyid=\"{}\";alg=\"ed25519\";expires={};tag=\"{TAG}\"",
            self.keyid,
            created + LIFETIME
        );
        let base = format!(
            "\"@authority\": {}\n\"signature-agent\": {}\n\"@signature-params\": {params}",
            authority(url),
            self.agent
        );
        let signature = STANDARD.encode(self.key.sign(base.as_bytes()).to_bytes());
        [
            ("Signature-Agent", self.agent.clone()),
            ("Signature-Input", format!("{LABEL}={params}")),
            ("Signature", format!("{LABEL}=:{signature}:")),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_drops_userinfo_and_default_port() {
        assert_eq!(authority("https://user@Example.COM:443/a"), "example.com");
        assert_eq!(authority("http://example.com:8080/a"), "example.com:8080");
    }
}

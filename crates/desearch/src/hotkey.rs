//! sr25519 hotkeys as bittensor's `Keypair` handles them: derived from a Substrate secret URI, addressed in SS58, signing and verifying.

use anyhow::{bail, Context, Result};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Blake2b512, Digest};
use schnorrkel::derive::{ChainCode, Derivation};
use schnorrkel::{signing_context, ExpansionMode, Keypair, MiniSecretKey, PublicKey, SecretKey, Signature};
use sha2::Sha256;
use unicode_normalization::UnicodeNormalization;

const DEV_PHRASE: &str = "bottom drive obey lake curtain smoke basket hold race lonely fit walk";
const SIGNING_CONTEXT: &[u8] = b"substrate";
const SS58_PREFIX: u8 = 42;

pub struct Hotkey {
    pair: Keypair,
}

impl Hotkey {
    /// A key from `phrase//hard/soft///password`, where the phrase is a BIP39 mnemonic or a 0x-prefixed 32-byte seed, and the dev phrase when left out.
    pub fn from_uri(uri: &str) -> Result<Self> {
        let SecretUri { phrase, junctions, password } = SecretUri::parse(uri)?;
        let mini = match phrase.strip_prefix("0x") {
            Some(hex) => {
                let seed = decode_hex(hex).context("the seed is not hex")?;
                MiniSecretKey::from_bytes(&seed).map_err(|_| anyhow::anyhow!("the seed is not 32 bytes"))?
            }
            None => {
                let phrase: String = phrase.nfkd().collect();
                let mnemonic = bip39::Mnemonic::parse_in_normalized(bip39::Language::English, &phrase).context("not a BIP39 phrase")?;
                substrate_bip39::mini_secret_from_entropy(&mnemonic.to_entropy(), password.unwrap_or("")).map_err(|_| anyhow::anyhow!("not a BIP39 phrase"))?
            }
        };
        let mut secret: SecretKey = mini.expand(ExpansionMode::Ed25519);
        for (hard, code) in junctions {
            secret = if hard {
                secret.hard_derive_mini_secret_key(Some(ChainCode(code)), b"").0.expand(ExpansionMode::Ed25519)
            } else {
                secret.derived_key_simple(ChainCode(code), []).0
            };
        }
        Ok(Hotkey { pair: secret.to_keypair() })
    }

    pub fn public(&self) -> [u8; 32] {
        self.pair.public.to_bytes()
    }

    pub fn ss58(&self) -> String {
        ss58(&self.public())
    }

    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.pair.sign(signing_context(SIGNING_CONTEXT).bytes(message)).to_bytes()
    }
}

/// A signature over `message` itself or over `<Bytes>message</Bytes>`, which bittensor accepts too.
pub fn verify(public: &[u8], message: &[u8], signature: &[u8]) -> bool {
    let (Ok(public), Ok(signature)) = (PublicKey::from_bytes(public), Signature::from_bytes(signature)) else {
        return false;
    };
    if public.verify_simple(SIGNING_CONTEXT, message, &signature).is_ok() {
        return true;
    }
    let wrapped = [b"<Bytes>".as_slice(), message, b"</Bytes>"].concat();
    public.verify_simple(SIGNING_CONTEXT, &wrapped, &signature).is_ok()
}

/// The public key an SS58 address names, whatever its network prefix, when its checksum holds.
pub fn public_of(address: &str) -> Option<[u8; 32]> {
    let raw = bs58::decode(address).into_vec().ok()?;
    let prefix = match raw.first()? {
        0..=63 => 1,
        64..=127 => 2,
        _ => return None,
    };
    if raw.len() != prefix + 32 + 2 {
        return None;
    }
    let checksum = Blake2b512::new().chain_update(b"SS58PRE").chain_update(&raw[..prefix + 32]).finalize();
    if checksum[..2] != raw[prefix + 32..] {
        return None;
    }
    raw[prefix..prefix + 32].try_into().ok()
}

/// A public key as a generic Substrate SS58 address.
pub fn ss58(public: &[u8; 32]) -> String {
    let mut payload = Vec::with_capacity(35);
    payload.push(SS58_PREFIX);
    payload.extend_from_slice(public);
    let checksum = Blake2b512::new().chain_update(b"SS58PRE").chain_update(&payload).finalize();
    payload.extend_from_slice(&checksum[..2]);
    bs58::encode(payload).into_string()
}

/// What the task API's signature covers, as `desearch.client.TaskApiClient` builds it.
pub fn signing_payload(method: &str, path: &str, body: &[u8], timestamp: &str, nonce: &str) -> Vec<u8> {
    let digest: String = Sha256::digest(body).iter().map(|b| format!("{b:02x}")).collect();
    format!("{method}\n{path}\n{digest}\n{timestamp}\n{nonce}").into_bytes()
}

/// The four auth headers of one request.
pub fn auth_headers(hotkey: &Hotkey, method: &str, path: &str, body: &[u8], timestamp: i64, nonce: &str) -> [(&'static str, String); 4] {
    let timestamp = timestamp.to_string();
    let signature = hotkey.sign(&signing_payload(method, path, body, &timestamp, nonce));
    [
        ("X-Hotkey", hotkey.ss58()),
        ("X-Timestamp", timestamp),
        ("X-Nonce", nonce.to_string()),
        ("X-Signature", signature.iter().map(|b| format!("{b:02x}")).collect()),
    ]
}

struct SecretUri<'a> {
    phrase: &'a str,
    /// Hard or soft, with the chain code of each.
    junctions: Vec<(bool, [u8; 32])>,
    password: Option<&'a str>,
}

impl<'a> SecretUri<'a> {
    fn parse(uri: &'a str) -> Result<Self> {
        let (rest, password) = match uri.find("///") {
            Some(at) => (&uri[..at], Some(&uri[at + 3..])),
            None => (uri, None),
        };
        let (phrase, path) = rest.find('/').map_or((rest, ""), |at| rest.split_at(at));
        let phrase = if phrase.is_empty() { DEV_PHRASE } else { phrase };
        let mut junctions = Vec::new();
        let mut parts = path.split('/').skip(1);
        while let Some(part) = parts.next() {
            let hard = part.is_empty();
            let code = if hard { parts.next().unwrap_or_default() } else { part };
            if code.is_empty() {
                bail!("an empty junction in the secret URI");
            }
            junctions.push((hard, chain_code(code)));
        }
        Ok(SecretUri { phrase, junctions, password })
    }
}

/// A junction's chain code: a number as its little-endian u64, a name as its SCALE encoding, hashed when longer than 32 bytes.
fn chain_code(junction: &str) -> [u8; 32] {
    let encoded = match junction.parse::<u64>() {
        Ok(number) => number.to_le_bytes().to_vec(),
        Err(_) => {
            let mut out = compact_length(junction.len());
            out.extend_from_slice(junction.as_bytes());
            out
        }
    };
    let mut code = [0u8; 32];
    if encoded.len() > 32 {
        code.copy_from_slice(&Blake2b::<U32>::digest(&encoded));
    } else {
        code[..encoded.len()].copy_from_slice(&encoded);
    }
    code
}

fn compact_length(length: usize) -> Vec<u8> {
    match length {
        0..=0x3f => vec![(length as u8) << 2],
        0x40..=0x3fff => (((length as u16) << 2) | 1).to_le_bytes().to_vec(),
        _ => (((length as u32) << 2) | 2).to_le_bytes().to_vec(),
    }
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris_split_like_substrate() {
        let SecretUri { phrase, junctions, password } = SecretUri::parse("//desearch//0/7///secret").unwrap();
        assert_eq!((phrase, password), (DEV_PHRASE, Some("secret")));
        assert_eq!(junctions.iter().map(|j| j.0).collect::<Vec<_>>(), [true, true, false]);
        assert_eq!(junctions[1].1[..8], 0u64.to_le_bytes());
        assert_eq!(&junctions[0].1[..9], b"\x20desearch");
        assert!(SecretUri::parse("//a//").is_err());
    }

    #[test]
    fn a_signature_verifies_and_a_changed_message_does_not() {
        let key = Hotkey::from_uri("//test").unwrap();
        let signature = key.sign(b"payload");
        assert!(verify(&key.public(), b"payload", &signature));
        assert!(!verify(&key.public(), b"payload!", &signature));
        assert!(verify(&key.public(), b"payload", &key.sign(b"<Bytes>payload</Bytes>")));
    }

    #[test]
    fn keys_derive_as_bittensor_derives_them() {
        let cases = [
            ("0x1111111111111111111111111111111111111111111111111111111111111111", "5DtDLm5rQHShDqojQpsvcN8tRXHVFaecfDoRet1SU6BFD9Fi"),
            ("0x1111111111111111111111111111111111111111111111111111111111111111//hard/soft", "5CD5kdqYApcnR7fm8dBnVptjLdtCvfKQLM6geaaVhJG2fkBj"),
            ("//TaskApi", "5GYeh6mGT4knyPMmF4F6QpBeCKrp7qpqSmHhQW2a9GMHjGLp"),
            ("bottom drive obey lake curtain smoke basket hold race lonely fit walk//desearch///pw", "5He3yEYtHsQthqzW6dvuEwb9so3T6S3V5JYzk5ieqfuF4r6c"),
        ];
        for (uri, address) in cases {
            assert_eq!(Hotkey::from_uri(uri).unwrap().ss58(), address, "{uri}");
        }
    }

    #[test]
    fn addresses_match_bittensor() {
        let alice = Hotkey::from_uri("//Alice").unwrap();
        assert_eq!(alice.ss58(), "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY");
        assert_eq!(public_of(&alice.ss58()), Some(alice.public()));
        assert_eq!(public_of("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQZ"), None);
    }
}

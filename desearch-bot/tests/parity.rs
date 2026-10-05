//! The Rust crawler must key, join and date everything exactly as the Python crawler did.

use std::fs::File;
use std::io::{BufRead, BufReader};

use desearch_bot::{exclusions, hotkey, isodate, langid, signing, sitemaps, suffixes, urls};
use flate2::read::GzDecoder;
use serde_json::Value;

fn vectors(name: &str) -> Vec<Value> {
    let file = File::open(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    BufReader::new(GzDecoder::new(file)).lines().map(|line| serde_json::from_str(&line.unwrap()).unwrap()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn report(differ: &[String], total: usize) {
    assert!(differ.is_empty(), "{} of {total} differ from Python:\n{}", differ.len(), differ[..differ.len().min(15)].join("\n"));
}

#[test]
fn url_keys_match_python() {
    let rows = vectors("url-vectors.jsonl.gz");
    let mut differ = Vec::new();
    for row in &rows {
        let (host, url) = (row["host"].as_str().unwrap(), row["url"].as_str().unwrap());
        let got = urls::parse(url, host).map(|u| (hex(&u.key), u64::from(u.flags)));
        let want = row["key"].as_str().map(|key| (key.to_string(), row["flags"].as_u64().unwrap()));
        if got != want {
            differ.push(format!("{url:?} on {host}: rust {got:?} python {want:?}"));
        }
    }
    report(&differ, rows.len());
}

#[test]
fn joins_match_python() {
    let rows = vectors("join-vectors.jsonl.gz");
    let mut differ = Vec::new();
    for row in &rows {
        let (base, url) = (row["base"].as_str().unwrap(), row["url"].as_str().unwrap());
        let got = urls::join(base, url).ok();
        let want = row["joined"].as_str().map(str::to_string);
        if got != want {
            differ.push(format!("{base:?} + {url:?}: rust {got:?} python {want:?}"));
        }
    }
    report(&differ, rows.len());
}

#[test]
fn lastmods_match_python() {
    let rows = vectors("lastmod-vectors.jsonl.gz");
    let mut differ = Vec::new();
    for row in &rows {
        let value = row["value"].as_str().unwrap();
        let got = (isodate::parse_lastmod(Some(value)), sitemaps::has_time(Some(value).filter(|v| !v.is_empty())));
        let want = (row["us"].as_i64(), row["timed"].as_bool().unwrap());
        if got != want {
            differ.push(format!("{value:?}: rust {got:?} python {want:?}"));
        }
    }
    report(&differ, rows.len());
}

#[test]
fn signatures_match_python() {
    let file = File::open(format!("{}/tests/data/signing-vectors.json", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let vectors: Value = serde_json::from_reader(file).unwrap();
    let signer = signing::Signer::load(vectors["pem"].as_str().unwrap()).unwrap();
    for row in vectors["signatures"].as_array().unwrap() {
        let headers = signer.headers(row["url"].as_str().unwrap(), row["created"].as_i64().unwrap());
        for (name, value) in headers {
            assert_eq!(Some(value.as_str()), row["headers"][name].as_str(), "{name} for {}", row["url"]);
        }
    }
    let psl = std::env::var("PSL_FILE").ok().map(|path| suffixes::PublicSuffixList::load(path.as_ref()).unwrap());
    for row in vectors["hosts"].as_array().unwrap() {
        let host = row["host"].as_str().unwrap();
        let group = suffixes::tld_group(host);
        assert_eq!(Some(group), row["group"].as_str(), "group of {host}");
        assert_eq!(exclusions::exclusion_reason(host, group), row["reason"].as_str(), "reason for {host}");
        if let Some(psl) = &psl {
            assert_eq!(psl.registrable(host).as_deref(), row["registrable"].as_str(), "registrable {host}");
        }
    }
}

#[test]
fn hotkeys_and_task_api_signatures_match_python() {
    let file = File::open(format!("{}/tests/data/hotkey-vectors.json", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let vectors: Value = serde_json::from_reader(file).unwrap();
    for row in vectors["keys"].as_array().unwrap() {
        let uri = row["uri"].as_str().unwrap();
        let key = hotkey::Hotkey::from_uri(uri).unwrap();
        assert_eq!(hex(&key.public()), row["public"].as_str().unwrap(), "public key of {uri}");
        assert_eq!(key.ss58(), row["ss58"].as_str().unwrap(), "address of {uri}");
    }
    let signer = hotkey::Hotkey::from_uri(vectors["signer"].as_str().unwrap()).unwrap();
    for row in vectors["requests"].as_array().unwrap() {
        let text = |name: &str| row[name].as_str().unwrap();
        let payload = hotkey::signing_payload(text("method"), text("path"), text("body").as_bytes(), text("timestamp"), text("nonce"));
        assert_eq!(payload, text("payload").as_bytes());
        let python: Vec<u8> = (0..128).step_by(2).map(|i| u8::from_str_radix(&text("signature")[i..i + 2], 16).unwrap()).collect();
        assert!(hotkey::verify(&signer.public(), &payload, &python), "Python's signature of {} verifies", text("path"));
        assert!(hotkey::verify(&signer.public(), &payload, &signer.sign(&payload)));
        assert!(!hotkey::verify(&signer.public(), b"another payload", &python));
    }
}

#[test]
fn languages_match_python() {
    let Ok(path) = std::env::var("LANGID_FILE") else {
        return;
    };
    let model = langid::LangId::load(path.as_ref()).unwrap();
    let rows = vectors("langid-vectors.jsonl.gz");
    let mut differ = Vec::new();
    for row in &rows {
        let (text, want) = (row["text"].as_str().unwrap(), row["lang"].as_str().unwrap());
        let got = model.classify(text);
        if got != want {
            differ.push(format!("{:?}: rust {got} python {want} ({})", &text[..text.len().min(60)], row["score"]));
        }
    }
    report(&differ, rows.len());
}

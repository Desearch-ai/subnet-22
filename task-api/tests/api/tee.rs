//! The API in a CVM: its own key, the settings it shows, and a quote that binds both.

use sha2::{Digest, Sha256};
use task_api::attestation::{decode_hex, report_data};
use task_api::py::canonical;
use task_api::settings::KeySource;

use crate::fakedstack::{self, FakeDstack};
use crate::harness::{Harness, TempDir};

async fn in_cvm() -> (Harness, std::sync::Arc<FakeDstack>, TempDir) {
    // A socket path must fit in 104 bytes, which macOS's temporary directory does not leave room for.
    let dir = TempDir(std::path::PathBuf::from("/tmp").join(format!("sn22-{}", &uuid::Uuid::new_v4().simple().to_string()[..12])));
    let agent = FakeDstack::start(&dir.0).await;
    let socket = agent.socket.display().to_string();
    let h = Harness::with(move |s| {
        s.tee.key_source = KeySource::Cvm;
        s.tee.dstack_socket = socket;
    })
    .await;
    (h, agent, dir)
}

#[tokio::test]
async fn the_config_lists_every_setting_and_no_secret() {
    let h = Harness::start().await;
    let (status, config) = h.public("/v1/config").await;
    assert_eq!(status, 200);
    let settings = config["settings"].as_object().unwrap();
    assert_eq!(settings["TASK_API_CLAIM_TTL"], 180.0);
    assert_eq!(settings["TASK_API_KEY_SOURCE"], "uri");
    assert_eq!(settings["TASK_API_CHAIN_QUORUM"], 1);
    let shown = config.to_string();
    assert!(!shown.contains("\"test\""), "the storage keys are not shown: {shown}");
    assert_eq!(config["secrets_set"], serde_json::json!(["CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY"]));
    assert_eq!(h.public("/v1/attestation").await.0, 404, "not in a TEE");
}

#[tokio::test]
async fn the_quote_binds_the_signer_the_settings_and_a_fresh_nonce() {
    let (h, agent, _dir) = in_cvm().await;
    let signer = h.public("/v1/key").await.1["signer"].as_str().unwrap().to_string();
    assert_ne!(signer, task_api::state::State::dev_signer(), "not the development key");
    let (status, attested) = h.public("/v1/attestation").await;
    assert_eq!(status, 200, "{attested}");
    let document = &attested["document"];
    assert_eq!((document["kind"].as_str(), document["version"].as_i64()), (Some("sn22-task-api-attestation"), Some(1)));
    assert_eq!(document["signer"], signer.as_str());
    assert_eq!(document["handover_key"].as_str().unwrap().len(), 64);
    let config = h.public("/v1/config").await.1;
    assert_eq!(document["config_sha256"].as_str().unwrap(), desearch::canonical::hex(&Sha256::digest(canonical(&config))));
    assert!(document["tls_spki_sha256"].is_null(), "no certificate configured");

    let bound = report_data(document);
    assert_eq!(attested["report_data"].as_str().unwrap(), desearch::canonical::hex(&bound));
    let quote = decode_hex(attested["quote"].as_str().unwrap()).unwrap();
    assert_eq!(&quote[48 + 520..48 + 584], &bound[..], "the quote's own report data is the binding");
    assert_eq!(attested["app_compose"], fakedstack::APP_COMPOSE);
    assert_eq!(attested["info"]["compose_hash"], fakedstack::compose_hash().as_str());
    assert_eq!(attested["event_log"].as_array().unwrap().len(), 4);

    assert_eq!(h.public("/v1/attestation").await.1, attested, "the same quote while nothing it binds changed");
    assert_eq!(agent.calls("/GetQuote"), 1);
    let fresh = h.public("/v1/attestation?nonce=00ff").await.1;
    assert_eq!(fresh["document"]["nonce"], "00ff");
    assert_eq!(fresh["report_data"].as_str().unwrap(), desearch::canonical::hex(&report_data(&fresh["document"])));
    assert_eq!(agent.calls("/GetQuote"), 2);
    assert_eq!(h.public("/v1/attestation?nonce=xyz").await.0, 422);
    assert_eq!(agent.calls("/Info"), 1, "the app's identity is read once");
}

#[tokio::test]
async fn the_key_is_kept_across_a_restart_and_never_taken_from_the_environment() {
    let (h, _agent, _dir) = in_cvm().await;
    let first = h.state.signer();
    let again = task_api::attestation::cvm_key(&h.state.settings.data).unwrap();
    assert_eq!(again.ss58(), first);
    let refused =
        temp_env(&[("TASK_API_REGISTRY", "local"), ("TASK_API_KEY_SOURCE", "cvm"), ("TASK_API_KEY_URI", "//Alice")], task_api::settings::Settings::from_env);
    let error = refused.unwrap_err().to_string();
    assert!(error.contains("TASK_API_KEY_URI must not be set"), "{error}");
}

/// Runs `f` with these variables set, then unsets them; the tests that read the environment run one at a time.
fn temp_env<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    for (name, value) in vars {
        std::env::set_var(name, value);
    }
    let out = f();
    for (name, _) in vars {
        std::env::remove_var(name);
    }
    out
}

/// The `environment:` names of the dstack compose's `api` service.
fn compose_settings() -> std::collections::BTreeSet<String> {
    let compose = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy/dstack/docker-compose.yml")).unwrap();
    let api = compose.split("\n  redis:").next().unwrap();
    let environment = api.split("    environment:").nth(1).unwrap().split("\n    volumes:").next().unwrap();
    environment
        .lines()
        .filter_map(|line| line.trim().split_once(':').map(|(name, _)| name.to_string()))
        .filter(|name| !name.starts_with('#') && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
        .collect()
}

#[tokio::test]
async fn the_measured_compose_writes_every_setting_the_config_shows() {
    let h = Harness::start().await;
    let shown: std::collections::BTreeSet<String> = h.public("/v1/config").await.1["settings"].as_object().unwrap().keys().cloned().collect();
    let written = compose_settings();
    let missing: Vec<_> = shown.difference(&written).collect();
    assert!(missing.is_empty(), "settings the compose leaves to defaults: {missing:?}");
    let credentials = task_api::verify::CREDENTIALS.map(String::from);
    let unknown: Vec<_> = written.iter().filter(|name| !shown.contains(*name) && !credentials.contains(name) && *name != "PAGES_S3_BUCKET").collect();
    assert!(unknown.is_empty(), "settings the API does not read: {unknown:?}");
    assert!(!written.contains("TASK_API_KEY_URI"), "the key is made in the CVM");
}

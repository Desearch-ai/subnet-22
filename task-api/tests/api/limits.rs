//! Redis going away, request bodies over their limit, and the addresses limits are counted by.

use std::time::Duration;

use futures::stream;
use serde_json::json;
use task_api::lifecycle;
use task_api::settings::RegistryMode;
use task_api::state::State;

use crate::harness::{urls, Harness};

#[tokio::test]
async fn a_claim_while_redis_is_down_finds_no_tasks_and_anything_else_is_asked_to_retry() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    h.proxy.cut();
    let answer = h.miner.claim().await;
    assert_eq!((answer["tasks"].clone(), answer["refusal"]["code"].clone()), (json!([]), json!("UNAVAILABLE")));
    assert!(answer["refusal"]["inputs"]["retry_after"].as_f64().unwrap() > 0.0);
    let refused = h.miner.post("/v1/tasks/0123456789abcdef/abandon", json!({})).await.unwrap_err();
    assert_eq!(refused.status, 503);
    let (status, _) = h.public("/v1/room").await;
    assert_eq!(status, 503);
    h.proxy.restore();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while h.miner.claim().await["tasks"].as_array().is_none_or(Vec::is_empty) {
        assert!(std::time::Instant::now() < deadline, "claims come back with Redis");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn a_claim_held_across_a_restart_expires_without_a_strike() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    h.miner.claim().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut restarted = State::new(h.state.settings.clone(), h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.unwrap();
    restarted.started_at = desearch::time::now();
    tokio::time::sleep(Duration::from_millis(50)).await;
    h.rival.claim().await;
    let reclaimed = lifecycle::reclaim_expired(&restarted, desearch::time::now() + 3600.0).await.unwrap();
    assert_eq!(reclaimed.len(), 2, "both claims go back to the queue");
    let strikes = |hotkey: String| async {
        restarted
            .db
            .run(move |conn| {
                Ok(conn
                    .prepare("SELECT reason FROM strikes WHERE hotkey = ?")?
                    .query_map([hotkey], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?)
            })
            .await
            .unwrap()
    };
    assert!(strikes(h.miner.ss58()).await.is_empty(), "the claim held through the restart is forgiven");
    assert_eq!(strikes(h.rival.ss58()).await, ["claim_expired"], "a claim taken after the restart is not");
}

fn junk(request: reqwest::RequestBuilder, address: &str) -> reqwest::RequestBuilder {
    request
        .header("CF-Connecting-IP", address)
        .header("X-Hotkey", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
        .header("X-Timestamp", "1")
        .header("X-Nonce", "0".repeat(32))
        .header("X-Signature", "00".repeat(64))
}

#[tokio::test]
async fn a_body_over_its_limit_is_refused_without_being_read() {
    let h = Harness::start().await;
    h.enqueue(urls(6)).await;
    let claim = format!("{}/v1/tasks/claim", h.url);
    let big = junk(h.http.post(&claim), "1.1.1.1").body(vec![b'x'; 64_001]).send().await.unwrap();
    assert_eq!(big.status(), 413);
    assert!(big.json::<serde_json::Value>().await.unwrap()["detail"].as_str().unwrap().contains("64000"));
    let chunks = stream::iter((0..66).map(|_| Ok::<_, std::io::Error>(vec![b'x'; 1000])));
    let streamed = junk(h.http.post(&claim), "1.1.1.1").body(reqwest::Body::wrap_stream(chunks)).send().await.unwrap();
    assert_eq!(streamed.status(), 413, "with no declared length the body is counted as it arrives");
    let small = junk(h.http.post(&claim), "1.1.1.1").body("{}").send().await.unwrap();
    assert_eq!(small.status(), 401, "a small body reaches the signature check");
    assert!(h.miner.claim().await["tasks"].as_array().is_some_and(|t| !t.is_empty()));
}

#[tokio::test]
async fn a_verdict_and_an_enqueue_may_be_larger_than_a_claim() {
    use task_api::http::body_limit;
    assert_eq!(body_limit("/v1/validation/t1/score"), 16_000_000);
    assert_eq!(body_limit("/v1/admin/enqueue"), 16_000_000);
    assert_eq!(body_limit("/v1/tasks/claim"), 64_000);
    assert_eq!(body_limit("/v1/tasks/t1/complete"), 64_000);
}

#[tokio::test]
async fn an_address_that_keeps_failing_to_sign_in_is_refused() {
    let h = Harness::with(|s| s.failed_writes_per_minute = 3).await;
    h.enqueue(urls(6)).await;
    let claim = format!("{}/v1/tasks/claim", h.url);
    let mut answers = Vec::new();
    for _ in 0..5 {
        answers.push(junk(h.http.post(&claim), "1.1.1.1").body("{}").send().await.unwrap().status().as_u16());
    }
    assert_eq!(answers, [401, 401, 401, 429, 429]);
    let refused = junk(h.http.post(&claim), "1.1.1.1").body("{}").send().await.unwrap();
    let wait: i64 = refused.headers()["retry-after"].to_str().unwrap().parse().unwrap();
    assert!(wait > 0 && wait <= 60);
    assert_eq!(h.public_from("/v1/health", "1.1.1.1").await.0, 200, "reads are counted apart from writes");
    assert!(h.miner.claim().await["tasks"].as_array().is_some_and(|t| !t.is_empty()), "a miner at another address is not affected");
}

#[tokio::test]
async fn reads_are_counted_per_cloudflare_caller_not_per_forwarded_header() {
    let h = Harness::with(|s| s.reads_per_minute = 2).await;
    let get = |caller: &'static str| {
        let request = h.http.get(format!("{}/v1/shares", h.url)).header("CF-Connecting-IP", caller).header("X-Forwarded-For", "9.9.9.9");
        async move { request.send().await.unwrap().status().as_u16() }
    };
    let first = [get("1.1.1.1").await, get("1.1.1.1").await, get("1.1.1.1").await];
    assert_eq!(first, [200, 200, 429]);
    assert_eq!(get("2.2.2.2").await, 200, "a shared forwarded header does not share the limit");
}

#[tokio::test]
async fn without_cloudflare_in_front_the_connection_address_is_used() {
    let h = Harness::with(|s| s.reads_per_minute = 2).await;
    let statuses: Vec<u16> = futures::future::join_all((0..3).map(|_| h.http.get(format!("{}/v1/shares", h.url)).send()))
        .await
        .into_iter()
        .map(|r| r.unwrap().status().as_u16())
        .collect();
    let mut sorted = statuses.clone();
    sorted.sort();
    assert_eq!(sorted, [200, 200, 429]);
}

#[tokio::test]
async fn an_unregistered_hotkey_is_turned_away_before_its_signature_is_checked() {
    let h = Harness::with(|s| s.registry = RegistryMode::Chain { netuid: 22, network: "finney".into() }).await;
    h.state.registry.load(Vec::new());
    let stranger = desearch::hotkey::Hotkey::from_uri("//not-on-the-subnet").unwrap().ss58();
    let response = h
        .http
        .post(format!("{}/v1/tasks/claim", h.url))
        .header("X-Hotkey", stranger)
        .header("X-Timestamp", (desearch::time::now() as i64).to_string())
        .header("X-Nonce", "0".repeat(32))
        .header("X-Signature", "00".repeat(64))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 403);
    assert!(response.text().await.unwrap().contains("not registered"));
}

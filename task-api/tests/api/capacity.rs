//! The link a miner puts its capacity test reveal through.

use serde_json::json;

use crate::harness::Harness;

const REVEAL: &str = "/v1/capacity/reveal";

async fn test_block(h: &Harness) -> i64 {
    h.state.seeds.current_block().await.unwrap()
}

#[tokio::test]
async fn a_miner_gets_a_link_to_its_own_key_uploads_and_opens_it() {
    let h = Harness::start().await;
    let block = test_block(&h).await;
    let link = h.miner.post(REVEAL, json!({"test_block": block})).await.unwrap();
    let key = format!("capacity/{block}/{}.reveal", h.miner.ss58());
    assert_eq!(link["key"], json!(key));
    assert_eq!(link["content_type"], json!("application/octet-stream"));
    h.put(link["url"].as_str().unwrap(), b"a reveal".to_vec()).await;
    assert_eq!(h.object(&key).as_deref(), Some(&b"a reveal"[..]));
    let opened = h.miner.post(REVEAL, json!({"test_block": block, "uploaded": true})).await.unwrap();
    assert_eq!(opened["key"], json!(key));
}

#[tokio::test]
async fn nothing_uploaded_is_refused() {
    let h = Harness::start().await;
    let block = test_block(&h).await;
    let refused = h.miner.post(REVEAL, json!({"test_block": block, "uploaded": true})).await.unwrap_err();
    assert_eq!(refused.status, 422);
}

#[tokio::test]
async fn a_block_too_old_or_in_the_future_or_not_a_number_is_refused() {
    let h = Harness::start().await;
    let block = test_block(&h).await;
    for body in [json!({"test_block": block - 38}), json!({"test_block": block + 5}), json!({"test_block": "soon"}), json!({})] {
        let refused = h.miner.post(REVEAL, body).await.unwrap_err();
        assert_eq!(refused.status, 422);
    }
}

#[tokio::test]
async fn an_oversized_reveal_is_deleted_not_opened() {
    let h = Harness::start().await;
    let block = test_block(&h).await;
    let link = h.miner.post(REVEAL, json!({"test_block": block})).await.unwrap();
    let key = link["key"].as_str().unwrap().to_string();
    h.put(link["url"].as_str().unwrap(), vec![0; 600_001]).await;
    let refused = h.miner.post(REVEAL, json!({"test_block": block, "uploaded": true})).await.unwrap_err();
    assert_eq!(refused.status, 413);
    assert!(h.object(&key).is_none());
}

#[tokio::test]
async fn a_caller_only_ever_gets_its_own_hotkeys_key() {
    let h = Harness::start().await;
    let block = test_block(&h).await;
    let mine = h.miner.post(REVEAL, json!({"test_block": block, "hotkey": h.rival.ss58()})).await.unwrap();
    assert!(mine["key"].as_str().unwrap().contains(&h.miner.ss58()));
}

//! The chain as the API uses it: read now and then, counted on in between, paused rather than hammered when it refuses.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde_json::json;
use task_api::chain::Chain;
use task_api::registry::{self, Entry};
use task_api::settings::RegistryMode;
use task_api::state::State;

use crate::fakechain::FakeChain;
use crate::harness::Harness;

#[tokio::test]
async fn the_best_block_is_read_once_and_counted_on_after() {
    let fake = FakeChain::start(5_000, 12.0).await;
    let chain = Arc::new(Chain::new(&fake.url).unwrap());
    let blocks = join_all((0..100).map(|_| chain.current_block())).await;
    assert!(blocks.iter().all(|block| *block.as_ref().unwrap() == 5_000));
    assert_eq!(fake.calls("chain_getHeader"), 1);
}

#[tokio::test]
async fn a_refusing_chain_is_paused_instead_of_called_for_every_request() {
    let fake = FakeChain::start(5_000, 12.0).await;
    fake.refuse(true);
    let chain = Arc::new(Chain::new(&fake.url).unwrap());
    let tried = join_all((0..50).map(|_| chain.current_block())).await;
    assert!(tried.iter().all(Result::is_err));
    assert_eq!(fake.calls("chain_getHeader"), 1, "one call; the others wait out the pause");
    assert!(chain.block_hash(4_990).await.is_err());
    assert_eq!(fake.calls("chain_getBlockHash"), 0, "the pause holds for every kind of call");
}

#[tokio::test]
async fn a_block_not_made_yet_is_asked_about_once_a_moment_not_once_per_round() {
    let fake = FakeChain::start(5_000, 12.0).await;
    let chain = Arc::new(Chain::new(&fake.url).unwrap());
    for block in [5_003, 5_003, 5_004, 5_010] {
        assert_eq!(chain.block_hash(block).await.unwrap(), None);
    }
    assert_eq!(fake.calls("chain_getBlockHash"), 1);
    assert!(chain.block_hash(4_999).await.unwrap().is_some(), "an earlier block is still asked for");
}

#[tokio::test]
async fn a_block_kept_by_another_copy_is_counted_on_without_the_chain() {
    let fake = FakeChain::start(5_000, 12.0).await;
    fake.refuse(true);
    let chain = Arc::new(Chain::new(&fake.url).unwrap());
    chain.assume(4_000, desearch::time::now() - 25.0);
    assert_eq!(chain.current_block().await.unwrap(), 4_002);
    assert_eq!(fake.calls("chain_getHeader"), 0, "younger than a refresh, so not read again");
}

#[tokio::test]
async fn an_old_best_block_is_read_again_in_the_background() {
    let fake = FakeChain::start(5_000, 12.0).await;
    let chain = Arc::new(Chain::new(&fake.url).unwrap());
    chain.assume(4_000, desearch::time::now() - 60.0);
    assert_eq!(chain.current_block().await.unwrap(), 4_005, "answered at once from the count");
    let deadline = Instant::now() + Duration::from_secs(3);
    while chain.current_block().await.unwrap() != 5_000 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(chain.current_block().await.unwrap(), 5_000);
    assert_eq!(fake.calls("chain_getHeader"), 1);
}

#[tokio::test]
async fn a_kept_metagraph_lets_a_restarted_copy_serve_while_the_chain_refuses() {
    let fake = FakeChain::start(1_000, 12.0).await;
    fake.refuse(true);
    let h = Harness::start().await;
    let kept = vec![Entry { hotkey: h.miner.ss58(), uid: Some(3), is_validator: false, coldkey: Some("ck".into()) }];
    h.state.db.run(move |conn| registry::save(conn, &kept)).await.unwrap();
    let mut settings = h.state.settings.clone();
    settings.registry = RegistryMode::Chain { netuid: 22, network: fake.url.clone() };
    let restarted = State::new(settings, h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.unwrap();
    let started = Instant::now();
    assert_eq!(restarted.registry.lookup(&h.miner.ss58()).await.unwrap().unwrap().uid, Some(3));
    assert!(restarted.registry.lookup(&h.rival.ss58()).await.unwrap().is_none());
    assert!(started.elapsed() < Duration::from_secs(1), "no wait on the chain");
    assert!(restarted.registry.ready());
}

#[tokio::test]
async fn with_no_metagraph_yet_a_caller_is_asked_to_come_back() {
    let fake = FakeChain::start(1_000, 12.0).await;
    fake.refuse(true);
    let h = Harness::with(|s| s.registry = RegistryMode::Chain { netuid: 22, network: fake.url.clone() }).await;
    assert_eq!(h.public("/v1/ping").await.0, 503, "not healthy, so no traffic is routed to it");
    let started = Instant::now();
    let refused = h.miner.post("/v1/tasks/claim", json!({})).await.unwrap_err();
    assert_eq!(refused.status, 503);
    assert!(started.elapsed() < Duration::from_secs(8), "turned away, not held");
}

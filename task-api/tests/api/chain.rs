//! The chain as the API uses it: read now and then, counted on in between, paused rather than hammered when it refuses.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde_json::json;
use task_api::chain::Chain;
use task_api::registry::{self, Entry};
use task_api::settings::ChainOptions;
use task_api::settings::RegistryMode;
use task_api::state::State;

use crate::fakechain::{hash_of, FakeChain};
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
async fn a_refusing_endpoint_is_skipped_for_the_next_one_listed() {
    let (down, up) = (FakeChain::start(5_000, 12.0).await, FakeChain::start(5_000, 12.0).await);
    down.refuse(true);
    let chain = Arc::new(Chain::new(&format!("{}, {}", down.url, up.url)).unwrap());
    assert_eq!(chain.current_block().await.unwrap(), 5_000, "answered by the second endpoint");
    assert!(chain.block_hash(4_990).await.unwrap().is_some());
    assert_eq!((down.calls("chain_getHeader"), down.calls("chain_getBlockHash")), (1, 0), "the refusing one is paused, not asked again");
    assert_eq!(up.calls("chain_getBlockHash"), 1);
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

fn quorum(of: usize) -> ChainOptions {
    ChainOptions { quorum: of, finalized: true, count_on: false, refresh_s: 1.0 }
}

#[tokio::test]
async fn a_quorum_takes_the_finalized_head_and_hashes_only_when_enough_endpoints_agree() {
    let honest = [FakeChain::start(5_000, 12.0).await, FakeChain::start(5_000, 12.0).await];
    let liar = FakeChain::start(5_000, 12.0).await;
    liar.lie(true);
    let network = format!("{},{},{}", honest[0].url, liar.url, honest[1].url);
    let chain = Arc::new(Chain::with(&network, &quorum(2)).unwrap());
    assert_eq!(chain.current_block().await.unwrap(), 4_997, "the finalized head, not the liar's 1000 blocks ahead");
    assert_eq!(chain.block_hash(4_990).await.unwrap(), Some(hash_of(4_990)), "the two honest endpoints outvote the liar");
    assert_eq!(chain.block_hash(4_999).await.unwrap(), None, "past the finalized head is not a seed yet");

    honest[1].refuse(true);
    let alone = Arc::new(Chain::with(&network, &quorum(2)).unwrap());
    assert_eq!(alone.current_block().await.unwrap(), 4_997, "the second highest head is the honest one: a liar can only hold time back");
    assert_eq!(alone.block_hash(4_980).await.unwrap(), None, "a hash without a quorum is taken as not made yet");
    honest[0].refuse(true);
    let none = Arc::new(Chain::with(&network, &quorum(2)).unwrap());
    assert!(none.current_block().await.is_err(), "the liar alone is not a quorum");
    assert!(Chain::with(&honest[0].url, &quorum(2)).is_err(), "a quorum of two needs two endpoints");
}

#[tokio::test]
async fn without_counting_on_the_clock_the_block_only_moves_when_read() {
    let fake = FakeChain::start(5_000, 0.05).await;
    let chain = Arc::new(Chain::with(&fake.url, &ChainOptions { quorum: 1, finalized: false, count_on: false, refresh_s: 30.0 }).unwrap());
    let read = chain.current_block().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(chain.current_block().await.unwrap(), read, "blocks passed on the fake chain, but none was read");
    let counting = Arc::new(Chain::new(&fake.url).unwrap());
    counting.assume(read, desearch::time::now() - 25.0);
    assert!(counting.current_block().await.unwrap() > read, "counting on the clock moves it");
}

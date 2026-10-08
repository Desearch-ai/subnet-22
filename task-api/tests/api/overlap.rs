//! Two copies of the API at once, as during a rolling deploy.

use std::time::Duration;

use redis::AsyncCommands;
use task_api::janitor::{self, LEASE_KEY};
use task_api::roundlog::{self, Receipt};

use crate::harness::Harness;

#[tokio::test]
async fn one_copy_holds_the_janitor_lease_until_it_lets_go() {
    let h = Harness::start().await;
    assert!(janitor::hold(&h.state, "old").await.unwrap());
    assert!(!janitor::hold(&h.state, "new").await.unwrap(), "the old copy still runs the passes");
    assert!(janitor::hold(&h.state, "old").await.unwrap(), "and renews its lease");
    janitor::release(&h.state, "new").await.unwrap();
    assert!(!janitor::hold(&h.state, "new").await.unwrap(), "only the holder can let go");
    janitor::release(&h.state, "old").await.unwrap();
    assert!(janitor::hold(&h.state, "new").await.unwrap());
}

#[tokio::test]
async fn a_stopped_janitor_hands_its_lease_over_at_once() {
    let h = Harness::start().await;
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let running = tokio::spawn(janitor::run(h.state.clone(), stopped));
    for _ in 0..50 {
        let holder: Option<String> = h.redis.clone().get(LEASE_KEY).await.unwrap();
        if holder.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!janitor::hold(&h.state, "next").await.unwrap(), "the running janitor holds the lease");
    stop.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(10), running).await.expect("the janitor stops").unwrap();
    assert!(janitor::hold(&h.state, "next").await.unwrap());
}

#[tokio::test]
async fn a_refusal_naming_a_sealed_round_is_logged_outside_it() {
    let h = Harness::start().await;
    let root = h.state.db.run(|conn| roundlog::anchor(conn, "sealed", 1.0)).await.unwrap();
    let refusal = |round_id: &str| Receipt {
        round_id: round_id.into(),
        hotkey: "5Miner".into(),
        outcome: "refused",
        seq: 1,
        refusal: Some(serde_json::json!({"code": "QUEUE_EMPTY"})),
        ..Receipt::default()
    };
    let shown = h.state.record(vec![refusal("sealed"), refusal("open")]).await.unwrap();
    assert_eq!((shown[0]["body"]["round_id"].as_str(), shown[1]["body"]["round_id"].as_str()), (Some(""), Some("open")));
    let (sealed, again) = h.state.db.run(|conn| Ok((roundlog::entries(conn, "sealed")?.len(), roundlog::anchor(conn, "sealed", 2.0)?))).await.unwrap();
    assert_eq!((sealed, again), (0, root), "the sealed round's log and root are unchanged");
}

#[tokio::test]
async fn a_draining_copy_fails_its_health_check_but_still_serves() {
    let h = Harness::start().await;
    assert_eq!(h.public("/v1/ping").await.0, 200);
    h.state.draining.store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(h.public("/v1/ping").await.0, 503);
    assert_eq!(h.public("/v1/rounds").await.0, 200);
}

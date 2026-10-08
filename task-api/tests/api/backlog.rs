//! A backlog of rounds waiting for their seed, as after a burst of enqueues: claims must not wait behind it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use task_api::lifecycle;
use task_api::rounds::Url;

use crate::harness::Harness;

const ROUNDS: usize = 300;
const ROUND_URLS: usize = 10_000;

fn round_urls(round: usize) -> Vec<Url> {
    (0..ROUND_URLS)
        .map(|i| {
            let host = format!("news-site-{}.example.com", (round * 7 + i) % 500);
            Url { url: format!("https://{host}/2026/10/08/section/a-fairly-long-article-slug-{round}-{i}?ref=homepage"), host }
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted.get(((sorted.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "load test: run with --ignored --nocapture"]
async fn a_backlog_of_rounds_clears_without_holding_up_writes() {
    let h = Harness::start().await;
    let started = Instant::now();
    for round in 0..ROUNDS {
        lifecycle::open_round(&h.state, round_urls(round)).await.unwrap().unwrap();
    }
    eprintln!("{ROUNDS} rounds of {ROUND_URLS} URLs saved in {:.1}s", started.elapsed().as_secs_f64());
    tokio::time::sleep(Duration::from_millis(500)).await;

    let done = Arc::new(AtomicBool::new(false));
    let writes = {
        let (state, done) = (h.state.clone(), done.clone());
        tokio::spawn(async move {
            let mut waits = Vec::new();
            while !done.load(Ordering::Relaxed) {
                let asked = Instant::now();
                state.db.run(|_| Ok(())).await.unwrap();
                waits.push(asked.elapsed().as_secs_f64());
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            waits
        })
    };
    let rooms = {
        let (http, url, done) = (h.http.clone(), h.url.clone(), done.clone());
        tokio::spawn(async move {
            let mut took = Vec::new();
            while !done.load(Ordering::Relaxed) {
                let asked = Instant::now();
                let _ = http.get(format!("{url}/v1/room")).send().await;
                took.push(asked.elapsed().as_secs_f64());
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
            took
        })
    };
    let arriving = {
        let (state, done) = (h.state.clone(), done.clone());
        tokio::spawn(async move {
            let mut round = ROUNDS;
            while !done.load(Ordering::Relaxed) {
                lifecycle::open_round(&state, round_urls(round)).await.unwrap().unwrap();
                round += 1;
                tokio::time::sleep(Duration::from_millis(2000)).await;
            }
        })
    };

    let clearing = Instant::now();
    let mut passes = Vec::new();
    loop {
        let pass = Instant::now();
        lifecycle::reveal_pending(&h.state).await.unwrap();
        lifecycle::fill_missing(&h.state).await.unwrap();
        passes.push(pass.elapsed().as_secs_f64());
        let waiting = h.public("/v1/room").await.1["unrevealed"].as_i64().unwrap();
        if waiting <= 20 || clearing.elapsed() > Duration::from_secs(900) {
            break;
        }
    }
    let cleared = clearing.elapsed().as_secs_f64();
    done.store(true, Ordering::Relaxed);
    let _ = arriving.await;
    let mut waits = writes.await.unwrap();
    let mut rooms = rooms.await.unwrap();
    waits.sort_by(f64::total_cmp);
    rooms.sort_by(f64::total_cmp);
    passes.sort_by(f64::total_cmp);
    eprintln!(
        "cleared in {cleared:.1}s over {} passes (longest {:.1}s); a write waited p50 {:.3}s p99 {:.3}s max {:.3}s; /v1/room took p50 {:.3}s max {:.3}s",
        passes.len(),
        passes.last().copied().unwrap_or(0.0),
        percentile(&waits, 0.5),
        percentile(&waits, 0.99),
        waits.last().copied().unwrap_or(0.0),
        percentile(&rooms, 0.5),
        rooms.last().copied().unwrap_or(0.0),
    );
    assert!(percentile(&waits, 0.99) < 0.25, "writes waited behind the backlog");
    assert!(passes.last().copied().unwrap_or(0.0) < 30.0, "one pass held the janitor too long");
}

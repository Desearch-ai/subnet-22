//! Signed claim polls from many hotkeys at once: `cargo run --release --example load -- <url> <hotkeys> <seconds>`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use desearch::hotkey::{auth_headers, Hotkey};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let base = args.next().unwrap_or_else(|| "http://127.0.0.1:8080".into());
    let hotkeys: usize = args.next().map_or(Ok(256), |n| n.parse())?;
    let seconds: u64 = args.next().map_or(Ok(20), |n| n.parse())?;
    let http = reqwest::Client::builder().pool_max_idle_per_host(hotkeys).build()?;
    let until = Instant::now() + Duration::from_secs(seconds);
    let latencies = Arc::new(Mutex::new(Vec::new()));
    let statuses: Arc<[AtomicU64; 6]> = Arc::new(Default::default());
    let mut workers = Vec::new();
    for n in 0..hotkeys {
        let (http, base, latencies, statuses) = (http.clone(), base.clone(), latencies.clone(), statuses.clone());
        let hotkey = Hotkey::from_uri(&format!("//load-{n}"))?;
        workers.push(tokio::spawn(async move {
            while Instant::now() < until {
                let nonce = format!("{:032x}", rand::random::<u128>());
                let mut request = http.post(format!("{base}/v1/tasks/claim"));
                for (name, value) in auth_headers(&hotkey, "POST", "/v1/tasks/claim", b"", desearch::time::now() as i64, &nonce) {
                    request = request.header(name, value);
                }
                let started = Instant::now();
                let class = match request.send().await {
                    Ok(response) => {
                        let status = response.status().as_u16();
                        let _ = response.bytes().await;
                        usize::from(status / 100).min(5)
                    }
                    Err(_) => 0,
                };
                latencies.lock().unwrap().push(started.elapsed());
                statuses[class].fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for worker in workers {
        worker.await?;
    }
    let mut latencies = latencies.lock().unwrap().clone();
    latencies.sort();
    let at = |q: f64| latencies[((latencies.len() as f64 - 1.0) * q) as usize].as_secs_f64() * 1000.0;
    println!(
        "{} requests in {seconds}s = {:.0}/s; latency p50 {:.1} ms, p99 {:.1} ms; 2xx {} 4xx {} 5xx {} failed {}",
        latencies.len(),
        latencies.len() as f64 / seconds as f64,
        at(0.5),
        at(0.99),
        statuses[2].load(Ordering::Relaxed),
        statuses[4].load(Ordering::Relaxed),
        statuses[5].load(Ordering::Relaxed),
        statuses[0].load(Ordering::Relaxed),
    );
    Ok(())
}

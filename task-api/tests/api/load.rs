//! Production load on the task API: a bot at 50M URLs an hour with its five-minute burst, 200 miners, two validators,
//! a publisher, a chain that blocks every 12 s and refuses for two minutes, and a second copy taking over mid-run.
//! `LOAD_MINUTES=24 cargo test --release --test api load -- --ignored --nocapture`

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use desearch::hotkey::{auth_headers, Hotkey};
use rand::Rng;
use redis::Script;
use serde_json::{json, Value};
use task_api::janitor;
use task_api::manifest::OPEN_LIST_KEY;
use task_api::queues::{self, PCLAIMS, PPENDING, PUBLISH, PUBLISHED};
use task_api::settings::SeedsMode;
use task_api::state::State;
use tokio::net::TcpListener;
use tokio::sync::{watch, Semaphore};

use crate::fakechain::FakeChain;
use crate::harness::{Harness, ADMIN, OTHER_VALIDATOR, VALIDATOR};

const MINERS: usize = 200;
const URLS_PER_HOUR: f64 = 50_000_000.0;
const BOT_PASS: Duration = Duration::from_secs(15);
const BATCH_URLS: usize = 10_000;
const SENDS_AT_ONCE: usize = 4;
const SAVED_SECONDS: f64 = 300.0;
const MINER_HOLDS: usize = 8;
const CHECKS_AT_ONCE: usize = 8;
const PUBLISH_BATCH: i64 = 80;
const PUBLISH_PER_MIN: f64 = 1_200.0;
const TASK_URLS: usize = 1000;
const WANTED: f64 = URLS_PER_HOUR / 60.0 / TASK_URLS as f64;

static PCLAIM: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| Script::new(include_str!("../../src/lua/pclaim.lua")));
static PACK: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| Script::new(include_str!("../../src/lua/pack.lua")));

/// Every request's endpoint, seconds, status and when it started.
#[derive(Default)]
struct Log {
    calls: Mutex<Vec<(&'static str, f64, u16, f64)>>,
    waits: Mutex<Vec<(&'static str, f64, f64)>>,
    snapshots: Mutex<Vec<Value>>,
    issued: AtomicU64,
    completed: AtomicU64,
    scored: AtomicU64,
    published: AtomicU64,
    enqueued_urls: AtomicU64,
    retried: AtomicU64,
}

impl Log {
    fn call(&self, endpoint: &'static str, took: f64, status: u16, at: f64) {
        self.calls.lock().unwrap().push((endpoint, took, status, at));
    }
}

/// The copies of the API in front of the clients, taken in turn as a proxy would.
struct Targets {
    urls: RwLock<Vec<String>>,
    next: AtomicUsize,
}

impl Targets {
    fn pick(&self) -> String {
        let urls = self.urls.read().unwrap();
        urls[self.next.fetch_add(1, Ordering::Relaxed) % urls.len()].clone()
    }
}

struct Signer {
    key: Hotkey,
    http: reqwest::Client,
    targets: Arc<Targets>,
    log: Arc<Log>,
    started: Instant,
}

impl Signer {
    async fn post(&self, endpoint: &'static str, path: &str, body: &Value) -> (u16, Value) {
        let (asked, at) = (Instant::now(), self.started.elapsed().as_secs_f64());
        let (status, answer) = self.send(path, body).await;
        self.log.call(endpoint, asked.elapsed().as_secs_f64(), status, at);
        (status, answer)
    }

    /// As validators send a verdict: a 5xx is sent again after a pause, a few times; only the last answer is logged.
    async fn post_retrying(&self, endpoint: &'static str, path: &str, body: &Value) -> (u16, Value) {
        let (asked, at) = (Instant::now(), self.started.elapsed().as_secs_f64());
        let mut answer = self.send(path, body).await;
        for _ in 1..3 {
            if answer.0 < 500 && answer.0 != 0 {
                break;
            }
            self.log.retried.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_secs(2)).await;
            answer = self.send(path, body).await;
        }
        self.log.call(endpoint, asked.elapsed().as_secs_f64(), answer.0, at);
        answer
    }

    async fn send(&self, path: &str, body: &Value) -> (u16, Value) {
        let raw = serde_json::to_vec(body).unwrap();
        let nonce: String = (0..16).map(|_| format!("{:02x}", rand::thread_rng().gen::<u8>())).collect();
        let mut request = self.http.post(format!("{}{path}", self.targets.pick())).body(raw.clone()).header("content-type", "application/json");
        for (name, value) in auth_headers(&self.key, "POST", path, &raw, chrono_now(), &nonce) {
            request = request.header(name, value);
        }
        match request.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                (status, response.json().await.unwrap_or(Value::Null))
            }
            Err(_) => (0, Value::Null),
        }
    }
}

fn chrono_now() -> i64 {
    desearch::time::now() as i64
}

fn minutes() -> f64 {
    std::env::var("LOAD_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(24.0)
}

async fn serve(state: Arc<State>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = task_api::http::router(state);
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    url
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "load test: run with --ignored --nocapture"]
async fn the_api_carries_50m_urls_an_hour() {
    let total = Duration::from_secs_f64(minutes() * 60.0);
    let fake = FakeChain::start(6_000_000, 12.0).await;
    let chain_url = fake.url.clone();
    let h = Harness::with(move |s| {
        s.task_urls = TASK_URLS;
        s.queue_target = 8000;
        s.check_share = 0.05;
        s.checks_per_hour = 400.0;
        s.poll_rate = 2.0;
        s.seeds = SeedsMode::Chain { network: chain_url };
    })
    .await;
    let miners: Vec<Hotkey> = (0..MINERS).map(|i| Hotkey::from_uri(&format!("//load-miner-{i}")).unwrap()).collect();
    let seeded: Vec<String> = miners.iter().map(Hotkey::ss58).collect();
    h.state
        .db
        .run(move |conn| {
            for hotkey in &seeded {
                conn.execute("INSERT INTO miners (hotkey, pool, budget, verified) VALUES (?, 'crawl', 100, 100000)", [hotkey])?;
                for n in 0..10 {
                    task_api::checks::record(conn, hotkey, &format!("seed-{n}"), true, 0.0, 0, 0, desearch::time::now() - 7200.0)?;
                }
            }
            Ok(())
        })
        .await
        .unwrap();

    let log = Arc::new(Log::default());
    let started = Instant::now();
    let done = Arc::new(AtomicBool::new(false));
    let targets = Arc::new(Targets { urls: RwLock::new(vec![h.url.clone()]), next: AtomicUsize::new(0) });
    let http = reqwest::Client::builder().pool_max_idle_per_host(256).timeout(Duration::from_secs(30)).build().unwrap();
    let signer = |key: Hotkey| Arc::new(Signer { key, http: http.clone(), targets: targets.clone(), log: log.clone(), started });

    let (stop_a, stopped_a) = watch::channel(false);
    let janitor_a = tokio::spawn(janitor::run(h.state.clone(), stopped_a));
    let mut tasks = Vec::new();

    // The bot: room every pass, batches of 10k URLs four at a time, a pace that saves up five minutes.
    {
        let (bot, done, log) = (signer(Hotkey::from_uri(ADMIN).unwrap()), done.clone(), log.clone());
        let base = format!("{:x}", rand::random::<u32>());
        tasks.push(tokio::spawn(async move {
            let mut allowance = URLS_PER_HOUR * SAVED_SECONDS / 3600.0;
            let mut last = Instant::now();
            let mut sent = 0usize;
            let mut pending: Vec<(usize, usize)> = Vec::new();
            let batch = |(first, count): (usize, usize)| {
                let urls: Vec<Value> = (first..first + count)
                    .map(|n| json!({"host": format!("site{}.example", n % 5000), "url": format!("https://site{}.example/{base}/{n}", n % 5000)}))
                    .collect();
                json!({"urls": urls, "batch_id": format!("{base}-{first}-{count}")})
            };
            while !done.load(Ordering::Relaxed) {
                allowance = (allowance + last.elapsed().as_secs_f64() * URLS_PER_HOUR / 3600.0).min(URLS_PER_HOUR * SAVED_SECONDS / 3600.0);
                last = Instant::now();
                let (status, room) = bot.post_get("room", "/v1/room").await;
                if status == 200 && pending.is_empty() {
                    let budget = (room["room_tasks"].as_f64().unwrap_or(0.0) * TASK_URLS as f64).min(allowance) as usize;
                    for start in (0..budget).step_by(BATCH_URLS) {
                        let count = BATCH_URLS.min(budget - start);
                        pending.push((sent, count));
                        sent += count;
                    }
                }
                while !pending.is_empty() {
                    let wave: Vec<(usize, usize)> = pending.drain(..SENDS_AT_ONCE.min(pending.len())).collect();
                    let bodies: Vec<Value> = wave.iter().map(|part| batch(*part)).collect();
                    let answers = futures::future::join_all(bodies.iter().map(|body| bot.post("enqueue", "/v1/admin/enqueue", body))).await;
                    drop(bodies);
                    let mut failed = false;
                    for (part, (status, _)) in wave.into_iter().zip(answers) {
                        if status == 200 {
                            allowance -= part.1 as f64;
                            log.enqueued_urls.fetch_add(part.1 as u64, Ordering::Relaxed);
                        } else {
                            pending.push(part);
                            failed = true;
                        }
                    }
                    if failed {
                        break;
                    }
                }
                tokio::time::sleep(BOT_PASS).await;
            }
        }));
    }

    // Miners: hold up to a few tasks each, crawl 30-120 s, upload, complete; idle polls honour retry_after.
    for key in miners {
        let (miner, done, log) = (signer(key), done.clone(), log.clone());
        tasks.push(tokio::spawn(async move {
            let holding = Arc::new(Semaphore::new(MINER_HOLDS));
            while !done.load(Ordering::Relaxed) {
                let free = holding.available_permits();
                if free == 0 {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
                let (status, answer) = miner.post("claim", "/v1/tasks/claim", &json!({"count": free.min(4)})).await;
                let claimed = answer["tasks"].as_array().cloned().unwrap_or_default();
                if status != 200 || claimed.is_empty() {
                    let wait = answer["refusal"]["inputs"]["retry_after"].as_f64().unwrap_or(5.0).clamp(1.0, 10.0);
                    tokio::time::sleep(Duration::from_secs_f64(wait)).await;
                    continue;
                }
                log.issued.fetch_add(claimed.len() as u64, Ordering::Relaxed);
                for task in claimed {
                    let permit = holding.clone().acquire_owned().await.unwrap();
                    let (miner, log) = (miner.clone(), log.clone());
                    tokio::spawn(async move {
                        let crawl = rand::thread_rng().gen_range(30.0..120.0);
                        tokio::time::sleep(Duration::from_secs_f64(crawl)).await;
                        let body = [b"PAR1".as_slice(), task["task_id"].as_str().unwrap_or_default().as_bytes(), b"PAR1"].concat();
                        let put = miner
                            .http
                            .put(task["upload"]["url"].as_str().unwrap_or_default())
                            .header("content-type", "application/vnd.apache.parquet")
                            .body(body.clone());
                        if put.send().await.is_err() {
                            drop(permit);
                            return;
                        }
                        let rows = task["urls"].as_array().map_or(0, Vec::len) as i64;
                        let errors = rows / 20;
                        let report = json!({"key": task["upload"]["key"], "rows": rows, "ok": rows - errors, "errors": errors, "bytes": body.len()});
                        let (status, _) =
                            miner.post("complete", &format!("/v1/tasks/{}/complete", task["task_id"].as_str().unwrap_or_default()), &report).await;
                        if status == 200 {
                            log.completed.fetch_add(1, Ordering::Relaxed);
                        }
                        drop(permit);
                    });
                }
            }
        }));
    }

    // Validators: read the open list from storage, check each listed upload once, 20-40 s a check, with per-URL detail.
    for uri in [VALIDATOR, OTHER_VALIDATOR] {
        let (validator, done, log) = (signer(Hotkey::from_uri(uri).unwrap()), done.clone(), log.clone());
        let stub = h.stub.state.clone();
        tasks.push(tokio::spawn(async move {
            let mut seen: HashSet<String> = HashSet::new();
            let checking = Arc::new(Semaphore::new(CHECKS_AT_ONCE));
            while !done.load(Ordering::Relaxed) {
                let listed: Value =
                    stub.object(crate::harness::BUCKET, OPEN_LIST_KEY).and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or(Value::Null);
                for upload in listed["uploads"].as_array().cloned().unwrap_or_default() {
                    let task_id = upload["task_id"].as_str().unwrap_or_default().to_string();
                    if task_id.is_empty() || !seen.insert(task_id.clone()) {
                        continue;
                    }
                    let permit = checking.clone().acquire_owned().await.unwrap();
                    let (validator, log) = (validator.clone(), log.clone());
                    tokio::spawn(async move {
                        let check = rand::thread_rng().gen_range(20.0..40.0);
                        tokio::time::sleep(Duration::from_secs_f64(check)).await;
                        let urls: Vec<Value> = upload["urls"].as_array().cloned().unwrap_or_default();
                        let mut body = crate::harness::score("pass", urls.len() as i64, "ok", None, urls.first().and_then(Value::as_str).unwrap_or(""));
                        body["urls"] = urls
                            .iter()
                            .map(|url| {
                                json!({"url": url, "status": 200, "text_chars": 2400, "sampled": false, "outcome": "ok", "miner_chars": 2400,
                                       "miner_snippet": "A fairly long opening paragraph of the article that the validator keeps as a snippet for the log page, about two hundred characters long in practice."})
                            })
                            .collect::<Vec<_>>()
                            .into();
                        let (status, _) = validator.post_retrying("score", &format!("/v1/validation/{task_id}/score"), &body).await;
                        if status == 200 {
                            log.scored.fetch_add(1, Ordering::Relaxed);
                        }
                        drop(permit);
                    });
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }));
    }

    // The publisher: up to 80 jobs a claim at 1,200 jobs a minute, each acknowledged.
    {
        let (redis, done, log) = (h.redis.clone(), done.clone(), log.clone());
        tasks.push(tokio::spawn(async move {
            while !done.load(Ordering::Relaxed) {
                let expiry = desearch::time::now() + 600.0;
                let jobs: Vec<String> =
                    PCLAIM.key(PUBLISH).key(PCLAIMS).arg(PUBLISH_BATCH).arg(expiry).invoke_async(&mut redis.clone()).await.unwrap_or_default();
                if jobs.is_empty() {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
                tokio::time::sleep(Duration::from_secs_f64(jobs.len() as f64 * 60.0 / PUBLISH_PER_MIN)).await;
                for job in jobs {
                    let parsed = queues::unpack_job(&job).or_else(|| serde_json::from_str::<Value>(&job).ok()).unwrap_or(Value::Null);
                    let task_id = parsed["task_id"].as_str().unwrap_or_default().to_string();
                    let _: i64 = PACK.key(PCLAIMS).key(PPENDING).key(PUBLISHED).arg(&task_id).invoke_async(&mut redis.clone()).await.unwrap_or(0);
                    log.published.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Probes: how long the writer and the lookup readers take to start a job, and how long a ping takes.
    let primary = Arc::new(RwLock::new(h.state.clone()));
    {
        let (primary, done, log, targets, http) = (primary.clone(), done.clone(), log.clone(), targets.clone(), http.clone());
        tasks.push(tokio::spawn(async move {
            while !done.load(Ordering::Relaxed) {
                let state = primary.read().unwrap().clone();
                let at = started.elapsed().as_secs_f64();
                let asked = Instant::now();
                let _ = state.db.run(|_| Ok(())).await;
                log.waits.lock().unwrap().push(("writer", asked.elapsed().as_secs_f64(), at));
                let asked = Instant::now();
                let _ = state.lookups.read(|_| Ok(())).await;
                log.waits.lock().unwrap().push(("lookups", asked.elapsed().as_secs_f64(), at));
                let asked = Instant::now();
                let status = http.get(format!("{}/v1/ping", targets.pick())).send().await.map_or(0, |r| r.status().as_u16());
                log.call("ping", asked.elapsed().as_secs_f64(), status, at);
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }));
    }

    // Snapshots every 30 s: backlogs, counters, chain calls, Redis memory, our memory.
    {
        let (done, log, targets, http, fake, redis) = (done.clone(), log.clone(), targets.clone(), http.clone(), fake.clone(), h.redis.clone());
        tasks.push(tokio::spawn(async move {
            while !done.load(Ordering::Relaxed) {
                let base = targets.pick();
                let room: Value = match http.get(format!("{base}/v1/room")).send().await {
                    Ok(r) => r.json().await.unwrap_or(Value::Null),
                    Err(_) => Value::Null,
                };
                let health: Value = match http.get(format!("{base}/v1/health")).send().await {
                    Ok(r) => r.json().await.unwrap_or(Value::Null),
                    Err(_) => Value::Null,
                };
                let info: String = redis::cmd("INFO").arg("memory").query_async(&mut redis.clone()).await.unwrap_or_default();
                let redis_mb = info.lines().find_map(|l| l.strip_prefix("used_memory:")).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0) / 1e6;
                let rss_mb = std::process::Command::new("ps")
                    .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok())
                    .unwrap_or(0.0)
                    / 1024.0;
                log.snapshots.lock().unwrap().push(json!({
                    "min": started.elapsed().as_secs_f64() / 60.0,
                    "unrevealed": room["unrevealed"], "queue": room["queue"], "room": room["room_tasks"], "in_system": room["in_system"],
                    "published_per_min": room["published_per_min"],
                    "seeding": health["seeding"], "validating": health["validation_depth"], "publishing": health["publishing"],
                    "oldest_publish_s": health["oldest_publish_s"], "oldest_validation_s": health["oldest_validation_s"],
                    "issued": log.issued.load(Ordering::Relaxed), "completed": log.completed.load(Ordering::Relaxed),
                    "scored": log.scored.load(Ordering::Relaxed), "published": log.published.load(Ordering::Relaxed),
                    "enqueued_urls": log.enqueued_urls.load(Ordering::Relaxed),
                    "header_calls": fake.calls("chain_getHeader"), "hash_calls": fake.calls("chain_getBlockHash"),
                    "redis_mb": redis_mb, "rss_mb": rss_mb,
                }));
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        }));
    }

    // The script: chain refusals, then a second copy takes over as in a rolling deploy.
    let at = |fraction: f64| total.mul_f64(fraction);
    let mut events = Vec::new();
    tokio::time::sleep(at(6.0 / 24.0)).await;
    fake.refuse(true);
    events.push((started.elapsed().as_secs_f64() / 60.0, "chain refuses"));
    tokio::time::sleep(at(2.0 / 24.0)).await;
    fake.refuse(false);
    events.push((started.elapsed().as_secs_f64() / 60.0, "chain answers"));
    tokio::time::sleep(at(3.0 / 24.0)).await;
    let copy_b = Arc::new(State::new(h.state.settings.clone(), h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.unwrap());
    let url_b = serve(copy_b.clone()).await;
    let (stop_b, stopped_b) = watch::channel(false);
    let janitor_b = tokio::spawn(janitor::run(copy_b.clone(), stopped_b));
    targets.urls.write().unwrap().push(url_b.clone());
    events.push((started.elapsed().as_secs_f64() / 60.0, "copy B joins"));
    tokio::time::sleep(at(1.0 / 24.0)).await;
    let _ = stop_a.send(true);
    h.state.draining.store(true, Ordering::Relaxed);
    *targets.urls.write().unwrap() = vec![url_b];
    *primary.write().unwrap() = copy_b.clone();
    events.push((started.elapsed().as_secs_f64() / 60.0, "copy A drains and stops"));
    let _ = janitor_a.await;
    let left = total.saturating_sub(started.elapsed());
    tokio::time::sleep(left).await;
    done.store(true, Ordering::Relaxed);
    let _ = stop_b.send(true);
    let _ = janitor_b.await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    for task in tasks {
        task.abort();
    }

    report(&log, &events, minutes());
    judge(&log, minutes());
}

/// The release bar: steady throughput at the wanted pace over the last ten minutes, quick answers, no failures, few chain calls.
fn judge(log: &Log, minutes: f64) {
    let (rates, span, headers_per_min) = {
        let snapshots = log.snapshots.lock().unwrap();
        let window: Vec<&Value> = snapshots.iter().filter(|s| s["min"].as_f64().unwrap_or(0.0) >= minutes - 10.5).collect();
        let (first, last) = (window.first().unwrap(), window.last().unwrap());
        let span = last["min"].as_f64().unwrap() - first["min"].as_f64().unwrap();
        let per_min = |name: &str| (last[name].as_f64().unwrap_or(0.0) - first[name].as_f64().unwrap_or(0.0)) / span;
        let rates = [("issued", per_min("issued")), ("completed", per_min("completed")), ("published", per_min("published"))];
        (rates, span, last["header_calls"].as_f64().unwrap_or(0.0) / last["min"].as_f64().unwrap_or(1.0))
    };
    let (p99s, failed) = {
        let calls = log.calls.lock().unwrap();
        let p99 = |name: &str| {
            let mut found: Vec<f64> = calls.iter().filter(|(n, ..)| *n == name).map(|(_, t, ..)| *t).collect();
            found.sort_by(f64::total_cmp);
            percentile(&found, 0.99)
        };
        (
            ["claim", "complete", "score", "enqueue"].map(|name| (name, p99(name))),
            calls.iter().filter(|(_, _, status, _)| *status >= 500 || *status == 0).count(),
        )
    };
    let writer = {
        let waits = log.waits.lock().unwrap();
        let mut writer: Vec<f64> = waits.iter().filter(|(n, ..)| *n == "writer").map(|(_, w, _)| *w).collect();
        writer.sort_by(f64::total_cmp);
        writer
    };
    println!("\n== judged over the last {span:.1} min");
    for (name, value) in rates {
        println!("  {name:9} {value:6.0}/min (wanted {WANTED:.0})");
    }
    println!(
        "  p99 {p99s:?}; writer wait p99 {:.3}s max {:.3}s; failed {failed}; verdicts sent again {}; chain headers {headers_per_min:.1}/min",
        percentile(&writer, 0.99),
        writer.last().copied().unwrap_or(0.0),
        log.retried.load(Ordering::Relaxed)
    );
    assert!(rates.iter().all(|(_, rate)| *rate >= 0.95 * WANTED), "steady pace below 50M/h");
    assert!(p99s.iter().all(|(_, p99)| *p99 < 0.5), "slow answers");
    assert_eq!(failed, 0, "failed requests");
    assert!(percentile(&writer, 0.99) < 0.05, "writes waited");
    assert!(headers_per_min < 3.0, "too many chain calls");
}

impl Signer {
    async fn post_get(&self, endpoint: &'static str, path: &str) -> (u16, Value) {
        let asked = Instant::now();
        let at = self.started.elapsed().as_secs_f64();
        let (status, answer) = match self.http.get(format!("{}{path}", self.targets.pick())).send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                (status, response.json().await.unwrap_or(Value::Null))
            }
            Err(_) => (0, Value::Null),
        };
        self.log.call(endpoint, asked.elapsed().as_secs_f64(), status, at);
        (status, answer)
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted.get(((sorted.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(0.0)
}

fn report(log: &Log, events: &[(f64, &str)], minutes: f64) {
    println!("\n== events");
    for (at, what) in events {
        println!("  {at:5.1} min  {what}");
    }
    let calls = log.calls.lock().unwrap();
    let mut by_endpoint: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut statuses: BTreeMap<(&str, u16), usize> = BTreeMap::new();
    for (endpoint, took, status, _) in calls.iter() {
        by_endpoint.entry(endpoint).or_default().push(*took);
        *statuses.entry((endpoint, *status)).or_default() += 1;
    }
    println!("\n== latency (s): calls, p50, p95, p99, max");
    for (endpoint, took) in &mut by_endpoint {
        took.sort_by(f64::total_cmp);
        println!(
            "  {endpoint:9} {:7} {:7.3} {:7.3} {:7.3} {:7.3}",
            took.len(),
            percentile(took, 0.5),
            percentile(took, 0.95),
            percentile(took, 0.99),
            took.last().copied().unwrap_or(0.0)
        );
    }
    println!("\n== statuses");
    for ((endpoint, status), count) in &statuses {
        println!("  {endpoint:9} {status:4} {count}");
    }
    let waits = log.waits.lock().unwrap();
    println!("\n== queue waits (s) per minute: writer p50/p99/max | lookups p99/max");
    for minute in 0..minutes.ceil() as usize {
        let pick = |name: &str| {
            let mut found: Vec<f64> = waits.iter().filter(|(n, _, at)| *n == name && (*at / 60.0) as usize == minute).map(|(_, w, _)| *w).collect();
            found.sort_by(f64::total_cmp);
            found
        };
        let (writer, lookups) = (pick("writer"), pick("lookups"));
        println!(
            "  min {minute:2}: {:6.3} {:6.3} {:6.3} | {:6.3} {:6.3}",
            percentile(&writer, 0.5),
            percentile(&writer, 0.99),
            writer.last().copied().unwrap_or(0.0),
            percentile(&lookups, 0.99),
            lookups.last().copied().unwrap_or(0.0)
        );
    }
    println!("\n== minute by minute (claims and completes p99 s)");
    for minute in 0..minutes.ceil() as usize {
        let pick = |name: &str| {
            let mut found: Vec<f64> = calls.iter().filter(|(n, _, _, at)| *n == name && (*at / 60.0) as usize == minute).map(|(_, t, _, _)| *t).collect();
            found.sort_by(f64::total_cmp);
            found
        };
        let failed = calls.iter().filter(|(n, _, s, at)| *n != "room" && (*at / 60.0) as usize == minute && (*s >= 500 || *s == 0)).count();
        println!(
            "  min {minute:2}: claim p99 {:6.3} complete p99 {:6.3} score p99 {:6.3} enqueue p99 {:6.3} ping max {:6.3}  failed {failed}",
            percentile(&pick("claim"), 0.99),
            percentile(&pick("complete"), 0.99),
            percentile(&pick("score"), 0.99),
            percentile(&pick("enqueue"), 0.99),
            pick("ping").last().copied().unwrap_or(0.0)
        );
    }
    println!("\n== snapshots (wanted ~{WANTED:.0} tasks/min)");
    let snapshots = log.snapshots.lock().unwrap();
    let mut previous: Option<&Value> = None;
    for snapshot in snapshots.iter() {
        let rate = |name: &str| match previous {
            Some(before) => {
                (snapshot[name].as_f64().unwrap_or(0.0) - before[name].as_f64().unwrap_or(0.0))
                    / (snapshot["min"].as_f64().unwrap_or(0.0) - before["min"].as_f64().unwrap_or(0.0)).max(1e-9)
            }
            None => 0.0,
        };
        println!(
            "  {:5.1} min issued/min {:5.0} completed/min {:5.0} published/min {:5.0} urls/min {:7.0} | unrevealed {} queue {} seeding {} validating {} publishing {} oldest pub {}s | headers {} hashes {} | redis {:.0} MB rss {:.0} MB",
            snapshot["min"].as_f64().unwrap_or(0.0),
            rate("issued"),
            rate("completed"),
            rate("published"),
            rate("enqueued_urls"),
            snapshot["unrevealed"],
            snapshot["queue"],
            snapshot["seeding"],
            snapshot["validating"],
            snapshot["publishing"],
            snapshot["oldest_publish_s"],
            snapshot["header_calls"],
            snapshot["hash_calls"],
            snapshot["redis_mb"].as_f64().unwrap_or(0.0),
            snapshot["rss_mb"].as_f64().unwrap_or(0.0),
        );
        previous = Some(snapshot);
    }
}

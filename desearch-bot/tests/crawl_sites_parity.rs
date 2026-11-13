//! `desearch-bot crawl-sites` against the replay server, compared with the frozen golden runs in `tests/data/crawl_sites/golden.json`.

#[allow(dead_code)]
#[path = "support/site_replay.rs"]
mod site_replay;

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

/// Hosts with two requests in flight, whose order between those two is a race on both sides.
const UNORDERED: [&str; 1] = ["slots.test"];
/// Request starts are paced; the server sees them a TLS handshake later, which varies by a few milliseconds.
const JITTER: f64 = 0.05;

fn load(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/crawl_sites").join(name)).unwrap()).unwrap()
}

fn lines(path: &Path, items: &Value) {
    let text: String = items.as_array().unwrap().iter().map(|l| format!("{}\n", l.as_str().unwrap())).collect();
    std::fs::write(path, text).unwrap();
}

async fn run_one(name: &str, run: &Value, fixture: &Value) -> (Value, Vec<site_replay::Hit>) {
    let dir = std::env::temp_dir().join(format!("crawl-sites-parity-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    lines(&dir.join("hosts.txt"), &run["hosts_file"]);
    let mut args: Vec<String> = vec!["crawl-sites".into(), dir.join("hosts.txt").display().to_string(), "--out".into(), dir.join("out").display().to_string()];
    args.extend(run["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()));
    for (key, flag, file) in [("skip", "--skip", "skip.txt"), ("seeds", "--seeds-file", "seeds.txt")] {
        if let Some(items) = run.get(key) {
            lines(&dir.join(file), items);
            args.extend([flag.to_string(), dir.join(file).display().to_string()]);
        }
    }
    if let Some(set) = run.get("set") {
        std::fs::write(dir.join("set.json"), set.to_string()).unwrap();
        args.extend(["--set".to_string(), dir.join("set.json").display().to_string()]);
    }
    if run.get("found").is_some() {
        args.extend(["--found".to_string(), dir.join("found.txt").display().to_string()]);
    }
    let replay = site_replay::start(fixture, 0, None).await;
    std::fs::write(dir.join("ca.pem"), &replay.ca_pem).unwrap();
    args.extend(["--test-resolve-all".to_string(), replay.addr.to_string(), "--test-ca-cert".to_string(), dir.join("ca.pem").display().to_string()]);
    let done = tokio::task::spawn_blocking(move || Command::new(env!("CARGO_BIN_EXE_desearch-bot")).args(&args).output().unwrap()).await.unwrap();
    assert!(done.status.success(), "run {name}: {}", String::from_utf8_lossy(&done.stderr));
    let stdout = String::from_utf8(done.stdout).unwrap();
    let stderr = String::from_utf8(done.stderr).unwrap();
    let total = stdout.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).find_map(|v| v.get("total").cloned()).unwrap();
    let mut hosts = serde_json::Map::new();
    for line in stderr.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        let mut counts = line.as_object().unwrap().clone();
        let host = counts.remove("host").unwrap();
        hosts.insert(host.as_str().unwrap().to_string(), Value::Object(counts));
    }
    let hits = replay.hits.lock().unwrap().clone();
    let mut requests: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for hit in &hits {
        requests.entry(hit.host.clone()).or_default().push(hit.target.clone());
    }
    let out = dir.join("out");
    let mut records = desearch_bot::sitecrawl::store::read(std::slice::from_ref(&out)).unwrap();
    for r in &mut records {
        r.as_object_mut().unwrap().remove("fetched_at");
    }
    let files = std::fs::read_dir(&out).unwrap().count();
    let found = run.get("found").map(|_| {
        let mut urls: Vec<String> = std::fs::read_to_string(dir.join("found.txt")).unwrap().lines().map(str::to_string).collect();
        urls.sort();
        urls
    });
    let agents: std::collections::BTreeSet<&str> = hits.iter().map(|h| h.agent.as_str()).collect();
    std::fs::remove_dir_all(&dir).unwrap();
    (json!({"total": total, "hosts": hosts, "requests": requests, "records": records, "files": files, "found": found, "agents": agents}), hits)
}

fn by_host(records: &Value) -> BTreeMap<String, Vec<Value>> {
    let mut out: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for r in records.as_array().unwrap() {
        out.entry(r["url"].as_str().unwrap().split('/').nth(2).unwrap().to_string()).or_default().push(r.clone());
    }
    out
}

/// The least time between two requests to each host, robots.txt left out: its read and the home page go back to back on both sides.
fn gaps(hits: &[site_replay::Hit]) -> BTreeMap<String, f64> {
    let mut times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for hit in hits.iter().filter(|h| h.target != "/robots.txt") {
        times.entry(hit.host.clone()).or_default().push(hit.t);
    }
    times.into_iter().filter(|(_, t)| t.len() > 1).map(|(host, t)| (host, t.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min))).collect()
}

/// The golden run with the hosts the robots.txt rule keeps off put as `strict_robots.json` says.
fn expected(golden: &Value, strict: &Value) -> Value {
    let mut want = golden.clone();
    let Some(hosts) = strict.as_object() else { return want };
    for (host, run) in hosts {
        let total = want["total"].as_object_mut().unwrap();
        for (key, n) in golden["hosts"][host].as_object().unwrap() {
            let left = total[key].as_i64().unwrap() - n.as_i64().unwrap();
            if left == 0 && n.as_i64().unwrap() > 0 {
                total.remove(key);
            } else {
                total.insert(key.clone(), json!(left));
            }
        }
        for (key, n) in run["counts"].as_object().unwrap() {
            let now = total.get(key).and_then(Value::as_i64).unwrap_or(0);
            total.insert(key.clone(), json!(now + n.as_i64().unwrap()));
        }
        want["hosts"][host] = run["counts"].clone();
        want["requests"][host] = run["requests"].clone();
        want["gaps"].as_object_mut().unwrap().remove(host);
        want["records"].as_array_mut().unwrap().retain(|r| r["url"].as_str().unwrap().split('/').nth(2) != Some(host.as_str()));
    }
    want
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crawl_sites_runs_as_the_golden_runs_did() {
    let (fixture, golden, strict) = (load("fixture.json"), load("golden.json"), load("strict_robots.json"));
    let runs: Vec<(String, Value)> = fixture["runs"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let mut started = Vec::new();
    for (name, run) in runs {
        let fixture = fixture.clone();
        started.push((name.clone(), tokio::spawn(async move { run_one(&name, &run, &fixture).await })));
    }
    let mut wrong: Vec<String> = Vec::new();
    for (name, task) in started {
        let (got, hits) = task.await.unwrap();
        let want = &expected(&golden["runs"][&name], &strict["runs"][&name]);
        for key in ["total", "hosts", "files", "found", "agents"] {
            if got[key] != want[key] {
                wrong.push(format!("run {name} {key}:\n  want {}\n  got  {}", want[key], got[key]));
            }
        }
        let hosts: std::collections::BTreeSet<&String> =
            want["requests"].as_object().unwrap().keys().chain(got["requests"].as_object().unwrap().keys()).collect();
        for host in hosts {
            let (mut w, mut g) = (want["requests"][host].clone(), got["requests"][host].clone());
            if UNORDERED.contains(&host.as_str()) {
                for list in [&mut w, &mut g] {
                    list.as_array_mut().unwrap().sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                }
            }
            if w != g {
                wrong.push(format!("run {name} requests to {host}:\n  want {w}\n  got  {g}"));
            }
        }
        let (mut w, mut g) = (by_host(&want["records"]), by_host(&got["records"]));
        for host in UNORDERED {
            for side in [&mut w, &mut g] {
                if let Some(list) = side.get_mut(host) {
                    list.sort_by(|a, b| a["url"].as_str().cmp(&b["url"].as_str()));
                }
            }
        }
        if w != g {
            wrong.push(format!("run {name} records:\n  want {}\n  got  {}", json!(w), json!(g)));
        }
        for (host, gap) in gaps(&hits) {
            let kept = want["gaps"][&host].as_f64().unwrap_or(0.0);
            if kept >= 0.9 && gap < kept.min(if kept >= 1.9 { 2.0 } else { 1.0 }) - JITTER {
                wrong.push(format!("run {name}: requests to {host} {gap:.3} s apart, the golden run kept {kept:.3} s"));
            }
        }
        if name == "a" {
            let at = |target: &str| hits.iter().find(|h| h.host == "ratelimit.test" && h.target == target).unwrap().t;
            assert!(at("/p2") - at("/p1") >= 6.0 - JITTER, "the first 429 waits out its Retry-After");
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_golden_run_kept_a_second_between_paced_requests() {
    let golden = load("golden.json");
    for (name, run) in golden["runs"].as_object().unwrap() {
        for (host, gap) in run["gaps"].as_object().unwrap() {
            let gap = gap.as_f64().unwrap();
            let known_fast = ["errors.test", "slots.test", "homeredirect.test"].contains(&host.as_str());
            assert!(known_fast || gap >= 1.0 - JITTER, "run {name}: {host} {gap}");
        }
    }
}

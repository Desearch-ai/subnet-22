#[allow(dead_code)]
#[path = "support/replay.rs"]
mod replay;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bulk_bittensor::fetch::{self, http::TestRoute, robots::Robots, yarl::Url, Anchors, Options, Pace};
use bulk_core::worker::{Worker, PAGE_WORKER};
use serde_json::Value;

const UA: &str = fetch::USER_AGENT;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/bittensor-fetch")
}

fn load(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap();
    let bytes = if path.extension().is_some_and(|e| e == "gz") {
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(bytes.as_slice()), &mut out).unwrap();
        out
    } else {
        bytes
    };
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn urls_follow_yarl_and_urllib() {
    let vectors = load(&fixtures().join("vectors.json"));
    for case in vectors["yarl"].as_array().unwrap() {
        let url = Url::parse(case[0].as_str().unwrap()).unwrap();
        assert_eq!(url.to_string(), case[1].as_str().unwrap(), "{}", case[0]);
        assert_eq!(url.raw_path_qs(), case[2].as_str().unwrap(), "{}", case[0]);
    }
    for case in vectors["join"].as_array().unwrap() {
        let base = Url::parse(case[0].as_str().unwrap()).unwrap();
        let joined = base.join(&Url::parse(case[1].as_str().unwrap()).unwrap());
        assert_eq!(joined.to_string(), case[2].as_str().unwrap(), "{} + {}", case[0], case[1]);
    }
    for case in vectors["urljoin"].as_array().unwrap() {
        let joined = bulk_core::urlparse::urljoin(case[0].as_str().unwrap(), case[1].as_str().unwrap()).unwrap();
        assert_eq!(joined, case[2].as_str().unwrap(), "{} + {}", case[0], case[1]);
    }
}

#[test]
fn robots_follow_python_robotparser() {
    let vectors = load(&fixtures().join("vectors.json"));
    let urls: Vec<&str> = vectors["robot_urls"].as_array().unwrap().iter().map(|u| u.as_str().unwrap()).collect();
    for case in vectors["robots"].as_array().unwrap() {
        let text = case["robots"].as_str().unwrap();
        let rules = Robots::parse(text).unwrap();
        assert_eq!(rules.crawl_delay(UA).map(|d| d as i64), case["delay"].as_i64(), "{text}");
        for (url, allowed) in urls.iter().zip(case["allowed"].as_array().unwrap()) {
            assert_eq!(rules.can_fetch(UA, url).unwrap(), allowed.as_bool().unwrap(), "{text} {url}");
        }
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn files(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                found.push((path.strip_prefix(root).unwrap().to_string_lossy().into_owned(), std::fs::read(&path).unwrap()));
            }
        }
    }
    found.sort();
    found
}

fn assert_same_files(got: &Path, want: &Path) {
    let (got, want) = (files(got), files(want));
    let names = |files: &[(String, Vec<u8>)]| files.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&got), names(&want));
    for ((name, a), (_, b)) in got.iter().zip(&want) {
        assert!(a == b, "{name} differs:\n{}\n---\n{}", String::from_utf8_lossy(a), String::from_utf8_lossy(b));
    }
}

fn no_anchors() -> Arc<Anchors> {
    Arc::new(Anchors::new(Box::new(|| anyhow::bail!("no page worker in this test"))))
}

fn summary(out: &str, counts: &bulk_core::Counts) -> String {
    let mut line = serde_json::Map::new();
    line.insert("out".into(), Value::String(out.into()));
    for (key, n) in counts.iter() {
        line.insert(key.into(), n.into());
    }
    bulk_core::pyjson::dumps(&Value::Object(line), None)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_chain_read_matches_the_golden_run_at_its_block() {
    let dir = fixtures().join("chain");
    let recorded = load(&dir.join("replay.json.gz"));
    let server = replay::start(&recorded["exchanges"], vec!["127.0.0.1".into()], 0).await;
    let block = load(&dir.join("expected/subnets-chain.json"))["block"].as_i64().unwrap();
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("src");
    let options = Options {
        out: out.clone(),
        previous: None,
        block: Some(block),
        chain: format!("http://{}", server.addr),
        chain_snapshot: None,
        only_chain: true,
        pace: Pace { waits: false },
        route: TestRoute::default(),
        token: String::new(),
    };
    fetch::fetch(&options, no_anchors()).await.unwrap();
    assert!(server.misses.lock().unwrap().is_empty(), "{:?}", server.misses.lock().unwrap());
    assert_same_files(&out, &dir.join("expected"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_replayed_fetch_matches_the_golden_run() {
    let Some(python) = std::env::var("BULK_PYTHON").ok().filter(|p| !p.is_empty()) else {
        eprintln!("skipped: set BULK_PYTHON to a Python with lxml for the page worker");
        return;
    };
    let dir = fixtures().join("replay");
    let recorded = load(&dir.join("replay.json.gz"));
    let hosts: Vec<String> = recorded["hosts"].as_array().unwrap().iter().map(|h| h.as_str().unwrap().to_string()).collect();
    let server = replay::start(&recorded["exchanges"], hosts, 0).await;
    let work = tempfile::tempdir().unwrap();
    let ca = work.path().join("ca.pem");
    std::fs::write(&ca, &server.ca_pem).unwrap();
    copy_dir(&dir.join("input/previous"), &work.path().join("previous"));
    let out = work.path().join("src");
    let options = Options {
        out: out.clone(),
        previous: Some(work.path().join("previous")),
        block: None,
        chain: "https://entrypoint-finney.opentensor.ai".into(),
        chain_snapshot: Some(dir.join("input/snapshot.json")),
        only_chain: false,
        pace: Pace { waits: false },
        route: TestRoute { resolve_all: Some(server.addr), ca_cert: Some(ca) },
        token: String::new(),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let anchors = Arc::new(Anchors::new(Box::new(move || Worker::spawn(&python, &root, PAGE_WORKER, &[]))));
    let counts = fetch::fetch(&options, anchors).await.unwrap();
    assert!(server.misses.lock().unwrap().is_empty(), "{:?}", server.misses.lock().unwrap());
    let mut written = std::fs::read_to_string(out.join(fetch::FETCH)).unwrap();
    written = written.replace(&bulk_core::run::path_str(&work.path().join("previous")), "previous");
    std::fs::write(out.join(fetch::FETCH), written).unwrap();
    assert_same_files(&out, &dir.join("expected"));
    let stdout = std::fs::read_to_string(dir.join("stdout.txt")).unwrap();
    assert_eq!(summary("src", &counts), stdout.trim_end());
}

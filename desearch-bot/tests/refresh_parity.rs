//! `desearch-bot refresh` against the replay server, compared with the Python crawler's frozen runs in `tests/data/refresh/golden.json`.

#[allow(dead_code)]
#[path = "support/site_replay.rs"]
mod site_replay;

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use arrow_array::{Array, Int32Array, LargeStringArray, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Map, Value};

const ROW_COLUMNS: [&str; 16] = [
    "url",
    "final_url",
    "status",
    "content_type",
    "page_type",
    "title",
    "description",
    "lang",
    "canonical",
    "published",
    "text_mode",
    "extraction",
    "text_sha256",
    "html_sha256",
    "html_bytes",
    "headings",
];

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/refresh").join(name)
}

fn load(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(data(name)).unwrap()).unwrap()
}

fn cell(column: &dyn Array, i: usize) -> Value {
    let any = column.as_any();
    if let Some(a) = any.downcast_ref::<StringArray>() {
        return json!(a.value(i));
    }
    if let Some(a) = any.downcast_ref::<LargeStringArray>() {
        return json!(a.value(i));
    }
    if let Some(a) = any.downcast_ref::<Int32Array>() {
        return json!(a.value(i));
    }
    let list = any.downcast_ref::<ListArray>().expect("a string, int or list column");
    let items = list.value(i);
    let items = items.as_any().downcast_ref::<StringArray>().unwrap();
    json!((0..items.len()).map(|j| items.value(j)).collect::<Vec<_>>())
}

fn rows(out: &Path) -> (Value, Value) {
    let Ok(text) = std::fs::read_to_string(out.join("manifest.json")) else { return (Value::Null, Value::Null) };
    let mut manifest: Value = serde_json::from_str(&text).unwrap();
    manifest.as_object_mut().unwrap().remove("as_of");
    let mut found = Vec::new();
    for name in manifest["files"].as_array().unwrap() {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(out.join(name.as_str().unwrap())).unwrap()).unwrap().build().unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            for i in 0..batch.num_rows() {
                let row: Map<String, Value> = ROW_COLUMNS.iter().map(|c| (c.to_string(), cell(batch.column_by_name(c).unwrap().as_ref(), i))).collect();
                found.push(Value::Object(row));
            }
        }
    }
    found.sort_by(|a, b| a["url"].as_str().cmp(&b["url"].as_str()));
    (json!(found), manifest)
}

fn records(folder: &Path) -> Value {
    if !folder.exists() {
        return Value::Null;
    }
    let mut found = desearch_bot::sitecrawl::store::read(&[folder.to_path_buf()]).unwrap();
    for r in &mut found {
        r.as_object_mut().unwrap().remove("fetched_at");
    }
    found.sort_by(|a, b| a["url"].as_str().cmp(&b["url"].as_str()));
    json!(found)
}

async fn run_one(name: &str, run: &Value, fixture: &Value) -> Value {
    let dir = std::env::temp_dir().join(format!("refresh-parity-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("set.json"), fixture["set"].to_string()).unwrap();
    let source = run["args"][0].as_str().unwrap();
    let out = dir.join("build").join(source);
    let subnet22 = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let path = |p: &Path| p.display().to_string();
    let mut args: Vec<String> = vec!["refresh".into()];
    args.extend(run["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()));
    args.extend(["--out".into(), path(&out), "--set".into(), path(&dir.join("set.json")), "--subnet22".into(), path(subnet22)]);
    if let Some(urls) = run.get("urls") {
        let text: String = urls.as_array().unwrap().iter().map(|u| format!("{}\n", u.as_str().unwrap())).collect();
        std::fs::write(dir.join("urls.txt"), text).unwrap();
        args.extend(["--urls".into(), path(&dir.join("urls.txt"))]);
    }
    if run.get("exists").is_some() {
        std::fs::create_dir_all(&out).unwrap();
    }
    let replay = site_replay::start(fixture, 0, None).await;
    std::fs::write(dir.join("ca.pem"), &replay.ca_pem).unwrap();
    args.extend([
        "--python".into(),
        "python3".into(),
        "--test-resolve-all".into(),
        replay.addr.to_string(),
        "--test-ca-cert".into(),
        path(&dir.join("ca.pem")),
    ]);
    let done = tokio::task::spawn_blocking(move || Command::new(env!("CARGO_BIN_EXE_desearch-bot")).args(&args).output().unwrap()).await.unwrap();
    let mut requests: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut agents = std::collections::BTreeSet::new();
    for hit in replay.hits.lock().unwrap().iter() {
        requests.entry(hit.host.clone()).or_default().push(hit.target.clone());
        agents.insert(hit.agent.clone());
    }
    let (rows, manifest) = rows(&out);
    let got = json!({
        "exit": done.status.code(),
        "stdout": String::from_utf8_lossy(&done.stdout).trim().replace(&path(&dir), "{dir}"),
        "stderr": String::from_utf8_lossy(&done.stderr),
        "requests": requests,
        "agents": agents,
        "rows": rows,
        "manifest": manifest,
        "records": records(&out.with_file_name(format!("{source}.records"))),
    });
    std::fs::remove_dir_all(&dir).unwrap();
    got
}

/// A failure line's counts, which concurrent EDGAR fetches count in no fixed order.
fn unordered(stdout: &str) -> Value {
    let Ok(line) = serde_json::from_str::<Value>(stdout) else { return json!(stdout) };
    let Some(message) = line.get("failed").and_then(Value::as_str) else { return line };
    let Some((why, rest)) = message.split_once(": {") else { return line };
    let counts: BTreeMap<&str, &str> =
        rest.trim_end_matches('}').split(", ").filter_map(|kv| kv.split_once(": ")).map(|(k, v)| (k.trim_matches('\''), v)).collect();
    json!({"failed": why, "counts": counts})
}

fn python_has_lxml() -> bool {
    Command::new("python3").args(["-c", "import lxml"]).status().is_ok_and(|s| s.success())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refresh_runs_as_the_python_crawler() {
    if !python_has_lxml() {
        eprintln!("no python3 with lxml for the extraction workers; skipped");
        return;
    }
    let (fixture, golden) = (load("fixture.json"), load("golden.json"));
    let mut started = Vec::new();
    for (name, run) in fixture["runs"].as_object().unwrap().clone() {
        let fixture = fixture.clone();
        started.push((name.clone(), tokio::spawn(async move { run_one(&name, &run, &fixture).await })));
    }
    let mut wrong = Vec::new();
    for (name, task) in started {
        let got = task.await.unwrap();
        let want = &golden["runs"][&name];
        let edgar = name.starts_with("edgar");
        for key in ["exit", "stdout", "requests", "agents", "rows", "manifest", "records"] {
            let (mut w, mut g) = (want[key].clone(), got[key].clone());
            if edgar && key == "stdout" {
                (w, g) = (unordered(w.as_str().unwrap()), unordered(g.as_str().unwrap()));
            }
            if edgar && key == "requests" {
                for side in [&mut w, &mut g] {
                    for list in side.as_object_mut().unwrap().values_mut() {
                        list.as_array_mut().unwrap().sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                    }
                }
            }
            if w != g {
                wrong.push(format!("run {name} {key}:\n  python {w}\n  rust   {g}\n  stderr {}", got["stderr"]));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

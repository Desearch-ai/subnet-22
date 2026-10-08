//! The sample published against a pre-filled index: `prefill SAMPLE WORK PAGES`, then `run SAMPLE WORK THREADS [PASSES]`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::Instant;

use desearch::canonical::sha1_hex;
use publisher::index::{Current, VersionIndex};
use publisher::local::{LocalFeed, LocalUploads};
use publisher::records::{iso, SECOND};
use publisher::worker::{self, Job, Page};
use serde_json::json;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const NOW: i64 = 1_791_311_000 * SECOND;
const OLDER: &str = "2026-10-01T00:00:00+00:00";
const NEWER: &str = "2026-12-01T00:00:00+00:00";
const CACHE_BYTES: usize = 1 << 30;

struct Usage {
    cpu: f64,
    peak: u64,
}

fn usage() -> Usage {
    let mut raw: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut raw) };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    let peak = if cfg!(target_os = "macos") { raw.ru_maxrss as u64 } else { raw.ru_maxrss as u64 * 1024 };
    Usage { cpu: seconds(raw.ru_utime) + seconds(raw.ru_stime), peak }
}

fn jobs(sample: &Path) -> Vec<Job> {
    serde_json::from_slice(&std::fs::read(sample.join("jobs.json")).unwrap()).unwrap()
}

fn pages(reads: Vec<worker::Read>) -> Vec<Page> {
    reads.into_iter().flatten().filter_map(Result::ok).flat_map(|(pages, _)| pages).collect()
}

/// The Python harness's seed: half the sample's pages already indexed, a tenth of them at another version.
fn seed(pages: &[Page]) -> Vec<(String, Current)> {
    let mut seen = std::collections::HashSet::new();
    let mut seed = Vec::new();
    for page in pages {
        if !seen.insert(page.key.clone()) {
            continue;
        }
        let bucket = u32::from_str_radix(&sha1_hex(page.key.as_bytes())[..8], 16).unwrap() % 20;
        if bucket < 10 {
            continue;
        }
        let r = &page.record;
        let mut current = Current {
            url: r.url.clone(),
            version: page.version.clone(),
            fetched_at: if bucket < 14 { r.fetched_at.clone() } else { OLDER.into() },
            task_id: "seeded".into(),
            content_sha1: r.content_sha1.clone(),
            change_seq: None,
            change_row: None,
        };
        if bucket == 18 {
            (current.version, current.fetched_at, current.content_sha1) = ("0".repeat(40), OLDER.into(), "1".repeat(40));
        }
        if bucket == 19 {
            (current.version, current.fetched_at, current.content_sha1) = ("f".repeat(40), NEWER.into(), "2".repeat(40));
        }
        seed.push((page.key.clone(), current));
    }
    seed
}

/// A page another task published earlier, shaped like a real news URL.
fn synthetic(i: u64) -> (String, Current) {
    let domain = format!("news{}.example{}.com", i % 20_011, i % 7);
    let digest = sha1_hex(&i.to_le_bytes());
    let url = format!("https://www.{domain}/2026/10/{i:08}-{}-story-about-something", &digest[..24]);
    let key = format!("pages/{domain}/{}", sha1_hex(url.as_bytes()));
    let current = Current {
        url,
        version: sha1_hex(format!("v{i}").as_bytes()),
        fetched_at: iso(NOW - (i as i64 % 2_592_000) * SECOND),
        task_id: format!("{:016x}", (i / 1000).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        content_sha1: sha1_hex(format!("c{i}").as_bytes()),
        change_seq: Some((i / 20_000) as i64),
        change_row: Some((i % 20_000) as i64),
    };
    (key, current)
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()?.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()
}

fn prefill(sample: &Path, work: &Path, synthetic_pages: u64) {
    let template = work.join("rust-template");
    let _ = std::fs::remove_dir_all(&template);
    let jobs = jobs(sample);
    let uploads = LocalUploads::new(sample);
    let sample_pages = pages(worker::read_all(&jobs, &uploads, NOW, 8));
    let seeded = seed(&sample_pages);
    let index = VersionIndex::open(&template, CACHE_BYTES).unwrap();
    let started = Instant::now();
    index.store(&seeded).unwrap();
    let chunk = 100_000;
    for start in (0..synthetic_pages).step_by(chunk) {
        let entries: Vec<_> = (start..(start + chunk as u64).min(synthetic_pages)).map(synthetic).collect();
        index.store(&entries).unwrap();
    }
    index.flush().unwrap();
    index.compact();
    let total = synthetic_pages + seeded.len() as u64;
    let families = index.disk_bytes().unwrap();
    let tables: u64 = families.iter().map(|(_, b)| b).sum();
    drop(index);
    let report = json!({
        "pages": total,
        "seeded": seeded.len(),
        "seconds": started.elapsed().as_secs_f64(),
        "sst_bytes": families.iter().map(|(name, bytes)| (name.to_string(), *bytes)).collect::<HashMap<_, _>>(),
        "sst_bytes_per_page": tables as f64 / total as f64,
        "dir_bytes_per_page": dir_bytes(&template) as f64 / total as f64,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn run(sample: &Path, work: &Path, threads: usize, passes: usize) {
    let jobs = jobs(sample);
    let run_dir = work.join(format!("rust-run-{threads}"));
    let _ = std::fs::remove_dir_all(&run_dir);
    std::fs::create_dir_all(&run_dir).unwrap();
    let index_dir = run_dir.join("index");
    assert!(Command::new("cp").arg("-R").arg(work.join("rust-template")).arg(&index_dir).status().unwrap().success());
    let index = VersionIndex::open(&index_dir, CACHE_BYTES).unwrap();
    let uploads = LocalUploads::new(sample);
    let feed = LocalFeed::new(run_dir.join("changes"), 1).unwrap();

    let mut read_wall = Vec::new();
    let mut read_cpu = Vec::new();
    let mut reads = Vec::new();
    for _ in 0..passes.max(1) {
        uploads.traffic.bytes.store(0, Ordering::Relaxed);
        uploads.traffic.requests.store(0, Ordering::Relaxed);
        let before = usage();
        let started = Instant::now();
        reads = worker::read_all(&jobs, &uploads, NOW, threads);
        read_wall.push(started.elapsed().as_secs_f64());
        read_cpu.push(usage().cpu - before.cpu);
    }
    let retried: Vec<String> = reads.iter().flatten().filter_map(|r| r.as_ref().err()).map(|e| format!("{e:?}")).collect();
    let bytes_in = uploads.traffic.bytes.load(Ordering::Relaxed);
    let requests = uploads.traffic.requests.load(Ordering::Relaxed);
    let before = usage();
    let started = Instant::now();
    let batch = worker::write(&jobs, reads, &index, &feed, NOW).unwrap();
    let write_wall = started.elapsed().as_secs_f64();
    let write_cpu = usage().cpu - before.cpu;
    let tasks = jobs.len() as f64;
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (read_wall, read_cpu) = (median(&mut read_wall), median(&mut read_cpu));
    let new = batch.changes.iter().filter(|c| c.kind == publisher::changes::Kind::New).count();
    let report = json!({
        "threads": threads,
        "tasks": jobs.len(),
        "not_read": retried,
        "pages_new": new,
        "pages_changed": batch.changes.len() - new,
        "pages_unchanged": batch.unchanged.len(),
        "urls_failed": batch.failed.len(),
        "read_wall_s": read_wall,
        "read_cpu_s": read_cpu,
        "write_wall_s": write_wall,
        "write_cpu_s": write_cpu,
        "cpu_s_per_task": (read_cpu + write_cpu) / tasks,
        "read_cpu_s_per_task": read_cpu / tasks,
        "write_cpu_s_per_task": write_cpu / tasks,
        "wall_s_per_task": (read_wall + write_wall) / tasks,
        "tasks_per_min_per_core": 60.0 * tasks / (read_cpu + write_cpu),
        "tasks_per_min_pipelined": 60.0 * tasks / read_wall.max(write_wall),
        "peak_rss_bytes": usage().peak,
        "r2_bytes_in_per_task": bytes_in as f64 / tasks,
        "r2_requests_per_task": requests as f64 / tasks,
        "change_file_bytes": batch.change_bytes,
        "change_file_bytes_per_page": batch.change_bytes as f64 / batch.changes.len().max(1) as f64,
        "change_file_bytes_per_task": batch.change_bytes as f64 / tasks,
    });
    println!("{}", serde_json::to_string(&report).unwrap());
    drop(index);
    let _ = std::fs::remove_dir_all(&run_dir);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (sample, work) = (PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    std::fs::create_dir_all(&work).unwrap();
    match args[1].as_str() {
        "prefill" => prefill(&sample, &work, args[4].parse().unwrap()),
        "run" => run(&sample, &work, args[4].parse().unwrap(), args.get(5).map_or(3, |p| p.parse().unwrap())),
        other => panic!("unknown command {other}"),
    }
}

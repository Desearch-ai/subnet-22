//! Publishing a sample batch: `run SAMPLE WORK THREADS [PASSES]`, SAMPLE holding `jobs.json` and each job's `<task_id>.parquet`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use publisher::local::{LocalFeed, LocalUploads};
use publisher::records::SECOND;
use publisher::worker::{self, Job};
use serde_json::json;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const NOW: i64 = 1_791_311_000 * SECOND;

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

fn run(sample: &Path, work: &Path, threads: usize, passes: usize) {
    let jobs = jobs(sample);
    let run_dir = work.join(format!("rust-run-{threads}"));
    let _ = std::fs::remove_dir_all(&run_dir);
    std::fs::create_dir_all(&run_dir).unwrap();
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
    let batch = worker::write(&jobs, reads, &HashSet::new(), &feed, NOW).unwrap();
    let write_wall = started.elapsed().as_secs_f64();
    let write_cpu = usage().cpu - before.cpu;
    let tasks = jobs.len() as f64;
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (read_wall, read_cpu) = (median(&mut read_wall), median(&mut read_cpu));
    let report = json!({
        "threads": threads,
        "tasks": jobs.len(),
        "not_read": retried,
        "pages": batch.changes.len(),
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
    let _ = std::fs::remove_dir_all(&run_dir);
}

/// The tasks `synthetic` gave the first `tasks` thousand pages to.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (sample, work) = (PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    std::fs::create_dir_all(&work).unwrap();
    match args[1].as_str() {
        "run" => run(&sample, &work, args[4].parse().unwrap(), args.get(5).map_or(3, |p| p.parse().unwrap())),
        other => panic!("unknown command {other}"),
    }
}

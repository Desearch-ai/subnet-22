//! A heartbeat file an orchestrator can stat, so a stalled publisher is visible without an HTTP port.
//!
//! The file is refreshed after init, after each batch that was actually published, and when the
//! backlog is empty. A failed claim or a failed publish pass does not refresh it: idle stays
//! distinguishable from dead. The compose healthcheck (shipped disabled) fails once the timestamp
//! in the file is older than `PUBLISHER_HEARTBEAT_MAX_AGE_S`.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Beside the RocksDB directory on the container volume (`PUBLISHER_INDEX=/data/index`), not inside it.
/// A file inside the database directory would be an extra file RocksDB does not own.
pub const DEFAULT_PATH: &str = "/data/heartbeat";

/// Placeholder only. Leave the healthcheck disabled until logs show the real max staleness.
pub const DEFAULT_MAX_AGE_S: u64 = 600;

/// What one iteration of the publish loop did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// `write_batch` finished. Refresh the file and record the gap since the previous success.
    Published,
    /// `start_batch` returned no jobs. The process is waiting, not stuck. Refresh the file.
    EmptyBacklog,
    /// `start_batch` returned an error and this iteration published nothing. Do not refresh.
    ClaimFailed,
    /// `write_batch` returned an error. Do not refresh, even if earlier claims in this iteration failed too.
    PublishFailed,
}

/// Counters stored in the heartbeat so a stale file still says how far publishing got.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub batches: u64,
    pub tasks: u64,
    pub changed: u64,
    pub unchanged: u64,
    pub removed: u64,
    pub failed_urls: u64,
    pub lost: u64,
    pub retried: u64,
}

/// Gap observed when a batch is published. Idle time is included in `inter_batch` and excluded
/// from `staleness`, because an idle pass refreshes the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    pub inter_batch: Duration,
    pub staleness: Duration,
    pub max_inter_batch: Duration,
    pub max_staleness: Duration,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Record {
    unix_secs: u64,
    counters: Counters,
}

pub struct Tracker {
    path: PathBuf,
    /// When the file was last written. `None` until startup or the first refresh.
    written_at: Option<Instant>,
    /// When the previous successful batch finished. Idle refreshes do not move this.
    last_success: Option<Instant>,
    max_batch_gap: Duration,
    max_staleness: Duration,
    successes: u64,
}

impl Tracker {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Tracker {
            path: path.into(),
            written_at: None,
            last_success: None,
            max_batch_gap: Duration::ZERO,
            max_staleness: Duration::ZERO,
            successes: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// After Redis, R2, and the version index are up. Not a successful batch, so it does not start the gap.
    pub fn startup(&mut self, counters: &Counters) -> Result<()> {
        self.startup_at(counters, Instant::now(), unix_now())?;
        log::info!(
            "heartbeat file {} written after init; the compose healthcheck stays disabled until PUBLISHER_HEARTBEAT_MAX_AGE_S is set above the measured max staleness",
            self.path.display()
        );
        Ok(())
    }

    /// Apply one loop outcome. Only `Published` and `EmptyBacklog` write the file.
    pub fn record(&mut self, outcome: Outcome, counters: &Counters) -> Result<Option<Gap>> {
        match outcome {
            Outcome::Published => Ok(Some(self.published(counters)?)),
            Outcome::EmptyBacklog => {
                self.idle(counters)?;
                Ok(None)
            }
            // A failure must not refresh. Touching the file here would make a stuck publisher look idle.
            Outcome::ClaimFailed | Outcome::PublishFailed => Ok(None),
        }
    }

    pub fn log_periodic(&self) {
        let age = match self.written_at {
            Some(written) => format!("{:.1}s", written.elapsed().as_secs_f64()),
            None => "unwritten".to_string(),
        };
        log::info!(
            "max inter-batch gap {:.1}s over {} successful batches; max heartbeat staleness {:.1}s; heartbeat file age {age}; set PUBLISHER_HEARTBEAT_MAX_AGE_S above the max staleness before enabling the healthcheck",
            self.max_batch_gap.as_secs_f64(),
            self.successes,
            self.max_staleness.as_secs_f64(),
        );
    }

    fn published(&mut self, counters: &Counters) -> Result<Gap> {
        self.publish_at(counters, Instant::now(), unix_now())
    }

    fn idle(&mut self, counters: &Counters) -> Result<()> {
        self.idle_at(counters, Instant::now(), unix_now())
    }

    fn startup_at(&mut self, counters: &Counters, now: Instant, unix_secs: u64) -> Result<()> {
        // Startup is not a sample of either gap. It only gives the file a time to age from.
        self.write_new(counters, now, unix_secs)?;
        Ok(())
    }

    fn publish_at(&mut self, counters: &Counters, now: Instant, unix_secs: u64) -> Result<Gap> {
        let staleness = self.age_since_write(now);
        let inter_batch = match self.last_success {
            Some(previous) => now.saturating_duration_since(previous),
            // No batch yet: the gap is the time since the startup (or idle) write, which is `staleness`.
            None => staleness,
        };
        self.write_new(counters, now, unix_secs)?;
        if inter_batch > self.max_batch_gap {
            self.max_batch_gap = inter_batch;
        }
        self.last_success = Some(now);
        self.successes += 1;
        let gap = Gap {
            inter_batch,
            staleness,
            max_inter_batch: self.max_batch_gap,
            max_staleness: self.max_staleness,
        };
        log::info!(
            "successful batch: inter-batch gap {:.1}s (max {:.1}s), heartbeat staleness {:.1}s (max {:.1}s)",
            gap.inter_batch.as_secs_f64(),
            gap.max_inter_batch.as_secs_f64(),
            gap.staleness.as_secs_f64(),
            gap.max_staleness.as_secs_f64(),
        );
        Ok(gap)
    }

    fn idle_at(&mut self, counters: &Counters, now: Instant, unix_secs: u64) -> Result<()> {
        // Refresh so an empty queue does not age the file. Do not move `last_success`:
        // the inter-batch gap is the wall time between publishes, idle included.
        // Staleness excludes that idle time because this write is a healthy refresh.
        self.write_new(counters, now, unix_secs)?;
        Ok(())
    }

    fn age_since_write(&self, now: Instant) -> Duration {
        self.written_at
            .map(|written| now.saturating_duration_since(written))
            .unwrap_or_default()
    }

    fn write_new(&mut self, counters: &Counters, now: Instant, unix_secs: u64) -> Result<()> {
        let staleness = self.age_since_write(now);
        if self.written_at.is_some() && staleness > self.max_staleness {
            self.max_staleness = staleness;
        }
        write_atomic(&self.path, &render(unix_secs, counters))?;
        self.written_at = Some(now);
        Ok(())
    }
}

/// The healthcheck's rule, shared with `healthcheck.sh`: older than the limit fails, equal to it does not.
pub fn is_stale(written_unix_secs: u64, now_unix_secs: u64, max_age_s: u64) -> bool {
    now_unix_secs.saturating_sub(written_unix_secs) > max_age_s
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// `unix_secs` is the first line. `healthcheck.sh` reads that line with sed; keep the key stable.
fn render(unix_secs: u64, counters: &Counters) -> String {
    format!(
        "unix_secs={unix_secs}\nbatches={}\ntasks={}\nchanged={}\nunchanged={}\nremoved={}\nfailed_urls={}\nlost={}\nretried={}\n",
        counters.batches, counters.tasks, counters.changed, counters.unchanged, counters.removed, counters.failed_urls, counters.lost, counters.retried,
    )
}

#[cfg(test)]
fn parse(text: &str) -> std::result::Result<Record, String> {
    let mut unix_secs = None;
    let mut counters = Counters::default();
    let mut fields = 0usize;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("not a heartbeat field: {line:?}"));
        };
        let value = value
            .parse::<u64>()
            .map_err(|_| format!("not a number: {line:?}"))?;
        match key {
            "unix_secs" => unix_secs = Some(value),
            "batches" => counters.batches = value,
            "tasks" => counters.tasks = value,
            "changed" => counters.changed = value,
            "unchanged" => counters.unchanged = value,
            "removed" => counters.removed = value,
            "failed_urls" => counters.failed_urls = value,
            "lost" => counters.lost = value,
            "retried" => counters.retried = value,
            other => return Err(format!("unknown heartbeat field {other}")),
        }
        fields += 1;
    }
    if fields != 9 {
        return Err(format!("heartbeat has {fields} fields, want 9"));
    }
    Ok(Record {
        unix_secs: unix_secs.ok_or_else(|| "heartbeat has no unix_secs".to_string())?,
        counters,
    })
}

/// Write `path` by creating a temp file in the same directory, syncing it, and renaming it over `path`.
/// Rename on one filesystem is atomic, so a reader sees the previous file or the new one, never a torn write.
pub fn write_atomic(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = temp_path(path);
    let wrote = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        file.write_all(body.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = wrote {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("writing {}", tmp.display()));
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("replacing {}", path.display()));
    }
    Ok(())
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.tmp", std::process::id()));
    path.with_file_name(name)
}

struct StdoutLog;

impl log::Log for StdoutLog {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            println!("INFO {}: {}", record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: StdoutLog = StdoutLog;

/// Operational lines go to stdout, the same stream the rest of `publisher serve` uses.
pub fn init_info_log() {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("publisher-heartbeat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn counters(batches: u64, tasks: u64) -> Counters {
        Counters {
            batches,
            tasks,
            changed: batches,
            ..Counters::default()
        }
    }

    #[test]
    fn staleness_fails_only_when_the_file_is_older_than_the_limit() {
        assert!(!is_stale(1_000, 1_600, 600));
        assert!(!is_stale(1_000, 1_000, 600));
        assert!(is_stale(1_000, 1_601, 600));
        assert!(
            !is_stale(2_000, 1_000, 600),
            "a timestamp ahead of the clock is not stale"
        );
    }

    #[test]
    fn a_successful_batch_writes_the_heartbeat_and_the_gap() {
        let dir = scratch("success");
        let path = dir.join("heartbeat");
        let mut tracker = Tracker::new(&path);
        let t0 = Instant::now();
        tracker.startup_at(&Counters::default(), t0, 1_000).unwrap();
        let started = parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(started.unix_secs, 1_000);
        assert_eq!(started.counters, Counters::default());

        let gap = tracker
            .publish_at(&counters(1, 40), t0 + Duration::from_secs(12), 1_012)
            .unwrap();
        assert_eq!(gap.inter_batch, Duration::from_secs(12));
        assert_eq!(gap.staleness, Duration::from_secs(12));
        assert_eq!(gap.max_inter_batch, Duration::from_secs(12));
        assert_eq!(gap.max_staleness, Duration::from_secs(12));
        assert_eq!(tracker.successes, 1);

        let written = parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written.unix_secs, 1_012);
        assert_eq!(written.counters.batches, 1);
        assert_eq!(written.counters.tasks, 40);

        let later = tracker
            .publish_at(&counters(2, 80), t0 + Duration::from_secs(20), 1_020)
            .unwrap();
        assert_eq!(later.inter_batch, Duration::from_secs(8));
        assert_eq!(
            later.max_inter_batch,
            Duration::from_secs(12),
            "the running max keeps the longer gap"
        );
        assert_eq!(tracker.successes, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_idle_pass_refreshes_the_heartbeat_and_a_failed_pass_does_not() {
        let dir = scratch("idle-fail");
        let path = dir.join("heartbeat");
        let mut tracker = Tracker::new(&path);
        let t0 = Instant::now();
        tracker.startup_at(&Counters::default(), t0, 5_000).unwrap();

        let gap = tracker.record(Outcome::Published, &counters(1, 4)).unwrap();
        // `record` uses the wall clock; the controlled-clock assertions are below.
        assert!(gap.is_some());
        let after_publish = std::fs::read_to_string(&path).unwrap();
        assert!(after_publish.contains("batches=1"));
        assert_eq!(tracker.successes, 1);
        let max_after_publish = tracker.max_batch_gap;

        tracker
            .record(Outcome::ClaimFailed, &counters(9, 9))
            .unwrap();
        tracker
            .record(Outcome::PublishFailed, &counters(8, 8))
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            after_publish,
            "a failed claim or publish pass must not refresh"
        );
        assert_eq!(tracker.successes, 1);
        assert_eq!(tracker.max_batch_gap, max_after_publish);

        tracker
            .record(Outcome::EmptyBacklog, &counters(1, 7))
            .unwrap();
        let idle = parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            idle.counters.tasks, 7,
            "an empty backlog refreshes the file"
        );
        assert_eq!(
            idle.counters.batches, 1,
            "idle is not another successful batch"
        );
        assert_eq!(tracker.successes, 1);
        assert_eq!(
            tracker.max_batch_gap, max_after_publish,
            "idle does not count as an inter-batch gap"
        );

        // Idle stretches the wall gap between batches and must not stretch heartbeat staleness the same way.
        tracker.startup_at(&counters(1, 7), t0, 5_000).unwrap();
        tracker
            .publish_at(&counters(2, 10), t0 + Duration::from_secs(10), 5_010)
            .unwrap();
        tracker
            .idle_at(&counters(2, 10), t0 + Duration::from_secs(100), 5_100)
            .unwrap();
        let after_idle = tracker
            .publish_at(&counters(3, 12), t0 + Duration::from_secs(106), 5_106)
            .unwrap();
        assert_eq!(
            after_idle.inter_batch,
            Duration::from_secs(96),
            "inter-batch gap is wall time between publishes"
        );
        assert_eq!(
            after_idle.staleness,
            Duration::from_secs(6),
            "staleness is only the time since the idle refresh"
        );
        assert_eq!(
            after_idle.max_staleness,
            Duration::from_secs(90),
            "the long idle itself was a refresh, so its wait was sampled then"
        );
        assert!(is_stale(5_100, 5_106, 5));
        assert!(!is_stale(5_100, 5_106, 6));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_heartbeat_is_replaced_by_rename_so_a_reader_never_sees_a_torn_file() {
        let dir = scratch("atomic");
        let path = dir.join("heartbeat");
        let mut tracker = Tracker::new(&path);
        let t0 = Instant::now();
        tracker.startup_at(&Counters::default(), t0, 100).unwrap();
        let mut held = std::fs::File::open(&path).unwrap();

        tracker
            .publish_at(&counters(4, 11), t0 + Duration::from_secs(3), 103)
            .unwrap();
        let mut previous = String::new();
        held.read_to_string(&mut previous).unwrap();
        let current = std::fs::read_to_string(&path).unwrap();
        assert!(
            previous.starts_with("unix_secs=100\n"),
            "the fd opened before the write still sees the old inode: {previous:?}"
        );
        assert!(current.starts_with("unix_secs=103\n"));
        assert!(current.contains("batches=4"));
        assert_ne!(previous, current);
        parse(&previous).unwrap();
        parse(&current).unwrap();

        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["heartbeat".to_string()],
            "the temp file does not stay behind"
        );

        let stop = Arc::new(AtomicBool::new(false));
        let reading = path.clone();
        let stop_flag = stop.clone();
        let reader = std::thread::spawn(move || {
            let mut seen = 0u64;
            while !stop_flag.load(Ordering::Relaxed) {
                match std::fs::read_to_string(&reading) {
                    Ok(body) => {
                        parse(&body)
                            .unwrap_or_else(|error| panic!("torn heartbeat {error}: {body:?}"));
                        seen += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => panic!("{error}"),
                }
            }
            seen
        });
        for n in 0..40u64 {
            tracker
                .publish_at(&counters(n, n), t0 + Duration::from_secs(10 + n), 200 + n)
                .unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        assert!(
            reader.join().unwrap() > 0,
            "the reader ran alongside the writes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_healthcheck_fails_when_the_file_is_missing_or_older_than_the_limit() {
        let dir = scratch("healthcheck");
        let path = dir.join("heartbeat");
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("healthcheck.sh");
        let run = |max: &str| {
            std::process::Command::new("sh")
                .arg(&script)
                .env("PUBLISHER_HEARTBEAT_PATH", &path)
                .env("PUBLISHER_HEARTBEAT_MAX_AGE_S", max)
                .output()
                .unwrap()
        };

        let missing = run("600");
        assert!(!missing.status.success(), "a missing file is unhealthy");
        assert!(String::from_utf8_lossy(&missing.stderr).contains("missing"));

        let now = unix_now();
        std::fs::write(&path, render(now, &counters(3, 9))).unwrap();
        let fresh = run("600");
        assert!(
            fresh.status.success(),
            "a file written now is healthy: {}",
            String::from_utf8_lossy(&fresh.stderr)
        );

        std::fs::write(&path, render(now.saturating_sub(86_400), &counters(3, 9))).unwrap();
        let stale = run("600");
        assert!(
            !stale.status.success(),
            "a day-old timestamp is older than 600s"
        );
        let stderr = String::from_utf8_lossy(&stale.stderr);
        assert!(stderr.contains("stale"), "{stderr}");
        assert!(is_stale(now.saturating_sub(86_400), now, 600));

        std::fs::write(&path, "not a heartbeat\n").unwrap();
        let garbled = run("600");
        assert!(
            !garbled.status.success(),
            "a file without unix_secs is unhealthy"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

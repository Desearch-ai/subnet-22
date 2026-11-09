//! Bulk sources as pages: the batch files `python -m bulk` writes, published into the bulk feed as rows built exactly as crawled pages are, each under an id no task holds.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::Deserialize;

use crate::changes;
use crate::reading::{read_rows, LocalFile};
use crate::records::{build_record, iso, record_key, record_version, Context as RecordContext, Row, Window, MIN_US};
use crate::worker::{page_changes, publishable, ChangeFeed, Chosen, Page};

/// Every bulk row's id starts with this; a task the task API hands out never does, so no miner's take-back names a bulk page.
pub const TASK_PREFIX: &str = "bulk-";
pub const MANIFEST: &str = "manifest.json";
/// The batch files of a run already published, one `<file> <first seq> <last seq>` line each, so a run picks up where it stopped.
pub const PUBLISHED: &str = "published.txt";

/// What `python -m bulk` writes beside a run's batch files once the run is complete.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub source: String,
    pub run: String,
    pub as_of: String,
    pub files: Vec<String>,
}

impl Manifest {
    pub fn read(dir: &Path) -> Result<Manifest> {
        let path = dir.join(MANIFEST);
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {}; is the run complete?", path.display()))?)?;
        if manifest.source.is_empty() || manifest.run.is_empty() || manifest.source.contains(char::is_whitespace) {
            bail!("{} names no usable source and run", path.display());
        }
        Ok(manifest)
    }

    /// The id of one row: the run, the batch file's place in it and the row's place in the file, so one take-back names one page.
    pub fn row_id(&self, file: usize, row: usize) -> String {
        format!("{TASK_PREFIX}{}-{}-{file:06}-{row}", self.source, self.run)
    }
}

pub struct Settings {
    /// Pages in one change file at most.
    pub max_rows: usize,
    /// Page text in one bulk file at most, in bytes, so readers that load a whole file stay light.
    pub max_text_bytes: usize,
    /// Pages published an hour at most, so the planner's budget takes them as they come; 0 publishes as fast as files are written.
    pub pages_per_hour: f64,
}

#[derive(Debug, Default)]
pub struct Report {
    pub files: usize,
    pub pages: usize,
    pub change_files: Vec<u64>,
}

/// The pages of one batch file, each under `row_id` of its place in the file: each row's own fetch time stands, never later than now, and no miner or validator is named.
pub fn pages_of(rows: &[Row], row_id: impl Fn(usize) -> String, now: i64) -> Result<Vec<Page>> {
    let urls: HashSet<&str> = rows.iter().filter_map(|row| row.url.as_deref()).collect();
    let mut chosen = Chosen::default();
    for (at, row) in rows.iter().enumerate().filter(|(_, row)| publishable(row, &urls)) {
        let task_id = row_id(at);
        let context = RecordContext {
            task_id: &task_id,
            miner: "",
            window: Window { earliest: MIN_US, latest: now },
            captured_at: now,
            validator: None,
            validators: &[],
        };
        let record = build_record(row, &context)?;
        chosen.offer(Page { key: record_key(&record)?, version: record_version(&record), record });
    }
    Ok(chosen.into_pages())
}

/// Consecutive runs of pages, each within the row and text limits.
pub fn split(pages: Vec<Page>, settings: &Settings) -> Vec<Vec<Page>> {
    let mut chunks: Vec<Vec<Page>> = Vec::new();
    let mut held = 0;
    for page in pages {
        let text = page.record.text.len();
        match chunks.last_mut() {
            Some(chunk) if chunk.len() < settings.max_rows && held + text <= settings.max_text_bytes => {
                held += text;
                chunk.push(page);
            }
            _ => {
                held = text;
                chunks.push(vec![page]);
            }
        }
    }
    chunks
}

/// Publishes every batch file of the run in `dir` not published yet into the bulk feed, as page rows only; `now` gives microseconds since the epoch.
pub fn publish_run(dir: &Path, feed: &dyn ChangeFeed, settings: &Settings, now: &dyn Fn() -> i64) -> Result<Report> {
    let manifest = Manifest::read(dir)?;
    let done = published(dir)?;
    let mut report = Report::default();
    let started = Instant::now();
    for (file, name) in manifest.files.iter().enumerate().filter(|(_, name)| !done.contains(name.as_str())) {
        let rows = batch_rows(&dir.join(name))?;
        let pages = pages_of(&rows, |row| manifest.row_id(file + 1, row), now())?;
        let mut seqs = Vec::new();
        for chunk in split(pages, settings) {
            report.pages += chunk.len();
            let rows = chunk.len();
            seqs.push(feed.append(changes::encode(&page_changes(chunk, &iso(now())))?, rows)?);
            pace(settings, report.pages, started);
        }
        note_published(dir, name, &seqs)?;
        report.change_files.extend(&seqs);
        report.files += 1;
    }
    Ok(report)
}

fn batch_rows(path: &Path) -> Result<Vec<Row>> {
    let reader = SerializedFileReader::new(File::open(path)?).with_context(|| format!("reading the footer of {}", path.display()))?;
    let rows = usize::try_from(reader.metadata().file_metadata().num_rows())?;
    read_rows(&LocalFile::open(path)?, rows)?.with_context(|| format!("{} is not a readable batch file", path.display()))
}

fn published(dir: &Path) -> Result<HashSet<String>> {
    match std::fs::read_to_string(dir.join(PUBLISHED)) {
        Ok(text) => Ok(text.lines().filter_map(|line| line.split_whitespace().next()).map(str::to_string).collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(HashSet::new()),
        Err(error) => Err(error.into()),
    }
}

fn note_published(dir: &Path, name: &str, seqs: &[u64]) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(dir.join(PUBLISHED))?;
    let (first, last) = (seqs.first().copied().unwrap_or(0), seqs.last().copied().unwrap_or(0));
    writeln!(file, "{name} {first} {last}")?;
    Ok(file.sync_data()?)
}

fn pace(settings: &Settings, pages: usize, started: Instant) {
    if settings.pages_per_hour > 0.0 {
        let due = Duration::from_secs_f64(pages as f64 / settings.pages_per_hour * 3600.0);
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            std::thread::sleep(wait);
        }
    }
}

//! A build's run folder against the source's state: its new and changed pages written as a run of their own.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use arrow_array::{Array, BooleanArray, LargeStringArray, RecordBatch, StringArray};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_select::filter::filter_record_batch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rusqlite::{params, Connection};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::sources::Build;

pub const MANIFEST: &str = "manifest.json";
pub const REMOVED: &str = "removed.txt";
pub const ROWS_PER_FILE: usize = 5_000;
pub const MIN_SEEN_SHARE: f64 = 0.95;
pub const LARGE_SHARE: f64 = 0.30;
/// Columns that change on every fetch without the page changing.
const UNHASHED: &[&str] = &["fetched_at", "elapsed_ms", "html", "html_bytes", "html_sha256"];

/// A run that publishes and hides nothing, because its input looks broken.
#[derive(Debug)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

#[derive(Debug, Default, Serialize)]
pub struct Counts {
    pub seen: usize,
    pub new: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub removed: usize,
    /// Pages missing only because their whole site is missing this run, kept until the source names them removed.
    pub held: usize,
    pub known: usize,
    pub published: String,
    pub large: bool,
    #[serde(skip)]
    pub hashes: HashMap<String, String>,
    #[serde(skip)]
    pub gone: Vec<String>,
}

pub fn open_state(path: &Path) -> Result<Connection> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let con = Connection::open(path)?;
    con.execute_batch(
        "CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, build TEXT NOT NULL, hash TEXT NOT NULL,
         run TEXT NOT NULL, seen TEXT NOT NULL, removed TEXT)",
    )?;
    Ok(con)
}

fn manifest(folder: &Path) -> Result<Value> {
    let path = folder.join(MANIFEST);
    serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {}; is the run complete?", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn files(manifest: &Value) -> Vec<String> {
    manifest["files"].as_array().map(|files| files.iter().filter_map(|f| f.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

fn urls(batch: &RecordBatch) -> Result<Vec<String>> {
    let column = batch.column_by_name("url").context("a row file without a url column")?;
    if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(strings.iter().map(|u| u.unwrap_or_default().to_string()).collect());
    }
    if let Some(strings) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(strings.iter().map(|u| u.unwrap_or_default().to_string()).collect());
    }
    bail!("the url column is not text")
}

struct Hasher(Sha256);

impl std::fmt::Write for Hasher {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.update(s.as_bytes());
        Ok(())
    }
}

/// The text with the build's volatile lines left out.
pub fn stable_text(text: &str, volatile_lines: &[&str]) -> String {
    text.split('\n').filter(|line| !volatile_lines.iter().any(|prefix| line.starts_with(prefix))).collect::<Vec<_>>().join("\n")
}

/// One hash a row over every column but the fetch's own and the build's volatile ones, so the same page fetched again hashes the same.
pub fn row_hashes(batch: &RecordBatch, build: &Build) -> Result<Vec<String>> {
    let options = FormatOptions::default().with_null("\u{0}null");
    let schema = batch.schema();
    let mut columns = Vec::new();
    for (i, field) in schema.fields().iter().enumerate() {
        let name = field.name().as_str();
        let derived_text = name == "text_sha256" && !build.volatile_lines.is_empty();
        if !UNHASHED.contains(&name) && !build.volatile_columns.contains(&name) && !derived_text {
            columns.push((field.name().clone(), ArrayFormatter::try_new(batch.column(i).as_ref(), &options)?));
        }
    }
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut hasher = Hasher(Sha256::new());
        for (name, column) in &columns {
            hasher.0.update(name.as_bytes());
            hasher.0.update([0]);
            if name == "text" && !build.volatile_lines.is_empty() {
                hasher.0.update(stable_text(&column.value(row).to_string(), build.volatile_lines).as_bytes());
            } else {
                column.value(row).write(&mut hasher)?;
            }
            hasher.0.update([0x1e]);
        }
        let mut hex = String::with_capacity(64);
        for byte in hasher.0.finalize() {
            write!(hex, "{byte:02x}")?;
        }
        out.push(hex);
    }
    Ok(out)
}

/// Writes rows into numbered parquet files of at most about `ROWS_PER_FILE` rows, and the manifest once closed.
struct RunWriter<'a> {
    out: &'a Path,
    source: String,
    writer: Option<ArrowWriter<File>>,
    files: Vec<String>,
    in_file: usize,
    rows: usize,
}

impl<'a> RunWriter<'a> {
    fn new(out: &'a Path, source: &str) -> Result<Self> {
        if out.join(MANIFEST).exists() {
            bail!("{} already holds a run; give each run its own folder", out.display());
        }
        std::fs::create_dir_all(out)?;
        Ok(RunWriter { out, source: source.to_string(), writer: None, files: Vec::new(), in_file: 0, rows: 0 })
    }

    fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.writer.is_none() {
            let name = format!("{}-{:06}.parquet", self.source, self.files.len() + 1);
            let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::default())).build();
            self.writer = Some(ArrowWriter::try_new(File::create(self.out.join(&name))?, batch.schema(), Some(props))?);
            self.files.push(name);
        }
        self.writer.as_mut().unwrap().write(batch)?;
        self.in_file += batch.num_rows();
        self.rows += batch.num_rows();
        if self.in_file >= ROWS_PER_FILE {
            self.close_file()?;
        }
        Ok(())
    }

    fn close_file(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        self.in_file = 0;
        Ok(())
    }

    fn close(mut self, run: &str, as_of: &str, counts: Value) -> Result<()> {
        self.close_file()?;
        let manifest = json!({
            "source": self.source, "run": run, "as_of": as_of, "files": self.files,
            "removed": null, "rows": self.rows, "counts": counts,
        });
        std::fs::write(self.out.join(MANIFEST), serde_json::to_vec_pretty(&manifest)?)?;
        Ok(())
    }
}

fn each_batch(folder: &Path, manifest: &Value, mut f: impl FnMut(RecordBatch) -> Result<()>) -> Result<()> {
    for name in files(manifest) {
        let path = folder.join(&name);
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).with_context(|| format!("opening {}", path.display()))?)?
            .with_batch_size(1024)
            .build()?;
        for batch in reader {
            f(batch?)?;
        }
    }
    Ok(())
}

/// The host of a URL, lowercase as written; a site whose every page is missing may be down, not gone.
pub fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())]
}

fn removed_of(folder: &Path) -> Result<Vec<String>> {
    let path = folder.join(REMOVED);
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(std::fs::read_to_string(path)?.split_whitespace().map(str::to_string).collect())
}

/// Writes the build's new and changed pages to `out` as run `run`, and counts what changed.
pub fn diff(con: &Connection, build: &Build, folder: &Path, out: &Path, run: &str, as_of: &str) -> Result<Counts> {
    // A page is compared with its last hash under any build of the source, so a page that moves between split builds is no change; hiding still goes by this build's pages.
    let mut known: HashMap<String, String> = HashMap::new();
    let mut query = con.prepare("SELECT url, hash FROM pages WHERE build = ?1 AND removed IS NULL")?;
    for row in query.query_map(params![build.name], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))? {
        let (url, hash) = row?;
        known.insert(url, hash);
    }
    let mut elsewhere: HashMap<String, String> = HashMap::new();
    let mut query = con.prepare("SELECT url, hash FROM pages WHERE build != ?1 AND removed IS NULL")?;
    for row in query.query_map(params![build.name], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))? {
        let (url, hash) = row?;
        elsewhere.insert(url, hash);
    }
    let manifest = manifest(folder)?;
    let source = manifest["source"].as_str().context("the build's manifest names no source")?.to_string();
    let mut counts = Counts { known: known.len(), ..Counts::default() };
    let mut writer = RunWriter::new(out, &source)?;
    each_batch(folder, &manifest, |batch| {
        let urls = urls(&batch)?;
        let hashes = row_hashes(&batch, build)?;
        let mut keep = Vec::with_capacity(urls.len());
        for (url, hash) in urls.into_iter().zip(hashes) {
            if counts.hashes.contains_key(&url) {
                keep.push(false);
                continue;
            }
            let changed = match known.get(&url).or_else(|| elsewhere.get(&url)) {
                Some(before) if *before == hash => {
                    counts.unchanged += 1;
                    false
                }
                Some(_) => {
                    counts.changed += 1;
                    true
                }
                None => {
                    counts.new += 1;
                    true
                }
            };
            keep.push(changed);
            counts.hashes.insert(url, hash);
        }
        writer.add(&filter_record_batch(&batch, &BooleanArray::from(keep))?)
    })?;
    counts.seen = counts.hashes.len();
    if build.full_list && !known.is_empty() && (counts.seen as f64) < MIN_SEEN_SHARE * known.len() as f64 {
        return Err(Refused(format!(
            "{}: {} pages, under {:.0}% of the {} known",
            build.name,
            counts.seen,
            MIN_SEEN_SHARE * 100.0,
            known.len()
        ))
        .into());
    }
    let named: HashSet<String> = removed_of(folder)?.into_iter().filter(|u| !counts.hashes.contains_key(u)).collect();
    let mut gone: HashSet<String> = named.clone();
    if build.full_list {
        let live_hosts: HashSet<&str> = counts.hashes.keys().map(|u| host_of(u)).collect();
        for url in known.keys().filter(|u| !counts.hashes.contains_key(*u) && !named.contains(*u)) {
            if live_hosts.contains(host_of(url)) {
                gone.insert(url.clone());
            } else {
                counts.held += 1;
            }
        }
    }
    counts.gone = gone.into_iter().collect();
    counts.gone.sort();
    counts.removed = counts.gone.len();
    counts.large = !build.volatile && !known.is_empty() && (counts.new + counts.changed) as f64 > LARGE_SHARE * known.len() as f64;
    writer.close(run, as_of, json!({"pages": counts.new + counts.changed, "new": counts.new, "changed": counts.changed}))?;
    Ok(counts)
}

/// Records a published diff, so the next run compares with it.
pub fn keep(con: &mut Connection, build: &Build, counts: &Counts, run: &str, as_of: &str) -> Result<()> {
    let tx = con.transaction()?;
    {
        let mut upsert = tx.prepare(
            "INSERT INTO pages (url, build, hash, run, seen, removed) VALUES (?1, ?2, ?3, ?4, ?5, NULL)
             ON CONFLICT(url) DO UPDATE SET build = excluded.build, seen = excluded.seen, removed = NULL,
             run = CASE WHEN pages.hash = excluded.hash AND pages.removed IS NULL THEN pages.run ELSE excluded.run END,
             hash = excluded.hash",
        )?;
        for (url, hash) in &counts.hashes {
            upsert.execute(params![url, build.name, hash, run, as_of])?;
        }
        let mut gone = tx.prepare("UPDATE pages SET removed = ?1 WHERE url = ?2 AND build = ?3")?;
        for url in &counts.gone {
            gone.execute(params![run, url, build.name])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn hidden_urls(con: &Connection) -> Result<Vec<String>> {
    let mut query = con.prepare("SELECT url FROM pages WHERE removed IS NOT NULL ORDER BY url")?;
    let urls = query.query_map([], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(urls)
}

/// Records runs published before the runner, so a source's first refresh sends only what changed since.
pub fn seed(con: &mut Connection, build: &Build, folders: &[std::path::PathBuf]) -> Result<usize> {
    let mut pages = 0;
    for folder in folders {
        let manifest = manifest(folder)?;
        let run = manifest["run"].as_str().context("the manifest names no run")?.to_string();
        let as_of = manifest["as_of"].as_str().context("the manifest names no as_of")?.to_string();
        let mut counts = Counts::default();
        each_batch(folder, &manifest, |batch| {
            for (url, hash) in urls(&batch)?.into_iter().zip(row_hashes(&batch, build)?) {
                counts.hashes.insert(url, hash);
            }
            Ok(())
        })?;
        pages += counts.hashes.len();
        keep(con, build, &counts, &run, &as_of)?;
    }
    Ok(pages)
}

/// The rows of a run folder, for tests and checks.
pub fn read_urls(folder: &Path) -> Result<Vec<String>> {
    let manifest = manifest(folder)?;
    let mut out = Vec::new();
    each_batch(folder, &manifest, |batch| {
        out.extend(urls(&batch)?);
        Ok(())
    })?;
    Ok(out)
}

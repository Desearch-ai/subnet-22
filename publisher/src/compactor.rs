//! Brings the current state up to the change log a cycle at a time: deltas per shard, merges, then the manifests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use desearch::feeds::CHANGES;
use desearch::r2::{self, Bucket};
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;

use crate::current::{self, shard_dir, shard_of, ChangeRow, FileRef, Manifest, PageRow, Retired, Withdrawals, SHARDS};
use crate::index::VersionIndex;

/// Objects in a bucket or a folder, as the compactor reads and writes them.
pub trait Store: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Bytes>>;
    fn put(&self, key: &str, body: Bytes) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
    /// When the object was last written, in seconds since the epoch; None when it does not exist.
    fn modified(&self, key: &str) -> Result<Option<f64>>;
}

pub struct LocalStore {
    pub root: PathBuf,
}

impl Store for LocalStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        match std::fs::read(self.root.join(key)) {
            Ok(body) => Ok(Some(Bytes::from(body))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {key}")),
        }
    }

    fn put(&self, key: &str, body: Bytes) -> Result<()> {
        let path = self.root.join(key);
        std::fs::create_dir_all(path.parent().expect("a key has a folder"))?;
        let partial = path.with_extension("partial");
        std::fs::write(&partial, body)?;
        Ok(std::fs::rename(partial, path)?)
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.root.join(key)) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
            _ => Ok(()),
        }
    }

    fn modified(&self, key: &str) -> Result<Option<f64>> {
        match std::fs::metadata(self.root.join(key)) {
            Ok(meta) => Ok(Some(meta.modified()?.duration_since(UNIX_EPOCH)?.as_secs_f64())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}

pub struct R2Store {
    pub bucket: Bucket,
    pub handle: Handle,
}

/// The current state under `prefix` in one store and everything else, the change log and snapshots, in another.
pub struct Split {
    pub prefix: String,
    pub current: Box<dyn Store>,
    pub log: Box<dyn Store>,
}

impl Split {
    fn holding(&self, key: &str) -> &dyn Store {
        if key.starts_with(&self.prefix) {
            self.current.as_ref()
        } else {
            self.log.as_ref()
        }
    }
}

impl Store for Split {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        self.holding(key).get(key)
    }

    fn put(&self, key: &str, body: Bytes) -> Result<()> {
        self.holding(key).put(key, body)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.holding(key).delete(key)
    }

    fn modified(&self, key: &str) -> Result<Option<f64>> {
        self.holding(key).modified(key)
    }
}

const PART_BYTES: usize = 32 << 20;

impl Store for R2Store {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        match self.handle.block_on(self.bucket.get(key)) {
            Ok(body) => Ok(Some(body)),
            Err(r2::Error::Missing) => Ok(None),
            Err(error) => Err(anyhow!("reading {key}: {error}")),
        }
    }

    fn put(&self, key: &str, body: Bytes) -> Result<()> {
        let kind = if key.ends_with(".json") { r2::JSON } else { "application/vnd.apache.parquet" };
        let put = if body.len() > PART_BYTES {
            self.handle.block_on(self.bucket.put_in_parts(key, body, kind, PART_BYTES, 4))
        } else {
            self.handle.block_on(self.bucket.put(key, body, kind, None))
        };
        put.map_err(|error| anyhow!("writing {key}: {error}"))
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.handle.block_on(self.bucket.delete(key)).map_err(|error| anyhow!("deleting {key}: {error}"))
    }

    fn modified(&self, key: &str) -> Result<Option<f64>> {
        let head = self.handle.block_on(self.bucket.head(key)).map_err(|error| anyhow!("reading {key}: {error}"))?;
        Ok(head.and_then(|head| head.modified).map(|at| at as f64))
    }
}

pub struct Settings {
    /// Where the shards live, e.g. `current/`.
    pub prefix: String,
    /// Local room for rows waiting to be sorted into their shard.
    pub work: PathBuf,
    pub files_per_cycle: i64,
    /// Once caught up, a cycle starts this long after the last one, so each delta holds a while of the log.
    pub every_s: f64,
    /// A shard's base is rewritten once its deltas hold this share of it.
    pub merge_ratio: f64,
    /// Deltas of one size combined into one of the next size, so a reader opens few files and a base is not rewritten for them.
    pub fanout: usize,
    pub merges_per_cycle: usize,
    /// Replaced files stay this long, so a reader mid-read is not cut off.
    pub retire_after_s: f64,
    /// Rows held in memory before they are written to the shard's spill files.
    pub spill_bytes: usize,
    pub threads: usize,
    /// Shards merged at once; each holds about a row group of every file it merges plus the new base.
    pub merge_threads: usize,
    /// A number still unwritten this long after a later one was written was never going to be, and is skipped.
    pub hole_wait_s: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            prefix: "current/".into(),
            work: PathBuf::from("compactor-work"),
            files_per_cycle: 300,
            merge_ratio: 1.0,
            every_s: 900.0,
            fanout: 12,
            merges_per_cycle: 64,
            retire_after_s: 86_400.0,
            spill_bytes: 1 << 30,
            threads: 8,
            merge_threads: 4,
            hole_wait_s: 600.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub from: i64,
    pub through: i64,
    pub rows: u64,
    pub withdrawn: u64,
    pub deltas: usize,
    pub merged: usize,
    pub combined: usize,
    pub deleted: usize,
    /// Numbers never written, passed over.
    pub skipped: Vec<i64>,
    /// The cycle stopped at a number not written yet.
    pub waiting: bool,
}

/// Tasks taken back; their rows count as removals until every shard's base is rewritten past the last row of theirs read.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Withdrawn {
    pub tasks: BTreeMap<String, Withdrawal>,
}

impl Withdrawn {
    pub fn by_task(&self) -> Withdrawals {
        self.tasks.iter().map(|(task_id, taken)| (task_id.clone(), taken.seq)).collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Withdrawal {
    /// The change file that took the task back.
    pub seq: i64,
    /// The last change file read with a row of the task: a publisher racing the take-back may log one after it.
    pub last: i64,
}

pub fn manifest_key(prefix: &str, shard: usize) -> String {
    format!("{prefix}{}/manifest.json", shard_dir(shard))
}

pub fn withdrawn_key(prefix: &str) -> String {
    format!("{prefix}withdrawn.json")
}

fn file_key(prefix: &str, shard: usize, kind: &str, through: i64) -> String {
    format!("{prefix}{}/{kind}-{through:012}.parquet", shard_dir(shard))
}

/// Runs `work` on every item across `threads` threads, keeping the results in item order.
pub(crate) fn each<T: Sync, R: Send>(items: &[T], threads: usize, work: impl Fn(&T) -> Result<R> + Sync) -> Result<Vec<R>> {
    let next = AtomicUsize::new(0);
    let done = Mutex::new(Vec::with_capacity(items.len()));
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, items.len().max(1)) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i) else { return };
                let result = work(item);
                done.lock().unwrap().push((i, result));
            });
        }
    });
    let mut done = done.into_inner().unwrap();
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, result)| result).collect()
}

pub fn load_manifests(store: &dyn Store, prefix: &str, threads: usize) -> Result<Vec<Manifest>> {
    let shards: Vec<usize> = (0..SHARDS).collect();
    each(&shards, threads, |&shard| {
        let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest; build or start the current state first"))?;
        Ok(serde_json::from_slice(&raw)?)
    })
}

pub fn load_withdrawn(store: &dyn Store, prefix: &str) -> Result<Withdrawn> {
    Ok(store.get(&withdrawn_key(prefix))?.map(|raw| serde_json::from_slice(&raw)).transpose()?.unwrap_or_default())
}

/// Empty shards that start following the change log after `through_seq`.
pub fn start_empty(store: &dyn Store, prefix: &str, through_seq: i64, threads: usize) -> Result<()> {
    let shards: Vec<usize> = (0..SHARDS).collect();
    each(&shards, threads, |&shard| -> Result<()> {
        if store.get(&manifest_key(prefix, shard))?.is_none() {
            let manifest = Manifest { shard, through_seq, merged_through: through_seq, ..Manifest::default() };
            store.put(&manifest_key(prefix, shard), Bytes::from(serde_json::to_vec(&manifest)?))?;
        }
        Ok(())
    })?;
    Ok(())
}

/// What the change files `seqs` say about one page key, in log order: each version and each removal.
pub fn history(store: &dyn Store, key: &str, seqs: &[i64], threads: usize) -> Result<Vec<String>> {
    let found = each(seqs, threads, |&seq| -> Result<Vec<String>> {
        let Some(file_key) = change_key(store, seq)? else { return Ok(Vec::new()) };
        let file = store.get(&file_key)?.ok_or_else(|| anyhow!("change file {seq} is missing"))?;
        Ok(current::decode_changes(file, seq)?
            .into_iter()
            .filter_map(|row| match row {
                ChangeRow::Page(page) if page.key == key => Some(format!("{seq}: page fetched {} task {}", page.record.fetched_at, page.record.task_id)),
                ChangeRow::Removed(removed) if removed == key => Some(format!("{seq}: removed")),
                _ => None,
            })
            .collect())
    })?;
    Ok(found.into_iter().flatten().collect())
}

/// Of these pages absent from the current state, the ones whose last logged event after `(key, seq)` is a removal.
fn removed_later(store: &dyn Store, prefix: &str, absent: &[(String, i64)], threads: usize) -> Result<HashSet<String>> {
    let Some(from) = absent.iter().map(|(_, seq)| seq + 1).min() else { return Ok(HashSet::new()) };
    let mut through = i64::MAX;
    for (key, _) in absent {
        let shard = shard_of(key).ok_or_else(|| anyhow!("{key} is not a page key"))?;
        let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest"))?;
        through = through.min(serde_json::from_slice::<Manifest>(&raw)?.through_seq);
    }
    let logged: HashMap<&str, i64> = absent.iter().map(|(key, seq)| (key.as_str(), *seq)).collect();
    let seqs: Vec<i64> = (from..=through).collect();
    let events = each(&seqs, threads, |&seq| -> Result<Vec<(String, bool)>> {
        let Some(file_key) = change_key(store, seq)? else { return Ok(Vec::new()) };
        let file = store.get(&file_key)?.ok_or_else(|| anyhow!("change file {seq} is missing"))?;
        let after = |key: &str| logged.get(key).is_some_and(|&logged| seq > logged);
        Ok(current::decode_changes(file, seq)?
            .into_iter()
            .filter_map(|row| match row {
                ChangeRow::Page(page) if after(&page.key) => Some((page.key, false)),
                ChangeRow::Removed(key) if after(&key) => Some((key, true)),
                _ => None,
            })
            .collect())
    })?;
    let last: HashMap<String, bool> = events.into_iter().flatten().collect();
    Ok(last.into_iter().filter(|&(_, removed)| removed).map(|(key, _)| key).collect())
}

/// Samples pages of `shards` logged in change files `seqs` and checks each reads back from the current state as that version or a newer one.
pub fn check(store: &dyn Store, prefix: &str, seqs: &[i64], per_file: usize, shards: &Range<usize>, threads: usize) -> Result<(usize, usize, Vec<String>)> {
    let mut sampled: Vec<(PageRow, i64)> = Vec::new();
    for &seq in seqs {
        let Some(key) = change_key(store, seq)? else { continue };
        let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} ({key}) is missing"))?;
        let pages: Vec<PageRow> = current::decode_changes(file, seq)?
            .into_iter()
            .filter_map(|row| match row {
                ChangeRow::Page(page) if shard_of(&page.key).is_some_and(|shard| shards.contains(&shard)) => Some(*page),
                _ => None,
            })
            .collect();
        let step = (pages.len() / per_file.max(1)).max(1);
        sampled.extend(pages.into_iter().step_by(step).take(per_file).map(|page| (page, seq)));
    }
    let mut by_shard: BTreeMap<usize, HashSet<String>> = BTreeMap::new();
    for (page, _) in &sampled {
        by_shard.entry(shard_of(&page.key).expect("a sampled key has a shard")).or_default().insert(page.key.clone());
    }
    let asked: Vec<(usize, HashSet<String>)> = by_shard.into_iter().collect();
    let found: HashMap<String, PageRow> = each(&asked, threads, |(shard, keys)| winners(store, prefix, *shard, keys.clone()))?.into_iter().flatten().collect();
    let (mut removed, mut wrong, mut absent) = (0, Vec::new(), Vec::new());
    for (page, seq) in &sampled {
        match found.get(&page.key) {
            Some(found) if found.removed && found.change_seq > *seq => removed += 1,
            Some(found) if !found.removed && !page.replaces(found) => {}
            Some(found) => wrong.push(format!(
                "{} in change file {seq}: current is {} {} from change file {}, logged {} {}",
                page.key,
                if found.removed { "a removal" } else { found.record.fetched_at.as_str() },
                found.version,
                found.change_seq,
                page.record.fetched_at,
                page.version
            )),
            None => absent.push((page.key.clone(), *seq)),
        }
    }
    let removed_since = removed_later(store, prefix, &absent, threads)?;
    for (key, seq) in absent {
        if removed_since.contains(&key) {
            removed += 1;
        } else {
            wrong.push(format!("{key} in change file {seq}: not in the current state"));
        }
    }
    Ok((sampled.len(), removed, wrong))
}

/// The newest numbered change file.
pub fn latest_seq(store: &dyn Store) -> Result<i64> {
    let raw = store.get(&CHANGES.latest_key())?.ok_or_else(|| anyhow!("the change log has no {}", CHANGES.latest_key()))?;
    let latest: serde_json::Value = serde_json::from_slice(&raw)?;
    latest["seq"].as_i64().ok_or_else(|| anyhow!("{} has no seq", CHANGES.latest_key()))
}

/// The change file numbered `seq`, None while its number is not written yet.
pub fn change_key(store: &dyn Store, seq: i64) -> Result<Option<String>> {
    let Some(raw) = store.get(&CHANGES.seq_key(seq as u64))? else { return Ok(None) };
    let pointer: serde_json::Value = serde_json::from_slice(&raw)?;
    Ok(pointer["key"].as_str().map(String::from))
}

/// Rows on their way to their shard: in memory up to a budget, then in spill files.
pub(crate) struct Spill {
    dir: PathBuf,
    budget: usize,
    held: usize,
    rows: Vec<Mutex<Vec<PageRow>>>,
    files: Vec<usize>,
}

impl Spill {
    pub(crate) fn new(dir: &Path, budget: usize) -> Result<Spill> {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir)?;
        Ok(Spill { dir: dir.to_path_buf(), budget, held: 0, rows: (0..SHARDS).map(|_| Mutex::default()).collect(), files: vec![0; SHARDS] })
    }

    fn path(&self, shard: usize, n: usize) -> PathBuf {
        self.dir.join(format!("{shard:04}-{n:06}.parquet"))
    }

    pub(crate) fn push(&mut self, shard: usize, row: PageRow) -> Result<()> {
        self.held += row.record.held_bytes();
        self.rows[shard].get_mut().unwrap().push(row);
        if self.held > self.budget {
            for shard in 0..SHARDS {
                let rows = std::mem::take(self.rows[shard].get_mut().unwrap());
                if !rows.is_empty() {
                    std::fs::write(self.path(shard, self.files[shard]), current::encode(&rows)?)?;
                    self.files[shard] += 1;
                }
            }
            self.held = 0;
        }
        Ok(())
    }

    /// The shard's rows in the order they were pushed.
    pub(crate) fn take(&self, shard: usize) -> Result<Vec<PageRow>> {
        let mut rows = Vec::new();
        for n in 0..self.files[shard] {
            let path = self.path(shard, n);
            rows.extend(current::decode(Bytes::from(std::fs::read(&path)?))?);
            std::fs::remove_file(path)?;
        }
        rows.append(&mut self.rows[shard].lock().unwrap());
        Ok(rows)
    }
}

/// Later numbers looked at to tell a number never written from one on its way.
const HOLE_LOOKAHEAD: i64 = 64;

/// Whether number `seq` will never be written: the next number written after it was written `wait_s` ago.
fn never_written(store: &dyn Store, seq: i64, latest: i64, wait_s: f64, now: f64) -> Result<bool> {
    for later in seq + 1..=latest.min(seq + HOLE_LOOKAHEAD) {
        if let Some(at) = store.modified(&CHANGES.seq_key(later as u64))? {
            return Ok(now - at >= wait_s);
        }
    }
    Ok(false)
}

/// Writes this cycle's deltas, combined deltas and merged bases and notes them in `manifests`, which are not saved yet; returns how many of each.
fn write_files(
    store: &dyn Store,
    settings: &Settings,
    manifests: &mut [Manifest],
    spill: &Spill,
    gone: &Withdrawals,
    (from, through, now): (i64, i64, f64),
    written: &Mutex<Vec<String>>,
) -> Result<(usize, usize, usize)> {
    let prefix = settings.prefix.as_str();
    let shards: Vec<usize> = (0..SHARDS).collect();
    let deltas = each(&shards, settings.threads, |&shard| -> Result<Option<FileRef>> {
        let rows = current::merge(spill.take(shard)?, gone, true);
        if rows.is_empty() || through < from {
            return Ok(None);
        }
        let key = file_key(prefix, shard, "delta", through);
        let file = current::encode(&rows)?;
        let bytes = file.len() as u64;
        store.put(&key, Bytes::from(file))?;
        written.lock().unwrap().push(key.clone());
        Ok(Some(FileRef { key, rows: rows.len() as u64, bytes, through_seq: through, level: 0 }))
    })?;
    let wrote_deltas = deltas.iter().flatten().count();
    for (manifest, delta) in manifests.iter_mut().zip(deltas) {
        if let Some(delta) = delta {
            manifest.deltas.push(delta);
        }
        manifest.through_seq = manifest.through_seq.max(through);
    }

    let mut due: Vec<Rewrite> = (0..SHARDS).filter_map(|shard| rewrite_due(&manifests[shard], settings).map(|level| Rewrite { shard, level })).collect();
    due.sort_by_key(|rewrite| (rewrite.level.is_some(), std::cmp::Reverse(manifests[rewrite.shard].delta_rows())));
    due.truncate(settings.merges_per_cycle);
    let made = each(&due, settings.merge_threads, |rewrite| -> Result<FileRef> {
        let manifest = &manifests[rewrite.shard];
        let inputs: Vec<&FileRef> = match rewrite.level {
            None => manifest.files().collect(),
            Some(level) => manifest.deltas.iter().filter(|delta| delta.level == level).collect(),
        };
        let files =
            inputs.iter().map(|file| store.get(&file.key)?.ok_or_else(|| anyhow!("{} is in a manifest but missing", file.key))).collect::<Result<Vec<_>>>()?;
        let (file, rows) = current::merge_files(files, gone, rewrite.level.is_some())?;
        let (key, level) = match rewrite.level {
            None => (file_key(prefix, rewrite.shard, "base", manifest.through_seq), 0),
            Some(level) => (file_key(prefix, rewrite.shard, &format!("delta{}", level + 1), manifest.through_seq), level + 1),
        };
        let bytes = file.len() as u64;
        store.put(&key, Bytes::from(file))?;
        written.lock().unwrap().push(key.clone());
        Ok(FileRef { key, rows, bytes, through_seq: inputs.iter().map(|f| f.through_seq).max().unwrap_or(manifest.through_seq), level })
    })?;
    let (mut merged, mut combined) = (0, 0);
    for (rewrite, made) in due.iter().zip(made) {
        let manifest = &mut manifests[rewrite.shard];
        match rewrite.level {
            None => {
                let replaced: Vec<Retired> = manifest.files().filter(|f| f.key != made.key).map(|f| Retired { key: f.key.clone(), at: now }).collect();
                manifest.retired.extend(replaced);
                manifest.merged_through = manifest.through_seq;
                manifest.base = Some(made);
                manifest.deltas.clear();
                merged += 1;
            }
            Some(level) => {
                let (old, kept): (Vec<FileRef>, Vec<FileRef>) = manifest.deltas.drain(..).partition(|delta| delta.level == level);
                manifest.retired.extend(old.into_iter().filter(|f| f.key != made.key).map(|f| Retired { key: f.key, at: now }));
                manifest.deltas = kept;
                manifest.deltas.push(made);
                combined += 1;
            }
        }
    }
    Ok((wrote_deltas, merged, combined))
}

/// A base rewrite (`level` None) or a combining of one size of deltas.
struct Rewrite {
    shard: usize,
    level: Option<u32>,
}

/// What a shard's files are due: its base rewritten once its deltas reach `merge_ratio` of it, else the smallest size of delta with `fanout` files combined.
fn rewrite_due(manifest: &Manifest, settings: &Settings) -> Option<Option<u32>> {
    if manifest.deltas.is_empty() {
        return None;
    }
    let base_rows = manifest.base.as_ref().map_or(0, |b| b.rows) as f64;
    if manifest.delta_rows() as f64 >= settings.merge_ratio * base_rows {
        return Some(None);
    }
    let mut levels: BTreeMap<u32, usize> = BTreeMap::new();
    for delta in &manifest.deltas {
        *levels.entry(delta.level).or_default() += 1;
    }
    levels.into_iter().find(|&(_, count)| count >= settings.fanout.max(2)).map(|(level, _)| Some(level))
}

/// One pass: the change log read up to `files_per_cycle` files past the shard furthest behind, then merges and cleanup.
pub fn cycle(store: &dyn Store, settings: &Settings, now: f64) -> Result<Report> {
    let prefix = settings.prefix.as_str();
    let mut manifests = load_manifests(store, prefix, settings.threads)?;
    let mut withdrawn = load_withdrawn(store, prefix)?;
    let from = manifests.iter().map(|m| m.through_seq).min().unwrap_or(0) + 1;
    let latest = latest_seq(store)?;
    let mut report = Report { from, through: from - 1, ..Report::default() };
    let mut spill = Spill::new(&settings.work.join("spill"), settings.spill_bytes)?;
    let wanted: Vec<i64> = (from..=latest.min(from + settings.files_per_cycle - 1)).collect();
    'read: for chunk in wanted.chunks(settings.threads.max(1) * 2) {
        let decoded = each(chunk, settings.threads, |&seq| -> Result<Option<Vec<ChangeRow>>> {
            let Some(key) = change_key(store, seq)? else { return Ok(None) };
            let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} ({key}) is numbered but missing"))?;
            Ok(Some(current::decode_changes(file, seq).with_context(|| format!("change file {seq} ({key})"))?))
        })?;
        for (&seq, rows) in chunk.iter().zip(decoded) {
            let rows = match rows {
                Some(rows) => rows,
                None if never_written(store, seq, latest, settings.hole_wait_s, now)? => {
                    report.skipped.push(seq);
                    report.through = seq;
                    continue;
                }
                None => {
                    report.waiting = true;
                    break 'read;
                }
            };
            for row in rows {
                let (shard, row) = match row {
                    ChangeRow::TaskWithdrawn(task_id) => {
                        let taken = withdrawn.tasks.entry(task_id).or_insert(Withdrawal { seq, last: seq });
                        taken.last = taken.last.max(seq);
                        report.withdrawn += 1;
                        continue;
                    }
                    ChangeRow::Page(page) => {
                        if let Some(taken) = withdrawn.tasks.get_mut(&page.record.task_id) {
                            taken.last = taken.last.max(seq);
                        }
                        (shard_of(&page.key), *page)
                    }
                    ChangeRow::Removed(key) => (shard_of(&key), PageRow::removal(key, seq)),
                };
                let Some(shard) = shard else { continue };
                if seq > manifests[shard].through_seq {
                    spill.push(shard, row)?;
                    report.rows += 1;
                }
            }
            report.through = seq;
        }
    }
    let through = report.through;
    // Kept before any manifest moves past the rows that withdrew them.
    store.put(&withdrawn_key(prefix), Bytes::from(serde_json::to_vec(&withdrawn)?))?;
    let gone = withdrawn.by_task();

    let referenced: HashSet<String> = manifests.iter().flat_map(|m| m.files().map(|f| f.key.clone())).collect();
    let written = Mutex::new(Vec::new());
    match write_files(store, settings, &mut manifests, &spill, &gone, (from, through, now), &written) {
        Ok((deltas, merged, combined)) => (report.deltas, report.merged, report.combined) = (deltas, merged, combined),
        Err(error) => {
            for key in written.into_inner().unwrap().iter().filter(|key| !referenced.contains(*key)) {
                let _ = store.delete(key);
            }
            return Err(error);
        }
    }

    let mut expired = Vec::new();
    for manifest in &mut manifests {
        // With no files a shard holds nothing a withdrawal could still need dropped.
        if manifest.base.is_none() && manifest.deltas.is_empty() {
            manifest.merged_through = manifest.through_seq;
        }
        let (old, keep): (Vec<Retired>, Vec<Retired>) = manifest.retired.drain(..).partition(|r| now - r.at >= settings.retire_after_s);
        manifest.retired = keep;
        expired.extend(old.into_iter().map(|r| r.key));
    }
    each(&manifests, settings.threads, |manifest| store.put(&manifest_key(prefix, manifest.shard), Bytes::from(serde_json::to_vec(manifest)?)))?;
    for key in &expired {
        store.delete(key)?;
    }
    report.deleted = expired.len();

    let merged_everywhere = manifests.iter().map(|m| m.merged_through).min().unwrap_or(0);
    let before = withdrawn.tasks.len();
    withdrawn.tasks.retain(|_, taken| taken.last > merged_everywhere);
    if withdrawn.tasks.len() != before {
        store.put(&withdrawn_key(prefix), Bytes::from(serde_json::to_vec(&withdrawn)?))?;
    }
    Ok(report)
}

/// The current version of one page from R2 alone: its shard's base and deltas, newest wins, withdrawn tasks dropped.
pub fn lookup(store: &dyn Store, prefix: &str, key: &str) -> Result<Option<PageRow>> {
    Ok(winner(store, prefix, key)?.filter(|row| !row.removed))
}

/// The newest row kept for one page, a removal included; withdrawn tasks dropped.
pub fn winner(store: &dyn Store, prefix: &str, key: &str) -> Result<Option<PageRow>> {
    let shard = shard_of(key).ok_or_else(|| anyhow!("{key} is not a page key"))?;
    Ok(winners(store, prefix, shard, HashSet::from([key.to_string()]))?.remove(key))
}

/// The newest row kept for each of `keys` in one shard; the withdrawn tasks are read before the manifest, so a task pruned meanwhile cannot show.
fn winners(store: &dyn Store, prefix: &str, shard: usize, keys: HashSet<String>) -> Result<HashMap<String, PageRow>> {
    let gone = load_withdrawn(store, prefix)?.by_task();
    let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest"))?;
    let manifest: Manifest = serde_json::from_slice(&raw)?;
    let keys = Arc::new(keys);
    let mut found = Vec::new();
    for file in manifest.files() {
        let wanted = keys.clone();
        let body = store.get(&file.key)?.ok_or_else(|| anyhow!("{} is missing", file.key))?;
        for row in current::rows(body, Some(Box::new(move |key| wanted.contains(key))))? {
            found.push(row?);
        }
    }
    let mut standing = HashMap::new();
    for row in current::merge(found, &gone, true) {
        if !row.removed || !standing.contains_key(&row.key) {
            standing.insert(row.key.clone(), row);
        }
    }
    Ok(standing)
}

/// Every current page of a shard, as a reader merges it.
pub fn shard_rows(store: &dyn Store, prefix: &str, shard: usize) -> Result<Vec<PageRow>> {
    let gone = load_withdrawn(store, prefix)?.by_task();
    let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest"))?;
    let manifest: Manifest = serde_json::from_slice(&raw)?;
    let mut rows = Vec::new();
    for file in manifest.files() {
        rows.extend(current::decode(store.get(&file.key)?.ok_or_else(|| anyhow!("{} is missing", file.key))?)?);
    }
    Ok(current::merge(rows, &gone, false))
}

/// How the current state differs from the publisher's version index, page by page.
#[derive(Debug, Default)]
pub struct Compared {
    /// Pages expected: in the index, or in the fold of the change log.
    pub index_pages: u64,
    pub current_pages: u64,
    /// In the index only because its task is withdrawn and the index has not drained it yet.
    pub draining: u64,
    pub missing: Vec<String>,
    pub extra: Vec<String>,
    pub other_version: Vec<String>,
}

/// Compares every page of `shards` in the current state under `prefix` with `index`; holds those shards in memory.
pub fn compare(store: &dyn Store, prefix: &str, index: &VersionIndex, shards: &Range<usize>, threads: usize) -> Result<Compared> {
    let draining: HashSet<String> = index.withdrawn()?.into_iter().map(|(task_id, _)| task_id).collect();
    let mut expected = Vec::new();
    index.scan(|key, page| {
        if shard_of(&key).is_some_and(|shard| shards.contains(&shard)) {
            let waiting = draining.contains(&page.task_id);
            expected.push((key, page.version, waiting));
        }
        Ok(())
    })?;
    compared(current_versions(store, prefix, shards, threads)?, expected)
}

/// Compares every page of `shards` in the current state under `prefix` with a plain fold of the change log it covers, without shards or files; reads the whole log, so for test runs.
pub fn compare_log(store: &dyn Store, prefix: &str, shards: &Range<usize>, threads: usize) -> Result<Compared> {
    let through = load_manifests(store, prefix, threads)?.iter().map(|m| m.through_seq).min().unwrap_or(0);
    let seqs: Vec<i64> = (1..=through).collect();
    let files = each(&seqs, threads, |&seq| -> Result<Vec<ChangeRow>> {
        let Some(key) = change_key(store, seq)? else { return Ok(Vec::new()) };
        let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} is missing"))?;
        current::decode_changes(file, seq)
    })?;
    let (mut rows, mut withdrawn) = (Vec::new(), Withdrawals::new());
    let wanted = |key: &str| shard_of(key).is_some_and(|shard| shards.contains(&shard));
    for (&seq, file) in seqs.iter().zip(files) {
        for row in file {
            match row {
                ChangeRow::Page(page) if wanted(&page.key) => rows.push(*page),
                ChangeRow::Removed(key) if wanted(&key) => rows.push(PageRow::removal(key, seq)),
                ChangeRow::Page(_) | ChangeRow::Removed(_) => {}
                ChangeRow::TaskWithdrawn(task_id) => {
                    withdrawn.entry(task_id).or_insert(seq);
                }
            }
        }
    }
    let expected = current::merge(rows, &withdrawn, false).into_iter().map(|row| (row.key, row.version, false)).collect();
    compared(current_versions(store, prefix, shards, threads)?, expected)
}

fn current_versions(store: &dyn Store, prefix: &str, shards: &Range<usize>, threads: usize) -> Result<HashMap<String, String>> {
    let shards: Vec<usize> = shards.clone().collect();
    Ok(each(&shards, threads, |&shard| shard_rows(store, prefix, shard))?
        .into_iter()
        .flatten()
        .filter(|row| !row.removed)
        .map(|row| (row.key, row.version))
        .collect())
}

/// `expected` holds each page's key, version, and whether it is waiting to leave because its task was withdrawn.
fn compared(mut current: HashMap<String, String>, expected: Vec<(String, String, bool)>) -> Result<Compared> {
    let mut compared = Compared { current_pages: current.len() as u64, ..Compared::default() };
    for (key, version, waiting) in expected {
        compared.index_pages += 1;
        match current.remove(&key) {
            Some(found) if found == version => {}
            Some(found) => compared.other_version.push(format!("{key}: expected {version} current {found}")),
            None if waiting => compared.draining += 1,
            None => compared.missing.push(key),
        }
    }
    compared.extra = current.into_keys().collect();
    Ok(compared)
}

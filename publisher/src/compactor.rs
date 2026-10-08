//! Brings the current state up to the change log: each cycle reads the change files the shards have not seen,
//! writes one delta per shard, merges shards whose deltas outgrew their base, then swaps the manifests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use desearch::feeds::CHANGES;
use desearch::r2::{self, Bucket};
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;

use crate::current::{self, shard_dir, shard_of, ChangeRow, FileRef, Manifest, PageRow, Retired, SHARDS};
use crate::records::Record;

/// Objects in a bucket or a folder, as the compactor reads and writes them.
pub trait Store: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Bytes>>;
    fn put(&self, key: &str, body: Bytes) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
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
}

pub struct R2Store {
    pub bucket: Bucket,
    pub handle: Handle,
}

/// Larger files go up in parts.
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
}

pub struct Settings {
    /// Where the shards live, e.g. `current/`.
    pub prefix: String,
    /// Local room for rows waiting to be sorted into their shard.
    pub work: PathBuf,
    pub files_per_cycle: i64,
    /// A shard merges once its deltas hold this share of its base, or `max_deltas` files.
    pub merge_ratio: f64,
    pub max_deltas: usize,
    pub merges_per_cycle: usize,
    /// Replaced files stay this long, so a reader mid-read is not cut off.
    pub retire_after_s: f64,
    /// Rows held in memory before they are written to the shard's spill files.
    pub spill_bytes: usize,
    pub threads: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            prefix: "current/".into(),
            work: PathBuf::from("compactor-work"),
            files_per_cycle: 300,
            merge_ratio: 1.0,
            max_deltas: 48,
            merges_per_cycle: 64,
            retire_after_s: 86_400.0,
            spill_bytes: 1 << 30,
            threads: 8,
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
    pub deleted: usize,
}

/// Tasks taken back and the change file each was withdrawn in; readers drop their rows until every shard has merged past it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Withdrawn {
    pub tasks: BTreeMap<String, i64>,
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

/// Samples page rows of these change files and checks each reads back from the current state as that version or a newer one.
pub fn check(store: &dyn Store, prefix: &str, seqs: &[i64], per_file: usize, threads: usize) -> Result<(usize, usize, Vec<String>)> {
    let (mut checked, mut removed) = (0, 0);
    let (mut wrong, mut absent) = (Vec::new(), Vec::new());
    for &seq in seqs {
        let Some(key) = change_key(store, seq)? else { continue };
        let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} ({key}) is missing"))?;
        let pages: Vec<PageRow> = current::decode_changes(file, seq)?
            .into_iter()
            .filter_map(|row| match row {
                ChangeRow::Page(page) => Some(*page),
                _ => None,
            })
            .collect();
        let step = (pages.len() / per_file.max(1)).max(1);
        for page in pages.iter().step_by(step).take(per_file) {
            checked += 1;
            match winner(store, prefix, &page.key)? {
                Some(found) if found.removed && found.change_seq > seq => removed += 1,
                Some(found) if !found.removed && !page.replaces(&found) => {}
                Some(found) => wrong.push(format!(
                    "{} in change file {seq}: current is {} {} from change file {}, logged {} {}",
                    page.key,
                    if found.removed { "a removal" } else { found.record.fetched_at.as_str() },
                    found.version,
                    found.change_seq,
                    page.record.fetched_at,
                    page.version
                )),
                None => absent.push((page.key.clone(), seq)),
            }
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
    Ok((checked, removed, wrong))
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
        self.held += row.record.text.len() + 512;
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

    pub(crate) fn take(&self, shard: usize) -> Result<Vec<PageRow>> {
        let mut rows = std::mem::take(&mut *self.rows[shard].lock().unwrap());
        for n in 0..self.files[shard] {
            let path = self.path(shard, n);
            rows.extend(current::decode(Bytes::from(std::fs::read(&path)?))?);
            std::fs::remove_file(path)?;
        }
        Ok(rows)
    }
}

fn removal(key: String, change_seq: i64) -> PageRow {
    PageRow { key, removed: true, change_seq, version: String::new(), published_at: String::new(), record: Record::default() }
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
    for chunk in wanted.chunks(settings.threads.max(1) * 2) {
        let decoded = each(chunk, settings.threads, |&seq| -> Result<Option<Vec<ChangeRow>>> {
            let Some(key) = change_key(store, seq)? else { return Ok(None) };
            let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} ({key}) is numbered but missing"))?;
            Ok(Some(current::decode_changes(file, seq).with_context(|| format!("change file {seq} ({key})"))?))
        })?;
        let mut stopped = false;
        for (&seq, rows) in chunk.iter().zip(decoded) {
            // A number not written yet ends the cycle there, so no file is ever skipped.
            let Some(rows) = rows else {
                stopped = true;
                break;
            };
            for row in rows {
                let (shard, row) = match row {
                    ChangeRow::TaskWithdrawn(task_id) => {
                        withdrawn.tasks.entry(task_id).or_insert(seq);
                        report.withdrawn += 1;
                        continue;
                    }
                    ChangeRow::Page(page) => (shard_of(&page.key), *page),
                    ChangeRow::Removed(key) => (shard_of(&key), removal(key, seq)),
                };
                let Some(shard) = shard else { continue };
                if seq > manifests[shard].through_seq {
                    spill.push(shard, row)?;
                    report.rows += 1;
                }
            }
            report.through = seq;
        }
        if stopped {
            break;
        }
    }
    let through = report.through;
    // Kept before any manifest moves past the rows that withdrew them.
    store.put(&withdrawn_key(prefix), Bytes::from(serde_json::to_vec(&withdrawn)?))?;
    let gone: HashSet<String> = withdrawn.tasks.keys().cloned().collect();

    let shards: Vec<usize> = (0..SHARDS).collect();
    let written = each(&shards, settings.threads, |&shard| -> Result<Option<FileRef>> {
        let rows = current::merge(spill.take(shard)?, &gone, true);
        if rows.is_empty() || through < from {
            return Ok(None);
        }
        let key = file_key(prefix, shard, "delta", through);
        let file = current::encode(&rows)?;
        let bytes = file.len() as u64;
        store.put(&key, Bytes::from(file))?;
        Ok(Some(FileRef { key, rows: rows.len() as u64, bytes, through_seq: through }))
    })?;
    for (manifest, delta) in manifests.iter_mut().zip(written) {
        if let Some(delta) = delta {
            manifest.deltas.push(delta);
            report.deltas += 1;
        }
        manifest.through_seq = manifest.through_seq.max(through);
    }

    let mut due: Vec<usize> = (0..SHARDS)
        .filter(|&shard| {
            let manifest = &manifests[shard];
            let base_rows = manifest.base.as_ref().map_or(0, |b| b.rows) as f64;
            !manifest.deltas.is_empty() && (manifest.delta_rows() as f64 >= settings.merge_ratio * base_rows || manifest.deltas.len() >= settings.max_deltas)
        })
        .collect();
    due.sort_by_key(|&shard| std::cmp::Reverse(manifests[shard].delta_rows()));
    due.truncate(settings.merges_per_cycle);
    let merged = each(&due, settings.threads, |&shard| -> Result<FileRef> {
        let manifest = &manifests[shard];
        let mut rows = Vec::new();
        for file in manifest.files() {
            rows.extend(current::decode(store.get(&file.key)?.ok_or_else(|| anyhow!("{} is in a manifest but missing", file.key))?)?);
        }
        let rows = current::merge(rows, &gone, false);
        let key = file_key(prefix, shard, "base", manifest.through_seq);
        let file = current::encode(&rows)?;
        let bytes = file.len() as u64;
        store.put(&key, Bytes::from(file))?;
        Ok(FileRef { key, rows: rows.len() as u64, bytes, through_seq: manifest.through_seq })
    })?;
    for (&shard, base) in due.iter().zip(merged) {
        let manifest = &mut manifests[shard];
        let replaced: Vec<Retired> = manifest.files().filter(|f| f.key != base.key).map(|f| Retired { key: f.key.clone(), at: now }).collect();
        manifest.retired.extend(replaced);
        manifest.merged_through = base.through_seq;
        manifest.base = Some(base);
        manifest.deltas.clear();
        report.merged += 1;
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
    withdrawn.tasks.retain(|_, at| *at > merged_everywhere);
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
    let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest"))?;
    let manifest: Manifest = serde_json::from_slice(&raw)?;
    let gone: HashSet<String> = load_withdrawn(store, prefix)?.tasks.into_keys().collect();
    let mut found = Vec::new();
    for file in manifest.files() {
        let rows = current::decode(store.get(&file.key)?.ok_or_else(|| anyhow!("{} is missing", file.key))?)?;
        found.extend(rows.into_iter().filter(|row| row.key == key));
    }
    Ok(current::merge(found, &gone, true).pop())
}

/// Every current page of a shard, as a reader merges it.
pub fn shard_rows(store: &dyn Store, prefix: &str, shard: usize) -> Result<Vec<PageRow>> {
    let raw = store.get(&manifest_key(prefix, shard))?.ok_or_else(|| anyhow!("shard {shard} has no manifest"))?;
    let manifest: Manifest = serde_json::from_slice(&raw)?;
    let gone: HashSet<String> = load_withdrawn(store, prefix)?.tasks.into_keys().collect();
    let mut rows = Vec::new();
    for file in manifest.files() {
        rows.extend(current::decode(store.get(&file.key)?.ok_or_else(|| anyhow!("{} is missing", file.key))?)?);
    }
    Ok(current::merge(rows, &gone, false))
}

/// How many rows each shard's files hold, for checking a build against what it was built from.
pub fn shard_counts(manifests: &[Manifest]) -> HashMap<usize, u64> {
    manifests.iter().map(|m| (m.shard, m.files().map(|f| f.rows).sum())).collect()
}

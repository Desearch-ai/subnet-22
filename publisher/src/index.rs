//! The publisher's own record of every page's latest version and of the tasks taken back, in RocksDB.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DBCompressionType, Direction, IteratorMode, Options, WriteBatch, DB,
};

use crate::records::parse_iso;
use crate::records::{iso, SECOND};

pub const WITHDRAWN_KEEP_S: f64 = 7.0 * 86_400.0;
/// The index outgrows memory, so lookups are random disk reads; several readers at once keep the disk busy.
pub const READERS: usize = 8;
const READ_CHUNK: usize = 500;
const PAGES: &str = "pages";
/// Task id, then page key: the pages a task's version is current for.
const TASKS: &str = "tasks";
const WITHDRAWN: &str = "withdrawn";
const HEX_VERSION: u8 = 1;
const HEX_SHA1: u8 = 2;
const PACKED_TIME: u8 = 4;
const HAS_SEQ: u8 = 8;
const HAS_ROW: u8 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Current {
    pub url: String,
    pub version: String,
    pub fetched_at: String,
    pub task_id: String,
    pub content_sha1: String,
    pub change_seq: Option<i64>,
    pub change_row: Option<i64>,
}

/// A page whose current version came from a withdrawn task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Withdrawn {
    pub key: String,
    pub url: String,
    pub version: String,
    pub content_sha1: String,
}

pub struct VersionIndex {
    db: DB,
    readers: usize,
    /// Writes read before they write, so only one runs at a time.
    writing: Mutex<()>,
}

impl VersionIndex {
    pub fn open(path: &Path, cache_bytes: usize) -> Result<Self> {
        let cache = Cache::new_lru_cache(cache_bytes);
        let db = DB::open_cf_descriptors(&options(), path, families(&cache)).with_context(|| format!("opening the version index at {}", path.display()))?;
        Ok(VersionIndex { db, readers: READERS, writing: Mutex::new(()) })
    }

    /// An index another process may be writing, or a checkpoint of one, for reading only.
    pub fn open_read_only(path: &Path, cache_bytes: usize) -> Result<Self> {
        let cache = Cache::new_lru_cache(cache_bytes);
        let db = DB::open_cf_descriptors_read_only(&options(), path, families(&cache), false)
            .with_context(|| format!("opening the version index at {} to read", path.display()))?;
        Ok(VersionIndex { db, readers: READERS, writing: Mutex::new(()) })
    }

    /// A consistent copy of the index in `dir`, made of hard links to the same files, so it costs no copying.
    pub fn checkpoint(&self, dir: &Path) -> Result<()> {
        rocksdb::checkpoint::Checkpoint::new(&self.db)?.create_checkpoint(dir).with_context(|| format!("checkpointing the index into {}", dir.display()))
    }

    pub fn with_readers(mut self, readers: usize) -> Self {
        self.readers = readers.max(1);
        self
    }

    fn family(&self, name: &str) -> &ColumnFamily {
        self.db.cf_handle(name).expect("column family opened with the index")
    }

    pub fn current(&self, key: &str) -> Result<Option<Current>> {
        self.db.get_pinned_cf(self.family(PAGES), key)?.map(|value| decode(&value)).transpose()
    }

    /// The current version of each page the index holds, looked up in sorted chunks on parallel readers.
    pub fn current_many(&self, keys: &[&str]) -> Result<HashMap<String, Current>> {
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        let chunks: Vec<&[&str]> = keys.chunks(READ_CHUNK).collect();
        let readers = self.readers.min(chunks.len()).max(1);
        if readers == 1 {
            let mut found = HashMap::with_capacity(keys.len());
            for chunk in chunks {
                self.read_chunk(chunk, &mut found)?;
            }
            return Ok(found);
        }
        let next = AtomicUsize::new(0);
        let parts: Vec<Result<HashMap<String, Current>>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..readers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut found = HashMap::new();
                        while let Some(chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                            self.read_chunk(chunk, &mut found)?;
                        }
                        Ok(found)
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().expect("index reader panicked")).collect()
        });
        let mut found = HashMap::with_capacity(keys.len());
        for part in parts {
            found.extend(part?);
        }
        Ok(found)
    }

    fn read_chunk(&self, keys: &[&str], found: &mut HashMap<String, Current>) -> Result<()> {
        let values = self.db.batched_multi_get_cf(self.family(PAGES), keys.iter().map(|k| k.as_bytes()), true);
        for (key, value) in keys.iter().zip(values) {
            if let Some(value) = value? {
                found.insert(key.to_string(), decode(&value)?);
            }
        }
        Ok(())
    }

    fn currents(&self, keys: &[&str]) -> Result<Vec<Option<Current>>> {
        let values = self.db.batched_multi_get_cf(self.family(PAGES), keys.iter().map(|k| k.as_bytes()), true);
        values.into_iter().map(|value| value?.map(|v| decode(&v)).transpose()).collect()
    }

    /// Called only after the change file holding these pages is written.
    pub fn store(&self, pages: &[(String, Current)]) -> Result<()> {
        let _writing = self.writing.lock().unwrap();
        let mut sorted: Vec<&(String, Current)> = pages.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let keys: Vec<&str> = sorted.iter().map(|(key, _)| key.as_str()).collect();
        let before = self.currents(&keys)?;
        let mut task_of: HashMap<&str, String> = HashMap::new();
        for (key, old) in keys.iter().zip(before) {
            if let Some(old) = old {
                task_of.insert(key, old.task_id);
            }
        }
        let mut batch = WriteBatch::default();
        for (key, current) in sorted {
            if let Some(old_task) = task_of.get(key.as_str()) {
                if *old_task != current.task_id {
                    batch.delete_cf(self.family(TASKS), task_key(old_task, key)?);
                }
            }
            batch.put_cf(self.family(PAGES), key, encode(current));
            batch.put_cf(self.family(TASKS), task_key(&current.task_id, key)?, b"");
            task_of.insert(key, current.task_id.clone());
        }
        Ok(self.db.write(batch)?)
    }

    /// A later fetch that found the same content moves the fetch time forward.
    pub fn touch(&self, seen: &[(String, String)]) -> Result<()> {
        let _writing = self.writing.lock().unwrap();
        let mut sorted: Vec<&(String, String)> = seen.iter().collect();
        sorted.sort();
        let keys: Vec<&str> = sorted.iter().map(|(key, _)| key.as_str()).collect();
        let mut latest: HashMap<&str, Current> = HashMap::new();
        for (key, current) in keys.iter().zip(self.currents(&keys)?) {
            if let Some(current) = current {
                latest.insert(key, current);
            }
        }
        let mut batch = WriteBatch::default();
        for (key, fetched_at) in sorted {
            if let Some(current) = latest.get_mut(key.as_str()) {
                if *fetched_at > current.fetched_at {
                    current.fetched_at = fetched_at.clone();
                    batch.put_cf(self.family(PAGES), key, encode(current));
                }
            }
        }
        Ok(self.db.write(batch)?)
    }

    /// Takes up to `limit` of a withdrawn task's pages listed after `after` out of the index, handing the ones still current to `removed` first; returns the last key taken while more may follow.
    pub fn drain(&self, task_id: &str, after: Option<&str>, limit: usize, removed: impl FnOnce(&[Withdrawn]) -> Result<()>) -> Result<Option<String>> {
        let limit = limit.max(1);
        let prefix = task_key(task_id, "")?;
        let start = task_key(task_id, after.unwrap_or_default())?;
        let mut keys = Vec::with_capacity(limit.min(READ_CHUNK));
        for item in self.db.iterator_cf(self.family(TASKS), IteratorMode::From(&start, Direction::Forward)) {
            let (entry, _) = item?;
            let Some(key) = entry.strip_prefix(prefix.as_slice()) else {
                break;
            };
            if keys.len() == limit {
                break;
            }
            keys.push(String::from_utf8(key.to_vec()).context("a page key that is not UTF-8")?);
        }
        if keys.is_empty() {
            return Ok(None);
        }
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let mut pages = Vec::new();
        for (key, current) in keys.iter().zip(self.currents(&keys)?) {
            if let Some(current) = current.filter(|c| c.task_id == task_id) {
                pages.push(Withdrawn { key: key.to_string(), url: current.url, version: current.version, content_sha1: current.content_sha1 });
            }
        }
        removed(&pages)?;
        let _writing = self.writing.lock().unwrap();
        let mut batch = WriteBatch::default();
        for (key, current) in keys.iter().zip(self.currents(&keys)?) {
            batch.delete_cf(self.family(TASKS), task_key(task_id, key)?);
            if current.is_some_and(|c| c.task_id == task_id) {
                batch.delete_cf(self.family(PAGES), key);
            }
        }
        self.db.write(batch)?;
        Ok((keys.len() == limit).then(|| keys[limit - 1].to_string()))
    }

    /// Remembered, so a withdrawn task's pages are never published even if its job comes later.
    pub fn withdraw(&self, task_ids: &[String], now: f64) -> Result<()> {
        let _writing = self.writing.lock().unwrap();
        let family = self.family(WITHDRAWN);
        let mut batch = WriteBatch::default();
        let mut added = HashSet::new();
        for task_id in task_ids {
            if self.db.get_pinned_cf(family, task_id)?.is_none() && added.insert(task_id) {
                batch.put_cf(family, task_id, now.to_le_bytes());
            }
        }
        Ok(self.db.write(batch)?)
    }

    pub fn is_withdrawn(&self, task_id: &str) -> Result<bool> {
        Ok(self.db.get_pinned_cf(self.family(WITHDRAWN), task_id)?.is_some())
    }

    /// Which of these tasks are withdrawn.
    pub fn withdrawn_among(&self, task_ids: &[&str]) -> Result<HashSet<String>> {
        let mut task_ids = task_ids.to_vec();
        task_ids.sort_unstable();
        task_ids.dedup();
        let values = self.db.batched_multi_get_cf(self.family(WITHDRAWN), task_ids.iter().map(|t| t.as_bytes()), true);
        let mut found = HashSet::new();
        for (task_id, value) in task_ids.iter().zip(values) {
            if value?.is_some() {
                found.insert(task_id.to_string());
            }
        }
        Ok(found)
    }

    /// Stops refusing these tasks' jobs; only for tasks drained and withdrawn long ago.
    pub fn forget_withdrawn(&self, task_ids: &[String]) -> Result<()> {
        let mut batch = WriteBatch::default();
        for task_id in task_ids {
            batch.delete_cf(self.family(WITHDRAWN), task_id);
        }
        Ok(self.db.write(batch)?)
    }

    /// A withdrawal removes a page only while the withdrawn version is still current.
    pub fn remove(&self, removed: &[(String, String)]) -> Result<()> {
        let _writing = self.writing.lock().unwrap();
        let mut wanted: Vec<(&str, &str)> = removed.iter().map(|(key, version)| (key.as_str(), version.as_str())).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let mut keys: Vec<&str> = wanted.iter().map(|(key, _)| *key).collect();
        keys.dedup();
        let mut batch = WriteBatch::default();
        for (key, current) in keys.iter().zip(self.currents(&keys)?) {
            let Some(current) = current else { continue };
            if wanted.binary_search(&(key, current.version.as_str())).is_ok() {
                batch.delete_cf(self.family(PAGES), key);
                batch.delete_cf(self.family(TASKS), task_key(&current.task_id, key)?);
            }
        }
        Ok(self.db.write(batch)?)
    }

    /// Every page the index holds, in key order.
    pub fn pages(&self) -> Result<Vec<(String, Current)>> {
        let mut pages = Vec::new();
        self.scan(|key, current| {
            pages.push((key, current));
            Ok(())
        })?;
        Ok(pages)
    }

    /// Every page in key order, one at a time, so the whole index never sits in memory.
    pub fn scan(&self, mut each: impl FnMut(String, Current) -> Result<()>) -> Result<()> {
        let mut read = rocksdb::ReadOptions::default();
        read.fill_cache(false);
        read.set_readahead_size(4 << 20);
        for item in self.db.iterator_cf_opt(self.family(PAGES), read, IteratorMode::Start) {
            let (key, value) = item?;
            each(String::from_utf8(key.to_vec()).context("a page key that is not UTF-8")?, decode(&value)?)?;
        }
        Ok(())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.db.iterator_cf(self.family(PAGES), IteratorMode::Start).next().transpose()?.is_none())
    }

    /// Withdrawn tasks and when they were withdrawn, in Unix seconds.
    pub fn withdrawn(&self) -> Result<Vec<(String, f64)>> {
        let mut found = Vec::new();
        for item in self.db.iterator_cf(self.family(WITHDRAWN), IteratorMode::Start) {
            let (task_id, at) = item?;
            let at = f64::from_le_bytes(at.as_ref().try_into().context("a withdrawal time that is not 8 bytes")?);
            found.push((String::from_utf8(task_id.to_vec())?, at));
        }
        Ok(found)
    }

    pub fn compact(&self) {
        for name in [PAGES, TASKS, WITHDRAWN] {
            self.db.compact_range_cf(self.family(name), None::<&[u8]>, None::<&[u8]>);
        }
    }

    /// Bytes of table files each column family holds on disk.
    pub fn disk_bytes(&self) -> Result<Vec<(&'static str, u64)>> {
        [PAGES, TASKS, WITHDRAWN]
            .into_iter()
            .map(|name| Ok((name, self.db.property_int_value_cf(self.family(name), "rocksdb.total-sst-files-size")?.unwrap_or(0))))
            .collect()
    }

    pub fn flush(&self) -> Result<()> {
        for name in [PAGES, TASKS, WITHDRAWN] {
            self.db.flush_cf(self.family(name))?;
        }
        Ok(())
    }
}

fn options() -> Options {
    let mut options = Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get()) as i32;
    options.set_max_background_jobs(cores.clamp(2, 8));
    options.set_max_subcompactions(2);
    options.set_max_open_files(4096);
    options
}

fn families(cache: &Cache) -> [ColumnFamilyDescriptor; 3] {
    // The task list is only ever scanned by prefix, so a whole-key filter would be memory spent for nothing.
    [(PAGES, true), (TASKS, false), (WITHDRAWN, true)].map(|(name, filtered)| ColumnFamilyDescriptor::new(name, family(cache, filtered)))
}

fn family(cache: &Cache, filtered: bool) -> Options {
    let mut table = BlockBasedOptions::default();
    table.set_block_cache(cache);
    if filtered {
        table.set_bloom_filter(10.0, false);
    }
    table.set_block_size(16 << 10);
    // Filters and indexes grow with every key; kept in small partitions inside the cache, they stay within its budget.
    table.set_cache_index_and_filter_blocks(true);
    table.set_pin_l0_filter_and_index_blocks_in_cache(true);
    table.set_partition_filters(true);
    table.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
    table.set_pin_top_level_index_and_filter(true);
    table.set_metadata_block_size(4096);
    let mut options = Options::default();
    options.set_block_based_table_factory(&table);
    options.set_compression_type(DBCompressionType::Zstd);
    options.set_bottommost_compression_type(DBCompressionType::Zstd);
    options.set_level_compaction_dynamic_level_bytes(true);
    options.set_write_buffer_size(64 << 20);
    options
}

fn task_key(task_id: &str, key: &str) -> Result<Vec<u8>> {
    let Ok(length) = u16::try_from(task_id.len()) else {
        bail!("a task id longer than 65535 bytes");
    };
    let mut out = Vec::with_capacity(2 + task_id.len() + key.len());
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(task_id.as_bytes());
    out.extend_from_slice(key.as_bytes());
    Ok(out)
}

/// Flags, then version, content hash, fetch time, change number and row, task id; the URL takes the rest.
fn encode(current: &Current) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + current.url.len() + current.task_id.len());
    out.push(0);
    let mut flags = 0;
    if put_sha1(&mut out, &current.version) {
        flags |= HEX_VERSION;
    }
    if put_sha1(&mut out, &current.content_sha1) {
        flags |= HEX_SHA1;
    }
    match parse_iso(&current.fetched_at) {
        Some(seconds) => {
            put_varint(&mut out, zigzag(seconds));
            flags |= PACKED_TIME;
        }
        None => put_text(&mut out, &current.fetched_at),
    }
    if let Some(seq) = current.change_seq {
        put_varint(&mut out, zigzag(seq));
        flags |= HAS_SEQ;
    }
    if let Some(row) = current.change_row {
        put_varint(&mut out, zigzag(row));
        flags |= HAS_ROW;
    }
    put_text(&mut out, &current.task_id);
    out.extend_from_slice(current.url.as_bytes());
    out[0] = flags;
    out
}

fn decode(value: &[u8]) -> Result<Current> {
    let (&flags, mut rest) = value.split_first().context("an empty index value")?;
    let version = take_sha1(&mut rest, flags & HEX_VERSION != 0)?;
    let content_sha1 = take_sha1(&mut rest, flags & HEX_SHA1 != 0)?;
    let fetched_at = if flags & PACKED_TIME != 0 { iso(unzigzag(take_varint(&mut rest)?) * SECOND) } else { take_text(&mut rest)? };
    let change_seq = (flags & HAS_SEQ != 0).then(|| take_varint(&mut rest).map(unzigzag)).transpose()?;
    let change_row = (flags & HAS_ROW != 0).then(|| take_varint(&mut rest).map(unzigzag)).transpose()?;
    let task_id = take_text(&mut rest)?;
    let url = String::from_utf8(rest.to_vec()).context("an index URL that is not UTF-8")?;
    Ok(Current { url, version, fetched_at, task_id, content_sha1, change_seq, change_row })
}

fn put_sha1(out: &mut Vec<u8>, text: &str) -> bool {
    let b = text.as_bytes();
    if b.len() == 40 && b.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        out.extend(b.chunks(2).map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1])));
        true
    } else {
        put_text(out, text);
        false
    }
}

fn take_sha1(rest: &mut &[u8], packed: bool) -> Result<String> {
    if !packed {
        return take_text(rest);
    }
    if rest.len() < 20 {
        bail!("a truncated index value");
    }
    let (digest, tail) = rest.split_at(20);
    *rest = tail;
    Ok(desearch::canonical::hex(digest))
}

fn nibble(c: u8) -> u8 {
    if c.is_ascii_digit() {
        c - b'0'
    } else {
        c - b'a' + 10
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    put_varint(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

fn take_text(rest: &mut &[u8]) -> Result<String> {
    let length = take_varint(rest)? as usize;
    if rest.len() < length {
        bail!("a truncated index value");
    }
    let (text, tail) = rest.split_at(length);
    *rest = tail;
    String::from_utf8(text.to_vec()).context("index text that is not UTF-8")
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_varint(rest: &mut &[u8]) -> Result<u64> {
    let mut value = 0u64;
    for (i, &byte) in rest.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte < 0x80 {
            *rest = &rest[i + 1..];
            return Ok(value);
        }
    }
    bail!("a truncated index value")
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let packed = Current {
            url: "https://example.com/a".into(),
            version: "0123456789abcdef0123456789abcdef01234567".into(),
            fetched_at: "2026-10-06T22:10:00+00:00".into(),
            task_id: "7be798edabde4b06".into(),
            content_sha1: "fedcba9876543210fedcba9876543210fedcba98".into(),
            change_seq: Some(12),
            change_row: Some(0),
        };
        let loose =
            Current { version: "V".into(), content_sha1: String::new(), fetched_at: "yesterday".into(), change_seq: None, change_row: None, ..packed.clone() };
        for current in [packed, loose] {
            assert_eq!(decode(&encode(&current)).unwrap(), current);
        }
    }
}

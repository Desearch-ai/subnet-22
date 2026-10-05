//! Domains split into fixed buckets, each bucket in its own RocksDB store.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result};
use blake2::digest::consts::{U2, U8};
use blake2::{Blake2b, Digest};
use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, Cache, ColumnFamilyDescriptor, DBCompressionType, Direction, IteratorMode, Options, ReadOptions,
    WriteBatch, WriteBufferManager, DB,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::listed::{self, ListedSet};
use crate::urls::{Listing, Record, Url, TIMED};

pub const BUCKETS: usize = 256;
const DOMAIN: u8 = b'D';
/// What each sitemap listed when last read, so unchanged entries skip their lookups.
const LISTED: u8 = b'L';
const SITEMAP: u8 = b'S';
pub(crate) const URL: u8 = b'U';
const META: u8 = b'M';

pub type Json = Map<String, Value>;

/// The bucket a domain belongs to; it never changes.
pub fn bucket_of(host: &str) -> usize {
    Blake2b::<U2>::digest(host.as_bytes())[1] as usize
}

/// A stable id for a sitemap file, the same wherever it is computed.
pub fn sitemap_id(url: &str) -> i64 {
    (u64::from_be_bytes(Blake2b::<U8>::digest(url.as_bytes()).into()) >> 1) as i64
}

/// The block cache and memtable budget every store in the process shares.
pub struct Resources {
    pub(crate) cache: Cache,
    memtables: WriteBufferManager,
    background_jobs: i32,
    open_files: i32,
}

impl Resources {
    pub fn new(cache_bytes: usize, memtable_bytes: usize, stores: usize) -> Self {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get()) as i32;
        Resources {
            cache: Cache::new_lru_cache(cache_bytes),
            // Writers wait rather than let memtables outgrow the budget while flushes catch up.
            memtables: WriteBufferManager::new_write_buffer_manager(memtable_bytes, true),
            // One process flushes and compacts every store it owns, so it needs the threads seven processes had.
            background_jobs: (cores * 2).max(4),
            // Every store keeps its own table files open; together they must stay well under the fd limit.
            open_files: (40_000 / stores.max(1)).clamp(64, 512) as i32,
        }
    }

    fn options(&self) -> Options {
        let mut table = BlockBasedOptions::default();
        table.set_block_cache(&self.cache);
        table.set_bloom_filter(10.0, false);
        // Bigger blocks let zstd find more of what neighbouring URLs of one domain share.
        table.set_block_size(16 << 10);
        // Filters and indexes grow with every key; kept in small partitions inside the cache, they stay within its budget.
        table.set_cache_index_and_filter_blocks(true);
        table.set_pin_l0_filter_and_index_blocks_in_cache(true);
        table.set_partition_filters(true);
        table.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
        table.set_pin_top_level_index_and_filter(true);
        table.set_metadata_block_size(4096);
        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_block_based_table_factory(&table);
        options.set_compression_type(DBCompressionType::Zstd);
        options.set_write_buffer_size(16 << 20);
        options.set_max_write_buffer_number(3);
        options.set_write_buffer_manager(&self.memtables);
        options.set_level_compaction_dynamic_level_bytes(true);
        // Hundreds of stores share one memtable budget and flush small files; merging more of them at once rewrites less.
        options.set_level_zero_file_num_compaction_trigger(8);
        options.set_max_background_jobs(self.background_jobs);
        options.set_max_subcompactions(2);
        options.set_max_open_files(self.open_files);
        options
    }
}

/// Domain and sitemap records bound for one store, written together.
#[derive(Default)]
pub struct Changes {
    batch: WriteBatch,
}

impl Changes {
    pub fn domain(&mut self, host: &str, record: &Json) {
        self.batch.put(domain_key(host), encode(record));
    }

    pub fn sitemap(&mut self, host: &str, url: &str, record: &Json) {
        self.batch.put(sitemap_key(host, url), encode(record));
    }
}

/// What the timetable needs from a domain record.
#[derive(Deserialize)]
pub struct Due {
    pub state: String,
    pub due: Option<i64>,
    pub rank: Option<i64>,
}

/// One bucket's domains, their sitemaps and every URL they list.
pub struct BucketStore {
    pub path: PathBuf,
    pub(crate) db: DB,
    /// Held while URL records and ready lists change together, so neither a stamp nor a count is lost.
    ready_lock: Mutex<()>,
    /// Domains whose ready lists grew since the dispatcher last looked, and by how much; None until a dispatcher looks.
    pub(crate) noticed: Mutex<Option<HashMap<String, u64>>>,
}

impl BucketStore {
    pub fn open(path: &Path, resources: &Resources) -> Result<Self> {
        let options = resources.options();
        // Compression, filters, cache and memtable sizes are column family options; the default family must get them too.
        let family = ColumnFamilyDescriptor::new("default", options.clone());
        let db = DB::open_cf_descriptors(&options, path, [family]).with_context(|| format!("opening {}", path.display()))?;
        Ok(BucketStore { path: path.to_path_buf(), db, ready_lock: Mutex::new(()), noticed: Mutex::new(None) })
    }

    pub fn write(&self, changes: Changes) -> Result<()> {
        Ok(self.db.write(changes.batch)?)
    }

    pub fn meta(&self, name: &str) -> Result<Option<Value>> {
        self.json(&meta_key(name))
    }

    pub fn set_meta(&self, name: &str, value: &Value) -> Result<()> {
        Ok(self.db.put(meta_key(name), serde_json::to_vec(value)?)?)
    }

    pub(crate) fn lock_ready(&self) -> MutexGuard<'_, ()> {
        self.ready_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn domain(&self, host: &str) -> Result<Option<Json>> {
        self.json(&domain_key(host))
    }

    /// Every domain's host and schedule, in key order.
    pub fn schedule(&self) -> Result<Vec<(String, Due)>> {
        self.scan(&[DOMAIN], |key, raw| Ok((String::from_utf8_lossy(&key[1..]).into_owned(), serde_json::from_slice(raw)?)))
    }

    pub fn sitemap(&self, host: &str, url: &str) -> Result<Option<Json>> {
        self.json(&sitemap_key(host, url))
    }

    /// How many sitemap records a domain has, counted without decoding them.
    pub fn sitemap_count(&self, host: &str) -> Result<usize> {
        let prefix = sitemap_key(host, "");
        let mut upper = prefix.clone();
        *upper.last_mut().unwrap() += 1;
        let mut bounds = ReadOptions::default();
        bounds.set_iterate_upper_bound(upper);
        let mut records = self.db.raw_iterator_opt(bounds);
        records.seek(&prefix);
        let mut count = 0;
        while records.valid() {
            count += 1;
            records.next();
        }
        records.status()?;
        Ok(count)
    }

    pub fn sitemaps(&self, host: &str) -> Result<Vec<(String, Json)>> {
        let prefix = sitemap_key(host, "");
        self.scan(&prefix, |key, raw| {
            Ok((String::from_utf8_lossy(&key[prefix.len()..]).into_owned(), serde_json::from_slice(raw)?))
        })
    }

    /// Store what one sitemap lists now, looking up only entries it did not list the same way when last read, and put pages due a send on their ready lists.
    pub fn record_listing(&self, host: &str, sitemap_id: i64, mut entries: Vec<(Url, u32, bool)>, now: u32) -> Result<Listing> {
        for entry in entries.iter_mut() {
            entry.0.key.insert(0, URL);
        }
        entries.sort_by(|a, b| a.0.key.cmp(&b.0.key));
        entries.dedup_by(|later, first| later.0.key == first.0.key);
        if entries.is_empty() {
            return Ok(Listing::default());
        }
        let listed_key = listed_key(host, sitemap_id);
        let stored = self.db.get(&listed_key)?;
        let before = stored.as_deref().and_then(ListedSet::decode);
        let prints: Vec<u64> = entries.iter().map(|(url, lastmod, _)| listed::fingerprint(&url.key, *lastmod)).collect();
        let changed: Vec<&(Url, u32, bool)> =
            entries.iter().zip(&prints).filter(|(_, print)| !before.as_ref().is_some_and(|set| set.contains(**print))).map(|(entry, _)| entry).collect();
        let mut batch = WriteBatch::default();
        let mut listing = Listing { listed: entries.len(), ..Listing::default() };
        let after = ListedSet::encode(&prints);
        if stored.as_deref() != Some(&after[..]) {
            batch.put(&listed_key, after);
        }
        if changed.is_empty() {
            if !batch.is_empty() {
                self.db.write(batch)?;
            }
            return Ok(listing);
        }
        let guard = self.lock_ready();
        let cf = self.db.cf_handle("default").context("default column family")?;
        let found = self.db.batched_multi_get_cf(cf, changed.iter().map(|e| &e.0.key), true);
        let (mut fresh, mut moved) = (Vec::new(), Vec::new());
        for ((url, lastmod, timed), raw) in changed.into_iter().zip(found) {
            let timed_flag = if *timed { TIMED } else { 0 };
            let record = match raw?.as_deref().and_then(Record::unpack) {
                None => {
                    listing.new += 1;
                    let record = Record { sitemap_id, lastmod: *lastmod, first_seen: now, last_seen: now, flags: url.flags | timed_flag, ..Record::default() };
                    fresh.push((url.key.clone(), record));
                    record
                }
                Some(mut record) => {
                    if *lastmod != 0 && *lastmod != record.lastmod {
                        record.lastmod = *lastmod;
                        record.flags = (record.flags & !TIMED) | timed_flag;
                        listing.moved += 1;
                        if record.pushed_at != 0 {
                            moved.push((url.key.clone(), *lastmod));
                        }
                    }
                    record.sitemap_id = sitemap_id;
                    record.last_seen = now;
                    record
                }
            };
            batch.put(&url.key, record.pack());
        }
        let (ready, counts) = self.ready_from_listing(&mut batch, &fresh, &moved, listing.listed)?;
        listing.ready = ready;
        self.db.write(batch)?;
        drop(guard);
        self.notice(&counts);
        Ok(listing)
    }

    pub fn url(&self, url: &Url) -> Result<Option<Record>> {
        Ok(self.db.get([&[URL], &url.key[..]].concat())?.as_deref().and_then(Record::unpack))
    }

    /// Memory RocksDB holds for this store outside the block cache: table readers and memtables.
    pub fn memory(&self) -> (u64, u64) {
        let read = |name: &str| self.db.property_int_value(name).ok().flatten().unwrap_or(0);
        (read("rocksdb.estimate-table-readers-mem"), read("rocksdb.cur-size-all-mem-tables"))
    }

    /// Rewrite every file of the store, reclaiming space held by old versions and older, looser blocks.
    pub fn compact(&self) {
        let mut options = rocksdb::CompactOptions::default();
        options.set_bottommost_level_compaction(rocksdb::BottommostLevelCompaction::Force);
        self.db.compact_range_opt(None::<&[u8]>, None::<&[u8]>, &options);
    }

    pub fn flush(&self) -> Result<()> {
        Ok(self.db.flush()?)
    }

    pub fn estimate(&self) -> u64 {
        self.db.property_int_value("rocksdb.estimate-num-keys").ok().flatten().unwrap_or(0)
    }

    fn json<T: serde::de::DeserializeOwned>(&self, key: &[u8]) -> Result<Option<T>> {
        match self.db.get_pinned(key)? {
            Some(raw) => Ok(Some(serde_json::from_slice(&raw)?)),
            None => Ok(None),
        }
    }

    fn scan<T>(&self, prefix: &[u8], mut each: impl FnMut(&[u8], &[u8]) -> Result<T>) -> Result<Vec<T>> {
        let mut bounds = ReadOptions::default();
        let mut upper = prefix.to_vec();
        *upper.last_mut().unwrap() += 1;
        bounds.set_iterate_upper_bound(upper);
        let mut out = Vec::new();
        for item in self.db.iterator_opt(IteratorMode::From(prefix, Direction::Forward), bounds) {
            let (key, raw) = item?;
            if !key.starts_with(prefix) {
                break;
            }
            out.push(each(&key, &raw)?);
        }
        Ok(out)
    }
}

/// The stores of the buckets one process owns.
pub struct Buckets {
    root: PathBuf,
    cache: Cache,
    stores: Vec<Option<BucketStore>>,
}

/// Where RocksDB's memory goes, in bytes.
pub struct Memory {
    pub table_readers: u64,
    pub memtables: u64,
    pub block_cache: u64,
}

impl Buckets {
    pub fn open(root: &Path, owned: &[usize], resources: &Resources) -> Result<Self> {
        let mut stores: Vec<Option<BucketStore>> = (0..BUCKETS).map(|_| None).collect();
        let opened: Vec<Result<(usize, BucketStore)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = owned
                .iter()
                .map(|&bucket| scope.spawn(move || Ok((bucket, BucketStore::open(&root.join(format!("{bucket:03}")), resources)?))))
                .collect();
            handles.into_iter().map(|h| h.join().expect("opening a store panicked")).collect()
        });
        for store in opened {
            let (bucket, store) = store?;
            stores[bucket] = Some(store);
        }
        Ok(Buckets { root: root.to_path_buf(), cache: resources.cache.clone(), stores })
    }

    pub fn memory(&self) -> Memory {
        let (table_readers, memtables) = self.stores().map(BucketStore::memory).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        Memory { table_readers, memtables, block_cache: self.cache.get_usage() as u64 }
    }

    /// Free space on the disk the stores live on.
    pub fn free_disk(&self) -> Option<u64> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(self.root.as_os_str().as_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
            return None;
        }
        Some(stat.f_bavail as u64 * stat.f_frsize as u64)
    }

    pub fn owns(&self, host: &str) -> bool {
        self.stores[bucket_of(host)].is_some()
    }

    pub fn store(&self, host: &str) -> &BucketStore {
        self.stores[bucket_of(host)].as_ref().expect("a host outside this process's buckets")
    }

    /// The store of a host named by data from outside, if this process owns it.
    pub fn store_for(&self, host: &str) -> Option<&BucketStore> {
        self.stores[bucket_of(host)].as_ref()
    }

    pub fn stores(&self) -> impl Iterator<Item = &BucketStore> {
        self.stores.iter().flatten()
    }

    pub fn owned(&self) -> Vec<usize> {
        (0..BUCKETS).filter(|&b| self.stores[b].is_some()).collect()
    }
}

pub(crate) fn meta_key(name: &str) -> Vec<u8> {
    [&[META], name.as_bytes()].concat()
}

fn domain_key(host: &str) -> Vec<u8> {
    [&[DOMAIN], host.as_bytes()].concat()
}

fn listed_key(host: &str, sitemap_id: i64) -> Vec<u8> {
    [&[LISTED], host.as_bytes(), b"\0", &sitemap_id.to_be_bytes()].concat()
}

fn sitemap_key(host: &str, url: &str) -> Vec<u8> {
    [&[SITEMAP], host.as_bytes(), b"\0", url.as_bytes()].concat()
}

fn encode(record: &Json) -> Vec<u8> {
    serde_json::to_vec(record).expect("records are plain JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_compress_and_share_the_cache() {
        let dir = std::env::temp_dir().join(format!("desearch-bot-compression-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let resources = Resources::new(8 << 20, 8 << 20, 1);
        let store = BucketStore::open(&dir, &resources).unwrap();
        let entries: Vec<(Url, u32, bool)> = (0..50_000)
            .map(|i| (Url { key: format!("example.com\0example.com/products/category/item-{i}").into_bytes(), flags: 1 }, 0, false))
            .collect();
        let raw: usize = entries.iter().map(|e| e.0.key.len() + 1 + Record::SIZE).sum();
        store.record_listing("example.com", 7, entries, 1).unwrap();
        store.flush().unwrap();
        let stored: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "sst"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(stored > 0 && (stored as usize) < raw / 4, "{stored} bytes stored for {raw} raw");
        assert!(store.url(&Url { key: b"example.com\0example.com/products/category/item-9".to_vec(), flags: 1 }).unwrap().is_some());
        assert!(resources.cache.get_usage() > 0, "reads go through the shared block cache");
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unchanged_entries_skip_their_lookups() {
        let dir = std::env::temp_dir().join(format!("desearch-bot-listed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let resources = Resources::new(8 << 20, 8 << 20, 1);
        let store = BucketStore::open(&dir, &resources).unwrap();
        let url = |path: &str| Url { key: format!("example.com\0example.com/{path}").into_bytes(), flags: 1 };
        let entry = |path: &str, lastmod: u32| (url(path), lastmod, false);
        let read = |path: &str| store.url(&url(path)).unwrap().unwrap();
        let counts = |l: Listing| (l.listed, l.new, l.moved);

        assert_eq!(counts(store.record_listing("example.com", 7, vec![entry("a", 10), entry("b", 20)], 100).unwrap()), (2, 2, 0));
        assert_eq!(counts(store.record_listing("example.com", 7, vec![entry("b", 20), entry("a", 10)], 200).unwrap()), (2, 0, 0));
        assert_eq!(read("a").last_seen, 100, "an entry listed the same way is not rewritten");
        assert_eq!(counts(store.record_listing("example.com", 7, vec![entry("a", 10), entry("b", 30), entry("c", 5)], 300).unwrap()), (3, 1, 1));
        let b = read("b");
        assert_eq!((b.lastmod, b.first_seen, b.last_seen), (30, 100, 300));
        assert_eq!(counts(store.record_listing("example.com", 8, vec![entry("a", 10)], 400).unwrap()), (1, 0, 0));
        assert_eq!(read("a").sitemap_id, 8, "each sitemap remembers only what it listed itself");
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hashes_match_python() {
        assert_eq!(bucket_of("example.com"), 201);
        assert_eq!(sitemap_id("https://example.com/sitemap.xml"), 3_838_397_266_945_991_335);
    }
}

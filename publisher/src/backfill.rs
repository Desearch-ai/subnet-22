//! Reads `current/` a shard at a time for pages to embed: each page's standing version is picked from a few small columns of every file, then only the picks are decoded whole.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use desearch::r2::{self, Bucket};
use futures::{stream, StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaDataReader;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

use crate::compactor::{manifest_key, withdrawn_key, Withdrawn};
use crate::current::{rows_in, standing, Manifest, PageRow, Withdrawals};
use crate::reading::{coalesce, footer_length_at_most, Prefetched, MAX_CURRENT_FOOTER_BYTES};

/// The columns that pick a page's standing version and tell whether it may be wanted.
const PICK_COLUMNS: [&str; 13] =
    ["key", "removed", "change_seq", "version", "fetched_at", "content_sha1", "extraction", "lang", "url", "published", "domain", "task_id", "page_type"];
/// Row groups of one file fetched at once while picking.
const GROUPS_AT_ONCE: usize = 32;
/// Picks read whole at once, in key order.
const PICKS_AT_ONCE: usize = 10_000;
/// Runs of whole row groups fetched ahead of the one being decoded.
const RUNS_AHEAD: usize = 16;
/// Files of a shard whose footers are read at once.
const OPENS_AT_ONCE: usize = 16;
/// Chunks of a file's small columns fetched ahead of the one being decoded.
const CHUNKS_AHEAD: usize = 4;
/// Compressed bytes of whole row groups in one run; `RUNS_AHEAD` runs are fetched at once.
pub const READ_BYTES_AT_ONCE: u64 = 16 << 20;
const TAIL_BYTES: u64 = 1 << 20;

/// One pages file of a shard, its footer read.
struct ShardFile {
    key: String,
    size: u64,
    metadata: ArrowReaderMetadata,
}

/// A standing version chosen to be read whole: its key, and the file and row group that hold it.
struct Pick {
    key: String,
    file: usize,
    group: usize,
}

pub struct ShardReader {
    pub bucket: Bucket,
    pub prefix: String,
    pub handle: Handle,
    pub ranges_at_once: usize,
    /// Compressed bytes of whole row groups fetched at once, `READ_BYTES_AT_ONCE` in service.
    pub bytes_at_once: u64,
}

impl ShardReader {
    /// Hands `each` the shard's standing pages that `keep` accepts, whole and in key order from after `after`, until it returns false; how many it was handed.
    pub fn read(
        &self,
        shard: usize,
        after: Option<&str>,
        mut withdrawn: Withdrawals,
        keep: impl Fn(&PageRow) -> bool,
        mut each: impl FnMut(PageRow) -> bool,
    ) -> Result<usize> {
        let Some(manifest) = self.manifest(shard)? else { return Ok(0) };
        for (task_id, taken) in self.withdrawn()? {
            withdrawn.entry(task_id).or_insert(taken);
        }
        let gets = Arc::new(Semaphore::new(self.ranges_at_once.max(1)));
        let opening = stream::iter(manifest.files().map(|file| open(self.bucket.clone(), file.key.clone(), file.bytes, gets.clone())).collect::<Vec<_>>());
        let files: Vec<ShardFile> = self.handle.block_on(opening.buffered(OPENS_AT_ONCE).try_collect())?;
        let picks = self.pick(&files, after, &withdrawn, keep, &gets)?;
        // Each window's runs of whole row groups, fetched a few runs ahead of the one being decoded.
        let mut jobs: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        let mut wanted: Vec<HashMap<(usize, usize), HashSet<&str>>> = Vec::new();
        for (w, window) in picks.chunks(PICKS_AT_ONCE).enumerate() {
            let mut keys: HashMap<(usize, usize), HashSet<&str>> = HashMap::new();
            for pick in window {
                keys.entry((pick.file, pick.group)).or_default().insert(pick.key.as_str());
            }
            for (at, file) in files.iter().enumerate() {
                let mut groups: Vec<usize> = keys.keys().filter(|(f, _)| *f == at).map(|(_, group)| *group).collect();
                groups.sort_unstable();
                let size = |group: usize| file.metadata.metadata().row_group(group).compressed_size() as u64;
                jobs.extend(runs(&groups, size, self.bytes_at_once).into_iter().map(|run| (w, at, run)));
            }
            wanted.push(keys);
        }
        let fetches = jobs.iter().map(|(_, at, run)| {
            let file = &files[*at];
            Abort(self.handle.spawn(fetch(self.bucket.clone(), file.key.clone(), file.size, file.metadata.clone(), run.clone(), None, gets.clone())))
        });
        let mut fetched = stream::iter(fetches).buffered(RUNS_AHEAD);
        let (mut handed, mut rows) = (0, Vec::new());
        for (n, (w, at, run)) in jobs.iter().enumerate() {
            let file = &files[*at];
            let got = self.handle.block_on(fetched.next()).expect("a fetch for each run").map_err(anyhow::Error::from).and_then(|got| got)?;
            for &group in run {
                let keys = &wanted[*w][&(*at, group)];
                let wanted = |key: &str| keys.contains(key);
                rows.extend(
                    rows_in(got.clone(), file.metadata.clone(), vec![group], ProjectionMask::all(), Some(&wanted)).with_context(|| format!("decoding {}", file.key))?,
                );
            }
            if jobs.get(n + 1).is_none_or(|(next, _, _)| next != w) {
                rows.sort_by(|a: &PageRow, b: &PageRow| a.key.cmp(&b.key));
                for row in rows.drain(..) {
                    if !each(row) {
                        return Ok(handed);
                    }
                    handed += 1;
                }
            }
        }
        Ok(handed)
    }

    /// The tasks the compactor has seen taken back, each with the change file that took it back.
    fn withdrawn(&self) -> Result<Withdrawals> {
        let key = withdrawn_key(&self.prefix);
        match self.handle.block_on(self.bucket.get(&key)) {
            Ok(raw) => Ok(serde_json::from_slice::<Withdrawn>(&raw).with_context(|| format!("reading {key}"))?.by_task()),
            Err(r2::Error::Missing) => Ok(Withdrawals::new()),
            Err(error) => Err(anyhow!("reading {key}: {error}")),
        }
    }

    fn manifest(&self, shard: usize) -> Result<Option<Manifest>> {
        let key = manifest_key(&self.prefix, shard);
        match self.handle.block_on(self.bucket.get(&key)) {
            Ok(raw) => Ok(Some(serde_json::from_slice(&raw).with_context(|| format!("reading {key}"))?)),
            Err(r2::Error::Missing) => Ok(None),
            Err(error) => Err(anyhow!("reading {key}: {error}")),
        }
    }

    /// Every key's standing version across the shard's files, by the compactor's rule with its take-backs, where `keep` accepts it.
    fn pick(&self, files: &[ShardFile], after: Option<&str>, withdrawn: &Withdrawals, keep: impl Fn(&PageRow) -> bool, gets: &Arc<Semaphore>) -> Result<Vec<Pick>> {
        let mut sources: Vec<_> = files.iter().enumerate().map(|(at, file)| self.small_rows(at, file, gets.clone()).peekable()).collect();
        let mut picks = Vec::new();
        loop {
            let mut least: Option<String> = None;
            for source in &mut sources {
                match source.peek() {
                    Some(Ok((_, _, row))) if least.as_ref().is_none_or(|least| row.key < *least) => least = Some(row.key.clone()),
                    Some(Err(_)) => return Err(source.next().expect("peeked").expect_err("peeked an error")),
                    _ => {}
                }
            }
            let Some(key) = least else { break };
            let mut group = Vec::new();
            for source in &mut sources {
                while source.peek().is_some_and(|row| row.as_ref().is_ok_and(|(_, _, row)| row.key == key)) {
                    group.push(source.next().expect("peeked")?);
                }
            }
            if after.is_some_and(|after| key.as_str() <= after) {
                continue;
            }
            // A row a withdrawn task logged after its take-back is dropped, as the compactor drops it.
            let (places, rows): (Vec<(usize, usize)>, Vec<Option<PageRow>>) = group
                .into_iter()
                .filter(|(_, _, row)| row.removed || withdrawn.get(&row.record.task_id).is_none_or(|&taken| row.change_seq <= taken))
                .map(|(file, group, row)| ((file, group), Some(row)))
                .unzip();
            let (_, _, page) = standing(&rows, withdrawn);
            if let Some(at) = page.filter(|&at| rows[at].as_ref().is_some_and(&keep)) {
                let (file, group) = places[at];
                picks.push(Pick { key, file, group });
            }
        }
        Ok(picks)
    }

    /// One file's rows with only `PICK_COLUMNS` set, in file order, with the file's place and each row's row group; a few chunks are fetched ahead of the one decoded.
    fn small_rows<'a>(&'a self, at: usize, file: &'a ShardFile, gets: Arc<Semaphore>) -> impl Iterator<Item = Result<(usize, usize, PageRow)>> + 'a {
        let groups = file.metadata.metadata().num_row_groups();
        let roots = file.metadata.parquet_schema().root_schema().get_fields();
        let indices: Vec<usize> = roots.iter().enumerate().filter(|(_, field)| PICK_COLUMNS.contains(&field.name())).map(|(i, _)| i).collect();
        let mask = ProjectionMask::roots(file.metadata.parquet_schema(), indices);
        // `None` ends the file; a channel closed without it is a fetch task that died.
        let (send, mut chunks) = mpsc::channel::<Result<Option<(Vec<usize>, Prefetched)>>>(CHUNKS_AHEAD);
        let (bucket, key, size, metadata) = (self.bucket.clone(), file.key.clone(), file.size, file.metadata.clone());
        self.handle.spawn(async move {
            for first in (0..groups).step_by(GROUPS_AT_ONCE) {
                let ids: Vec<usize> = (first..groups.min(first + GROUPS_AT_ONCE)).collect();
                let got = fetch(bucket.clone(), key.clone(), size, metadata.clone(), ids.clone(), Some(&PICK_COLUMNS), gets.clone()).await;
                let failed = got.is_err();
                if send.send(got.map(|got| Some((ids, got)))).await.is_err() || failed {
                    return;
                }
            }
            let _ = send.send(Ok(None)).await;
        });
        let mut ended = false;
        std::iter::from_fn(move || match (ended, chunks.blocking_recv()) {
            (true, _) | (false, Some(Ok(None))) => {
                ended = true;
                None
            }
            (false, Some(Ok(Some(chunk)))) => Some(Ok(chunk)),
            (false, Some(Err(error))) => {
                ended = true;
                Some(Err(error))
            }
            (false, None) => {
                ended = true;
                Some(Err(anyhow!("reading {}: its fetches stopped before its end", file.key)))
            }
        })
        .flat_map(move |chunk| {
            let read = || -> Result<Vec<(usize, usize, PageRow)>> {
                let (ids, fetched) = chunk?;
                let mut rows = Vec::new();
                for group in ids {
                    let decoded =
                        rows_in(fetched.clone(), file.metadata.clone(), vec![group], mask.clone(), None).with_context(|| format!("decoding {}", file.key))?;
                    rows.extend(decoded.into_iter().map(|row| (at, group, row)));
                }
                Ok(rows)
            };
            match read() {
                Ok(rows) => rows.into_iter().map(Ok).collect::<Vec<_>>(),
                Err(error) => vec![Err(error)],
            }
        })
    }
}

/// A spawned fetch, stopped when dropped, so a read given up never keeps downloading.
struct Abort<T>(JoinHandle<T>);

impl<T> Drop for Abort<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl<T> std::future::Future for Abort<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.0).poll(cx)
    }
}

/// One pages file of a shard, its footer read: the last `TAIL_BYTES` first, then the rest of a longer footer.
async fn open(bucket: Bucket, key: String, size: u64, gets: Arc<Semaphore>) -> Result<ShardFile> {
    let range = |range: Range<u64>| {
        let (bucket, key, gets) = (bucket.clone(), key.clone(), gets.clone());
        async move {
            let _get = gets.acquire_owned().await?;
            bucket.get_range(&key, range, None).await.map_err(|error| anyhow!("reading {key}: {error}"))
        }
    };
    let mut tail_start = size.saturating_sub(TAIL_BYTES);
    let (head, mut tail) = futures::try_join!(range(0..4), range(tail_start..size))?;
    let length = footer_length_at_most(size, &head, &tail, MAX_CURRENT_FOOTER_BYTES).ok_or_else(|| anyhow!("{key} is not a Parquet file"))?;
    let footer_start = size - 8 - length;
    if footer_start < tail_start {
        tail = [range(footer_start..tail_start).await?, tail].concat().into();
        tail_start = footer_start;
    }
    let at = (footer_start - tail_start) as usize;
    let metadata = ParquetMetaDataReader::decode_metadata(&tail[at..at + length as usize]).with_context(|| format!("the footer of {key}"))?;
    let metadata = ArrowReaderMetadata::try_new(metadata.into(), ArrowReaderOptions::new())?;
    Ok(ShardFile { key, size, metadata })
}

/// The byte ranges of some row groups of a file, every column or only those named, each range a GET within the shard's limit of GETs at once.
async fn fetch(
    bucket: Bucket,
    key: String,
    size: u64,
    metadata: ArrowReaderMetadata,
    groups: Vec<usize>,
    columns: Option<&'static [&'static str]>,
    gets: Arc<Semaphore>,
) -> Result<Prefetched> {
    let mut chunks: Vec<Range<u64>> = Vec::new();
    for group in groups {
        for column in metadata.metadata().row_group(group).columns() {
            let root = column.column_path().parts().first().cloned().unwrap_or_default();
            if columns.is_some_and(|columns| !columns.contains(&root.as_str())) {
                continue;
            }
            let start = column.dictionary_page_offset().unwrap_or(column.data_page_offset()) as u64;
            chunks.push(start..start + column.compressed_size() as u64);
        }
    }
    let ranges = coalesce(chunks);
    let reads = stream::iter(ranges.iter().cloned())
        .map(|range| {
            let (bucket, key, gets) = (bucket.clone(), key.clone(), gets.clone());
            async move {
                let _get = gets.acquire_owned().await?;
                bucket.get_range(&key, range, None).await.map_err(|error| anyhow!("reading {key}: {error}"))
            }
        })
        .buffered(ranges.len().max(1));
    let fetched: Vec<Bytes> = reads.try_collect().await?;
    Ok(Prefetched::new(size, ranges.iter().map(|range| range.start).zip(fetched).collect()))
}

/// `groups` in order, cut into runs of at most `most` bytes, or of one group larger than that.
fn runs(groups: &[usize], size: impl Fn(usize) -> u64, most: u64) -> Vec<Vec<usize>> {
    let (mut runs, mut bytes): (Vec<Vec<usize>>, u64) = (Vec::new(), 0);
    for &group in groups {
        match runs.last_mut() {
            Some(run) if bytes + size(group) <= most => run.push(group),
            _ => {
                runs.push(vec![group]);
                bytes = 0;
            }
        }
        bytes += size(group);
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_current_file_s_footer_may_be_larger_than_an_upload_s() {
        let tail = |length: u32| [&length.to_le_bytes()[..], b"PAR1"].concat();
        let size = 4_000_000_000;
        assert_eq!(crate::reading::footer_length(size, b"PAR1", &tail(20_181_489)), None);
        assert_eq!(footer_length_at_most(size, b"PAR1", &tail(20_181_489), MAX_CURRENT_FOOTER_BYTES), Some(20_181_489));
    }

    #[test]
    fn row_groups_are_read_in_runs_of_bounded_bytes_in_order() {
        let sizes = [30, 30, 50, 10, 120, 5, 5];
        assert_eq!(runs(&[0, 1, 2, 3, 4, 5, 6], |group| sizes[group], 64), [vec![0, 1], vec![2, 3], vec![4], vec![5, 6]]);
        assert_eq!(runs(&[1, 3, 6], |group| sizes[group], 64), [vec![1, 3, 6]]);
        assert!(runs(&[], |group| sizes[group], 64).is_empty());
    }
}

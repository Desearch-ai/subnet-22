//! Reading an upload no validator checked: only the columns published, in byte ranges, refused before decoding when too large.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use bytes::{Buf, Bytes};
use desearch::r2::{Bucket, Head};
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use parquet::errors::ParquetError;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::reader::{ChunkReader, Length};
use tokio::runtime::Handle;

use crate::records::{Row, MAX_US, MIN_US, ROW_COLUMNS};

pub const MAGIC: &[u8; 4] = b"PAR1";
pub const MAX_FOOTER_BYTES: u64 = 2_000_000;
pub const MAX_ROW_BYTES: i64 = 6_000_000;
pub const MAX_ROW_GROUP_BYTES: i64 = 512_000_000;
/// The footer and its length are read in one request whenever the footer is this small.
const TAIL_BYTES: u64 = 64 << 10;
/// Column chunks closer than this are fetched in one request, as pyarrow's pre-buffering does.
const HOLE_BYTES: u64 = 8 << 10;
const RANGE_LIMIT: u64 = 32 << 20;
const BATCH_ROWS: usize = 1024;

/// A stored object read in byte ranges: a local file now, an R2 object next.
pub trait RangeRead: Send + Sync {
    fn size(&self) -> u64;
    fn read(&self, range: Range<u64>) -> io::Result<Bytes>;

    /// Several ranges in order; a remote source fetches them at once.
    fn read_many(&self, ranges: &[Range<u64>]) -> io::Result<Vec<Bytes>> {
        ranges.iter().map(|range| self.read(range.clone())).collect()
    }
}

pub struct LocalFile {
    file: File,
    size: u64,
}

impl LocalFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        Ok(LocalFile { file, size })
    }
}

impl RangeRead for LocalFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read(&self, range: Range<u64>) -> io::Result<Bytes> {
        let mut buffer = vec![0u8; (range.end - range.start) as usize];
        self.file.read_exact_at(&mut buffer, range.start)?;
        Ok(buffer.into())
    }
}

/// Bytes and requests a source served, as R2 would bill them.
#[derive(Default)]
pub struct Traffic {
    pub bytes: AtomicU64,
    pub requests: AtomicU64,
}

pub struct Counted<R> {
    pub inner: R,
    pub traffic: Arc<Traffic>,
}

impl<R: RangeRead> RangeRead for Counted<R> {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn read(&self, range: Range<u64>) -> io::Result<Bytes> {
        self.read_many(std::slice::from_ref(&range)).map(|mut found| found.remove(0))
    }

    fn read_many(&self, ranges: &[Range<u64>]) -> io::Result<Vec<Bytes>> {
        self.traffic.requests.fetch_add(ranges.len() as u64, Ordering::Relaxed);
        self.traffic.bytes.fetch_add(ranges.iter().map(|r| r.end - r.start).sum(), Ordering::Relaxed);
        self.inner.read_many(ranges)
    }
}

/// An upload read in ranges from R2, several ranges at once, each refused if the object changed since its HEAD.
pub struct Remote {
    pub bucket: Bucket,
    pub key: String,
    pub head: Head,
    pub handle: Handle,
    pub ranges_at_once: usize,
    pub traffic: Arc<Traffic>,
}

impl Remote {
    fn count(&self, ranges: &[Range<u64>]) {
        self.traffic.requests.fetch_add(ranges.len() as u64, Ordering::Relaxed);
        self.traffic.bytes.fetch_add(ranges.iter().map(|r| r.end - r.start).sum(), Ordering::Relaxed);
    }
}

impl RangeRead for Remote {
    fn size(&self) -> u64 {
        self.head.size
    }

    fn read(&self, range: Range<u64>) -> io::Result<Bytes> {
        self.count(std::slice::from_ref(&range));
        let etag = Some(self.head.etag.as_str()).filter(|e| !e.is_empty());
        self.handle.block_on(self.bucket.get_range(&self.key, range, etag)).map_err(io::Error::other)
    }

    fn read_many(&self, ranges: &[Range<u64>]) -> io::Result<Vec<Bytes>> {
        self.count(ranges);
        let etag = Some(self.head.etag.as_str()).filter(|e| !e.is_empty());
        let reads =
            futures::stream::iter(ranges.iter().cloned()).map(|range| self.bucket.get_range(&self.key, range, etag)).buffered(self.ranges_at_once.max(1));
        self.handle.block_on(reads.try_collect()).map_err(io::Error::other)
    }
}

/// The footer's length when both magic markers are present and the footer is small enough to parse.
pub fn footer_length(size: u64, head: &[u8], tail: &[u8]) -> Option<u64> {
    if size < 12 || head != MAGIC || tail.len() < 8 || &tail[tail.len() - 4..] != MAGIC {
        return None;
    }
    let at = tail.len() - 8;
    let length = u64::from(u32::from_le_bytes(tail[at..at + 4].try_into().ok()?));
    (length > 0 && length <= MAX_FOOTER_BYTES.min(size - 12)).then_some(length)
}

/// Checked on the footer, before decompressing anything.
pub fn too_big(metadata: &ParquetMetaData, assigned: usize) -> bool {
    let groups: Vec<i64> = metadata
        .row_groups()
        .iter()
        .map(|group| group.columns().iter().filter(|c| wanted(&c.column_path().string())).map(|c| c.uncompressed_size()).sum())
        .collect();
    let rows = assigned.max(1) as i128;
    i128::from(metadata.file_metadata().num_rows()) > 2 * rows
        || groups.iter().map(|&g| i128::from(g)).sum::<i128>() > rows * i128::from(MAX_ROW_BYTES)
        || groups.iter().max().is_some_and(|&g| g > MAX_ROW_GROUP_BYTES)
}

fn wanted(path: &str) -> bool {
    ROW_COLUMNS.contains(&path.split('.').next().unwrap_or(""))
}

/// The published columns of every row; None when the file cannot be decoded safely, an error only when the source fails.
pub fn read_rows(source: &dyn RangeRead, assigned: usize) -> io::Result<Option<Vec<Row>>> {
    let size = source.size();
    if size < 12 {
        return Ok(None);
    }
    let mut tail_start = size.saturating_sub(TAIL_BYTES);
    let (head, mut tail) = if tail_start == 0 {
        let whole = source.read(0..size)?;
        (whole.slice(..4), whole)
    } else {
        let mut found = source.read_many(&[0..4, tail_start..size])?;
        let tail = found.pop().unwrap_or_default();
        (found.pop().unwrap_or_default(), tail)
    };
    let Some(length) = footer_length(size, &head, &tail) else {
        return Ok(None);
    };
    let footer_start = size - 8 - length;
    if footer_start < tail_start {
        let mut whole = source.read(footer_start..tail_start)?.to_vec();
        whole.extend_from_slice(&tail);
        (tail, tail_start) = (whole.into(), footer_start);
    }
    let at = (footer_start - tail_start) as usize;
    let Ok(metadata) = ParquetMetaDataReader::decode_metadata(&tail[at..at + length as usize]) else {
        return Ok(None);
    };
    if too_big(&metadata, assigned) {
        return Ok(None);
    }
    let mut chunks = Vec::new();
    for group in metadata.row_groups() {
        for column in group.columns().iter().filter(|c| wanted(&c.column_path().string())) {
            let (Ok(start), Ok(length)) =
                (u64::try_from(column.dictionary_page_offset().unwrap_or(column.data_page_offset())), u64::try_from(column.compressed_size()))
            else {
                return Ok(None);
            };
            if start.checked_add(length).is_none_or(|end| end > footer_start) {
                return Ok(None);
            }
            chunks.push(start..start + length);
        }
    }
    let ranges = coalesce(chunks);
    let fetched = source.read_many(&ranges)?;
    let ranges = ranges.iter().map(|r| r.start).zip(fetched).collect();
    Ok(decode(Prefetched { size, ranges }, metadata, assigned))
}

fn coalesce(mut chunks: Vec<Range<u64>>) -> Vec<Range<u64>> {
    chunks.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::new();
    for chunk in chunks.into_iter().filter(|c| !c.is_empty()) {
        match merged.last_mut() {
            Some(last) if chunk.start <= last.end + HOLE_BYTES && chunk.end.max(last.end) - last.start <= RANGE_LIMIT => {
                last.end = last.end.max(chunk.end);
            }
            _ => merged.push(chunk),
        }
    }
    merged
}

/// The byte ranges fetched for one file, served to the Parquet decoder.
struct Prefetched {
    size: u64,
    ranges: Vec<(u64, Bytes)>,
}

impl Prefetched {
    fn find(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let i = self.ranges.partition_point(|(s, _)| *s <= start);
        let (s, data) = i.checked_sub(1).map(|i| &self.ranges[i]).ok_or_else(|| ParquetError::EOF("outside the fetched ranges".into()))?;
        let offset = (start - s) as usize;
        if offset + length > data.len() {
            return Err(ParquetError::EOF("outside the fetched ranges".into()));
        }
        Ok(data.slice(offset..))
    }
}

impl Length for Prefetched {
    fn len(&self) -> u64 {
        self.size
    }
}

impl ChunkReader for Prefetched {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        Ok(self.find(start, 0)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        Ok(self.find(start, length)?.slice(..length))
    }
}

fn decode(file: Prefetched, metadata: ParquetMetaData, assigned: usize) -> Option<Vec<Row>> {
    let metadata = ArrowReaderMetadata::try_new(Arc::new(metadata), ArrowReaderOptions::new()).ok()?;
    let roots = metadata.parquet_schema().root_schema().get_fields();
    let mut indices = Vec::with_capacity(ROW_COLUMNS.len());
    for name in ROW_COLUMNS {
        indices.push(roots.iter().position(|field| field.name() == name)?);
    }
    let mask = ProjectionMask::roots(metadata.parquet_schema(), indices);
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata).with_projection(mask).with_batch_size(BATCH_ROWS).build().ok()?;
    // The footer is the miner's word; the decoded columns are not.
    let limit = assigned.max(1) as u64 * MAX_ROW_BYTES as u64;
    let mut decoded = 0u64;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.ok()?;
        decoded += batch.columns().iter().map(|c| c.to_data().get_slice_memory_size().unwrap_or(usize::MAX) as u64).sum::<u64>();
        if decoded > limit {
            return None;
        }
        rows.extend(batch_rows(&batch)?);
    }
    Some(rows)
}

fn batch_rows(batch: &RecordBatch) -> Option<Vec<Row>> {
    let column = |name: &str| batch.column_by_name(name);
    let text = |name: &str| strings(column(name)?);
    let mut url = text("url")?.into_iter();
    let mut final_url = text("final_url")?.into_iter();
    let mut status = ints(column("status")?)?.into_iter();
    let mut error = text("error")?.into_iter();
    let mut fetched_at = times(column("fetched_at")?)?.into_iter();
    let mut page_type = text("page_type")?.into_iter();
    let mut title = text("title")?.into_iter();
    let mut description = text("description")?.into_iter();
    let mut lang = text("lang")?.into_iter();
    let mut canonical = text("canonical")?.into_iter();
    let mut published = text("published")?.into_iter();
    let mut author = text("author")?.into_iter();
    let mut json_ld_types = lists(column("json_ld_types")?)?.into_iter();
    let mut headings = lists(column("headings")?)?.into_iter();
    let mut body = text("text")?.into_iter();
    let mut text_sha256 = text("text_sha256")?.into_iter();
    let mut rows = Vec::with_capacity(batch.num_rows());
    for _ in 0..batch.num_rows() {
        rows.push(Row {
            url: url.next()?,
            final_url: final_url.next()?,
            status: status.next()?,
            error: error.next()?,
            fetched_at: fetched_at.next()?,
            page_type: page_type.next()?,
            title: title.next()?,
            description: description.next()?,
            lang: lang.next()?,
            canonical: canonical.next()?,
            published: published.next()?,
            author: author.next()?,
            json_ld_types: json_ld_types.next()?,
            headings: headings.next()?,
            text: body.next()?,
            text_sha256: text_sha256.next()?,
        });
    }
    Some(rows)
}

/// A text column's values, None when the column holds something other than text.
fn strings(array: &ArrayRef) -> Option<Vec<Option<String>>> {
    let owned = |v: Option<&str>| v.map(str::to_string);
    Some(match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().iter().map(owned).collect(),
        DataType::LargeUtf8 => array.as_string::<i64>().iter().map(owned).collect(),
        DataType::Utf8View => array.as_string_view().iter().map(owned).collect(),
        DataType::Dictionary(_, values) if matches!(**values, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) => {
            return strings(&arrow_cast::cast(array, &DataType::LargeUtf8).ok()?);
        }
        _ => return None,
    })
}

fn ints(array: &ArrayRef) -> Option<Vec<Option<i32>>> {
    fn widen<T: arrow_array::ArrowPrimitiveType>(array: &ArrayRef) -> Option<Vec<Option<i32>>>
    where
        T::Native: TryInto<i32>,
    {
        array.as_primitive::<T>().iter().map(|v| v.map(|v| v.try_into().ok()).map_or(Some(None), |v| v.map(Some))).collect()
    }
    match array.data_type() {
        DataType::Int8 => widen::<Int8Type>(array),
        DataType::Int16 => widen::<Int16Type>(array),
        DataType::Int32 => widen::<Int32Type>(array),
        DataType::Int64 => widen::<Int64Type>(array),
        DataType::UInt8 => widen::<UInt8Type>(array),
        DataType::UInt16 => widen::<UInt16Type>(array),
        DataType::UInt32 => widen::<UInt32Type>(array),
        DataType::UInt64 => widen::<UInt64Type>(array),
        _ => None,
    }
}

/// Fetch times in microseconds; a column of anything but timestamps is no time at all, as Python's `_utc` treats it.
fn times(array: &ArrayRef) -> Option<Vec<Option<i64>>> {
    let DataType::Timestamp(unit, _) = array.data_type() else {
        return Some(vec![None; array.len()]);
    };
    let values = arrow_cast::cast(array, &DataType::Int64).ok()?;
    let micros = |v: i64| -> Option<i64> {
        let us = match unit {
            TimeUnit::Second => v.checked_mul(1_000_000)?,
            TimeUnit::Millisecond => v.checked_mul(1_000)?,
            TimeUnit::Microsecond => v,
            TimeUnit::Nanosecond if v % 1_000 == 0 => v / 1_000,
            TimeUnit::Nanosecond => return None,
        };
        (MIN_US..=MAX_US).contains(&us).then_some(us)
    };
    values.as_primitive::<Int64Type>().iter().map(|v| v.map_or(Some(None), |v| micros(v).map(Some))).collect()
}

fn lists(array: &ArrayRef) -> Option<Vec<Option<Vec<Option<String>>>>> {
    let (offsets, values): (Vec<usize>, ArrayRef) = match array.data_type() {
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            (list.value_offsets().iter().map(|&o| o as usize).collect(), list.values().clone())
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            (list.value_offsets().iter().map(|&o| o as usize).collect(), list.values().clone())
        }
        _ => return None,
    };
    let values = strings(&values)?;
    Some((0..array.len()).map(|i| array.is_valid(i).then(|| values[offsets[i]..offsets[i + 1]].to_vec())).collect())
}

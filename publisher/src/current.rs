//! The current state of every page: 1,024 shards by URL hash, each a base file plus deltas of rows sorted by key, and a manifest.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::builder::{BooleanBuilder, Int32Builder, Int64Builder, LargeStringBuilder, ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_buffer::NullBuffer;
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

use crate::records::{record_version, Record, SOURCE};

pub const SHARDS: usize = 1024;
/// One lookup reads one row group.
pub const ROW_GROUP: usize = 1000;
const ZSTD_LEVEL: i32 = 3;
/// Rows decoded at a time when a pages file is read in order; a merge holds this many of every file it reads.
const READ_BATCH: usize = 256;

pub static PAGES_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let text = |name: &str| Field::new(name, DataType::Utf8, true);
    let list = |name: &str| Field::new(name, DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true);
    Arc::new(Schema::new(vec![
        text("key"),
        Field::new("removed", DataType::Boolean, false),
        Field::new("change_seq", DataType::Int64, false),
        text("version"),
        text("published_at"),
        text("url"),
        text("domain"),
        text("doc_id"),
        text("title"),
        text("published"),
        text("author"),
        text("lang"),
        Field::new("text", DataType::LargeUtf8, true),
        text("fetched_at"),
        text("content_sha1"),
        text("source"),
        text("captured_at"),
        text("assigned_url"),
        text("final_url"),
        text("canonical"),
        Field::new("status", DataType::Int32, true),
        text("page_type"),
        text("description"),
        list("json_ld_types"),
        list("headings"),
        text("text_sha256"),
        text("task_id"),
        text("miner"),
        text("validator"),
        list("validators"),
        text("extraction"),
        text("text_mode"),
        text("modified"),
        text("site_name"),
        text("image"),
        text("section"),
        list("tags"),
        list("breadcrumbs"),
        Field::new("entity", DataType::LargeUtf8, true),
        Field::new("ld", DataType::LargeUtf8, true),
        Field::new("paywalled", DataType::Boolean, true),
        list("robots"),
    ]))
});

/// The shard of a page key: the first 10 bits of the URL hash that ends the key.
pub fn shard_of(key: &str) -> Option<usize> {
    let hash = key.rsplit('/').next()?;
    let head = u16::from_str_radix(hash.get(..3)?, 16).ok()?;
    Some(usize::from(head >> 2))
}

pub fn shard_dir(shard: usize) -> String {
    format!("shard={shard:04}")
}

/// One version of one page as the current state keeps it; a removed row stands for the page's removal.
#[derive(Clone, Debug, PartialEq)]
pub struct PageRow {
    pub key: String,
    pub removed: bool,
    pub change_seq: i64,
    pub version: String,
    pub published_at: String,
    pub record: Record,
}

impl PageRow {
    pub fn removal(key: String, change_seq: i64) -> PageRow {
        PageRow { key, removed: true, change_seq, version: String::new(), published_at: String::new(), record: Record::default() }
    }

    /// Whether this row replaces `other` for the same key: removals go by log order, pages by the later fetch as the publisher decides.
    pub fn replaces(&self, other: &PageRow) -> bool {
        if self.removed || other.removed {
            return self.change_seq > other.change_seq;
        }
        (self.record.fetched_at.as_str(), self.version.as_str(), self.change_seq) > (other.record.fetched_at.as_str(), other.version.as_str(), other.change_seq)
    }
}

/// One row of a change file, in file order.
#[derive(Clone, Debug, PartialEq)]
pub enum ChangeRow {
    Page(Box<PageRow>),
    Removed(String),
    TaskWithdrawn(String),
}

/// Tasks taken back, each with the change file that took it back.
pub type Withdrawals = HashMap<String, i64>;

/// Each key as it stands, sorted by key: after its last removal the later fetch wins, and a winner from a withdrawn task is taken down where the take-back was logged.
pub fn merge(rows: Vec<PageRow>, withdrawn: &Withdrawals, keep_removals: bool) -> Vec<PageRow> {
    let mut rows: Vec<Option<PageRow>> =
        rows.into_iter().filter(|row| row.removed || withdrawn.get(&row.record.task_id).is_none_or(|&taken| row.change_seq <= taken)).map(Some).collect();
    rows.sort_by(|a, b| a.as_ref().map(|r| &r.key).cmp(&b.as_ref().map(|r| &r.key)));
    let mut merged = Vec::with_capacity(rows.len());
    let mut start = 0;
    while start < rows.len() {
        let key = rows[start].as_ref().map(|r| r.key.clone()).unwrap_or_default();
        let end = start + rows[start..].iter().take_while(|r| r.as_ref().is_some_and(|r| r.key == key)).count();
        let (removal, barrier, page) = standing(&rows[start..end], withdrawn);
        if keep_removals {
            match removal {
                Some(i) => merged.extend(rows[start + i].take()),
                None if barrier > i64::MIN => merged.push(PageRow::removal(key, barrier)),
                None => {}
            }
        }
        merged.extend(page.and_then(|i| rows[start + i].take()));
        start = end;
    }
    merged
}

/// One key's rows: the removal row that stands, if a logged one does; where everything before is hidden; and the page that wins after it.
fn standing(group: &[Option<PageRow>], withdrawn: &Withdrawals) -> (Option<usize>, i64, Option<usize>) {
    let row = |i: usize| group[i].as_ref().expect("a row not taken yet");
    let removal = (0..group.len()).filter(|&i| row(i).removed).max_by_key(|&i| row(i).change_seq);
    let mut barrier = removal.map_or(i64::MIN, |i| row(i).change_seq);
    loop {
        let best =
            (0..group.len()).filter(|&i| !row(i).removed && row(i).change_seq > barrier).reduce(|kept, i| if row(i).replaces(row(kept)) { i } else { kept });
        match best.and_then(|i| withdrawn.get(&row(i).record.task_id)) {
            Some(&taken) => barrier = barrier.max(taken),
            None => return (removal.filter(|&i| row(i).change_seq == barrier), barrier, best),
        }
    }
}

/// Merges pages files, each sorted by key and given oldest first, into one, holding about a row group of each at a time.
pub fn merge_files(files: Vec<Bytes>, withdrawn: &Withdrawals, keep_removals: bool) -> Result<(Vec<u8>, u64)> {
    let mut sources = files.into_iter().map(|file| Ok(rows(file, None)?.peekable())).collect::<Result<Vec<_>>>()?;
    let mut out = PagesWriter::new()?;
    loop {
        let mut least: Option<String> = None;
        for source in &mut sources {
            match source.peek() {
                Some(Ok(row)) if least.as_ref().is_none_or(|least| row.key < *least) => least = Some(row.key.clone()),
                Some(Err(_)) => return Err(source.next().expect("peeked").expect_err("peeked an error")),
                _ => {}
            }
        }
        let Some(key) = least else { break };
        let mut group = Vec::new();
        for source in &mut sources {
            while source.peek().is_some_and(|row| row.as_ref().is_ok_and(|row| row.key == key)) {
                group.push(source.next().expect("peeked")?);
            }
        }
        out.push(merge(group, withdrawn, keep_removals))?;
    }
    out.finish()
}

/// Rows written to one pages file as they come, a few row groups at a time.
struct PagesWriter {
    writer: ArrowWriter<Vec<u8>>,
    pending: Vec<PageRow>,
    rows: u64,
}

impl PagesWriter {
    fn new() -> Result<PagesWriter> {
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL)?))
            .set_max_row_group_row_count(Some(ROW_GROUP))
            .build();
        Ok(PagesWriter { writer: ArrowWriter::try_new(Vec::new(), PAGES_SCHEMA.clone(), Some(properties))?, pending: Vec::new(), rows: 0 })
    }

    fn push(&mut self, rows: Vec<PageRow>) -> Result<()> {
        self.pending.extend(rows);
        if self.pending.len() >= ROW_GROUP * 4 {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.pending.is_empty() {
            self.writer.write(&RecordBatch::try_new(PAGES_SCHEMA.clone(), columns(&self.pending))?)?;
            self.rows += self.pending.len() as u64;
            self.pending.clear();
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(Vec<u8>, u64)> {
        self.flush()?;
        Ok((self.writer.into_inner()?, self.rows))
    }
}

pub fn encode(rows: &[PageRow]) -> Result<Vec<u8>> {
    let mut out = PagesWriter::new()?;
    for chunk in rows.chunks(ROW_GROUP * 4) {
        out.push(chunk.to_vec())?;
    }
    Ok(out.finish()?.0)
}

fn columns(rows: &[PageRow]) -> Vec<ArrayRef> {
    let text = |value: fn(&PageRow) -> Option<&str>| -> ArrayRef {
        let mut builder = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
        for row in rows {
            builder.append_option(value(row));
        }
        Arc::new(builder.finish())
    };
    let list = |value: fn(&Record) -> Vec<Option<&str>>| -> ArrayRef {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in rows {
            for item in value(&row.record) {
                builder.values().append_option(item);
            }
            builder.append(true);
        }
        Arc::new(builder.finish())
    };
    // A v1 page's v2 lists are null, as in files written before them.
    let v2_list = |value: fn(&Record) -> &[Option<String>]| -> ArrayRef {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in rows {
            if row.record.is_v1() {
                builder.append(false);
                continue;
            }
            for item in value(&row.record) {
                builder.values().append_option(item.as_deref());
            }
            builder.append(true);
        }
        Arc::new(builder.finish())
    };
    let large = |value: fn(&Record) -> Option<&str>| -> ArrayRef {
        let mut builder = LargeStringBuilder::new();
        for row in rows {
            builder.append_option(value(&row.record));
        }
        Arc::new(builder.finish())
    };
    let mut removed = BooleanBuilder::with_capacity(rows.len());
    let mut seq = Int64Builder::with_capacity(rows.len());
    let mut status = Int32Builder::with_capacity(rows.len());
    let mut paywalled = BooleanBuilder::with_capacity(rows.len());
    let mut body = LargeStringBuilder::with_capacity(rows.len(), rows.iter().map(|r| r.record.text.len()).sum());
    for row in rows {
        removed.append_value(row.removed);
        seq.append_value(row.change_seq);
        status.append_option(row.record.status);
        paywalled.append_option(row.record.paywalled);
        body.append_value(&row.record.text);
    }
    vec![
        text(|r| Some(&r.key)),
        Arc::new(removed.finish()),
        Arc::new(seq.finish()),
        text(|r| Some(&r.version)),
        text(|r| Some(&r.published_at)),
        text(|r| Some(&r.record.url)),
        text(|r| Some(&r.record.domain)),
        text(|r| Some(&r.record.doc_id)),
        text(|r| Some(&r.record.title)),
        text(|r| Some(&r.record.published)),
        text(|r| Some(&r.record.author)),
        text(|r| Some(&r.record.lang)),
        Arc::new(body.finish()),
        text(|r| Some(&r.record.fetched_at)),
        text(|r| Some(&r.record.content_sha1)),
        text(|_| Some(SOURCE)),
        text(|r| Some(&r.record.captured_at)),
        text(|r| Some(&r.record.assigned_url)),
        text(|r| r.record.final_url.as_deref()),
        text(|r| r.record.canonical.as_deref()),
        Arc::new(status.finish()),
        text(|r| r.record.page_type.as_deref()),
        text(|r| r.record.description.as_deref()),
        list(|r| r.json_ld_types.iter().map(Option::as_deref).collect()),
        list(|r| r.headings.iter().map(Option::as_deref).collect()),
        text(|r| r.record.text_sha256.as_deref()),
        text(|r| Some(&r.record.task_id)),
        text(|r| Some(&r.record.miner)),
        text(|r| r.record.validator.as_deref()),
        list(|r| r.validators.iter().map(|v| Some(v.as_str())).collect()),
        text(|r| r.record.extraction.as_deref()),
        text(|r| r.record.text_mode.as_deref()),
        text(|r| r.record.modified.as_deref()),
        text(|r| r.record.site_name.as_deref()),
        text(|r| r.record.image.as_deref()),
        text(|r| r.record.section.as_deref()),
        v2_list(|r| &r.tags),
        v2_list(|r| &r.breadcrumbs),
        large(|r| r.entity.as_deref()),
        large(|r| r.ld.as_deref()),
        Arc::new(paywalled.finish()),
        v2_list(|r| &r.robots),
    ]
}

/// A batch's columns by name, every text column as UTF-8 whichever string type its writer chose, looked up once.
struct Columns {
    rows: usize,
    texts: HashMap<String, StringArray>,
    lists: HashMap<String, (Vec<usize>, StringArray, Option<NullBuffer>)>,
    ints: HashMap<String, Int64Array>,
    flags: HashMap<String, BooleanArray>,
}

impl Columns {
    fn new(batch: RecordBatch) -> Result<Columns> {
        let mut columns = Columns { rows: batch.num_rows(), texts: HashMap::new(), lists: HashMap::new(), ints: HashMap::new(), flags: HashMap::new() };
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            let name = field.name().clone();
            match column.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                    columns.texts.insert(name, cast(column, &DataType::Utf8)?.as_string::<i32>().clone());
                }
                DataType::List(_) | DataType::LargeList(_) => {
                    let (offsets, values, nulls): (Vec<usize>, ArrayRef, Option<NullBuffer>) = match column.data_type() {
                        DataType::List(_) => {
                            let list = column.as_list::<i32>();
                            (list.value_offsets().iter().map(|o| *o as usize).collect(), list.values().clone(), list.nulls().cloned())
                        }
                        _ => {
                            let list = column.as_list::<i64>();
                            (list.value_offsets().iter().map(|o| *o as usize).collect(), list.values().clone(), list.nulls().cloned())
                        }
                    };
                    columns.lists.insert(name, (offsets, cast(&values, &DataType::Utf8)?.as_string::<i32>().clone(), nulls));
                }
                DataType::Int32 | DataType::Int64 => {
                    columns.ints.insert(name, cast(column, &DataType::Int64)?.as_primitive::<Int64Type>().clone());
                }
                DataType::Boolean => {
                    columns.flags.insert(name, column.as_boolean().clone());
                }
                _ => {}
            }
        }
        Ok(columns)
    }

    fn text(&self, name: &str, row: usize) -> Option<String> {
        let column = self.texts.get(name)?;
        column.is_valid(row).then(|| column.value(row).to_string())
    }

    fn texts(&self, name: &str, row: usize) -> Vec<Option<String>> {
        let Some((offsets, values, nulls)) = self.lists.get(name) else { return Vec::new() };
        if nulls.as_ref().is_some_and(|nulls| nulls.is_null(row)) {
            return Vec::new();
        }
        (offsets[row]..offsets[row + 1]).map(|i| values.is_valid(i).then(|| values.value(i).to_string())).collect()
    }

    fn int(&self, name: &str, row: usize) -> Option<i64> {
        let column = self.ints.get(name)?;
        column.is_valid(row).then(|| column.value(row))
    }

    fn flag(&self, name: &str, row: usize) -> Option<bool> {
        let column = self.flags.get(name)?;
        column.is_valid(row).then(|| column.value(row))
    }

    fn page_rows(&self, wanted: Option<&(dyn Fn(&str) -> bool + Send)>) -> Result<Vec<PageRow>> {
        let removed = self.flags.get("removed").ok_or_else(|| anyhow!("a pages file without removed"))?;
        let mut rows = Vec::new();
        for row in 0..self.rows {
            let key = self.text("key", row).unwrap_or_default();
            if wanted.is_some_and(|wanted| !wanted(&key)) {
                continue;
            }
            rows.push(PageRow {
                key,
                removed: removed.value(row),
                change_seq: self.int("change_seq", row).unwrap_or_default(),
                version: self.text("version", row).unwrap_or_default(),
                published_at: self.text("published_at", row).unwrap_or_default(),
                record: self.record(row),
            });
        }
        Ok(rows)
    }

    fn record(&self, row: usize) -> Record {
        let text = |name: &str| self.text(name, row).unwrap_or_default();
        Record {
            url: text("url"),
            domain: text("domain"),
            title: text("title"),
            published: text("published"),
            author: text("author"),
            lang: text("lang"),
            text: text("text"),
            fetched_at: text("fetched_at"),
            content_sha1: text("content_sha1"),
            captured_at: text("captured_at"),
            doc_id: text("doc_id"),
            assigned_url: text("assigned_url"),
            final_url: self.text("final_url", row),
            canonical: self.text("canonical", row),
            status: self.int("status", row).and_then(|s| i32::try_from(s).ok()),
            page_type: self.text("page_type", row),
            description: self.text("description", row),
            json_ld_types: self.texts("json_ld_types", row),
            headings: self.texts("headings", row),
            text_sha256: self.text("text_sha256", row),
            task_id: text("task_id"),
            miner: text("miner"),
            validator: self.text("validator", row),
            validators: self.texts("validators", row).into_iter().flatten().collect(),
            extraction: self.text("extraction", row),
            text_mode: self.text("text_mode", row),
            modified: self.text("modified", row),
            site_name: self.text("site_name", row),
            image: self.text("image", row),
            section: self.text("section", row),
            tags: self.texts("tags", row),
            breadcrumbs: self.texts("breadcrumbs", row),
            entity: self.text("entity", row),
            ld: self.text("ld", row),
            paywalled: self.flag("paywalled", row),
            robots: self.texts("robots", row),
        }
    }
}

fn batches(file: Bytes) -> Result<Vec<Columns>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(ROW_GROUP * 8).build()?;
    reader.map(|batch| Columns::new(batch?)).collect()
}

/// Every row of a change file in file order, from the Rust publisher or the Python one before it.
pub fn decode_changes(file: Bytes, change_seq: i64) -> Result<Vec<ChangeRow>> {
    Ok(decode_changes_where(file, change_seq, |_| true)?.into_iter().map(|(_, row)| row).collect())
}

/// The rows of a change file whose position `keep` accepts, with their positions; the others are not decoded.
pub fn decode_changes_where(file: Bytes, change_seq: i64, keep: impl Fn(usize) -> bool) -> Result<Vec<(usize, ChangeRow)>> {
    let mut rows = Vec::new();
    let mut at = 0;
    for columns in batches(file)? {
        for row in 0..columns.rows {
            let position = at + row;
            if !keep(position) {
                continue;
            }
            let kind = columns.text("kind", row).unwrap_or_default();
            let decoded = match kind.as_str() {
                "task_withdrawn" => ChangeRow::TaskWithdrawn(columns.text("task_id", row).context("a task_withdrawn row without a task id")?),
                "removed" => ChangeRow::Removed(columns.text("key", row).context("a removed row without a key")?),
                "new" | "changed" | "page" => {
                    let record = columns.record(row);
                    ChangeRow::Page(Box::new(PageRow {
                        key: columns.text("key", row).context("a page row without a key")?,
                        removed: false,
                        change_seq,
                        version: record_version(&record),
                        published_at: columns.text("published_at", row).unwrap_or_default(),
                        record,
                    }))
                }
                other => bail!("a change row of kind {other:?}"),
            };
            rows.push((position, decoded));
        }
        at += columns.rows;
    }
    Ok(rows)
}

pub fn decode(file: Bytes) -> Result<Vec<PageRow>> {
    rows(file, None)?.collect()
}

pub type Wanted = Box<dyn Fn(&str) -> bool + Send>;

/// The rows of a pages file in file order, decoded a row group at a time; `wanted` picks keys before the rest of a row is decoded.
pub fn rows(file: Bytes, wanted: Option<Wanted>) -> Result<Rows> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(READ_BATCH).build()?;
    Ok(Rows { reader, batch: Vec::new().into_iter(), wanted })
}

pub struct Rows {
    reader: ParquetRecordBatchReader,
    batch: std::vec::IntoIter<PageRow>,
    wanted: Option<Wanted>,
}

impl Iterator for Rows {
    type Item = Result<PageRow>;

    fn next(&mut self) -> Option<Result<PageRow>> {
        loop {
            if let Some(row) = self.batch.next() {
                return Some(Ok(row));
            }
            let decoded = match self.reader.next()? {
                Ok(batch) => Columns::new(batch).and_then(|columns| columns.page_rows(self.wanted.as_deref())),
                Err(error) => Err(error.into()),
            };
            match decoded {
                Ok(rows) => self.batch = rows.into_iter(),
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

/// One file of a shard and how much it holds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileRef {
    pub key: String,
    pub rows: u64,
    pub bytes: u64,
    pub through_seq: i64,
    /// 0 for a cycle's delta; each combining of `fanout` deltas of one level makes one of the next.
    #[serde(default)]
    pub level: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retired {
    pub key: String,
    pub at: f64,
}

/// A shard's files and how far into the change log they reach; one write of it swaps the shard's state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub shard: usize,
    pub through_seq: i64,
    /// Where the base was last rewritten; a task withdrawn before it is out of every file of the shard.
    pub merged_through: i64,
    pub base: Option<FileRef>,
    pub deltas: Vec<FileRef>,
    pub retired: Vec<Retired>,
}

impl Manifest {
    pub fn delta_rows(&self) -> u64 {
        self.deltas.iter().map(|d| d.rows).sum()
    }

    pub fn files(&self) -> impl Iterator<Item = &FileRef> {
        self.base.iter().chain(&self.deltas)
    }
}

//! The current state of every page: 1,024 shards by URL hash, each a base file plus deltas of rows sorted by key, and a manifest.

use std::collections::{HashMap, HashSet};
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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

use crate::records::{record_version, Record, SOURCE};

pub const SHARDS: usize = 1024;
/// One lookup reads one row group.
pub const ROW_GROUP: usize = 1000;
const ZSTD_LEVEL: i32 = 3;

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

/// The winning version of every key across these rows, sorted by key; a key whose winner is removed or from a withdrawn task is gone.
pub fn merge(mut rows: Vec<PageRow>, withdrawn: &HashSet<String>, keep_removals: bool) -> Vec<PageRow> {
    rows.sort_by(|a, b| a.key.cmp(&b.key));
    let mut merged: Vec<PageRow> = Vec::with_capacity(rows.len());
    for row in rows {
        match merged.last_mut() {
            Some(last) if last.key == row.key => {
                if row.replaces(last) {
                    *last = row;
                }
            }
            _ => merged.push(row),
        }
    }
    merged.retain(|row| !withdrawn.contains(&row.record.task_id) && (keep_removals || !row.removed));
    merged
}

pub fn encode(rows: &[PageRow]) -> Result<Vec<u8>> {
    let properties =
        WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL)?)).set_max_row_group_row_count(Some(ROW_GROUP)).build();
    let mut file = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut file, PAGES_SCHEMA.clone(), Some(properties))?;
    for chunk in rows.chunks(ROW_GROUP * 16) {
        writer.write(&RecordBatch::try_new(PAGES_SCHEMA.clone(), columns(chunk))?)?;
    }
    writer.close()?;
    Ok(file)
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
    let mut removed = BooleanBuilder::with_capacity(rows.len());
    let mut seq = Int64Builder::with_capacity(rows.len());
    let mut status = Int32Builder::with_capacity(rows.len());
    let mut body = LargeStringBuilder::with_capacity(rows.len(), rows.iter().map(|r| r.record.text.len()).sum());
    for row in rows {
        removed.append_value(row.removed);
        seq.append_value(row.change_seq);
        status.append_option(row.record.status);
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
    ]
}

/// A batch's columns by name, every text column as UTF-8 whichever string type its writer chose, looked up once.
struct Columns {
    rows: usize,
    texts: HashMap<String, StringArray>,
    lists: HashMap<String, (Vec<usize>, StringArray, Option<NullBuffer>)>,
    ints: HashMap<String, Int64Array>,
    removed: Option<BooleanArray>,
}

impl Columns {
    fn new(batch: RecordBatch) -> Result<Columns> {
        let mut columns = Columns { rows: batch.num_rows(), texts: HashMap::new(), lists: HashMap::new(), ints: HashMap::new(), removed: None };
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
                DataType::Boolean if name == "removed" => columns.removed = Some(column.as_boolean().clone()),
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
    let mut rows = Vec::new();
    for columns in batches(file)? {
        let removed = columns.removed.clone().ok_or_else(|| anyhow!("a pages file without removed"))?;
        for row in 0..columns.rows {
            rows.push(PageRow {
                key: columns.text("key", row).unwrap_or_default(),
                removed: removed.value(row),
                change_seq: columns.int("change_seq", row).unwrap_or_default(),
                version: columns.text("version", row).unwrap_or_default(),
                published_at: columns.text("published_at", row).unwrap_or_default(),
                record: columns.record(row),
            });
        }
    }
    Ok(rows)
}

/// One file of a shard and how much it holds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileRef {
    pub key: String,
    pub rows: u64,
    pub bytes: u64,
    pub through_seq: i64,
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

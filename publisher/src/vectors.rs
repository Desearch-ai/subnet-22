//! Accepted embed uploads as vector files: each vector joined to the page of its text, one file per task under `vectors/<model>/log/`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use anyhow::{anyhow, bail, Result};
use arrow_array::builder::{BinaryBuilder, Int32Builder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, BinaryArray, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

const ROW_GROUP: usize = 1000;
const ZSTD_LEVEL: i32 = 1;
/// How far a vector's length may be from 1, as validators allow.
const NORM_TOLERANCE: f64 = 0.01;

pub static VECTORS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let text = |name: &str| Field::new(name, DataType::Utf8, false);
    Arc::new(Schema::new(vec![
        text("key"),
        text("url"),
        text("content_sha1"),
        text("kind"),
        Field::new("index", DataType::Int32, false),
        Field::new("vector", DataType::Binary, false),
        text("task_id"),
        text("miner"),
        text("published_at"),
    ]))
});

/// One text's vector and the page version it was taken from.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorRow {
    pub key: String,
    pub url: String,
    pub content_sha1: String,
    pub kind: String,
    pub index: i32,
    pub vector: Vec<u8>,
}

pub fn file_key(model: &str, day: &str, id: &str) -> String {
    format!("vectors/{model}/log/dt={day}/{id}.parquet")
}

/// Every text of the input with its one vector from the upload, sorted by page; refused unless each text has exactly one well-formed vector, all of one size.
pub fn join(input: Bytes, upload: Bytes) -> Result<Vec<VectorRow>> {
    let mut vectors = upload_vectors(upload)?;
    let mut size = None;
    let mut rows = Vec::new();
    for batch in batches(input)? {
        let (ids, keys, urls, hashes, kinds) =
            (strings(&batch, "text_id")?, strings(&batch, "page_key")?, strings(&batch, "url")?, strings(&batch, "content_sha1")?, strings(&batch, "kind")?);
        let index = cast(column(&batch, "index")?, &DataType::Int32)?;
        let index = index.as_primitive::<Int32Type>();
        for row in 0..batch.num_rows() {
            let text_id = ids.value(row);
            let vector = vectors.remove(text_id).ok_or_else(|| anyhow!("no vector for text {text_id}"))?;
            if *size.get_or_insert(vector.len()) != vector.len() || !well_formed(&vector) {
                bail!("the vector for text {text_id} is not a unit vector of the task's size");
            }
            rows.push(VectorRow {
                key: keys.value(row).to_string(),
                url: urls.value(row).to_string(),
                content_sha1: hashes.value(row).to_string(),
                kind: kinds.value(row).to_string(),
                index: index.value(row),
                vector,
            });
        }
    }
    if let Some(extra) = vectors.keys().next() {
        bail!("a vector for text {extra}, which the input does not hold");
    }
    rows.sort_by(|a, b| (&a.key, &a.kind, a.index).cmp(&(&b.key, &b.kind, b.index)));
    Ok(rows)
}

/// Finite float16 values, little-endian, whose length is within the tolerance of 1.
pub fn well_formed(vector: &[u8]) -> bool {
    if vector.is_empty() || !vector.len().is_multiple_of(2) {
        return false;
    }
    let mut squares = 0.0f64;
    for pair in vector.as_chunks::<2>().0 {
        let value = f64::from(half::f16::from_le_bytes(*pair).to_f32());
        if !value.is_finite() {
            return false;
        }
        squares += value * value;
    }
    (squares.sqrt() - 1.0).abs() <= NORM_TOLERANCE
}

pub fn encode(rows: &[VectorRow], task_id: &str, miner: &str, published_at: &str) -> Result<Vec<u8>> {
    let mut columns: Vec<StringBuilder> = (0..4).map(|_| StringBuilder::new()).collect();
    let mut index = Int32Builder::new();
    let mut vector = BinaryBuilder::new();
    for row in rows {
        for (builder, value) in columns.iter_mut().zip([&row.key, &row.url, &row.content_sha1, &row.kind]) {
            builder.append_value(value);
        }
        index.append_value(row.index);
        vector.append_value(&row.vector);
    }
    let repeated = |value: &str| -> ArrayRef { Arc::new(StringArray::from(vec![value; rows.len()])) };
    let mut arrays: Vec<ArrayRef> = columns.iter_mut().map(|builder| Arc::new(builder.finish()) as ArrayRef).collect();
    arrays.push(Arc::new(index.finish()));
    arrays.push(Arc::new(vector.finish()));
    arrays.extend([repeated(task_id), repeated(miner), repeated(published_at)]);
    let batch = RecordBatch::try_new(VECTORS_SCHEMA.clone(), arrays)?;
    let properties =
        WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL)?)).set_max_row_group_row_count(Some(ROW_GROUP)).build();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, VECTORS_SCHEMA.clone(), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(out)
}

/// The rows of a vector file, in file order.
pub fn decode(file: Bytes) -> Result<Vec<VectorRow>> {
    let mut rows = Vec::new();
    for batch in batches(file)? {
        let (keys, urls, hashes, kinds) = (strings(&batch, "key")?, strings(&batch, "url")?, strings(&batch, "content_sha1")?, strings(&batch, "kind")?);
        let index = cast(column(&batch, "index")?, &DataType::Int32)?;
        let index = index.as_primitive::<Int32Type>();
        let vectors = binaries(&batch, "vector")?;
        for row in 0..batch.num_rows() {
            rows.push(VectorRow {
                key: keys.value(row).to_string(),
                url: urls.value(row).to_string(),
                content_sha1: hashes.value(row).to_string(),
                kind: kinds.value(row).to_string(),
                index: index.value(row),
                vector: vectors.value(row).to_vec(),
            });
        }
    }
    Ok(rows)
}

fn upload_vectors(upload: Bytes) -> Result<HashMap<String, Vec<u8>>> {
    let mut vectors = HashMap::new();
    for batch in batches(upload)? {
        let (ids, values) = (strings(&batch, "text_id")?, binaries(&batch, "vector")?);
        for row in 0..batch.num_rows() {
            if vectors.insert(ids.value(row).to_string(), values.value(row).to_vec()).is_some() {
                bail!("text {} has two vectors", ids.value(row));
            }
        }
    }
    Ok(vectors)
}

fn batches(file: Bytes) -> Result<Vec<RecordBatch>> {
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?.build()?.collect::<Result<Vec<_>, _>>()?)
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch.column_by_name(name).ok_or_else(|| anyhow!("no {name} column"))
}

fn strings(batch: &RecordBatch, name: &str) -> Result<StringArray> {
    let column = cast(column(batch, name)?, &DataType::Utf8)?;
    if column.null_count() > 0 {
        bail!("the {name} column has empty values");
    }
    Ok(column.as_string::<i32>().clone())
}

fn binaries(batch: &RecordBatch, name: &str) -> Result<BinaryArray> {
    let column = cast(column(batch, name)?, &DataType::Binary)?;
    if column.null_count() > 0 {
        bail!("the {name} column has empty values");
    }
    Ok(column.as_binary::<i32>().clone())
}

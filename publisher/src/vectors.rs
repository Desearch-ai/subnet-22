//! Embed uploads as vector files: each vector joined to the page of its text, one file per task under `vectors/<model>/log/`, and withdrawal entries for tasks taken back.

use std::collections::HashSet;
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
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::Serialize;

use crate::changes::Kind;
use crate::planner::RECIPE_V1;

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
        text("recipe"),
    ]))
});

/// One text's vector, the page version it was taken from and the recipe that made the text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorRow {
    pub key: String,
    pub url: String,
    pub content_sha1: String,
    pub kind: String,
    pub index: i32,
    pub vector: Vec<u8>,
    pub recipe: String,
}

pub fn file_key(model: &str, day: &str, id: &str) -> String {
    format!("vectors/{model}/log/dt={day}/{id}.parquet")
}

/// A page sent back to the planner, as the task API lists an embed task's pages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DroppedPage {
    pub page_key: String,
    pub content_sha1: String,
    pub url: String,
}

/// A task's vectors joined to its input: the rows to publish, and the pages left out whole because a vector of theirs is malformed.
#[derive(Debug, Default)]
pub struct Joined {
    pub rows: Vec<VectorRow>,
    pub dropped: Vec<DroppedPage>,
    pub malformed: usize,
}

/// Every text of the input with its vector from the raw upload, which holds one float16 vector per text in the input's order; rows sorted by page, refused unless the upload splits into one vector of one size per text.
pub fn join(input: Bytes, upload: &[u8]) -> Result<Joined> {
    let texts = read_rows(input, "page_key", false)?;
    if texts.is_empty() || upload.is_empty() || !upload.len().is_multiple_of(2 * texts.len()) {
        bail!("{} bytes are not one float16 vector for each of {} texts", upload.len(), texts.len());
    }
    let vectors = upload.chunks_exact(upload.len() / texts.len());
    let mut joined = Joined::default();
    let mut bad = HashSet::new();
    for (text, vector) in texts.iter().zip(vectors.clone()) {
        if !well_formed(vector) {
            joined.malformed += 1;
            bad.insert(text.key.as_str());
        }
    }
    let mut sent = HashSet::new();
    for (text, vector) in texts.iter().zip(vectors) {
        if !bad.contains(text.key.as_str()) {
            joined.rows.push(VectorRow { vector: vector.to_vec(), ..text.clone() });
        } else if sent.insert(text.key.as_str()) {
            joined.dropped.push(DroppedPage { page_key: text.key.clone(), content_sha1: text.content_sha1.clone(), url: text.url.clone() });
        }
    }
    joined.rows.sort_by(|a, b| (&a.key, &a.kind, a.index).cmp(&(&b.key, &b.kind, b.index)));
    joined.dropped.sort_by(|a, b| a.page_key.cmp(&b.page_key));
    Ok(joined)
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
    write(rows, &vec![task_id; rows.len()], &vec![miner; rows.len()], published_at)
}

/// A withdrawal entry: a `task_withdrawn` row for each (task, miner) taken back, with only its kind, task, miner and time set; readers drop every vector row of those tasks.
pub fn encode_withdrawn(tasks: &[(String, String)], published_at: &str) -> Result<Vec<u8>> {
    let rows = vec![VectorRow { kind: Kind::TaskWithdrawn.as_str().to_string(), ..VectorRow::default() }; tasks.len()];
    let (task_ids, miners): (Vec<&str>, Vec<&str>) = tasks.iter().map(|(task_id, miner)| (task_id.as_str(), miner.as_str())).unzip();
    write(&rows, &task_ids, &miners, published_at)
}

fn write(rows: &[VectorRow], task_ids: &[&str], miners: &[&str], published_at: &str) -> Result<Vec<u8>> {
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
    let mut arrays: Vec<ArrayRef> = columns.iter_mut().map(|builder| Arc::new(builder.finish()) as ArrayRef).collect();
    arrays.push(Arc::new(index.finish()));
    arrays.push(Arc::new(vector.finish()));
    arrays.push(Arc::new(StringArray::from(task_ids.to_vec())));
    arrays.push(Arc::new(StringArray::from(miners.to_vec())));
    arrays.push(Arc::new(StringArray::from(vec![published_at; rows.len()])));
    arrays.push(Arc::new(StringArray::from(rows.iter().map(|row| row.recipe.as_str()).collect::<Vec<_>>())));
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
    read_rows(file, "key", true)
}

/// The pages of a vector file, each once, read without its vectors.
pub fn pages(file: Bytes) -> Result<Vec<DroppedPage>> {
    let mut seen = HashSet::new();
    Ok(read_rows(file, "key", false)?
        .into_iter()
        .filter(|row| seen.insert(row.key.clone()))
        .map(|row| DroppedPage { page_key: row.key, content_sha1: row.content_sha1, url: row.url })
        .collect())
}

/// The rows of a file whose `key_column` names the page; their vectors only when `with_vectors`, and page-v1 for a file without recipes.
fn read_rows(file: Bytes, key_column: &str, with_vectors: bool) -> Result<Vec<VectorRow>> {
    let mut columns = vec![key_column, "url", "content_sha1", "kind", "index", "recipe"];
    if with_vectors {
        columns.push("vector");
    }
    let mut rows = Vec::new();
    for batch in batches(file, &columns)? {
        let (keys, urls, hashes, kinds) = (strings(&batch, key_column)?, strings(&batch, "url")?, strings(&batch, "content_sha1")?, strings(&batch, "kind")?);
        let index = cast(column(&batch, "index")?, &DataType::Int32)?;
        let index = index.as_primitive::<Int32Type>();
        let vectors = with_vectors.then(|| binaries(&batch, "vector")).transpose()?;
        let recipes = batch.column_by_name("recipe").map(|_| strings(&batch, "recipe")).transpose()?;
        for row in 0..batch.num_rows() {
            rows.push(VectorRow {
                key: keys.value(row).to_string(),
                url: urls.value(row).to_string(),
                content_sha1: hashes.value(row).to_string(),
                kind: kinds.value(row).to_string(),
                index: index.value(row),
                vector: vectors.as_ref().map_or_else(Vec::new, |vectors| vectors.value(row).to_vec()),
                recipe: recipes.as_ref().map_or(RECIPE_V1, |recipes| recipes.value(row)).to_string(),
            });
        }
    }
    Ok(rows)
}

fn batches(file: Bytes, columns: &[&str]) -> Result<Vec<RecordBatch>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::columns(builder.parquet_schema(), columns.iter().copied());
    Ok(builder.with_projection(projection).build()?.collect::<Result<Vec<_>, _>>()?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{encode_input, Planned, Text, RECIPE_V2};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn unit(dims: usize, axis: usize) -> Vec<u8> {
        (0..dims).flat_map(|i| half::f16::from_f32(if i == axis { 1.0 } else { 0.0 }).to_le_bytes()).collect()
    }

    fn page(key: &str, texts: &[(&'static str, i32)]) -> Planned {
        Planned {
            key: key.into(),
            url: format!("https://example.com/{key}"),
            host: "example.com".into(),
            content_sha1: format!("sha-{key}"),
            recipe: RECIPE_V2,
            fetched_at: String::new(),
            texts: texts.iter().map(|&(kind, index)| Text { text_id: format!("{key}#{kind}{index}"), kind, index, text: format!("text {index}") }).collect(),
            page_only: false,
        }
    }

    fn input(pages: &[Planned]) -> Bytes {
        let rows: Vec<(&Planned, &Text)> = pages.iter().flat_map(|page| page.texts.iter().map(move |text| (page, text))).collect();
        Bytes::from(encode_input(&rows).unwrap())
    }

    #[test]
    fn a_raw_upload_holds_one_vector_per_text_in_the_inputs_order() {
        let pages = [page("p2", &[("page", 0)]), page("p1", &[("passage", 1), ("page", 0)])];
        let upload = [unit(8, 1), unit(8, 2), unit(8, 0)].concat();
        let joined = join(input(&pages), &upload).unwrap();
        let placed: Vec<(&str, &str, i32, Vec<u8>)> = joined.rows.iter().map(|r| (r.key.as_str(), r.kind.as_str(), r.index, r.vector.clone())).collect();
        assert_eq!(placed, [("p1", "page", 0, unit(8, 0)), ("p1", "passage", 1, unit(8, 2)), ("p2", "page", 0, unit(8, 1))]);
        assert_eq!((joined.malformed, joined.dropped.len()), (0, 0));
        assert_eq!((joined.rows[0].url.as_str(), joined.rows[0].content_sha1.as_str()), ("https://example.com/p1", "sha-p1"));

        let longer = [upload.as_slice(), &[0u8, 0][..]].concat();
        for wrong in [&upload[..upload.len() - 2], longer.as_slice(), &upload[..upload.len() - 1], &[][..]] {
            assert!(join(input(&pages), wrong).is_err(), "{} bytes for 3 texts", wrong.len());
        }
    }

    #[test]
    fn a_page_with_a_malformed_vector_is_left_out_whole() {
        let pages = [page("p1", &[("page", 0), ("passage", 1)]), page("p2", &[("page", 0)]), page("p3", &[("page", 0)])];
        let mut not_a_number = unit(8, 0);
        not_a_number[2..4].copy_from_slice(&half::f16::NAN.to_le_bytes());
        let long: Vec<u8> = (0..8).flat_map(|i| half::f16::from_f32(if i < 2 { 0.75 } else { 0.0 }).to_le_bytes()).collect();
        let upload = [unit(8, 0), not_a_number, unit(8, 1), long].concat();
        let joined = join(input(&pages), &upload).unwrap();
        assert_eq!(joined.malformed, 2);
        let kept: Vec<&str> = joined.rows.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(kept, ["p2"], "p1's good page vector goes with its bad passage");
        let dropped: Vec<&str> = joined.dropped.iter().map(|d| d.page_key.as_str()).collect();
        assert_eq!(dropped, ["p1", "p3"]);
        assert_eq!(joined.dropped[0], DroppedPage { page_key: "p1".into(), content_sha1: "sha-p1".into(), url: "https://example.com/p1".into() });
    }

    /// A file as written before it had a recipe column.
    fn without_recipes(file: Bytes) -> Bytes {
        let full = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap().next().unwrap().unwrap();
        let kept: Vec<usize> = (0..full.num_columns()).filter(|&i| full.schema().field(i).name() != "recipe").collect();
        assert_eq!(kept.len() + 1, full.num_columns());
        let projected = full.project(&kept).unwrap();
        let mut out = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut out, projected.schema(), None).unwrap();
        writer.write(&projected).unwrap();
        writer.close().unwrap();
        Bytes::from(out)
    }

    #[test]
    fn vectors_carry_their_input_s_recipe_and_files_without_one_are_page_v1() {
        let pages = [page("p1", &[("page", 0), ("passage", 1)])];
        let upload = [unit(8, 0), unit(8, 1)].concat();
        let joined = join(input(&pages), &upload).unwrap();
        assert!(joined.rows.iter().all(|row| row.recipe == RECIPE_V2));
        let file = Bytes::from(encode(&joined.rows, "t1", "5A", "2026-10-23T12:00:00+00:00").unwrap());
        assert_eq!(decode(file.clone()).unwrap(), joined.rows);
        assert!(decode(without_recipes(file)).unwrap().iter().all(|row| row.recipe == RECIPE_V1));
        let old_input = join(without_recipes(input(&pages)), &upload).unwrap();
        assert!(old_input.rows.iter().all(|row| row.recipe == RECIPE_V1), "an input written before recipes holds page-v1 texts");
    }

    #[test]
    fn vectors_are_unit_length_and_finite() {
        assert!(well_formed(&unit(2560, 7)));
        let tilted: Vec<u8> = [0.6f32, 0.8].iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        assert!(well_formed(&tilted));
        let short: Vec<u8> = [0.6f32, 0.7].iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        assert!(!well_formed(&short), "length 0.92");
        let infinite: Vec<u8> = [half::f16::INFINITY, half::f16::ZERO].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(!well_formed(&infinite));
        assert!(!well_formed(&[0; 16]) && !well_formed(&[]) && !well_formed(&[0x3c]));
    }

    #[test]
    fn a_withdrawal_entry_names_each_task_taken_back() {
        let file = encode_withdrawn(&[("t1".into(), "5A".into()), ("t2".into(), "5B".into())], "2026-10-19T12:00:00+00:00").unwrap();
        let batch = &batches(Bytes::from(file), &["kind", "task_id", "miner", "key"]).unwrap()[0];
        let text = |name: &str| strings(batch, name).unwrap().iter().map(|v| v.unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!([text("kind"), text("task_id"), text("miner"), text("key")], [["task_withdrawn"; 2], ["t1", "t2"], ["5A", "5B"], [""; 2]]);
    }
}

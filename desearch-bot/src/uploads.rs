//! Upload-shaped rows in the numbered parquet files and manifest `publisher bulk` reads, as `bulk/rows.py` writes them.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow_array::builder::{BooleanBuilder, Int32Builder, LargeBinaryBuilder, LargeStringBuilder, ListBuilder, StringBuilder, TimestampMicrosecondBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use chrono::{DateTime, FixedOffset, Utc};
use indexmap::IndexMap;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::pystr;

pub const ROWS_PER_FILE: usize = 5_000;
pub const ROW_GROUP_ROWS: usize = 500;
pub const MANIFEST: &str = "manifest.json";
pub const EXTRACTION_V2: &str = "v2";

/// `desearch.extraction.Page`, as the extraction worker sends it.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Page {
    pub page_type: String,
    pub text_mode: String,
    pub title: String,
    pub description: String,
    pub lang: String,
    pub canonical: String,
    pub published: String,
    pub modified: String,
    pub author: String,
    pub site_name: String,
    pub image: String,
    pub section: String,
    pub tags: Vec<String>,
    pub breadcrumbs: Vec<String>,
    pub json_ld_types: Vec<String>,
    pub headings: Vec<String>,
    pub entity: String,
    pub ld: String,
    pub paywalled: Option<bool>,
    pub robots: Vec<String>,
    pub text: String,
}

/// A row as a miner's upload holds a fetched page, without the HTML.
pub struct Row {
    pub url: String,
    pub fetched_at: DateTime<Utc>,
    pub content_type: String,
    pub page: Page,
}

pub fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

fn list() -> DataType {
    DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true)))
}

/// `desearch.extraction.schema.PAGE_SCHEMA`.
pub fn page_schema() -> SchemaRef {
    let utc = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let s = |name: &str| Field::new(name, DataType::Utf8, false);
    Arc::new(Schema::new(vec![
        s("url"),
        s("final_url"),
        Field::new("status", DataType::Int32, false),
        Field::new("error", DataType::Utf8, true),
        Field::new("fetched_at", utc, false),
        Field::new("elapsed_ms", DataType::Int32, false),
        s("content_type"),
        Field::new("html_bytes", DataType::Int32, false),
        Field::new("html", DataType::LargeBinary, true),
        s("html_sha256"),
        s("page_type"),
        s("title"),
        s("description"),
        s("lang"),
        s("canonical"),
        s("published"),
        s("author"),
        Field::new("json_ld_types", list(), false),
        Field::new("headings", list(), false),
        Field::new("text", DataType::LargeUtf8, false),
        s("text_sha256"),
        s("extraction"),
        s("text_mode"),
        s("modified"),
        s("site_name"),
        s("image"),
        s("section"),
        Field::new("tags", list(), false),
        Field::new("breadcrumbs", list(), false),
        Field::new("entity", DataType::LargeUtf8, false),
        Field::new("ld", DataType::LargeUtf8, false),
        Field::new("paywalled", DataType::Boolean, true),
        Field::new("robots", list(), false),
    ]))
}

fn strings(rows: &[Row], get: impl Fn(&Row) -> &str) -> ArrayRef {
    let mut b = StringBuilder::new();
    rows.iter().for_each(|r| b.append_value(get(r)));
    Arc::new(b.finish())
}

fn large_strings(rows: &[Row], get: impl Fn(&Row) -> &str) -> ArrayRef {
    let mut b = LargeStringBuilder::new();
    rows.iter().for_each(|r| b.append_value(get(r)));
    Arc::new(b.finish())
}

fn lists(rows: &[Row], get: impl Fn(&Row) -> &Vec<String>) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for r in rows {
        get(r).iter().for_each(|v| b.values().append_value(v));
        b.append(true);
    }
    Arc::new(b.finish())
}

fn int32(rows: &[Row], value: i32) -> ArrayRef {
    let mut b = Int32Builder::new();
    rows.iter().for_each(|_| b.append_value(value));
    Arc::new(b.finish())
}

pub fn batch(schema: &SchemaRef, rows: &[Row]) -> Result<RecordBatch> {
    let mut fetched = TimestampMicrosecondBuilder::new().with_timezone("UTC");
    rows.iter().for_each(|r| fetched.append_value(r.fetched_at.timestamp_micros()));
    let mut error = StringBuilder::new();
    let mut html = LargeBinaryBuilder::new();
    let mut paywalled = BooleanBuilder::new();
    for r in rows {
        error.append_null();
        html.append_null();
        paywalled.append_option(r.page.paywalled);
    }
    let columns: Vec<ArrayRef> = vec![
        strings(rows, |r| &r.url),
        strings(rows, |r| &r.url),
        int32(rows, 200),
        Arc::new(error.finish()),
        Arc::new(fetched.finish()),
        int32(rows, 0),
        strings(rows, |r| &r.content_type),
        int32(rows, 0),
        Arc::new(html.finish()),
        strings(rows, |_| ""),
        strings(rows, |r| &r.page.page_type),
        strings(rows, |r| &r.page.title),
        strings(rows, |r| &r.page.description),
        strings(rows, |r| &r.page.lang),
        strings(rows, |r| &r.page.canonical),
        strings(rows, |r| &r.page.published),
        strings(rows, |r| &r.page.author),
        lists(rows, |r| &r.page.json_ld_types),
        lists(rows, |r| &r.page.headings),
        large_strings(rows, |r| &r.page.text),
        Arc::new({
            let mut b = StringBuilder::new();
            rows.iter().for_each(|r| b.append_value(sha256_hex(&r.page.text)));
            b.finish()
        }),
        strings(rows, |_| EXTRACTION_V2),
        strings(rows, |r| &r.page.text_mode),
        strings(rows, |r| &r.page.modified),
        strings(rows, |r| &r.page.site_name),
        strings(rows, |r| &r.page.image),
        strings(rows, |r| &r.page.section),
        lists(rows, |r| &r.page.tags),
        lists(rows, |r| &r.page.breadcrumbs),
        large_strings(rows, |r| &r.page.entity),
        large_strings(rows, |r| &r.page.ld),
        Arc::new(paywalled.finish()),
        lists(rows, |r| &r.page.robots),
    ];
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

/// Rows written to numbered parquet files in one folder, with a manifest naming the source, run and time.
pub struct Batches {
    out: PathBuf,
    source: String,
    run: String,
    as_of: DateTime<FixedOffset>,
    schema: SchemaRef,
    files: Vec<String>,
    rows: usize,
    pending: Vec<Row>,
    writer: Option<ArrowWriter<File>>,
    in_file: usize,
}

impl Batches {
    pub fn new(out: &Path, source: &str, run: &str, as_of: DateTime<FixedOffset>) -> Result<Self> {
        std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
        if out.join(MANIFEST).exists() {
            bail!("{} already holds a run; give each run its own folder", out.display());
        }
        Ok(Self {
            out: out.to_path_buf(),
            source: source.into(),
            run: run.into(),
            as_of,
            schema: page_schema(),
            files: Vec::new(),
            rows: 0,
            pending: Vec::new(),
            writer: None,
            in_file: 0,
        })
    }

    pub fn add(&mut self, row: Row) -> Result<()> {
        if self.writer.is_none() {
            let name = format!("{}-{:06}.parquet", self.source, self.files.len() + 1);
            let file = File::create(self.out.join(&name))?;
            let props = WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
                .build();
            self.writer = Some(ArrowWriter::try_new(file, self.schema.clone(), Some(props))?);
            self.files.push(name);
        }
        self.pending.push(row);
        self.in_file += 1;
        self.rows += 1;
        if self.pending.len() >= ROW_GROUP_ROWS {
            self.flush()?;
        }
        if self.in_file >= ROWS_PER_FILE {
            self.close_file()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let (false, Some(w)) = (self.pending.is_empty(), self.writer.as_mut()) {
            w.write(&batch(&self.schema, &self.pending)?)?;
            w.flush()?;
            self.pending.clear();
        }
        Ok(())
    }

    fn close_file(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        self.in_file = 0;
        Ok(())
    }

    /// Closes the last file and writes the manifest, which marks the run complete.
    pub fn close(mut self, counts: &IndexMap<String, u64>) -> Result<()> {
        self.close_file()?;
        let manifest = manifest_json(&self.source, &self.run, &self.as_of, &self.files, self.rows, counts);
        std::fs::write(self.out.join(MANIFEST), manifest)?;
        Ok(())
    }
}

/// The manifest as Python's `json.dumps(manifest, indent=1)` writes it.
pub fn manifest_json(source: &str, run: &str, as_of: &DateTime<FixedOffset>, files: &[String], rows: usize, counts: &IndexMap<String, u64>) -> String {
    let value = serde_json::json!({
        "source": source,
        "run": run,
        "as_of": as_of.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "files": files,
        "removed": null,
        "rows": rows,
        "counts": counts,
    });
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, serde_json::ser::PrettyFormatter::with_indent(b" "));
    value.serialize(&mut ser).expect("a manifest serialises");
    pystr::ascii_json(&String::from_utf8(out).expect("JSON is UTF-8"))
}

/// Python's `json.dumps(counts)` with its default separators.
pub fn counts_json(counts: &IndexMap<String, u64>) -> String {
    let body: Vec<String> = counts.iter().map(|(k, v)| format!("{}: {v}", serde_json::to_string(k).expect("a key serialises"))).collect();
    pystr::ascii_json(&format!("{{{}}}", body.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_read_as_pythons() {
        let mut counts = IndexMap::new();
        counts.insert("page_site".to_string(), 3);
        counts.insert("wall".to_string(), 1);
        let at = DateTime::parse_from_rfc3339("2026-11-11T00:00:00Z").unwrap();
        let got = manifest_json("crawl", "rs-é", &at, &["crawl-000001.parquet".into()], 3, &counts);
        let want = "{\n \"source\": \"crawl\",\n \"run\": \"rs-\\u00e9\",\n \"as_of\": \"2026-11-11T00:00:00Z\",\n \"files\": [\n  \"crawl-000001.parquet\"\n ],\n \"removed\": null,\n \"rows\": 3,\n \"counts\": {\n  \"page_site\": 3,\n  \"wall\": 1\n }\n}";
        assert_eq!(got, want);
        assert_eq!(counts_json(&counts), r#"{"page_site": 3, "wall": 1}"#);
        assert!(manifest_json("crawl", "r", &at, &[], 0, &IndexMap::new()).contains("\"files\": [],\n"));
    }
}

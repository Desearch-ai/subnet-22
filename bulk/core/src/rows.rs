//! Upload-shaped rows and the folder of batch files `publisher bulk` publishes.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, BooleanArray, Int32Array, LargeBinaryArray, LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::counts::Counts;
use crate::isotime::fromisoformat;
use crate::pyjson;
use crate::text::{clean, finish, text_headings, Line, MAX_TEXT_CHARS};

pub const EXTRACTION_V2: &str = "v2";
pub const MAIN: &str = "main";
pub const PAPER: &str = "paper";
pub const ROWS_PER_FILE: usize = 5_000;
pub const ROW_GROUP_ROWS: usize = 500;
pub const MANIFEST: &str = "manifest.json";
pub const REMOVED: &str = "removed.txt";
pub const MAX_TAGS: usize = 20;

/// `desearch.extraction.Page`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

/// The keyword arguments of `typed_page` and `paper_page` after the title and lines.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaperMeta {
    pub lang: String,
    pub url: String,
    pub published: String,
    pub modified: String,
    pub author: String,
    pub site_name: String,
    pub tags: Vec<String>,
}

/// One PAGE_SCHEMA row: `upload_row`'s fixed columns are written by `Batches`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub url: String,
    pub fetched_at_us: i64,
    pub content_type: String,
    pub page: Page,
}

/// A built page as one upload row.
pub fn upload_row(url: &str, page: Page, fetched_at_us: i64, content_type: &str) -> Row {
    Row { url: url.to_string(), fetched_at_us, content_type: content_type.to_string(), page }
}

/// A paper page: the title line, then cleaned lines with `##` headings.
pub fn paper_page(title: &str, lines: Vec<Line>, meta: PaperMeta) -> Page {
    typed_page(PAPER, title, lines, meta)
}

/// A page of `page_type` in v2's text conventions, the title line then cleaned lines.
pub fn typed_page(page_type: &str, title: &str, lines: Vec<Line>, meta: PaperMeta) -> Page {
    let text = finish(title, &clean(lines, false), MAX_TEXT_CHARS);
    let mut tags = meta.tags;
    tags.truncate(MAX_TAGS);
    Page {
        page_type: page_type.to_string(),
        text_mode: MAIN.to_string(),
        title: title.to_string(),
        lang: meta.lang,
        canonical: meta.url,
        published: meta.published,
        modified: meta.modified,
        author: meta.author,
        site_name: meta.site_name,
        tags,
        headings: text_headings(&text),
        text,
        ..Page::default()
    }
}

/// The first `most` author names, then "et al."; `most` is 10 in every source.
pub fn authors_line(names: &[String], most: usize) -> String {
    let names: Vec<&str> = names.iter().map(String::as_str).filter(|n| !n.is_empty()).collect();
    if names.len() > most {
        return format!("{}, et al.", names[..most].join(", "));
    }
    names.join(", ")
}

/// Microseconds of 0001-01-01 and 10000-01-01, the ends of Python's datetime range.
const MIN_US: i64 = -62_135_596_800_000_000;
const END_US: i64 = 253_402_300_800_000_000;

/// `--as-of`, an ISO time with a zone, as microseconds since the epoch in UTC.
pub fn parse_as_of(text: &str) -> Result<i64> {
    let replaced = text.replace('Z', "+00:00");
    let Some(moment) = fromisoformat(&replaced) else {
        bail!("Invalid isoformat string: {}", py_repr(&replaced));
    };
    let Some(offset) = moment.offset_us else {
        bail!("--as-of {} needs a time zone, as in 2026-11-01T00:00:00Z", py_repr(text));
    };
    let utc = moment.wall_us - offset;
    if offset != 0 && !(MIN_US..END_US).contains(&utc) {
        bail!("date value out of range: {}", py_repr(text));
    }
    Ok(utc)
}

fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// `strftime` of a UTC time in microseconds, for `%Y`, `%m`, `%d`, `%H`, `%M` and `%S` only.
pub fn utc_format(us: i64, format: &str) -> String {
    let [year, month, day, hour, minute, second] = desearch::time::utc(us.div_euclid(1_000_000));
    format
        .replace("%Y", &format!("{year:04}"))
        .replace("%m", &format!("{month:02}"))
        .replace("%d", &format!("{day:02}"))
        .replace("%H", &format!("{hour:02}"))
        .replace("%M", &format!("{minute:02}"))
        .replace("%S", &format!("{second:02}"))
}

/// `desearch.extraction.schema.sha256_hex` of text.
pub fn sha256_hex(data: &str) -> String {
    let digest = Sha256::digest(data.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// pyarrow writes Parquet-compliant lists, so its files read back with the child named `element`.
/// Writes `bytes` to `path` and syncs it to disk.
pub fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn string_item() -> Field {
    Field::new("element", DataType::Utf8, true)
}

fn strings() -> DataType {
    DataType::List(Arc::new(string_item()))
}

/// `desearch.extraction.schema.PAGE_SCHEMA`.
pub fn page_schema() -> SchemaRef {
    let text = |name: &str| Field::new(name, DataType::Utf8, false);
    Arc::new(Schema::new(vec![
        text("url"),
        text("final_url"),
        Field::new("status", DataType::Int32, false),
        Field::new("error", DataType::Utf8, true),
        Field::new("fetched_at", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
        Field::new("elapsed_ms", DataType::Int32, false),
        text("content_type"),
        Field::new("html_bytes", DataType::Int32, false),
        Field::new("html", DataType::LargeBinary, true),
        text("html_sha256"),
        text("page_type"),
        text("title"),
        text("description"),
        text("lang"),
        text("canonical"),
        text("published"),
        text("author"),
        Field::new("json_ld_types", strings(), false),
        Field::new("headings", strings(), false),
        Field::new("text", DataType::LargeUtf8, false),
        text("text_sha256"),
        text("extraction"),
        text("text_mode"),
        text("modified"),
        text("site_name"),
        text("image"),
        text("section"),
        Field::new("tags", strings(), false),
        Field::new("breadcrumbs", strings(), false),
        Field::new("entity", DataType::LargeUtf8, false),
        Field::new("ld", DataType::LargeUtf8, false),
        Field::new("paywalled", DataType::Boolean, true),
        Field::new("robots", strings(), false),
    ]))
}

fn string_column<'a>(rows: &'a [Row], value: impl Fn(&'a Row) -> &'a str) -> ArrayRef {
    Arc::new(rows.iter().map(|row| Some(value(row))).collect::<StringArray>())
}

fn large_string_column<'a>(rows: &'a [Row], value: impl Fn(&'a Row) -> &'a str) -> ArrayRef {
    Arc::new(rows.iter().map(|row| Some(value(row))).collect::<LargeStringArray>())
}

fn list_column<'a>(rows: &'a [Row], value: impl Fn(&'a Row) -> &'a [String]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new()).with_field(string_item());
    for row in rows {
        for item in value(row) {
            builder.values().append_value(item);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn int_column(rows: &[Row], value: i32) -> ArrayRef {
    Arc::new(Int32Array::from(vec![value; rows.len()]))
}

/// The rows as one PAGE_SCHEMA batch, with `upload_row`'s fixed columns.
pub fn record_batch(rows: &[Row]) -> Result<RecordBatch> {
    let columns: Vec<ArrayRef> = vec![
        string_column(rows, |r| &r.url),
        string_column(rows, |r| &r.url),
        int_column(rows, 200),
        Arc::new(StringArray::new_null(rows.len())),
        Arc::new(TimestampMicrosecondArray::from(rows.iter().map(|r| r.fetched_at_us).collect::<Vec<_>>()).with_timezone("UTC")),
        int_column(rows, 0),
        string_column(rows, |r| &r.content_type),
        int_column(rows, 0),
        Arc::new(LargeBinaryArray::new_null(rows.len())),
        string_column(rows, |_| ""),
        string_column(rows, |r| &r.page.page_type),
        string_column(rows, |r| &r.page.title),
        string_column(rows, |r| &r.page.description),
        string_column(rows, |r| &r.page.lang),
        string_column(rows, |r| &r.page.canonical),
        string_column(rows, |r| &r.page.published),
        string_column(rows, |r| &r.page.author),
        list_column(rows, |r| &r.page.json_ld_types),
        list_column(rows, |r| &r.page.headings),
        large_string_column(rows, |r| &r.page.text),
        Arc::new(rows.iter().map(|r| Some(sha256_hex(&r.page.text))).collect::<StringArray>()),
        string_column(rows, |_| EXTRACTION_V2),
        string_column(rows, |r| &r.page.text_mode),
        string_column(rows, |r| &r.page.modified),
        string_column(rows, |r| &r.page.site_name),
        string_column(rows, |r| &r.page.image),
        string_column(rows, |r| &r.page.section),
        list_column(rows, |r| &r.page.tags),
        list_column(rows, |r| &r.page.breadcrumbs),
        large_string_column(rows, |r| &r.page.entity),
        large_string_column(rows, |r| &r.page.ld),
        Arc::new(rows.iter().map(|r| r.page.paywalled).collect::<BooleanArray>()),
        list_column(rows, |r| &r.page.robots),
    ];
    Ok(RecordBatch::try_new(page_schema(), columns)?)
}

/// What `Batches::close` reports for the run's summary line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub out: PathBuf,
    pub files: usize,
    pub rows: u64,
    pub removed: usize,
}

/// Rows written to numbered parquet files in one folder, with a manifest naming the source, run and time.
pub struct Batches {
    out: PathBuf,
    source: String,
    run: String,
    as_of_us: i64,
    rows_per_file: usize,
    files: Vec<String>,
    removed: Vec<String>,
    rows: u64,
    pending: Vec<Row>,
    writer: Option<ArrowWriter<File>>,
    in_file: usize,
}

impl Batches {
    pub fn new(out: &Path, source: &str, run: &str, as_of_us: i64) -> Result<Self> {
        Self::with_rows_per_file(out, source, run, as_of_us, ROWS_PER_FILE)
    }

    pub fn with_rows_per_file(out: &Path, source: &str, run: &str, as_of_us: i64, rows_per_file: usize) -> Result<Self> {
        std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
        if out.join(MANIFEST).exists() {
            bail!("{} already holds a run; give each run its own folder", out.display());
        }
        Ok(Batches {
            out: out.to_path_buf(),
            source: source.to_string(),
            run: run.to_string(),
            as_of_us,
            rows_per_file,
            files: Vec::new(),
            removed: Vec::new(),
            rows: 0,
            pending: Vec::with_capacity(ROW_GROUP_ROWS),
            writer: None,
            in_file: 0,
        })
    }

    pub fn add(&mut self, row: Row) -> Result<()> {
        if self.writer.is_none() {
            let name = format!("{}-{:06}.parquet", self.source, self.files.len() + 1);
            let path = self.out.join(&name);
            let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
            let props = WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .set_max_row_group_row_count(Some(ROW_GROUP_ROWS.max(self.rows_per_file)))
                .build();
            self.writer = Some(ArrowWriter::try_new(file, page_schema(), Some(props))?);
            self.files.push(name);
        }
        self.pending.push(row);
        self.in_file += 1;
        self.rows += 1;
        if self.pending.len() >= ROW_GROUP_ROWS {
            self.flush()?;
        }
        if self.in_file >= self.rows_per_file {
            self.close_file()?;
        }
        Ok(())
    }

    /// A page the source deleted, taken down when the run is published.
    pub fn remove(&mut self, url: &str) {
        self.removed.push(url.to_string());
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn files(&self) -> &[String] {
        &self.files
    }

    pub fn removed(&self) -> &[String] {
        &self.removed
    }

    pub fn out(&self) -> &Path {
        &self.out
    }

    fn flush(&mut self) -> Result<()> {
        if let (false, Some(writer)) = (self.pending.is_empty(), self.writer.as_mut()) {
            writer.write(&record_batch(&self.pending)?)?;
            writer.flush()?;
            self.pending.clear();
        }
        Ok(())
    }

    fn close_file(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        self.in_file = 0;
        Ok(())
    }

    /// The manifest's JSON text, as `json.dumps(manifest, indent=1)` writes it.
    pub fn manifest(&self, counts: &Counts) -> String {
        let counts: Map<String, Value> = counts.iter().map(|(k, n)| (k.to_string(), json!(n))).collect();
        let manifest = json!({
            "source": self.source,
            "run": self.run,
            "as_of": utc_format(self.as_of_us, "%Y-%m-%dT%H:%M:%SZ"),
            "files": self.files,
            "removed": if self.removed.is_empty() { Value::Null } else { json!(REMOVED) },
            "rows": self.rows,
            "counts": counts,
        });
        pyjson::dumps(&manifest, Some(1))
    }

    /// Closes the last file and writes the manifest, which marks the run complete.
    pub fn close(mut self, counts: &Counts) -> Result<Summary> {
        self.close_file()?;
        for name in &self.files {
            std::fs::File::open(self.out.join(name))?.sync_all()?;
        }
        if !self.removed.is_empty() {
            let text: String = self.removed.iter().map(|url| format!("{url}\n")).collect();
            write_synced(&self.out.join(REMOVED), text.as_bytes())?;
        }
        // The manifest marks the run complete, so it lands whole and only after every file is on disk.
        let partial = self.out.join(format!("{MANIFEST}.tmp"));
        write_synced(&partial, self.manifest(counts).as_bytes())?;
        std::fs::rename(&partial, self.out.join(MANIFEST))?;
        std::fs::File::open(&self.out)?.sync_all()?;
        Ok(Summary { out: self.out.clone(), files: self.files.len(), rows: self.rows, removed: self.removed.len() })
    }
}

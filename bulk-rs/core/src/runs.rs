//! `bulk diff`: whether two run folders hold the same manifest and the same rows, column by column.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde_json::{json, Value};

use crate::pyjson;
use crate::rows::{MANIFEST, REMOVED};

#[derive(clap::Args, Clone, Debug)]
pub struct DiffArgs {
    pub a: PathBuf,
    pub b: PathBuf,
    /// Differing rows to describe.
    #[arg(long, default_value_t = 20)]
    pub show: usize,
    /// Match rows by URL instead of by place, for runs whose rows come in no fixed order; the run name, as-of time and
    /// file names are then not compared.
    #[arg(long = "by-url")]
    pub by_url: bool,
    /// Columns left out of the comparison, such as fetch times, with --by-url.
    #[arg(long, value_delimiter = ',')]
    pub ignore: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diff {
    pub rows_a: u64,
    pub rows_b: u64,
    pub rows_differing: u64,
    pub manifest_differences: Vec<String>,
    pub examples: Vec<String>,
}

impl Diff {
    pub fn same(&self) -> bool {
        self.rows_differing == 0 && self.manifest_differences.is_empty() && self.rows_a == self.rows_b
    }
}

const MANIFEST_FIELDS: [&str; 7] = ["source", "run", "as_of", "files", "removed", "rows", "counts"];
const SHOWN_CHARS: usize = 80;

/// One cell's value, borrowed from its column.
#[derive(Debug, PartialEq)]
enum Cell<'a> {
    Null,
    Text(&'a str),
    Bytes(&'a [u8]),
    Int(i64),
    Bool(bool),
    Other(String),
}

fn cell(array: &dyn Array, i: usize) -> Cell<'_> {
    if array.is_null(i) {
        return Cell::Null;
    }
    match array.data_type() {
        DataType::Utf8 => Cell::Text(array.as_string::<i32>().value(i)),
        DataType::LargeUtf8 => Cell::Text(array.as_string::<i64>().value(i)),
        DataType::Binary => Cell::Bytes(array.as_binary::<i32>().value(i)),
        DataType::LargeBinary => Cell::Bytes(array.as_binary::<i64>().value(i)),
        DataType::Int32 => Cell::Int(i64::from(array.as_primitive::<Int32Type>().value(i))),
        DataType::Int64 => Cell::Int(array.as_primitive::<Int64Type>().value(i)),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => Cell::Int(array.as_primitive::<TimestampMicrosecondType>().value(i)),
        DataType::Boolean => Cell::Bool(array.as_boolean().value(i)),
        DataType::List(_) => list(array.as_list::<i32>().value(i)),
        DataType::LargeList(_) => list(array.as_list::<i64>().value(i)),
        other => Cell::Other(format!("{other} value")),
    }
}

fn list(values: ArrayRef) -> Cell<'static> {
    let strings = |i: usize| -> Option<String> { (!values.is_null(i)).then(|| text_of(values.as_ref(), i)) };
    Cell::Other(pyjson::dumps(&Value::Array((0..values.len()).map(|i| strings(i).map_or(Value::Null, Value::String)).collect()), None))
}

fn text_of(array: &dyn Array, i: usize) -> String {
    match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(i).to_string(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(i).to_string(),
        other => format!("{other} value"),
    }
}

fn shown(value: &Cell, other: &Cell) -> String {
    match (value, other) {
        (Cell::Text(a), Cell::Text(b)) => {
            let at = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            let start = at.saturating_sub(SHOWN_CHARS / 4);
            let window: String = a.chars().skip(start).take(SHOWN_CHARS).collect();
            let lead = if start > 0 { format!("[{start}]...") } else { String::new() };
            let tail = if a.chars().count() > start + SHOWN_CHARS { "..." } else { "" };
            pyjson::quote(&format!("{lead}{window}{tail}"))
        }
        (Cell::Text(a), _) => pyjson::quote(desearch::text::head(a, SHOWN_CHARS)),
        (Cell::Bytes(a), _) => format!("<{} bytes>", a.len()),
        (Cell::Int(n), _) => n.to_string(),
        (Cell::Bool(b), _) => b.to_string(),
        (Cell::Null, _) => "null".into(),
        (Cell::Other(text), _) => desearch::text::head(text, SHOWN_CHARS * 2).to_string(),
    }
}

struct Run {
    root: PathBuf,
    manifest: Value,
    text: String,
}

fn open_run(root: &Path) -> Result<Run> {
    let path = root.join(MANIFEST);
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let manifest = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Run { root: root.to_path_buf(), manifest, text })
}

fn files(run: &Run) -> Vec<String> {
    run.manifest["files"].as_array().map(|files| files.iter().filter_map(|f| f.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn removed_lines(run: &Run) -> Result<Option<Vec<String>>> {
    let path = run.root.join(REMOVED);
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(std::fs::read_to_string(&path)?.lines().map(String::from).collect()))
}

/// Rows of one file in order, read a batch at a time.
struct Rows {
    reader: Option<ParquetRecordBatchReader>,
    schema: Option<SchemaRef>,
    batch: Option<RecordBatch>,
    at: usize,
    rows: u64,
}

impl Rows {
    fn empty() -> Rows {
        Rows { reader: None, schema: None, batch: None, at: 0, rows: 0 }
    }

    fn open(path: &Path) -> Result<Rows> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path).with_context(|| format!("opening {}", path.display()))?)?;
        let rows = builder.metadata().file_metadata().num_rows() as u64;
        let schema = builder.schema().clone();
        Ok(Rows { reader: Some(builder.with_batch_size(1024).build()?), schema: Some(schema), batch: None, at: 0, rows })
    }

    /// The batch holding the next row and its index there, or None at the end.
    fn next(&mut self) -> Result<Option<(&RecordBatch, usize)>> {
        while self.batch.as_ref().is_none_or(|batch| self.at >= batch.num_rows()) {
            let Some(reader) = self.reader.as_mut() else { return Ok(None) };
            match reader.next() {
                Some(batch) => {
                    self.batch = Some(batch?);
                    self.at = 0;
                }
                None => {
                    self.reader = None;
                    return Ok(None);
                }
            }
        }
        self.at += 1;
        Ok(self.batch.as_ref().map(|batch| (batch, self.at - 1)))
    }
}

/// A listed file's rows, or none with a difference noted when the file is not there.
fn file_rows(root: &Path, name: &str, side: &str, differences: &mut Vec<String>) -> Result<Rows> {
    if name.is_empty() {
        return Ok(Rows::empty());
    }
    let path = root.join(name);
    if !path.exists() {
        differences.push(format!("{name}: listed but missing in {side}"));
        return Ok(Rows::empty());
    }
    Rows::open(&path)
}

fn schema_differences(name: &str, a: &SchemaRef, b: &SchemaRef, out: &mut Vec<String>) {
    for field in a.fields() {
        match b.field_with_name(field.name()) {
            Err(_) => out.push(format!("{name}: column {} only in a", field.name())),
            Ok(other) if other.data_type() != field.data_type() || other.is_nullable() != field.is_nullable() => out.push(format!(
                "{name}: column {} is {}{} in a and {}{} in b",
                field.name(),
                field.data_type(),
                if field.is_nullable() { "" } else { " not null" },
                other.data_type(),
                if other.is_nullable() { "" } else { " not null" },
            )),
            Ok(_) => {}
        }
    }
    for field in b.fields() {
        if a.field_with_name(field.name()).is_err() {
            out.push(format!("{name}: column {} only in b", field.name()));
        }
    }
    let names = |s: &SchemaRef| s.fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>();
    if a.fields().len() == b.fields().len() && names(a) != names(b) {
        out.push(format!("{name}: columns in another order"));
    }
}

pub fn diff(a: &Path, b: &Path, show: usize) -> Result<Diff> {
    let (run_a, run_b) = (open_run(a)?, open_run(b)?);
    let mut out = Diff::default();
    for field in MANIFEST_FIELDS {
        let (x, y) = (pyjson::dumps(&run_a.manifest[field], None), pyjson::dumps(&run_b.manifest[field], None));
        if x != y {
            out.manifest_differences.push(format!("{field}: {} != {}", desearch::text::head(&x, 200), desearch::text::head(&y, 200)));
        }
    }
    if out.manifest_differences.is_empty() && run_a.text != run_b.text {
        out.manifest_differences.push(format!("{MANIFEST} text differs"));
    }
    if removed_lines(&run_a)? != removed_lines(&run_b)? {
        out.manifest_differences.push(format!("{REMOVED} differs"));
    }
    let (files_a, files_b) = (files(&run_a), files(&run_b));
    for i in 0..files_a.len().max(files_b.len()) {
        let name_a = files_a.get(i).map(String::as_str).unwrap_or("");
        let name_b = files_b.get(i).map(String::as_str).unwrap_or("");
        let name = if name_a.is_empty() { name_b } else { name_a };
        let mut rows_a = file_rows(a, name_a, "a", &mut out.manifest_differences)?;
        let mut rows_b = file_rows(b, name_b, "b", &mut out.manifest_differences)?;
        out.rows_a += rows_a.rows;
        out.rows_b += rows_b.rows;
        if rows_a.rows != rows_b.rows {
            out.manifest_differences.push(format!("{name}: {} rows in a, {} in b", rows_a.rows, rows_b.rows));
        }
        if let (Some(x), Some(y)) = (&rows_a.schema, &rows_b.schema) {
            schema_differences(name, x, y, &mut out.manifest_differences);
        }
        let mut row = 0u64;
        loop {
            let (x, y) = (rows_a.next()?, rows_b.next()?);
            let differing = match (&x, &y) {
                (None, None) => break,
                (Some(_), None) => Some(("(row)".to_string(), "present".to_string(), "missing".to_string())),
                (None, Some(_)) => Some(("(row)".to_string(), "missing".to_string(), "present".to_string())),
                (Some((batch_a, i)), Some((batch_b, j))) => first_difference(batch_a, *i, batch_b, *j),
            };
            if let Some((column, value_a, value_b)) = differing {
                out.rows_differing += 1;
                if out.examples.len() < show {
                    out.examples.push(format!("{name} row {row} {column}: {value_a} != {value_b}"));
                }
            }
            row += 1;
        }
    }
    Ok(out)
}

/// The first column whose values differ, with both values shortened, and how many other columns differ.
fn first_difference(a: &RecordBatch, i: usize, b: &RecordBatch, j: usize) -> Option<(String, String, String)> {
    let mut found: Option<(String, String, String)> = None;
    let mut more = 0;
    for (index, field) in a.schema().fields().iter().enumerate() {
        let Some(other) = b.column_by_name(field.name()) else { continue };
        let (x, y) = (cell(a.column(index).as_ref(), i), cell(other.as_ref(), j));
        if x == y {
            continue;
        }
        if found.is_none() {
            found = Some((field.name().clone(), shown(&x, &y), shown(&y, &x)));
        } else {
            more += 1;
        }
    }
    found.map(|(column, x, y)| if more > 0 { (format!("{column} (+{more} columns)"), x, y) } else { (column, x, y) })
}

/// A cell's value, owned so rows from both runs can be held together.
#[derive(Clone, Debug, PartialEq)]
enum Owned {
    Null,
    Text(String),
    Bytes(Vec<u8>),
    Int(i64),
    Bool(bool),
    Other(String),
}

impl Owned {
    fn of(cell: &Cell) -> Owned {
        match cell {
            Cell::Null => Owned::Null,
            Cell::Text(s) => Owned::Text(s.to_string()),
            Cell::Bytes(b) => Owned::Bytes(b.to_vec()),
            Cell::Int(n) => Owned::Int(*n),
            Cell::Bool(b) => Owned::Bool(*b),
            Cell::Other(s) => Owned::Other(s.clone()),
        }
    }

    fn cell(&self) -> Cell<'_> {
        match self {
            Owned::Null => Cell::Null,
            Owned::Text(s) => Cell::Text(s),
            Owned::Bytes(b) => Cell::Bytes(b),
            Owned::Int(n) => Cell::Int(*n),
            Owned::Bool(b) => Cell::Bool(*b),
            Owned::Other(s) => Cell::Other(s.clone()),
        }
    }
}

type ByUrl = BTreeMap<String, Vec<(String, Owned)>>;

/// Every row of a run by URL, without the ignored columns; a URL listed twice is a difference.
fn rows_by_url(run: &Run, ignore: &[String], side: &str, differences: &mut Vec<String>) -> Result<(ByUrl, u64)> {
    let mut found = ByUrl::new();
    let mut count = 0;
    for name in files(run) {
        let mut rows = file_rows(&run.root, &name, side, differences)?;
        while let Some((batch, i)) = rows.next()? {
            count += 1;
            let mut url = None;
            let mut values = Vec::new();
            for (index, field) in batch.schema().fields().iter().enumerate() {
                let value = Owned::of(&cell(batch.column(index).as_ref(), i));
                if field.name() == "url" {
                    url = Some(match &value {
                        Owned::Text(s) => s.clone(),
                        other => format!("{other:?}"),
                    });
                }
                if !ignore.iter().any(|c| c == field.name()) {
                    values.push((field.name().clone(), value));
                }
            }
            let url = url.with_context(|| format!("{name} has no url column"))?;
            if found.insert(url.clone(), values).is_some() {
                differences.push(format!("{url} is listed twice in {side}"));
            }
        }
    }
    Ok((found, count))
}

/// `diff` with rows matched by URL and the `ignore` columns left out.
pub fn diff_by_url(a: &Path, b: &Path, ignore: &[String], show: usize) -> Result<Diff> {
    let (run_a, run_b) = (open_run(a)?, open_run(b)?);
    let mut out = Diff::default();
    for field in ["source", "removed", "rows", "counts"] {
        let (x, y) = (pyjson::dumps(&run_a.manifest[field], None), pyjson::dumps(&run_b.manifest[field], None));
        if x != y {
            out.manifest_differences.push(format!("{field}: {} != {}", desearch::text::head(&x, 200), desearch::text::head(&y, 200)));
        }
    }
    let sorted = |run: &Run| -> Result<Option<Vec<String>>> {
        Ok(removed_lines(run)?.map(|mut lines| {
            lines.sort();
            lines
        }))
    };
    if sorted(&run_a)? != sorted(&run_b)? {
        out.manifest_differences.push(format!("{REMOVED} differs"));
    }
    let (rows_a, count_a) = rows_by_url(&run_a, ignore, "a", &mut out.manifest_differences)?;
    let (rows_b, count_b) = rows_by_url(&run_b, ignore, "b", &mut out.manifest_differences)?;
    out.rows_a = count_a;
    out.rows_b = count_b;
    let note = |out: &mut Diff, text: String| {
        out.rows_differing += 1;
        if out.examples.len() < show {
            out.examples.push(text);
        }
    };
    for (url, values) in &rows_a {
        let Some(other) = rows_b.get(url) else {
            note(&mut out, format!("{url}: only in a"));
            continue;
        };
        let differing: Vec<&(String, Owned)> =
            values.iter().filter(|(column, value)| other.iter().find(|(c, _)| c == column).map(|(_, v)| v) != Some(value)).collect();
        if let Some((column, value)) = differing.first() {
            let theirs = other.iter().find(|(c, _)| c == column).map_or(Owned::Null, |(_, v)| v.clone());
            let more = if differing.len() > 1 { format!(" (+{} columns)", differing.len() - 1) } else { String::new() };
            note(&mut out, format!("{url}: {column}{more}: {} != {}", shown(&value.cell(), &theirs.cell()), shown(&theirs.cell(), &value.cell())));
        }
    }
    for url in rows_b.keys().filter(|url| !rows_a.contains_key(*url)) {
        note(&mut out, format!("{url}: only in b"));
    }
    Ok(out)
}

/// Prints the diff as one JSON line; an error when the runs differ.
pub fn main(args: DiffArgs) -> Result<()> {
    let found = if args.by_url { diff_by_url(&args.a, &args.b, &args.ignore, args.show)? } else { diff(&args.a, &args.b, args.show)? };
    let line = json!({
        "a": args.a.to_string_lossy(),
        "b": args.b.to_string_lossy(),
        "same": found.same(),
        "rows_a": found.rows_a,
        "rows_b": found.rows_b,
        "rows_differing": found.rows_differing,
        "manifest_differences": found.manifest_differences,
        "examples": found.examples,
    });
    println!("{}", pyjson::dumps(&line, None));
    if !found.same() {
        bail!("the runs differ");
    }
    Ok(())
}

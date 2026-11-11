//! Upload-shaped rows and the folder of batch files `publisher bulk` publishes, as `bulk/rows.py` writes them.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::counts::Counts;
use crate::text::Line;

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

/// The keyword arguments of `paper_page` after the title and lines.
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

/// `bulk.rows.upload_row`.
pub fn upload_row(url: &str, page: Page, fetched_at_us: i64, content_type: &str) -> Row {
    Row { url: url.to_string(), fetched_at_us, content_type: content_type.to_string(), page }
}

/// `bulk.rows.paper_page`: the title line, then cleaned lines with `##` headings.
pub fn paper_page(title: &str, lines: Vec<Line>, meta: PaperMeta) -> Page {
    todo!("{title} {} {meta:?}", lines.len())
}

/// `bulk.rows.authors_line`, `most` being 10 in every source.
pub fn authors_line(names: &[String], most: usize) -> String {
    todo!("{} {most}", names.len())
}

/// `bulk.rows.parse_as_of`, as microseconds since the epoch in UTC.
pub fn parse_as_of(text: &str) -> Result<i64> {
    todo!("{text}")
}

/// What `Batches::close` reports for the run's summary line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub out: PathBuf,
    pub files: usize,
    pub rows: u64,
    pub removed: usize,
}

/// `bulk.rows.Batches`: rows written to numbered parquet files in one folder, with a manifest naming the source, run and time.
pub struct Batches {
    _private: (),
}

impl Batches {
    pub fn new(out: &Path, source: &str, run: &str, as_of_us: i64) -> Result<Self> {
        todo!("{out:?} {source} {run} {as_of_us}")
    }

    pub fn with_rows_per_file(out: &Path, source: &str, run: &str, as_of_us: i64, rows_per_file: usize) -> Result<Self> {
        todo!("{out:?} {source} {run} {as_of_us} {rows_per_file}")
    }

    pub fn add(&mut self, row: Row) -> Result<()> {
        todo!("{}", row.url)
    }

    /// A page the source deleted, taken down when the run is published.
    pub fn remove(&mut self, url: &str) {
        todo!("{url}")
    }

    pub fn rows(&self) -> u64 {
        todo!()
    }

    /// Closes the last file and writes the manifest, which marks the run complete.
    pub fn close(self, counts: &Counts) -> Result<Summary> {
        todo!("{counts:?}")
    }
}

//! arXiv full text: Common Pile's `arxiv_papers` joined to our OAI records, for papers under CC BY, CC BY-SA or CC0.

mod body;
mod read;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use bulk_arxiv::Paper;
use bulk_core::{upload_row, Counts, Line, Page, RunArgs};
use rayon::prelude::*;
use serde_json::Value;

pub use body::{AFFILIATION_CHARS, REFERENCES};

/// Shorter full texts are failed conversions; the abstract page covers the paper.
pub const MIN_TEXT_CHARS: usize = 2_000;
pub const CONTENT_TYPE: &str = "text/html";
const BATCH_LINES: usize = 512;
const BATCH_BYTES: usize = 256 << 20;

pub fn body_lines(markdown: &str) -> Vec<String> {
    body::body_lines(markdown)
}

pub fn full_text_page(paper: &Paper, body: Vec<String>) -> (String, Page) {
    let mut lines = bulk_arxiv::heading_lines(paper);
    lines.extend(body.into_iter().map(Line::new));
    bulk_arxiv::framed(paper, &bulk_arxiv::html_url(&paper.arxiv_id), lines)
}

enum Outcome {
    NotOpen,
    Short,
    Page(String, Box<Page>),
}

/// One Common Pile line as `run_text` judges it, its errors where Python's would raise.
fn outcome(line: &[u8], papers: &HashMap<String, Paper>) -> Result<Outcome> {
    let line = std::str::from_utf8(line).context("a Common Pile line is not UTF-8")?;
    let item: Value = serde_json::from_str(&bulk_core::text::drop_lone_surrogates(line))?;
    let item = item.as_object().ok_or_else(|| anyhow!("a Common Pile line is not an object"))?;
    let paper = match item.get("id").ok_or_else(|| anyhow!("a Common Pile line has no id"))? {
        Value::String(id) => papers.get(id),
        Value::Array(_) | Value::Object(_) => return Err(anyhow!("a Common Pile id is not hashable")),
        _ => None,
    };
    let Some(paper) = paper else {
        return Ok(Outcome::NotOpen);
    };
    let text = match item.get("text") {
        Some(Value::String(text)) => text,
        Some(_) => return Err(anyhow!("the Common Pile text of {} is not a string", paper.arxiv_id)),
        None => return Err(anyhow!("the Common Pile line of {} has no text", paper.arxiv_id)),
    };
    let body = body_lines(text);
    if body.iter().map(|t| t.chars().count()).sum::<usize>() < MIN_TEXT_CHARS {
        return Ok(Outcome::Short);
    }
    let (url, page) = full_text_page(paper, body);
    Ok(Outcome::Page(url, Box::new(page)))
}

pub fn run_text(files: &[PathBuf], papers: &HashMap<String, Paper>, batches: &mut bulk_core::Batches, as_of_us: i64, limit: u64) -> Result<Counts> {
    let mut counts = Counts::default();
    for path in files {
        let mut lines = read::GzLines::open(path).with_context(|| format!("opening {}", path.display()))?;
        let limited = std::thread::scope(|scope| -> Result<bool> {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            scope.spawn(move || loop {
                let (batch, error) = lines.batch(BATCH_LINES, BATCH_BYTES);
                let last = batch.is_empty() || error.is_some();
                if send.send((batch, error)).is_err() || last {
                    break;
                }
            });
            for (batch, error) in receive {
                let outcomes: Vec<Result<Outcome>> = batch.par_iter().map(|line| outcome(line, papers)).collect();
                for found in outcomes {
                    match found.with_context(|| format!("reading {}", path.display()))? {
                        Outcome::NotOpen => counts.add("not_open", 1),
                        Outcome::Short => counts.add("short", 1),
                        Outcome::Page(url, page) => {
                            batches.add(upload_row(&url, *page, as_of_us, CONTENT_TYPE))?;
                            counts.add("pages", 1);
                            if limit > 0 && counts.get("pages") >= limit {
                                return Ok(true);
                            }
                        }
                    }
                }
                if let Some(error) = error {
                    return Err(error).with_context(|| format!("reading {}", path.display()));
                }
            }
            Ok(false)
        })?;
        if limited {
            break;
        }
    }
    Ok(counts)
}

/// `sorted(folder.glob("*.xml"))`: hidden names and folders included, in code point order; an error when the folder is missing or holds none.
pub fn oai_pages(folder: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(folder).with_context(|| format!("reading --oai {}", folder.display()))?;
    let mut names: Vec<_> =
        entries.filter_map(|entry| entry.ok()).map(|entry| entry.file_name()).filter(|name| name.as_encoded_bytes().ends_with(b".xml")).collect();
    if names.is_empty() {
        return Err(anyhow!("--oai {} holds no .xml pages", folder.display()));
    }
    names.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
    Ok(names.into_iter().map(|name| folder.join(name)).collect())
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// Common Pile's arXiv full texts (.jsonl.gz).
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
    /// A folder of OAI-PMH pages from `arxiv-harvest`, whose licences decide which papers are taken.
    #[arg(long, required = true)]
    pub oai: PathBuf,
}

/// `bulk arxiv-text`.
pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, "arxiv-text")?;
    let papers = bulk_arxiv::open_papers(&oai_pages(&args.oai)?)?;
    let counts = run_text(&args.files, &papers, &mut batches, as_of_us, args.run.limit)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

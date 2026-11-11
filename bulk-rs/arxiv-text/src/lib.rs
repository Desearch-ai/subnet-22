//! arXiv full text: Common Pile's `arxiv_papers` joined to our OAI records, for papers under CC BY, CC BY-SA or CC0.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use bulk_arxiv::Paper;
use bulk_core::{Counts, RunArgs};

/// `bulk.arxiv.body_lines`.
pub fn body_lines(markdown: &str) -> Vec<String> {
    todo!("{}", markdown.len())
}

/// `bulk.arxiv.run_text`.
pub fn run_text(files: &[PathBuf], papers: &HashMap<String, Paper>, batches: &mut bulk_core::Batches, as_of_us: i64, limit: u64) -> Result<Counts> {
    todo!("{} {} {as_of_us} {limit} {}", files.len(), papers.len(), batches.rows())
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

/// `python -m bulk arxiv-text`.
pub fn main(args: Args) -> Result<()> {
    todo!("{args:?}")
}

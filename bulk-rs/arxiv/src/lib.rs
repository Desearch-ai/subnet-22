//! arXiv: the OAI-PMH harvest and one abstract page a paper, as `bulk/arxiv.py` builds them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use bulk_core::{Counts, Line, Page, RunArgs};

pub const DELAY_S: f64 = 15.0;

/// `bulk.arxiv.Paper`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Paper {
    pub arxiv_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub categories: String,
    pub created: String,
    pub updated: String,
    pub comments: String,
    pub journal: String,
    pub doi: String,
    pub licence: String,
}

pub fn url_of(arxiv_id: &str) -> String {
    todo!("{arxiv_id}")
}

pub fn html_url(arxiv_id: &str) -> String {
    todo!("{arxiv_id}")
}

pub fn is_open(licence: &str) -> bool {
    todo!("{licence}")
}

/// `bulk.arxiv.framed`: the paper's page at `url` from its lines.
pub fn framed(paper: &Paper, url: &str, lines: Vec<Line>) -> (String, Page) {
    todo!("{} {url} {}", paper.arxiv_id, lines.len())
}

/// `bulk.arxiv.harvest`: saves every OAI-PMH page for the date range, returning how many.
pub fn harvest(out: &Path, since: &str, until: &str, user_agent: &str) -> Result<usize> {
    todo!("{out:?} {since} {until} {user_agent}")
}

/// `bulk.arxiv.run` over OAI-PMH pages (.xml) or the Kaggle snapshot (.json).
pub fn run(files: &[PathBuf], batches: &mut bulk_core::Batches, as_of_us: i64, limit: u64) -> Result<Counts> {
    todo!("{} {as_of_us} {limit} {}", files.len(), batches.rows())
}

/// `bulk.arxiv.open_papers`: the openly licensed papers of OAI-PMH pages, by id.
pub fn open_papers(paths: &[PathBuf]) -> Result<HashMap<String, Paper>> {
    todo!("{}", paths.len())
}

#[derive(clap::Args, Clone, Debug)]
pub struct HarvestArgs {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long = "from")]
    pub since: String,
    #[arg(long)]
    pub until: String,
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// arXiv OAI-PMH pages (.xml) or the Kaggle metadata snapshot (.json).
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

/// `python -m bulk arxiv-harvest`.
pub fn harvest_main(args: HarvestArgs) -> Result<()> {
    todo!("{args:?}")
}

/// `python -m bulk arxiv`.
pub fn main(args: Args) -> Result<()> {
    todo!("{args:?}")
}

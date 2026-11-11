//! PubMed: baseline and update files as paper pages at pubmed.ncbi.nlm.nih.gov, as `bulk/pubmed.py` builds them.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use bulk_core::{Counts, RunArgs};

/// `bulk.pubmed.run`.
pub fn run(files: &[PathBuf], batches: &mut bulk_core::Batches, as_of_us: i64, skip: &HashSet<String>, limit: u64) -> Result<Counts> {
    todo!("{} {as_of_us} {} {limit} {}", files.len(), skip.len(), batches.rows())
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// PubMed baseline or update files (.xml.gz).
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
    /// PMIDs to leave out, one per line, such as those the PMC run took.
    #[arg(long = "skip-pmids")]
    pub skip_pmids: Option<PathBuf>,
}

/// `python -m bulk pubmed`.
pub fn main(args: Args) -> Result<()> {
    todo!("{args:?}")
}

//! The flags every source command takes, and the start and summary line every run shares.

use std::path::PathBuf;

use anyhow::Result;

use crate::counts::Counts;
use crate::rows::Batches;

#[derive(clap::Args, Clone, Debug)]
pub struct RunArgs {
    /// A new folder for this run's batch files.
    #[arg(long)]
    pub out: PathBuf,
    /// When the source's data was taken, as an ISO time with a zone; the pages' fetch time.
    #[arg(long = "as-of")]
    pub as_of: String,
    /// The run's name in its task id, bulk-<source>-<run>; the as-of date by default.
    #[arg(long, default_value = "")]
    pub run: String,
    /// Stop after this many pages, for trials.
    #[arg(long, default_value_t = 0)]
    pub limit: u64,
}

/// The run's batches and its as-of time in microseconds, as `python -m bulk` opens them for `source`.
pub fn start(args: &RunArgs, source: &str) -> Result<(Batches, i64)> {
    todo!("{args:?} {source}")
}

/// Closes the run and prints its summary line exactly as `python -m bulk` prints it.
pub fn finish(args: &RunArgs, batches: Batches, counts: &Counts) -> Result<()> {
    todo!("{args:?} {counts:?} {}", batches.rows())
}

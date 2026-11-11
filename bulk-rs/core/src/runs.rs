//! `bulk diff`: whether two run folders hold the same manifest and the same rows, column by column.

use std::path::{Path, PathBuf};

use anyhow::Result;

#[derive(clap::Args, Clone, Debug)]
pub struct DiffArgs {
    pub a: PathBuf,
    pub b: PathBuf,
    /// Differing rows to describe.
    #[arg(long, default_value_t = 20)]
    pub show: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diff {
    pub rows_a: u64,
    pub rows_b: u64,
    pub rows_differing: u64,
    pub manifest_differences: Vec<String>,
    pub examples: Vec<String>,
}

pub fn diff(a: &Path, b: &Path, show: usize) -> Result<Diff> {
    todo!("{a:?} {b:?} {show}")
}

/// Prints the diff as one JSON line; an error when the runs differ.
pub fn main(args: DiffArgs) -> Result<()> {
    todo!("{args:?}")
}

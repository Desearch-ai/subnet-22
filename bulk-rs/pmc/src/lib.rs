//! PMC: open-access articles under commercial-use licences, as `bulk/pmc.py` builds them; bodies render in the Python worker.

use std::path::PathBuf;

use anyhow::Result;
use bulk_core::RunArgs;

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The bucket's inventory CSV files; read from the bucket.
    #[arg(long, num_args = 0..)]
    pub inventory: Vec<PathBuf>,
    /// A folder of <name>.json and <name>.xml pairs instead of the bucket.
    #[arg(long)]
    pub folder: Option<PathBuf>,
    /// Version names to read from the bucket in this order, one per line, as `pmc-order` writes them.
    #[arg(long)]
    pub names: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    pub workers: usize,
    /// Write the PMIDs taken here, for the PubMed run's --skip-pmids.
    #[arg(long = "pmids-out")]
    pub pmids_out: Option<PathBuf>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct InventoryArgs {
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(clap::Args, Clone, Debug)]
pub struct OrderArgs {
    #[arg(long, num_args = 1.., required = true)]
    pub inventory: Vec<PathBuf>,
    /// NCBI's PMCID table.
    #[arg(long = "pmc-ids")]
    pub pmc_ids: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub workers: usize,
}

#[derive(clap::Args, Clone, Debug)]
pub struct PmidsArgs {
    #[arg(long, num_args = 1.., required = true)]
    pub inventory: Vec<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub workers: usize,
}

/// `python -m bulk pmc`.
pub fn main(args: Args) -> Result<()> {
    todo!("{args:?}")
}

/// `python -m bulk pmc-inventory`.
pub fn inventory_main(args: InventoryArgs) -> Result<()> {
    todo!("{args:?}")
}

/// `python -m bulk pmc-order`.
pub fn order_main(args: OrderArgs) -> Result<()> {
    todo!("{args:?}")
}

/// `python -m bulk pmc-pmids`.
pub fn pmids_main(args: PmidsArgs) -> Result<()> {
    todo!("{args:?}")
}

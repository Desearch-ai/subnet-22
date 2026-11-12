//! bulk: turn one bulk source into a folder of batch files for `publisher bulk`.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bulk", about = "turn one bulk source into a folder of batch files for `publisher bulk`")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// PubMed baseline or update files (.xml.gz).
    Pubmed(bulk_pubmed::Args),
    /// arXiv OAI-PMH pages (.xml) or the Kaggle metadata snapshot (.json).
    Arxiv(bulk_arxiv::Args),
    /// Common Pile's arXiv full texts (.jsonl.gz), for papers under CC BY, CC BY-SA or CC0.
    ArxivText(bulk_arxiv_text::Args),
    /// PMC open-access articles under commercial-use licenses.
    Pmc(bulk_pmc::Args),
    /// Save arXiv OAI-PMH pages for a date range.
    ArxivHarvest(bulk_arxiv::HarvestArgs),
    /// Download the PMC bucket's newest inventory CSV files.
    PmcInventory(bulk_pmc::InventoryArgs),
    /// The newest version of each PMC article, most cited per year first, by OpenAlex's counts.
    PmcOrder(bulk_pmc::OrderArgs),
    /// PMIDs of the PMC versions taken, from metadata alone.
    PmcPmids(bulk_pmc::PmidsArgs),
    /// Wikimedia Enterprise snapshots of English Wikipedia and its sister projects.
    Wikipedia(bulk_wikipedia::Args),
    /// Print the hidden categories of a category-namespace snapshot.
    HiddenCategories(bulk_wikipedia::HiddenArgs),
    /// A page per Bittensor subnet from its chain identity.
    Subnets(bulk_bittensor::SubnetsArgs),
    /// The Bittensor subnets' fetched website pages.
    SubnetSites(bulk_bittensor::SitesArgs),
    /// The Bittensor subnets' GitHub repositories and READMEs.
    SubnetGithub(bulk_bittensor::GithubArgs),
    /// Whether two run folders hold the same manifest and rows.
    Diff(bulk_core::runs::DiffArgs),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Pubmed(args) => bulk_pubmed::main(args),
        Command::Arxiv(args) => bulk_arxiv::main(args),
        Command::ArxivText(args) => bulk_arxiv_text::main(args),
        Command::Pmc(args) => bulk_pmc::main(args),
        Command::ArxivHarvest(args) => bulk_arxiv::harvest_main(args),
        Command::PmcInventory(args) => bulk_pmc::inventory_main(args),
        Command::PmcOrder(args) => bulk_pmc::order_main(args),
        Command::PmcPmids(args) => bulk_pmc::pmids_main(args),
        Command::Wikipedia(args) => bulk_wikipedia::main(args),
        Command::HiddenCategories(args) => bulk_wikipedia::hidden_main(args),
        Command::Subnets(args) => bulk_bittensor::subnets_main(args),
        Command::SubnetSites(args) => bulk_bittensor::sites_main(args),
        Command::SubnetGithub(args) => bulk_bittensor::github_main(args),
        Command::Diff(args) => bulk_core::runs::main(args),
    }
}

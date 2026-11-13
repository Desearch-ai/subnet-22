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
    /// Save the subnets' chain identities, websites and READMEs for `subnets`, `subnet-sites` and `subnet-github`.
    BittensorFetch(bulk_bittensor::fetch::FetchArgs),
    /// ClinicalTrials.gov, a page per study, from the all-studies JSON zip.
    Ctgov(bulk_ctgov::Args),
    /// DailyMed's human prescription and OTC drug labels, from its full release zips.
    Dailymed(bulk_dailymed::Args),
    /// The eCFR, a page per section, from govinfo's title XML and eCFR's structures.
    Ecfr(bulk_ecfr::Args),
    /// EU acts in force and CJEU judgments in English, from the texts `eurlex-fetch` saved.
    Eurlex(bulk_eurlex::Args),
    /// List EUR-Lex's English acts in force and judgments through Cellar's SPARQL and save their texts from Cellar.
    EurlexFetch(bulk_eurlex::FetchArgs),
    /// The United States Code, a page per section, from an OLRC release point's zip.
    Uscode(bulk_uscode::Args),
    /// The Federal Register, a page per document, from govinfo's yearly zips and the index `fedreg-index` saved.
    Fedreg(bulk_fedreg::Args),
    /// Index the Federal Register's documents by day through federalregister.gov's API.
    FedregIndex(bulk_fedreg::IndexArgs),
    /// Project Gutenberg's English public-domain books, each cut to its opening.
    Gutenberg(bulk_gutenberg::Args),
    /// The World Factbook's final edition, a page per entity, from factbook.json.
    Factbook(bulk_factbook::Args),
    /// MedlinePlus health topics, English, from the daily topics XML.
    Medlineplus(bulk_medlineplus::Args),
    /// PubChem compounds with a description from an openly licensed source, from the pages `pubchem-fetch` saved.
    Pubchem(bulk_pubchem::Args),
    /// Save every page of a PubChem heading's compound annotations from PUG-View.
    PubchemFetch(bulk_pubchem::FetchArgs),
    /// UniProtKB/Swiss-Prot reviewed entries, from the release's XML.
    Uniprot(bulk_uniprot::Args),
    /// Save the newest MedlinePlus topics XML and the Genetics summaries.
    MedlineplusFetch(bulk_medlineplus::FetchArgs),
    /// Save the US Code's current release point.
    UscodeFetch(bulk_uscode::FetchArgs),
    /// Save the eCFR's titles from govinfo and their structures from eCFR.
    EcfrFetch(bulk_ecfr::FetchArgs),
    /// Save DailyMed's daily update zips since a day.
    DailymedFetch(bulk_dailymed::FetchArgs),
    /// Index the Federal Register's days through the API and save their issues from govinfo.
    FedregFetch(bulk_fedreg::FetchArgs),
    /// Save one file by its address, for a source's fetch.
    Download(bulk_core::fetch::DownloadArgs),
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
        Command::BittensorFetch(args) => bulk_bittensor::fetch::main(args),
        Command::Ctgov(args) => bulk_ctgov::main(args),
        Command::Dailymed(args) => bulk_dailymed::main(args),
        Command::Ecfr(args) => bulk_ecfr::main(args),
        Command::Eurlex(args) => bulk_eurlex::main(args),
        Command::EurlexFetch(args) => bulk_eurlex::fetch_main(args),
        Command::Uscode(args) => bulk_uscode::main(args),
        Command::Gutenberg(args) => bulk_gutenberg::main(args),
        Command::Fedreg(args) => bulk_fedreg::main(args),
        Command::FedregIndex(args) => bulk_fedreg::index_main(args),
        Command::Factbook(args) => bulk_factbook::main(args),
        Command::Medlineplus(args) => bulk_medlineplus::main(args),
        Command::Pubchem(args) => bulk_pubchem::main(args),
        Command::PubchemFetch(args) => bulk_pubchem::fetch_main(args),
        Command::Uniprot(args) => bulk_uniprot::main(args),
        Command::MedlineplusFetch(args) => bulk_medlineplus::fetch_main(args),
        Command::UscodeFetch(args) => bulk_uscode::fetch_main(args),
        Command::EcfrFetch(args) => bulk_ecfr::fetch_main(args),
        Command::DailymedFetch(args) => bulk_dailymed::fetch_main(args),
        Command::FedregFetch(args) => bulk_fedreg::fetch_main(args),
        Command::Download(args) => bulk_core::fetch::download_main(args),
        Command::Diff(args) => bulk_core::runs::main(args),
    }
}

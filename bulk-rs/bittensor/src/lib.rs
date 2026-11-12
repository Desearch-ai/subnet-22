//! Bittensor subnets as `bulk/bittensor.py` builds them, the shared extraction's part in `python -m bulk.page_worker`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use bulk_core::worker::{PythonArgs, Worker, PAGE_WORKER};
use bulk_core::{py, urlparse, Batches, Counts, RunArgs};
use indexmap::IndexSet;
use regex::Regex;

pub mod fetch;
mod github;
mod sites;
mod subnets;

pub use github::{readme_page, run_github, GITHUB_SOURCE};
pub use sites::{app_page, run_sites, SITES_SOURCE};
pub use subnets::{run_subnets, subnet_page, url_of, SUBNETS_SOURCE};

/// `desearch.extraction.classify.COMPANY`.
pub const COMPANY: &str = "company";
/// `desearch.extraction.classify.DOCS`.
pub const DOCS: &str = "docs";
pub const REMOVED_FILE: &str = "removed.json";

static TRACKING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:utm_[a-z]+|ref|fbclid|gclid)\n?\z").expect("pattern compiles"));

#[derive(clap::Args, Clone, Debug)]
pub struct SubnetsArgs {
    #[command(flatten)]
    pub run: RunArgs,
    /// The subnets' chain JSON.
    pub chain: PathBuf,
}

#[derive(clap::Args, Clone, Debug)]
pub struct SitesArgs {
    #[command(flatten)]
    pub run: RunArgs,
    /// The folder with sites.json and html/.
    pub folder: PathBuf,
    /// Processes that extract pages; the CPU count by default.
    #[arg(long)]
    pub workers: Option<usize>,
    #[command(flatten)]
    pub python: PythonArgs,
}

#[derive(clap::Args, Clone, Debug)]
pub struct GithubArgs {
    #[command(flatten)]
    pub run: RunArgs,
    /// The folder with github.json and the READMEs.
    pub folder: PathBuf,
    /// Processes that render READMEs; the CPU count by default.
    #[arg(long)]
    pub workers: Option<usize>,
    #[command(flatten)]
    pub python: PythonArgs,
}

/// `without_tracking`: the URL with utm_, ref, fbclid and gclid parameters dropped and its query re-encoded.
pub fn without_tracking(url: &str) -> Result<String> {
    let mut parts = urlparse::urlsplit(url)?;
    let kept: Vec<(String, String)> =
        urlparse::parse_qsl_blank(&parts.query).into_iter().filter(|(k, _)| !TRACKING.is_match(&desearch::text::fold_turkish_i(k))).collect();
    parts.query = urlparse::urlencode(&kept);
    Ok(urlparse::urlunsplit(&parts))
}

/// `remove_named`: pages the fetch found refused, gone or no longer listed, unless this build holds them.
pub fn remove_named(folder: &Path, batches: &mut Batches, kept: &HashSet<String>, counts: &mut Counts) -> Result<()> {
    let named = folder.join(REMOVED_FILE);
    if !named.exists() {
        return Ok(());
    }
    let removed = py::load_json(&named)?;
    let mut urls = IndexSet::new();
    for record in py::list(&removed, REMOVED_FILE)? {
        urls.insert(without_tracking(py::str_of(py::item(record, "url")?, "a removed url")?)?);
    }
    for url in urls {
        if !kept.contains(&url) {
            batches.remove(&url);
            counts.add("removed", 1);
        }
    }
    Ok(())
}

fn worker_count(workers: Option<usize>) -> usize {
    workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())).max(1)
}

pub fn subnets_main(args: SubnetsArgs) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SUBNETS_SOURCE)?;
    let counts = run_subnets(&args.chain, &mut batches, as_of_us)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

pub fn sites_main(args: SitesArgs) -> Result<()> {
    let python_path = args.python.python_path(PAGE_WORKER);
    let spawn = || Worker::spawn(&args.python.python, &python_path, PAGE_WORKER, &[]);
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SITES_SOURCE)?;
    let counts = run_sites(&args.folder, &mut batches, as_of_us, worker_count(args.workers), &spawn)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

pub fn github_main(args: GithubArgs) -> Result<()> {
    let python_path = args.python.python_path(PAGE_WORKER);
    let spawn = || Worker::spawn(&args.python.python, &python_path, PAGE_WORKER, &[]);
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, GITHUB_SOURCE)?;
    let counts = run_github(&args.folder, &mut batches, as_of_us, worker_count(args.workers), &spawn)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_parameters_go() {
        let cases = [
            ("https://www.natix.network/?utm_source=bittensor", "https://www.natix.network/"),
            ("https://vidaio.io/sign-in?authRedir=/dashboard", "https://vidaio.io/sign-in?authRedir=%2Fdashboard"),
            ("https://actual.inc/?home", "https://actual.inc/?home="),
            ("https://x.io/?UTM_Medium=a&ref%0A=b&utm_\u{131}=c&refs=d&a+b=%7e#top", "https://x.io/?refs=d&a+b=~#top"),
            ("https://x.io/p?", "https://x.io/p"),
        ];
        for (url, want) in cases {
            assert_eq!(without_tracking(url).unwrap(), want, "{url}");
        }
    }
}

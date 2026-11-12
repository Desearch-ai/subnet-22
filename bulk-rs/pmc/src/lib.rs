//! PMC: open-access articles under commercial-use licences, as `bulk/pmc.py` builds them; bodies render in the Python worker.

use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use bulk_core::http::{Client, USER_AGENT};
use bulk_core::{upload_row, Batches, Counts, RunArgs};
use serde_json::Value;

pub mod ahead;
pub mod bucket;
pub mod csv;
pub mod order;
pub mod page;
pub mod py;
pub mod render;

pub use ahead::bounded_map;
pub use bucket::{
    fetch, from_bucket, from_folder, inventory_keys, latest_from_inventory, latest_inventory, latest_versions, left_out, taken_pmids, Article,
    AHEAD_PER_WORKER, BUCKET,
};
pub use order::{citations, order, pmc_ids};
pub use page::{article_page, parse_front, Outcome, Parsed};
pub use render::{RenderArgs, Rendered, Renderer};

pub const CONTENT_TYPE: &str = "application/xml";

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
    #[command(flatten)]
    pub render: RenderArgs,
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

/// What `run` counts for one version, worked out on a render thread.
fn outcome(meta: &Value, xml: Option<&[u8]>, renderer: &Renderer) -> Result<Result<Outcome, &'static str>> {
    if let Some(why) = left_out(meta) {
        return Ok(Err(why));
    }
    let Some(xml) = xml else { return Ok(Err("no_xml")) };
    Ok(Ok(article_page(xml, meta, renderer)?))
}

/// `bulk.pmc.run`: pages of the versions taken; `pmids` collects their PMIDs for the PubMed run.
pub fn run<I>(articles: I, batches: &mut Batches, as_of_us: i64, limit: u64, mut pmids: Option<&mut Vec<String>>, renderer: Arc<Renderer>) -> Result<Counts>
where
    I: Iterator<Item = Result<Article>>,
{
    let threads = renderer.workers();
    let outcomes = bounded_map(articles, threads, threads * AHEAD_PER_WORKER, move |article: Result<Article>| {
        let (meta, xml) = article?;
        let found = outcome(&meta, xml.as_deref(), &renderer)?;
        Ok::<_, anyhow::Error>((meta, found))
    });
    let mut counts = Counts::default();
    for item in outcomes {
        let (meta, found) = item?;
        match found {
            Err(why) => counts.add(why, 1),
            Ok(Outcome::BadXml) => counts.add("bad_xml", 1),
            Ok(Outcome::Skipped(why)) => counts.add(why, 1),
            Ok(Outcome::Page { url, page, no_body }) => {
                if no_body {
                    counts.add("no_body", 1);
                }
                batches.add(upload_row(&url, *page, as_of_us, CONTENT_TYPE))?;
                counts.add("pages", 1);
                if let Some(pmids) = pmids.as_deref_mut() {
                    if py::truthy(meta.get("pmid")) {
                        pmids.push(meta.get("pmid").map(py::str_of).unwrap_or_default());
                    }
                }
                if limit > 0 && counts.get("pages") >= limit {
                    break;
                }
            }
        }
    }
    Ok(counts)
}

fn lines(items: &[String]) -> String {
    items.iter().map(|item| format!("{item}\n")).collect()
}

/// `bulk pmc`.
pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, "pmc")?;
    let renderer = Arc::new(Renderer::start(&args.render)?);
    let articles: Box<dyn Iterator<Item = Result<Article>>> = match (&args.folder, &args.names) {
        (Some(folder), _) => Box::new(from_folder(folder)?),
        (None, names) => {
            let names = match names {
                Some(path) => bucket::read_names(path)?,
                None => bucket::sorted_names(&args.inventory)?,
            };
            let client = Arc::new(Client::new(USER_AGENT)?);
            Box::new(from_bucket(names.into_iter(), client, BUCKET.to_string(), args.workers))
        }
    };
    let mut pmids = Vec::new();
    let counts = run(articles, &mut batches, as_of_us, args.run.limit, Some(&mut pmids), renderer)?;
    if let Some(path) = &args.pmids_out {
        fs::write(path, lines(&pmids))?;
    }
    bulk_core::run::finish(&args.run, batches, &counts)
}

/// `bulk pmc-inventory`.
pub fn inventory_main(args: InventoryArgs) -> Result<()> {
    let client = Client::new(USER_AGENT)?;
    for path in latest_inventory(&args.out, &client, BUCKET)? {
        println!("{}", bulk_core::run::path_str(&path));
    }
    Ok(())
}

/// `bulk pmc-order`.
pub fn order_main(args: OrderArgs) -> Result<()> {
    let names = latest_from_inventory(&args.inventory)?;
    let ids = pmc_ids(&args.pmc_ids)?;
    let wanted: HashSet<String> = names.keys().filter_map(|p| ids.get(p)).map(|(pmid, _)| pmid.clone()).filter(|pmid| !pmid.is_empty()).collect();
    let client = Client::new(USER_AGENT)?;
    let files = order::openalex_files(&client)?;
    let with_pmid = wanted.len();
    let cited = citations(&files, Arc::new(order::RangeClient::new(USER_AGENT)?), Arc::new(wanted), args.workers)?;
    let ordered = order(&names, &ids, &cited, order::this_year());
    fs::write(&args.out, lines(&ordered))?;
    println!(
        "{{\"versions\": {}, \"with_pmid\": {}, \"cited\": {}, \"out\": {}}}",
        ordered.len(),
        with_pmid,
        cited.len(),
        bulk_core::pyjson::quote(&bulk_core::run::path_str(&args.out))
    );
    Ok(())
}

/// `bulk pmc-pmids`.
pub fn pmids_main(args: PmidsArgs) -> Result<()> {
    let names = bucket::sorted_names(&args.inventory)?;
    // A list cut short by a failed read must not pass for whole as a PubMed run's --skip-pmids.
    let partial = PathBuf::from(format!("{}.partial", args.out.display()));
    let mut out = BufWriter::new(fs::File::create(&partial)?);
    let client = Arc::new(Client::new(USER_AGENT)?);
    for pmid in taken_pmids(names.into_iter(), client, BUCKET.to_string(), args.workers) {
        writeln!(out, "{}", pmid?)?;
    }
    out.flush()?;
    drop(out);
    fs::rename(&partial, &args.out)?;
    Ok(())
}

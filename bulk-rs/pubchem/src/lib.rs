//! PubChem: a page per compound that has a description from a source whose licence allows reuse, from PUG-View's heading annotations, each description credited to its source.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bulk_core::http::USER_AGENT;
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use indexmap::IndexMap;
use serde_json::Value;

pub const SOURCE: &str = "pubchem";
pub const SITE: &str = "PubChem";
pub const COMPOUND: &str = "article";
const PUG_VIEW: &str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON";
const CONTENT_TYPE: &str = "application/json";

/// Description sources whose terms allow reuse, with the licence named on each description; every other source is left out.
pub const ALLOWED: &[(&str, &str)] = &[
    ("Agency for Toxic Substances and Disease Registry (ATSDR)", "US government work"),
    ("ChEBI", "CC BY 4.0"),
    ("LOTUS - the natural products occurrence database", "CC0"),
    ("NCI Thesaurus (NCIt)", "US government work"),
    ("Medical Subject Headings (MeSH)", "US government work"),
    ("LiverTox", "US government work"),
    ("FDA Pharm Classes", "US government work"),
    ("Hazardous Substances Data Bank (HSDB)", "US government work"),
    ("DTP/NCI", "US government work"),
    ("Dietary Supplement Label Database (DSLD)", "CC0"),
    ("Wikidata", "CC0"),
    ("FDA Global Substance Registration System (GSRS)", "US government work"),
    ("EPA DSSTox", "US government work"),
    ("ChemIDplus", "US government work"),
    ("EPA Hazardous Air Pollutants", "US government work"),
    ("GlyCosmos Glycoscience Portal", "CC BY 4.0"),
];
/// A compound page shorter than this is a one-line stub, such as a lone natural-product occurrence sentence.
const MIN_TEXT_CHARS: usize = 300;

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for the pages, page-<n>.json; a fetch skips pages already saved.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "Record Description")]
    pub heading: String,
    /// Seconds between requests, PubChem's robots.txt crawl delay.
    #[arg(long, default_value_t = 5.0)]
    pub delay: f64,
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The folder `pubchem-fetch` filled.
    pub folder: PathBuf,
}

fn page_url(heading: &str, page: usize) -> String {
    format!("{PUG_VIEW}?heading={}&heading_type=Compound&page={page}", heading.replace(' ', "%20"))
}

/// Every page of a heading's compound annotations, under the crawl delay.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let client = reqwest::blocking::Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(300)).build()?;
    let gap = Duration::from_secs_f64(args.delay);
    let mut last = Instant::now() - gap;
    let (mut page, mut total, mut fetched) = (1usize, 1usize, 0usize);
    while page <= total {
        let path = args.out.join(format!("page-{page}.json"));
        let body: Value = if path.exists() {
            serde_json::from_slice(&std::fs::read(&path)?)?
        } else {
            std::thread::sleep(gap.saturating_sub(last.elapsed()));
            last = Instant::now();
            let bytes = get(&client, &page_url(&args.heading, page))?;
            std::fs::write(path.with_extension("part"), &bytes)?;
            std::fs::rename(path.with_extension("part"), &path)?;
            fetched += 1;
            serde_json::from_slice(&bytes)?
        };
        total = body.pointer("/Annotations/TotalPages").and_then(Value::as_u64).context("a page without TotalPages")? as usize;
        page += 1;
    }
    println!("{}", serde_json::json!({"pages": total, "fetched": fetched}));
    Ok(())
}

fn get(client: &reqwest::blocking::Client, url: &str) -> Result<Vec<u8>> {
    for attempt in 0..6u32 {
        match client.get(url).send() {
            Ok(r) if r.status().is_success() => return Ok(r.bytes()?.to_vec()),
            Ok(r) if attempt == 5 => bail!("PubChem answered {} for {url}", r.status()),
            Err(e) if attempt == 5 => return Err(e.into()),
            _ => std::thread::sleep(Duration::from_secs(30 * (attempt as u64 + 1))),
        }
    }
    unreachable!("the last attempt returns")
}

/// One description of a compound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Description {
    pub source: String,
    pub licence: String,
    pub name: String,
    pub text: String,
}

fn texts(annotation: &Value) -> Vec<String> {
    let mut found = Vec::new();
    for data in annotation.get("Data").and_then(Value::as_array).into_iter().flatten() {
        for s in data.pointer("/Value/StringWithMarkup").and_then(Value::as_array).into_iter().flatten() {
            if let Some(t) = s.get("String").and_then(Value::as_str) {
                let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
                if !t.is_empty() && !found.contains(&t) {
                    found.push(t);
                }
            }
        }
    }
    found
}

/// The allowed descriptions of every page, by compound, in the order the pages give them.
pub fn descriptions(folder: &Path, counts: &mut Counts) -> Result<IndexMap<u64, Vec<Description>>> {
    let mut pages: Vec<(usize, PathBuf)> = std::fs::read_dir(folder)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| p.file_name()?.to_str()?.strip_prefix("page-")?.strip_suffix(".json")?.parse().ok().map(|n| (n, p.clone())))
        .collect();
    pages.sort();
    let mut by_cid: IndexMap<u64, Vec<Description>> = IndexMap::new();
    for (_, path) in pages {
        let body: Value = serde_json::from_slice(&std::fs::read(&path)?).with_context(|| format!("reading {}", path.display()))?;
        for a in body.pointer("/Annotations/Annotation").and_then(Value::as_array).into_iter().flatten() {
            let source = a.get("SourceName").and_then(Value::as_str).unwrap_or("");
            let Some((_, licence)) = ALLOWED.iter().find(|(s, _)| *s == source) else {
                counts.add("left_out_source", 1);
                continue;
            };
            let name = a.get("Name").and_then(Value::as_str).unwrap_or("").to_string();
            let cids: Vec<u64> = a.pointer("/LinkedRecords/CID").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64).collect();
            for text in texts(a) {
                for cid in &cids {
                    let found = by_cid.entry(*cid).or_default();
                    if !found.iter().any(|d| d.text == text) {
                        found.push(Description { source: source.to_string(), licence: licence.to_string(), name: name.clone(), text: text.clone() });
                    }
                }
            }
        }
    }
    Ok(by_cid)
}

/// A compound's page: its most given name as the title, then each description with its source and licence.
pub fn page_of(cid: u64, found: &[Description]) -> (String, Page) {
    let mut names: IndexMap<&str, usize> = IndexMap::new();
    for d in found.iter().filter(|d| !d.name.is_empty()) {
        *names.entry(d.name.as_str()).or_default() += 1;
    }
    let title = names.iter().max_by_key(|(_, n)| **n).map(|(name, _)| name.to_string()).unwrap_or_else(|| format!("CID {cid}"));
    let url = format!("https://pubchem.ncbi.nlm.nih.gov/compound/{cid}");
    let mut lines = vec![Line::new(format!("Source: PubChem, National Center for Biotechnology Information, compound CID {cid}; each description is credited to its source below."))];
    let mut sources: Vec<&str> = Vec::new();
    for d in found {
        lines.push(Line::new(format!("{} (Source: {}, {})", d.text, d.source, d.licence)));
        if !sources.contains(&d.source.as_str()) {
            sources.push(&d.source);
        }
    }
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), site_name: SITE.into(), tags: sources.iter().map(|s| s.to_string()).collect(), ..PaperMeta::default() };
    (url, typed_page(COMPOUND, &title, lines, meta))
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let mut counts = Counts::default();
    let by_cid = descriptions(&args.folder, &mut counts)?;
    let mut seen = HashSet::new();
    for (cid, found) in &by_cid {
        if !seen.insert(*cid) || found.is_empty() {
            continue;
        }
        let (url, page) = page_of(*cid, found);
        if page.text.chars().count() < MIN_TEXT_CHARS {
            counts.add("thin", 1);
            continue;
        }
        batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
        counts.add("pages", 1);
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            break;
        }
    }
    Ok(counts)
}

pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn allowed_sources_only_credited_on_the_page() {
        let dir = tempfile::tempdir().unwrap();
        let page = json!({"Annotations": {"Page": 1, "TotalPages": 1, "Annotation": [
            {"SourceName": "ChEBI", "Name": "aspirin", "LinkedRecords": {"CID": [2244]}, "Data": [{"Value": {"StringWithMarkup": [{"String": "Aspirin is a member of the benzoic acids."}]}}]},
            {"SourceName": "DrugBank", "Name": "Acetylsalicylic acid", "LinkedRecords": {"CID": [2244]}, "Data": [{"Value": {"StringWithMarkup": [{"String": "Licensed text."}]}}]},
            {"SourceName": "Medical Subject Headings (MeSH)", "Name": "Aspirin", "LinkedRecords": {"CID": [2244]}, "Data": [{"Value": {"StringWithMarkup": [{"String": "The prototypical analgesic."}]}}]}
        ]}});
        std::fs::write(dir.path().join("page-1.json"), page.to_string()).unwrap();
        let mut counts = Counts::default();
        let by_cid = descriptions(dir.path(), &mut counts).unwrap();
        assert_eq!(counts.get("left_out_source"), 1);
        let (url, page) = page_of(2244, &by_cid[&2244]);
        assert_eq!(url, "https://pubchem.ncbi.nlm.nih.gov/compound/2244");
        assert!(!page.text.contains("Licensed text"));
        assert!(page.text.contains("Aspirin is a member of the benzoic acids. (Source: ChEBI, CC BY 4.0)"));
        assert!(page.text.contains("The prototypical analgesic. (Source: Medical Subject Headings (MeSH), US government work)"));
    }
}

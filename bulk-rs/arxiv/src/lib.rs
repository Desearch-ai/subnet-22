//! arXiv: the OAI-PMH harvest and one abstract page a paper, as `bulk/arxiv.py` builds them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use bulk_core::{authors_line, paper_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use rayon::prelude::*;

mod harvest;
mod kaggle;
mod oai;

pub use harvest::{harvest_with, oai_error, resumption_token, urlencode, BUSY_TRIES, NO_RECORDS, OAI, TRIES};
pub use kaggle::{kaggle_date, kaggle_record, kaggle_records};
pub use oai::{oai_papers, oai_records, papers_in, OaiItem, ARXIV_NS, OAI_NS};

/// arXiv's robots.txt asks for 15 s between requests.
pub const DELAY_S: f64 = 15.0;
pub const SITE: &str = "arXiv";
/// Licences under which a commercial index may keep and show a paper's full text.
pub const OPEN: [&str; 5] = [
    "creativecommons.org/licenses/by/4.0/",
    "creativecommons.org/licenses/by/3.0/",
    "creativecommons.org/licenses/by-sa/4.0/",
    "creativecommons.org/publicdomain/zero/1.0/",
    "creativecommons.org/licenses/publicdomain/",
];
/// arXiv makes an HTML view of every TeX submission from December 2023 on; ar5iv holds the earlier ones.
pub const HTML_FROM: &str = "2312";
const MOST_AUTHORS: usize = 10;
const CONTENT_TYPE: &str = "application/xml";

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

/// What a source file holds for `run`: a deleted paper's id, or an abstract page at its URL.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum Record {
    Deleted(String),
    Page(String, Page),
}

pub fn url_of(arxiv_id: &str) -> String {
    format!("https://arxiv.org/abs/{arxiv_id}")
}

/// The HTML view that shows a paper's full text.
pub fn html_url(arxiv_id: &str) -> String {
    if !arxiv_id.contains('/') && desearch::text::head(arxiv_id, 4) >= HTML_FROM {
        format!("https://arxiv.org/html/{arxiv_id}")
    } else {
        format!("https://ar5iv.labs.arxiv.org/html/{arxiv_id}")
    }
}

pub fn is_open(licence: &str) -> bool {
    let licence = desearch::text::strip(licence);
    let rest = licence.split_once("://").map_or(licence, |(_, rest)| rest);
    OPEN.contains(&rest)
}

/// Python's `" ".join((text or "").split())`.
pub fn squash(text: &str) -> String {
    desearch::text::collapse(text)
}

/// The authors line and the `arXiv:id · categories · Submitted …` line every arXiv page opens with.
pub fn heading_lines(paper: &Paper) -> Vec<Line> {
    let mut stamp = format!("Submitted {}", paper.created);
    if !paper.updated.is_empty() && paper.updated != paper.created {
        stamp.push_str(&format!(", updated {}", paper.updated));
    }
    let id = format!("arXiv:{}", paper.arxiv_id);
    let parts: Vec<&str> = [id.as_str(), paper.categories.as_str(), stamp.as_str()].into_iter().filter(|p| !p.is_empty()).collect();
    vec![Line::new(authors_line(&paper.authors, MOST_AUTHORS)), Line::new(parts.join(" · "))]
}

/// `bulk.arxiv.framed`: the paper's page at `url` from its lines.
pub fn framed(paper: &Paper, url: &str, lines: Vec<Line>) -> (String, Page) {
    let meta = PaperMeta {
        lang: "en".to_string(),
        url: url.to_string(),
        published: paper.created.clone(),
        modified: if paper.updated != paper.created { paper.updated.clone() } else { String::new() },
        author: authors_line(&paper.authors, MOST_AUTHORS),
        site_name: SITE.to_string(),
        tags: desearch::text::words(&paper.categories).map(String::from).collect(),
    };
    (url.to_string(), paper_page(&paper.title, lines, meta))
}

/// `bulk.arxiv.page_of`: the abstract page.
pub fn page_of(paper: &Paper) -> (String, Page) {
    let mut lines = heading_lines(paper);
    lines.push(Line::new("## Abstract"));
    lines.push(Line::new(paper.abstract_text.clone()));
    for (name, value) in [("Comments", &paper.comments), ("Journal reference", &paper.journal), ("DOI", &paper.doi)] {
        if !value.is_empty() {
            lines.push(Line::new(format!("{name}: {value}")));
        }
    }
    framed(paper, &url_of(&paper.arxiv_id), lines)
}

/// `bulk.arxiv.harvest`: saves every OAI-PMH page for the date range, returning how many.
pub fn harvest(out: &Path, since: &str, until: &str, user_agent: &str) -> Result<usize> {
    harvest_with(out, since, until, user_agent, OAI, Duration::from_secs_f64(DELAY_S))
}

/// Python's `path.suffix in (".json", ".jsonl")`.
fn is_kaggle(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    let suffix = match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => &name[i..],
        _ => "",
    };
    suffix == ".json" || suffix == ".jsonl"
}

/// Takes one record into the batches, returning whether the page limit is reached.
fn take(record: Record, batches: &mut Batches, as_of_us: i64, limit: u64, counts: &mut Counts) -> Result<bool> {
    match record {
        Record::Deleted(id) => {
            batches.remove(&url_of(&id));
            counts.add("deleted", 1);
            Ok(false)
        }
        Record::Page(url, page) => {
            if page.title.is_empty() || page.text.chars().count() <= page.title.chars().count() + 1 {
                counts.add("empty", 1);
                return Ok(false);
            }
            batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
            counts.add("pages", 1);
            Ok(limit > 0 && counts.get("pages") >= limit)
        }
    }
}

/// Snapshot lines framed in parallel this many at a time.
const KAGGLE_CHUNK: usize = 4096;

fn run_kaggle(path: &Path, batches: &mut Batches, as_of_us: i64, limit: u64, counts: &mut Counts) -> Result<bool> {
    let mut lines = kaggle::lines(path)?;
    loop {
        let chunk: Vec<Result<String>> = lines.by_ref().take(KAGGLE_CHUNK).collect();
        if chunk.is_empty() {
            return Ok(false);
        }
        let records: Vec<Result<Record>> = chunk.into_par_iter().map(|line| line.and_then(|line| kaggle_record(&line))).collect();
        for record in records {
            if take(record.map_err(|e| e.context(format!("in {}", path.display())))?, batches, as_of_us, limit, counts)? {
                return Ok(true);
            }
        }
    }
}

/// `bulk.arxiv.run` over OAI-PMH pages (.xml) or the Kaggle snapshot (.json).
pub fn run(files: &[PathBuf], batches: &mut bulk_core::Batches, as_of_us: i64, limit: u64) -> Result<Counts> {
    let mut counts = Counts::default();
    let window = 2 * rayon::current_num_threads().max(1);
    let mut i = 0;
    while i < files.len() {
        if is_kaggle(&files[i]) {
            if run_kaggle(&files[i], batches, as_of_us, limit, &mut counts)? {
                return Ok(counts);
            }
            i += 1;
            continue;
        }
        let end = (i..files.len().min(i + window)).find(|&j| is_kaggle(&files[j])).unwrap_or(files.len().min(i + window));
        let parsed: Vec<Result<Vec<Result<Record>>>> = files[i..end].par_iter().map(|path| oai_records(path)).collect();
        for file in parsed {
            for record in file? {
                if take(record?, batches, as_of_us, limit, &mut counts)? {
                    return Ok(counts);
                }
            }
        }
        i = end;
    }
    Ok(counts)
}

/// `bulk.arxiv.open_papers`: the openly licensed papers of OAI-PMH pages, by id.
pub fn open_papers(paths: &[PathBuf]) -> Result<HashMap<String, Paper>> {
    let parsed: Vec<Result<Vec<Result<Paper>>>> = paths
        .par_iter()
        .map(|path| {
            Ok(oai_papers(path)?
                .into_iter()
                .filter_map(|item| match item {
                    Ok(OaiItem::Paper(paper)) if is_open(&paper.licence) => Some(Ok(paper)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect())
        })
        .collect();
    let mut papers = HashMap::new();
    for file in parsed {
        for paper in file? {
            let paper = paper?;
            papers.insert(paper.arxiv_id.clone(), paper);
        }
    }
    Ok(papers)
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
    let pages = harvest(&args.out, &args.since, &args.until, bulk_core::http::USER_AGENT)?;
    println!("{pages} pages saved in {}", bulk_core::run::path_str(&args.out));
    Ok(())
}

/// `python -m bulk arxiv`.
pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, "arxiv")?;
    let counts = run(&args.files, &mut batches, as_of_us, args.run.limit)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

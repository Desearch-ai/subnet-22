//! PubMed: baseline and update files as paper pages at pubmed.ncbi.nlm.nih.gov, with the PMIDs of deleted citations.

pub mod xml;

use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

use anyhow::{anyhow, Context, Result};
use bulk_core::{authors_line, paper_page, upload_row, Counts, Line, Page, PaperMeta, Row, RunArgs};
use desearch::text::{collapse, head, strip, words};
use flate2::read::MultiGzDecoder;
use rayon::prelude::*;

pub use xml::{Element, Keep, Node, Records};

pub const SITE: &str = "PubMed";
/// The parts of each record `run` reads; the rest is parsed and checked but not kept.
pub const RECORDS: &[(&str, Keep)] = &[("PubmedArticle", ARTICLE), ("DeleteCitation", Keep::Children(&[("PMID", Keep::Text)]))];
const DATE: Keep = Keep::Children(&[("Year", Keep::Text), ("Month", Keep::Text), ("Day", Keep::Text), ("MedlineDate", Keep::Text)]);
const AUTHOR: Keep = Keep::Children(&[("CollectiveName", Keep::Text), ("ForeName", Keep::Text), ("LastName", Keep::Text)]);
const ARTICLE: Keep = Keep::Children(&[
    (
        "MedlineCitation",
        Keep::Children(&[
            ("PMID", Keep::Text),
            (
                "Article",
                Keep::Children(&[
                    ("Journal", Keep::Children(&[("Title", Keep::Text), ("JournalIssue", Keep::Children(&[("PubDate", DATE)]))])),
                    ("ArticleTitle", Keep::AllText),
                    ("VernacularTitle", Keep::AllText),
                    ("Abstract", Keep::Children(&[("AbstractText", Keep::AllText)])),
                    ("AuthorList", Keep::Children(&[("Author", AUTHOR)])),
                    ("Language", Keep::Text),
                    ("ArticleDate", DATE),
                ]),
            ),
            ("KeywordList", Keep::Children(&[("Keyword", Keep::AllText)])),
            ("MeshHeadingList", Keep::Children(&[("MeshHeading", Keep::Children(&[("DescriptorName", Keep::AllText)]))])),
        ]),
    ),
    ("PubmedData", Keep::Children(&[("ArticleIdList", Keep::Children(&[("ArticleId", Keep::Text)]))])),
]);
const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const CONTENT_TYPE: &str = "application/xml";
const MOST_AUTHORS: usize = 10;
const CHUNK_RECORDS: usize = 128;
const FILES_AHEAD: usize = 4;
// Chunks a file further down the list may build before the run reaches it: about a whole file.
const BUILT_AHEAD: usize = 256;

pub fn url_of(pmid: &str) -> String {
    format!("https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
}

pub fn text_of(node: Option<&Element>) -> String {
    node.map(|n| collapse(&n.itertext())).unwrap_or_default()
}

/// YYYY-MM-DD when the node names a year, month and day, else empty.
pub fn full_date(node: Option<&Element>) -> String {
    let Some(node) = node else {
        return String::new();
    };
    let [year, month, day] = ["Year", "Month", "Day"].map(|k| node.findtext(k).unwrap_or(""));
    let key = head(month, 3).to_lowercase();
    let month = match MONTHS.iter().position(|m| *m == key) {
        Some(i) => (i + 1).to_string(),
        None => month.to_string(),
    };
    if !(is_digits(year) && is_digits(&month) && is_digits(day)) {
        return String::new();
    }
    format!("{}-{}-{}", padded(year, 4), padded(&month, 2), padded(day, 2))
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `f"{int(digits):0{width}d}"` for any number of digits.
fn padded(digits: &str, width: usize) -> String {
    let n = digits.trim_start_matches('0');
    format!("{:0>width$}", if n.is_empty() { "0" } else { n })
}

pub fn author_name(author: &Element) -> String {
    if let Some(collective) = author.findtext("CollectiveName").filter(|c| !c.is_empty()) {
        return collapse(collective);
    }
    joined(&[author.findtext("ForeName"), author.findtext("LastName")], " ")
}

fn joined(parts: &[Option<&str>], sep: &str) -> String {
    parts.iter().flatten().filter(|p| !p.is_empty()).copied().collect::<Vec<_>>().join(sep)
}

pub fn abstract_lines(abstract_: &Element) -> Vec<Line> {
    let mut lines = Vec::new();
    for part in abstract_.iterfind("AbstractText") {
        let label = strip(part.get("Label").unwrap_or(""));
        if !label.is_empty() {
            let label = if is_upper(label) { capitalize(label) } else { label.to_string() };
            lines.push(Line::new(format!("## {label}")));
        }
        let text = text_of(Some(part));
        if !text.is_empty() {
            lines.push(Line::new(text));
        }
    }
    lines
}

/// Python's `str.isupper`.
pub fn is_upper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() || is_titlecase(c) {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

fn is_titlecase(c: char) -> bool {
    matches!(c, '\u{1C5}' | '\u{1C8}' | '\u{1CB}' | '\u{1F2}' | '\u{1F88}'..='\u{1F8F}' | '\u{1F98}'..='\u{1F9F}' | '\u{1FA8}'..='\u{1FAF}' | '\u{1FBC}' | '\u{1FCC}' | '\u{1FFC}')
}

/// Python's `str.capitalize` on a string `is_upper` accepts, whose first character is uppercase or uncased.
fn capitalize(s: &str) -> String {
    let Some(first) = s.chars().next() else {
        return String::new();
    };
    let title = match first {
        '\u{1C4}' => '\u{1C5}',
        '\u{1C7}' => '\u{1C8}',
        '\u{1CA}' => '\u{1CB}',
        '\u{1F1}' => '\u{1F2}',
        c => c,
    };
    // Lowering the whole string keeps final sigma's context, as Python's capitalize does.
    let mut out = String::from(title);
    out.extend(s.to_lowercase().chars().skip(first.to_lowercase().count()));
    out
}

/// What `article_page` makes of one PubmedArticle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Article {
    Page {
        url: String,
        page: Box<Page>,
    },
    /// The counts key naming why there is no page.
    Skipped(&'static str),
}

/// The citation's page, or why it has none; an error when it lacks a part every citation has.
pub fn article_page(article: &Element) -> Result<Article> {
    let citation = article.find("MedlineCitation").ok_or_else(|| anyhow!("a PubmedArticle has no MedlineCitation"))?;
    let pmid = citation.findtext("PMID").unwrap_or("");
    let body = citation.find("Article").ok_or_else(|| anyhow!("PubmedArticle {pmid:?} has no Article"))?;
    let abstract_ = body.find("Abstract");
    let Some(abstract_) = abstract_.filter(|a| !pmid.is_empty() && a.find("AbstractText").is_some()) else {
        return Ok(Article::Skipped("no_abstract"));
    };
    let mut title = text_of(body.find("ArticleTitle"));
    if title.is_empty() {
        title = text_of(body.find("VernacularTitle"));
    }
    if title.is_empty() {
        return Ok(Article::Skipped("no_title"));
    }
    let journal = body.find("Journal").ok_or_else(|| anyhow!("PubmedArticle {pmid:?} has no Journal"))?;
    let issue = journal.find("JournalIssue/PubDate");
    let mut published = full_date(body.find("ArticleDate"));
    if published.is_empty() {
        published = full_date(issue);
    }
    let when = if published.is_empty() {
        let issue = issue.ok_or_else(|| anyhow!("PubmedArticle {pmid:?} has no JournalIssue/PubDate"))?;
        match joined(&[issue.findtext("Year"), issue.findtext("Month")], " ") {
            ym if ym.is_empty() => issue.findtext("MedlineDate").unwrap_or("").to_string(),
            ym => ym,
        }
    } else {
        published.clone()
    };
    let doi = article
        .iterfind("PubmedData/ArticleIdList/ArticleId")
        .into_iter()
        .filter(|i| i.get("IdType") == Some("doi"))
        .find_map(|i| i.text().filter(|t| !t.is_empty()))
        .unwrap_or("");
    let names: Vec<String> = body.iterfind("AuthorList/Author").into_iter().map(author_name).collect();
    let authors = authors_line(&names, MOST_AUTHORS);
    let doi = if doi.is_empty() { String::new() } else { format!("doi: {doi}") };
    let source = joined(&[journal.findtext("Title"), Some(&when), Some(&doi)], ". ");
    let mut lines: Vec<Line> = [&authors, &source].into_iter().filter(|t| !t.is_empty()).map(|t| Line::new(t.as_str())).collect();
    lines.extend(abstract_lines(abstract_));
    let languages: Vec<&str> = body.iterfind("Language").into_iter().filter_map(|l| l.text()).filter(|t| !t.is_empty()).collect();
    let lang = if languages.contains(&"eng") { "en" } else { languages.first().copied().unwrap_or("") };
    let tags: Vec<String> = citation
        .iterfind("KeywordList/Keyword")
        .into_iter()
        .chain(citation.iterfind("MeshHeadingList/MeshHeading/DescriptorName"))
        .map(|k| text_of(Some(k)))
        .filter(|t| !t.is_empty())
        .collect();
    let url = url_of(pmid);
    let meta = PaperMeta { lang: lang.to_string(), url: url.clone(), published, modified: String::new(), author: authors, site_name: SITE.to_string(), tags };
    Ok(Article::Page { page: Box::new(paper_page(&title, lines, meta)), url })
}

/// The file's XML, through gzip when its name ends in `.gz`.
pub fn open(path: &Path) -> Result<Box<dyn BufRead + Send>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(if path.extension().is_some_and(|e| e == "gz") {
        Box::new(BufReader::with_capacity(1 << 18, MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(1 << 18, file))
    })
}

/// Each PubmedArticle and DeleteCitation of one file, in file order.
pub fn records(path: &Path) -> Result<Records<Box<dyn BufRead + Send>>> {
    Ok(Records::new(open(path)?, RECORDS))
}

/// What one record does to the run, in the order the run applies it.
enum Step {
    Remove(String),
    Count(&'static str),
    Page(Box<Row>),
    Fail(anyhow::Error),
}

fn steps_of(record: &Element, skip: &HashSet<String>, as_of_us: i64) -> Vec<Step> {
    if record.tag != "PubmedArticle" {
        let mut out = Vec::new();
        for pmid in record.iterfind("PMID") {
            let Some(text) = pmid.text() else {
                out.push(Step::Fail(anyhow!("a DeleteCitation PMID has no text")));
                break;
            };
            out.push(Step::Remove(url_of(strip(text))));
            out.push(Step::Count("deleted"));
        }
        return out;
    }
    if skip.contains(record.findtext("MedlineCitation/PMID").unwrap_or("")) {
        return vec![Step::Count("skipped_pmid")];
    }
    vec![match article_page(record) {
        Ok(Article::Page { url, page }) => Step::Page(Box::new(upload_row(&url, *page, as_of_us, CONTENT_TYPE))),
        Ok(Article::Skipped(why)) => Step::Count(why),
        Err(e) => Step::Fail(e),
    }]
}

type Parsed = Result<Vec<Element>>;
type Built = Result<Vec<Vec<Step>>>;

/// Reads one file's records in chunks until the file ends, it fails, or the run stops listening.
fn parse_file(path: &Path, out: SyncSender<Parsed>) {
    let mut found = match records(path) {
        Ok(found) => found,
        Err(e) => {
            let _ = out.send(Err(e));
            return;
        }
    };
    loop {
        let mut chunk = Vec::with_capacity(CHUNK_RECORDS);
        let end = loop {
            if chunk.len() == CHUNK_RECORDS {
                break Ok(false);
            }
            match found.next_record() {
                Ok(Some(record)) => chunk.push(record),
                Ok(None) => break Ok(true),
                Err(e) => break Err(e.context(format!("reading {}", path.display()))),
            }
        };
        if !chunk.is_empty() && out.send(Ok(chunk)).is_err() {
            return;
        }
        match end {
            Ok(false) => {}
            Ok(true) => return,
            Err(e) => {
                let _ = out.send(Err(e));
                return;
            }
        }
    }
}

/// Builds each parsed chunk's pages in parallel, keeping their order.
fn build_pages(parsed: Receiver<Parsed>, skip: &HashSet<String>, as_of_us: i64, out: SyncSender<Built>) {
    for chunk in parsed {
        let built = chunk.map(|records| records.par_iter().map(|r| steps_of(r, skip, as_of_us)).collect());
        if out.send(built).is_err() {
            return;
        }
    }
}

pub fn run(files: &[PathBuf], batches: &mut bulk_core::Batches, as_of_us: i64, skip: &HashSet<String>, limit: u64) -> Result<Counts> {
    let mut counts = Counts::default();
    let mut pages = 0u64;
    std::thread::scope(|scope| -> Result<()> {
        let mut ahead: VecDeque<Receiver<Built>> = VecDeque::new();
        let mut next = files.iter();
        loop {
            while ahead.len() < FILES_AHEAD {
                let Some(path) = next.next() else { break };
                let (parsed_tx, parsed_rx) = sync_channel(2);
                let (built_tx, built_rx) = sync_channel(BUILT_AHEAD);
                scope.spawn(move || parse_file(path, parsed_tx));
                scope.spawn(move || build_pages(parsed_rx, skip, as_of_us, built_tx));
                ahead.push_back(built_rx);
            }
            let Some(file) = ahead.pop_front() else {
                return Ok(());
            };
            for chunk in file {
                for step in chunk?.into_iter().flatten() {
                    match step {
                        Step::Remove(url) => batches.remove(&url),
                        Step::Count(key) => counts.add(key, 1),
                        Step::Page(row) => {
                            batches.add(*row)?;
                            counts.add("pages", 1);
                            pages += 1;
                            if limit > 0 && pages >= limit {
                                return Ok(());
                            }
                        }
                        Step::Fail(e) => return Err(e),
                    }
                }
            }
        }
    })?;
    Ok(counts)
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

/// `set(path.read_text().split())`.
pub fn read_skip(path: &Path) -> Result<HashSet<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(words(&text).map(str::to_string).collect())
}

/// `bulk pubmed`.
pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, "pubmed")?;
    let skip = match &args.skip_pmids {
        Some(path) => read_skip(path)?,
        None => HashSet::new(),
    };
    let counts = run(&args.files, &mut batches, as_of_us, &skip, args.run.limit)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

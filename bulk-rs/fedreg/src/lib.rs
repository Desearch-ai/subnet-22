//! The Federal Register (free to reproduce, 1 CFR 2.6): each document's address from federalregister.gov's API, its text from govinfo's daily XML.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bulk_core::http::USER_AGENT;
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use chrono::{Datelike, NaiveDate, Weekday};
use regex::Regex;
use roxmltree::Node;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SOURCE: &str = "fedreg";
pub const SITE: &str = "Federal Register";
pub const DOCUMENT: &str = "article";
const API: &str = "https://www.federalregister.gov/api/v1/documents.json";
const DONE: &str = "days-done.txt";
const DOCUMENTS: [&str; 4] = ["RULE", "PRORULE", "NOTICE", "PRESDOCU"];
static FRDOC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"FR Doc\.?\s*([A-Z0-9]+-[0-9]+(?:-[0-9]+)?)").expect("a valid pattern"));

/// One document as the API lists it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub document_number: String,
    pub html_url: String,
    pub title: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub publication_date: String,
    #[serde(default)]
    pub agencies: Vec<String>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct IndexArgs {
    /// The folder for the index, a <year>.jsonl a year; an index picks up the days it has not done.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub from: NaiveDate,
    #[arg(long)]
    pub to: NaiveDate,
    /// API requests a second.
    #[arg(long, default_value_t = 1.0)]
    pub per_second: f64,
}

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for the index (index/) and the daily issues (FR-<date>.xml).
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub from: NaiveDate,
    #[arg(long)]
    pub to: NaiveDate,
    /// API requests a second.
    #[arg(long, default_value_t = 1.0)]
    pub per_second: f64,
}

/// The days' index through the API, then each day's issue from govinfo.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    index_main(IndexArgs { out: args.out.join("index"), from: args.from, to: args.to, per_second: args.per_second })?;
    let client = bulk_core::fetch::client()?;
    let mut day = args.from;
    let mut issues = 0;
    while day <= args.to {
        if !matches!(day.weekday(), Weekday::Sat | Weekday::Sun) {
            let url = format!("https://www.govinfo.gov/bulkdata/FR/{}/{:02}/FR-{day}.xml", day.year(), day.month());
            if bulk_core::fetch::save(&client, &url, &args.out.join(format!("FR-{day}.xml")))? {
                issues += 1;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        day = day.succ_opt().context("a day after the last")?;
    }
    println!("{}", serde_json::json!({"issues": issues}));
    Ok(())
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The folder `fedreg-index` filled.
    #[arg(long)]
    pub index: PathBuf,
    /// govinfo's yearly zips, FR-<year>.zip, or daily issues, FR-<date>.xml.
    #[arg(required = true)]
    pub zips: Vec<PathBuf>,
}

fn names(v: &Value) -> Vec<String> {
    v.as_array().into_iter().flatten().filter_map(|a| a.get("name").or_else(|| a.get("raw_name")).and_then(Value::as_str)).map(String::from).collect()
}

fn entry_of(d: &Value) -> Entry {
    let s = |k: &str| d.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    Entry { document_number: s("document_number"), html_url: s("html_url"), title: s("title"), kind: s("type"), publication_date: s("publication_date"), agencies: names(&d["agencies"]) }
}

fn day_url(day: NaiveDate) -> String {
    let fields = ["document_number", "html_url", "title", "type", "publication_date", "agencies"].map(|f| format!("&fields[]={f}")).concat();
    format!("{API}?per_page=1000&order=document_number&conditions[publication_date][is]={day}{fields}")
}

/// Every day from `from` to `to` the Register is published on, Monday to Friday, listed through the API under the pace.
pub fn index_main(args: IndexArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let done: HashSet<String> = std::fs::read_to_string(args.out.join(DONE)).unwrap_or_default().lines().map(String::from).collect();
    let mut done_file = OpenOptions::new().create(true).append(true).open(args.out.join(DONE))?;
    let client = reqwest::blocking::Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(120)).build()?;
    let gap = Duration::from_secs_f64(1.0 / args.per_second.max(0.01));
    let mut last = Instant::now() - gap;
    let (mut days, mut documents) = (0usize, 0usize);
    let mut day = args.from;
    while day <= args.to {
        let key = day.to_string();
        if !matches!(day.weekday(), Weekday::Sat | Weekday::Sun) && !done.contains(&key) {
            let mut url = Some(day_url(day));
            let mut found = Vec::new();
            while let Some(next) = url {
                std::thread::sleep(gap.saturating_sub(last.elapsed()));
                last = Instant::now();
                let body = get_json(&client, &next)?;
                found.extend(body["results"].as_array().into_iter().flatten().map(entry_of));
                url = body.get("next_page_url").and_then(Value::as_str).map(String::from);
            }
            let mut year = OpenOptions::new().create(true).append(true).open(args.out.join(format!("{}.jsonl", day.year())))?;
            for e in &found {
                writeln!(year, "{}", serde_json::to_string(e)?)?;
            }
            writeln!(done_file, "{key}")?;
            days += 1;
            documents += found.len();
            if days % 100 == 0 {
                println!("{}", serde_json::json!({"day": key, "days": days, "documents": documents}));
            }
        }
        day = day.succ_opt().context("a day after the last")?;
    }
    println!("{}", serde_json::json!({"days": days, "documents": documents}));
    Ok(())
}

fn get_json(client: &reqwest::blocking::Client, url: &str) -> Result<Value> {
    for attempt in 0..6u32 {
        match client.get(url).send() {
            Ok(r) if r.status().is_success() => return Ok(serde_json::from_slice(&r.bytes()?)?),
            Ok(r) if attempt == 5 => bail!("the API answered {} for {url}", r.status()),
            Err(e) if attempt == 5 => return Err(e.into()),
            _ => std::thread::sleep(Duration::from_secs(10 * (attempt as u64 + 1))),
        }
    }
    unreachable!("the last attempt returns")
}

/// The index, by document number; a number listed twice keeps its first entry.
pub fn read_index(folder: &Path) -> Result<HashMap<String, Entry>> {
    let mut found = HashMap::new();
    for item in std::fs::read_dir(folder)? {
        let path = item?.path();
        if path.extension().is_some_and(|e| e == "jsonl") {
            for line in BufReader::new(File::open(&path)?).lines() {
                let e: Entry = serde_json::from_str(&line?)?;
                found.entry(e.document_number.clone()).or_insert(e);
            }
        }
    }
    Ok(found)
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn all_text(node: Node) -> String {
    let parts: Vec<&str> = node.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect();
    squash(&parts.join(" ")).replace(" ,", ",").replace(" ;", ";").replace(" .", ".").replace("( ", "(").replace(" )", ")")
}

fn table_rows(table: Node) -> Vec<String> {
    let mut rows = Vec::new();
    let heads: Vec<String> = table.descendants().filter(|d| d.has_tag_name("CHED")).map(all_text).filter(|h| !h.is_empty()).collect();
    if !heads.is_empty() {
        rows.push(heads.join(" | "));
    }
    for row in table.descendants().filter(|d| d.has_tag_name("ROW")) {
        let cells: Vec<String> = row.children().filter(|c| c.has_tag_name("ENT")).map(all_text).collect();
        if cells.iter().any(|c| !c.is_empty()) {
            rows.push(cells.join(" | "));
        }
    }
    rows
}

/// A document's text lines, in order: headings as `##`, paragraphs, tables and signatures; page marks and filing lines left out.
fn lines_of(node: Node, out: &mut Vec<String>) {
    for c in node.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "PRTPAGE" | "FRDOC" | "BILCOD" | "GPH" | "AGENCY" | "SUBAGY" | "SUBJECT" | "CFR" | "DEPDOC" | "RIN" => {}
            "HD" => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(format!("## {}", text.trim_end_matches(':')));
                }
            }
            "GPOTABLE" => out.extend(table_rows(c)),
            "P" | "FP" | "AMDPAR" | "SECTNO" | "SUBJECT_" | "NAME" | "TITLE" | "DATED" | "FTNT" | "PSIG" => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(text);
                }
            }
            _ if c.children().any(|g| g.is_element()) => lines_of(c, out),
            _ => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(text);
                }
            }
        }
    }
}

fn child_texts(node: Node, tag: &str) -> Vec<String> {
    node.descendants().filter(|d| d.has_tag_name(tag)).map(all_text).filter(|t| !t.is_empty()).collect()
}

/// The document number of a document's `FRDOC` line, `[FR Doc. 2026-23060 Filed 11-10-26; 8:45 am]`.
pub fn document_number(document: Node) -> Option<String> {
    let line = document.descendants().find(|d| d.has_tag_name("FRDOC")).map(all_text)?;
    FRDOC.captures(&line).map(|c| c[1].to_string())
}

pub fn page_of(document: Node, entry: &Entry) -> Option<Page> {
    let mut body = Vec::new();
    lines_of(document, &mut body);
    if body.is_empty() {
        return None;
    }
    let subject = child_texts(document, "SUBJECT").join(" ");
    let title = if entry.title.is_empty() { subject } else { entry.title.clone() };
    let mut lines = vec![Line::new(format!(
        "Source: Federal Register ({}, published {}, FR Doc. {}), free to reproduce (1 CFR 2.6). FederalRegister.gov is an unofficial edition; the official one is on govinfo.gov.",
        entry.kind, entry.publication_date, entry.document_number
    ))];
    let mut agencies = child_texts(document, "AGENCY");
    agencies.extend(child_texts(document, "SUBAGY"));
    if agencies.is_empty() {
        agencies = entry.agencies.clone();
    }
    if !agencies.is_empty() {
        lines.push(Line::new(format!("Agencies: {}.", agencies.join("; "))));
    }
    for (label, tag) in [("CFR", "CFR"), ("Docket", "DEPDOC"), ("RIN", "RIN")] {
        let found = child_texts(document, tag);
        if !found.is_empty() {
            lines.push(Line::new(format!("{label}: {}", found.join("; "))));
        }
    }
    lines.extend(body.into_iter().map(Line::new));
    let meta = PaperMeta {
        lang: "en".into(),
        url: entry.html_url.clone(),
        published: entry.publication_date.clone(),
        site_name: SITE.into(),
        tags: entry.agencies.clone(),
        ..PaperMeta::default()
    };
    Some(typed_page(DOCUMENT, &title, lines, meta))
}

/// The pages of one daily issue: each document the index gives an address.
pub fn issue_pages(xml: &str, index: &HashMap<String, Entry>, counts: &mut Counts) -> Result<Vec<(String, Page)>> {
    let options = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = roxmltree::Document::parse_with_options(xml, options).context("parsing a daily issue")?;
    let mut pages = Vec::new();
    for document in doc.descendants().filter(|n| DOCUMENTS.contains(&n.tag_name().name())) {
        let Some(number) = document_number(document) else {
            counts.add("no_document_number", 1);
            continue;
        };
        let Some(entry) = index.get(&number) else {
            counts.add("not_in_index", 1);
            continue;
        };
        match page_of(document, entry) {
            Some(page) => pages.push((entry.html_url.clone(), page)),
            None => counts.add("empty", 1),
        }
    }
    Ok(pages)
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let index = read_index(&args.index)?;
    let mut counts = Counts::default();
    let mut seen = HashSet::new();
    for path in &args.zips {
        let issues: Vec<String> = if path.extension().is_some_and(|e| e == "xml") {
            vec![std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?]
        } else {
            let mut archive = zip::ZipArchive::new(File::open(path).with_context(|| format!("opening {}", path.display()))?)?;
            let mut names: Vec<String> = archive.file_names().filter(|n| n.ends_with(".xml")).map(String::from).collect();
            names.sort();
            let mut found = Vec::new();
            for name in names {
                let mut xml = String::new();
                archive.by_name(&name)?.read_to_string(&mut xml)?;
                found.push(xml);
            }
            found
        };
        for xml in issues {
            counts.add("issues", 1);
            for (url, page) in issue_pages(&xml, &index, &mut counts)? {
                if url.is_empty() || !seen.insert(url.clone()) {
                    counts.add("duplicate", 1);
                    continue;
                }
                batches.add(upload_row(&url, page, as_of_us, "application/xml"))?;
                counts.add("pages", 1);
                if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
                    return Ok(counts);
                }
            }
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

    const XML: &str = r#"<FEDREG><NOTICES><NOTICE><PREAMB><PRTPAGE P="72846"/><AGENCY TYPE="F">DEPARTMENT OF AGRICULTURE</AGENCY><SUBAGY>Agricultural Marketing Service</SUBAGY>
<DEPDOC>[Doc. No. AMS-ST-26-2410]</DEPDOC><SUBJECT>Plant Variety Protection Office: Notice of Request</SUBJECT>
<AGY><HD SOURCE="HED">AGENCY:</HD><P>Agricultural Marketing Service, USDA.</P></AGY>
<SUM><HD SOURCE="HED">SUMMARY:</HD><P>This notice announces an <E T="03">extension</E> of a collection.</P></SUM></PREAMB>
<SUPLINF><HD SOURCE="HED">SUPPLEMENTARY INFORMATION:</HD><GPOTABLE><BOXHD><CHED>Item</CHED><CHED>Hours</CHED></BOXHD><ROW><ENT>Forms</ENT><ENT>30</ENT></ROW></GPOTABLE></SUPLINF>
<SIG><NAME>Erin Morris,</NAME><TITLE>Associate Administrator.</TITLE></SIG>
<FRDOC>[FR Doc. 2026-23060 Filed 11-10-26; 8:45 am]</FRDOC><BILCOD>BILLING CODE 3410-02-P</BILCOD></NOTICE>
<NOTICE><PREAMB><SUBJECT>Unlisted</SUBJECT><P>x</P></PREAMB><FRDOC>[FR Doc. 2026-99999 Filed 11-10-26; 8:45 am]</FRDOC></NOTICE></NOTICES></FEDREG>"#;

    fn index() -> HashMap<String, Entry> {
        let e = Entry {
            document_number: "2026-23060".into(),
            html_url: "https://www.federalregister.gov/documents/2026/11/12/2026-23060/plant-variety-protection-office".into(),
            title: "Plant Variety Protection Office: Notice of Request".into(),
            kind: "Notice".into(),
            publication_date: "2026-11-12".into(),
            agencies: vec!["Agricultural Marketing Service".into()],
        };
        HashMap::from([(e.document_number.clone(), e)])
    }

    #[test]
    fn documents_at_their_api_addresses() {
        let mut counts = Counts::default();
        let pages = issue_pages(XML, &index(), &mut counts).unwrap();
        assert_eq!((pages.len(), counts.get("not_in_index")), (1, 1));
        let (url, page) = &pages[0];
        assert_eq!(url, "https://www.federalregister.gov/documents/2026/11/12/2026-23060/plant-variety-protection-office");
        assert_eq!(
            page.text.lines().skip(1).collect::<Vec<_>>(),
            [
                "Source: Federal Register (Notice, published 2026-11-12, FR Doc. 2026-23060), free to reproduce (1 CFR 2.6). FederalRegister.gov is an unofficial edition; the official one is on govinfo.gov.",
                "Agencies: DEPARTMENT OF AGRICULTURE; Agricultural Marketing Service.",
                "Docket: [Doc. No. AMS-ST-26-2410]",
                "## AGENCY",
                "Agricultural Marketing Service, USDA.",
                "## SUMMARY",
                "This notice announces an extension of a collection.",
                "## SUPPLEMENTARY INFORMATION",
                "Item | Hours",
                "Forms | 30",
                "Erin Morris,",
                "Associate Administrator.",
            ]
        );
        assert_eq!(page.published, "2026-11-12");
    }

    #[test]
    fn document_numbers_and_corrections() {
        let doc = roxmltree::Document::parse("<R><FRDOC>[FR Doc. C1-2026-21030 Filed 11-10-26; 8:45 am]</FRDOC></R>").unwrap();
        assert_eq!(document_number(doc.root_element()).as_deref(), Some("C1-2026-21030"));
    }
}

//! The Electronic Code of Federal Regulations (public domain): a page per section from govinfo's title XML, at the ecfr.gov address its structure gives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use roxmltree::Node;
use serde_json::Value;

pub const SOURCE: &str = "ecfr";
pub const SITE: &str = "eCFR";
pub const LAW: &str = "article";
pub const CURRENT: &str = "https://www.ecfr.gov/current";
const CONTENT_TYPE: &str = "application/xml";
/// govinfo's division types above a section, named in each page's place line.
const ABOVE: [&str; 7] = ["TITLE", "SUBTITLE", "CHAPTER", "SUBCHAP", "PART", "SUBPART", "SUBJGRP"];

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// eCFR's titles.json, for each title's date.
    #[arg(long)]
    pub titles: PathBuf,
    /// A folder of govinfo's ECFR-title<n>.xml and eCFR's structure-<n>.json, one pair a title.
    pub folder: PathBuf,
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn all_text(node: Node) -> String {
    let parts: Vec<&str> = node.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect();
    squash(&parts.join(" ")).replace(" ,", ",").replace(" ;", ";").replace(" .", ".").replace("( ", "(").replace(" )", ")")
}

/// Each section's address under its title, by section number, from eCFR's structure of the title.
pub fn addresses(structure: &Value) -> HashMap<String, String> {
    fn walk(node: &Value, path: &mut Vec<String>, out: &mut HashMap<String, String>) {
        let kind = node.get("type").and_then(Value::as_str).unwrap_or("");
        let id = node.get("identifier").and_then(Value::as_str).unwrap_or("");
        let part = match kind {
            "title" => Some(format!("title-{id}")),
            "subtitle" | "chapter" | "subchapter" | "part" | "subpart" => Some(format!("{kind}-{id}")),
            "subject_group" => Some(format!("subject-group-{id}")),
            "section" => {
                if node.get("reserved").and_then(Value::as_bool) != Some(true) {
                    out.insert(id.to_string(), format!("{CURRENT}/{}/section-{id}", path.join("/")));
                }
                None
            }
            _ => None,
        };
        let pushed = part.map(|p| path.push(p)).is_some();
        for child in node.get("children").and_then(Value::as_array).into_iter().flatten() {
            walk(child, path, out);
        }
        if pushed {
            path.pop();
        }
    }
    let mut out = HashMap::new();
    walk(structure, &mut Vec::new(), &mut out);
    out
}

fn kind(node: Node) -> &'static str {
    let Some(t) = node.attribute("TYPE") else { return "" };
    ABOVE.iter().chain(["SECTION", "APPENDIX"].iter()).find(|k| **k == t).copied().unwrap_or("")
}

fn head(node: Node) -> String {
    node.children().find(|c| c.has_tag_name("HEAD")).map(all_text).unwrap_or_default()
}

/// A section's place: the heads of the divisions above it.
pub fn place(section: Node) -> String {
    let mut heads: Vec<String> = section.ancestors().filter(|a| ABOVE.contains(&kind(*a)) && kind(*a) != "TITLE").map(head).filter(|h| !h.is_empty()).collect();
    heads.reverse();
    heads.join("; ")
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

/// A section's body lines, in order: paragraphs, headings, tables, notes and its citation.
fn body_lines(node: Node, out: &mut Vec<String>) {
    for c in node.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "HEAD" | "PRTPAGE" | "GPH" | "img" | "IMG" => {}
            "HD" | "HED" => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(format!("### {text}"));
                }
            }
            "GPOTABLE" => out.extend(table_rows(c)),
            "EXTRACT" | "NOTE" | "NOTES" | "EXAMPLE" | "APPENDIX" | "DIV" => body_lines(c, out),
            "CITA" => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(format!("Citation: {text}"));
                }
            }
            _ => {
                let text = all_text(c);
                if !text.is_empty() {
                    out.push(text);
                }
            }
        }
    }
}

/// `40 CFR 60.1 Applicability.` from `§ 60.1 Applicability.`
pub fn title_of(cfr_title: &str, head: &str) -> String {
    let rest = head.trim_start_matches('§').trim_start_matches("§§").trim();
    squash(&format!("{cfr_title} CFR {rest}"))
}

pub fn page_of(section: Node, cfr_title: &str, title_name: &str, url: &str, as_of: &str) -> Option<Page> {
    let head = head(section);
    if head.contains("[Reserved]") {
        return None;
    }
    let mut body = Vec::new();
    body_lines(section, &mut body);
    if body.is_empty() {
        return None;
    }
    let mut lines = vec![Line::new(format!(
        "Source: Electronic Code of Federal Regulations, Title {cfr_title} ({title_name}), current as of {as_of} (public domain). Authoritative but unofficial; the Federal Register and the annual CFR are the official editions."
    ))];
    let place = place(section);
    if !place.is_empty() {
        lines.push(Line::new(place));
    }
    lines.extend(body.into_iter().map(Line::new));
    let meta = PaperMeta { lang: "en".into(), url: url.to_string(), modified: as_of.to_string(), site_name: SITE.into(), ..PaperMeta::default() };
    Some(typed_page(LAW, &title_of(cfr_title, &head), lines, meta))
}

/// Section number from govinfo's `N`, `§ 60.1` as `60.1`.
pub fn number(n: &str) -> String {
    n.trim_start_matches('§').trim().to_string()
}

/// The pages of one title: each section its structure gives an address, in document order.
pub fn title_pages(xml: &str, structure: &Value, cfr_title: &str, title_name: &str, as_of: &str, counts: &mut Counts) -> Result<Vec<(String, Page)>> {
    let doc = roxmltree::Document::parse(xml).context("parsing an eCFR title")?;
    let addresses = addresses(structure);
    let mut pages = Vec::new();
    for section in doc.descendants().filter(|n| kind(*n) == "SECTION") {
        let n = number(section.attribute("N").unwrap_or(""));
        let Some(url) = addresses.get(&n) else {
            counts.add("no_address", 1);
            continue;
        };
        match page_of(section, cfr_title, title_name, url, as_of) {
            Some(page) => pages.push((url.clone(), page)),
            None => counts.add("reserved_or_empty", 1),
        }
    }
    Ok(pages)
}

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for titles.json and each title's XML and structure.
    #[arg(long)]
    pub out: PathBuf,
}

/// eCFR's title list, then each title's XML from govinfo and its structure from eCFR at the title's date, a second apart.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let client = bulk_core::fetch::client()?;
    let titles_text = bulk_core::fetch::text(&client, "https://www.ecfr.gov/api/versioner/v1/titles.json")?;
    std::fs::write(args.out.join("titles.json"), &titles_text)?;
    let titles: Value = serde_json::from_str(&titles_text)?;
    let mut saved = 0;
    for t in titles.get("titles").and_then(Value::as_array).into_iter().flatten() {
        if t.get("reserved").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let number = t.get("number").map(|n| n.to_string()).unwrap_or_default();
        let Some(date) = t.get("up_to_date_as_of").and_then(Value::as_str) else {
            anyhow::bail!("title {number} has no up_to_date_as_of; a title left out would hide all its sections");
        };
        let xml = format!("https://www.govinfo.gov/bulkdata/ECFR/title-{number}/ECFR-title{number}.xml");
        let structure = format!("https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-{number}.json");
        for (url, name) in [(xml, format!("ECFR-title{number}.xml")), (structure, format!("structure-{number}.json"))] {
            if !bulk_core::fetch::save(&client, &url, &args.out.join(name))? {
                anyhow::bail!("{url} is not there; a title left out would hide all its sections");
            }
            saved += 1;
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    println!("{}", serde_json::json!({"files": saved}));
    Ok(())
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let titles: Value = serde_json::from_str(&read(&args.titles)?)?;
    let mut counts = Counts::default();
    let mut seen = std::collections::HashSet::new();
    for t in titles.get("titles").and_then(Value::as_array).into_iter().flatten() {
        let number = t.get("number").map(|n| n.to_string()).unwrap_or_default();
        let (xml, structure) = (args.folder.join(format!("ECFR-title{number}.xml")), args.folder.join(format!("structure-{number}.json")));
        if t.get("reserved").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if !xml.exists() || !structure.exists() {
            anyhow::bail!("title {number} is missing from {}; a title left out would hide all its sections", args.folder.display());
        }
        let name = t.get("name").and_then(Value::as_str).unwrap_or("");
        let as_of = t.get("up_to_date_as_of").and_then(Value::as_str).unwrap_or("");
        let structure: Value = serde_json::from_str(&read(&structure)?)?;
        for (url, page) in title_pages(&read(&xml)?, &structure, &number, name, as_of, &mut counts)? {
            if !seen.insert(url.clone()) {
                counts.add("duplicate", 1);
                continue;
            }
            batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
            counts.add("pages", 1);
            if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
                return Ok(counts);
            }
        }
        counts.add("titles", 1);
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

    const XML: &str = r#"<DLPSTEXTCLASS><TEXT><BODY><ECFRBRWS><DIV1 N="14" TYPE="TITLE"><HEAD>Title 14—Aeronautics and Space</HEAD>
<DIV3 N="I" TYPE="CHAPTER"><HEAD>CHAPTER I—FEDERAL AVIATION ADMINISTRATION</HEAD>
<DIV4 N="C" TYPE="SUBCHAP"><HEAD>SUBCHAPTER C—AIRCRAFT</HEAD>
<DIV5 N="25" TYPE="PART"><HEAD>PART 25—AIRWORTHINESS STANDARDS</HEAD>
<DIV6 N="B" TYPE="SUBPART"><HEAD>Subpart B—Flight</HEAD>
<DIV7 N="General" TYPE="SUBJGRP"><HEAD>General</HEAD>
<DIV8 N="§ 25.21" TYPE="SECTION"><HEAD>§ 25.21 Proof of compliance.</HEAD>
<P>(a) Each requirement of this subpart must be met.</P>
<GPOTABLE><BOXHD><CHED>Weight</CHED><CHED>Limit</CHED></BOXHD><ROW><ENT>Max</ENT><ENT>10 percent</ENT></ROW></GPOTABLE>
<CITA>[Doc. No. 5066, 29 FR 18291, Dec. 24, 1964]</CITA></DIV8>
<DIV8 N="§ 25.22" TYPE="SECTION"><HEAD>§ 25.22 [Reserved]</HEAD></DIV8>
</DIV7></DIV6></DIV5></DIV4></DIV3></DIV1></ECFRBRWS></BODY></TEXT></DLPSTEXTCLASS>"#;

    fn structure() -> Value {
        json!({"type": "title", "identifier": "14", "children": [{"type": "chapter", "identifier": "I", "children": [{"type": "subchapter", "identifier": "C", "children": [
            {"type": "part", "identifier": "25", "children": [{"type": "subpart", "identifier": "B", "children": [{"type": "subject_group", "identifier": "ECFR557993314ee6275", "children": [
                {"type": "section", "identifier": "25.21"}, {"type": "section", "identifier": "25.22", "reserved": true}]}]}]}]}]}]})
    }

    #[test]
    fn sections_at_their_full_ecfr_addresses() {
        let a = addresses(&structure());
        assert_eq!(a["25.21"], "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-B/subject-group-ECFR557993314ee6275/section-25.21");
        assert!(!a.contains_key("25.22"));
    }

    #[test]
    fn section_page_with_place_table_and_citation() {
        let mut counts = Counts::default();
        let pages = title_pages(XML, &structure(), "14", "Aeronautics and Space", "2026-11-10", &mut counts).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(counts.get("no_address"), 1);
        let page = &pages[0].1;
        assert_eq!(page.title, "14 CFR 25.21 Proof of compliance.");
        assert_eq!(
            page.text.lines().skip(2).collect::<Vec<_>>(),
            [
                "CHAPTER I—FEDERAL AVIATION ADMINISTRATION; SUBCHAPTER C—AIRCRAFT; PART 25—AIRWORTHINESS STANDARDS; Subpart B—Flight; General",
                "(a) Each requirement of this subpart must be met.",
                "Weight | Limit",
                "Max | 10 percent",
                "Citation: [Doc. No. 5066, 29 FR 18291, Dec. 24, 1964]",
            ]
        );
        assert_eq!(page.modified, "2026-11-10");
    }
}

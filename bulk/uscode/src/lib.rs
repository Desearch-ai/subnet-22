//! The United States Code (public domain): a page per section from an OLRC release point's USLM XML, at its uscode.house.gov address.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use roxmltree::Node;

pub const SOURCE: &str = "uscode";
pub const SITE: &str = "United States Code";
pub const LAW: &str = "article";
pub const HOST: &str = "https://uscode.house.gov";
const CONTENT_TYPE: &str = "application/xml";
/// The levels below a section, each a line of its number, heading and text.
const LEVELS: [&str; 8] = ["subsection", "paragraph", "subparagraph", "clause", "subclause", "item", "subitem", "subsubitem"];
/// The levels above a section, named in each page's place line.
const ABOVE: [&str; 6] = ["title", "subtitle", "chapter", "subchapter", "part", "subpart"];

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The release point's public law, as in 119-138, for the credit line; read from the zip's name when not given.
    #[arg(long = "release-point", default_value = "")]
    pub release_point: String,
    /// The release point's all-titles zip, xml_uscAll@<release point>.zip.
    pub zip: PathBuf,
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every text below a node, in order, with spaces between elements.
pub fn all_text(node: Node) -> String {
    let parts: Vec<&str> = node.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect();
    squash(&parts.join(" ")).replace(" ,", ",").replace(" ;", ";").replace(" .", ".").replace("( ", "(").replace(" )", ")")
}

fn child<'a, 'i>(node: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|c| c.has_tag_name(tag))
}

fn child_text(node: Node, tag: &str) -> String {
    child(node, tag).map(all_text).unwrap_or_default()
}

fn table_rows(table: Node) -> Vec<String> {
    table
        .descendants()
        .filter(|d| d.has_tag_name("tr"))
        .map(|tr| tr.children().filter(|c| c.has_tag_name("td") || c.has_tag_name("th")).map(all_text).collect::<Vec<_>>().join(" | "))
        .filter(|row| !row.trim_matches(|c| c == '|' || c == ' ').is_empty())
        .collect()
}

/// A text block's lines: each `<p>` or table row its own line, else the whole text.
fn block_lines(node: Node) -> Vec<String> {
    if node.children().any(|c| c.has_tag_name("p") || c.has_tag_name("table")) {
        let mut lines = Vec::new();
        for c in node.children().filter(|c| c.is_element()) {
            if c.has_tag_name("table") {
                lines.extend(table_rows(c));
            } else {
                lines.push(all_text(c));
            }
        }
        lines.into_iter().filter(|l| !l.is_empty()).collect()
    } else {
        let text = all_text(node);
        if text.is_empty() {
            Vec::new()
        } else {
            vec![text]
        }
    }
}

/// A level's lines: its number, heading and opening text on one line, then its parts, then any closing text.
fn level_lines(node: Node, out: &mut Vec<String>) {
    let mut head: Vec<String> = [child_text(node, "num"), child_text(node, "heading")].into_iter().filter(|t| !t.is_empty()).collect();
    let mut first_done = false;
    for c in node.children().filter(|c| c.is_element()) {
        let tag = c.tag_name().name();
        if tag == "num" || tag == "heading" {
            continue;
        }
        if LEVELS.contains(&tag) {
            if !first_done {
                push_head(&mut head, out);
                first_done = true;
            }
            level_lines(c, out);
        } else if tag == "chapeau" || tag == "content" || tag == "continuation" || tag == "p" {
            let mut lines = block_lines(c);
            if !first_done && !lines.is_empty() {
                head.push(lines.remove(0));
                push_head(&mut head, out);
                first_done = true;
            }
            out.extend(lines);
        } else if tag == "table" {
            out.extend(table_rows(c));
        }
    }
    if !first_done {
        push_head(&mut head, out);
    }
}

fn push_head(head: &mut Vec<String>, out: &mut Vec<String>) {
    if !head.is_empty() {
        out.push(head.join(" "));
        head.clear();
    }
}

/// A section's place in its title: each level above it as its number and heading.
pub fn place(section: Node) -> String {
    let mut levels: Vec<String> = section
        .ancestors()
        .filter(|a| ABOVE.contains(&a.tag_name().name()))
        .map(|a| squash(&format!("{} {}", child_text(a, "num"), child_text(a, "heading"))))
        .filter(|l| !l.is_empty())
        .collect();
    levels.reverse();
    levels.join("; ")
}

/// One section of one title, `/us/usc/t<title>/s<section>`: not a range of several, nor an appendix's public-law section.
pub fn is_section(identifier: &str) -> bool {
    let Some(rest) = identifier.strip_prefix("/us/usc/t") else { return false };
    let mut parts = rest.split('/');
    let (title, section, more) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next());
    more.is_none()
        && !title.is_empty()
        && title.chars().all(|c| c.is_ascii_alphanumeric())
        && section.len() > 1
        && section.starts_with('s')
        && !section.chars().any(char::is_whitespace)
}

/// The page address of a section identifier, `/us/usc/t17/s107` at `https://uscode.house.gov/us/usc/t17/s107/`.
pub fn url_of(identifier: &str) -> String {
    format!("{HOST}{identifier}/")
}

/// `17 U.S.C. § 107` for `/us/usc/t17/s107`.
pub fn citation(identifier: &str) -> String {
    let mut parts = identifier.trim_start_matches("/us/usc/").split('/');
    let title = parts.next().unwrap_or("").trim_start_matches('t');
    let section = parts.next().unwrap_or("").trim_start_matches('s');
    format!("{title} U.S.C. § {section}")
}

pub fn page_of(section: Node, release_point: &str) -> Option<(String, Page)> {
    let identifier = section.attribute("identifier")?;
    if !is_section(identifier) {
        return None;
    }
    let heading = child_text(section, "heading");
    let title = squash(&format!("{} {heading}", citation(identifier)));
    let mut lines = vec![Line::new(format!(
        "Source: United States Code, Office of the Law Revision Counsel, U.S. House of Representatives, current through Public Law {release_point} (public domain)."
    ))];
    let place = place(section);
    if !place.is_empty() {
        lines.push(Line::new(place));
    }
    let mut body = Vec::new();
    for c in section.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "num" | "heading" | "notes" | "sourceCredit" | "toc" | "layout" => {}
            tag if LEVELS.contains(&tag) => level_lines(c, &mut body),
            "table" => body.extend(table_rows(c)),
            _ => body.extend(block_lines(c)),
        }
    }
    if body.is_empty() {
        return None;
    }
    lines.extend(body.into_iter().map(Line::new));
    let credit = child_text(section, "sourceCredit");
    if !credit.is_empty() {
        lines.push(Line::new(format!("Enacted and amended: {credit}")));
    }
    if let Some(notes) = child(section, "notes") {
        let mut noted = Vec::new();
        for note in notes.children().filter(|c| c.has_tag_name("note")) {
            let heading = child_text(note, "heading");
            let text: Vec<String> = note.children().filter(|c| c.is_element() && !c.has_tag_name("heading")).flat_map(block_lines).collect();
            if !text.is_empty() {
                if !heading.is_empty() {
                    noted.push(format!("### {heading}"));
                }
                noted.extend(text);
            }
        }
        if !noted.is_empty() {
            lines.push(Line::new("## Notes"));
            lines.extend(noted.into_iter().map(Line::new));
        }
    }
    let url = url_of(identifier);
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), site_name: SITE.into(), ..PaperMeta::default() };
    Some((url, typed_page(LAW, &title, lines, meta)))
}

/// Each section of one title's USLM document, in document order.
pub fn sections(xml: &str, release_point: &str) -> Result<Vec<(String, Page)>> {
    let doc = roxmltree::Document::parse(xml).context("parsing a USLM title")?;
    Ok(doc.descendants().filter(|n| n.has_tag_name("section") && n.attribute("identifier").is_some()).filter_map(|s| page_of(s, release_point)).collect())
}

/// The release point in a zip's name, `xml_uscAll@119-138.zip` as `119-138`.
pub fn release_point_of(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    Some(name.split_once('@')?.1.strip_suffix(".zip")?.to_string())
}

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for the current release point's all-titles zip.
    #[arg(long)]
    pub out: PathBuf,
}

/// The all-titles zip of the release point uscode.house.gov's download page names.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let client = bulk_core::fetch::client()?;
    let page = bulk_core::fetch::text(&client, &format!("{HOST}/download.html"))?;
    let pattern = regex::Regex::new(r#"href="([^"]*xml_uscAll@([0-9A-Za-z-]+)\.zip)""#).expect("a valid pattern");
    let found = pattern.captures(&page).context("no all-titles zip on the download page")?;
    let href = &found[1];
    let url = if href.starts_with("http") { href.to_string() } else { format!("{HOST}/{}", href.trim_start_matches("./").trim_start_matches('/')) };
    let to = args.out.join(format!("xml_uscAll@{}.zip", &found[2]));
    if !bulk_core::fetch::save(&client, &url, &to)? {
        anyhow::bail!("{url} is not there");
    }
    println!("{}", serde_json::json!({"release_point": &found[2]}));
    Ok(())
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let release_point = if args.release_point.is_empty() { release_point_of(&args.zip).unwrap_or_default() } else { args.release_point.clone() };
    let file = std::fs::File::open(&args.zip).with_context(|| format!("opening {}", args.zip.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut names: Vec<String> = archive.file_names().filter(|n| n.ends_with(".xml")).map(String::from).collect();
    names.sort();
    let mut counts = Counts::default();
    let mut seen = std::collections::HashSet::new();
    for name in names {
        let mut xml = String::new();
        archive.by_name(&name)?.read_to_string(&mut xml)?;
        counts.add("titles", 1);
        for (url, page) in sections(&xml, &release_point)? {
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

    const XML: &str = r#"<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0"><main><title identifier="/us/usc/t17"><num value="17">Title 17—</num><heading>COPYRIGHTS</heading>
<chapter identifier="/us/usc/t17/ch1"><num value="1">CHAPTER 1—</num><heading>SUBJECT MATTER AND SCOPE OF COPYRIGHT</heading>
<section identifier="/us/usc/t17/s107"><num value="107">§ 107.</num><heading> Limitations on exclusive rights: Fair use</heading>
<chapeau>Notwithstanding the provisions of sections 106 and 106A, the fair use is not an infringement of copyright. The factors shall include—</chapeau>
<paragraph identifier="/us/usc/t17/s107/1"><num value="1">(1)</num><content> the purpose and character of the use;</content></paragraph>
<paragraph identifier="/us/usc/t17/s107/2"><num value="2">(2)</num><content> the nature of the copyrighted work.</content></paragraph>
<continuation>The fact that a work is unpublished shall not itself bar a finding of fair use.</continuation>
<sourceCredit>(<ref href="/us/pl/94/553/tI/s101">Pub. L. 94–553, title I, § 101</ref>, <date date="1976-10-19">Oct. 19, 1976</date>.)</sourceCredit>
<notes type="uscNote"><note topic="amendments"><heading>Amendments</heading><p><b>1992</b>—Pub. L. 102–492 inserted a sentence.</p></note></notes>
</section>
<section identifier="/us/usc/t17/s108"><num value="108">§ 108.</num><heading> Limitations on exclusive rights: Reproduction by libraries and archives</heading>
<subsection identifier="/us/usc/t17/s108/a"><num value="a">(a)</num><chapeau>Except as otherwise provided—</chapeau>
<paragraph identifier="/us/usc/t17/s108/a/1"><num value="1">(1)</num><content>the reproduction is made without commercial advantage;</content></paragraph></subsection>
</section></chapter></title></main></uscDoc>"#;

    #[test]
    fn sections_as_pages_at_their_identifiers() {
        let pages = sections(XML, "119-138").unwrap();
        assert_eq!(pages.len(), 2);
        let (url, page) = &pages[0];
        assert_eq!(url, "https://uscode.house.gov/us/usc/t17/s107/");
        assert_eq!(page.title, "17 U.S.C. § 107 Limitations on exclusive rights: Fair use");
        assert_eq!(
            page.text.lines().collect::<Vec<_>>(),
            [
                "17 U.S.C. § 107 Limitations on exclusive rights: Fair use",
                "Source: United States Code, Office of the Law Revision Counsel, U.S. House of Representatives, current through Public Law 119-138 (public domain).",
                "Title 17— COPYRIGHTS; CHAPTER 1— SUBJECT MATTER AND SCOPE OF COPYRIGHT",
                "Notwithstanding the provisions of sections 106 and 106A, the fair use is not an infringement of copyright. The factors shall include—",
                "(1) the purpose and character of the use;",
                "(2) the nature of the copyrighted work.",
                "The fact that a work is unpublished shall not itself bar a finding of fair use.",
                "Enacted and amended: (Pub. L. 94–553, title I, § 101, Oct. 19, 1976.)",
                "## Notes",
                "### Amendments",
                "1992 —Pub. L. 102–492 inserted a sentence.",
            ]
        );
        let lines: Vec<&str> = pages[1].1.text.lines().collect();
        assert_eq!(lines[3..], ["(a) Except as otherwise provided—", "(1) the reproduction is made without commercial advantage;"]);
    }

    #[test]
    fn single_sections_only() {
        assert!(is_section("/us/usc/t17/s107") && is_section("/us/usc/t42/s1395w-4") && is_section("/us/usc/t5a/s1"));
        assert!(!is_section("/us/usc/t2/s3 /us/usc/t2/s4") && !is_section("/us/usc/t5a/pl/95/452/s1") && !is_section("/us/usc/t17") && !is_section("/us/usc/t17/s"));
    }

    #[test]
    fn release_points_of_zip_names() {
        assert_eq!(release_point_of(std::path::Path::new("/x/xml_uscAll@119-138.zip")).as_deref(), Some("119-138"));
    }

    #[test]
    fn citations_of_identifiers() {
        assert_eq!(citation("/us/usc/t42/s1395w-4"), "42 U.S.C. § 1395w-4");
        assert_eq!(url_of("/us/usc/t5a/s1"), "https://uscode.house.gov/us/usc/t5a/s1/");
    }
}

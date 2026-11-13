//! DailyMed (FDA labeling hosted by NLM): a page per human prescription or OTC drug label from the full release zips, the label's text without its images.

use std::io::{Cursor, Read};
use std::path::PathBuf;

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use roxmltree::Node;

pub const SOURCE: &str = "dailymed";
pub const SITE: &str = "DailyMed";
pub const LABEL: &str = "article";
const CONTENT_TYPE: &str = "application/xml";
/// The label kinds taken, by their document's LOINC code.
const KINDS: [(&str, &str); 3] = [("34391-3", "human prescription drug label"), ("45129-4", "human prescription drug label"), ("34390-5", "human OTC drug label")];
/// The SPL product data section, left out: its facts are in the product line.
const PRODUCT_DATA: &str = "48780-1";

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// DailyMed full release zips, dm_spl_release_human_rx_part<n>.zip or _otc_.
    #[arg(required = true)]
    pub zips: Vec<PathBuf>,
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every text below a node, a `<br>` as a space.
pub fn all_text(node: Node) -> String {
    let parts: Vec<&str> = node.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect();
    squash(&parts.join(" ")).replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace("( ", "(").replace(" )", ")").replace(" ®", "®").replace(" ™", "™")
}

fn child<'a, 'i>(node: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|c| c.has_tag_name(tag))
}

fn text_lines(text: Node, out: &mut Vec<String>) {
    for c in text.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "paragraph" => {
                let t = all_text(c);
                if !t.is_empty() {
                    out.push(t);
                }
            }
            "list" => {
                for item in c.children().filter(|i| i.has_tag_name("item")) {
                    let t = all_text(item);
                    if !t.is_empty() {
                        out.push(format!("- {t}"));
                    }
                }
            }
            "table" => {
                for tr in c.descendants().filter(|d| d.has_tag_name("tr")) {
                    let cells: Vec<String> = tr.children().filter(|x| x.has_tag_name("td") || x.has_tag_name("th")).map(all_text).collect();
                    if cells.iter().any(|x| !x.is_empty()) {
                        out.push(cells.join(" | "));
                    }
                }
            }
            "renderMultiMedia" => {}
            _ => {
                let t = all_text(c);
                if !t.is_empty() {
                    out.push(t);
                }
            }
        }
    }
}

/// A section's lines: its title as a heading of its depth, its text, then its subsections.
fn section_lines(section: Node, depth: usize, out: &mut Vec<String>) {
    if child(section, "code").and_then(|c| c.attribute("code")) == Some(PRODUCT_DATA) || child(section, "subject").is_some() {
        return;
    }
    let title = child(section, "title").map(all_text).unwrap_or_default();
    if !title.is_empty() {
        out.push(format!("{} {title}", "#".repeat(depth.clamp(2, 4))));
    }
    if let Some(text) = child(section, "text") {
        text_lines(text, out);
    }
    if let Some(excerpt) = child(section, "excerpt") {
        for highlight in excerpt.descendants().filter(|d| d.has_tag_name("text")) {
            text_lines(highlight, out);
        }
    }
    for component in section.children().filter(|c| c.has_tag_name("component")) {
        for sub in component.children().filter(|c| c.has_tag_name("section")) {
            section_lines(sub, depth + 1, out);
        }
    }
}

/// The label's products: each name with its active ingredients and form.
fn products(doc: &roxmltree::Document) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for product in doc.descendants().filter(|d| d.has_tag_name("manufacturedProduct") && d.parent().is_some_and(|p| p.has_tag_name("subject"))) {
        let Some(medicine) = product.children().find(|c| c.is_element() && c.tag_name().name().starts_with("manufactured")) else { continue };
        let name = child(medicine, "name").map(all_text).unwrap_or_default();
        let form = child(medicine, "formCode").and_then(|f| f.attribute("displayName")).unwrap_or("").to_lowercase();
        let actives: Vec<String> = medicine
            .descendants()
            .filter(|d| d.has_tag_name("activeIngredientSubstance") || d.has_tag_name("ingredientSubstance") && d.parent().is_some_and(|p| p.attribute("classCode") == Some("ACTIB") || p.attribute("classCode") == Some("ACTIM")))
            .filter_map(|s| child(s, "name").map(all_text))
            .filter(|n| !n.is_empty())
            .collect();
        let mut line = name;
        if !actives.is_empty() {
            line.push_str(&format!(" ({})", actives.join(", ").to_lowercase()));
        }
        if !form.is_empty() {
            line.push_str(&format!(", {form}"));
        }
        if !line.trim().is_empty() && !found.contains(&line) {
            found.push(line);
        }
    }
    found
}

fn yyyymmdd(value: &str) -> String {
    if value.len() >= 8 && value[..8].chars().all(|c| c.is_ascii_digit()) {
        format!("{}-{}-{}", &value[..4], &value[4..6], &value[6..8])
    } else {
        String::new()
    }
}

/// A label's page, or None when it is not a human prescription or OTC label.
pub fn page_of(xml: &str, listed_set_id: Option<&str>) -> Result<Option<(String, Page)>> {
    let doc = roxmltree::Document::parse(xml).context("parsing a label")?;
    let root = doc.root_element();
    let code = child(root, "code").and_then(|c| c.attribute("code")).unwrap_or("");
    let Some((_, kind)) = KINDS.iter().find(|(c, _)| *c == code) else { return Ok(None) };
    let Some(set_id) = listed_set_id.or_else(|| child(root, "setId").and_then(|s| s.attribute("root"))).filter(|id| is_set_id(id)) else { return Ok(None) };
    let set_id = set_id.to_lowercase();
    let products = products(&doc);
    let title = child(root, "title").map(all_text).filter(|t| !t.is_empty()).or_else(|| products.first().cloned()).unwrap_or_else(|| set_id.clone());
    let version = child(root, "versionNumber").and_then(|v| v.attribute("value")).unwrap_or("");
    let effective = yyyymmdd(child(root, "effectiveTime").and_then(|e| e.attribute("value")).unwrap_or(""));
    let labeler = root.descendants().find(|d| d.has_tag_name("representedOrganization")).and_then(|o| child(o, "name")).map(all_text).unwrap_or_default().trim_end_matches('.').to_string();
    let mut lines = vec![Line::new(format!(
        "Source: DailyMed, U.S. National Library of Medicine; {} submitted to the FDA{}. Label version {version}, effective {effective}.",
        kind,
        if labeler.is_empty() { String::new() } else { format!(" by {labeler}") }
    ))];
    lines.push(Line::new(format!("Set ID: {set_id}")));
    if !products.is_empty() {
        lines.push(Line::new(format!("Products: {}.", products.join("; "))));
    }
    let mut body = Vec::new();
    if let Some(structured) = root.descendants().find(|d| d.has_tag_name("structuredBody")) {
        for component in structured.children().filter(|c| c.has_tag_name("component")) {
            for section in component.children().filter(|c| c.has_tag_name("section")) {
                section_lines(section, 2, &mut body);
            }
        }
    }
    if body.iter().all(|line| line.starts_with('#')) {
        return Ok(None);
    }
    lines.extend(body.into_iter().map(Line::new));
    let url = format!("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}");
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), modified: effective.clone(), published: effective, author: labeler, site_name: SITE.into(), ..PaperMeta::default() };
    Ok(Some((url, typed_page(LABEL, &title, lines, meta))))
}

/// The set ID in a release entry's name, `prescription/20060131_<set id>.zip`, as DailyMed lists the label.
pub fn listed_set_id(name: &str) -> Option<&str> {
    let stem = name.rsplit('/').next()?.strip_suffix(".zip")?;
    let (_, id) = stem.split_once('_')?;
    is_set_id(id).then_some(id)
}

/// A set ID as DailyMed's addresses take it, a 36-character UUID.
pub fn is_set_id(id: &str) -> bool {
    id.len() == 36 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let mut counts = Counts::default();
    let mut seen = std::collections::HashSet::new();
    for path in &args.zips {
        let mut outer = zip::ZipArchive::new(std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?)?;
        let mut names: Vec<String> = outer.file_names().filter(|n| n.ends_with(".zip")).map(String::from).collect();
        // Entry names start with the version's date, so newest first keeps each set's latest label.
        names.sort_by(|a, b| b.cmp(a));
        for name in names {
            let mut bytes = Vec::new();
            outer.by_name(&name)?.read_to_end(&mut bytes)?;
            let mut inner = zip::ZipArchive::new(Cursor::new(bytes)).with_context(|| format!("opening {name}"))?;
            let Some(xml_name) = inner.file_names().find(|n| n.ends_with(".xml")).map(String::from) else {
                counts.add("no_xml", 1);
                continue;
            };
            let mut xml = String::new();
            inner.by_name(&xml_name)?.read_to_string(&mut xml)?;
            match page_of(&xml, listed_set_id(&name)) {
                Ok(Some((url, page))) => {
                    if !seen.insert(url.clone()) {
                        counts.add("duplicate", 1);
                        continue;
                    }
                    batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
                    counts.add("pages", 1);
                }
                Ok(None) => counts.add("other_kind_or_empty", 1),
                Err(_) => counts.add("unreadable", 1),
            }
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

    const XML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<document xmlns="urn:hl7-org:v3"><code code="34391-3" displayName="HUMAN PRESCRIPTION DRUG LABEL"/>
<title>Halotestin<sup>&#174;</sup> CIII<br/>fluoxymesterone<br/>tablets, USP</title>
<effectiveTime value="20060206"/><setId root="09BAFC2D-1893-4618-86DC-E9403407CD41"/><versionNumber value="1"/>
<author><assignedEntity><representedOrganization><name>Pharmacia and Upjohn Company</name></representedOrganization></assignedEntity></author>
<component><structuredBody>
<component><section><code code="48780-1"/><subject><manufacturedProduct><manufacturedMedicine><name>Halotestin</name><formCode displayName="TABLET"/>
<activeIngredient><activeIngredientSubstance><name>FLUOXYMESTERONE</name></activeIngredientSubstance></activeIngredient></manufacturedMedicine></manufacturedProduct></subject></section></component>
<component><section><code code="34067-9"/><title>INDICATIONS AND USAGE</title><text><paragraph>Males: replacement therapy.</paragraph>
<list><item>Primary hypogonadism</item></list></text>
<component><section><title>Females</title><text><paragraph>Secondary breast cancer.</paragraph><renderMultiMedia referencedObject="MM1"/></text></section></component></section></component>
<component><section><title>HOW SUPPLIED</title><text><table><tr><td>Strength</td><td>NDC</td></tr><tr><td>2 mg</td><td>0009-0014</td></tr></table></text></section></component>
</structuredBody></component></document>"#;

    #[test]
    fn label_page_with_sections_products_and_no_images() {
        let (url, page) = page_of(XML, None).unwrap().unwrap();
        assert_eq!(url, "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=09bafc2d-1893-4618-86dc-e9403407cd41");
        assert_eq!(page.title, "Halotestin® CIII fluoxymesterone tablets, USP");
        assert_eq!(
            page.text.lines().skip(1).collect::<Vec<_>>(),
            [
                "Source: DailyMed, U.S. National Library of Medicine; human prescription drug label submitted to the FDA by Pharmacia and Upjohn Company. Label version 1, effective 2006-02-06.",
                "Set ID: 09bafc2d-1893-4618-86dc-e9403407cd41",
                "Products: Halotestin (fluoxymesterone), tablet.",
                "## INDICATIONS AND USAGE",
                "Males: replacement therapy.",
                "- Primary hypogonadism",
                "### Females",
                "Secondary breast cancer.",
                "## HOW SUPPLIED",
                "Strength | NDC",
                "2 mg | 0009-0014",
            ]
        );
        assert_eq!(page.author, "Pharmacia and Upjohn Company");
    }

    #[test]
    fn other_label_kinds_left_out() {
        let xml = XML.replace("34391-3", "53410-8");
        assert!(page_of(&xml, None).unwrap().is_none());
        assert_eq!(listed_set_id("prescription/20060131_ABD6ECF0-DC8E-41DE-89F2-1E36ED9D6535.zip"), Some("ABD6ECF0-DC8E-41DE-89F2-1E36ED9D6535"));
        assert_eq!(listed_set_id("prescription/20060131_short.zip"), None);
    }
}

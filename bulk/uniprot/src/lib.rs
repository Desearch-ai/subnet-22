//! UniProtKB/Swiss-Prot (CC BY 4.0): a page per reviewed protein entry from the release's XML, credited to the UniProt Consortium.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use flate2::read::MultiGzDecoder;
use roxmltree::Node;

pub const SOURCE: &str = "uniprot";
pub const SITE: &str = "UniProt";
pub const ENTRY: &str = "article";
const CONTENT_TYPE: &str = "application/xml";
/// Comment types written as sections, in this order, with their headings.
const SECTIONS: [(&str, &str); 11] = [
    ("function", "Function"),
    ("catalytic activity", "Catalytic activity"),
    ("pathway", "Pathway"),
    ("subunit", "Subunit"),
    ("subcellular location", "Subcellular location"),
    ("tissue specificity", "Tissue specificity"),
    ("developmental stage", "Developmental stage"),
    ("induction", "Induction"),
    ("disease", "Involvement in disease"),
    ("PTM", "Post-translational modification"),
    ("similarity", "Family"),
];

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The release, as in 2026_04, for the credit line; read from reldate.txt beside the XML when not given.
    #[arg(long, default_value = "")]
    pub release: String,
    /// First entry to take, by position in the file.
    #[arg(long, default_value_t = 0)]
    pub from: usize,
    /// Position to stop before; every entry after `from` when 0.
    #[arg(long, default_value_t = 0)]
    pub to: usize,
    /// uniprot_sprot.xml.gz.
    pub xml: PathBuf,
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn child<'a, 'i>(node: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|c| c.has_tag_name(tag))
}

fn text_of(node: Option<Node>) -> String {
    node.map(|n| squash(&n.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect::<Vec<_>>().join(" "))).unwrap_or_default()
}

fn names(node: Node, tag: &str) -> Vec<String> {
    node.children().filter(|c| c.has_tag_name(tag)).map(|n| text_of(child(n, "fullName"))).filter(|n| !n.is_empty()).collect()
}

/// A comment's lines: its texts, a reaction's text, a disease's name and description, or its subcellular locations.
fn comment_lines(comment: Node) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(disease) = child(comment, "disease") {
        let name = text_of(child(disease, "name"));
        let acronym = text_of(child(disease, "acronym"));
        let description = text_of(child(disease, "description"));
        let head = if acronym.is_empty() { name } else { format!("{name} ({acronym})") };
        lines.push(if description.is_empty() { head } else { format!("{head}: {description}") });
    }
    if let Some(reaction) = child(comment, "reaction") {
        lines.push(text_of(child(reaction, "text")));
    }
    for location in comment.children().filter(|c| c.has_tag_name("subcellularLocation")) {
        let parts: Vec<String> = location.children().filter(|c| c.has_tag_name("location") || c.has_tag_name("topology")).map(|c| text_of(Some(c))).filter(|t| !t.is_empty()).collect();
        if !parts.is_empty() {
            lines.push(parts.join(", "));
        }
    }
    for text in comment.children().filter(|c| c.has_tag_name("text")) {
        lines.push(text_of(Some(text)));
    }
    lines.into_iter().filter(|l| !l.is_empty()).collect()
}

pub fn page_of(entry: Node, release: &str) -> Option<(String, Page)> {
    let accession = text_of(child(entry, "accession"));
    if accession.is_empty() {
        return None;
    }
    let protein = child(entry, "protein");
    let recommended = protein.and_then(|p| child(p, "recommendedName")).map(|r| text_of(child(r, "fullName"))).unwrap_or_default();
    let submitted = protein.map(|p| names(p, "submittedName")).unwrap_or_default();
    let name = if recommended.is_empty() { submitted.first().cloned().unwrap_or_else(|| accession.clone()) } else { recommended };
    let gene = child(entry, "gene").and_then(|g| g.children().find(|n| n.has_tag_name("name") && n.attribute("type") == Some("primary")).or_else(|| child(g, "name"))).map(|n| text_of(Some(n))).unwrap_or_default();
    let organism = child(entry, "organism");
    let organism_name = |kind: &str| organism.and_then(|o| o.children().find(|n| n.has_tag_name("name") && n.attribute("type") == Some(kind))).map(|n| text_of(Some(n))).unwrap_or_default();
    let (scientific, common) = (organism_name("scientific"), organism_name("common"));
    let species = if common.is_empty() { scientific.clone() } else { format!("{scientific} ({common})") };
    let mut title = name.clone();
    if !gene.is_empty() {
        title.push_str(&format!(" ({gene})"));
    }
    if !species.is_empty() {
        title.push_str(&format!(", {species}"));
    }
    let mut lines = vec![Line::new(format!(
        "Source: UniProtKB/Swiss-Prot release {release}, UniProt Consortium. Copyrighted by the UniProt Consortium, see https://www.uniprot.org/terms. Distributed under the Creative Commons Attribution (CC BY 4.0) License."
    ))];
    let mut facts = vec![format!("Accession: {accession}"), format!("Entry name: {}", text_of(child(entry, "name")))];
    if !gene.is_empty() {
        facts.push(format!("Gene: {gene}"));
    }
    if !species.is_empty() {
        facts.push(format!("Organism: {species}"));
    }
    if let Some(sequence) = child(entry, "sequence") {
        if let Some(length) = sequence.attribute("length") {
            facts.push(format!("Length: {length} amino acids"));
        }
        if let Some(mass) = sequence.attribute("mass") {
            facts.push(format!("Mass: {mass} Da"));
        }
    }
    if let Some(existence) = child(entry, "proteinExistence").and_then(|p| p.attribute("type")) {
        facts.push(format!("Protein existence: {existence}"));
    }
    lines.push(Line::new(facts.join(". ") + "."));
    let alternatives = protein.map(|p| names(p, "alternativeName")).unwrap_or_default();
    if !alternatives.is_empty() {
        lines.push(Line::new(format!("Alternative names: {}.", alternatives.join("; "))));
    }
    for (kind, heading) in SECTIONS {
        let body: Vec<String> = entry.children().filter(|c| c.has_tag_name("comment") && c.attribute("type") == Some(kind)).flat_map(comment_lines).collect();
        if !body.is_empty() {
            lines.push(Line::new(format!("## {heading}")));
            lines.extend(body.into_iter().map(Line::new));
        }
    }
    let keywords: Vec<String> = entry.children().filter(|c| c.has_tag_name("keyword")).map(|k| text_of(Some(k))).filter(|k| !k.is_empty()).collect();
    if !keywords.is_empty() {
        lines.push(Line::new(format!("Keywords: {}.", keywords.join("; "))));
    }
    let url = format!("https://www.uniprot.org/uniprotkb/{accession}/entry");
    let meta = PaperMeta {
        lang: "en".into(),
        url: url.clone(),
        published: entry.attribute("created").unwrap_or("").to_string(),
        modified: entry.attribute("modified").unwrap_or("").to_string(),
        site_name: SITE.into(),
        tags: keywords,
        ..PaperMeta::default()
    };
    Some((url, typed_page(ENTRY, &title, lines, meta)))
}

/// Each `<entry>` element's XML, read line by line from the gzipped release.
pub fn entries(path: &PathBuf) -> Result<impl Iterator<Item = Result<String>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut lines = BufReader::with_capacity(1 << 20, MultiGzDecoder::new(file)).lines();
    Ok(std::iter::from_fn(move || {
        let mut chunk = String::new();
        for line in lines.by_ref() {
            let line = match line {
                Ok(line) => line,
                Err(e) => return Some(Err(e.into())),
            };
            if chunk.is_empty() && !line.trim_start().starts_with("<entry ") {
                continue;
            }
            chunk.push_str(&line);
            chunk.push('\n');
            if line.trim_end().ends_with("</entry>") {
                return Some(Ok(chunk));
            }
        }
        None
    }))
}

/// The Swiss-Prot release a reldate.txt names, `Release 2026_04 of 04-Nov-2026` as `2026_04`.
pub fn release_of(reldate: &str) -> Option<String> {
    reldate.lines().find(|l| l.contains("Swiss-Prot"))?.split_whitespace().skip_while(|w| *w != "Release").nth(1).map(String::from)
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let release = if args.release.is_empty() {
        let reldate = args.xml.with_file_name("reldate.txt");
        release_of(&std::fs::read_to_string(&reldate).with_context(|| format!("reading {}", reldate.display()))?).context("no Swiss-Prot release in reldate.txt")?
    } else {
        args.release.clone()
    };
    let mut counts = Counts::default();
    for (i, chunk) in entries(&args.xml)?.enumerate() {
        if args.to > 0 && i >= args.to {
            break;
        }
        let chunk = chunk?;
        if i < args.from {
            continue;
        }
        let doc = roxmltree::Document::parse(&chunk).with_context(|| format!("parsing entry {i}"))?;
        match page_of(doc.root_element(), &release) {
            Some((url, page)) => {
                batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
                counts.add("pages", 1);
            }
            None => counts.add("no_accession", 1),
        }
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

    const ENTRY_XML: &str = r#"<entry dataset="Swiss-Prot" created="1986-07-21" modified="2026-10-01" version="250" xmlns="https://uniprot.org/uniprot">
<accession>P69905</accession><name>HBA_HUMAN</name>
<protein><recommendedName><fullName>Hemoglobin subunit alpha</fullName></recommendedName><alternativeName><fullName>Alpha-globin</fullName></alternativeName></protein>
<gene><name type="primary">HBA1</name></gene>
<organism><name type="scientific">Homo sapiens</name><name type="common">Human</name></organism>
<comment type="function"><text evidence="1">Involved in oxygen transport from the lung to the various peripheral tissues.</text></comment>
<comment type="disease"><disease id="DI-02025"><name>Heinz body anemias</name><acronym>HEIBAN</acronym><description>Form of non-spherocytic hemolytic anemia.</description></disease></comment>
<comment type="subcellular location"><subcellularLocation><location>Cytoplasm</location></subcellularLocation></comment>
<keyword id="KW-0349">Heme</keyword><proteinExistence type="evidence at protein level"/>
<sequence length="142" mass="15258" checksum="x">MVLS</sequence></entry>"#;

    #[test]
    fn release_read_from_reldate() {
        let reldate = "UniProt Knowledgebase Release 2026_04 consists of:\nUniProtKB/Swiss-Prot Release 2026_04 of 04-Nov-2026\n";
        assert_eq!(release_of(reldate).as_deref(), Some("2026_04"));
    }

    #[test]
    fn entry_page_credits_uniprot() {
        let doc = roxmltree::Document::parse(ENTRY_XML).unwrap();
        let (url, page) = page_of(doc.root_element(), "2026_04").unwrap();
        assert_eq!(url, "https://www.uniprot.org/uniprotkb/P69905/entry");
        assert_eq!(page.title, "Hemoglobin subunit alpha (HBA1), Homo sapiens (Human)");
        assert_eq!(
            page.text.lines().skip(2).collect::<Vec<_>>(),
            [
                "Accession: P69905. Entry name: HBA_HUMAN. Gene: HBA1. Organism: Homo sapiens (Human). Length: 142 amino acids. Mass: 15258 Da. Protein existence: evidence at protein level.",
                "Alternative names: Alpha-globin.",
                "## Function",
                "Involved in oxygen transport from the lung to the various peripheral tissues.",
                "## Subcellular location",
                "Cytoplasm",
                "## Involvement in disease",
                "Heinz body anemias (HEIBAN): Form of non-spherocytic hemolytic anemia.",
                "Keywords: Heme.",
            ]
        );
        assert!(page.text.contains("CC BY 4.0"));
        assert_eq!((page.published.as_str(), page.modified.as_str()), ("1986-07-21", "2026-10-01"));
    }
}

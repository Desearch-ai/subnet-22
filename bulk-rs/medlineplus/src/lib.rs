//! MedlinePlus (public domain, credit NLM): a page per English health topic and per Genetics condition, gene and chromosome, their summaries read by the shared extraction.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use bulk_core::worker::{map_in_order, page_from_json, PythonArgs, Worker, PAGE_WORKER};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};

pub const SOURCE: &str = "medlineplus";
pub const SITE: &str = "MedlinePlus";
pub const ARTICLE: &str = "article";
const CONTENT_TYPE: &str = "application/xml";
const CREDIT: &str = "Source: MedlinePlus, National Library of Medicine (public domain).";
const ENGLISH: &str = "English";
const EXTRACT: &[u8] = b"e";
const WORKERS: usize = 4;

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    #[command(flatten)]
    pub python: PythonArgs,
    /// Page worker processes.
    #[arg(long, default_value_t = WORKERS)]
    pub workers: usize,
    /// The health topics XML, mplus_topics_<date>.xml from medlineplus.gov/xml.html.
    #[arg(long)]
    pub topics: PathBuf,
    /// The Genetics summaries, ghr-summaries.xml from medlineplus.gov/download/.
    #[arg(long)]
    pub genetics: Option<PathBuf>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for topics.xml and genetics.xml.
    #[arg(long)]
    pub out: PathBuf,
}

/// The newest topics XML that medlineplus.gov/xml.html lists, and the Genetics summaries.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let client = bulk_core::fetch::client()?;
    let page = bulk_core::fetch::text(&client, "https://medlineplus.gov/xml.html")?;
    let pattern = regex::Regex::new(r"https://medlineplus\.gov/xml/mplus_topics_(\d{4}-\d{2}-\d{2})\.xml").expect("a valid pattern");
    let newest = pattern.captures_iter(&page).map(|c| c[1].to_string()).max().context("no topics XML listed")?;
    let topics = format!("https://medlineplus.gov/xml/mplus_topics_{newest}.xml");
    for (url, name) in [(topics.as_str(), "topics.xml"), ("https://medlineplus.gov/download/ghr-summaries.xml", "genetics.xml")] {
        if !bulk_core::fetch::save(&client, url, &args.out.join(name))? {
            anyhow::bail!("{url} is not there");
        }
    }
    println!("{}", serde_json::json!({"topics": newest}));
    Ok(())
}

/// One page to build: its fields, the lines around its summary, and the summary's HTML.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub url: String,
    pub title: String,
    pub description: String,
    pub published: String,
    pub before: Vec<String>,
    pub html: String,
    pub after: Vec<String>,
    pub tags: Vec<String>,
}

/// `MM/DD/YYYY`, with or without a time after it, as `YYYY-MM-DD`; empty when it is not such a date.
pub fn us_date(text: &str) -> String {
    let day = text.split_whitespace().next().unwrap_or("");
    let parts: Vec<&str> = day.split('/').collect();
    match parts.as_slice() {
        [m, d, y] if m.len() == 2 && d.len() == 2 && y.len() == 4 && day.chars().all(|c| c.is_ascii_digit() || c == '/') => format!("{y}-{m}-{d}"),
        _ => String::new(),
    }
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn child<'a, 'i>(node: roxmltree::Node<'a, 'i>, tag: &str) -> Option<roxmltree::Node<'a, 'i>> {
    node.children().find(|c| c.has_tag_name(tag))
}

fn child_text(node: roxmltree::Node, tag: &str) -> String {
    child(node, tag).map(|c| squash(&c.text().unwrap_or(""))).unwrap_or_default()
}

fn texts(node: roxmltree::Node, tag: &str) -> Vec<String> {
    node.children().filter(|c| c.has_tag_name(tag)).map(|c| squash(c.text().unwrap_or(""))).filter(|t| !t.is_empty()).collect()
}

fn listed(label: &str, items: &[String]) -> Option<String> {
    (!items.is_empty()).then(|| format!("{label}: {}.", items.join("; ")))
}

fn parse(xml: &str) -> Result<roxmltree::Document<'_>> {
    let options = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    roxmltree::Document::parse_with_options(xml, options).context("parsing the XML")
}

/// The English health topics, in file order, and the day the file was generated.
pub fn topics(xml: &str) -> Result<(Vec<Entry>, String)> {
    let doc = parse(xml)?;
    let root = doc.root_element();
    let generated = us_date(root.attribute("date-generated").unwrap_or(""));
    let mut found = Vec::new();
    for node in root.children().filter(|c| c.has_tag_name("health-topic")) {
        if node.attribute("language") != Some(ENGLISH) {
            continue;
        }
        let attr = |name: &str| squash(node.attribute(name).unwrap_or(""));
        let also_called = texts(node, "also-called");
        let groups = texts(node, "group");
        let mesh: Vec<String> = node.children().filter(|c| c.has_tag_name("mesh-heading")).flat_map(|m| texts(m, "descriptor")).collect();
        found.push(Entry {
            url: attr("url"),
            title: attr("title"),
            description: attr("meta-desc"),
            published: us_date(&attr("date-created")),
            before: (!also_called.is_empty()).then(|| format!("Also called: {}", also_called.join(", "))).into_iter().collect(),
            html: child(node, "full-summary").and_then(|s| s.text()).unwrap_or("").to_string(),
            after: [listed("Health topic groups", &groups), listed("Related health topics", &texts(node, "related-topic")), listed("MeSH headings", &mesh)]
                .into_iter()
                .flatten()
                .collect(),
            tags: groups,
        });
    }
    Ok((found, generated))
}

/// A Genetics page address as its canonical form spells it, with a trailing slash.
pub fn genetics_url(page: &str) -> String {
    if page.is_empty() || page.ends_with('/') {
        page.to_string()
    } else {
        format!("{page}/")
    }
}

/// The XHTML of each `<text>`'s `<html>`, its `html:` prefixes dropped, so it reads as HTML.
fn summary_html(xml: &str, node: roxmltree::Node) -> String {
    let Some(list) = child(node, "text-list") else { return String::new() };
    let mut out = String::new();
    for text in list.children().filter(|c| c.has_tag_name("text")) {
        if let Some(html) = child(text, "html") {
            let inner: String = html.children().map(|c| &xml[c.range()]).collect();
            out.push_str(&inner.replace("<html:", "<").replace("</html:", "</"));
        }
    }
    out
}

fn related(node: roxmltree::Node, list: &str, item: &str) -> Vec<String> {
    child(node, list).map(|l| l.children().filter(|c| c.has_tag_name(item)).map(|c| child_text(c, "name")).filter(|n| !n.is_empty()).collect()).unwrap_or_default()
}

/// The Genetics conditions, genes and chromosomes, in file order.
pub fn genetics(xml: &str) -> Result<Vec<Entry>> {
    let doc = parse(xml)?;
    let mut found = Vec::new();
    for node in doc.root_element().children().filter(|c| c.is_element()) {
        let name = child_text(node, "name");
        let title = match node.tag_name().name() {
            "health-condition-summary" => name.clone(),
            "gene-summary" => format!("{} gene: {name}", child_text(node, "gene-symbol")),
            "chromosome-summary" => format!("Chromosome {name}"),
            _ => continue,
        };
        let synonyms = child(node, "synonym-list").map(|l| texts(l, "synonym")).unwrap_or_default();
        let inheritance: Vec<String> = child(node, "inheritance-pattern-list")
            .map(|l| l.children().filter(|c| c.has_tag_name("inheritance-pattern")).map(|c| child_text(c, "memo")).filter(|m| !m.is_empty()).collect())
            .unwrap_or_default();
        let genes: Vec<String> = child(node, "related-gene-list")
            .map(|l| l.children().filter(|c| c.has_tag_name("related-gene")).map(|c| child_text(c, "gene-symbol")).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        let chromosomes = related(node, "related-gene-list", "related-chromosome");
        let conditions = related(node, "related-health-condition-list", "related-health-condition");
        found.push(Entry {
            url: genetics_url(&child_text(node, "ghr-page")),
            title,
            published: child_text(node, "published"),
            before: listed("Other names", &synonyms).into_iter().collect(),
            html: summary_html(xml, node),
            after: [
                listed("Inheritance", &inheritance),
                listed("Related genes", &genes),
                listed("Related chromosomes", &chromosomes),
                listed("Related conditions", &conditions),
            ]
            .into_iter()
            .flatten()
            .collect(),
            ..Entry::default()
        });
    }
    Ok(found)
}

/// The summary as a page for the shared extraction: the entry's title as its heading, then the summary's HTML.
pub fn summary_document(title: &str, html: &str) -> String {
    let title = escape(title);
    format!("<html><head><title>{title}</title></head><body><main><h1>{title}</h1>{html}</main></body></html>")
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// The extraction's text lines after its title heading.
pub fn body_lines(text: &str, title: &str) -> Vec<Line> {
    let mut lines: Vec<&str> = text.lines().collect();
    if lines.first().is_some_and(|first| first.trim_start_matches('#').trim() == title) {
        lines.remove(0);
    }
    lines.into_iter().filter(|line| !line.trim().is_empty()).map(Line::new).collect()
}

/// An entry's page: the credit, the lines before the summary, the summary, then the lines after it.
pub fn page_of(entry: &Entry, body: Vec<Line>, modified: &str) -> Page {
    let mut lines = vec![Line::new(CREDIT)];
    lines.extend(entry.before.iter().map(Line::new));
    lines.extend(body);
    lines.extend(entry.after.iter().map(Line::new));
    let meta = PaperMeta {
        lang: "en".into(),
        url: entry.url.clone(),
        published: entry.published.clone(),
        modified: modified.to_string(),
        site_name: SITE.into(),
        tags: entry.tags.clone(),
        ..PaperMeta::default()
    };
    let mut page = typed_page(ARTICLE, &entry.title, lines, meta);
    page.description = entry.description.clone();
    page
}

fn summary_text(worker: &mut Worker, entry: &Entry) -> Result<String> {
    let html = summary_document(&entry.title, &entry.html);
    let size = u32::try_from(entry.url.len())?.to_be_bytes();
    let answer = worker.ask_json(&[EXTRACT, &size, entry.url.as_bytes(), html.as_bytes()])?;
    Ok(page_from_json(answer.get("page").ok_or_else(|| anyhow!("an extract answer without a page"))?)?.text)
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

/// Every English topic, then every Genetics summary, that has a summary; each URL once.
pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64, spawn: &(dyn Fn() -> Result<Worker> + Sync)) -> Result<Counts> {
    let (mut entries, generated) = topics(&read(&args.topics)?)?;
    let topics_count = entries.len();
    if let Some(path) = &args.genetics {
        entries.extend(genetics(&read(path)?)?);
    }
    let mut counts = Counts::default();
    let texts = map_in_order(&entries, args.workers, spawn, &summary_text)?;
    let mut seen = std::collections::HashSet::new();
    for (i, (entry, text)) in entries.iter().zip(texts).enumerate() {
        let body = body_lines(&text, &entry.title);
        if entry.url.is_empty() || body.is_empty() {
            counts.add("no_summary", 1);
            continue;
        }
        if !seen.insert(entry.url.clone()) {
            counts.add("duplicate", 1);
            continue;
        }
        batches.add(upload_row(&entry.url, page_of(entry, body, &generated), as_of_us, CONTENT_TYPE))?;
        counts.add(if i < topics_count { "topics" } else { "genetics" }, 1);
        counts.add("pages", 1);
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            break;
        }
    }
    Ok(counts)
}

pub fn main(args: Args) -> Result<()> {
    let python_path = args.python.python_path(PAGE_WORKER);
    let spawn = || Worker::spawn(&args.python.python, &python_path, PAGE_WORKER, &[]);
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us, &spawn)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOPICS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE health-topics PUBLIC "-//NLM//DTD health-topics //EN" "https://medlineplus.gov/xml/mplus_topics.dtd">
<health-topics total="2" date-generated="11/12/2026 02:30:39">
<health-topic meta-desc="Stomach aches can be painful." title="Abdominal Pain" url="https://medlineplus.gov/abdominalpain.html" id="3061" language="English" date-created="01/07/2003">
<also-called>Bellyache</also-called>
<full-summary>&lt;p&gt;Your abdomen extends from below your chest.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Sudden pain&lt;/li&gt;&lt;/ul&gt;</full-summary>
<group url="https://medlineplus.gov/digestivesystem.html" id="7">Digestive System</group>
<group url="https://medlineplus.gov/symptoms.html" id="40">Symptoms</group>
<mesh-heading><descriptor id="D015746">Abdominal Pain</descriptor></mesh-heading>
<related-topic url="https://medlineplus.gov/stomachdisorders.html" id="1">Stomach Disorders</related-topic>
<site title="Abdominal pain (A.D.A.M.)" url="https://medlineplus.gov/ency/article/003120.htm" language="English"><information-category>Symptoms</information-category></site>
</health-topic>
<health-topic title="Dolor abdominal" url="https://medlineplus.gov/spanish/abdominalpain.html" id="3062" language="Spanish" date-created="01/07/2003">
<full-summary>&lt;p&gt;Su abdomen.&lt;/p&gt;</full-summary>
</health-topic>
</health-topics>"#;

    const GENETICS: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<summaries xmlns="https://medlineplus.gov/download/ghr-summaries-20250602.xsd" xmlns:html="http://www.w3.org/1999/xhtml">
<health-condition-summary id="21631">
<name>10q26 deletion syndrome</name>
<ghr-page>https://medlineplus.gov/genetics/condition/10q26-deletion-syndrome</ghr-page>
<text-list><text><text-role>description</text-role><html><html:p>A deletion on chromosome 10.</html:p></html></text></text-list>
<inheritance-pattern-list><inheritance-pattern><code>ad</code><memo>Autosomal dominant</memo></inheritance-pattern></inheritance-pattern-list>
<related-gene-list><related-chromosome><name>10</name><ghr-page>https://medlineplus.gov/genetics/chromosome/10</ghr-page></related-chromosome></related-gene-list>
<synonym-list><synonym>10qter deletion</synonym></synonym-list>
<published>2017-05-01</published>
</health-condition-summary>
<gene-summary id="21929">
<gene-symbol>AAAS</gene-symbol>
<name>aladin WD repeat nucleoporin</name>
<ghr-page>https://medlineplus.gov/genetics/gene/aaas</ghr-page>
<text-list><text><text-role>function</text-role><html><html:p>Makes ALADIN.</html:p></html></text></text-list>
<related-health-condition-list><related-health-condition><name>Triple A syndrome</name><ghr-page>https://medlineplus.gov/genetics/condition/triple-a-syndrome</ghr-page></related-health-condition></related-health-condition-list>
</gene-summary>
</summaries>"#;

    #[test]
    fn english_topics_only_with_their_fields() {
        let (found, generated) = topics(TOPICS).unwrap();
        assert_eq!(generated, "2026-11-12");
        assert_eq!(found.len(), 1);
        let t = &found[0];
        assert_eq!((t.url.as_str(), t.title.as_str(), t.published.as_str()), ("https://medlineplus.gov/abdominalpain.html", "Abdominal Pain", "2003-01-07"));
        assert_eq!(t.before, ["Also called: Bellyache"]);
        assert_eq!(t.tags, ["Digestive System", "Symptoms"]);
        assert!(t.html.starts_with("<p>Your abdomen"));
    }

    #[test]
    fn topic_page_credits_nlm_and_leaves_out_other_sites() {
        let (found, generated) = topics(TOPICS).unwrap();
        let body = body_lines("# Abdominal Pain\nYour abdomen extends from below your chest.\n- Sudden pain", "Abdominal Pain");
        let page = page_of(&found[0], body, &generated);
        assert_eq!(
            page.text.lines().collect::<Vec<_>>(),
            [
                "Abdominal Pain",
                CREDIT,
                "Also called: Bellyache",
                "Your abdomen extends from below your chest.",
                "- Sudden pain",
                "Health topic groups: Digestive System; Symptoms.",
                "Related health topics: Stomach Disorders.",
                "MeSH headings: Abdominal Pain.",
            ]
        );
        assert!(!page.text.contains("A.D.A.M."));
        assert_eq!((page.page_type.as_str(), page.lang.as_str(), page.published.as_str(), page.modified.as_str()), (ARTICLE, "en", "2003-01-07", "2026-11-12"));
        assert_eq!(page.description, "Stomach aches can be painful.");
    }

    #[test]
    fn genetics_conditions_and_genes_with_canonical_urls() {
        let found = genetics(GENETICS).unwrap();
        assert_eq!(found.len(), 2);
        let (condition, gene) = (&found[0], &found[1]);
        assert_eq!(condition.url, "https://medlineplus.gov/genetics/condition/10q26-deletion-syndrome/");
        assert_eq!(condition.html, "<p>A deletion on chromosome 10.</p>");
        assert_eq!(condition.before, ["Other names: 10qter deletion."]);
        assert_eq!(condition.after, ["Inheritance: Autosomal dominant.", "Related chromosomes: 10."]);
        assert_eq!(condition.published, "2017-05-01");
        assert_eq!((gene.title.as_str(), gene.url.as_str()), ("AAAS gene: aladin WD repeat nucleoporin", "https://medlineplus.gov/genetics/gene/aaas/"));
        assert_eq!(gene.after, ["Related conditions: Triple A syndrome."]);
    }

    #[test]
    fn dates_read_as_us_dates() {
        assert_eq!(us_date("11/12/2026 02:30:39"), "2026-11-12");
        assert_eq!(us_date("2026-11-12"), "");
        assert_eq!(us_date(""), "");
    }

    #[test]
    fn summary_document_escapes_the_title() {
        assert_eq!(summary_document("A & B", "<p>x</p>"), "<html><head><title>A &amp; B</title></head><body><main><h1>A &amp; B</h1><p>x</p></main></body></html>");
    }
}

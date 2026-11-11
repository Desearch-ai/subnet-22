//! `bulk.pmc.article_page`: the front matter read here, the abstracts and body rendered by the Python worker.

use std::borrow::Cow;

use anyhow::{anyhow, Result};
use bulk_core::{authors_line, paper_page, Line, Page, PaperMeta};
use desearch::text::{collapse, head, strip};
use roxmltree::{Document, Node, ParsingOptions};
use serde_json::Value;

use crate::py::{decimal, str_of};
use crate::render::{Parts, Rendered, Renderer};

pub const SITE: &str = "PMC";
pub const MOST_AUTHORS: usize = 10;
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";
const DATE_KEYS: [&str; 4] = ["epub", "pub", "ppub", "collection"];

pub fn url_of(pmcid: &str) -> String {
    format!("https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
}

/// What `article_page` reads from `front/article-meta` and the root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Front {
    pub title: String,
    pub authors: Vec<String>,
    pub doi: String,
    pub published: String,
    pub lang: String,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parsed {
    NoFront,
    NoTitle,
    Front(Front),
}

/// What `article_page` and `run` make of one taken version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    BadXml,
    /// `no_front` or `no_title`.
    Skipped(&'static str),
    Page {
        url: String,
        page: Box<Page>,
        no_body: bool,
    },
}

fn named(node: &Node, name: &str) -> bool {
    node.is_element() && node.tag_name().namespace().is_none() && node.tag_name().name() == name
}

fn children<'a, 'i>(node: Node<'a, 'i>, name: &'a str) -> impl Iterator<Item = Node<'a, 'i>> + 'a {
    node.children().filter(move |c| named(c, name))
}

/// ElementPath's `find("a/b/...")`: the first match in document order.
fn find<'a, 'i>(node: Node<'a, 'i>, path: &[&'a str]) -> Option<Node<'a, 'i>> {
    let (first, rest) = path.split_first()?;
    children(node, first).find_map(|child| if rest.is_empty() { Some(child) } else { find(child, rest) })
}

/// lxml's `.text`: the text before the first child.
fn own_text<'a>(node: Node<'a, '_>) -> &'a str {
    node.text().unwrap_or("")
}

/// `bulk.pmc.text_of`: the subtree's text, whitespace collapsed.
pub fn text_of(node: Option<Node>) -> String {
    let Some(node) = node else { return String::new() };
    let text: String = node.descendants().filter(|n| n.is_text()).filter_map(|n| n.text()).collect();
    collapse(&text)
}

fn full_date(node: Node) -> String {
    let part = |name| children(node, name).next().map(own_text).unwrap_or("");
    let (Some(year), Some(month), Some(day)) = (decimal(part("year")), decimal(part("month")), decimal(part("day"))) else {
        return String::new();
    };
    format!("{year:0>4}-{month:0>2}-{day:0>2}")
}

fn declared_encoding(xml: &[u8]) -> Option<String> {
    if !xml.starts_with(b"<?xml") {
        return None;
    }
    let end = xml.windows(2).position(|w| w == b"?>")?;
    let declaration = std::str::from_utf8(&xml[..end]).ok()?;
    let rest = &declaration[declaration.find("encoding")? + "encoding".len()..];
    let rest = rest.trim_start().strip_prefix('=')?.trim_start();
    let quote = rest.chars().next().filter(|q| *q == '"' || *q == '\'')?;
    let value = &rest[1..];
    Some(value[..value.find(quote)?].to_ascii_lowercase())
}

/// The article as text, by its byte order mark or declared encoding as lxml reads it; None for encodings not read here.
fn decode(xml: &[u8]) -> Option<Cow<'_, str>> {
    let xml = xml.strip_prefix(b"\xef\xbb\xbf").unwrap_or(xml);
    if xml.is_ascii() {
        return std::str::from_utf8(xml).ok().map(Cow::Borrowed);
    }
    match declared_encoding(xml).as_deref() {
        None | Some("utf-8" | "utf8") => std::str::from_utf8(xml).ok().map(Cow::Borrowed),
        Some("iso-8859-1" | "iso8859-1" | "iso_8859-1" | "latin-1" | "latin1" | "l1") => Some(Cow::Owned(xml.iter().map(|&b| b as char).collect())),
        _ => None,
    }
}

/// The front matter, or None when the XML does not parse here.
pub fn parse_front(xml: &[u8]) -> Option<Parsed> {
    let text = decode(xml)?;
    let doc = Document::parse_with_options(&text, ParsingOptions { allow_dtd: true, nodes_limit: u32::MAX }).ok()?;
    let root = doc.root_element();
    let Some(front) = find(root, &["front", "article-meta"]) else { return Some(Parsed::NoFront) };
    let title = text_of(find(front, &["title-group", "article-title"]));
    if title.is_empty() {
        return Some(Parsed::NoTitle);
    }
    let mut authors = Vec::new();
    for contrib in children(front, "contrib-group").flat_map(|group| children(group, "contrib")) {
        if contrib.attribute("contrib-type").unwrap_or("author") != "author" {
            continue;
        }
        if let Some(name) = children(contrib, "name").next() {
            let parts = [text_of(children(name, "given-names").next()), text_of(children(name, "surname").next())];
            authors.push(parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" "));
        } else if let Some(collab) = children(contrib, "collab").next() {
            authors.push(text_of(Some(collab)));
        }
    }
    let doi = text_of(children(front, "article-id").find(|id| id.attribute("pub-id-type") == Some("doi")));
    let mut dates: Vec<(Option<&str>, Node)> = Vec::new();
    for date in children(front, "pub-date") {
        let key = date.attribute("pub-type").filter(|k| !k.is_empty()).or_else(|| date.attribute("date-type"));
        match dates.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = date,
            None => dates.push((key, date)),
        }
    }
    let published = DATE_KEYS
        .iter()
        .filter_map(|k| dates.iter().find(|(key, _)| *key == Some(k)).map(|(_, node)| full_date(*node)))
        .find(|d| !d.is_empty())
        .unwrap_or_default();
    let lang = root.attribute((XML_NS, "lang")).filter(|l| !l.is_empty()).unwrap_or("en").to_lowercase();
    let tags = children(front, "kwd-group").flat_map(|group| children(group, "kwd")).map(|k| text_of(Some(k))).collect();
    Some(Parsed::Front(Front { title, authors, doi, published, lang: head(&lang, 2).to_string(), tags }))
}

/// The lines and page `article_page` builds from the front matter, the metadata and the rendered parts.
pub fn assemble(front: Front, meta: &Value, parts: Parts) -> Result<(String, Page)> {
    let author = authors_line(&front.authors, MOST_AUTHORS);
    let citation = match meta.get("citation") {
        Some(Value::String(s)) => collapse(s),
        _ => String::new(),
    };
    let doi = if front.doi.is_empty() { String::new() } else { format!("DOI: {}", front.doi) };
    let mut lines: Vec<Line> = [author.clone(), citation, doi].into_iter().filter(|t| !t.is_empty()).map(Line::new).collect();
    for abstract_lines in parts.abstracts {
        lines.push(Line::new("## Abstract"));
        lines.extend(abstract_lines);
    }
    lines.extend(parts.body.unwrap_or_default());
    lines.retain(|line| !strip(&line.text).is_empty());
    let pmcid = meta.get("pmcid").ok_or_else(|| anyhow!("metadata without a pmcid"))?;
    let url = url_of(&str_of(pmcid));
    let meta =
        PaperMeta { lang: front.lang, url: url.clone(), published: front.published, modified: String::new(), author, site_name: SITE.into(), tags: front.tags };
    Ok((url.clone(), paper_page(&front.title, lines, meta)))
}

/// One taken version's page, or why it has none.
pub fn article_page(xml: &[u8], meta: &Value, renderer: &Renderer) -> Result<Outcome> {
    let rendered = renderer.render(xml)?;
    if rendered == Rendered::BadXml {
        return Ok(Outcome::BadXml);
    }
    let front = match parse_front(xml) {
        None => {
            eprintln!("pmc: {} parses in lxml but not here; counted as bad_xml", meta.get("pmcid").map(str_of).unwrap_or_default());
            return Ok(Outcome::BadXml);
        }
        Some(Parsed::NoFront) => return Ok(Outcome::Skipped("no_front")),
        Some(Parsed::NoTitle) => return Ok(Outcome::Skipped("no_title")),
        Some(Parsed::Front(front)) => front,
    };
    let Rendered::Parts(parts) = rendered else { return Ok(Outcome::BadXml) };
    let no_body = parts.body.is_none();
    let (url, page) = assemble(front, meta, parts)?;
    Ok(Outcome::Page { url, page: Box::new(page), no_body })
}

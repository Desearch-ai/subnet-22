//! OAI-PMH pages read as lxml reads them: `find` on direct children, `.text` up to the first child node.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use roxmltree::{Document, Node, ParsingOptions};

use crate::{page_of, squash, Paper, Record};

pub const OAI_NS: &str = "http://www.openarchives.org/OAI/2.0/";
pub const ARXIV_NS: &str = "http://arxiv.org/OAI/arXiv/";
const ID_PREFIX: &str = "oai:arXiv.org:";

/// One record of an OAI-PMH page, as `oai_papers` yields it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum OaiItem {
    Deleted(String),
    Paper(Paper),
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Document<'_>> {
    let text = std::str::from_utf8(bytes).context("not UTF-8")?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    Ok(Document::parse_with_options(text, ParsingOptions { allow_dtd: true, ..ParsingOptions::default() })?)
}

fn is(node: &Node, ns: &str, name: &str) -> bool {
    node.is_element() && node.tag_name().name() == name && node.tag_name().namespace() == Some(ns)
}

fn children<'a, 'i: 'a>(node: Node<'a, 'i>, ns: &'a str, name: &'a str) -> impl Iterator<Item = Node<'a, 'i>> + 'a {
    node.children().filter(move |c| is(c, ns, name))
}

/// `find("x/y")`: the first `y` child of any `x` child.
fn find<'a, 'i: 'a>(node: Node<'a, 'i>, outer: (&'a str, &'a str), inner: (&'a str, &'a str)) -> Option<Node<'a, 'i>> {
    children(node, outer.0, outer.1).flat_map(move |c| children(c, inner.0, inner.1)).next()
}

/// lxml's `.text`: the element's text before its first child element, comment or instruction.
pub(crate) fn text_of<'a>(node: Node<'a, '_>) -> &'a str {
    node.first_child().filter(Node::is_text).and_then(|c| c.text()).unwrap_or("")
}

fn findtext<'a>(node: Node<'a, '_>, ns: &str, name: &str) -> &'a str {
    node.children().find(|c| is(c, ns, name)).map(text_of).unwrap_or("")
}

/// `.//o:x`: the first descendant of the root element with that name.
pub(crate) fn first_descendant<'a, 'i>(doc: &'a Document<'i>, ns: &str, name: &str) -> Option<Node<'a, 'i>> {
    doc.root_element().descendants().skip(1).find(|n| is(n, ns, name))
}

/// Each record's item, stopping after the first record lxml's reading would raise on.
pub fn papers_in(bytes: &[u8]) -> Result<Vec<Result<OaiItem>>> {
    let doc = parse(bytes)?;
    let mut out = Vec::new();
    for record in doc.root_element().descendants().skip(1).filter(|n| is(n, OAI_NS, "record")) {
        let item = item_of(record);
        let failed = item.is_err();
        out.push(item);
        if failed {
            break;
        }
    }
    Ok(out)
}

fn item_of(record: Node) -> Result<OaiItem> {
    let header = children(record, OAI_NS, "header").next().ok_or_else(|| anyhow!("a record without a header"))?;
    if header.attribute("status") == Some("deleted") {
        let id = find(record, (OAI_NS, "header"), (OAI_NS, "identifier")).map(text_of).unwrap_or("");
        return Ok(OaiItem::Deleted(id.strip_prefix(ID_PREFIX).unwrap_or(id).to_string()));
    }
    let meta = find(record, (OAI_NS, "metadata"), (ARXIV_NS, "arXiv")).ok_or_else(|| {
        let id = find(record, (OAI_NS, "header"), (OAI_NS, "identifier")).map(text_of).unwrap_or("");
        anyhow!("record {id:?} has no arXiv metadata")
    })?;
    let authors = children(meta, ARXIV_NS, "authors")
        .flat_map(|a| children(a, ARXIV_NS, "author"))
        .map(|a| {
            let parts = ["forenames", "keyname", "suffix"].map(|name| findtext(a, ARXIV_NS, name));
            squash(&parts.iter().filter(|p| !p.is_empty()).copied().collect::<Vec<_>>().join(" "))
        })
        .collect();
    let squashed = |name: &str| squash(findtext(meta, ARXIV_NS, name));
    let stripped = |name: &str| desearch::text::strip(findtext(meta, ARXIV_NS, name)).to_string();
    Ok(OaiItem::Paper(Paper {
        arxiv_id: stripped("id"),
        title: squashed("title"),
        authors,
        abstract_text: squashed("abstract"),
        categories: squashed("categories"),
        created: stripped("created"),
        updated: stripped("updated"),
        comments: squashed("comments"),
        journal: squashed("journal-ref"),
        doi: squashed("doi"),
        licence: squashed("license"),
    }))
}

/// `bulk.arxiv.oai_papers`: an error for a page lxml cannot parse, else each record, ending at the first bad one.
pub fn oai_papers(path: &Path) -> Result<Vec<Result<OaiItem>>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let items = papers_in(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    Ok(items.into_iter().map(|item| item.with_context(|| format!("in {}", path.display()))).collect())
}

/// `bulk.arxiv.oai_records`: deleted ids and abstract pages.
pub fn oai_records(path: &Path) -> Result<Vec<Result<Record>>> {
    Ok(oai_papers(path)?
        .into_iter()
        .map(|item| {
            item.map(|item| match item {
                OaiItem::Deleted(id) => Record::Deleted(id),
                OaiItem::Paper(paper) => {
                    let (url, page) = page_of(&paper);
                    Record::Page(url, page)
                }
            })
        })
        .collect())
}

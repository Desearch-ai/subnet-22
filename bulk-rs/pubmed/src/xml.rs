//! One record at a time as a small owned tree, read by streaming, with lxml's `.text`, `itertext()` and path lookups.

use std::io::BufRead;

use anyhow::{anyhow, bail, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::name::{LocalName, NamespaceResolver, ResolveResult};
use quick_xml::reader::NsReader;
use quick_xml::XmlVersion;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Element {
    /// lxml's tag: the local name, or `{namespace}local` in a namespace.
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Element(Element),
    /// Adjacent character data and CDATA, joined as lxml joins them.
    Text(String),
    /// A comment or processing instruction: it ends `.text`, and only its tail is in `itertext()`.
    Other,
}

impl Element {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// lxml's `.text`: the character data before the first child node.
    pub fn text(&self) -> Option<&str> {
        match self.children.first() {
            Some(Node::Text(t)) => Some(t),
            _ => None,
        }
    }

    /// `"".join(element.itertext())`.
    pub fn itertext(&self) -> String {
        let mut out = String::new();
        self.push_text(&mut out);
        out
    }

    fn push_text(&self, out: &mut String) {
        for child in &self.children {
            match child {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => e.push_text(out),
                Node::Other => {}
            }
        }
    }

    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|c| match c {
            Node::Element(e) => Some(e),
            _ => None,
        })
    }

    /// ElementPath's `iterfind` for a path of child tags such as `JournalIssue/PubDate`.
    pub fn iterfind<'a>(&'a self, path: &str) -> Vec<&'a Element> {
        let steps: Vec<&str> = path.split('/').collect();
        let mut out = Vec::new();
        self.walk(&steps, &mut out, false);
        out
    }

    pub fn find(&self, path: &str) -> Option<&Element> {
        let steps: Vec<&str> = path.split('/').collect();
        let mut out = Vec::new();
        self.walk(&steps, &mut out, true);
        out.pop()
    }

    /// lxml's `findtext`: None when nothing matches, the text or "" when something does.
    pub fn findtext(&self, path: &str) -> Option<&str> {
        self.find(path).map(|e| e.text().unwrap_or(""))
    }

    fn walk<'a>(&'a self, steps: &[&str], out: &mut Vec<&'a Element>, first: bool) -> bool {
        let Some((step, rest)) = steps.split_first() else {
            return false;
        };
        for child in self.elements().filter(|e| e.tag == *step) {
            if rest.is_empty() {
                out.push(child);
                if first {
                    return true;
                }
            } else if child.walk(rest, out, first) {
                return true;
            }
        }
        false
    }
}

/// How much of an element a reader keeps; whatever is dropped is still parsed and checked.
#[derive(Clone, Copy, Debug)]
pub enum Keep {
    /// Everything, as lxml holds it.
    All,
    /// The attributes and the children named here, each kept as its entry says; no text.
    Children(&'static [(&'static str, Keep)]),
    /// The attributes and `.text`.
    Text,
    /// The attributes and what `"".join(itertext())` gives, as one text node.
    AllText,
}

struct Frame {
    element: Element,
    keep: Keep,
    /// Under `Keep::Text`, a child node has ended `.text`.
    text_done: bool,
}

/// Each element named in `roots` at any depth, in the order their end tags come, as lxml's `iterparse(tag=...)` yields them.
pub struct Records<R: BufRead> {
    reader: NsReader<R>,
    buf: Vec<u8>,
    roots: &'static [(&'static str, Keep)],
    stack: Vec<Frame>,
    /// Open elements below the top frame that it does not keep as elements.
    nested: usize,
    depth: usize,
    seen_root: bool,
}

impl<R: BufRead> Records<R> {
    pub fn new(source: R, roots: &'static [(&'static str, Keep)]) -> Self {
        Records { reader: NsReader::from_reader(source), buf: Vec::with_capacity(1 << 16), roots, stack: Vec::new(), nested: 0, depth: 0, seen_root: false }
    }

    pub fn next_record(&mut self) -> Result<Option<Element>> {
        let Records { reader, buf, roots, stack, nested, depth, seen_root } = self;
        loop {
            buf.clear();
            let at = reader.buffer_position();
            let event = reader.read_event_into(buf).map_err(|e| anyhow!("XML error at byte {at}: {e}"))?;
            let resolver = reader.resolver();
            if *depth == 0 {
                match &event {
                    Event::Start(_) | Event::Empty(_) if *seen_root => bail!("Extra content at the end of the document, at byte {at}"),
                    Event::Start(_) | Event::Empty(_) => *seen_root = true,
                    Event::Text(e) if !e.iter().all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')) => bail!("text outside the root element, at byte {at}"),
                    Event::CData(_) | Event::GeneralRef(_) => bail!("text outside the root element, at byte {at}"),
                    _ => {}
                }
            }
            match event {
                Event::Start(e) => {
                    *depth += 1;
                    match open(resolver, &e, roots, stack, *nested)? {
                        Some(frame) => stack.push(frame),
                        None if !stack.is_empty() => *nested += 1,
                        None => {}
                    }
                }
                Event::Empty(e) => {
                    if let Some(frame) = open(resolver, &e, roots, stack, *nested)? {
                        if let Some(record) = close(frame, stack) {
                            return Ok(Some(record));
                        }
                    }
                }
                Event::End(_) => {
                    *depth -= 1;
                    if *nested > 0 {
                        *nested -= 1;
                    } else if let Some(frame) = stack.pop() {
                        if let Some(record) = close(frame, stack) {
                            return Ok(Some(record));
                        }
                    }
                }
                Event::Text(e) => {
                    let text = e.xml10_content()?;
                    check_chars(&text)?;
                    push_text(stack, *nested, &text);
                }
                Event::CData(e) => {
                    let text = e.xml10_content()?;
                    check_chars(&text)?;
                    push_text(stack, *nested, &text);
                }
                Event::GeneralRef(e) => {
                    let c = match e.resolve_char_ref()? {
                        Some(c) if is_xml_char(c) => c,
                        Some(c) => bail!("invalid character reference &#{};", c as u32),
                        None => match &*e.decode()? {
                            "amp" => '&',
                            "lt" => '<',
                            "gt" => '>',
                            "quot" => '"',
                            "apos" => '\'',
                            name => bail!("Entity '{name}' not defined"),
                        },
                    };
                    push_text(stack, *nested, c.encode_utf8(&mut [0; 4]));
                }
                Event::Comment(_) | Event::PI(_) => {
                    if let Some(top) = stack.last_mut().filter(|_| *nested == 0) {
                        match top.keep {
                            Keep::All => top.element.children.push(Node::Other),
                            Keep::Text => top.text_done = true,
                            Keep::Children(_) | Keep::AllText => {}
                        }
                    }
                }
                Event::Decl(e) => {
                    if let Some(encoding) = e.encoding() {
                        let encoding = encoding?;
                        if !matches!(encoding.to_ascii_lowercase().as_slice(), b"utf-8" | b"utf8") {
                            bail!("unsupported XML encoding {}", String::from_utf8_lossy(&encoding));
                        }
                    }
                }
                Event::DocType(_) => {}
                Event::Eof => {
                    if *depth > 0 {
                        bail!("Premature end of data: the XML ends inside an element");
                    }
                    if !*seen_root {
                        bail!("the XML has no root element");
                    }
                    return Ok(None);
                }
            }
        }
    }
}

/// The frame for an element the reader keeps, or None; its name and attributes are checked either way.
fn open(resolver: &NamespaceResolver, e: &BytesStart, roots: &[(&str, Keep)], stack: &mut [Frame], nested: usize) -> Result<Option<Frame>> {
    let (ns, local) = resolver.resolve_element(e.name());
    let plain = match ns {
        ResolveResult::Unbound => Some(local.as_ref()),
        ResolveResult::Bound(_) => None,
        ResolveResult::Unknown(prefix) => {
            bail!("Namespace prefix {} on {} is not defined", String::from_utf8_lossy(&prefix), String::from_utf8_lossy(local.as_ref()))
        }
    };
    let named = |list: &[(&str, Keep)]| plain.and_then(|name| list.iter().find(|(tag, _)| tag.as_bytes() == name).map(|(_, keep)| *keep));
    let keep = match stack.last_mut() {
        None => named(roots),
        Some(_) if nested > 0 => None,
        Some(top) => match top.keep {
            Keep::All => Some(Keep::All),
            Keep::Children(list) => named(list),
            Keep::Text => {
                top.text_done = true;
                None
            }
            Keep::AllText => None,
        },
    };
    let attrs = attributes(resolver, e, keep.is_some())?;
    let Some(keep) = keep else {
        return Ok(None);
    };
    let tag = resolved(resolver.resolve_element(e.name()))?;
    Ok(Some(Frame { element: Element { tag, attrs, children: Vec::new() }, keep, text_done: false }))
}

/// The element on its parent, or the record itself when it has none.
fn close(frame: Frame, stack: &mut [Frame]) -> Option<Element> {
    match stack.last_mut() {
        Some(parent) => {
            parent.element.children.push(Node::Element(frame.element));
            None
        }
        None => Some(frame.element),
    }
}

fn push_text(stack: &mut [Frame], nested: usize, text: &str) {
    let Some(top) = stack.last_mut() else {
        return;
    };
    let kept = match top.keep {
        Keep::All | Keep::AllText => true,
        Keep::Text => nested == 0 && !top.text_done,
        Keep::Children(_) => false,
    };
    if kept {
        match top.element.children.last_mut() {
            Some(Node::Text(t)) => t.push_str(text),
            _ => top.element.children.push(Node::Text(text.to_string())),
        }
    }
}

/// The element's attributes as lxml names them, kept only when `keep`; checked either way.
fn attributes(resolver: &NamespaceResolver, e: &BytesStart, keep: bool) -> Result<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    for attr in e.attributes() {
        let attr = attr?;
        if attr.key.as_namespace_binding().is_some() {
            continue;
        }
        let name = resolver.resolve_attribute(attr.key);
        if let ResolveResult::Unknown(prefix) = &name.0 {
            bail!("Namespace prefix {} for attribute {} is not defined", String::from_utf8_lossy(prefix), String::from_utf8_lossy(name.1.as_ref()));
        }
        let value = attr.normalized_value(XmlVersion::Implicit1_0)?;
        check_chars(&value)?;
        if keep {
            attrs.push((resolved(name)?, value.into_owned()));
        }
    }
    Ok(attrs)
}

fn resolved((ns, local): (ResolveResult, LocalName)) -> Result<String> {
    let local = std::str::from_utf8(local.as_ref())?;
    Ok(match ns {
        ResolveResult::Bound(ns) => format!("{{{}}}{local}", std::str::from_utf8(ns.as_ref())?),
        _ => local.to_string(),
    })
}

fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

fn check_chars(text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let bad = |i: usize| match bytes[i] {
        b'\t' | b'\n' | b'\r' => false,
        b if b < 0x20 => true,
        0xEF => bytes.get(i + 1) == Some(&0xBF) && matches!(bytes.get(i + 2), Some(0xBE | 0xBF)),
        _ => false,
    };
    if (0..bytes.len()).any(bad) {
        let c = text.chars().find(|c| !is_xml_char(*c)).unwrap_or_default();
        bail!("invalid XML character U+{:04X}", c as u32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(tags: &[&'static str]) -> &'static [(&'static str, Keep)] {
        Box::leak(tags.iter().map(|t| (*t, Keep::All)).collect::<Vec<_>>().into_boxed_slice())
    }

    fn one(xml: &str, tag: &'static str) -> Element {
        Records::new(xml.as_bytes(), all(&[tag])).next_record().unwrap().unwrap()
    }

    #[test]
    fn text_stops_at_the_first_child_node_and_itertext_skips_comment_and_pi_text() {
        let b = one("<a><b>x<!--c-->y<?p q?>w<![CDATA[<k>]]>v<i>in</i>tail</b></a>", "b");
        assert_eq!(b.text(), Some("x"));
        assert_eq!(b.itertext(), "xyw<k>vintail");
        assert_eq!(one("<b><!--c-->x</b>", "b").text(), None);
        assert_eq!(one("<b></b>", "b").text(), None);
        assert_eq!(one("<b><![CDATA[]]></b>", "b").text(), Some(""));
    }

    #[test]
    fn line_ends_attributes_and_references_are_read_as_lxml_reads_them() {
        let b = one("<b L=\"A\tB\nC&#10;D\r\nE\" a=\"&amp;lt;\" c=\"&#38;#38;\">x\r\ny\rz&#13;&amp;&#x3B1;</b>", "b");
        assert_eq!(b.get("L"), Some("A B C\nD E"));
        assert_eq!(b.get("a"), Some("&lt;"));
        assert_eq!(b.get("c"), Some("&#38;"));
        assert_eq!(b.text(), Some("x\ny\nz\r&\u{3B1}"));
    }

    #[test]
    fn namespaced_tags_carry_their_namespace_and_unknown_prefixes_fail() {
        let a = one("<a xmlns:m=\"urn:m\"><m:b>1</m:b><b xmlns=\"urn:x\">2</b><b>3</b></a>", "a");
        let tags: Vec<&str> = a.elements().map(|e| e.tag.as_str()).collect();
        assert_eq!(tags, ["{urn:m}b", "{urn:x}b", "b"]);
        assert_eq!(a.findtext("b"), Some("3"));
        let mut bad = Records::new("<a><b>x<m:math>y</m:math></b></a>".as_bytes(), all(&["a"]));
        assert!(bad.next_record().is_err());
    }

    #[test]
    fn undefined_entities_and_bad_characters_fail() {
        for xml in [
            "<a>&nbsp;</a>",
            "<a>\u{1}</a>",
            "<a>&#1;</a>",
            "<a x=\"&foo;\"/>",
            "",
            " \n",
            "41575209\n",
            "<?xml version=\"1.0\"?>",
            "<a/><b/>",
            "<a><a>x</a></a>y",
            "x<a/>",
            "<a/>&amp;",
            "<a><a>",
        ] {
            let mut found = Records::new(xml.as_bytes(), all(&["a"]));
            assert!((0..3).any(|_| found.next_record().is_err()), "{xml:?}");
        }
    }

    #[test]
    fn paths_follow_elementpath_order() {
        let a = one("<a><j><k>1</k></j><j><k>2</k><k>3</k></j></a>", "a");
        let found: Vec<&str> = a.iterfind("j/k").into_iter().map(|e| e.text().unwrap()).collect();
        assert_eq!(found, ["1", "2", "3"]);
        assert_eq!(a.findtext("j/k"), Some("1"));
        assert_eq!(a.findtext("j/x"), None);
        let first_empty = one("<a><j/><j><k>2</k></j></a>", "a");
        assert_eq!(first_empty.findtext("j/k"), Some("2"));
    }

    #[test]
    fn records_come_in_end_tag_order_with_a_bom_and_doctype_before_them() {
        let xml = "\u{FEFF}<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<!DOCTYPE S PUBLIC \"-//X//EN\" \"x.dtd\">\n<S><R>1</R><D/><x><R>2</R></x></S>";
        let mut found = Records::new(xml.as_bytes(), all(&["R", "D"]));
        let mut seen = Vec::new();
        while let Some(r) = found.next_record().unwrap() {
            seen.push(format!("{}{}", r.tag, r.itertext()));
        }
        assert_eq!(seen, ["R1", "D", "R2"]);
        let mut found = Records::new("<a><R>1</R></a><!--c--> <?p?>\n".as_bytes(), all(&["R"]));
        assert!(found.next_record().unwrap().is_some() && found.next_record().unwrap().is_none());
        assert!(Records::new("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><R/>".as_bytes(), all(&["R"])).next_record().is_err());
    }

    #[test]
    fn kept_parts_are_what_each_keep_says_and_the_rest_is_still_checked() {
        const ROOTS: &[(&str, Keep)] = &[("r", Keep::Children(&[("t", Keep::Text), ("a", Keep::AllText), ("c", Keep::Children(&[("t", Keep::Text)]))]))];
        let xml =
            "<s><r>skip<x><t>no</t></x><t k=\"v\">x<i>in</i>tail</t><t><!--c-->y</t><a>p<i>q<!--z-->r</i>s</a><c>w<t>z</t></c><m:q xmlns:m=\"urn:m\"/></r></s>";
        let r = Records::new(xml.as_bytes(), ROOTS).next_record().unwrap().unwrap();
        type Shown = (String, Vec<(String, String)>, Option<String>, String);
        let shown: Vec<Shown> = r.elements().map(|e| (e.tag.clone(), e.attrs.clone(), e.text().map(str::to_string), e.itertext())).collect();
        let want = [("t", vec![("k", "v")], Some("x"), "x"), ("t", vec![], None, ""), ("a", vec![], Some("pqrs"), "pqrs"), ("c", vec![], None, "z")];
        let want: Vec<Shown> = want
            .iter()
            .map(|(t, a, x, i)| (t.to_string(), a.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), x.map(str::to_string), i.to_string()))
            .collect();
        assert_eq!(shown, want);
        assert_eq!(r.findtext("c/t"), Some("z"));
        for bad in ["<r><x y=\"&foo;\"/></r>", "<r><x><m:q/></x></r>", "<r><x>\u{1}</x></r>", "<r><x a=\"1\" a=\"2\"/></r>"] {
            assert!(Records::new(bad.as_bytes(), ROOTS).next_record().is_err(), "{bad}");
        }
    }
}

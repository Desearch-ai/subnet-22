//! The little HTML handling the frame needs: start tags with their attributes, the body's inner HTML, and Python's escape and unquote.

use std::borrow::Cow;

/// A start tag: its lowercased name and its attributes, values decoded, the first of a repeated name kept as libxml2 keeps it.
#[derive(Debug, PartialEq, Eq)]
pub struct Tag {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    /// The byte just past the tag's `>`.
    pub end: usize,
}

impl Tag {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|i| from + i)
}

fn find_ci(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w.eq_ignore_ascii_case(needle)).map(|i| from + i)
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'\x0c')
}

/// Every start tag of `doc` in document order, outside comments and the raw text of script and style.
pub fn start_tags(doc: &str) -> Vec<Tag> {
    let b = doc.as_bytes();
    let mut tags = Vec::new();
    let mut i = 0;
    while let Some(lt) = b[i..].iter().position(|&c| c == b'<').map(|p| i + p) {
        let rest = &b[lt..];
        if rest.starts_with(b"<!--") {
            i = find(b, b"-->", lt + 4).map_or(b.len(), |end| end + 3);
            continue;
        }
        match rest.get(1) {
            Some(b'!' | b'?' | b'/') => {
                i = find(b, b">", lt + 1).map_or(b.len(), |end| end + 1);
                continue;
            }
            Some(c) if c.is_ascii_alphabetic() => {}
            _ => {
                i = lt + 1;
                continue;
            }
        }
        let mut k = lt + 1;
        while k < b.len() && !is_space(b[k]) && b[k] != b'/' && b[k] != b'>' {
            k += 1;
        }
        let name = doc[lt + 1..k].to_ascii_lowercase();
        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            while k < b.len() && (is_space(b[k]) || b[k] == b'/') {
                k += 1;
            }
            if k >= b.len() {
                break;
            }
            if b[k] == b'>' {
                k += 1;
                break;
            }
            let start = k;
            while k < b.len() && !is_space(b[k]) && b[k] != b'=' && b[k] != b'>' && b[k] != b'/' {
                k += 1;
            }
            let key = doc[start..k].to_ascii_lowercase();
            while k < b.len() && is_space(b[k]) {
                k += 1;
            }
            let mut value = String::new();
            if k < b.len() && b[k] == b'=' {
                k += 1;
                while k < b.len() && is_space(b[k]) {
                    k += 1;
                }
                if k < b.len() && (b[k] == b'"' || b[k] == b'\'') {
                    let quote = b[k];
                    let end = b[k + 1..].iter().position(|&c| c == quote).map_or(b.len(), |p| k + 1 + p);
                    value = decode_entities(&doc[k + 1..end]).into_owned();
                    k = (end + 1).min(b.len());
                } else {
                    let start = k;
                    while k < b.len() && !is_space(b[k]) && b[k] != b'>' {
                        k += 1;
                    }
                    value = decode_entities(&doc[start..k]).into_owned();
                }
            }
            if !key.is_empty() && !attrs.iter().any(|(k, _)| *k == key) {
                attrs.push((key, value));
            }
        }
        let raw = matches!(name.as_str(), "script" | "style");
        let close = format!("</{name}");
        tags.push(Tag { name, attrs, end: k });
        i = if raw { find_ci(b, close.as_bytes(), k).unwrap_or(b.len()) } else { k };
    }
    tags
}

/// Character references in an attribute value: the numeric ones and the few names markup uses; others stay as written.
pub fn decode_entities(value: &str) -> Cow<'_, str> {
    if !value.contains('&') {
        return Cow::Borrowed(value);
    }
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let decoded = after.find(';').filter(|&semi| semi <= 10).and_then(|semi| {
            let name = &after[..semi];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ => {
                    let code = if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        name.strip_prefix('#').and_then(|dec| dec.parse::<u32>().ok())
                    };
                    code.and_then(char::from_u32).filter(|&c| c != '\0')
                }
            };
            c.map(|c| (c, semi))
        });
        match decoded {
            Some((c, semi)) => {
                out.push(c);
                rest = &after[semi + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// `inner` as lxml writes it back out: an empty `<li>` loses its `</li>`, so what follows it is the item's on the next
/// parse. `tags` are `start_tags` of the document `inner` is cut from, `at` bytes in.
pub fn as_lxml_writes<'a>(inner: &'a str, tags: &[Tag], at: usize) -> Cow<'a, str> {
    let cuts: Vec<(usize, usize)> = tags
        .iter()
        .filter(|t| t.name == "li" && t.end >= at && t.end <= at + inner.len())
        .filter_map(|t| {
            let rest = &inner.as_bytes()[t.end - at..];
            let close = rest.get(..4).filter(|b| b.eq_ignore_ascii_case(b"</li"))?;
            let spaces = rest[close.len()..].iter().take_while(|&&b| is_space(b)).count();
            (rest.get(close.len() + spaces) == Some(&b'>')).then_some((t.end - at, close.len() + spaces + 1))
        })
        .collect();
    if cuts.is_empty() {
        return Cow::Borrowed(inner);
    }
    let mut out = String::with_capacity(inner.len());
    let mut copied = 0;
    for (start, len) in cuts {
        out.push_str(&inner[copied..start]);
        copied = start + len;
    }
    out.push_str(&inner[copied..]);
    Cow::Owned(out)
}

/// The HTML between the document's `<body …>` and its last `</body>`, from the first tag on, as lxml's
/// serialisation of the body's children starts; None when there is no body tag to cut at.
pub fn body_inner(doc: &str) -> Option<&str> {
    let head_end = doc.find("</head>").unwrap_or(0);
    let start = head_end + doc[head_end..].find("<body")?;
    let open_end = start + doc[start..].find('>')?;
    let end = doc.rfind("</body>")?;
    if end < open_end {
        return None;
    }
    let inner = &doc[open_end + 1..end];
    Some(inner.find('<').map_or("", |first| &inner[first..]))
}

/// Python's `html.escape(s)`.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(c),
        }
    }
    out
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Python's `urllib.parse.unquote(s)`: each ASCII run percent-decoded and read as UTF-8 with replacement.
pub fn unquote(s: &str) -> Cow<'_, str> {
    if !s.contains('%') {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut run: Vec<u8> = Vec::new();
    let flush = |run: &mut Vec<u8>, out: &mut String| {
        let mut bytes = Vec::with_capacity(run.len());
        let mut i = 0;
        while i < run.len() {
            if run[i] == b'%' {
                if let (Some(hi), Some(lo)) = (run.get(i + 1).copied().and_then(hex_value), run.get(i + 2).copied().and_then(hex_value)) {
                    bytes.push(hi * 16 + lo);
                    i += 3;
                    continue;
                }
            }
            bytes.push(run[i]);
            i += 1;
        }
        out.push_str(&String::from_utf8_lossy(&bytes));
        run.clear();
    };
    for c in s.chars() {
        if c.is_ascii() {
            run.push(c as u8);
        } else {
            flush(&mut run, &mut out);
            out.push(c);
        }
    }
    flush(&mut run, &mut out);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_skip_comments_and_raw_text() {
        let doc = r#"<!DOCTYPE html><html><head><meta property="mw:PageProp/toc"/><style>a<link rel="x"></style></head>
<body><!-- <link rel="hidden"> --><p class=a data-x='1 &amp; 2'>t</p><LINK REL="mw:PageProp/Category" href="./Category:A&amp;B" rel="dup"/><script>if (a<b) {}</script><br/></body></html>"#;
        let tags = start_tags(doc);
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["html", "head", "meta", "style", "body", "p", "link", "script", "br"]);
        assert_eq!(tags[2].get("property"), Some("mw:PageProp/toc"));
        assert_eq!(tags[5].get("data-x"), Some("1 & 2"));
        assert_eq!(tags[5].get("class"), Some("a"));
        assert_eq!(tags[6].get("rel"), Some("mw:PageProp/Category"));
        assert_eq!(tags[6].get("href"), Some("./Category:A&B"));
    }

    #[test]
    fn entities() {
        assert_eq!(decode_entities("a &amp; b &#39;c&#x27; &bogus; &"), "a & b 'c' &bogus; &");
    }

    #[test]
    fn empty_items_lose_their_end_tag() {
        let doc = "<body><ol><li class=\"mw-empty-elt\"></li><li value=\"3\"></LI >\n<dl><dd>t</dd></dl><li> </li><li>x</li></ol></body>";
        let tags = start_tags(doc);
        let inner = body_inner(&format!("<html><head></head>{doc}</html>")).map(str::to_string).unwrap();
        let at = doc.find(&inner).unwrap();
        assert_eq!(as_lxml_writes(&inner, &tags, at), "<ol><li class=\"mw-empty-elt\"><li value=\"3\">\n<dl><dd>t</dd></dl><li> </li><li>x</li></ol>");
    }

    #[test]
    fn body_cut() {
        let doc = "<!DOCTYPE html><html><head><title>x</title></head><body id=\"b\" class=\"c\">\n<section>a</section>\n<p>b</p></body></html>";
        assert_eq!(body_inner(doc), Some("<section>a</section>\n<p>b</p>"));
        assert_eq!(body_inner("<html><head></head></html>"), None);
    }

    #[test]
    fn python_escape_and_unquote() {
        assert_eq!(escape(r#"a&b<c>"d'"#), "a&amp;b&lt;c&gt;&quot;d&#x27;");
        assert_eq!(unquote("Living_people"), "Living_people");
        assert_eq!(unquote("Caf%C3%A9_%26_bar%zz%2"), "Café_&_bar%zz%2");
        assert_eq!(unquote("%C3%28é%41"), "\u{fffd}(éA");
    }
}

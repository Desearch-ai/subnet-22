//! Opinion HTML and Harvard XML as paragraphs of plain text.

const BLOCKS: [&str; 52] = [
    "p",
    "div",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "blockquote",
    "center",
    "section",
    "article",
    "header",
    "footer",
    "hr",
    "dd",
    "dt",
    "dl",
    "body",
    "opinion",
    "author",
    "footnote",
    "footnotes",
    "casebody",
    "parties",
    "docketnumber",
    "court",
    "decisiondate",
    "otherdate",
    "attorneys",
    "headnotes",
    "summary",
    "syllabus",
    "disposition",
    "seealso",
    "judges",
    "citation",
    "correction",
    "page",
    "title",
    "caption",
    "sup-footnote",
    "headmatter",
    "brmapping",
    "line",
    "dissent",
];
const SKIPPED: [&str; 5] = ["script", "style", "head", "page-number", "xml_scan"];

fn named(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" | "ensp" | "emsp" | "thinsp" => " ",
        "shy" | "zwnj" | "zwj" => "",
        "sect" => "§",
        "para" => "¶",
        "mdash" => "—",
        "ndash" => "–",
        "hellip" => "…",
        "lsquo" => "‘",
        "rsquo" => "’",
        "ldquo" => "“",
        "rdquo" => "”",
        "laquo" => "«",
        "raquo" => "»",
        "deg" => "°",
        "frac12" => "½",
        "frac14" => "¼",
        "frac34" => "¾",
        "copy" => "©",
        "reg" => "®",
        "trade" => "™",
        "middot" => "·",
        "times" => "×",
        "bull" => "•",
        "dagger" => "†",
        "Dagger" => "‡",
        "cent" => "¢",
        "pound" => "£",
        "yen" => "¥",
        "euro" => "€",
        "aacute" => "á",
        "agrave" => "à",
        "acirc" => "â",
        "auml" => "ä",
        "eacute" => "é",
        "egrave" => "è",
        "ecirc" => "ê",
        "euml" => "ë",
        "iacute" => "í",
        "igrave" => "ì",
        "iuml" => "ï",
        "oacute" => "ó",
        "ograve" => "ò",
        "ocirc" => "ô",
        "ouml" => "ö",
        "uacute" => "ú",
        "ugrave" => "ù",
        "uuml" => "ü",
        "ntilde" => "ñ",
        "ccedil" => "ç",
        "Eacute" => "É",
        "szlig" => "ß",
        _ => return None,
    })
}

/// Character references decoded: numeric ones and the names opinions use; others stay as written.
pub fn decode(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let decoded = after.find(';').filter(|&semi| semi > 0 && semi <= 10).and_then(|semi| {
            let name = &after[..semi];
            let text = if let Some(code) = name.strip_prefix('#') {
                let code = match code.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => code.parse::<u32>().ok(),
                };
                code.and_then(char::from_u32).filter(|&c| c != '\0').map(String::from)
            } else {
                named(name).map(String::from)
            };
            text.map(|t| (t, semi))
        });
        match decoded {
            Some((t, semi)) => {
                out.push_str(&t);
                rest = &after[semi + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

struct Tag<'a> {
    name: String,
    closing: bool,
    self_closing: bool,
    raw: &'a str,
}

fn tag_at(markup: &str, at: usize) -> Option<(Tag<'_>, usize)> {
    let bytes = markup.as_bytes();
    let mut i = at + 1;
    let closing = bytes.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b':' | b'-')) {
        i += 1;
    }
    if i == start {
        return None;
    }
    let name = markup[start..i].to_ascii_lowercase();
    let mut quote = 0u8;
    while i < bytes.len() {
        let b = bytes[i];
        if quote != 0 {
            if b == quote {
                quote = 0;
            }
        } else if b == b'"' || b == b'\'' {
            quote = b;
        } else if b == b'>' {
            let raw = &markup[at..=i];
            return Some((Tag { name, closing, self_closing: raw.ends_with("/>"), raw }, i + 1));
        }
        i += 1;
    }
    None
}

fn skipped(tag: &Tag) -> bool {
    SKIPPED.contains(&tag.name.as_str()) || (tag.name == "span" && tag.raw.contains("star-pagination"))
}

/// The markup's text, a block element's edge as a blank line and `<pre>` text with its own line breaks.
fn flatten(markup: &str) -> String {
    let mut out = String::with_capacity(markup.len());
    let mut pre = 0usize;
    let mut skip: Option<(String, usize)> = None;
    let mut i = 0;
    let bytes = markup.as_bytes();
    let push_text = |out: &mut String, text: &str, pre: bool| {
        let text = decode(text);
        if pre {
            out.push_str(&text);
        } else {
            let mut space = out.ends_with(|c: char| c.is_whitespace());
            for c in text.chars() {
                if c.is_whitespace() {
                    if !space {
                        out.push(' ');
                        space = true;
                    }
                } else {
                    out.push(c);
                    space = false;
                }
            }
        }
    };
    while i < bytes.len() {
        let next = markup[i..].find('<').map_or(bytes.len(), |n| i + n);
        if next > i {
            if skip.is_none() {
                push_text(&mut out, &markup[i..next], pre > 0);
            }
            i = next;
            continue;
        }
        if markup[i..].starts_with("<!--") {
            i = markup[i..].find("-->").map_or(bytes.len(), |n| i + n + 3);
            continue;
        }
        if markup[i..].starts_with("<!") || markup[i..].starts_with("<?") {
            i = markup[i..].find('>').map_or(bytes.len(), |n| i + n + 1);
            continue;
        }
        let Some((tag, end)) = tag_at(markup, i) else {
            if skip.is_none() {
                push_text(&mut out, "<", pre > 0);
            }
            i += 1;
            continue;
        };
        i = end;
        if let Some((name, depth)) = skip.as_mut() {
            if tag.name == *name && !tag.self_closing {
                if tag.closing {
                    *depth -= 1;
                } else {
                    *depth += 1;
                }
                if *depth == 0 {
                    skip = None;
                }
            }
            continue;
        }
        if !tag.closing && !tag.self_closing && skipped(&tag) {
            skip = Some((tag.name.clone(), 1));
            continue;
        }
        if tag.name == "pre" {
            if tag.closing {
                pre = pre.saturating_sub(1);
            } else if !tag.self_closing {
                pre += 1;
            }
        } else if BLOCKS.contains(&tag.name.as_str()) {
            out.push_str("\n\n");
        } else if tag.name == "td" || tag.name == "th" {
            out.push(' ');
        }
    }
    out
}

/// Paragraphs of the markup's text, each one line: a blank line or an indented line starts the next.
pub fn paragraphs(markup: &str) -> Vec<String> {
    let flat = if markup.contains('<') { flatten(markup) } else { markup.to_string() };
    let mut out = Vec::new();
    let mut held = String::new();
    for line in flat.split(['\n', '\r', '\x0c', '\x0b']) {
        let blank = line.trim().is_empty();
        let indented = line.starts_with("   ") || line.starts_with('\t');
        if (blank || indented) && !held.is_empty() {
            out.push(std::mem::take(&mut held));
        }
        for word in line.split_whitespace() {
            if !held.is_empty() {
                held.push(' ');
            }
            held.push_str(word);
        }
    }
    if !held.is_empty() {
        out.push(held);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_with_citations_as_paragraphs() {
        let html = concat!(
            "<div><p>The judgment is <em>affirmed</em>. See <span class=\"citation\"><a href=\"/opinion/1/x/\">347 U.S. 483</a></span>,",
            " at 495.<span class=\"star-pagination\">*12</span></p><p>It is so ordered &amp; done&nbsp;&sect; 2.</p></div>",
        );
        assert_eq!(paragraphs(html), ["The judgment is affirmed. See 347 U.S. 483, at 495.", "It is so ordered & done § 2."]);
    }

    #[test]
    fn pre_text_split_at_blank_and_indented_lines() {
        let html = "<pre class=\"inline\">OPINION\n\nThe court\nheld that </pre><a href=\"/x\">5 F.2d 1</a><pre class=\"inline\"> applies.\n     Next paragraph\nwraps here.</pre>";
        assert_eq!(paragraphs(html), ["OPINION", "The court held that 5 F.2d 1 applies.", "Next paragraph wraps here."]);
    }

    #[test]
    fn harvard_xml_without_page_numbers() {
        let xml = "<opinion type=\"majority\"><author id=\"b1\">ELLIS, J.</author><p id=\"p1\">This is <page-number citation-index=\"1\" label=\"5\">*5</page-number>an appeal.</p><!-- note --></opinion>";
        assert_eq!(paragraphs(xml), ["ELLIS, J.", "This is an appeal."]);
    }

    #[test]
    fn entities() {
        assert_eq!(decode("a &#167; b &#x2014; c &bogus; &amp;"), "a § b — c &bogus; &");
    }
}

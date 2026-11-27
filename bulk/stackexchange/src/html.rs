//! A post's body HTML as v2 lines: paragraphs, lists, quotes and tables as text, code blocks fenced as written.

use bulk_core::Line;

const FENCE: &str = "```";

enum Token<'a> {
    Open(String),
    Close(String),
    Text(&'a str),
}

fn tag_name(rest: &str) -> String {
    rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase()
}

fn tokens(html: &str) -> Vec<Token<'_>> {
    let mut out = Vec::new();
    let mut at = 0;
    let bytes = html.as_bytes();
    while at < html.len() {
        let Some(lt) = html[at..].find('<').map(|i| at + i) else {
            out.push(Token::Text(&html[at..]));
            break;
        };
        if lt > at {
            out.push(Token::Text(&html[at..lt]));
        }
        let rest = &html[lt + 1..];
        if rest.starts_with("!--") {
            at = rest.find("-->").map_or(html.len(), |i| lt + 1 + i + 3);
            continue;
        }
        let closing = rest.starts_with('/');
        let name = tag_name(if closing { &rest[1..] } else { rest });
        if name.is_empty() || !bytes[lt + 1 + usize::from(closing)].is_ascii_alphabetic() {
            out.push(Token::Text(&html[lt..lt + 1]));
            at = lt + 1;
            continue;
        }
        let mut quote = None;
        let mut end = html.len();
        for (i, c) in rest.char_indices() {
            match (quote, c) {
                (Some(q), c) if c == q => quote = None,
                (Some(_), _) => {}
                (None, '"' | '\'') => quote = Some(c),
                (None, '>') => {
                    end = lt + 1 + i + 1;
                    break;
                }
                _ => {}
            }
        }
        out.push(if closing { Token::Close(name) } else { Token::Open(name) });
        at = end;
    }
    out
}

fn entity(name: &str) -> Option<char> {
    if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
        return u32::from_str_radix(hex, 16).ok().and_then(char::from_u32);
    }
    if let Some(dec) = name.strip_prefix('#') {
        return dec.parse().ok().and_then(char::from_u32);
    }
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "hellip" => '…',
        "mdash" => '—',
        "ndash" => '–',
        "lsquo" => '‘',
        "rsquo" => '’',
        "ldquo" => '“',
        "rdquo" => '”',
        "laquo" => '«',
        "raquo" => '»',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "times" => '×',
        "divide" => '÷',
        "deg" => '°',
        "plusmn" => '±',
        "middot" => '·',
        "bull" => '•',
        "euro" => '€',
        "pound" => '£',
        "yen" => '¥',
        "cent" => '¢',
        "sect" => '§',
        "para" => '¶',
        "larr" => '←',
        "rarr" => '→',
        "uarr" => '↑',
        "darr" => '↓',
        "harr" => '↔',
        "le" => '≤',
        "ge" => '≥',
        "ne" => '≠',
        "minus" => '−',
        "zwj" => '\u{200d}',
        "zwnj" => '\u{200c}',
        "shy" => '\u{ad}',
        _ => return None,
    })
}

/// HTML character references decoded; an unknown one is left as written.
pub fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp + 1..];
        let name_len = tail.find(';').filter(|&i| i > 0 && i <= 10 && tail[..i].chars().all(|c| c.is_ascii_alphanumeric() || c == '#'));
        match name_len.and_then(|i| entity(&tail[..i]).map(|c| (i, c))) {
            Some((i, c)) => {
                out.push(c);
                rest = &tail[i + 1..];
            }
            None => {
                out.push('&');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    out
}

struct List {
    ordered: bool,
    next: usize,
}

#[derive(Default)]
struct Render {
    lines: Vec<Line>,
    inline: String,
    marker: Option<String>,
    heading: bool,
    quote: usize,
    lists: Vec<List>,
    pre: Option<String>,
    table: Option<Vec<(bool, Vec<String>)>>,
    cell: Option<String>,
    skip: usize,
}

impl Render {
    fn indent(&self) -> String {
        "  ".repeat(self.lists.len().saturating_sub(1))
    }

    fn push_text(&mut self, text: &str) {
        if self.skip > 0 {
            return;
        }
        if let Some(pre) = self.pre.as_mut() {
            pre.push_str(&unescape(text));
            return;
        }
        let target = match self.cell.as_mut() {
            Some(cell) => cell,
            None if self.table.is_some() => return,
            None => &mut self.inline,
        };
        for c in unescape(text).chars() {
            if c.is_whitespace() {
                if !target.is_empty() && !target.ends_with(' ') {
                    target.push(' ');
                }
            } else {
                target.push(c);
            }
        }
    }

    fn push_mark(&mut self, mark: &str) {
        if self.skip > 0 || self.pre.is_some() {
            return;
        }
        match self.cell.as_mut() {
            Some(cell) => cell.push_str(mark),
            None => self.inline.push_str(mark),
        }
    }

    fn close_mark(&mut self) {
        if self.skip > 0 || self.pre.is_some() {
            return;
        }
        let target = self.cell.as_mut().unwrap_or(&mut self.inline);
        if target.ends_with('`') {
            target.pop();
        } else {
            target.push('`');
        }
    }

    fn flush(&mut self) {
        let text = self.inline.trim().to_string();
        self.inline.clear();
        if text.is_empty() {
            return;
        }
        let mut line = "> ".repeat(self.quote);
        if self.heading {
            line.push_str("### ");
        } else if let Some(marker) = self.marker.take() {
            line.push_str(&marker);
        } else if !self.lists.is_empty() {
            line.push_str(&self.indent());
            line.push_str("  ");
        }
        line.push_str(&text);
        self.lines.push(Line::new(line));
    }

    fn close_pre(&mut self) {
        let Some(code) = self.pre.take() else { return };
        let code = code.trim_end_matches(['\n', '\r', ' ', '\t']);
        let code = code.trim_start_matches(['\n', '\r']);
        if code.trim().is_empty() {
            return;
        }
        let fenced = |text: &str| Line { text: text.to_string(), verbatim: true, ..Line::default() };
        self.lines.push(fenced(FENCE));
        for line in code.lines() {
            self.lines.push(fenced(line.trim_end()));
        }
        self.lines.push(fenced(FENCE));
    }

    fn close_table(&mut self) {
        self.close_cell();
        let Some(rows) = self.table.take() else { return };
        let width = rows.iter().map(|(_, cells)| cells.len()).max().unwrap_or(0);
        if width == 0 {
            return;
        }
        let row_line = |cells: &[String]| {
            let mut padded: Vec<String> = cells.iter().map(|c| c.trim().replace('|', "\\|")).collect();
            padded.resize(width, String::new());
            Line { text: format!("| {} |", padded.join(" | ")), row: true, ..Line::default() }
        };
        for (i, (header, cells)) in rows.iter().enumerate() {
            self.lines.push(row_line(cells));
            if i == 0 && *header {
                self.lines.push(Line { text: format!("|{}", " --- |".repeat(width)), row: true, ..Line::default() });
            }
        }
    }

    fn close_cell(&mut self) {
        if let (Some(cell), Some(rows)) = (self.cell.take(), self.table.as_mut()) {
            match rows.last_mut() {
                Some((_, cells)) => cells.push(cell),
                None => rows.push((false, vec![cell])),
            }
        }
    }

    fn open(&mut self, name: &str) {
        if matches!(name, "script" | "style") {
            self.skip += 1;
            return;
        }
        if self.pre.is_some() {
            return;
        }
        match name {
            "p" | "div" | "dl" | "dt" | "dd" | "hr" | "center" | "details" | "summary" => self.flush(),
            "br" => match self.cell.as_mut() {
                Some(cell) => cell.push(' '),
                None => self.flush(),
            },
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.flush();
                self.heading = true;
            }
            "blockquote" => {
                self.flush();
                self.quote += 1;
            }
            "ul" | "ol" => {
                self.flush();
                self.lists.push(List { ordered: name == "ol", next: 1 });
            }
            "li" => {
                self.flush();
                let indent = self.indent();
                let marker = match self.lists.last_mut() {
                    Some(List { ordered: true, next }) => {
                        *next += 1;
                        format!("{indent}{}. ", *next - 1)
                    }
                    _ => format!("{indent}- "),
                };
                self.marker = Some(marker);
            }
            "pre" => {
                self.flush();
                self.pre = Some(String::new());
            }
            "code" | "kbd" | "tt" => self.push_mark("`"),
            "table" => {
                self.flush();
                self.table = Some(Vec::new());
            }
            "tr" => {
                self.close_cell();
                if let Some(rows) = self.table.as_mut() {
                    rows.push((false, Vec::new()));
                }
            }
            "td" | "th" => {
                self.close_cell();
                if let Some(rows) = self.table.as_mut() {
                    if rows.is_empty() {
                        rows.push((false, Vec::new()));
                    }
                    if name == "th" {
                        if let Some(row) = rows.last_mut() {
                            row.0 = true;
                        }
                    }
                    self.cell = Some(String::new());
                }
            }
            _ => {}
        }
    }

    fn close(&mut self, name: &str) {
        if matches!(name, "script" | "style") {
            self.skip = self.skip.saturating_sub(1);
            return;
        }
        if self.pre.is_some() && name != "pre" {
            return;
        }
        match name {
            "p" | "div" | "dl" | "dt" | "dd" | "li" | "center" | "details" | "summary" => self.flush(),
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.flush();
                self.heading = false;
            }
            "blockquote" => {
                self.flush();
                self.quote = self.quote.saturating_sub(1);
            }
            "ul" | "ol" => {
                self.flush();
                self.lists.pop();
            }
            "pre" => self.close_pre(),
            "code" | "kbd" | "tt" => self.close_mark(),
            "td" | "th" => self.close_cell(),
            "table" => self.close_table(),
            _ => {}
        }
    }
}

/// The body's lines, as a post renders them.
pub fn lines(html: &str) -> Vec<Line> {
    let mut render = Render::default();
    for token in tokens(html) {
        match token {
            Token::Open(name) => render.open(&name),
            Token::Close(name) => render.close(&name),
            Token::Text(text) => render.push_text(text),
        }
    }
    render.close_pre();
    render.close_table();
    render.flush();
    render.lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(html: &str) -> Vec<String> {
        lines(html).into_iter().map(|l| l.text).collect()
    }

    #[test]
    fn paragraphs_code_and_lists() {
        let html = "<p>How do I use <code>git reset</code> &amp; keep files?</p>\n\n<pre class=\"lang-bash\"><code>git reset --soft HEAD~1\n  echo &quot;x &lt; y&quot;\n</code></pre>\n\n<ul>\n<li>one <strong>bold</strong></li>\n<li>two<ol><li>inner</li></ol></li>\n</ul>";
        assert_eq!(
            texts(html),
            vec!["How do I use `git reset` & keep files?", "```", "git reset --soft HEAD~1", "  echo \"x < y\"", "```", "- one bold", "- two", "  1. inner"]
        );
        assert!(lines(html)[2].verbatim);
    }

    #[test]
    fn quotes_headings_breaks_and_tables() {
        let html = "<blockquote><p>quoted<br>text</p></blockquote><h2>Update</h2><p>a&nbsp;b &#39;c&#x27; &bogus; 1 < 2</p><table><thead><tr><th>A</th><th>B|C</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table><!-- note --><hr/>";
        assert_eq!(texts(html), vec!["> quoted", "> text", "### Update", "a b 'c' &bogus; 1 < 2", "| A | B\\|C |", "| --- | --- |", "| 1 | 2 |"]);
    }

    #[test]
    fn empty_code_and_images_dropped() {
        assert_eq!(texts("<p><img src=\"x.png\" alt=\"enter image description here\"></p><pre><code>\n</code></pre><p>after</p>"), vec!["after"]);
    }
}

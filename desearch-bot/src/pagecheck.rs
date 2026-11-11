//! What the crawler build does around extraction: the FRED table rewrite, wall and mirror checks, EDGAR items.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::pystr;

pub const MAX_TEXT_CHARS: usize = 200_000;
pub const WALL_CHARS: usize = 3000;
pub const MIRROR_CHARS: usize = 3000;
pub const MIN_SECTION_CHARS: usize = 400;
pub const SECTIONED: &[&str] = &["10-K", "10-Q", "20-F"];
const MAX_HEADINGS: usize = 30;
const MAX_HEADING_CHARS: usize = 300;
const HEADING_MARKS: &[&str] = &["# ", "## ", "### "];
const FENCE: &str = "```";

static FRED_DATA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^https?://fred\.stlouisfed\.org/data/[A-Za-z0-9_.-]+/?\n?$").unwrap());
static FRED_TABLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?s)<table id="data-table-observations".*?</table>"#).unwrap());
static FRED_ROW: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<tr>\s*<th scope="row"[^>]*>\s*([^<]+?)\s*</th>\s*<td[^>]*>\s*([^<]*?)\s*</td>\s*</tr>"#).unwrap()
});
static FRED_EXTRA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?s)<div id="extra-rows">(.*?)</div>"#).unwrap());
static WALL_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\W*(making sure you('|’)re not a bot|human verification|browser verification|bot verification|access restricted in your area|consent required|verify(ing)? your browser|security verification|just a moment|attention required|pardon our interruption|one more step|ddos-guard)\b",
    )
    .unwrap()
});
const WALL_PHRASES: &[&str] = &[
    "anubis could not load",
    "set up anubis to protect",
    "protected by xpguard",
    "uses xpguard",
    "consent required in your area",
    "access restricted in your area",
    "our automated systems indicate that you are accessing this website from",
    "your request was blocked because our system detected unusual activity",
    "confirm that your browser supports the security features",
    "running browser verification",
];
static MIRROR_CORE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?:simple[\s_-]?qa|seal[\s_-]?qa|browse[\s_-]?comp|frames[\s_-]benchmark)").unwrap());
static MIRROR_EVALS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)simple[\s_-]?evals").unwrap());
static ITEM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^\W*item\s+(\d{1,2}[a-c]?)\b[\s.:—–-]*(.{0,120})$").unwrap());

pub fn is_fred_data(url: &str) -> bool {
    FRED_DATA.is_match(url)
}

/// Python's `html.escape(s, quote=True)`.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

/// A FRED /data/ page with every observation in its table, the latest first, and the hidden extra rows removed.
pub fn fred_newest_first(page: &str) -> String {
    let Some(table) = FRED_TABLE.find(page) else { return page.to_string() };
    let mut rows: Vec<(String, String)> =
        FRED_ROW.captures_iter(table.as_str()).map(|c| (c[1].to_string(), c[2].to_string())).collect();
    if let Some(extra) = FRED_EXTRA.captures(page) {
        for part in extra[1].split('#').skip(1) {
            let (day, value) = part.split_once('|').unwrap_or((part, ""));
            if !pystr::strip(day).is_empty() {
                rows.push((pystr::strip(day).to_string(), pystr::strip(value).to_string()));
            }
        }
    }
    if rows.is_empty() {
        return page.to_string();
    }
    let body: String =
        rows.iter().rev().map(|(d, v)| format!(r#"<tr><th scope="row">{}</th><td>{}</td></tr>"#, escape(d), escape(v))).collect();
    let rebuilt = format!(
        r#"<table id="data-table-observations"><thead><tr><th scope="col">DATE</th><th scope="col">VALUE</th></tr></thead><tbody>{body}</tbody></table>"#
    );
    let out = format!("{}{}{}", &page[..table.start()], rebuilt, &page[table.end()..]);
    FRED_EXTRA.replacen(&out, 1, "").into_owned()
}

/// A short page whose title or body is a bot check or a regional consent wall.
pub fn is_wall(title: &str, text: &str) -> bool {
    if pystr::len(text) > WALL_CHARS {
        return false;
    }
    if WALL_TITLE.is_match(&pystr::squash(title)) {
        return true;
    }
    let low = pystr::squash(text).to_lowercase();
    WALL_PHRASES.iter().any(|p| low.contains(p))
}

/// A character `[a-z0-9]` matches in Python under IGNORECASE.
fn mirror_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '\u{130}' | '\u{131}' | '\u{17f}' | '\u{212a}')
}

/// The search-evals harness's test for pages that copy a benchmark's own questions.
pub fn is_mirror(s: &str) -> bool {
    if MIRROR_EVALS.is_match(s) {
        return true;
    }
    let mut before: Option<char> = None;
    for (i, c) in s.char_indices() {
        if !before.is_some_and(mirror_word) {
            if let Some(m) = MIRROR_CORE.find(&s[i..]) {
                if !s[i + m.end()..].chars().next().is_some_and(mirror_word) {
                    return true;
                }
            }
        }
        before = Some(c);
    }
    false
}

/// The text's h1 to h3 lines without their marks, outside fenced code.
pub fn text_headings(text: &str) -> Vec<String> {
    let lines = pystr::splitlines(text);
    let mut found = Vec::new();
    let mut i = 0;
    while i < lines.len() && found.len() < MAX_HEADINGS {
        let line = lines[i];
        if line.starts_with(FENCE) {
            if let Some(close) = (i + 1..lines.len()).find(|&j| lines[j].starts_with(FENCE)) {
                i = close + 1;
                continue;
            }
        }
        if HEADING_MARKS.iter().any(|m| line.starts_with(m)) {
            let rest = line.split_once(' ').map_or("", |(_, r)| r);
            let heading = pystr::cut(pystr::strip(rest), MAX_HEADING_CHARS);
            if !heading.is_empty() {
                found.push(heading.to_string());
            }
        }
        i += 1;
    }
    found
}

/// A 10-K/10-Q text as (item, text) sections, skipping the table of contents' list of items.
pub fn item_sections(text: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.split('\n').collect();
    let marks: Vec<(usize, String)> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !pystr::lstrip(line).starts_with('|'))
        .filter_map(|(i, line)| {
            let c = ITEM.captures(pystr::strip_chars(line, "#* "))?;
            (!pystr::strip_chars(&c[2], " .:—–-").is_empty()).then(|| (i, c[1].to_uppercase()))
        })
        .collect();
    if marks.len() < 2 {
        return Vec::new();
    }
    let spans: Vec<(String, String)> = marks
        .iter()
        .enumerate()
        .map(|(k, (i, item))| {
            let end = marks.get(k + 1).map_or(lines.len(), |m| m.0);
            (item.clone(), pystr::strip(&lines[*i..end].join("\n")).to_string())
        })
        .collect();
    let mut best: HashMap<&str, (usize, &str)> = HashMap::new();
    for (at, (item, body)) in spans.iter().enumerate() {
        let n = pystr::len(body);
        if n >= MIN_SECTION_CHARS && n >= best.get(item.as_str()).map_or(0, |b| pystr::len(b.1)) {
            best.insert(item, (at, body));
        }
    }
    let mut out: Vec<(usize, String, String)> = best.into_iter().map(|(item, (at, body))| (at, item.to_string(), body.to_string())).collect();
    out.sort_by_key(|(at, _, _)| *at);
    out.into_iter().map(|(_, item, body)| (item, body)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRED: &str = r#"<table id="data-table-observations" aria-labelledby="t" class="table">
<thead><tr><th scope="col">DATE</th><th scope="col">VALUE</th></tr></thead><tbody>
<tr>
  <th scope="row" class="pe-5">1962-01-02</th>
  <td class="pe-5"> 4.06</td>
</tr>
<tr>
  <th scope="row" class="pe-5">1962-01-03</th>
  <td class="pe-5"> 4.03</td>
</tr>
</tbody>
</table>
<div id="extra-rows">
    #1962-01-04| 3.99
    #1962-01-05|    .
    #2026-11-09| 4.88
</div><script>let rows = 1;</script>"#;

    #[test]
    fn fred_rows_come_newest_first_with_the_hidden_rows() {
        let out = fred_newest_first(FRED);
        let at: Vec<usize> = ["2026-11-09", "1962-01-05", "1962-01-04", "1962-01-03", "1962-01-02"].iter().map(|d| out.find(d).unwrap()).collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]));
        assert!(!out.contains("extra-rows") && out.contains("<td>4.88</td>") && out.contains("<td>.</td>"));
        assert_eq!(fred_newest_first("<html>no table</html>"), "<html>no table</html>");
        assert!(is_fred_data("https://fred.stlouisfed.org/data/DGS10"));
        assert!(!is_fred_data("https://fred.stlouisfed.org/series/DGS10"));
    }

    #[test]
    fn walls_and_their_lookalikes() {
        assert!(is_wall("Making sure you're not a bot!", "Loading..."));
        assert!(is_wall("UBC", "Your request was blocked because our system detected unusual activity."));
        assert!(!is_wall("How Anubis protects small sites", &"Anubis could not load its JavaScript. ".repeat(100)));
        assert!(!is_wall("Opencoarrays - Fortran Discourse", "based on Making sure you're not a bot! this wiki"));
    }

    #[test]
    fn mirrors_need_word_edges() {
        assert!(is_mirror("the SimpleQA set"));
        assert!(is_mirror("seal_qa"));
        assert!(is_mirror("xsimple-evals"));
        assert!(!is_mirror("xsimpleqa"));
        assert!(!is_mirror("simpleqas"));
        assert!(is_mirror("frames benchmark"));
        assert!(!is_mirror("framesbenchmark"));
    }

    #[test]
    fn headings_skip_fenced_code() {
        let text = "# Title\n```\n# not\n```\n## Two \n#nope\n### \n```\n# open fence";
        assert_eq!(text_headings(text), vec!["Title", "Two", "open fence"]);
    }

    #[test]
    fn items_skip_the_table_of_contents() {
        let body = "word ".repeat(100);
        let text = format!("| Item 1. Business | 3 |\nItem 1. Business\nItem 1A. Risk Factors\n\nITEM 1. BUSINESS\n{body}\nItem 1A. Risk Factors\n{body}\nItem 2\n");
        let got = item_sections(&text);
        assert_eq!(got.iter().map(|(i, _)| i.as_str()).collect::<Vec<_>>(), vec!["1", "1A"]);
        assert!(got[0].1.starts_with("ITEM 1. BUSINESS"));
    }
}

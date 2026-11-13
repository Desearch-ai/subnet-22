//! A paper's body lines and the patterns that find them, with Python's `\s`, `\w` and `\b`.

use std::sync::LazyLock;

use desearch::text::{collapse, fold, is_space, is_word, strip};
use regex::Regex;

pub const AFFILIATION_CHARS: usize = 500;
pub const REFERENCES: [&str; 4] = ["references", "bibliography", "references and notes", "literature cited"];

// Python's `\s` is `str.isspace`, which also holds \x1c-\x1f.
static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(#{1,6})(?:[\s\x1c-\x1f]+(.*?))?[\s\x1c-\x1f]*$").unwrap());
static NUMBERING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:\d+(?:\.\d+)*|[ivxlc]+)\.?[\s\x1c-\x1f]+").unwrap());
static ITEM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\s\x1c-\x1f]*(?:[*+-]|\d+\.)[\s\x1c-\x1f]").unwrap());
static EMAIL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+(?:\.[\p{L}\p{N}_-]+)+").unwrap());
static ESCAPED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\([.*_#\[\]()+!-])").unwrap());

/// `HEADING.match(row)` as its level and name, the name empty when absent.
fn heading(row: &str) -> Option<(usize, &str)> {
    if !row.starts_with('#') {
        return None;
    }
    let found = HEADING.captures(row)?;
    Some((found[1].len(), found.get(2).map_or("", |name| name.as_str())))
}

/// `ABSTRACT.match(row)`: `^\s*abstract\b`, ignoring case.
fn opens_with_abstract(row: &str) -> bool {
    let mut chars = row.trim_start_matches(is_space).chars();
    for want in "abstract".chars() {
        match chars.next() {
            Some(c) if fold(c) == want => {}
            _ => return false,
        }
    }
    !chars.next().is_some_and(is_word)
}

/// The body of a paper's Markdown, from its abstract to its references.
pub fn body_lines(markdown: &str) -> Vec<String> {
    let rows: Vec<&str> = markdown.split('\n').collect();
    let headings: Vec<usize> = (0..rows.len()).filter(|&i| heading(rows[i]).is_some()).collect();
    let abstract_at = headings
        .iter()
        .copied()
        .find(|&i| heading(rows[i]).is_some_and(|(_, name)| name.to_lowercase().starts_with("abstract")))
        .or_else(|| rows.iter().position(|row| opens_with_abstract(row)));
    let start = abstract_at.unwrap_or_else(|| headings.get(1).copied().unwrap_or(1));
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for row in &rows[start.min(rows.len())..] {
        let found = heading(row);
        let stripped = strip(row);
        if (found.is_some() || stripped.is_empty() || ITEM.is_match(row) || row.trim_start_matches(is_space).starts_with('|')) && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if let Some((level, name)) = found {
            if name.is_empty() {
                continue;
            }
            let lowered = name.to_lowercase();
            let key = NUMBERING.replace(&lowered, "");
            let key = key.trim_matches([' ', '.', ':']);
            if REFERENCES.contains(&key) {
                break;
            }
            if key.starts_with("abstract") {
                out.push("## Abstract".to_string());
            } else if level <= 4 {
                out.push(format!("## {name}"));
            } else {
                out.push(name.to_string());
            }
        } else if !stripped.is_empty() {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(stripped);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    let mut kept = Vec::with_capacity(out.len());
    for line in &out {
        let line = ESCAPED.replace_all(line, "$1");
        if EMAIL.is_match(&line) {
            if line.chars().count() < AFFILIATION_CHARS {
                continue;
            }
            kept.push(collapse(&EMAIL.replace_all(&line, "")));
        } else {
            kept.push(line.into_owned());
        }
    }
    kept
}

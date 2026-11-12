//! The v2 text conventions of `desearch.extraction` that bulk pages go through: line cleaning, finishing and headings.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::LazyLock;

use desearch::text::{fold_turkish_i, is_space, is_word, strip, words};
use regex::Regex;

use crate::casefold::FOLDS;

pub const MAX_TEXT_CHARS: usize = 200_000;
pub const MIN_LINE_CHARS: usize = 2;
pub const MIN_REPEAT_WORDS: usize = 8;
pub const MAX_CONTROL_WORDS: usize = 8;
pub const MAX_INTERFACE_WORDS: usize = 30;
pub const MIN_BLOB_CHARS: usize = 300;
pub const MAX_HEADINGS: usize = 30;
pub const MAX_HEADING_CHARS: usize = 300;
pub const FENCE: &str = "```";
pub const HEADING_MARKS: [&str; 3] = ["# ", "## ", "### "];

/// One line of rendered text, as `desearch.extraction.text.markdown.Line`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Line {
    pub text: String,
    pub linked: bool,
    pub verbatim: bool,
    pub row: bool,
}

impl Line {
    pub fn new(text: impl Into<String>) -> Self {
        Line { text: text.into(), ..Line::default() }
    }
}

// Python's `\b` after a word character, `\W` and `\w`; the regex crate's word class also holds marks and connectors.
const END: &str = r"(?-i:[^\p{L}\p{N}_]|\z)";
const NOT_WORD: &str = r"(?-i:[^\p{L}\p{N}_])";
const WORD: &str = r"(?-i:[\p{L}\p{N}_])";
const SPACE: &str = r"[\s\x1c-\x1f]";

fn pattern(text: &str) -> Regex {
    let text = text.replace("{END}", END).replace("{NOT_WORD}", NOT_WORD).replace("{WORD}", WORD).replace("{SPACE}", SPACE);
    Regex::new(&text).expect("pattern compiles")
}

static SKIP_LINK: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)^skip to (?:the )?(?:main )?(?:content|navigation|search|product(?: information)?){END}"));
static COPYRIGHT: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)^(?:©|\(c\)|copyright{END}|all rights reserved)"));
static CALL_TO_ACTION: LazyLock<Regex> = LazyLock::new(|| {
    pattern(concat!(
        r"(?i)^(?:(?:follow (?:us|me|topics|along)|subscribe|sign up|sign in|log in|register|donate|support (?:local|our|independent)",
        r"|become a (?:member|subscriber|supporter)|get (?:the|our) (?:app|newsletter)|download (?:the|our) app",
        r"|listen to (?:this|the) (?:article|story)|this (?:audio|article) (?:is|was) (?:auto|generated|read)",
        r"|add .{1,40}(?:google|preferred source)|want more|don.?t miss){END}|follow @{WORD})",
    ))
});
static GENERIC_ACTION: LazyLock<Regex> = LazyLock::new(|| {
    pattern(concat!(
        r"(?i)^(?:more|more info(?:rmation)?|more details|details|view details|read more|learn more|see more|view more",
        r"|show more|load more|find out more|discover more|explore(?: all| more)?|view all|see all|show all",
        r"|browse all|continue(?: reading)?|click here|get started|start now|shop now|buy now|order now",
        r"|book now|apply now|join now|watch now){NOT_WORD}*\z",
    ))
});
static AUDIO_PLAYER: LazyLock<Regex> = LazyLock::new(|| {
    pattern(concat!(
        r"(?i)^(?:getting your .{0,30}audio player ready|click here to listen to this (?:article|story)",
        r"|(?:this|the) (?:audio|article|story) is (?:auto-generated|read by an automated voice)",
        r"|this is read by an automated voice|please report any issues or inconsistencies here)",
    ))
});
static INTERFACE_LINE: LazyLock<Regex> = LazyLock::new(|| {
    pattern(concat!(
        r"(?i)^(?:you are using an? (?:out[- ]of[- ]date|unsupported|outdated) browser|you should upgrade or use an alternative browser",
        r"|it seems that you.?re using an outdated browser|please use .{1,20} browser for",
        r"|an official website of the united states|a \.gov website belongs to an official|share sensitive information only on official",
        r"|double tap to zoom|use (?:the )?left/right arrows to navigate|listen to (?:this|the) (?:article|story){NOT_WORD}*\z",
        r"|affiliate disclosure{END}|this (?:article|post|page) (?:may )?contains? affiliate links",
        r"|when you (?:buy|purchase|shop) (?:through|via|using) links (?:on|in) our",
        r"|(?:website|web design|site) by{END}|government websites by{END}|powered by wordpress)",
    ))
});
static GALLERY_COUNTER: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)^(?:image|photo|slide|item|picture) \d+ (?:of|/) \d+\n?\z"));
static JSON_TEXT: LazyLock<Regex> = LazyLock::new(|| pattern(r#"^[\[{]+{SPACE}*"[^"]{1,100}"{SPACE}*:"#));
static AD_LABEL: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(?i)^(?:advertisements?|sponsored(?: content)?|paid (?:post|content)|story continues below(?: advertisement)?)(?: \d+)?\n?\z"));
static VALUE: LazyLock<Regex> = LazyLock::new(|| pattern(r"[\d$€£¥%]"));

/// Python's `str.isalnum` on one character.
pub fn is_alnum(c: char) -> bool {
    c != '_' && is_word(c)
}

fn word_count(s: &str) -> usize {
    words(s).count()
}

/// `\S{300,}`: a run of 300 characters without whitespace.
fn has_blob(text: &str) -> bool {
    let mut run = 0;
    for c in text.chars() {
        run = if is_space(c) { 0 } else { run + 1 };
        if run >= MIN_BLOB_CHARS {
            return true;
        }
    }
    false
}

fn repeatable(line: &Line) -> bool {
    word_count(&line.text) < MIN_REPEAT_WORDS
}

fn dropped(line: &Line, cards: bool, seen: &HashSet<String>) -> bool {
    let bare = line.text.trim_matches(['-', '#', '|', '>', ' ']);
    let folded = fold_turkish_i(bare);
    let folded = folded.as_ref();
    let bare_words = word_count(bare);
    (bare.chars().count() <= MIN_LINE_CHARS && !(cards && VALUE.is_match(bare)))
        || (cards && !bare.chars().any(is_alnum))
        || SKIP_LINK.is_match(folded)
        || AD_LABEL.is_match(folded)
        || JSON_TEXT.is_match(bare)
        || AUDIO_PLAYER.is_match(folded)
        || has_blob(&line.text)
        || (COPYRIGHT.is_match(folded) && (cards || bare_words <= MAX_CONTROL_WORDS * 2))
        || (bare_words <= MAX_INTERFACE_WORDS && INTERFACE_LINE.is_match(folded))
        || GALLERY_COUNTER.is_match(folded)
        || (cards && bare_words < MIN_REPEAT_WORDS && CALL_TO_ACTION.is_match(folded))
        || (cards && GENERIC_ACTION.is_match(folded))
        || (seen.contains(&line.text) && !(cards && repeatable(line)))
}

/// `desearch.extraction.text.lines.clean`.
pub fn clean(lines: Vec<Line>, cards: bool) -> Vec<Line> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut tables: HashSet<Vec<String>> = HashSet::new();
    let mut kept = Vec::with_capacity(lines.len());
    let mut lines = lines.into_iter().peekable();
    while let Some(line) = lines.next() {
        if line.row {
            let mut table = vec![line];
            while let Some(next) = lines.next_if(|l| l.row) {
                table.push(next);
            }
            let rows: Vec<String> = table.iter().map(|l| l.text.clone()).collect();
            if !cards || !tables.contains(&rows) {
                tables.insert(rows);
                kept.extend(table);
            }
            continue;
        }
        if line.verbatim {
            kept.push(line);
            continue;
        }
        if dropped(&line, cards, &seen) {
            continue;
        }
        seen.insert(line.text.clone());
        kept.push(line);
    }
    kept
}

/// `desearch.extraction.text.markdown.heading_level`.
pub fn heading_level(line: &Line) -> usize {
    let marks = line.text.len() - line.text.trim_start_matches('#').len();
    if (1..=6).contains(&marks) && line.text[marks..].starts_with(' ') {
        marks
    } else {
        0
    }
}

/// Python's `str.casefold`.
pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match FOLDS.binary_search_by_key(&c, |(from, _)| *from) {
            Ok(i) => out.push_str(FOLDS[i].1),
            Err(_) => out.push(c),
        }
    }
    out
}

/// `desearch.extraction.text.markdown.with_title`: the title first unless the first line already says it.
pub fn with_title<'a>(title: &'a str, lines: &[&'a str]) -> Vec<&'a str> {
    let first = lines.first().map_or("", |line| strip(line.trim_start_matches('#')));
    if !title.is_empty() && casefold(first) != casefold(title) {
        let mut out = Vec::with_capacity(lines.len() + 1);
        out.push(title);
        out.extend_from_slice(lines);
        return out;
    }
    lines.to_vec()
}

/// `desearch.extraction.text.markdown.finish`: the title line, then the lines, cut at a line end within `cap` characters.
pub fn finish(title: &str, lines: &[Line], cap: usize) -> String {
    let texts: Vec<&str> = lines.iter().map(|line| line.text.as_str()).collect();
    let text = with_title(title, &texts).join("\n");
    let Some((end, _)) = text.char_indices().nth(cap) else {
        return text;
    };
    let cut = text[..end].rfind('\n').filter(|&cut| cut > 0).unwrap_or(end);
    text[..cut].to_string()
}

/// Python's `str.splitlines()`.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            continue;
        }
        out.push(&text[start..i]);
        start = i + c.len_utf8();
        if c == '\r' && chars.next_if(|&(_, n)| n == '\n').is_some() {
            start += 1;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `desearch.extraction.text_headings`.
pub fn text_headings(text: &str) -> Vec<String> {
    let lines = splitlines(text);
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
        if HEADING_MARKS.iter().any(|mark| line.starts_with(mark)) {
            let (_, rest) = line.split_once(' ').unwrap_or((line, ""));
            let heading = desearch::text::head(strip(rest), MAX_HEADING_CHARS);
            if !heading.is_empty() {
                found.push(heading.to_string());
            }
        }
        i += 1;
    }
    found
}

/// JSON text with lone surrogate escapes dropped, as Python's json.loads then `without_surrogates` leave the strings.
pub fn drop_lone_surrogates(json: &str) -> Cow<'_, str> {
    replace_lone_surrogates(json, "")
}

/// JSON text with each lone surrogate escape replaced, as `str.encode("utf-8", "replace")` turns one into `?`.
pub fn replace_lone_surrogates<'a>(json: &'a str, with: &str) -> Cow<'a, str> {
    let bytes = json.as_bytes();
    let escape_at = |i: usize| -> Option<u16> {
        let hex = bytes.get(i..i + 6)?;
        if hex[0] != b'\\' || hex[1] != b'u' || !hex[2..].iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u16::from_str_radix(std::str::from_utf8(&hex[2..]).ok()?, 16).ok()
    };
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        let Some(unit) = escape_at(i) else {
            i += 2;
            continue;
        };
        if (0xD800..0xDC00).contains(&unit) && escape_at(i + 6).is_some_and(|low| (0xDC00..0xE000).contains(&low)) {
            i += 12;
            continue;
        }
        if (0xD800..0xE000).contains(&unit) {
            let kept = out.get_or_insert_with(|| String::with_capacity(json.len()));
            kept.push_str(&json[copied..i]);
            kept.push_str(with);
            copied = i + 6;
        }
        i += 6;
    }
    match out {
        None => Cow::Borrowed(json),
        Some(mut kept) => {
            kept.push_str(&json[copied..]);
            Cow::Owned(kept)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_escapes() {
        let pair = "{\"a\": \"x\\ud83d\\ude00y\"}";
        assert!(matches!(drop_lone_surrogates(pair), Cow::Borrowed(_)));
        assert_eq!(drop_lone_surrogates("{\"a\": \"x\\ud83dy\\ude00\"}"), "{\"a\": \"xy\"}");
        assert_eq!(drop_lone_surrogates("[\"\\\\ud800\", \"\\ud800\\ud800\\udc00\"]"), "[\"\\\\ud800\", \"\\ud800\\udc00\"]");
        assert!(matches!(drop_lone_surrogates("\"plain \u{e9}\""), Cow::Borrowed(_)));
    }

    #[test]
    fn lines_split_as_python() {
        assert_eq!(splitlines("a\r\nb\rc\n\nd\u{85}e\u{2028}"), ["a", "b", "c", "", "d", "e"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n"), [""]);
    }

    #[test]
    fn casefold_is_full_folding() {
        assert_eq!(casefold("Straße ﬁ İ"), "strasse fi i\u{307}");
    }
}

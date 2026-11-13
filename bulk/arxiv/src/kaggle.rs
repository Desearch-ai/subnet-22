//! The Kaggle metadata snapshot: one JSON paper a line, with RFC 2822 version dates.

use std::borrow::Cow;
use std::io::BufRead;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::Page;
use serde_json::{Map, Value};

use crate::{page_of, squash, Paper, Record};

/// Stands in for a lone surrogate escape, which Rust strings cannot hold, until `without_surrogates` drops it.
const SURROGATE: char = '\u{ffff}';
const MONTHS: [&str; 24] = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const DAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
const ZONES: [(&str, i64); 14] = [
    ("UT", 0),
    ("UTC", 0),
    ("GMT", 0),
    ("Z", 0),
    ("AST", -400),
    ("ADT", -300),
    ("EST", -500),
    ("EDT", -400),
    ("CST", -600),
    ("CDT", -500),
    ("MST", -700),
    ("MDT", -600),
    ("PST", -800),
    ("PDT", -700),
];

/// Python's `int()` of a token without whitespace: a sign, then digits with single underscores between them.
fn py_int(s: &str) -> Option<i64> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let mut value: i64 = 0;
    for c in digits.chars().filter(|c| *c != '_') {
        value = value.checked_mul(10)?.checked_add(c.to_digit(10)? as i64)?;
    }
    Some(if negative { -value } else { value })
}

fn last_is(s: &str, c: char) -> bool {
    s.ends_with(c)
}

fn without_last(s: &str) -> &str {
    s.char_indices().next_back().map_or(s, |(i, _)| &s[..i])
}

/// `email.utils._parsedate_tz`: year, month, day, hour, minute, second and the zone offset in seconds.
fn parsedate_tz(text: &str) -> Option<([i64; 6], Option<i64>)> {
    let mut data: Vec<String> = desearch::text::words(text).map(String::from).collect();
    if data.is_empty() {
        return None;
    }
    if data[0].ends_with(',') || DAYS.contains(&data[0].to_lowercase().as_str()) {
        data.remove(0);
    } else if let Some(i) = data[0].rfind(',') {
        data[0] = data[0][i + 1..].to_string();
    }
    if data.len() == 3 {
        let stuff: Vec<String> = data[0].split('-').map(String::from).collect();
        if stuff.len() == 3 {
            data = stuff.into_iter().chain(data.drain(1..)).collect();
        }
    }
    if data.len() == 4 {
        let s = data[3].clone();
        let i = s.find('+').or_else(|| s.find('-'));
        match i {
            Some(i) if i > 0 => {
                data.truncate(3);
                data.push(s[..i].to_string());
                data.push(s[i..].to_string());
            }
            _ => data.push(String::new()),
        }
    }
    if data.len() < 5 {
        return None;
    }
    let [mut dd, mut mm, mut yy, mut tm, mut tz]: [String; 5] = data.into_iter().take(5).collect::<Vec<_>>().try_into().ok()?;
    if dd.is_empty() || mm.is_empty() || yy.is_empty() {
        return None;
    }
    mm = mm.to_lowercase();
    if !MONTHS.contains(&mm.as_str()) {
        (dd, mm) = (mm, dd.to_lowercase());
        if !MONTHS.contains(&mm.as_str()) {
            return None;
        }
    }
    let mut month = MONTHS.iter().position(|m| *m == mm)? as i64 + 1;
    if month > 12 {
        month -= 12;
    }
    if last_is(&dd, ',') {
        dd = without_last(&dd).to_string();
    }
    if yy.find(':').is_some_and(|i| i > 0) {
        std::mem::swap(&mut yy, &mut tm);
    }
    if last_is(&yy, ',') {
        yy = without_last(&yy).to_string();
        if yy.is_empty() {
            return None;
        }
    }
    if !yy.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        std::mem::swap(&mut yy, &mut tz);
    }
    if last_is(&tm, ',') {
        tm = without_last(&tm).to_string();
    }
    let mut clock: Vec<&str> = tm.split(':').collect();
    match clock.len() {
        2 => clock.push("0"),
        3 => {}
        1 if clock[0].contains('.') => {
            clock = clock[0].split('.').collect();
            match clock.len() {
                2 => clock.push("0"),
                3 => {}
                _ => return None,
            }
        }
        _ => return None,
    }
    let mut year = py_int(&yy)?;
    let day = py_int(&dd)?;
    let [hour, minute, second] = [py_int(clock[0])?, py_int(clock[1])?, py_int(clock[2])?];
    if year < 100 {
        year += if year > 68 { 1900 } else { 2000 };
    }
    let tz = tz.to_uppercase();
    let mut offset = match ZONES.iter().find(|(name, _)| *name == tz) {
        Some((_, value)) => Some(*value),
        None => py_int(&tz),
    };
    if offset == Some(0) && tz.starts_with('-') {
        offset = None;
    }
    let offset = offset.map(|o| o.signum().saturating_mul((o.saturating_abs() / 100).saturating_mul(3600).saturating_add(o.saturating_abs() % 100 * 60)));
    Some(([year, month, day, hour, minute, second], offset))
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The day of an RFC 2822 date in its own zone, or "" when Python's datetime rejects it.
pub fn kaggle_date(text: &str) -> String {
    let Some(([year, month, day, hour, minute, second], offset)) = parsedate_tz(text) else {
        return String::new();
    };
    let valid = (1..=9999).contains(&year)
        && (1..=days_in_month(year, month)).contains(&day)
        && (0..24).contains(&hour)
        && (0..60).contains(&minute)
        && (0..60).contains(&second)
        && offset.is_none_or(|o| o.abs() < 86_400);
    if valid {
        format!("{year:04}-{month:02}-{day:02}")
    } else {
        String::new()
    }
}

/// JSON text with each lone surrogate escape as `￿`, and pairs left for the parser to join.
pub(crate) fn mark_lone_surrogates(json: &str) -> Cow<'_, str> {
    if !json.contains("\\u") {
        return Cow::Borrowed(json);
    }
    let bytes = json.as_bytes();
    let unit = |at: usize| -> Option<u32> {
        if bytes.get(at) == Some(&b'\\') && bytes.get(at + 1) == Some(&b'u') {
            u32::from_str_radix(json.get(at + 2..at + 6)?, 16).ok()
        } else {
            None
        }
    };
    let mut out = String::with_capacity(json.len());
    let mut copied = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        match unit(i) {
            Some(0xd800..=0xdbff) if unit(i + 6).is_some_and(|low| (0xdc00..=0xdfff).contains(&low)) => i += 12,
            Some(0xd800..=0xdfff) => {
                out.push_str(&json[copied..i]);
                out.push_str("\\uffff");
                i += 6;
                copied = i;
            }
            Some(_) => i += 6,
            None => i += 2,
        }
    }
    if copied == 0 {
        return Cow::Borrowed(json);
    }
    out.push_str(&json[copied..]);
    Cow::Owned(out)
}

fn unmark(s: &mut String) {
    if s.contains(SURROGATE) {
        s.retain(|c| c != SURROGATE);
    }
}

/// `desearch.extraction.without_surrogates` for the marks `mark_lone_surrogates` left.
fn without_surrogates(page: &mut Page) {
    for s in [
        &mut page.page_type,
        &mut page.text_mode,
        &mut page.title,
        &mut page.description,
        &mut page.lang,
        &mut page.canonical,
        &mut page.published,
        &mut page.modified,
        &mut page.author,
        &mut page.site_name,
        &mut page.image,
        &mut page.section,
        &mut page.entity,
        &mut page.ld,
        &mut page.text,
    ] {
        unmark(s);
    }
    for list in [&mut page.tags, &mut page.breadcrumbs, &mut page.json_ld_types, &mut page.headings, &mut page.robots] {
        list.iter_mut().for_each(unmark);
    }
}

fn falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// `(value or "")` for a field `squash` or `kaggle_date` reads: Python fails on anything else that is not a string.
fn text_field<'a>(item: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match item.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(value) if !falsy(value) => bail!("{key} is not a string: {value}"),
        _ => Ok(""),
    }
}

fn author_of(parsed: &Value) -> Result<String> {
    let Value::Array(parts) = parsed else { bail!("an authors_parsed entry is not a list: {parsed}") };
    let key = parts.first().ok_or_else(|| anyhow!("an empty authors_parsed entry"))?;
    let names = [parts.get(1), Some(key), parts.get(2)];
    let mut kept = Vec::new();
    for name in names.into_iter().flatten() {
        match name {
            Value::String(s) if !s.is_empty() => kept.push(s.as_str()),
            value if falsy(value) => {}
            value => bail!("an author name is not a string: {value}"),
        }
    }
    Ok(squash(&kept.join(" ")))
}

fn version_date(version: &Value) -> Result<String> {
    let created = version.get("created").ok_or_else(|| anyhow!("a version without created: {version}"))?;
    match created {
        Value::String(s) => Ok(kaggle_date(s)),
        value if falsy(value) => Ok(String::new()),
        value => bail!("a version's created is not a string: {value}"),
    }
}

/// One line of the snapshot as `kaggle_records` makes it.
pub fn kaggle_record(line: &str) -> Result<Record> {
    let item: Value = serde_json::from_str(&mark_lone_surrogates(line))?;
    let Value::Object(item) = item else { bail!("a line that is not a JSON object") };
    let id = match item.get("id") {
        Some(Value::String(id)) if !id.contains(SURROGATE) => id.clone(),
        Some(Value::String(_)) => bail!("an id with a lone surrogate"),
        Some(value) => bail!("an id that is not a string: {value}"),
        None => bail!("a paper without an id"),
    };
    let versions = match item.get("versions") {
        Some(Value::Array(versions)) => versions.as_slice(),
        Some(value) if !falsy(value) => bail!("versions is not a list: {value}"),
        _ => &[],
    };
    let authors = match item.get("authors_parsed") {
        Some(Value::Array(parsed)) => parsed.iter().map(author_of).collect::<Result<_>>()?,
        Some(value) if !falsy(value) => bail!("authors_parsed is not a list: {value}"),
        _ => Vec::new(),
    };
    let squashed = |key: &str| text_field(&item, key).map(squash);
    let paper = Paper {
        arxiv_id: id,
        title: squashed("title")?,
        authors,
        abstract_text: squashed("abstract")?,
        categories: squashed("categories")?,
        created: versions.first().map(version_date).transpose()?.unwrap_or_default(),
        updated: versions.last().map(version_date).transpose()?.unwrap_or_default(),
        comments: squashed("comments")?,
        journal: squashed("journal-ref")?,
        doi: squashed("doi")?,
        licence: squashed("license")?,
    };
    let (url, mut page) = page_of(&paper);
    without_surrogates(&mut page);
    Ok(Record::Page(url, page))
}

/// The snapshot's lines as Python's universal newlines split them.
pub struct Lines<R> {
    reader: R,
    pending: std::collections::VecDeque<String>,
}

impl<R: BufRead> Iterator for Lines<R> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(line) = self.pending.pop_front() {
            return Some(Ok(line));
        }
        let mut raw = Vec::new();
        match self.reader.read_until(b'\n', &mut raw) {
            Ok(0) => return None,
            Ok(_) => {}
            Err(error) => return Some(Err(error.into())),
        }
        if raw.last() == Some(&b'\n') {
            raw.pop();
        }
        if raw.last() == Some(&b'\r') {
            raw.pop();
        }
        let text = match String::from_utf8(raw) {
            Ok(text) => text,
            Err(error) => return Some(Err(error.into())),
        };
        self.pending.extend(text.split('\r').map(String::from));
        self.pending.pop_front().map(Ok)
    }
}

pub fn lines(path: &Path) -> Result<Lines<std::io::BufReader<std::fs::File>>> {
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Lines { reader: std::io::BufReader::with_capacity(1 << 20, file), pending: Default::default() })
}

/// Each line's abstract page, in order, ending at the first error.
pub fn kaggle_records(path: &Path) -> Result<impl Iterator<Item = Result<Record>>> {
    Ok(lines(path)?.map(|line| line.and_then(|line| kaggle_record(&line))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_lone_surrogate_escapes_are_marked() {
        let u = |hex: &str| format!("\\u{hex}");
        let marked = |s: String| mark_lone_surrogates(&s).into_owned();
        assert_eq!(marked(format!("a {} b", u("d800"))), format!("a {} b", u("ffff")));
        assert_eq!(marked(format!("{}{}", u("d835"), u("dc9c"))), format!("{}{}", u("d835"), u("dc9c")));
        assert_eq!(marked(format!("{}{}{}", u("d835"), u("0041"), u("dc9c"))), format!("{}{}{}", u("ffff"), u("0041"), u("ffff")));
        assert_eq!(marked(format!("\\{} é", u("d800"))), format!("\\{} é", u("d800")), "an escaped backslash is not an escape");
        assert_eq!(marked(u("D83D")), u("ffff"));
    }

    #[test]
    fn lines_split_as_universal_newlines_do() {
        let split = |text: &str| Lines { reader: text.as_bytes(), pending: Default::default() }.map(Result::unwrap).collect::<Vec<_>>();
        assert_eq!(split("a\nb\r\nc\rd\r\r\ne"), ["a", "b", "c", "d", "", "e"]);
        assert_eq!(split("a\r"), ["a"]);
        assert_eq!(split("a\n\n"), ["a", ""]);
        assert!(split("").is_empty());
    }
}

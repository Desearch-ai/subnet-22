//! The v1 set file's rule and the planner's page checks: which pages each set site admits, and in which language.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use regex::Regex;
use serde_json::Value;

use crate::pystr;
use crate::urls;

pub const LANDING_DEPTH: i64 = 1;
pub const LANGS: &[&str] = &["en"];
pub const MIN_CHARS: usize = 300;
pub const EXTRACTION: &str = "v2";
const RULES: &[&str] = &["whole", "news", "bulk", "landing", "out"];

static ENGLISH_LOCALE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)^en(?:[-_][a-z]{2})?$").unwrap());
static LOCALE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)^[a-z]{2}[-_][a-z]{2}$").unwrap());

/// Lower case, no `www.`, no mobile `m.` label: `en.m.wikipedia.org` is `en.wikipedia.org`.
pub fn normal_host(host: &str) -> String {
    let lower = host.to_lowercase();
    let h = lower.trim_matches('.');
    let h = h.strip_prefix("www.").unwrap_or(h);
    let mut labels: Vec<&str> = h.split('.').collect();
    if labels[0] == "m" && labels.len() > 2 {
        labels.remove(0);
    } else if labels.len() > 3 && labels[1] == "m" {
        labels.remove(1);
    }
    labels.join(".")
}

/// Python's `date.fromisoformat` for the date forms it reads.
pub fn iso_date(s: &str) -> Option<NaiveDate> {
    let b = s.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = b.get(r)?;
        part.iter().all(u8::is_ascii_digit).then(|| part.iter().fold(0, |n, d| n * 10 + u32::from(d - b'0')))
    };
    let year = num(0..4)? as i32;
    if year < 1 {
        return None;
    }
    let dash = b.get(4) == Some(&b'-');
    let rest = if dash { 5 } else { 4 };
    if b.get(rest) == Some(&b'W') {
        let week = num(rest + 1..rest + 3)?;
        let day = match b.len() - (rest + 3) {
            0 => 1,
            n if dash && n == 2 && b[rest + 3] == b'-' => num(rest + 4..rest + 5)?,
            1 if !dash => num(rest + 3..rest + 4)?,
            _ => return None,
        };
        if !(1..=7).contains(&day) {
            return None;
        }
        return NaiveDate::from_isoywd_opt(year, week, chrono::Weekday::try_from(day as u8 - 1).ok()?);
    }
    let (month, day) = match (dash, b.len()) {
        (true, 10) if b[7] == b'-' => (num(5..7)?, num(8..10)?),
        (false, 8) => (num(4..6)?, num(6..8)?),
        _ => return None,
    };
    NaiveDate::from_ymd_opt(year, month, day)
}

fn digit(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_digit())
}

/// The first `DATE_IN_PATH` match as Python's backtracking finds it: (year, month, day).
fn date_in_path(path: &str) -> Option<(u32, u32, Option<u32>)> {
    let chars: Vec<char> = path.chars().collect();
    let at = |i: usize| chars.get(i).copied();
    let ahead = |i: usize| match at(i) {
        None => true,
        Some('\n') => i + 1 == chars.len(),
        Some(c) => matches!(c, '/' | '.' | '_' | '-'),
    };
    let value = |from: usize, to: usize| chars[from..to].iter().collect::<String>().parse::<u32>().ok();
    let one_to_31 = |i: usize| -> Vec<usize> {
        let mut ends = Vec::new();
        if at(i) == Some('0') && matches!(at(i + 1), Some('1'..='9')) {
            ends.push(i + 2);
        }
        if matches!(at(i), Some('1'..='9')) {
            ends.push(i + 1);
        }
        ends
    };
    for start in 0..chars.len() {
        if at(start) != Some('/') {
            continue;
        }
        let y = start + 1;
        if !(matches!((at(y), at(y + 1)), (Some('1'), Some('9')) | (Some('2'), Some('0'))) && digit(at(y + 2)) && digit(at(y + 3))) {
            continue;
        }
        if !matches!(at(y + 4), Some('/' | '-')) {
            continue;
        }
        let m = y + 5;
        let mut months = one_to_31(m);
        if at(m) == Some('1') && matches!(at(m + 1), Some('0'..='2')) {
            months.push(m + 2);
        }
        for m_end in months {
            if matches!(at(m_end), Some('/' | '-')) {
                let d = m_end + 1;
                let mut days = one_to_31(d);
                if matches!(at(d), Some('1' | '2')) && digit(at(d + 1)) {
                    days.push(d + 2);
                }
                if at(d) == Some('3') && matches!(at(d + 1), Some('0' | '1')) {
                    days.push(d + 2);
                }
                for d_end in days {
                    if ahead(d_end) {
                        return Some((value(y, y + 4)?, value(m, m_end)?, value(d, d_end)));
                    }
                }
            }
            if ahead(m_end) {
                return Some((value(y, y + 4)?, value(m, m_end)?, None));
            }
        }
    }
    None
}

/// The page's `published` date, else a date in its URL path.
pub fn page_date(published: Option<&str>, url: &str) -> Option<NaiveDate> {
    if let Some(p) = published.filter(|p| !p.is_empty()) {
        if let Some(d) = iso_date(pystr::cut(p, 10)) {
            return Some(d);
        }
    }
    let path = urls::split(url).ok()?.path;
    let (y, m, d) = date_in_path(&path)?;
    NaiveDate::from_ymd_opt(y as i32, m, d.unwrap_or(1))
}

/// A homepage or a page at most `depth` levels down with no query, in no other locale.
pub fn landing_page(path: &str, query: &str, depth: i64) -> bool {
    let mut levels: Vec<&str> = path.split('/').filter(|l| !l.is_empty()).collect();
    if levels.first().is_some_and(|l| ENGLISH_LOCALE.is_match(l)) {
        levels.remove(0);
    } else if levels.first().is_some_and(|l| LOCALE.is_match(l)) {
        return false;
    }
    query.is_empty() && levels.len() as i64 <= depth
}

pub struct HubPages {
    types: HashSet<String>,
    words: HashSet<String>,
}

fn hub_page(hub: Option<&HubPages>, path: &str, page_type: Option<&str>) -> bool {
    let Some(hub) = hub else { return false };
    if page_type.is_some_and(|t| hub.types.contains(t)) {
        return true;
    }
    path.split('/').filter(|l| !l.is_empty()).any(|level| {
        let level = level.to_ascii_lowercase();
        hub.words.contains(&level) || level.split(['-', '_', '.']).any(|w| hub.words.contains(w))
    })
}

struct Site {
    rule: String,
    paths: Vec<String>,
    hubs: bool,
    own_host: bool,
    depth: Option<i64>,
}

impl Site {
    fn own_host_only(&self) -> bool {
        self.rule == "landing" || (self.rule == "whole" && self.own_host)
    }
}

pub struct SetRule {
    sites: HashMap<String, Site>,
    since: HashMap<String, NaiveDate>,
    undated: String,
    hub: Option<HubPages>,
    pages: HashSet<String>,
}

fn strings(v: Option<&Value>) -> Option<Vec<String>> {
    match v {
        None | Some(Value::Null) => Some(Vec::new()),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string)).collect(),
        _ => None,
    }
}

fn flag(v: Option<&Value>) -> Option<bool> {
    match v {
        None | Some(Value::Null) => Some(false),
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

impl SetRule {
    /// A set file, gzipped or not.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let data = if raw.starts_with(&[0x1f, 0x8b]) {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&raw[..]).read_to_end(&mut out)?;
            out
        } else {
            raw
        };
        Self::parse(&data)
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        let d: Value = serde_json::from_slice(data)?;
        let Some(list) = d.get("sites").and_then(Value::as_array) else { bail!("a set file has a list of sites") };
        let undated = match d.get("undated") {
            None | Some(Value::Null) => "out".to_string(),
            Some(Value::String(s)) if s.is_empty() => "out".to_string(),
            Some(Value::String(s)) => s.clone(),
            _ => bail!("a set file has a string undated"),
        };
        let hub = match d.get("hub_pages") {
            None | Some(Value::Null) => None,
            Some(h) => {
                let (Some(types), Some(words)) = (h.get("types"), h.get("words")) else { bail!("hub_pages not understood") };
                match (types.as_array().and(strings(Some(types))), words.as_array().and(strings(Some(words)))) {
                    (Some(t), Some(w)) => Some(HubPages { types: t.into_iter().collect(), words: w.into_iter().collect() }),
                    _ => bail!("hub_pages not understood"),
                }
            }
        };
        let mut sites = HashMap::new();
        let mut since = HashMap::new();
        for s in list {
            let bad = || anyhow::anyhow!("set entry not understood: {}", pystr::cut(&s.to_string(), 200));
            let host = s.get("host").and_then(Value::as_str).ok_or_else(bad)?;
            let rule = s.get("rule").and_then(Value::as_str).filter(|r| RULES.contains(r)).ok_or_else(bad)?;
            let since_at = match s.get("since") {
                None | Some(Value::Null | Value::Bool(false)) => None,
                Some(Value::String(v)) => Some(v.as_str()).filter(|v| !v.is_empty()),
                _ => return Err(bad()),
            };
            let paths = strings(s.get("paths")).ok_or_else(bad)?;
            let hubs = flag(s.get("hubs")).ok_or_else(bad)?;
            let own_host = flag(s.get("own_host")).ok_or_else(bad)?;
            let depth = match s.get("depth") {
                None | Some(Value::Null) => None,
                Some(Value::Number(n)) if n.as_i64().is_some_and(|d| d >= 1) => n.as_i64(),
                _ => return Err(bad()),
            };
            let key = normal_host(host);
            since.remove(&key);
            if let Some(at) = since_at {
                let day = iso_date(at).with_context(|| format!("since {at:?} of {host}"))?;
                since.insert(key.clone(), day);
            }
            sites.insert(key, Site { rule: rule.to_string(), paths, hubs, own_host, depth });
        }
        let pages = match d.get("pages") {
            None | Some(Value::Null) => HashSet::new(),
            Some(p) => p.as_array().and(strings(Some(p))).ok_or_else(|| anyhow::anyhow!("pages not understood"))?.into_iter().collect(),
        };
        Ok(Self { sites, since, undated, hub, pages })
    }

    fn entry(&self, host: &str) -> Option<(&str, &Site)> {
        let h = normal_host(host);
        let mut h = h.as_str();
        while !h.is_empty() {
            if let Some((k, s)) = self.sites.get_key_value(h) {
                return Some((k, s));
            }
            h = h.split_once('.')?.1;
        }
        None
    }

    /// Whether the set can admit the URL's page before it is fetched, as the crawler's `scope.may_admit`; date rules wait for the page.
    pub fn may_admit(&self, url: &str) -> bool {
        if self.pages.contains(url) {
            return true;
        }
        let Ok(u) = urls::split(url) else { return false };
        let hostname = u.hostname().unwrap_or_default();
        let Some((h, e)) = self.entry(&hostname) else { return false };
        if e.rule == "out" || e.own_host_only() && normal_host(&hostname) != h {
            return false;
        }
        match e.rule.as_str() {
            "whole" => {
                let path = if u.path.is_empty() { "/" } else { &u.path };
                e.paths.is_empty() || e.paths.iter().any(|p| path.starts_with(p.as_str()))
            }
            "landing" => normal_host(&hostname) == h && landing_page(&u.path, u.query.as_deref().unwrap_or(""), e.depth.unwrap_or(LANDING_DEPTH)),
            rule => rule == "news" || rule == "bulk",
        }
    }

    /// The paths a `whole` entry limits the host to, as start points for a site crawl.
    pub fn seed_paths(&self, host: &str) -> Vec<String> {
        match self.entry(host) {
            Some((_, e)) if e.rule == "whole" => e.paths.clone(),
            _ => Vec::new(),
        }
    }

    /// Whether the set takes this page; `cap` bounds the crawl, not the set.
    pub fn admits(&self, url: &str, published: Option<&str>, page_type: Option<&str>) -> bool {
        if self.pages.contains(url) {
            return true;
        }
        let Ok(u) = urls::split(url) else { return false };
        let hostname = u.hostname().unwrap_or_default();
        let Some((h, e)) = self.entry(&hostname) else { return false };
        let query = u.query.as_deref().unwrap_or("");
        match e.rule.as_str() {
            "news" => match page_date(published, url) {
                None => {
                    if self.undated == "article" && page_type == Some("article") {
                        return true;
                    }
                    e.hubs && hub_page(self.hub.as_ref(), &u.path, page_type)
                }
                Some(d) => self.since.get(h).is_none_or(|s| d >= *s),
            },
            "whole" => {
                if e.own_host_only() && normal_host(&hostname) != h {
                    return false;
                }
                let path = if u.path.is_empty() { "/" } else { &u.path };
                e.paths.is_empty() || e.paths.iter().any(|p| path.starts_with(p.as_str()))
            }
            "landing" => normal_host(&hostname) == h && landing_page(&u.path, query, e.depth.unwrap_or(LANDING_DEPTH)),
            rule => rule == "bulk",
        }
    }
}

const ENGLISH: &str = "about all an and are as at be been but by can for from has have in is it not of on or our \
that the their they this to was we were which will with you your";
const OTHER: &str = "ada af akan alla anche ang att au aux ay bei bij bir bu ce ces che con como da dalla \
das dat de degli del della delle dem den dengan der des det deze di dos du een ein eine el \
elle em en er es esta este et gli har het ich ikke ile inte ini itu jak jest la las le \
les leur los maar mais mga mit muy na naar nach ne nel nella nicht nie noch nous ng och \
oder og ook op oraz ou para pas pela pelo pero por przez que questo qui se seu ses sich \
sie sind som sono sont su sua sur sus tidak um una une und untuk van ve von voor vous \
werden wie wir wird wordt yang ze zijn zu";
static ENGLISH_WORDS: LazyLock<HashSet<&str>> = LazyLock::new(|| ENGLISH.split(' ').collect());
static OTHER_WORDS: LazyLock<HashSet<&str>> = LazyLock::new(|| OTHER.split(' ').collect());

/// Whether a text's start reads as English: enough ASCII words, little else, English words.
pub fn english(text: &str) -> bool {
    let sample = pystr::cut(text, 4000);
    let words: Vec<String> =
        sample.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()).map(str::to_ascii_lowercase).collect();
    let letters: usize = words.iter().map(String::len).sum();
    let non_ascii = sample.chars().filter(|c| !c.is_ascii()).count();
    let english_words = words.iter().filter(|w| ENGLISH_WORDS.contains(w.as_str())).count();
    let other_words = words.iter().filter(|w| OTHER_WORDS.contains(w.as_str())).count();
    words.len() >= 30 && non_ascii * 10 <= letters && english_words * 100 >= 8 * words.len() && english_words > 2 * other_words
}

/// The page's own language, or `en` when it names none and its text reads as English.
pub fn page_lang<'a>(lang: &'a str, text: &str) -> &'a str {
    if pystr::strip(lang).is_empty() && english(text) {
        "en"
    } else {
        lang
    }
}

/// The planner's checks before the set rule: language, extraction, length.
pub fn page_eligible(lang: &str, extraction: &str, text: &str) -> bool {
    let lang = page_lang(lang, text).to_lowercase();
    LANGS.iter().any(|w| lang == *w || lang.starts_with(&format!("{w}-")) || lang.starts_with(&format!("{w}_")))
        && extraction == EXTRACTION
        && pystr::len(text) >= MIN_CHARS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_in_paths_backtrack_like_python() {
        assert_eq!(date_in_path("/2020/10/15/x"), Some((2020, 10, Some(15))));
        assert_eq!(date_in_path("/2020/1/5"), Some((2020, 1, Some(5))));
        assert_eq!(date_in_path("/2020/10/"), Some((2020, 10, None)));
        assert_eq!(date_in_path("/2020/13/"), None);
        assert_eq!(date_in_path("/a/1999-02-30.html"), Some((1999, 2, Some(30))));
        assert_eq!(date_in_path("/2020/105"), None);
        assert_eq!(date_in_path("/x/2020/10/150/2021/03"), Some((2020, 10, None)));
    }

    #[test]
    fn iso_dates_like_python() {
        assert_eq!(iso_date("2026-11-10"), NaiveDate::from_ymd_opt(2026, 11, 10));
        assert_eq!(iso_date("20261110"), NaiveDate::from_ymd_opt(2026, 11, 10));
        assert_eq!(iso_date("2026-W01-1"), NaiveDate::from_ymd_opt(2025, 12, 29));
        assert_eq!(iso_date("2026W011"), NaiveDate::from_ymd_opt(2025, 12, 29));
        assert_eq!(iso_date("2026-W01"), NaiveDate::from_ymd_opt(2025, 12, 29));
        assert_eq!(iso_date("2026-11-1T"), None);
        assert_eq!(iso_date("2026-02-30"), None);
        assert_eq!(iso_date("2026-11"), None);
    }

    #[test]
    fn hosts_and_landing_pages() {
        assert_eq!(normal_host("WWW.En.M.Wikipedia.org."), "en.wikipedia.org");
        assert_eq!(normal_host("m.example.com"), "example.com");
        assert!(landing_page("/en-us/about", "", 1));
        assert!(!landing_page("/fr-fr/", "", 1));
        assert!(!landing_page("/a/b", "", 1));
        assert!(!landing_page("/", "q=1", 1));
    }

    #[test]
    fn set_rules_admit_as_python_does() {
        let set = br#"{"sites": [
            {"host": "news.example", "rule": "news", "since": "2026-01-01", "hubs": true},
            {"host": "whole.example", "rule": "whole", "paths": ["/docs/"]},
            {"host": "own.example", "rule": "whole", "own_host": true},
            {"host": "land.example", "rule": "landing", "depth": 2},
            {"host": "bulk.example", "rule": "bulk"},
            {"host": "out.example", "rule": "out"}],
            "undated": "article", "hub_pages": {"types": ["listing"], "words": ["data", "stats"]}}"#;
        let rule = SetRule::parse(set).unwrap();
        assert!(rule.admits("https://news.example/2026/03/x", None, None));
        assert!(!rule.admits("https://news.example/2025/03/x", None, None));
        assert!(!rule.admits("https://news.example/x", Some("2025-12-31"), None));
        assert!(rule.admits("https://news.example/x", None, Some("article")));
        assert!(rule.admits("https://news.example/key-stats.html", None, None));
        assert!(!rule.admits("https://news.example/x", None, None));
        assert!(rule.admits("https://sub.whole.example/docs/a", None, None));
        assert!(!rule.admits("https://whole.example/blog/a", None, None));
        assert!(!rule.admits("https://sub.own.example/a", None, None));
        assert!(rule.admits("https://www.own.example/a", None, None));
        assert!(rule.admits("https://land.example/a/b", None, None));
        assert!(!rule.admits("https://land.example/a/b/c", None, None));
        assert!(!rule.admits("https://x.land.example/", None, None));
        assert!(rule.admits("https://a.bulk.example/anything?x", None, None));
        assert!(!rule.admits("https://out.example/", None, None));
        assert!(!rule.admits("https://other.example/", None, None));
    }

    #[test]
    fn a_listed_page_is_in_whatever_its_host_rule_says() {
        let rule = SetRule::parse(br#"{"sites": [{"host": "forbes.com", "rule": "news"}], "pages": ["https://www.forbes.com/billionaires/"]}"#).unwrap();
        assert!(rule.admits("https://www.forbes.com/billionaires/", None, Some("listing")));
        assert!(rule.may_admit("https://www.forbes.com/billionaires/"));
        assert!(!rule.admits("https://www.forbes.com/lists/", None, Some("listing")));
        assert!(SetRule::parse(br#"{"sites": [], "pages": "x"}"#).is_err());
    }

    #[test]
    fn english_text_counts_as_english() {
        let text = "This is the page that we were reading, and it has all of the words you want to see in it. ".repeat(3);
        assert!(english(&text));
        assert!(page_eligible("", "v2", &text.repeat(3)));
        assert!(!page_eligible("de", "v2", &text.repeat(3)));
        assert!(page_eligible("en-GB", "v2", &text.repeat(3)));
        assert!(!page_eligible("en", "v1", &text.repeat(3)));
    }
}

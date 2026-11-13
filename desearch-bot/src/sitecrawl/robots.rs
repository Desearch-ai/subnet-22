//! robots.txt per host: Protego's parse for our token, and why a host is out.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use regex::Regex;

use super::fetch::Http;
use super::Crash;
use crate::pystr;
use crate::urls;

pub const TOKEN: &str = "DesearchBot";
pub const RETRY_PAUSE: Duration = Duration::from_secs(5);
pub const MAX_BYTES: usize = 512 << 10;
const REDIRECTS: [u16; 5] = [301, 302, 303, 307, 308];

const DISALLOW: [&str; 6] = ["disallow", "dissallow", "dissalow", "disalow", "diasllow", "disallaw"];
const USER_AGENT: [&str; 3] = ["user-agent", "useragent", "user agent"];
const SITEMAP: [&str; 3] = ["sitemap", "sitemaps", "site-map"];
const CRAWL_DELAY: [&str; 2] = ["crawl-delay", "crawl delay"];
const OTHER: [&str; 6] = ["allow", "request-rate", "request rate", "visit-time", "visit time", "host"];
const MAX_DIRECTIVE_WORDS: usize = 2;

static QUOTED_URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*)?(/[A-Za-z0-9_.~/=-]*)$").unwrap());
static QUOTED_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z0-9_.~/=*-]+\$?$").unwrap());
static STARS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*+").unwrap());

fn is_directive(field: &str) -> bool {
    DISALLOW.contains(&field) || USER_AGENT.contains(&field) || SITEMAP.contains(&field) || CRAWL_DELAY.contains(&field) || OTHER.contains(&field)
}

fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

/// Python's `float(text)`: surrounding whitespace, `_` between digits, inf and nan.
pub fn py_float(text: &str) -> Option<f64> {
    let value = pystr::strip(text);
    let bytes = value.as_bytes();
    let joined =
        bytes.iter().enumerate().all(|(i, &b)| b != b'_' || (i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)));
    if !joined || value.is_empty() || !value.is_ascii() {
        return None;
    }
    let value = value.replace('_', "");
    let lower = value.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    if lower.len() - unsigned.len() > 1 {
        return None;
    }
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        let v = if unsigned == "nan" { f64::NAN } else { f64::INFINITY };
        return Some(if lower.starts_with('-') { -v } else { v });
    }
    if !unsigned.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-')) {
        return None;
    }
    value.parse::<f64>().ok()
}

/// Protego's `_unquote`: %XX decoded unless it names one of `ignore`, bad bytes replaced.
fn unquote(url: &str, ignore: &str) -> String {
    if !url.contains('%') {
        return url.to_string();
    }
    let ignored: Vec<String> = ignore.chars().map(|c| format!("{:02X}", c as u32)).collect();
    let mut parts = url.split('%');
    let mut out: Vec<u8> = parts.next().unwrap_or("").as_bytes().to_vec();
    for part in parts {
        let b = part.as_bytes();
        if b.len() >= 2 && b[0].is_ascii_hexdigit() && b[1].is_ascii_hexdigit() {
            let code = part[..2].to_ascii_uppercase();
            let leftover = &part[2..];
            if !ignored.contains(&code) {
                out.push(u8::from_str_radix(&code, 16).unwrap());
                out.extend_from_slice(leftover.as_bytes());
                continue;
            }
            out.push(b'%');
            out.extend_from_slice(code.as_bytes());
            out.extend_from_slice(leftover.as_bytes());
            continue;
        }
        out.push(b'%');
        out.extend_from_slice(part.as_bytes());
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Python's `urllib.parse.quote(text, safe)`.
pub fn quote(text: &str, safe: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) || (b < 0x80 && safe.as_bytes().contains(&b)) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn requote(value: &str, kept: &str, safe: &str) -> String {
    quote(&unquote(value, kept), safe)
}

fn quote_path(url: &str) -> String {
    if let Some(found) = QUOTED_URL.captures(url).filter(|c| !c[1].starts_with("//")) {
        return found[1].to_string();
    }
    let url = url.split('#').next().unwrap_or("");
    let parts = urls::split(url).unwrap_or_default();
    let mut path = parts.path.clone();
    if url.contains('?') {
        path.push('?');
        path.push_str(parts.query.as_deref().unwrap_or(""));
    }
    let path = requote(&path, "/%", "/%=");
    if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    }
}

fn quote_pattern(pattern: &str) -> String {
    if QUOTED_PATTERN.is_match(pattern) {
        return pattern.to_string();
    }
    let mut pattern = pattern.to_string();
    if pattern.starts_with("https://") || pattern.starts_with("http://") {
        pattern = format!("/{pattern}");
    }
    let anchor = if pattern.ends_with('$') {
        pattern.pop();
        "$"
    } else {
        ""
    };
    requote(&pattern, "/%*", "/%*=") + anchor
}

#[derive(Clone, Debug)]
struct Pattern {
    pattern: String,
    priority: usize,
    asterisk: bool,
    dollar: bool,
    parts: Vec<String>,
    parts_dollar: bool,
}

impl Pattern {
    fn new(pattern: String) -> Self {
        let asterisk = pattern.contains('*');
        let dollar = pattern.ends_with('$');
        let (parts, parts_dollar) = if asterisk {
            let body = pattern.strip_suffix('$').unwrap_or(&pattern);
            (STARS.replace_all(body, "*").split('*').map(str::to_string).collect(), dollar)
        } else {
            (Vec::new(), false)
        };
        Pattern { priority: pattern.chars().count(), pattern, asterisk, dollar, parts, parts_dollar }
    }

    fn matches(&self, url: &str) -> bool {
        if !self.asterisk {
            return if self.dollar { url == &self.pattern[..self.pattern.len() - 1] } else { url.starts_with(&self.pattern) };
        }
        if !url.starts_with(self.parts[0].as_str()) {
            return false;
        }
        let mut pos = self.parts[0].len();
        for part in &self.parts[1..self.parts.len() - 1] {
            match url[pos..].find(part.as_str()) {
                Some(i) => pos += i + part.len(),
                None => return false,
            }
        }
        let last = self.parts.last().unwrap();
        if self.parts_dollar {
            return url.len() >= pos + last.len() && url.ends_with(last.as_str());
        }
        last.is_empty() || url[pos..].contains(last.as_str())
    }
}

#[derive(Clone, Debug, Default)]
struct RuleSet {
    agent: String,
    rules: Vec<(bool, Pattern)>,
    crawl_delay: Option<f64>,
}

impl RuleSet {
    fn applies_to(&self, robot: &str) -> usize {
        let robot = py_lower(pystr::strip(robot));
        if self.agent == "*" {
            return 1;
        }
        let mut from = 0;
        while let Some(i) = robot[from..].find(&self.agent).map(|i| i + from) {
            let before = robot[..i].chars().next_back();
            if before.is_none_or(|c| !(c.is_alphanumeric() || c == '-' || c == '_')) {
                return self.agent.chars().count();
            }
            from = i + robot[i..].chars().next().map_or(1, char::len_utf8);
        }
        0
    }

    fn add(&mut self, allow: bool, pattern: &str, cache: &mut HashMap<String, Option<Pattern>>) {
        if pattern.contains('$') {
            self.add(allow, &pattern.replace('$', "%24"), cache);
        }
        let found = cache
            .entry(pattern.to_string())
            .or_insert_with(|| {
                let quoted = quote_pattern(pattern);
                (!quoted.is_empty()).then(|| Pattern::new(quoted))
            })
            .clone();
        let Some(found) = found else { return };
        let quoted = found.pattern.clone();
        self.rules.push((allow, found));
        if allow {
            if let Some(dir) = quoted.strip_suffix("index.html").filter(|d| d.ends_with('/')) {
                self.rules.push((true, Pattern::new(format!("{dir}$"))));
            }
        }
    }

    fn can_fetch(&self, url: &str) -> bool {
        let url = quote_path(url);
        for (allow, rule) in &self.rules {
            if rule.matches(&url) {
                return *allow;
            }
        }
        true
    }
}

/// Protego's `_extract_directive`.
fn directive(line: &str) -> Option<(String, String)> {
    let mut found = None;
    if let Some((field, raw)) = line.split_once(':') {
        let field = py_lower(pystr::strip(field));
        let value = pystr::strip(raw).to_string();
        if is_directive(&field) || raw.is_empty() || raw.chars().next().is_some_and(pystr::is_space) {
            return Some((field, value));
        }
        found = Some((field, value));
    }
    let words: Vec<&str> = pystr::split_ws(line).collect();
    for count in 1..=MAX_DIRECTIVE_WORDS {
        let candidate = py_lower(&words[..count.min(words.len())].join(" "));
        if is_directive(&candidate) {
            return Some((candidate, words[count.min(words.len())..].join(" ")));
        }
    }
    found
}

/// A parsed robots.txt, as `Protego.parse`.
#[derive(Clone, Debug, Default)]
pub struct Protego {
    sets: Vec<RuleSet>,
    pub sitemaps: Vec<String>,
}

impl Protego {
    pub fn parse(content: &str) -> Self {
        let mut sets: Vec<RuleSet> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut cache = HashMap::new();
        let mut sitemaps = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        let mut previous: Option<String> = None;
        for raw in pystr::splitlines(content) {
            let mut line = raw;
            if let Some(at) = line.find('#') {
                line = pystr::strip(&line[..at]);
            }
            let line = pystr::strip(line);
            if line.is_empty() {
                continue;
            }
            let Some((field, value)) = directive(line) else { continue };
            let agent_field = USER_AGENT.contains(&field.as_str());
            if agent_field && previous.as_deref().is_some_and(|p| !USER_AGENT.contains(&p)) {
                current.clear();
            }
            if !SITEMAP.contains(&field.as_str()) && field != "host" {
                previous = Some(field.clone());
            }
            if value.is_empty() {
                continue;
            }
            if current.is_empty() && !agent_field && !SITEMAP.contains(&field.as_str()) && field != "host" {
                continue;
            }
            if agent_field {
                let agent = py_lower(pystr::strip(&value));
                let bare = (agent != "*" && agent.contains('*')).then(|| agent.replace('*', ""));
                for ua in [Some(agent), bare].into_iter().flatten().filter(|ua| !ua.is_empty()) {
                    match index.get(&ua) {
                        Some(&i) => {
                            if !current.contains(&i) {
                                current.push(i);
                            }
                        }
                        None => {
                            index.insert(ua.clone(), sets.len());
                            current.push(sets.len());
                            sets.push(RuleSet { agent: ua, ..RuleSet::default() });
                        }
                    }
                }
            } else if field == "allow" || DISALLOW.contains(&field.as_str()) {
                for &i in &current {
                    sets[i].add(field == "allow", &value, &mut cache);
                }
            } else if SITEMAP.contains(&field.as_str()) {
                sitemaps.push(value);
            } else if CRAWL_DELAY.contains(&field.as_str()) {
                if let Some(delay) = py_float(&value).filter(|d| d.is_finite() && *d >= 0.0) {
                    for &i in &current {
                        sets[i].crawl_delay = Some(delay);
                    }
                }
            }
        }
        for set in &mut sets {
            set.rules.sort_by_key(|(allow, p)| std::cmp::Reverse((p.priority, *allow)));
        }
        Protego { sets, sitemaps }
    }

    fn matching(&self, agent: &str) -> Option<&RuleSet> {
        let mut best: Option<(usize, &RuleSet)> = None;
        for set in &self.sets {
            let score = set.applies_to(agent);
            if best.is_none_or(|(top, _)| score > top) {
                best = Some((score, set));
            }
        }
        best.filter(|(score, _)| *score > 0).map(|(_, set)| set)
    }

    pub fn can_fetch(&self, url: &str, agent: &str) -> bool {
        if url.contains("/robots.txt") && super::links::urlparse(url, "").is_ok_and(|p| p[2] == "/robots.txt") {
            return true;
        }
        self.matching(agent).is_none_or(|set| set.can_fetch(url))
    }

    pub fn crawl_delay(&self, agent: &str) -> Option<f64> {
        self.matching(agent).and_then(|set| set.crawl_delay)
    }
}

/// What a host's robots.txt lets us do, or why the host is out.
#[derive(Clone, Debug, Default)]
pub struct Rules {
    pub host: String,
    pub refused: Option<String>,
    pub delay: Option<f64>,
    pub sitemaps: Vec<String>,
    parsed: Option<Protego>,
}

impl Rules {
    pub fn allowed(&self, url: &str) -> bool {
        if self.refused.is_some() {
            return false;
        }
        self.parsed.as_ref().is_none_or(|p| p.can_fetch(url, TOKEN))
    }
}

/// The robots.txt answers that refuse us the host for the run.
pub const REFUSING: [u16; 4] = [401, 402, 403, 451];

/// An answer that is asked again after a pause and, still so, keeps us off: no answer, a server error, and a 429.
pub fn ask_again(status: u16) -> bool {
    status == 0 || status >= 500 || status == 429
}

/// A 2xx is the rules; a 3xx or 4xx other than a refusal or a 429 is none; anything else keeps us off.
pub fn parse(host: &str, status: u16, body: &str) -> Rules {
    if (200..300).contains(&status) {
        let p = Protego::parse(body);
        let delay = p.crawl_delay(TOKEN).filter(|d| *d != 0.0);
        return Rules { host: host.to_string(), refused: None, delay, sitemaps: p.sitemaps.clone(), parsed: Some(p) };
    }
    let refused = REFUSING.contains(&status) || status == 429;
    if (300..500).contains(&status) && !refused {
        return Rules { host: host.to_string(), ..Rules::default() };
    }
    let why = if status == 0 { "unreachable".to_string() } else { status.to_string() };
    Rules { host: host.to_string(), refused: Some(format!("robots_{why}")), ..Rules::default() }
}

/// Every host's rules for the run, read once each.
pub struct Robots {
    http: Arc<Http>,
    scheme: String,
    cache: Mutex<HashMap<String, Arc<Rules>>>,
    pub retry_pause: Duration,
}

impl Robots {
    pub fn new(http: Arc<Http>, scheme: &str) -> Self {
        Robots { http, scheme: scheme.to_string(), cache: Mutex::new(HashMap::new()), retry_pause: RETRY_PAUSE }
    }

    pub async fn rules(&self, host: &str) -> Result<Arc<Rules>, Crash> {
        if let Some(found) = self.cache.lock().unwrap().get(host) {
            return Ok(found.clone());
        }
        let mut r = self.read(host, false).await?;
        if ask_again(r.status) {
            tokio::time::sleep(self.retry_pause).await;
            r = self.read(host, true).await?;
        }
        let text = if (200..300).contains(&r.status) { r.text() } else { String::new() };
        let rules = Arc::new(parse(host, r.status, &text));
        self.cache.lock().unwrap().insert(host.to_string(), rules.clone());
        Ok(rules)
    }

    async fn read(&self, host: &str, patient: bool) -> Result<super::fetch::Response, Crash> {
        let mut url = format!("{}://{host}/robots.txt", self.scheme);
        let mut r = self.http.get(&url, MAX_BYTES, patient).await;
        for _ in 0..5 {
            let Some(location) = r.header("location").filter(|_| REDIRECTS.contains(&r.status)) else { break };
            url = urls::join(&url, location).map_err(|_| Crash("ValueError"))?;
            r = self.http.get(&url, MAX_BYTES, patient).await;
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_for_our_token() {
        let r = parse("a.test", 200, "User-agent: *\nDisallow: /private/\nCrawl-delay: 3\nSitemap: https://a.test/sm.xml\n");
        assert!(r.allowed("https://a.test/page") && !r.allowed("https://a.test/private/x"));
        assert_eq!((r.delay, r.sitemaps.clone()), (Some(3.0), vec!["https://a.test/sm.xml".to_string()]));
        let named = parse("a.test", 200, "User-agent: DesearchBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n");
        assert!(!named.allowed("https://a.test/page"));
        let wild = parse("a.test", 200, "User-agent: *\nDisallow: /*?tab=\n");
        assert!(!wild.allowed("https://a.test/x?tab=1") && wild.allowed("https://a.test/x"));
        assert_eq!(parse("a.test", 200, "User-agent: *\nCrawl-delay: 0\n").delay, None, "a zero delay is none");
    }

    #[test]
    fn refusals_and_429s_keep_us_off_and_a_missing_file_allows_all() {
        for status in [202, 204, 301, 400, 404, 405, 410] {
            assert!(parse("a.test", status, "").allowed("https://a.test/anything"), "{status}");
        }
        assert!(!parse("a.test", 203, "User-agent: *\nDisallow: /\n").allowed("https://a.test/x"));
        for (status, why) in [
            (401, "robots_401"),
            (402, "robots_402"),
            (403, "robots_403"),
            (451, "robots_451"),
            (429, "robots_429"),
            (500, "robots_500"),
            (503, "robots_503"),
            (0, "robots_unreachable"),
        ] {
            let r = parse("a.test", status, "");
            assert!(r.refused.as_deref() == Some(why) && !r.allowed("https://a.test/"), "{status}");
        }
        assert!(ask_again(429) && ask_again(503) && ask_again(0) && !ask_again(403) && !ask_again(404));
    }

    #[test]
    fn reads_as_protego_does() {
        let p = Protego::parse("User-agent: desearch\nDisallow: /a\n\nUser-agent: *\nDisallow: /b\n");
        assert!(!p.can_fetch("https://x/a", TOKEN) && p.can_fetch("https://x/b", TOKEN), "a group naming part of our token is ours");
        let p = Protego::parse("useragent: *\ndissallow: /typo\nDisallow: /$\nAllow: /docs/index.html\nDisallow: /docs/\n");
        assert!(!p.can_fetch("https://x/typo", TOKEN) && !p.can_fetch("https://x/", TOKEN) && p.can_fetch("https://x/a", TOKEN));
        assert!(p.can_fetch("https://x/docs/", TOKEN) && p.can_fetch("https://x/docs/index.html", TOKEN) && !p.can_fetch("https://x/docs/a", TOKEN));
        let p = Protego::parse("User-agent: *\nDisallow: /caf%C3%A9\nDisallow: /a b\nDisallow: /*.pdf$\n");
        assert!(!p.can_fetch("https://x/café/menu", TOKEN) && !p.can_fetch("https://x/a%20b", TOKEN) && !p.can_fetch("https://x/f/x.pdf", TOKEN));
        assert!(p.can_fetch("https://x/f/x.pdf?y", TOKEN) && p.can_fetch("https://x/robots.txt", TOKEN));
        let p = Protego::parse("Disallow: /orphan\nUser-agent: *\nAllow: /\nUser-agent: other\nDisallow: /\n");
        assert!(p.can_fetch("https://x/orphan", TOKEN), "rules before any user agent count for none");
        assert!(!Protego::parse("User-agent: *\nUser-agent: other\nDisallow: /\n").can_fetch("https://x/", TOKEN));
        assert_eq!(Protego::parse("Sitemap: /s.xml\nsite-map: https://x/t.xml\n").sitemaps, ["/s.xml", "https://x/t.xml"]);
        assert_eq!(Protego::parse("User-agent: *\nCrawl-delay: 1_5\n").crawl_delay(TOKEN), Some(15.0));
        assert_eq!(Protego::parse("User-agent: *\nCrawl-delay: inf\n").crawl_delay(TOKEN), None);
    }

    #[test]
    fn python_floats() {
        for (text, value) in [("1", Some(1.0)), (" 2.5 ", Some(2.5)), ("1_000", Some(1000.0)), ("1e2", Some(100.0)), ("-3", Some(-3.0)), (".5", Some(0.5))] {
            assert_eq!(py_float(text), value, "{text}");
        }
        for text in ["", "1__0", "_1", "abc", "1,5", "0x10", "--1"] {
            assert_eq!(py_float(text), None, "{text}");
        }
        assert!(py_float("Infinity").unwrap().is_infinite() && py_float("nan").unwrap().is_nan());
    }
}

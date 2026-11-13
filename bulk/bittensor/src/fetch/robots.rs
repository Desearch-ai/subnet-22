//! robots.txt as Python 3.14's `urllib.robotparser.RobotFileParser` reads it: first matching rule, `*` group last.

use anyhow::{bail, Result};
use bulk_core::text::splitlines;
use bulk_core::urlparse::{quote_bytes, split_parts, unquote_bytes, unsplit_parts};
use desearch::text::strip;

#[derive(Clone, Debug, Default)]
struct Rule {
    path: String,
    allowance: bool,
}

#[derive(Clone, Debug, Default)]
struct Entry {
    agents: Vec<String>,
    rules: Vec<Rule>,
    delay: Option<u64>,
}

impl Entry {
    fn applies_to(&self, useragent: &str) -> bool {
        let useragent = useragent.split('/').next().unwrap_or("").to_lowercase();
        self.agents.iter().any(|agent| agent == "*" || useragent.contains(&agent.to_lowercase()))
    }

    fn allowance(&self, filename: &str) -> bool {
        self.rules.iter().find(|rule| rule.path == "*" || filename.starts_with(&rule.path)).is_none_or(|rule| rule.allowance)
    }
}

/// `normalize`: escapes decoded and the bytes quoted again, `/` kept.
fn normalize(path: &str) -> String {
    quote_bytes(&unquote_bytes(path), "/")
}

/// `normalize_path`: the path normalized, and each query name and value apart from `=` and `&`.
pub fn normalize_path(path: &str) -> String {
    let Some((path, query)) = path.split_once('?') else {
        return normalize(path);
    };
    let mut out = normalize(path);
    out.push('?');
    let mut run = String::new();
    for c in query.chars() {
        if c == '=' || c == '&' {
            out.push_str(&normalize(&run));
            run.clear();
            out.push(c);
        } else {
            run.push(c);
        }
    }
    out.push_str(&normalize(&run));
    out
}

/// Digits `str.isdigit` takes and `int` refuses: superscripts, subscripts and circled forms.
fn is_digit_only(c: char) -> bool {
    matches!(c as u32, 0xB2 | 0xB3 | 0xB9 | 0x2070 | 0x2074..=0x2079 | 0x2080..=0x2089 | 0x2460..=0x2468 | 0x2474..=0x247C | 0x2488..=0x2490 | 0x24EA
        | 0x24F5..=0x24FD | 0x24FF | 0x2776..=0x277E | 0x2780..=0x2788 | 0x278A..=0x2792 | 0x1369..=0x1371 | 0x19DA)
}

/// `if value.strip().isdigit(): int(value)`: decimal digits read, the digits `int` refuses failing as it fails.
fn delay_of(text: &str) -> Result<Option<u64>> {
    let text = strip(text);
    if text.is_empty() {
        return Ok(None);
    }
    if let Some(n) = bulk_core::py::decimal_int(text) {
        return Ok(Some(n));
    }
    if text.chars().all(|c| bulk_core::py::decimal_digit(c).is_some() || is_digit_only(c)) {
        bail!("ValueError: invalid literal for int() with base 10: {}", bulk_core::py::repr_str(text));
    }
    Ok(None)
}

#[derive(Clone, Debug, Default)]
pub struct Robots {
    entries: Vec<Entry>,
    default_entry: Option<Entry>,
}

impl Robots {
    fn add(&mut self, entry: Entry) {
        if entry.agents.iter().any(|a| a == "*") {
            if self.default_entry.is_none() {
                self.default_entry = Some(entry);
            }
        } else {
            self.entries.push(entry);
        }
    }

    /// `parse(lines)`.
    pub fn parse(text: &str) -> Result<Robots> {
        let mut robots = Robots::default();
        let mut state = 0;
        let mut entry = Entry::default();
        for raw in splitlines(text) {
            if raw.is_empty() {
                if state == 1 {
                    entry = Entry::default();
                    state = 0;
                } else if state == 2 {
                    robots.add(std::mem::take(&mut entry));
                    state = 0;
                }
            }
            let line = raw.split_once('#').map_or(raw, |(before, _)| before);
            let line = strip(line);
            if line.is_empty() {
                continue;
            }
            let Some((field, value)) = line.split_once(':') else { continue };
            let field = strip(field).to_lowercase();
            let value = strip(value);
            match field.as_str() {
                "user-agent" => {
                    if state == 2 {
                        robots.add(std::mem::take(&mut entry));
                    }
                    entry.agents.push(value.to_string());
                    state = 1;
                }
                "disallow" | "allow" => {
                    if state != 0 {
                        let allowance = field == "allow" || value.is_empty();
                        entry.rules.push(Rule { path: normalize_path(value), allowance });
                        state = 2;
                    }
                }
                "crawl-delay" => {
                    if state != 0 {
                        if let Some(delay) = delay_of(value)? {
                            entry.delay = Some(delay);
                        }
                        state = 2;
                    }
                }
                "request-rate" if state != 0 => state = 2,
                _ => {}
            }
        }
        if state == 2 {
            robots.add(entry);
        }
        Ok(robots)
    }

    fn entry_for(&self, useragent: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.applies_to(useragent)).or(self.default_entry.as_ref())
    }

    /// `can_fetch(useragent, url)`.
    pub fn can_fetch(&self, useragent: &str, url: &str) -> Result<bool> {
        let parts = split_parts(url)?;
        let target = normalize_path(&unsplit_parts(None, None, &parts.path, parts.query.as_deref(), parts.fragment.as_deref()));
        let target = if target.is_empty() { "/".to_string() } else { target };
        Ok(self.entry_for(useragent).is_none_or(|entry| entry.allowance(&target)))
    }

    /// `crawl_delay(useragent)`.
    pub fn crawl_delay(&self, useragent: &str) -> Option<u64> {
        self.entry_for(useragent).and_then(|entry| entry.delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UA: &str = "DesearchBulk/1.0 (+https://desearch.ai)";

    #[test]
    fn first_rule_and_default_group() {
        let robots = Robots::parse("User-agent: *\nAllow: /\nDisallow: /docker-credentials\n\nUser-agent: GPTBot\nDisallow: /\n").unwrap();
        assert!(robots.can_fetch(UA, "https://lium.io/docker-credentials").unwrap());
        let robots = Robots::parse("User-agent: *\nDisallow: /team\nCrawl-delay: 3\n").unwrap();
        assert!(!robots.can_fetch(UA, "https://a.io/team").unwrap());
        assert!(robots.can_fetch(UA, "https://a.io/").unwrap());
        assert_eq!(robots.crawl_delay(UA), Some(3));
        let robots = Robots::parse("User-agent: desearchbulk\nDisallow: /x\n\nUser-agent: *\nDisallow: /\n").unwrap();
        assert!(robots.can_fetch(UA, "https://a.io/y").unwrap());
        assert!(!robots.can_fetch(UA, "https://a.io/x").unwrap());
        assert!(Robots::parse("User-agent: *\nCrawl-delay: \u{b2}\n").is_err());
        assert_eq!(Robots::parse("User-agent: *\nCrawl-delay: \u{bd}\n").unwrap().crawl_delay(UA), None);
    }

    #[test]
    fn paths_normalize_as_python() {
        assert_eq!(normalize_path("/a b/%7e?x=1 2&y=%2F"), "/a%20b/~?x=1%202&y=/");
        assert_eq!(normalize_path("/caf\u{e9}"), "/caf%C3%A9");
        assert_eq!(normalize_path("/%ff"), "/%FF");
    }
}

//! What a site's robots.txt allows us to fetch, and which sitemaps it names.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::text;

pub const TOKEN: &str = "DesearchBot";

static SITEMAP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?im)^[\s\x1c-\x1f]*sitemap[\s\x1c-\x1f]*:[\s\x1c-\x1f]*([^\s\x1c-\x1f]+)").unwrap()
});

/// Whether our token may fetch the site's root, and any crawl delay it is asked to keep.
pub fn rules(robots: &str, token: &str) -> (bool, Option<f64>) {
    let mut groups: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut delays: HashMap<String, f64> = HashMap::new();
    let mut current: Vec<String> = Vec::new();
    let mut previous_was_agent = false;
    for raw in text::lines(robots) {
        let line = text::strip(raw.split('#').next().unwrap_or(""));
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let (name, value) = (text::strip(name).to_lowercase(), text::strip(value));
        if name == "user-agent" {
            if !previous_was_agent {
                current.clear();
            }
            let agent = value.to_lowercase();
            groups.entry(agent.clone()).or_default();
            current.push(agent);
            previous_was_agent = true;
            continue;
        }
        previous_was_agent = false;
        if name == "allow" || name == "disallow" {
            for agent in &current {
                groups.entry(agent.clone()).or_default().push((name.clone(), value.to_string()));
            }
        } else if name == "crawl-delay" {
            if let Some(delay) = seconds(value) {
                for agent in &current {
                    delays.insert(agent.clone(), delay);
                }
            }
        }
    }
    let token = token.to_lowercase();
    let agent = if groups.contains_key(&token) { token } else { "*".to_string() };
    let mut best: (Option<&str>, i64) = (None, -1);
    for (rule, path) in groups.get(&agent).into_iter().flatten() {
        let prefix = path.trim_end_matches('*');
        if path.is_empty() || !"/".starts_with(prefix) {
            continue;
        }
        let length = prefix.len() as i64;
        if length > best.1 || (length == best.1 && rule == "allow") {
            best = (Some(rule), length);
        }
    }
    (best.0 != Some("disallow"), delays.get(&agent).copied())
}

pub fn sitemaps(robots: &str) -> Vec<String> {
    SITEMAP.captures_iter(robots).map(|c| c[1].to_string()).collect()
}

/// Python's `float(value)`, keeping only finite, non-negative delays.
fn seconds(value: &str) -> Option<f64> {
    let bytes = value.as_bytes();
    let joined = bytes
        .iter()
        .enumerate()
        .all(|(i, &b)| b != b'_' || (i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)));
    if !joined {
        return None;
    }
    let delay: f64 = value.replace('_', "").parse().ok()?;
    (delay.is_finite() && delay >= 0.0).then_some(delay)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_the_most_specific_group() {
        let robots = "User-agent: *\nDisallow: /\n\nUser-agent: DesearchBot\nAllow: /\nCrawl-delay: 5\n";
        assert_eq!(rules(robots, TOKEN), (true, Some(5.0)));
        assert_eq!(rules("User-agent: *\nDisallow: /\n", TOKEN), (false, None));
        assert_eq!(rules("User-agent: *\nDisallow: /private\n", TOKEN), (true, None));
        assert_eq!(rules("User-agent: *\nDisallow: /*\nAllow: /\n", TOKEN), (true, None));
        assert_eq!(rules("User-agent: a\nUser-agent: *\nCrawl-delay: nan\n", TOKEN), (true, None));
    }

    #[test]
    fn finds_named_sitemaps() {
        assert_eq!(sitemaps("Sitemap: https://a.com/s.xml\n  sitemap :https://a.com/t.xml x\n"), ["https://a.com/s.xml", "https://a.com/t.xml"]);
    }
}

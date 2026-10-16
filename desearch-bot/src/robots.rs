//! What a site's robots.txt allows us to fetch, and which sitemaps it names.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::{text, urls};

pub const TOKEN: &str = "DesearchBot";

static SITEMAP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?im)^[\s\x1c-\x1f]*sitemap[\s\x1c-\x1f]*:[\s\x1c-\x1f]*([^\s\x1c-\x1f]+)").unwrap()
});

/// The Allow and Disallow paths of the group our token follows, escaped like stored URL paths.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Paths {
    pub allow: Vec<String>,
    pub disallow: Vec<String>,
}

impl Paths {
    /// Whether a path, with its query, may be fetched: the longest matching rule wins, Allow on a tie (RFC 9309).
    pub fn allows(&self, target: &str) -> bool {
        if target == "/robots.txt" {
            return true;
        }
        let longest = |rules: &[String]| rules.iter().filter(|rule| matches(rule, target)).map(String::len).max();
        match (longest(&self.allow), longest(&self.disallow)) {
            (_, None) => true,
            (None, Some(_)) => false,
            (Some(allow), Some(disallow)) => allow >= disallow,
        }
    }

    /// Whether a stored page (host and path, as `rest`) may be fetched; the file covers the domain and its www host only.
    pub fn allows_page(&self, domain: &str, rest: &[u8]) -> bool {
        let rest = String::from_utf8_lossy(rest);
        let (host, target) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
        host != domain || self.allows(if target.is_empty() { "/" } else { target })
    }
}

/// The paths our token may fetch, and any crawl delay it is asked to keep.
pub fn rules(robots: &str, token: &str) -> (Paths, Option<f64>) {
    let mut groups: HashMap<String, Paths> = HashMap::new();
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
            if value.is_empty() {
                continue;
            }
            // A trailing * changes nothing a prefix match does not already do.
            let path = value.trim_end_matches('*');
            let path = urls::tidy_target(if path.is_empty() { "*" } else { path });
            for agent in &current {
                let paths = groups.entry(agent.clone()).or_default();
                if name == "allow" { &mut paths.allow } else { &mut paths.disallow }.push(path.clone());
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
    let delay = delays.get(&agent).copied();
    (groups.remove(&agent).unwrap_or_default(), delay)
}

/// A rule matches from the start of the path; * stands for any run of characters and a final $ for the end.
fn matches(rule: &str, target: &str) -> bool {
    let (rule, anchored) = rule.strip_suffix('$').map_or((rule, false), |rule| (rule, true));
    let mut pieces = rule.split('*');
    let Some(mut rest) = target.strip_prefix(pieces.next().unwrap_or("")) else {
        return false;
    };
    let pieces: Vec<&str> = pieces.collect();
    for (i, piece) in pieces.iter().enumerate() {
        if anchored && i + 1 == pieces.len() {
            return rest.ends_with(piece);
        }
        let Some(at) = rest.find(piece) else {
            return false;
        };
        rest = &rest[at + piece.len()..];
    }
    !anchored || rest.is_empty()
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

    fn root(robots: &str) -> (bool, Option<f64>) {
        let (paths, delay) = rules(robots, TOKEN);
        (paths.allows("/"), delay)
    }

    #[test]
    fn follows_the_most_specific_group() {
        let robots = "User-agent: *\nDisallow: /\n\nUser-agent: DesearchBot\nAllow: /\nCrawl-delay: 5\n";
        assert_eq!(root(robots), (true, Some(5.0)));
        assert_eq!(root("User-agent: *\nDisallow: /\n"), (false, None));
        assert_eq!(root("User-agent: *\nDisallow: /private\n"), (true, None));
        assert_eq!(root("User-agent: *\nDisallow: /*\nAllow: /\n"), (true, None));
        assert_eq!(root("User-agent: a\nUser-agent: *\nCrawl-delay: nan\n"), (true, None));
        assert_eq!(root("User-agent: *\nDisallow:\n"), (true, None));
    }

    #[test]
    fn matches_paths_by_the_longest_rule() {
        let robots = "User-agent: *\nDisallow: /members/\nDisallow: /*.pdf$\nDisallow: /search?\nDisallow: /café\n\
                      Allow: /members/join\nDisallow: /*/print*\n";
        let (paths, _) = rules(robots, TOKEN);
        let allowed = |path: &str| paths.allows(path);
        assert!(!allowed("/members/") && !allowed("/members/a/b") && allowed("/members"));
        assert!(allowed("/members/join") && allowed("/members/join/now"));
        assert!(!allowed("/files/a.pdf") && allowed("/files/a.pdf?x=1") && allowed("/files/a.pdfx"));
        assert!(!allowed("/search?q=1") && allowed("/search"));
        assert!(!allowed("/caf%C3%A9/menu"));
        assert!(!allowed("/2024/story/print") && !allowed("/2024/print") && allowed("/print"));
        assert!(allowed("/robots.txt") && allowed("/news/story"));
        assert!(!rules("User-agent: *\nDisallow: *\n", TOKEN).0.allows("/anything"));
    }

    #[test]
    fn finds_named_sitemaps() {
        assert_eq!(sitemaps("Sitemap: https://a.com/s.xml\n  sitemap :https://a.com/t.xml x\n"), ["https://a.com/s.xml", "https://a.com/t.xml"]);
    }
}

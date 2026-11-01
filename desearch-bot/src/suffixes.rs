//! Registrable domains from the public suffix list, and the TLD groups candidates were filed under.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::text;

#[derive(Deserialize)]
pub(crate) struct Rules {
    pub english_market: HashSet<String>,
    pub generic: HashSet<String>,
    pub big_generic: HashSet<String>,
    pub blocked_operators: Vec<String>,
    pub not_blocked: HashSet<String>,
    pub dynamic_platforms: HashSet<String>,
    pub infrastructure: HashSet<String>,
    pub infrastructure_name: String,
}

pub(crate) static RULES: LazyLock<Rules> =
    LazyLock::new(|| serde_json::from_str(include_str!("../data/rules.json")).expect("data/rules.json is valid"));

pub struct PublicSuffixList {
    rules: HashSet<String>,
    exceptions: HashSet<String>,
}

impl PublicSuffixList {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        Ok(PublicSuffixList::parse(&std::fs::read_to_string(path)?))
    }

    pub fn parse(list: &str) -> Self {
        let mut rules = HashSet::new();
        let mut exceptions = HashSet::new();
        for line in text::lines(list) {
            let line = text::strip(line);
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            match line.strip_prefix('!') {
                Some(exception) => exceptions.insert(exception.to_string()),
                None => rules.insert(line.to_string()),
            };
        }
        PublicSuffixList { rules, exceptions }
    }

    pub fn registrable(&self, host: &str) -> Option<String> {
        let host = host.to_lowercase();
        let host = text::strip(&host).trim_matches('.');
        if host.is_empty() || host.contains([' ', '/', ':', '@']) {
            return None;
        }
        let labels: Vec<&str> = host.split('.').collect();
        if labels.len() < 2 || labels.iter().any(|label| label.is_empty()) {
            return None;
        }
        for i in 0..labels.len() {
            let candidate = labels[i..].join(".");
            if self.exceptions.contains(&candidate) {
                return Some(candidate);
            }
            let wildcard = format!("*.{}", labels[i + 1..].join("."));
            if self.rules.contains(&candidate) || self.rules.contains(&wildcard) {
                return (i > 0).then(|| labels[i - 1..].join("."));
            }
        }
        Some(labels[labels.len() - 2..].join("."))
    }
}

/// Names reserved for local and private networks, never delegated on the public internet.
const SPECIAL_USE: [&str; 13] = ["alt", "arpa", "corp", "example", "home", "internal", "intranet", "invalid", "lan", "local", "localhost", "onion", "test"];

/// The group of a registrable domain's public suffix; a top-level name longer than a country code is generic.
pub fn tld_group(domain: &str) -> &'static str {
    let suffix = domain.split_once('.').map_or("", |(_, suffix)| suffix);
    let under_english = || suffix.match_indices('.').any(|(i, _)| RULES.english_market.contains(&suffix[i + 1..]));
    let gtld = suffix.len() > 2
        && !suffix.contains('.')
        && !suffix.starts_with("xn--")
        && suffix.bytes().all(|b| b.is_ascii_lowercase())
        && !SPECIAL_USE.contains(&suffix);
    if RULES.big_generic.contains(suffix) {
        "big_generic"
    } else if RULES.english_market.contains(suffix) || under_english() {
        "en_cctld"
    } else if RULES.generic.contains(suffix) || gtld {
        "new_generic"
    } else {
        "other_cctld"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_registrable_domains() {
        let list = PublicSuffixList::parse("// comment\ncom\nco.uk\n*.ck\n!www.ck\n");
        assert_eq!(list.registrable("Blog.Example.COM.").as_deref(), Some("example.com"));
        assert_eq!(list.registrable("a.b.example.co.uk").as_deref(), Some("example.co.uk"));
        assert_eq!(list.registrable("x.y.ck").as_deref(), Some("x.y.ck"));
        assert_eq!(list.registrable("www.ck").as_deref(), Some("www.ck"));
        assert_eq!(list.registrable("co.uk"), None);
        assert_eq!(list.registrable("localhost"), None);
        assert_eq!(tld_group("example.com"), "big_generic");
    }

    #[test]
    fn groups_top_level_names_beyond_the_lists() {
        assert_eq!(tld_group("ms.now"), "new_generic");
        assert_eq!(tld_group("world.rugby"), "new_generic");
        assert_eq!(tld_group("last.fm"), "new_generic");
        assert_eq!(tld_group("www.nhs.uk"), "en_cctld");
        assert_eq!(tld_group("www.vic.gov.au"), "en_cctld");
        assert_eq!(tld_group("example.de"), "other_cctld");
        assert_eq!(tld_group("example.xn--p1ai"), "other_cctld");
        assert_eq!(tld_group("foo.github.io"), "other_cctld");
        assert_eq!(tld_group("a.s3.amazonaws.com"), "other_cctld");
        for host in ["mysite.local", "foo.test", "foo.localhost", "intranet.corp", "site.internal", "x.onion", "router.lan"] {
            assert_eq!(tld_group(host), "other_cctld", "{host}");
        }
    }
}

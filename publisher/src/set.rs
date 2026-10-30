//! The v1 set: which pages belong in the index, site by site, as `v1-set.json` lists them for the crawl list builder, the planner and the indexer.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use desearch::time::{civil_from_days, days_from_civil};
use serde_json::Value;

/// What the set takes of one site.
#[derive(Clone, Debug, PartialEq)]
pub enum Rule {
    /// Pages dated on or after `since`, an ISO date, or of any date without it, by `published` or a date in the URL path.
    News {
        since: Option<String>,
    },
    /// Every page, or every page under one of `paths`.
    Whole {
        paths: Vec<String>,
    },
    /// Pages from a bulk source, and the site's crawled pages too.
    Bulk,
    /// The host's own homepage and first-level pages, never the hosts under it.
    Landing,
    Out,
}

/// Path levels a landing page has at most, a leading English locale (`/en-us/`) not counted.
pub const LANDING_DEPTH: usize = 1;

/// Why a page is left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Unlisted,
    Undated,
    BeforeSince,
    OffPaths,
}

/// How a page the set takes is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admitted {
    ByRule,
    /// A news page with no date at all, in as an article because the file's `undated` is `article`.
    UndatedArticle,
}

#[derive(Debug, Default)]
pub struct Set {
    /// When the set was built, as its file says.
    pub built: Option<String>,
    sites: HashMap<String, Rule>,
    /// Undated news pages typed `article` are in.
    undated_articles: bool,
}

impl Set {
    pub fn parse(raw: &[u8]) -> Result<Set> {
        let file: Value = serde_json::from_slice(raw)?;
        let entries = file["sites"].as_array().ok_or_else(|| anyhow!("the set file has no sites"))?;
        let mut sites = HashMap::new();
        for entry in entries {
            let host = entry["host"].as_str().map(host_of).filter(|host| !host.is_empty()).ok_or_else(|| anyhow!("a set entry without a host: {entry}"))?;
            let rule = match entry["rule"].as_str() {
                Some("news") => match entry["since"].as_str() {
                    None => Rule::News { since: None },
                    Some(since) => Rule::News { since: Some(iso_date(since).ok_or_else(|| anyhow!("{host}: since {since:?} is not an ISO date"))?) },
                },
                Some("whole") => Rule::Whole { paths: entry["paths"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from).collect() },
                Some("bulk") => Rule::Bulk,
                Some("landing") => Rule::Landing,
                Some("out") => Rule::Out,
                other => bail!("{host}: unknown rule {other:?}"),
            };
            sites.insert(host, rule);
        }
        let undated_articles = match file["undated"].as_str() {
            None | Some("out") => false,
            Some("article") => true,
            Some(other) => bail!("undated {other:?} is not article or out"),
        };
        Ok(Set { built: file["built"].as_str().map(String::from), sites, undated_articles })
    }

    /// The listed host a host is, or is nearest under, and its rule.
    pub fn site(&self, host: &str) -> Option<(&str, &Rule)> {
        let host = host_of(host);
        let mut rest = host.as_str();
        loop {
            if let Some((listed, rule)) = self.sites.get_key_value(rest) {
                return Some((listed.as_str(), rule));
            }
            rest = rest.split_once('.')?.1;
        }
    }

    /// Whether the page at `url` is in, by the rule of its host's entry, and how.
    pub fn admit(&self, url: &str, published: &str, page_type: Option<&str>) -> Result<Admitted, Refusal> {
        let (host, path) = split_url(url);
        let Some((listed, rule)) = self.site(host) else { return Err(Refusal::Unlisted) };
        match rule {
            Rule::Landing if !own_host(host, listed) => Err(Refusal::Unlisted),
            Rule::Landing if !landing_page(url) => Err(Refusal::OffPaths),
            Rule::Landing => Ok(Admitted::ByRule),
            Rule::Out => Err(Refusal::Unlisted),
            Rule::News { since } => match page_date(published, url) {
                None if self.undated_articles && page_type == Some("article") => Ok(Admitted::UndatedArticle),
                None => Err(Refusal::Undated),
                Some(date) if since.as_ref().is_some_and(|since| date < *since) => Err(Refusal::BeforeSince),
                Some(_) => Ok(Admitted::ByRule),
            },
            Rule::Whole { paths } if !paths.is_empty() && !paths.iter().any(|prefix| path.starts_with(prefix.as_str())) => Err(Refusal::OffPaths),
            Rule::Whole { .. } | Rule::Bulk => Ok(Admitted::ByRule),
        }
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}

/// A URL's host and its path from the first `/`, without the query or fragment.
fn split_url(url: &str) -> (&str, &str) {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let rest = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    let (authority, path) = rest.find('/').map_or((rest, "/"), |at| (&rest[..at], &rest[at..]));
    let host = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    (host.split_once(':').map_or(host, |(host, _)| host), path)
}

/// Whether a host is the listed host itself, or its `www.` or `m.` host.
fn own_host(host: &str, listed: &str) -> bool {
    let host = host_of(host);
    host == listed || host.strip_prefix("m.") == Some(listed)
}

/// A homepage or first-level page with no query; a leading English locale (`en`, `en-us`) is no level, and a page under another locale (`de-de`) is none.
pub fn landing_page(url: &str) -> bool {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let rest = &rest[..rest.find('#').unwrap_or(rest.len())];
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let mut levels = rest.find('/').map_or("", |at| &rest[at..]).split('/').filter(|level| !level.is_empty()).peekable();
    match levels.peek() {
        Some(first) if first.eq_ignore_ascii_case("en") || (locale(first) && first[..2].eq_ignore_ascii_case("en")) => {
            levels.next();
        }
        Some(first) if locale(first) => return false,
        _ => {}
    }
    query.is_empty() && levels.count() <= LANDING_DEPTH
}

/// Two letters, `-` or `_`, two letters: `en-us`, `de_DE`.
fn locale(level: &str) -> bool {
    let b = level.as_bytes();
    b.len() == 5 && b[..2].iter().all(u8::is_ascii_alphabetic) && matches!(b[2], b'-' | b'_') && b[3..].iter().all(u8::is_ascii_alphabetic)
}

/// A host as the set names it: lowercase, without `www.`, and a mobile `<x>.m.<site>` as `<x>.<site>`.
pub fn host_of(host: &str) -> String {
    let host = host.trim().trim_end_matches('.').to_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    match host.split_once(".m.") {
        Some((first, rest)) if !first.contains('.') && rest.contains('.') => format!("{first}.{rest}"),
        _ => host.to_string(),
    }
}

/// A page's date: its `published` date when that opens with an ISO date, else a date in its URL path.
pub fn page_date(published: &str, url: &str) -> Option<String> {
    iso_date(published).or_else(|| url_date(url))
}

/// `YYYY-MM-DD` when `text` opens with a real calendar date, of any year.
pub fn iso_date(text: &str) -> Option<String> {
    let head = text.trim().get(..10)?;
    let b = head.as_bytes();
    let shaped = b[4] == b'-' && b[7] == b'-' && b.iter().enumerate().all(|(at, c)| at == 4 || at == 7 || c.is_ascii_digit());
    shaped.then(|| calendar(&head[..4], &head[5..7], &head[8..10])).flatten().map(|_| head.to_string())
}

/// A date in a URL's path: `/YYYY/MM/DD/`, `/YYYY/MM/` (the first of the month), or `YYYY-MM-DD` or `YYYYMMDD` standing apart from other digits.
pub fn url_date(url: &str) -> Option<String> {
    let parts: Vec<&str> = split_url(url).1.split('/').collect();
    for (at, part) in parts.iter().enumerate() {
        if part.len() == 4 {
            if let Some(month) = parts.get(at + 1).filter(|m| (1..=2).contains(&m.len())) {
                let day = parts.get(at + 2).filter(|d| (1..=2).contains(&d.len())).copied().unwrap_or("1");
                if let Some(found) = date(part, month, day).or_else(|| date(part, month, "1")) {
                    return Some(found);
                }
            }
        }
        if let Some(found) = digits_date(part) {
            return Some(found);
        }
    }
    None
}

fn digits_date(part: &str) -> Option<String> {
    let b = part.as_bytes();
    let digit = |at: usize| b.get(at).is_some_and(u8::is_ascii_digit);
    for at in (0..b.len()).filter(|&at| digit(at) && (at == 0 || !digit(at - 1))) {
        let piece = |from: usize, to: usize| part.get(at + from..at + to).unwrap_or("");
        let dashed = b.get(at + 4) == Some(&b'-') && b.get(at + 7) == Some(&b'-') && !digit(at + 10);
        if let Some(found) = dashed.then(|| date(piece(0, 4), piece(5, 7), piece(8, 10))).flatten() {
            return Some(found);
        }
        let compact = (at..at + 8).all(digit) && !digit(at + 8);
        if let Some(found) = compact.then(|| date(piece(0, 4), piece(4, 6), piece(6, 8))).flatten() {
            return Some(found);
        }
    }
    None
}

/// A URL's date: a real calendar date from 1990 to 2099.
fn date(year: &str, month: &str, day: &str) -> Option<String> {
    let (year, month, day) = calendar(year, month, day)?;
    (1990..=2099).contains(&year).then(|| format!("{year:04}-{month:02}-{day:02}"))
}

/// Year, month and day when they name a day that exists.
fn calendar(year: &str, month: &str, day: &str) -> Option<(i64, i64, i64)> {
    let found: (i64, i64, i64) = (year.parse().ok()?, month.parse().ok()?, day.parse().ok()?);
    ((1..=12).contains(&found.1) && civil_from_days(days_from_civil(found.0, found.1, found.2)) == found).then_some(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_come_from_published_or_the_url_path() {
        assert_eq!(page_date("2024-03-15T10:00:00+00:00", "https://a.example/x"), Some("2024-03-15".into()));
        assert_eq!(page_date("March 15, 2024", "https://a.example/2023/11/02/story"), Some("2023-11-02".into()));
        assert_eq!(url_date("https://a.example/news/2021/7/slug"), Some("2021-07-01".into()));
        assert_eq!(url_date("https://a.example/a/story-2022-05-09-slug"), Some("2022-05-09".into()));
        assert_eq!(url_date("https://a.example/a/2022-05-09-123"), Some("2022-05-09".into()));
        assert_eq!(url_date("https://a.example/20190412/slug.html"), Some("2019-04-12".into()));
        assert_eq!(url_date("https://a.example/a/123456789/slug"), None, "a long number is not a date");
        assert_eq!(url_date("https://a.example/p/2024"), None, "a year alone is not a date");
        assert_eq!(url_date("https://a.example/x?d=2024-01-01"), None, "only the path counts");
        assert_eq!(url_date("https://a.example/2024/13/x"), None);
        assert_eq!(url_date("https://a.example/a/2022-05-0\u{e9}"), None);
        assert_eq!(url_date("https://2024-01-01.example/a"), None, "the host is not the path");
        assert_eq!(url_date("https://a.example/2023/02/30/x"), Some("2023-02-01".into()), "a day that does not exist falls back to the month");
        assert_eq!(url_date("https://a.example/a/20230230/x"), None);
        assert_eq!(url_date("https://a.example/2024/02/29/x"), Some("2024-02-29".into()));
        assert_eq!(page_date("1985-06-01", "https://a.example/x"), Some("1985-06-01".into()), "a published date of any year");
        assert_eq!(page_date("2024-02-30", "https://a.example/2023/05/x"), Some("2023-05-01".into()));
    }

    #[test]
    fn a_host_takes_the_rule_of_the_nearest_listed_host_it_is_under() {
        let set = Set::parse(
            br#"{"sites": [
                {"host": "apnews.com", "rule": "news", "since": "2020-01-01"},
                {"host": "www.Nature.com", "rule": "whole"},
                {"host": "en.wikipedia.org", "rule": "bulk", "source": "wikipedia"},
                {"host": "blogs.nature.com", "rule": "out"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(set.site("www.media.nature.com").map(|s| s.0), Some("nature.com"));
        assert_eq!(set.site("en.m.wikipedia.org").map(|s| s.0), Some("en.wikipedia.org"));
        assert_eq!(set.admit("https://x.blogs.nature.com/a", "", None), Err(Refusal::Unlisted), "out shuts its host and everything under it");
        assert_eq!(
            set.admit("https://apnews.com/article/abc", "", Some("article")),
            Err(Refusal::Undated),
            "undated articles are out unless the file says otherwise"
        );
        assert_eq!(set.admit("https://apnews.com/article/abc", "2019-12-31", None), Err(Refusal::BeforeSince));
        assert_eq!(set.admit("https://APNEWS.com:443/2020/01/01/abc", "", None), Ok(Admitted::ByRule));
        assert!(Set::parse(br#"{"undated": "sometimes", "sites": []}"#).is_err());
        let landing = Set::parse(br#"{"sites": [{"host": "vendor.example", "rule": "landing"}]}"#).unwrap();
        assert_eq!(landing.admit("https://www.vendor.example/en_GB/about", "", None), Ok(Admitted::ByRule));
        assert_eq!(landing.admit("https://vendor.example/products/gpu", "", None), Err(Refusal::OffPaths));
        assert_eq!(landing.admit("https://shop.vendor.example/", "", None), Err(Refusal::Unlisted), "landing takes its own host only");
        assert!(!landing_page("https://vendor.example/fr-fr/") && !landing_page("https://vendor.example?p=1") && landing_page("https://vendor.example/a#b?c"));
        assert!(Set::parse(br#"{"sites": [{"host": "a.example", "rule": "news", "since": "soon"}]}"#).is_err());
    }
}

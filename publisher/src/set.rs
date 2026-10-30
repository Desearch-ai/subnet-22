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
    Out,
}

/// Why a page is left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Unlisted,
    Undated,
    BeforeSince,
    OffPaths,
}

/// How much a site's pages are worth embedding while the budget is short, best first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// A site a benchmark cites, whatever its rule.
    Benchmark,
    /// A news site taken for every year.
    News,
    Docs,
    Reference,
    /// A news site taken from its `since`.
    RecentNews,
    /// Every other site, whole academic and public ones among them.
    #[default]
    Other,
}

impl Class {
    pub const ALL: [Class; 6] = [Class::Benchmark, Class::News, Class::Docs, Class::Reference, Class::RecentNews, Class::Other];

    pub fn of(rule: &Rule, kind: Option<&str>, why: &[&str]) -> Class {
        match rule {
            Rule::Out => Class::Other,
            _ if why.iter().any(|word| !BACKING.contains(word)) => Class::Benchmark,
            Rule::News { since: None } => Class::News,
            Rule::News { since: Some(_) } => Class::RecentNews,
            _ if kind == Some("docs") => Class::Docs,
            _ if kind == Some("reference") => Class::Reference,
            _ => Class::Other,
        }
    }

    pub fn from_digit(digit: u8) -> Option<Class> {
        Class::ALL.get(usize::from(digit.checked_sub(b'0')?)).copied()
    }

    pub fn digit(self) -> char {
        char::from(b'0' + self as u8)
    }
}

/// The set's backings that name no benchmark; any other word in a site's `why` is a benchmark's name.
const BACKING: [&str; 10] = ["wikidata", "news-list", "news-major", "news", "judge", "academic", "public", "docs", "evergreen", "rank"];

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
    sites: HashMap<String, (Rule, Class)>,
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
                Some("out") => Rule::Out,
                other => bail!("{host}: unknown rule {other:?}"),
            };
            let why: Vec<&str> = entry["why"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
            let class = Class::of(&rule, entry["type"].as_str(), &why);
            sites.insert(host, (rule, class));
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
        self.entry(host).map(|(listed, (rule, _))| (listed, rule))
    }

    /// Whether the set takes pages of some site of `class`.
    pub fn has(&self, class: Class) -> bool {
        self.sites.values().any(|(rule, held)| *held == class && *rule != Rule::Out)
    }

    /// The class of the listed host a host is, or is nearest under; `Other` when none is listed.
    pub fn class(&self, host: &str) -> Class {
        self.entry(host).map_or(Class::Other, |(_, (_, class))| *class)
    }

    fn entry(&self, host: &str) -> Option<(&str, &(Rule, Class))> {
        let host = host_of(host);
        let mut rest = host.as_str();
        loop {
            if let Some((listed, entry)) = self.sites.get_key_value(rest) {
                return Some((listed.as_str(), entry));
            }
            rest = rest.split_once('.')?.1;
        }
    }

    /// Whether the page at `url` is in, by the rule of its host's entry, and how.
    pub fn admit(&self, url: &str, published: &str, page_type: Option<&str>) -> Result<Admitted, Refusal> {
        let (host, path) = split_url(url);
        let Some((_, rule)) = self.site(host) else { return Err(Refusal::Unlisted) };
        match rule {
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
        assert_eq!(set.class("apnews.com"), Class::RecentNews);
        assert!(Set::parse(br#"{"sites": [{"host": "a.example", "rule": "news", "since": "soon"}]}"#).is_err());
    }

    #[test]
    fn a_site_s_class_comes_from_its_why_then_its_rule_and_type() {
        let set = Set::parse(
            br#"{"sites": [
                {"host": "latimes.com", "rule": "news", "type": "news", "why": ["wikidata", "simpleqa"]},
                {"host": "apnews.com", "rule": "news", "type": "news", "why": ["wikidata", "news-list"]},
                {"host": "local.example", "rule": "news", "since": "2025-10-01", "type": "news", "why": ["news"]},
                {"host": "docs.python.org", "rule": "whole", "type": "docs", "why": ["docs"]},
                {"host": "britannica.com", "rule": "whole", "type": "reference", "why": ["judge", "wikidata"]},
                {"host": "en.wikipedia.org", "rule": "bulk", "source": "wikipedia", "type": "reference", "why": ["benchmark"]},
                {"host": "nature.com", "rule": "whole", "type": "academic", "why": ["academic", "wikidata"]},
                {"host": "sec.gov", "rule": "whole", "type": "public", "why": ["public"]},
                {"host": "nvidia.com", "rule": "whole", "type": "company"},
                {"host": "blogs.nature.com", "rule": "out", "why": ["spam"]}
            ]}"#,
        )
        .unwrap();
        let classes: Vec<Class> = [
            "latimes.com",
            "apnews.com",
            "local.example",
            "docs.python.org",
            "britannica.com",
            "en.m.wikipedia.org",
            "nature.com",
            "sec.gov",
            "nvidia.com",
            "unlisted.example",
        ]
        .iter()
        .map(|host| set.class(host))
        .collect();
        use Class::*;
        assert_eq!(classes, [Benchmark, News, RecentNews, Docs, Reference, Benchmark, Other, Other, Other, Other]);
        assert!(Class::ALL.windows(2).all(|pair| pair[0] < pair[1]), "best first");
        assert_eq!(set.class("blogs.nature.com"), Other, "an out site's reasons name no benchmark");
        assert!(Class::ALL.iter().all(|class| set.has(*class)));
        let news_only = Set::parse(br#"{"sites": [{"host": "a.example", "rule": "news"}, {"host": "b.example", "rule": "out", "why": ["spam"]}]}"#).unwrap();
        assert_eq!(Class::ALL.map(|class| news_only.has(class)), [false, true, false, false, false, false]);
        assert!(Class::ALL.iter().all(|class| Class::from_digit(class.digit() as u8) == Some(*class)));
    }
}

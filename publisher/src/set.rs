//! The v1 set: which pages belong in the index, site by site, as `v1-set.json` lists them for the crawl list builder, the planner and the indexer.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{anyhow, bail, Result};
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

#[derive(Debug, Default)]
pub struct Set {
    sites: HashMap<String, Rule>,
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
            sites.insert(host, rule);
        }
        Ok(Set { sites })
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

    /// Whether the page at `url` is in, by the rule of its host's entry.
    pub fn admit(&self, url: &str, published: &str) -> Result<(), Refusal> {
        let (host, path) = split_url(url);
        let Some((_, rule)) = self.site(host) else { return Err(Refusal::Unlisted) };
        match rule {
            Rule::Out => Err(Refusal::Unlisted),
            Rule::News { since } => match page_date(published, url) {
                None => Err(Refusal::Undated),
                Some(date) if since.as_ref().is_some_and(|since| date < *since) => Err(Refusal::BeforeSince),
                Some(_) => Ok(()),
            },
            Rule::Whole { paths } if !paths.is_empty() && !paths.iter().any(|prefix| path.starts_with(prefix.as_str())) => Err(Refusal::OffPaths),
            Rule::Whole { .. } | Rule::Bulk => Ok(()),
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

/// `YYYY-MM-DD` when `text` opens with a valid one.
pub fn iso_date(text: &str) -> Option<String> {
    let head = text.trim().get(..10)?;
    let b = head.as_bytes();
    let digits = |range: Range<usize>| b[range].iter().all(u8::is_ascii_digit);
    let shaped = digits(0..4) && b[4] == b'-' && digits(5..7) && b[7] == b'-' && digits(8..10);
    (shaped && ("01"..="12").contains(&&head[5..7]) && ("01"..="31").contains(&&head[8..10])).then(|| head.to_string())
}

/// The first of the month a URL's path names as `/YYYY/MM` or `/YYYY-MM`, followed by `/`, `-` or its end; the list builder's rule.
pub fn url_date(url: &str) -> Option<String> {
    url_month(url).map(|(year, month, _)| format!("{year}-{month}-01"))
}

/// `url_date` to the day when a two-digit day follows the month, for telling how new a page is.
pub fn url_day(url: &str) -> Option<String> {
    url_month(url).map(|(year, month, day)| format!("{year}-{month}-{}", day.unwrap_or("01")))
}

/// Year, month and any day of the first `/YYYY[/-]MM` date in a URL's path, years 1900-2099.
fn url_month(url: &str) -> Option<(&str, &str, Option<&str>)> {
    let path = split_url(url).1;
    let b = path.as_bytes();
    let digits = |from: usize, to: usize| to <= b.len() && b[from..to].iter().all(u8::is_ascii_digit);
    let mark = |at: usize| matches!(b.get(at), Some(b'/' | b'-'));
    (0..b.len()).filter(|&at| b[at] == b'/').find_map(|at| {
        let (y, m) = (at + 1, at + 6);
        let century = matches!(b.get(y..y + 2), Some(b"19" | b"20"));
        let month = digits(m, m + 2).then(|| &path[m..m + 2]).filter(|month| ("01"..="12").contains(month));
        let found = century && digits(y, y + 4) && mark(y + 4) && month.is_some() && (m + 2 == b.len() || mark(m + 2));
        let day =
            (mark(m + 2) && digits(m + 3, m + 5) && (m + 5 == b.len() || mark(m + 5))).then(|| &path[m + 3..m + 5]).filter(|day| ("01"..="31").contains(day));
        found.then(|| (&path[y..y + 4], month.expect("checked"), day))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_come_from_published_or_a_month_in_the_url_path_as_the_list_builder_reads_them() {
        assert_eq!(page_date("2024-03-15T10:00:00+00:00", "https://a.example/x"), Some("2024-03-15".into()));
        assert_eq!(page_date("March 15, 2024", "https://a.example/2023/11/02/story"), Some("2023-11-01".into()));
        assert_eq!(url_day("https://a.example/2023/11/02/story"), Some("2023-11-02".into()));
        assert_eq!(url_date("https://a.example/a/2022-05-09-123"), Some("2022-05-01".into()));
        assert_eq!(url_date("https://a.example/news/1998/03"), Some("1998-03-01".into()));
        for undated in [
            "https://a.example/news/2021/7/slug",
            "https://a.example/a/story-2022-05-09-slug",
            "https://a.example/20190412/slug.html",
            "https://a.example/a/123456789/slug",
            "https://a.example/p/2024",
            "https://a.example/x?d=2024-01-01",
            "https://a.example/2024/13/x",
            "https://a.example/2024/051/x",
            "https://2024-01-01.example/a",
        ] {
            assert_eq!(url_date(undated), None, "{undated}");
        }
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
        assert_eq!(set.admit("https://x.blogs.nature.com/a", ""), Err(Refusal::Unlisted), "out shuts its host and everything under it");
        assert_eq!(set.admit("https://apnews.com/article/abc", ""), Err(Refusal::Undated));
        assert_eq!(set.admit("https://apnews.com/article/abc", "2019-12-31"), Err(Refusal::BeforeSince));
        assert_eq!(set.admit("https://APNEWS.com:443/2020/01/01/abc", ""), Ok(()));
        assert!(Set::parse(br#"{"sites": [{"host": "a.example", "rule": "news", "since": "soon"}]}"#).is_err());
    }
}

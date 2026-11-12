//! `site_url`, `host_of`, `listed_site` and `home_links`: which pages of a subnet's site the fetch asks for.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use anyhow::Result;
use bulk_core::py::{self, ValueError};
use bulk_core::urlparse::{urljoin, urlsplit, urlunsplit, SplitResult};
use desearch::text::{fold_turkish_i, strip};
use regex::Regex;
use serde_json::Value;

pub const TOP_PAGES: usize = 4;

static PRIORITY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)about|docs?|documentation|team|faq|how|product|whitepaper|litepaper|mining|miner|validator|subnet|network|overview|learn|technology|research|blog")
        .expect("pattern compiles")
});
static SKIP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\.(?:png|jpe?g|gif|svg|webp|pdf|zip|mp4|ico|css|js)\n?\z|login|signin|sign-in|signup|register|privacy|terms|cookie|legal|/cart|mailto:|tel:",
    )
    .expect("pattern compiles")
});
static DOCS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\A(?:docs?|wiki|developers?|learn)\.|gitbook\.io|readthedocs|notion\.site|mintlify").expect("pattern compiles"));
static HTTP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\Ahttps?://").expect("pattern compiles"));

fn search(pattern: &Regex, text: &str) -> bool {
    pattern.is_match(&fold_turkish_i(text))
}

fn unsplit(scheme: &str, netloc: &str, path: &str, query: &str) -> String {
    urlunsplit(&SplitResult { scheme: scheme.into(), netloc: netloc.into(), path: path.into(), query: query.into(), fragment: String::new() })
}

/// `site_url`: the listed site as an https URL with a path and no fragment.
pub fn site_url(raw: &str) -> Result<String, ValueError> {
    let raw = strip(raw);
    let raw = if search(&HTTP, raw) { raw.to_string() } else { format!("https://{raw}") };
    let parts = urlsplit(&raw)?;
    let path = if parts.path.is_empty() { "/" } else { parts.path.as_str() };
    Ok(unsplit(&parts.scheme.to_lowercase(), &parts.netloc.to_lowercase(), path, &parts.query))
}

/// `host_of`: the URL's host, lowercased and without `www.`.
pub fn host_of(url: &str) -> Result<String, ValueError> {
    let netloc = urlsplit(url)?.netloc.to_lowercase();
    Ok(netloc.strip_prefix("www.").map_or(netloc.clone(), str::to_string))
}

/// `listed_site`: the subnet's listed website, stripped.
pub fn listed_site(subnet: &Value) -> Result<String> {
    let identity = py::dict_or_empty(py::get(subnet, "identity")?, "identity")?;
    match py::when_truthy(identity.get("subnet_url")) {
        Some(v) => Ok(strip(py::str_of(v, "subnet_url")?).to_string()),
        None => Ok(String::new()),
    }
}

fn without_www(host: &str) -> &str {
    host.strip_prefix("www.").unwrap_or(host)
}

/// `home_links` after lxml: same-site links worth fetching, priority paths first, and links to documentation sites.
pub fn home_links(hrefs: &[Option<String>], final_url: &str, home: &str) -> Result<(Vec<String>, Vec<String>), ValueError> {
    let own_netloc = urlsplit(final_url)?.netloc.to_lowercase();
    let own = without_www(&own_netloc).to_string();
    let mut links: Vec<String> = Vec::new();
    let mut docs: BTreeSet<String> = BTreeSet::new();
    for href in hrefs {
        let Some(href) = href.as_deref().filter(|h| !h.is_empty()) else { continue };
        let url = urljoin(final_url, href)?;
        let parts = urlsplit(&url)?;
        if parts.scheme != "http" && parts.scheme != "https" {
            continue;
        }
        let host = parts.netloc.to_lowercase();
        let path_docs = format!("{}.", parts.path.trim_start_matches('/'));
        if (search(&DOCS, &host) || search(&DOCS, &path_docs)) && (without_www(&host) != own || parts.path.starts_with("/docs")) {
            let path = if parts.path.is_empty() { "/" } else { parts.path.as_str() };
            docs.insert(unsplit(&parts.scheme, &host, path, ""));
        }
        if without_www(&host) != own || search(&SKIP, &url) {
            continue;
        }
        let trimmed = parts.path.trim_end_matches('/');
        let link = unsplit(&parts.scheme, &host, if trimmed.is_empty() { "/" } else { trimmed }, &parts.query);
        let bare = link.trim_end_matches('/');
        if bare == final_url.trim_end_matches('/') || bare == home.trim_end_matches('/') || links.contains(&link) {
            continue;
        }
        links.push(link);
    }
    let mut ranked: Vec<(u8, String)> = Vec::with_capacity(links.len());
    for link in links {
        let path = urlsplit(&link)?.path;
        ranked.push((if search(&PRIORITY, &path) { 0 } else { 1 }, link));
    }
    ranked.sort_by_key(|(rank, _)| *rank);
    Ok((ranked.into_iter().take(TOP_PAGES * 2).map(|(_, link)| link).collect(), docs.into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_links_stay_on_site_and_list_docs_sites() {
        let hrefs: Vec<Option<String>> =
            ["/about", "/pricing", "/login", "https://docs.example.ai/intro", "https://other.io/x"].iter().map(|h| Some(h.to_string())).collect();
        let (links, docs) = home_links(&hrefs, "https://www.example.ai/", "https://example.ai/").unwrap();
        assert_eq!(links, ["https://www.example.ai/about", "https://www.example.ai/pricing"]);
        assert_eq!(docs, ["https://docs.example.ai/intro"]);
    }

    #[test]
    fn listed_sites_become_https_urls() {
        assert_eq!(site_url(" Ridges.AI ").unwrap(), "https://ridges.ai/");
        assert_eq!(site_url("http://x.io/a?b#c").unwrap(), "http://x.io/a?b");
        assert_eq!(host_of("https://WWW.kept.io/").unwrap(), "kept.io");
    }
}

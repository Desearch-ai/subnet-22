//! `fetch_site` and `fetch_sites`: each listed site's home page and its top pages, within robots.txt, eight sites at a time.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use bulk_core::{py, pyjson, Counts};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use sha1::{Digest, Sha1};

use super::http::Http;
use super::links::{home_links, host_of, listed_site, site_url};
use super::robots::Robots;
use super::{Anchors, Pace};

pub const SITE_WORKERS: usize = 8;
pub const MIN_DELAY: f64 = 1.0;
pub const DENIED: &str = "denied";
pub const GONE: &str = "gone";
pub const UNLISTED: &str = "unlisted";
const ROBOTS_RETRY: Duration = Duration::from_secs(5);

fn removal(url: &str, why: &str) -> Value {
    json!({"url": url, "why": why})
}

fn unique<'a>(urls: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut seen = Vec::new();
    for url in urls {
        if !seen.contains(&url) {
            seen.push(url);
        }
    }
    seen
}

fn head(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// `robots_for`: the site's rules, none when it has none, and nothing fetched when robots.txt refuses us or can't be read.
async fn robots_for(http: &Http, base: &str, pace: Pace) -> (Option<Robots>, f64, String, Option<anyhow::Error>) {
    let mut got = None;
    for attempt in 0..2 {
        match http.get(&format!("{base}/robots.txt"), &[]).await {
            Ok(found) => {
                got = Some(found);
                break;
            }
            Err(_) if attempt == 1 => return (None, MIN_DELAY, "robots unreachable".into(), None),
            Err(_) => pace.sleep(ROBOTS_RETRY.as_secs_f64()).await,
        }
    }
    let got = got.expect("a robots answer");
    if got.status == 401 || got.status == 403 {
        return (None, MIN_DELAY, format!("robots {}: refused", got.status), None);
    }
    if got.status >= 500 {
        return (None, MIN_DELAY, format!("robots {}: unreachable", got.status), None);
    }
    if got.status != 200 {
        return (Some(Robots::default()), MIN_DELAY, "none".into(), None);
    }
    match Robots::parse(&String::from_utf8_lossy(&got.body)) {
        Ok(rules) => {
            let delay = rules.crawl_delay(super::USER_AGENT).filter(|d| *d > 0).map_or(MIN_DELAY, |d| d as f64);
            (Some(rules), MIN_DELAY.max(delay), "parsed".into(), None)
        }
        Err(error) => (None, MIN_DELAY, String::new(), Some(error)),
    }
}

/// `settle`: a site that gave no pages; its known pages are named removed when it refused us or is gone, else held for next time.
fn settle(mut found: Map<String, Value>, home: &str, known: &[String], why: Option<&str>) -> Map<String, Value> {
    found.insert("site".into(), json!(why.unwrap_or("held")));
    found.insert("known".into(), json!(known));
    if let Some(why) = why {
        let mut urls = vec![home];
        urls.extend(known.iter().map(String::as_str));
        if let Some(Value::Array(removed)) = found.get_mut("removed") {
            removed.extend(unique(urls).into_iter().map(|url| removal(url, why)));
        }
    }
    found
}

fn push(found: &mut Map<String, Value>, key: &str, item: Value) {
    if let Some(Value::Array(items)) = found.get_mut(key) {
        items.push(item);
    }
}

fn pages_len(found: &Map<String, Value>) -> usize {
    found.get("pages").and_then(Value::as_array).map_or(0, Vec::len)
}

/// `fetch_site`: None when the subnet lists no site.
pub async fn fetch_site(http: &Http, subnet: &Value, folder: &Path, known: Vec<String>, anchors: &Anchors, pace: Pace) -> Result<Option<Map<String, Value>>> {
    let raw = listed_site(subnet)?;
    if raw.is_empty() {
        return Ok(None);
    }
    let home = site_url(&raw)?;
    let mut found = Map::new();
    found.insert("netuid".into(), py::item(subnet, "netuid")?.clone());
    found.insert("listed".into(), json!(raw));
    found.insert("pages".into(), json!([]));
    found.insert("docs_links".into(), json!([]));
    found.insert("removed".into(), json!([]));
    let parts = bulk_core::urlparse::urlsplit(&home)?;
    let base = format!("{}://{}", parts.scheme, parts.netloc);
    let (rules, delay, robots, failed) = robots_for(http, &base, pace).await;
    if let Some(error) = failed {
        return Err(error);
    }
    found.insert("robots".into(), json!(robots));
    let Some(rules) = rules else {
        let why = robots.ends_with("refused").then_some(DENIED);
        return Ok(Some(settle(found, &home, &known, why)));
    };
    if !rules.can_fetch(super::USER_AGENT, &home)? {
        return Ok(Some(settle(found, &home, &known, Some(DENIED))));
    }
    let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::from([home.clone()]);
    let mut seen: HashSet<String> = HashSet::new();
    while !queue.is_empty() && pages_len(&found) < super::links::TOP_PAGES + 1 {
        let url = queue.pop_front().expect("a queued url");
        if !seen.insert(url.clone()) {
            continue;
        }
        if !rules.can_fetch(super::USER_AGENT, &url)? {
            push(&mut found, "pages", json!({"url": url, "skipped": "robots"}));
            push(&mut found, "removed", removal(&url, DENIED));
            continue;
        }
        pace.sleep(delay).await;
        let got = match http.get(&url, &[]).await {
            Ok(got) => got,
            Err(error) => {
                if url == home {
                    return Ok(Some(settle(found, &home, &known, None)));
                }
                push(&mut found, "pages", json!({"url": url, "error": head(&error.repr, 160)}));
                continue;
            }
        };
        let why = match got.status {
            401 | 403 => Some(DENIED),
            404 | 410 => Some(GONE),
            _ => None,
        };
        let html = got.status == 200 && got.content_type.contains("html");
        if url == home && !html {
            return Ok(Some(settle(found, &home, &known, why)));
        }
        if let Some(why) = why {
            for u in unique([url.as_str(), got.final_url.as_str()]) {
                push(&mut found, "removed", removal(u, why));
            }
        }
        let mut page = Map::new();
        page.insert("url".into(), json!(url));
        page.insert("final_url".into(), json!(got.final_url));
        page.insert("status".into(), json!(got.status));
        page.insert("content_type".into(), json!(got.content_type));
        if html {
            let digest: String = Sha1::digest(got.final_url.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
            let name = format!("{}.html", &digest[..16]);
            std::fs::write(folder.join("html").join(&name), &got.body)?;
            page.insert("file".into(), json!(name));
            page.insert("bytes".into(), json!(got.body.len()));
            if url == home {
                match anchors.hrefs(&got.body).await? {
                    Err(repr) => {
                        page.insert("link_error".into(), json!(head(&repr, 120)));
                    }
                    Ok(hrefs) => match home_links(&hrefs, &got.final_url, &home) {
                        Ok((links, docs)) => {
                            found.insert("docs_links".into(), json!(docs));
                            queue.extend(links);
                        }
                        Err(error) => {
                            page.insert("link_error".into(), json!(head(&error.repr(), 120)));
                        }
                    },
                }
            }
            push(&mut found, "pages", Value::Object(page));
        }
    }
    found.insert("site".into(), json!("ok"));
    Ok(Some(found))
}

fn text(value: &Value) -> &str {
    value.as_str().unwrap_or("")
}

/// `site_urls`: the pages a fetch built for a site, and those it carried over when the site gave none.
pub fn site_urls(site: &Value) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    for page in site.get("pages").and_then(Value::as_array).into_iter().flatten() {
        if py::truthy(page.get("file")) {
            urls.push(text(&page["final_url"]).to_string());
        }
    }
    for url in site.get("known").and_then(Value::as_array).into_iter().flatten() {
        urls.push(text(url).to_string());
    }
    let mut kept = Vec::new();
    for url in urls {
        if !kept.contains(&url) {
            kept.push(url);
        }
    }
    kept
}

/// `unlisted`: pages of sites no subnet lists any more, on hosts no subnet lists now.
pub fn unlisted(previous: &[Value], subnets: &[Value]) -> Result<Vec<Value>> {
    let mut listed = HashSet::new();
    for subnet in subnets {
        let raw = listed_site(subnet)?;
        if !raw.is_empty() {
            listed.insert(host_of(&site_url(&raw)?)?);
        }
    }
    let mut out = Vec::new();
    for site in previous {
        if !listed.contains(&host_of(&site_url(py::str_of(py::item(site, "listed")?, "listed")?)?)?) {
            out.extend(site_urls(site).iter().map(|url| removal(url, UNLISTED)));
        }
    }
    Ok(out)
}

/// `fetch_sites`: sites.json and removed.json in `folder`, with the site counts.
pub async fn fetch_sites(
    http: Arc<Http>,
    subnets: &[Value],
    folder: &Path,
    counts: &mut Counts,
    previous: &[Value],
    anchors: Arc<Anchors>,
    pace: Pace,
) -> Result<()> {
    std::fs::create_dir_all(folder.join("html"))?;
    let mut before: IndexMap<(String, String), Vec<String>> = IndexMap::new();
    for site in previous {
        let key = (py::item(site, "netuid")?.to_string(), host_of(&site_url(py::str_of(py::item(site, "listed")?, "listed")?)?)?);
        before.insert(key, site_urls(site));
    }
    let gate = Arc::new(tokio::sync::Semaphore::new(SITE_WORKERS));
    let mut tasks = Vec::with_capacity(subnets.len());
    for subnet in subnets {
        let raw = listed_site(subnet)?;
        let known = if raw.is_empty() {
            Vec::new()
        } else {
            before.get(&(py::item(subnet, "netuid")?.to_string(), host_of(&site_url(&raw)?)?)).cloned().unwrap_or_default()
        };
        let (http, gate, anchors, subnet, folder) = (http.clone(), gate.clone(), anchors.clone(), subnet.clone(), folder.to_path_buf());
        tasks.push(tokio::spawn(async move {
            let _permit = gate.acquire_owned().await?;
            fetch_site(&http, &subnet, &folder, known, &anchors, pace).await
        }));
    }
    let mut found: Vec<Map<String, Value>> = Vec::new();
    for task in tasks {
        if let Some(site) = task.await.map_err(|e| anyhow!("a site fetch stopped: {e}"))?? {
            found.push(site);
        }
    }
    let sites: Vec<Value> = found.iter().cloned().map(Value::Object).collect();
    std::fs::write(folder.join("sites.json"), pyjson::dumps(&Value::Array(sites.clone()), Some(1)))?;
    let mut removed: IndexMap<String, Value> = IndexMap::new();
    let mut all: Vec<Value> = found.iter().flat_map(|site| site.get("removed").and_then(Value::as_array).cloned().unwrap_or_default()).collect();
    all.extend(unlisted(previous, subnets)?);
    for record in all {
        removed.insert(text(&record["url"]).to_string(), record);
    }
    let removed: Vec<Value> = removed.into_values().collect();
    std::fs::write(folder.join(super::REMOVED), pyjson::dumps(&Value::Array(removed.clone()), Some(1)))?;
    for why in [DENIED, GONE, UNLISTED] {
        counts.set(&format!("site_removed_{why}"), removed.iter().filter(|r| text(&r["why"]) == why).count() as u64);
    }
    let pages = |site: &Map<String, Value>| site.get("pages").and_then(Value::as_array).cloned().unwrap_or_default();
    counts.set("sites_held", found.iter().filter(|s| s.get("site").and_then(Value::as_str) == Some("held")).count() as u64);
    counts.set("sites", found.len() as u64);
    counts.set("site_homes", found.iter().filter(|s| pages(s).first().is_some_and(|p| py::truthy(p.get("file")))).count() as u64);
    counts.set("site_pages", found.iter().map(|s| pages(s).iter().filter(|p| py::truthy(p.get("file"))).count() as u64).sum());
    counts.set("site_robots_skips", found.iter().map(|s| pages(s).iter().filter(|p| py::truthy(p.get("skipped"))).count() as u64).sum());
    counts.set("site_robots_refused", found.iter().filter(|s| s.get("robots").and_then(Value::as_str).is_some_and(|r| r.ends_with("refused"))).count() as u64);
    Ok(())
}

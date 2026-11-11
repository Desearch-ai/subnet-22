//! Whether a host has sitemaps the bot can make miner tasks from: robots.txt, then its named or guessed sitemaps, counted.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::net;
use crate::robots::{self, Paths};
use crate::sitemaps;
use crate::urls;
use crate::visit::gunzip;

pub const TOKEN: &str = "DesearchBot";
pub const ENOUGH: usize = 1_000;
const MAX_FILES: usize = 12;
const MAX_HOPS: usize = 5;
const MAX_BYTES: usize = 64 << 20;
const GUESSES: [&str; 2] = ["/sitemap.xml", "/sitemap_index.xml"];
const PACE: Duration = Duration::from_secs(1);

struct Host {
    client: reqwest::Client,
    host: String,
    last: Option<Instant>,
    fetches: usize,
}

impl Host {
    async fn get(&mut self, url: &str, limit: usize) -> Result<(u16, String, Vec<u8>), String> {
        let mut url = url.to_string();
        for _ in 0..=MAX_HOPS {
            if let Some(last) = self.last {
                let wait = PACE.saturating_sub(last.elapsed());
                tokio::time::sleep(wait).await;
            }
            self.last = Some(Instant::now());
            self.fetches += 1;
            let mut r = self.client.get(&url).send().await.map_err(|e| net::failure(&e, Duration::ZERO, None).to_string())?;
            let status = r.status().as_u16();
            if (300..400).contains(&status) {
                let Some(next) = r.headers().get("location").and_then(|l| l.to_str().ok()).and_then(|l| urls::join(&url, l).ok()) else {
                    return Ok((status, url, Vec::new()));
                };
                url = next;
                continue;
            }
            let body = net::read_body(&mut r, limit).await.map_err(|e| net::failure(&e, Duration::ZERO, None).to_string())?;
            return Ok((status, url, body));
        }
        Err("redirect_loop".into())
    }

    fn own(&self, url: &str) -> bool {
        let Ok(parts) = urls::split(url) else { return false };
        let host = parts.hostname().unwrap_or_default();
        host.strip_prefix("www.").unwrap_or(&host) == self.host && parts.port().ok().flatten().is_none_or(|p| p == 80 || p == 443)
    }
}

fn target(url: &str) -> String {
    let parts = urls::split(url).unwrap_or_default();
    let path = if parts.path.is_empty() { "/" } else { &parts.path };
    match parts.query {
        Some(q) => format!("{path}?{q}"),
        None => path.to_string(),
    }
}

/// One host's verdict: `refused`, `sitemap` (at least ENOUGH pages listed), `thin` (some), `none`, or `unreachable`.
pub async fn check(client: reqwest::Client, host: String) -> Value {
    let mut h = Host { client, host: host.clone(), last: None, fetches: 0 };
    let base = format!("https://{host}/");
    let (paths, named, robots_status) = match h.get(&format!("{base}robots.txt"), 1 << 20).await {
        Err(e) => return json!({"host": host, "class": "unreachable", "robots": e}),
        Ok((200, _, body)) => {
            let text = String::from_utf8_lossy(&body).into_owned();
            let (paths, _) = robots::rules(&text, TOKEN);
            (paths, robots::sitemaps(&text), 200)
        }
        Ok((404 | 410, _, _)) => (Paths::default(), Vec::new(), 404),
        Ok((status, _, _)) => return json!({"host": host, "class": "refused", "robots": status}),
    };
    if !paths.allows("/") {
        return json!({"host": host, "class": "refused", "robots": "disallow_root"});
    }
    let named_on_site: Vec<String> = named.iter().filter_map(|n| urls::join(&base, n).ok()).filter(|u| h.own(u) && paths.allows(&target(u))).collect();
    let guessed = named_on_site.is_empty();
    let mut queue: VecDeque<String> =
        if guessed { GUESSES.iter().map(|g| format!("https://{host}{g}")).collect() } else { named_on_site.iter().cloned().collect() };
    let mut seen = HashSet::new();
    let (mut pages, mut files, mut indexes, mut failed) = (0usize, 0usize, 0usize, 0usize);
    let mut sample = None;
    while let Some(url) = queue.pop_front() {
        if pages >= ENOUGH || files >= MAX_FILES || !seen.insert(url.clone()) {
            continue;
        }
        files += 1;
        let Ok((200, final_url, body)) = h.get(&url, MAX_BYTES).await else {
            failed += 1;
            continue;
        };
        if !h.own(&final_url) {
            failed += 1;
            continue;
        }
        let (kind, entries) = sitemaps::parse_entries(&gunzip(&body));
        match kind {
            "index" => {
                indexes += 1;
                queue.extend(entries.into_iter().map(|e| e.url).filter(|u| h.own(u) && paths.allows(&target(u))));
            }
            "urlset" => {
                let own: Vec<String> = entries.into_iter().map(|e| e.url).filter(|u| h.own(u)).collect();
                if sample.is_none() {
                    sample = own.first().cloned();
                }
                pages += own.len();
            }
            _ => failed += 1,
        }
    }
    let class = match pages {
        0 => "none",
        n if n >= ENOUGH => "sitemap",
        _ => "thin",
    };
    json!({"host": host, "class": class, "robots": robots_status, "named": named.len(), "named_on_site": named_on_site.len(),
           "guessed": guessed, "files": files, "indexes": indexes, "failed": failed, "pages": pages, "fetches": h.fetches, "sample": sample})
}

/// Every host's verdict, as JSON lines in the order they finish.
pub async fn run(hosts: Vec<String>, at_once: usize, read_timeout: Duration) -> anyhow::Result<()> {
    let client = net::client(net::PublicResolver::local(), read_timeout)?;
    let slots = Arc::new(Semaphore::new(at_once.max(1)));
    let mut tasks = tokio::task::JoinSet::new();
    for host in hosts {
        let permit = slots.clone().acquire_owned().await?;
        let client = client.clone();
        tasks.spawn(async move {
            let verdict = check(client, host).await;
            drop(permit);
            verdict
        });
        while let Some(done) = tasks.try_join_next() {
            println!("{}", done?);
        }
    }
    while let Some(done) = tasks.join_next().await {
        println!("{}", done?);
    }
    Ok(())
}

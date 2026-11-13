//! One site's crawl: robots.txt and the host's pace, sitemaps, then links breadth first.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

use super::fetch::{Http, Response, MAX_BYTES};
use super::hosts::{Host, MIN_DELAY};
use super::links::{self, host_of, in_scope, normalize, page_links, parse_html, robots_meta, Doc};
use super::robots::{Robots, Rules};
use super::scope;
use super::store::{self, Writer};
use super::{Counts, Crash};
use crate::setrule::SetRule;
use crate::urls;

pub const REDIRECTS: [u16; 5] = [301, 302, 303, 307, 308];
pub const MAX_SITEMAPS: usize = 50;
pub const MAX_HOPS: usize = 3;
pub const LANG_SAMPLE: i64 = 100;
/// One path's URLs that differ only in their query: past this many, the rest are a crawl trap.
pub const MAX_QUERY_VARIANTS: i64 = 5;
const SITEMAP_BYTES: usize = 64 << 20;
const HOP_BYTES: usize = 1 << 20;
const IDLE: Duration = Duration::from_millis(50);
const HTML: [&str; 2] = ["text/html", "application/xhtml+xml"];

/// One host to crawl: where to start, which paths count, and how much to take this run.
#[derive(Clone, Debug)]
pub struct Site {
    pub host: String,
    pub seeds: Vec<String>,
    pub prefixes: Vec<String>,
    pub max_pages: i64,
    pub max_depth: i64,
    pub sitemaps: bool,
    pub scheme: String,
    pub slots: Option<usize>,
    pub near: Vec<String>,
}

impl Site {
    pub fn new(host: &str) -> Self {
        Site {
            host: host.to_string(),
            seeds: Vec::new(),
            prefixes: Vec::new(),
            max_pages: 1_000,
            max_depth: 6,
            sitemaps: true,
            scheme: "https".into(),
            slots: None,
            near: Vec::new(),
        }
    }
}

/// URLs to fetch in order found, those under the site's `near` paths before the rest.
#[derive(Default)]
pub struct Frontier {
    near: Vec<String>,
    first: VecDeque<(String, i64)>,
    rest: VecDeque<(String, i64)>,
}

impl Frontier {
    pub fn new(near: &[String]) -> Self {
        Frontier { near: near.to_vec(), ..Frontier::default() }
    }

    fn lane(&mut self, url: &str) -> &mut VecDeque<(String, i64)> {
        let near = !self.near.is_empty() && urls::split(url).is_ok_and(|p| self.near.iter().any(|n| p.path.starts_with(n.as_str())));
        if near {
            &mut self.first
        } else {
            &mut self.rest
        }
    }

    pub fn append(&mut self, item: (String, i64)) {
        self.lane(&item.0).push_back(item);
    }

    pub fn appendleft(&mut self, item: (String, i64)) {
        self.lane(&item.0).push_front(item);
    }

    pub fn popleft(&mut self) -> Option<(String, i64)> {
        self.first.pop_front().or_else(|| self.rest.pop_front())
    }

    pub fn len(&self) -> usize {
        self.first.len() + self.rest.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn urls(&self) -> impl Iterator<Item = &str> {
        self.first.iter().chain(&self.rest).map(|(u, _)| u.as_str())
    }
}

/// What the run shares between its hosts.
pub struct Ctx {
    pub http: Arc<Http>,
    pub robots: Arc<Robots>,
    pub writer: Arc<Mutex<Writer>>,
    pub skip: Arc<HashSet<String>>,
    pub set: Option<Arc<SetRule>>,
    /// The pace floor between a host's requests, a second unless a test asks for less.
    pub min_delay: f64,
}

impl Ctx {
    pub fn admit(&self, url: &str) -> bool {
        self.set.as_ref().is_none_or(|set| set.may_admit(url))
    }

    fn cookies(&self, url: &str) -> indexmap::IndexMap<String, String> {
        let parts = urls::split(url).unwrap_or_default();
        let path = if parts.path.is_empty() { "/".to_string() } else { parts.path.clone() };
        self.http.jar.filter(parts.scheme.as_deref().unwrap_or(""), &parts.hostname().unwrap_or_default(), &path)
    }
}

pub struct Fetched {
    pub outcome: String,
    pub response: Response,
    pub doc: Option<Doc>,
    pub html: String,
    pub follow: bool,
    pub location: Option<String>,
}

/// One page in the host's turn: a keepable page, a redirect to look at, or why not.
pub async fn fetch_page(http: &Http, host: &Host, url: &str) -> Fetched {
    let r = {
        let _slot = host.slot().await;
        host.turn().await;
        let r = http.get(url, MAX_BYTES, false).await;
        host.answer(r.status, r.header("retry-after"));
        r
    };
    let fetched = |outcome: String, r: Response| Fetched { outcome, response: r, doc: None, html: String::new(), follow: false, location: None };
    if REDIRECTS.contains(&r.status) {
        if let Some(location) = r.header("location") {
            let location = normalize(location, url);
            return Fetched { location, ..fetched("redirect".into(), r) };
        }
    }
    if r.status != 200 {
        let outcome = if r.status != 0 { format!("status_{}", r.status) } else { format!("error_{}", r.error.as_deref().unwrap_or("None")) };
        return fetched(outcome, r);
    }
    if !HTML.contains(&r.content_type().as_str()) {
        return fetched("not_html".into(), r);
    }
    let text = r.text();
    let Some(doc) = parse_html(&text) else { return fetched("empty".into(), r) };
    let meta = robots_meta(&doc, r.header("x-robots-tag").unwrap_or(""));
    let outcome = if meta.contains("noindex") { "noindex" } else { "page" };
    Fetched { outcome: outcome.into(), follow: !meta.contains("nofollow"), doc: Some(doc), html: text, location: None, response: r }
}

/// Where an offsite redirect comes back to `url`'s host after giving it a cookie, else None.
pub async fn cookie_round_trip(ctx: &Ctx, url: &str, location: &str) -> Result<Option<String>, Crash> {
    let (origin, before) = (host_of(url), ctx.cookies(url));
    let mut hop = location.to_string();
    let mut hosts: HashMap<String, Host> = HashMap::new();
    for _ in 0..MAX_HOPS {
        let name = host_of(&hop);
        if name == origin {
            return Ok((ctx.cookies(url) != before).then_some(hop));
        }
        let rules = ctx.robots.rules(&name).await?;
        if !rules.allowed(&hop) {
            return Ok(None);
        }
        let auth = hosts.entry(name.clone()).or_insert_with(|| Host::new(&name, rules.delay, None, Some(ctx.min_delay)));
        auth.turn().await;
        let r = ctx.http.get(&hop, HOP_BYTES, false).await;
        auth.answer(r.status, r.header("retry-after"));
        let Some(location) = r.header("location").filter(|_| auth.stopped().is_none() && REDIRECTS.contains(&r.status)) else {
            return Ok(None);
        };
        match normalize(location, &hop) {
            Some(next) => hop = next,
            None => return Ok(None),
        }
    }
    Ok((host_of(&hop) == origin && ctx.cookies(url) != before).then_some(hop))
}

/// The host the site serves its home page from: www when the bare name is unreachable. An error here ends the whole run.
pub async fn home_host(ctx: &Ctx, name: &str) -> Result<String, Crash> {
    let mut name = name.to_string();
    let mut rules = ctx.robots.rules(&name).await?;
    if rules.refused.as_deref() == Some("robots_unreachable") && !name.starts_with("www.") {
        let www_name = format!("www.{name}");
        let www = ctx.robots.rules(&www_name).await?;
        if www.refused.is_none() {
            (name, rules) = (www_name, www);
        }
    }
    if rules.refused.is_some() {
        return Ok(name);
    }
    let r = ctx.http.get(&format!("https://{name}/"), MAX_BYTES, false).await;
    tokio::time::sleep(Duration::from_secs_f64(rules.delay.unwrap_or(0.0).max(ctx.min_delay.min(MIN_DELAY)))).await;
    if !REDIRECTS.contains(&r.status) {
        return Ok(name);
    }
    let to = host_of(&normalize(r.header("location").unwrap_or(""), &r.url).unwrap_or_default());
    let bare = |h: &str| h.strip_prefix("www.").unwrap_or(h).to_string();
    Ok(if !to.is_empty() && bare(&to) == bare(&name) { to } else { name })
}

async fn sitemap_urls(ctx: &Ctx, sitemaps: Vec<String>, host: &Host, counts: &Mutex<Counts>, limit: i64) -> Result<Vec<String>, Crash> {
    let mut pages = Vec::new();
    let mut todo: VecDeque<String> = sitemaps.into();
    let mut fetched = 0;
    while fetched < MAX_SITEMAPS && (pages.len() as i64) < limit && host.stopped().is_none() {
        let Some(url) = todo.pop_front() else { break };
        host.turn().await;
        let r = ctx.http.get(&url, SITEMAP_BYTES, false).await;
        host.answer(r.status, r.header("retry-after"));
        fetched += 1;
        if r.status != 200 {
            counts.lock().unwrap().add(&format!("sitemap_status_{}", r.status), 1);
            continue;
        }
        let (found, children) = links::sitemap_entries(&r.body)?;
        pages.extend(found);
        todo.extend(children);
        counts.lock().unwrap().add("sitemap", 1);
    }
    Ok(pages)
}

#[derive(Default)]
struct State {
    queue: Frontier,
    seen: HashSet<String>,
    foreign: HashSet<String>,
    tripped: HashSet<String>,
    variants: HashMap<String, i64>,
    judged: i64,
    other: i64,
    taken: i64,
    busy: usize,
    other_site: bool,
}

struct Crawl<'a> {
    ctx: &'a Ctx,
    site: &'a Site,
    rules: Arc<Rules>,
    host: Host,
    counts: Mutex<Counts>,
    st: Mutex<State>,
}

impl Crawl<'_> {
    fn count(&self, key: &str) {
        self.counts.lock().unwrap().add(key, 1);
    }

    fn push(&self, url: Option<&str>, depth: i64, seed: bool) {
        let Some(url) = url.filter(|u| !u.is_empty()) else { return };
        let mut st = self.st.lock().unwrap();
        if st.seen.contains(url) || !in_scope(url, &self.site.host, &self.site.prefixes) {
            return;
        }
        st.seen.insert(url.to_string());
        let parts = urls::split(url).unwrap_or_default();
        if parts.query.as_deref().is_some_and(|q| !q.is_empty()) && !seed {
            let n = st.variants.entry(parts.path.clone()).or_insert(0);
            *n += 1;
            if *n > MAX_QUERY_VARIANTS {
                self.count("query_variants_capped");
                return;
            }
        }
        if self.ctx.skip.contains(url) {
            self.count("skipped_held");
        } else if scope::locale_level(url).is_some_and(|level| st.foreign.contains(&level)) {
            self.count("other_language_path");
        } else if !self.rules.allowed(url) {
            self.count("robots_disallowed");
        } else if !seed && !self.ctx.admit(url) {
            self.count("out_of_set");
        } else {
            st.queue.append((url.to_string(), depth));
        }
    }

    async fn worker(&self) -> Result<(), Crash> {
        loop {
            let next = {
                let mut st = self.st.lock().unwrap();
                if !(st.taken < self.site.max_pages && self.host.stopped().is_none() && !st.other_site) {
                    return Ok(());
                }
                match st.queue.popleft() {
                    Some(item) => {
                        st.busy += 1;
                        Some(item)
                    }
                    None if st.busy == 0 => return Ok(()),
                    None => None,
                }
            };
            let Some((url, depth)) = next else {
                tokio::time::sleep(IDLE).await;
                continue;
            };
            let done = self.step(&url, depth).await;
            self.st.lock().unwrap().busy -= 1;
            done?;
        }
    }

    async fn step(&self, url: &str, depth: i64) -> Result<(), Crash> {
        if scope::locale_level(url).is_some_and(|level| self.st.lock().unwrap().foreign.contains(&level)) {
            self.count("other_language_path");
            return Ok(());
        }
        let f = fetch_page(&self.ctx.http, &self.host, url).await;
        if f.outcome == "redirect" {
            match f.location.as_deref() {
                Some(location) if host_of(location) == self.site.host => {
                    self.count("redirect");
                    self.push(Some(location), depth, false);
                }
                Some(location) if !self.st.lock().unwrap().tripped.contains(url) => match cookie_round_trip(self.ctx, url, location).await? {
                    Some(back) => {
                        self.st.lock().unwrap().tripped.insert(url.to_string());
                        self.count("cookie_round_trip");
                        if back == url {
                            self.st.lock().unwrap().queue.appendleft((url.to_string(), depth));
                        } else {
                            self.push(Some(&back), depth, false);
                        }
                    }
                    None => self.count("redirect_offsite"),
                },
                _ => self.count("redirect_offsite"),
            }
            return Ok(());
        }
        if let Some(doc) = &f.doc {
            let lang = scope::declared_lang(doc);
            let other = {
                let mut st = self.st.lock().unwrap();
                st.judged += 1;
                let other = !lang.is_empty() && !scope::wanted_lang(&lang);
                if other {
                    st.other += 1;
                    if let Some(level) = scope::locale_level(url) {
                        st.foreign.insert(level);
                    }
                }
                other
            };
            if other {
                self.count("other_language");
                if depth == 0 && f.follow {
                    for link in page_links(doc, url)? {
                        self.push(Some(&link), 1, false);
                    }
                }
                let mut st = self.st.lock().unwrap();
                if st.judged >= LANG_SAMPLE && st.other * 10 >= st.judged * 9 && !st.other_site {
                    self.count("other_language_site");
                    st.other_site = true;
                }
                return Ok(());
            }
            if f.follow && depth < self.site.max_depth {
                for link in page_links(doc, url)? {
                    self.push(Some(&link), depth + 1, false);
                }
            }
        }
        self.count(&f.outcome);
        if f.outcome == "page" && self.ctx.admit(url) {
            let mut st = self.st.lock().unwrap();
            if st.taken < self.site.max_pages {
                let line = store::record_line(&f.response, &f.html, "site");
                self.ctx.writer.lock().unwrap().add(&line).map_err(|_| Crash("OSError"))?;
                st.taken += 1;
            }
        }
        Ok(())
    }
}

/// Every future to its end, or the first error, which drops the rest.
pub(crate) async fn try_join_all<F: Future<Output = Result<(), Crash>>>(futures: Vec<F>) -> Result<(), Crash> {
    let mut futures: Vec<Option<Pin<Box<F>>>> = futures.into_iter().map(|f| Some(Box::pin(f))).collect();
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for slot in &mut futures {
            if let Some(f) = slot {
                match f.as_mut().poll(cx) {
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Ready(Ok(())) => *slot = None,
                    Poll::Pending => pending = true,
                }
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    })
    .await
}

/// Up to `max_pages` pages of the site into the writer; counts say what happened to each URL, `found` gets the URLs left queued.
pub async fn crawl(ctx: &Ctx, site: &Site, found: Option<&Mutex<Vec<String>>>) -> Result<Counts, Crash> {
    let rules = ctx.robots.rules(&site.host).await?;
    if let Some(why) = &rules.refused {
        let mut counts = Counts::default();
        counts.add(why, 1);
        return Ok(counts);
    }
    let host = Host::new(&site.host, rules.delay, site.slots, Some(ctx.min_delay));
    let root = format!("{}://{}/", site.scheme, site.host);
    let crawl = Crawl {
        ctx,
        site,
        rules: rules.clone(),
        host,
        counts: Mutex::new(Counts::default()),
        st: Mutex::new(State { queue: Frontier::new(&site.near), ..State::default() }),
    };
    for seed in std::iter::once(&root).chain(&site.seeds) {
        crawl.push(normalize(seed, "").as_deref(), 0, true);
    }
    if site.sitemaps {
        let mut listed = rules.sitemaps.iter().map(|s| urls::join(&root, s)).collect::<Result<Vec<_>, _>>().map_err(|_| Crash("ValueError"))?;
        if listed.is_empty() {
            listed.push(urls::join(&root, "/sitemap.xml").map_err(|_| Crash("ValueError"))?);
        }
        for url in sitemap_urls(ctx, listed, &crawl.host, &crawl.counts, 10 * site.max_pages).await? {
            crawl.push(normalize(&url, "").as_deref(), 1, false);
        }
    }
    try_join_all((0..crawl.host.slots).map(|_| crawl.worker()).collect()).await?;
    let mut counts = crawl.counts.into_inner().unwrap();
    if let Some(why) = crawl.host.stopped() {
        counts.add(&why, 1);
    }
    let st = crawl.st.into_inner().unwrap();
    counts.add("left_in_queue", st.queue.len() as i64);
    if let Some(found) = found {
        found.lock().unwrap().extend(st.queue.urls().map(str::to_string));
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frontier_serves_near_urls_before_the_rest() {
        let mut q = Frontier::new(&["/docs/".to_string()]);
        for url in ["https://a.test/x", "https://a.test/docs/1", "https://a.test/y", "https://a.test/docs/2"] {
            q.append((url.to_string(), 1));
        }
        q.appendleft(("https://a.test/z".to_string(), 0));
        assert_eq!(q.len(), 5);
        let order: Vec<String> = std::iter::from_fn(|| q.popleft().map(|(u, _)| u)).collect();
        assert_eq!(order, ["https://a.test/docs/1", "https://a.test/docs/2", "https://a.test/z", "https://a.test/x", "https://a.test/y"]);
    }
}

//! One visit to one domain: read its rules, read its sitemaps, keep what is new.

use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use flate2::{Decompress, FlushDecompress, Status};
use indexmap::IndexMap;
use reqwest::header::HeaderMap;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use crate::buckets::{sitemap_id, Buckets};
use crate::churn;
use crate::listed::fingerprint;
use crate::robots::{self, Paths};
use crate::schedule::{self, Trust, DAY, SECOND};
use crate::signing::Signer;
use crate::sitemaps::{self, Entry};
use crate::states::{Outcome, State, JITTER};
use crate::suffixes::PublicSuffixList;
use crate::urls::{self, Listing, Normaliser};
use crate::{homepage, isodate, net};

pub const MIN_HOST_INTERVAL: f64 = 1.0;
/// Some sites ask for hours between requests; past this we slow down no further.
const MAX_CRAWL_DELAY: f64 = 60.0;
const MAX_REDIRECTS: usize = 5;
const SITEMAP_GUESSES: [&str; 2] = ["/sitemap.xml", "/sitemap_index.xml"];
/// Tried after the sitemap guesses, on listed domains only: a feed names a site's newest pages when it lists no sitemap.
const FEED_GUESSES: [&str; 2] = ["/feed", "/rss.xml"];
const MAX_DEPTH: i64 = 3;
const MAX_FILES: usize = 40;
/// The most unread sitemap files one visit records for the visits after it.
const MAX_DEFERRED: usize = 10_000;
/// Three failed requests in a row end a visit; more would only add to the site's trouble.
const MAX_FAILURES_IN_ROW: usize = 3;
const MIN_URLS: i64 = 10;
const ROBOTS_EVERY: i64 = DAY;
const MAX_ROBOTS_BYTES: usize = 512 * 1024;
const MAX_HOMEPAGE_BYTES: usize = 512 * 1024;
/// The sitemap protocol caps a file at 50 MB uncompressed, and so do we.
pub const MAX_SITEMAP_BYTES: usize = 50 * 1024 * 1024;

/// A failure on our side, named like the Python exception; the domain is simply tried again later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crash(pub String);

impl Crash {
    fn new(name: &str) -> Self {
        Crash(name.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnownSitemap {
    pub id: i64,
    pub url: String,
    pub kind: Option<String>,
    pub depth: i64,
    pub parent_id: Option<i64>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub content_hash: Option<String>,
    pub interval: i64,
    pub next_check_at: Option<i64>,
    pub trust: Trust,
    pub index_lastmod: Option<String>,
    pub url_count: i64,
    /// The file answered last time, so its URLs count toward the domain's total.
    pub ok: bool,
    pub suspect_at: Option<i64>,
    pub sketch: Vec<u64>,
    pub generated_until: Option<i64>,
    /// A Google News sitemap when last read, so a quiet read keeps its news pace.
    pub google_news: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Known {
    pub host: String,
    pub state: State,
    pub failures: i64,
    pub last_ok_at: Option<i64>,
    pub robots_checked_at: Option<i64>,
    pub robots_allows: Option<bool>,
    pub robots_paths: Paths,
    pub crawl_delay: Option<f64>,
    pub language: Option<String>,
    pub categories: Vec<String>,
    pub sitemaps: IndexMap<String, KnownSitemap>,
    pub canonical_host: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SitemapUpdate {
    pub id: i64,
    pub url: String,
    pub kind: Option<String>,
    pub depth: i64,
    pub parent_id: Option<i64>,
    pub status: &'static str,
    pub error: Option<String>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub content_hash: Option<String>,
    pub changed: bool,
    pub url_count: i64,
    pub child_count: i64,
    pub interval: i64,
    pub next_check_at: Option<i64>,
    pub trust: Trust,
    pub index_lastmod: Option<String>,
    pub suspect_at: Option<i64>,
    pub sketch: Vec<u64>,
    pub generated_until: Option<i64>,
    pub google_news: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Visit {
    pub host: String,
    pub outcome: Outcome,
    pub reason: Option<String>,
    pub canonical_host: Option<String>,
    pub robots_read: bool,
    pub robots_status: Option<u16>,
    pub robots_allows: Option<bool>,
    pub robots_paths: Paths,
    pub crawl_delay: Option<f64>,
    pub language: Option<String>,
    pub declared_lang: Option<String>,
    pub home_chars: Option<i64>,
    pub sitemaps: Vec<SitemapUpdate>,
    pub listed: i64,
    pub new: i64,
    pub moved: i64,
    pub requests: i64,
    pub deferred: Vec<(String, i64, Option<i64>)>,
    pub cut_short: bool,
}

impl Visit {
    pub fn new(host: &str) -> Self {
        Visit {
            host: host.to_string(),
            outcome: Outcome::Sitemap,
            reason: None,
            canonical_host: None,
            robots_read: false,
            robots_status: None,
            robots_allows: None,
            robots_paths: Paths::default(),
            crawl_delay: None,
            language: None,
            declared_lang: None,
            home_chars: None,
            sitemaps: Vec::new(),
            listed: 0,
            new: 0,
            moved: 0,
            requests: 0,
            deferred: Vec::new(),
            cut_short: false,
        }
    }
}

pub struct Answer {
    pub status: u16,
    pub body: Vec<u8>,
    pub headers: HeaderMap,
    pub url: String,
    /// Held until the body is parsed, when it was fetched under the sitemap budget.
    pub permit: Option<OwnedSemaphorePermit>,
}

/// A fixed number of permits, and how many are taken.
#[derive(Clone)]
pub struct Slots {
    permits: Arc<Semaphore>,
    size: usize,
}

impl Slots {
    pub fn new(size: usize) -> Self {
        Slots { permits: Arc::new(Semaphore::new(size)), size }
    }

    pub fn busy(&self) -> usize {
        self.size - self.permits.available_permits()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// A permit if one is free right now.
    pub fn try_take(&self) -> Option<OwnedSemaphorePermit> {
        self.permits.clone().try_acquire_owned().ok()
    }

    pub async fn take(&self) -> Result<OwnedSemaphorePermit, Crash> {
        self.permits.clone().acquire_owned().await.map_err(|_| Crash::new("CancelledError"))
    }
}

/// One host's request clock, so a slow host never shares a budget with a fast one.
struct Pacer {
    interval: f64,
    next: Option<Instant>,
}

impl Pacer {
    fn new(delay: Option<f64>, floor: f64) -> Self {
        Pacer { interval: crawl_delay(delay).max(floor), next: None }
    }

    /// Adopt a longer interval, pushing back a request already booked at the shorter one.
    fn slow_to(&mut self, delay: Option<f64>) {
        let delay = crawl_delay(delay);
        if delay <= self.interval {
            return;
        }
        if let Some(next) = self.next.as_mut() {
            *next += Duration::from_secs_f64(delay - self.interval);
        }
        self.interval = delay;
    }

    /// Hold the next request back at least this long from now.
    fn rest(&mut self, seconds: f64) {
        let until = Instant::now() + Duration::from_secs_f64(seconds.clamp(0.0, MAX_CRAWL_DELAY));
        self.next = Some(self.next.map_or(until, |next| next.max(until)));
    }

    async fn wait(&mut self) {
        if let Some(next) = self.next {
            tokio::time::sleep_until(next).await;
        }
        self.next = Some(Instant::now() + Duration::from_secs_f64(self.interval));
    }
}

/// Everything a visit needs besides the domain itself.
pub struct Visitor {
    pub client: reqwest::Client,
    pub buckets: Arc<Buckets>,
    pub suffixes: Arc<PublicSuffixList>,
    pub signer: Option<Arc<Signer>>,
    /// The language of a homepage's text, or None when there is too little to tell.
    pub language: Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
    pub floor: f64,
    /// How long connecting may take before a request fails, to tell connect timeouts from read timeouts.
    pub connect_timeout: Duration,
    /// Parsing and storing run on the blocking pool, a few files per core at a time.
    pub cpu: Slots,
    /// Sitemap files fetched or waiting to be parsed at once; bounds the memory their bodies take.
    pub bodies: Slots,
    /// Set while the disk is nearly full: visits read no more sitemap files and leave the rest for later.
    pub pause: Arc<AtomicBool>,
    /// Visits at once to domains with thousands of sitemap records, which each hold those records in memory.
    pub heavy: Slots,
}

impl Visitor {
    /// Visit one domain and report everything the visit learned.
    pub async fn visit(&self, known: Arc<Known>, now: i64) -> Result<Visit, Crash> {
        let mut run = Run {
            visitor: self,
            pacer: Pacer::new(known.crawl_delay, self.floor),
            result: Visit {
                robots_allows: known.robots_allows,
                robots_paths: known.robots_paths.clone(),
                crawl_delay: known.crawl_delay,
                ..Visit::new(&known.host)
            },
            known,
            now,
            guesses: HashSet::new(),
            found: false,
            fetches: 0,
            failed_in_row: 0,
            kept: 0,
            answered: false,
            network_error: None,
        };
        run.go().await?;
        Ok(run.result)
    }

    async fn cpu<T: Send + 'static>(&self, work: impl FnOnce() -> T + Send + 'static) -> Result<T, Crash> {
        let _permit = self.cpu.take().await?;
        tokio::task::spawn_blocking(work).await.map_err(|_| Crash::new("RuntimeError"))
    }
}

type Child = (String, i64, Option<i64>, Option<String>, bool);

struct Run<'a> {
    visitor: &'a Visitor,
    known: Arc<Known>,
    now: i64,
    pacer: Pacer,
    result: Visit,
    guesses: HashSet<String>,
    found: bool,
    fetches: usize,
    failed_in_row: usize,
    kept: i64,
    answered: bool,
    network_error: Option<String>,
}

impl Run<'_> {
    async fn go(&mut self) -> Result<(), Crash> {
        let roots = if self.robots_due() {
            match self.robots().await? {
                Some(roots) => roots,
                None => return Ok(()),
            }
        } else {
            self.roots(&[])?
        };
        self.walk(roots).await?;
        self.settle();
        if self.result.outcome == Outcome::Sitemap && self.needs_language() {
            self.language().await?;
        }
        Ok(())
    }

    fn robots_due(&self) -> bool {
        // The daily recheck can land up to 10% early and should still read robots.txt.
        self.known.state != State::Active
            || self.known.robots_checked_at.is_none_or(|checked| (self.now - checked) as f64 >= ROBOTS_EVERY as f64 * (1.0 - JITTER))
    }

    fn roots(&mut self, named: &[String]) -> Result<Vec<String>, Crash> {
        let base = format!("https://{}/", self.known.host);
        let joined = named.iter().map(|url| urls::join(&base, url)).collect::<Result<Vec<_>, _>>().map_err(|_| Crash::new("ValueError"))?;
        let known = self.known.sitemaps.values().filter(|s| s.depth == 0).map(|s| s.url.clone());
        let mut seen = HashSet::new();
        let roots: Vec<String> = joined.into_iter().chain(known).filter(|url| seen.insert(url.clone())).collect();
        if !roots.is_empty() {
            return Ok(roots);
        }
        let guesses = guess_paths(self.visitor.buckets.lists(&self.known.host)).map(|path| urls::join(&base, path)).collect::<Result<Vec<_>, _>>().map_err(|_| Crash::new("ValueError"))?;
        self.guesses = guesses.iter().cloned().collect();
        Ok(guesses)
    }

    /// Read robots.txt; None means the visit ends here.
    async fn robots(&mut self) -> Result<Option<Vec<String>>, Crash> {
        let Some(answer) = self.first_answer("/robots.txt", MAX_ROBOTS_BYTES).await else {
            return Ok(None);
        };
        if self.moved_away(&answer) {
            return Ok(None);
        }
        self.result.robots_read = true;
        self.result.robots_status = Some(answer.status);
        if answer.status >= 500 {
            self.stop(Outcome::Unreachable, Some(format!("robots_{}", answer.status)));
            return Ok(None);
        }
        let (mut paths, mut delay, mut named) = (Paths::default(), None, Vec::new());
        if answer.status == 200 {
            let text = homepage::decode(&answer.body);
            (paths, delay) = robots::rules(&text, robots::TOKEN);
            named = robots::sitemaps(&text);
        }
        let allowed = paths.allows("/");
        self.result.robots_allows = Some(allowed);
        self.result.robots_paths = paths;
        self.result.crawl_delay = delay;
        self.pacer.slow_to(delay);
        if !allowed {
            self.stop(Outcome::Blocked, Some("robots_disallow".into()));
            return Ok(None);
        }
        self.roots(&named).map(Some)
    }

    async fn walk(&mut self, roots: Vec<String>) -> Result<(), Crash> {
        let known = self.known.clone();
        let mut queue: VecDeque<Child> = roots.into_iter().map(|url| (url, 0, None, None, false)).collect();
        queue.extend(
            known.sitemaps.values().filter(|s| s.depth > 0 && due(s, self.now)).map(|s| (s.url.clone(), s.depth, s.parent_id, None, false)),
        );
        let mut seen: HashSet<String> = HashSet::new();
        while self.fetches < MAX_FILES && self.failed_in_row < MAX_FAILURES_IN_ROW && !self.visitor.pause.load(Ordering::Relaxed) {
            let Some((url, depth, parent, index_date, force)) = queue.pop_front() else {
                break;
            };
            if seen.contains(&url) || (self.found && self.guesses.contains(&url)) {
                continue;
            }
            seen.insert(url.clone());
            if !self.robots_allow(&url) {
                continue;
            }
            let stored = known.sitemaps.get(&url);
            if stored.is_some_and(|s| !force && !due(s, self.now)) {
                continue;
            }
            let children = self.sitemap(url, depth, parent, index_date, stored).await?;
            queue.extend(children);
        }
        self.result.cut_short = self.failed_in_row >= MAX_FAILURES_IN_ROW;
        for (url, depth, parent, _, _) in queue {
            if self.result.deferred.len() >= MAX_DEFERRED {
                break;
            }
            if !seen.contains(&url) && !known.sitemaps.contains_key(&url) && !self.guesses.contains(&url) {
                seen.insert(url.clone());
                self.result.deferred.push((url, depth, parent));
            }
        }
        Ok(())
    }

    /// Whether robots.txt lets us fetch this URL; the file we read covers the domain and its www host only.
    fn robots_allow(&self, url: &str) -> bool {
        let Ok(parts) = urls::split(url) else {
            return true;
        };
        let host = parts.hostname().unwrap_or_default();
        if host.strip_prefix("www.").unwrap_or(&host) != self.known.host || parts.port().ok().flatten().is_some_and(|port| port != 80 && port != 443) {
            return true;
        }
        let target = match &parts.query {
            Some(query) => format!("{}?{query}", parts.path),
            None => parts.path.clone(),
        };
        self.result.robots_paths.allows(&urls::tidy_target(if target.is_empty() { "/" } else { &target }))
    }

    /// Read one sitemap file and return the child sitemaps worth reading next.
    async fn sitemap(
        &mut self,
        url: String,
        depth: i64,
        parent: Option<i64>,
        index_date: Option<String>,
        stored: Option<&KnownSitemap>,
    ) -> Result<Vec<Child>, Crash> {
        self.fetches += 1;
        let validators = stored.map(|s| (s.etag.clone(), s.last_modified.clone()));
        let budget = Some(self.visitor.bodies.clone());
        let (answer, error) = match self.get(&url, MAX_SITEMAP_BYTES, validators, budget).await {
            Ok(answer) => (Some(answer), None),
            Err(error) => {
                self.network_error = Some(error.clone());
                (None, Some(error))
            }
        };
        let failing = answer.as_ref().is_none_or(|a| a.status >= 500 || a.status == 429);
        self.failed_in_row = if failing { self.failed_in_row + 1 } else { 0 };
        if self.guesses.contains(&url) && answer.as_ref().is_none_or(|a| a.status != 200) {
            return Ok(Vec::new());
        }
        let i = self.update(&url, depth, parent, index_date, stored);
        let Some(answer) = answer else {
            self.failed(i, error);
            return Ok(Vec::new());
        };
        match answer.status {
            304 => {
                self.unchanged(i, stored);
                return Ok(Vec::new());
            }
            404 | 410 => {
                let update = &mut self.result.sitemaps[i];
                update.status = "gone";
                update.next_check_at = Some(self.now + schedule::MAX_INTERVAL);
                return Ok(Vec::new());
            }
            200 if !answer.body.is_empty() => {}
            status => {
                self.failed(i, Some(format!("http_{status}")));
                return Ok(Vec::new());
            }
        }
        let update = &mut self.result.sitemaps[i];
        update.etag = header(&answer.headers, "etag");
        update.last_modified = header(&answer.headers, "last-modified");
        let standing = stored.map_or_else(churn::Standing::default, |s| churn::Standing {
            listed_before: s.url_count > 0,
            suspect_at: s.suspect_at,
            sketch: s.sketch.clone(),
            generated_until: s.generated_until,
        });
        let job = FileJob {
            _permit: answer.permit,
            body: answer.body,
            standing,
            stored_hash: stored.and_then(|s| s.content_hash.clone()),
            now: self.now,
            trust: update.trust,
            url: url.clone(),
            id: update.id,
            depth,
            known: self.known.clone(),
            robots: self.result.robots_paths.clone(),
            buckets: self.visitor.buckets.clone(),
        };
        let mut read = match self.visitor.cpu(move || digest(job)).await?? {
            Digested::Unchanged => {
                self.unchanged(i, stored);
                return Ok(Vec::new());
            }
            Digested::Invalid if stored.is_none() && self.guesses.contains(&url) => {
                // A guess answered with a page that isn't a sitemap is no root, so the next visit guesses again.
                self.result.sitemaps.remove(i);
                return Ok(Vec::new());
            }
            Digested::Invalid => {
                self.failed(i, Some("invalid".into()));
                return Ok(Vec::new());
            }
            Digested::Read(read) => read,
        };
        self.found = true;
        let news = read.news || url.to_lowercase().contains("news") || self.known.categories.iter().any(|c| c == "news");
        let listed_news = self.listed_news();
        let update = &mut self.result.sitemaps[i];
        update.kind = Some(read.kind.to_string());
        update.content_hash = Some(read.hash);
        update.changed = true;
        update.trust = read.trust;
        update.google_news = read.news;
        let interval = match stored {
            Some(_) => schedule::next_interval(update.interval, true),
            None => schedule::first_interval(read.changefreq.as_deref(), news),
        };
        update.interval = schedule::reread_within(interval, listed_news, update.depth, read.kind == "index", read.news);
        update.next_check_at = Some(self.now + update.interval);
        match read.verdict.take() {
            Some(churn::Verdict::Suspect { sketch }) => {
                update.suspect_at = Some(self.now);
                update.sketch = sketch;
                update.next_check_at = Some(self.now + churn::CONFIRM_AFTER);
                // Held back, so the next read must not pass as unchanged.
                update.content_hash = stored.and_then(|s| s.content_hash.clone());
                update.etag = stored.and_then(|s| s.etag.clone());
                update.last_modified = stored.and_then(|s| s.last_modified.clone());
            }
            Some(churn::Verdict::Generated { until }) => {
                if update.generated_until.is_none_or(|was| was <= self.now) {
                    println!("[rs] {url} lists new URLs on every read; none of them is taken for a week");
                }
                update.generated_until = Some(until);
                update.suspect_at = None;
                update.sketch.clear();
            }
            Some(churn::Verdict::Clear) => {
                update.suspect_at = None;
                update.sketch.clear();
                update.generated_until = None;
            }
            None => {}
        }
        if read.kind == "index" {
            update.child_count = read.entries as i64;
            return Ok(read.children);
        }
        update.url_count = read.listing.listed as i64;
        self.result.listed += read.listing.listed as i64;
        self.result.new += read.listing.new as i64;
        self.result.moved += read.listing.moved as i64;
        Ok(Vec::new())
    }

    fn update(&mut self, url: &str, depth: i64, parent: Option<i64>, index_date: Option<String>, stored: Option<&KnownSitemap>) -> usize {
        let update = match stored {
            Some(stored) => SitemapUpdate {
                id: stored.id,
                url: url.to_string(),
                kind: stored.kind.clone(),
                depth,
                parent_id: parent,
                status: "ok",
                error: None,
                etag: stored.etag.clone(),
                last_modified: stored.last_modified.clone(),
                content_hash: stored.content_hash.clone(),
                changed: false,
                url_count: 0,
                child_count: 0,
                interval: stored.interval,
                next_check_at: None,
                trust: stored.trust,
                index_lastmod: index_date.or_else(|| stored.index_lastmod.clone()),
                suspect_at: stored.suspect_at,
                sketch: stored.sketch.clone(),
                generated_until: stored.generated_until,
                google_news: stored.google_news,
            },
            None => SitemapUpdate {
                id: sitemap_id(url),
                url: url.to_string(),
                kind: None,
                depth,
                parent_id: parent,
                status: "ok",
                error: None,
                etag: None,
                last_modified: None,
                content_hash: None,
                changed: false,
                url_count: 0,
                child_count: 0,
                interval: schedule::DEFAULT_INTERVAL,
                next_check_at: None,
                trust: Trust::Unknown,
                index_lastmod: index_date,
                suspect_at: None,
                sketch: Vec::new(),
                generated_until: None,
                google_news: false,
            },
        };
        self.result.sitemaps.push(update);
        self.result.sitemaps.len() - 1
    }

    fn unchanged(&mut self, i: usize, stored: Option<&KnownSitemap>) {
        self.found = true;
        self.kept += stored.map_or(0, |s| s.url_count);
        let listed_news = self.listed_news();
        let update = &mut self.result.sitemaps[i];
        let index = update.kind.as_deref() == Some("index");
        update.interval = schedule::reread_within(schedule::next_interval(update.interval, false), listed_news, update.depth, index, update.google_news);
        update.next_check_at = Some(self.now + update.interval);
    }

    fn listed_news(&self) -> bool {
        self.visitor.buckets.terms(&self.known.host).news
    }

    fn failed(&mut self, i: usize, error: Option<String>) {
        let update = &mut self.result.sitemaps[i];
        update.status = "error";
        update.error = error;
        update.next_check_at = Some(self.now + update.interval);
    }

    fn settle(&mut self) {
        if self.result.outcome != Outcome::Sitemap {
            return;
        }
        let updates = &self.result.sitemaps;
        if self.known.state == State::Active {
            let mut roots = updates.iter().filter(|u| u.depth == 0).peekable();
            if self.result.requests > 0 && !self.answered {
                self.stop(Outcome::Unreachable, self.network_error.clone());
            } else if roots.peek().is_some() && !self.found && roots.all(|u| u.status == "gone") {
                self.stop(Outcome::NoSitemap, Some("sitemap_gone".into()));
            }
            return;
        }
        if !self.found {
            self.stop(Outcome::NoSitemap, Some("no_sitemap".into()));
        } else if self.result.listed + self.kept < MIN_URLS && !updates.iter().any(|u| u.kind.as_deref() == Some("index")) {
            self.stop(Outcome::NoSitemap, Some("sitemap_too_small".into()));
        }
    }

    fn needs_language(&self) -> bool {
        self.known.language.is_none() || self.known.state == State::Ineligible
    }

    async fn language(&mut self) -> Result<(), Crash> {
        let Some(answer) = self.first_answer("/", MAX_HOMEPAGE_BYTES).await else {
            return Ok(());
        };
        if self.moved_away(&answer) {
            return Ok(());
        }
        if answer.status >= 500 {
            self.stop(Outcome::Unreachable, Some(format!("homepage_{}", answer.status)));
            return Ok(());
        }
        if answer.status != 200 {
            self.stop(Outcome::Ineligible, Some("homepage_error".into()));
            return Ok(());
        }
        let body = answer.body;
        let detect = self.visitor.language.clone();
        let page = self.visitor.cpu(move || homepage::read(&body, &*detect)).await?;
        self.result.home_chars = Some(page.chars as i64);
        self.result.declared_lang = page.declared;
        self.result.language = page.language;
        if let Some(problem) = page.problem {
            self.stop(Outcome::Ineligible, Some(problem.into()));
        }
        Ok(())
    }

    /// Try https, then http; stop at a DNS failure since the other scheme will not help.
    async fn first_answer(&mut self, path: &str, limit: usize) -> Option<Answer> {
        let mut last = "unknown".to_string();
        for scheme in ["https", "http"] {
            match self.get(&format!("{scheme}://{}{path}", self.known.host), limit, None, None).await {
                Ok(answer) => return Some(answer),
                Err(error) => last = error,
            }
            if last.contains("DNS") {
                break;
            }
        }
        self.stop(Outcome::Unreachable, Some(last));
        None
    }

    fn moved_away(&mut self, answer: &Answer) -> bool {
        let host = urls::split(&answer.url).ok().and_then(|parts| parts.hostname()).unwrap_or_default();
        let target = self.visitor.suffixes.registrable(&host);
        let home = self.visitor.suffixes.registrable(&self.known.host);
        // A subdomain kept as a domain of its own stays itself only on its own host; a registrable domain anywhere under its name.
        let target = if home.as_deref() == Some(self.known.host.as_str()) || host.is_empty() {
            target.filter(|target| *target != self.known.host)
        } else {
            let bare = host.strip_prefix("www.").unwrap_or(&host).to_string();
            (bare != self.known.host).then(|| if target == home { bare } else { target.unwrap_or(bare) })
        };
        let Some(target) = target else {
            return false;
        };
        self.result.canonical_host = Some(target);
        self.stop(Outcome::Redirect, Some("redirect".into()));
        true
    }

    /// Follow redirects by hand, pacing and signing each hop for the host it goes to.
    async fn get(
        &mut self,
        url: &str,
        limit: usize,
        validators: Option<(Option<String>, Option<String>)>,
        budget: Option<Slots>,
    ) -> Result<Answer, String> {
        let mut url = url.to_string();
        let mut permit = None;
        for _ in 0..=MAX_REDIRECTS {
            let host = urls::split(&url).map_err(|_| "ValueError".to_string())?.hostname().unwrap_or_default();
            if !net::public_host(&host) {
                return Err("OSError".into());
            }
            self.pacer.wait().await;
            if let (None, Some(budget)) = (&permit, &budget) {
                permit = Some(budget.take().await.map_err(|crash| crash.0)?);
            }
            self.result.requests += 1;
            let mut request = self.visitor.client.get(&url);
            if let Some(signer) = &self.visitor.signer {
                for (name, value) in signer.headers(&url, chrono::Utc::now().timestamp()) {
                    request = request.header(name, value);
                }
            }
            if let Some((etag, modified)) = &validators {
                if let Some(etag) = etag {
                    request = request.header("If-None-Match", etag);
                }
                if let Some(modified) = modified {
                    request = request.header("If-Modified-Since", modified);
                }
            }
            let started = Instant::now();
            let connecting = Some(self.visitor.connect_timeout);
            let mut response = request.send().await.map_err(|e| net::failure(&e, started.elapsed(), connecting).to_string())?;
            self.answered = true;
            let waited = started.elapsed().as_secs_f64();
            let status = response.status().as_u16();
            if status == 429 || status == 503 {
                let asked = retry_after(response.headers());
                self.pacer.slow_to(Some((self.pacer.interval * 2.0).max(asked)));
            }
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                if let Some(location) = header(response.headers(), "location").filter(|l| !l.is_empty()) {
                    url = urls::join(&url, &location).map_err(|_| "ValueError".to_string())?;
                    continue;
                }
            }
            let body = if status == 200 {
                net::read_body(&mut response, limit).await.map_err(|e| net::failure(&e, started.elapsed(), None).to_string())?
            } else {
                Vec::new()
            };
            // Slow to answer means busy: rest that long before the next request.
            self.pacer.rest(waited);
            return Ok(Answer { status, body, headers: response.headers().clone(), url, permit });
        }
        Err("TooManyRedirects".into())
    }

    fn stop(&mut self, outcome: Outcome, reason: Option<String>) {
        self.result.outcome = outcome;
        self.result.reason = reason;
    }
}

struct FileJob {
    standing: churn::Standing,
    _permit: Option<OwnedSemaphorePermit>,
    body: Vec<u8>,
    stored_hash: Option<String>,
    now: i64,
    trust: Trust,
    url: String,
    id: i64,
    depth: i64,
    known: Arc<Known>,
    robots: Paths,
    buckets: Arc<Buckets>,
}

struct Read {
    hash: String,
    kind: &'static str,
    trust: Trust,
    news: bool,
    changefreq: Option<String>,
    entries: usize,
    children: Vec<Child>,
    listing: Listing,
    verdict: Option<churn::Verdict>,
}

enum Digested {
    Unchanged,
    Invalid,
    Read(Read),
}

/// Everything CPU-bound about one sitemap file: decompress, hash, parse, date, and store what it lists.
fn digest(job: FileJob) -> Result<Digested, Crash> {
    let body = gunzip(&job.body);
    let hash: String = Sha256::digest(&body).iter().take(16).map(|b| format!("{b:02x}")).collect();
    if job.stored_hash.as_deref() == Some(hash.as_str()) {
        return Ok(Digested::Unchanged);
    }
    let (kind, entries) = sitemaps::parse_entries(&body);
    if kind == "invalid" {
        return Ok(Digested::Invalid);
    }
    let dates: Vec<Option<i64>> = entries
        .iter()
        .map(|e| schedule::plausible(isodate::parse_lastmod(e.lastmod.as_deref().or(e.published.as_deref())), job.now))
        .collect();
    let trust = schedule::assess_dates(&dates, job.now, job.trust);
    let mut read = Read {
        hash,
        kind,
        trust,
        news: sitemaps::is_news(&body),
        changefreq: dominant(entries.iter().filter_map(|e| e.changefreq.as_deref())),
        entries: entries.len(),
        children: Vec::new(),
        listing: Listing::default(),
        verdict: None,
    };
    if kind == "index" {
        if job.depth < MAX_DEPTH {
            read.children = children(&job, &entries, trust)?;
        }
        return Ok(Digested::Read(read));
    }
    let rows = match Normaliser::new(&job.known.host) {
        Ok(normaliser) => entries
            .iter()
            .zip(&dates)
            .filter_map(|(entry, date)| {
                let url = normaliser.parse(&entry.url)?;
                let rest = url.key.splitn(2, |&b| b == 0).nth(1)?;
                if !job.robots.allows_page(&job.known.host, rest) {
                    return None;
                }
                let timed = sitemaps::has_time(entry.lastmod.as_deref().or(entry.published.as_deref()));
                Some((url, epoch(*date) as u32, timed))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    let store = job.buckets.store(&job.known.host);
    let mut hashed: Vec<(u64, &[u8])> = rows.iter().map(|(url, _, _)| (fingerprint(&url.key, 0), url.key.as_slice())).collect();
    hashed.sort_unstable();
    hashed.dedup_by_key(|(hash, _)| *hash);
    hashed.truncate(churn::SKETCH);
    let sketch: Vec<u64> = hashed.iter().map(|(hash, _)| *hash).collect();
    let new_share = || store.new_share(hashed.iter().map(|(_, key)| *key)).unwrap_or(0.0);
    let verdict = churn::judge(&job.standing, rows.len(), sketch, new_share, job.now);
    read.listing = match verdict {
        churn::Verdict::Clear => store.record_listing(&job.known.host, job.id, rows, epoch(Some(job.now)) as u32).map_err(|_| Crash::new("StoreError"))?,
        churn::Verdict::Suspect { .. } | churn::Verdict::Generated { .. } => Listing { listed: rows.len(), ..Listing::default() },
    };
    read.verdict = Some(verdict);
    Ok(Digested::Read(read))
}

fn children(job: &FileJob, entries: &[Entry], trust: Trust) -> Result<Vec<Child>, Crash> {
    let trusted = schedule::relies_on_dates(trust);
    let mut children = Vec::with_capacity(entries.len());
    for entry in entries {
        let url = urls::join(&job.url, &entry.url).map_err(|_| Crash::new("ValueError"))?;
        let stored = job.known.sitemaps.get(&url);
        if let Some(stored) = stored {
            if trusted && entry.lastmod.is_some() && entry.lastmod == stored.index_lastmod {
                continue;
            }
        }
        let force = stored.is_none_or(|stored| trusted && entry.lastmod != stored.index_lastmod);
        children.push((url, job.depth + 1, Some(job.id), entry.lastmod.clone(), force));
    }
    Ok(children)
}

fn due(stored: &KnownSitemap, now: i64) -> bool {
    stored.next_check_at.is_none_or(|next| next <= now)
}

/// The most frequent value, the first seen winning a tie.
fn dominant<'a>(values: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut counts: IndexMap<&str, usize> = IndexMap::new();
    for value in values.filter(|v| !v.is_empty()) {
        *counts.entry(value).or_default() += 1;
    }
    let best = counts.values().copied().max()?;
    counts.into_iter().find(|&(_, count)| count == best).map(|(value, _)| value.to_string())
}

/// Unix seconds, as Python's `int(date.timestamp())`; 0 for no date.
pub fn epoch(micros: Option<i64>) -> i64 {
    micros.map_or(0, |us| us.div_euclid(SECOND))
}

/// Where to look for a sitemap no robots.txt names: the usual files, then on a listed domain its feed.
fn guess_paths(listed: bool) -> impl Iterator<Item = &'static str> {
    SITEMAP_GUESSES.into_iter().chain(FEED_GUESSES.into_iter().filter(move |_| listed))
}

fn crawl_delay(value: Option<f64>) -> f64 {
    match value {
        Some(delay) if delay.is_finite() && delay >= 0.0 => delay.min(MAX_CRAWL_DELAY),
        _ => 0.0,
    }
}

fn retry_after(headers: &HeaderMap) -> f64 {
    header(headers, "retry-after").and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0)
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

/// Decompress gzip up to the protocol's size limit, keeping whatever decoded before a cut.
pub fn gunzip(body: &[u8]) -> Cow<'_, [u8]> {
    if !body.starts_with(&[0x1f, 0x8b]) {
        return Cow::Borrowed(body);
    }
    let mut inflater = Decompress::new_gzip(15);
    let mut out: Vec<u8> = Vec::new();
    while out.len() < MAX_SITEMAP_BYTES {
        out.reserve((MAX_SITEMAP_BYTES - out.len()).min(1 << 20));
        let (read, written) = (inflater.total_in(), inflater.total_out());
        match inflater.decompress_vec(&body[read as usize..], &mut out, FlushDecompress::None) {
            Err(_) => return Cow::Borrowed(body),
            Ok(Status::StreamEnd) => break,
            Ok(_) if inflater.total_in() == read && inflater.total_out() == written => break,
            Ok(_) => {}
        }
    }
    out.truncate(MAX_SITEMAP_BYTES);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_listed_domain_with_no_sitemap_named_is_also_looked_at_for_a_feed() {
        assert_eq!(guess_paths(false).collect::<Vec<_>>(), ["/sitemap.xml", "/sitemap_index.xml"]);
        assert_eq!(guess_paths(true).collect::<Vec<_>>(), ["/sitemap.xml", "/sitemap_index.xml", "/feed", "/rss.xml"], "after the sitemaps, which go first");
    }

    #[test]
    fn gunzip_keeps_what_decoded_before_a_cut() {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&b"<url><loc>https://a.com/x</loc></url>".repeat(1000)).unwrap();
        let packed = encoder.finish().unwrap();
        assert_eq!(gunzip(&packed).len(), 37_000);
        let cut = gunzip(&packed[..packed.len() / 2]);
        assert!(!cut.is_empty() && cut.len() < 37_000);
        assert_eq!(&gunzip(b"\x1f\x8bnot gzip")[..], b"\x1f\x8bnot gzip");
        assert_eq!(dominant(["daily", "weekly", "weekly", "daily"].into_iter()).as_deref(), Some("daily"));
    }
}

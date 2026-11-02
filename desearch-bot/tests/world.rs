//! The crawl loop end to end against a small fake web.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use desearch_bot::allowed::{Allowed, Terms};
use desearch_bot::buckets::{bucket_of, sitemap_id, Buckets, Changes, Resources};
use desearch_bot::crawl::Loop;
use desearch_bot::records;
use desearch_bot::registry::Registry;
use desearch_bot::states::State;
use desearch_bot::suffixes::PublicSuffixList;
use desearch_bot::urls;
use desearch_bot::visit::{FollowLinks, Slots, Visitor};
use hickory_resolver::error::ResolveError;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const ENGLISH: &str = "The quick brown fox jumps over the lazy dog while the committee reviews the annual budget \
    report. Readers who follow the local news will find stories about schools, transport, weather and the people \
    who make this town a better place to live and work every single day of the year.";

#[derive(Clone)]
struct Page {
    status: u16,
    etag: Option<&'static str>,
    body: Vec<u8>,
    headers: Vec<(&'static str, String)>,
    /// Served instead when the request carries this `name=value` cookie.
    with_cookie: Option<(&'static str, Box<Page>)>,
}

#[derive(Default)]
struct Web {
    pages: HashMap<(String, String), Page>,
    hits: Vec<(String, String, Option<String>)>,
    /// Served instead once the path was asked this many times.
    later: HashMap<(String, String), (usize, Page)>,
    /// Asked for, and never answered.
    silent: HashSet<(String, String)>,
}

impl Web {
    fn add(&mut self, host: &str, path: &str, status: u16, etag: Option<&'static str>, body: &[u8]) {
        self.pages.insert((host.into(), path.into()), Page { status, etag, body: body.to_vec(), headers: Vec::new(), with_cookie: None });
    }

    fn redirect(&mut self, host: &str, path: &str, to: &str, set_cookie: Option<&str>) {
        let mut headers = vec![("Location", to.to_string())];
        headers.extend(set_cookie.map(|cookie| ("Set-Cookie", cookie.to_string())));
        self.pages.insert((host.into(), path.into()), Page { status: 302, etag: None, body: Vec::new(), headers, with_cookie: None });
    }

    fn with_cookie(&mut self, host: &str, path: &str, cookie: &'static str, page: Page) {
        if let Some(found) = self.pages.get_mut(&(host.to_string(), path.to_string())) {
            found.with_cookie = Some((cookie, Box::new(page)));
        }
    }
}

type Shared = Arc<Mutex<Web>>;

/// Plain HTTP/1.1 with keep-alive; anything else, such as a TLS handshake, is hung up on.
async fn serve(listener: TcpListener, web: Shared) {
    while let Ok((mut socket, _)) = listener.accept().await {
        let web = web.clone();
        tokio::spawn(async move {
            let mut buffer = Vec::new();
            loop {
                let end = loop {
                    if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break end;
                    }
                    if buffer.len() >= 4 && !buffer.starts_with(b"GET ") {
                        return;
                    }
                    let mut chunk = [0u8; 4096];
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                };
                let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
                buffer.drain(..end + 4);
                let mut lines = head.split("\r\n");
                let Some(path) = lines.next().and_then(|l| l.strip_prefix("GET ")).and_then(|r| r.split(' ').next()) else {
                    return;
                };
                let headers: HashMap<String, String> =
                    lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().into())).collect();
                let host = headers.get("host").map_or("", |h| h.split(':').next().unwrap_or("")).to_string();
                let validator = headers.get("if-none-match").cloned();
                let (page, silent) = {
                    let mut web = web.lock().unwrap();
                    let asked = web.hits.iter().filter(|(h, p, _)| *h == host && p == path).count();
                    web.hits.push((host.clone(), path.to_string(), validator.clone()));
                    let key = (host, path.to_string());
                    let silent = web.silent.contains(&key);
                    (web.later.get(&key).filter(|(after, _)| asked >= *after).map(|(_, page)| page.clone()).or_else(|| web.pages.get(&key).cloned()), silent)
                };
                if silent {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                let page = page.unwrap_or(Page { status: 404, etag: None, body: b"not found".to_vec(), headers: Vec::new(), with_cookie: None });
                let sent = headers.get("cookie").cloned().unwrap_or_default();
                let page = match &page.with_cookie {
                    Some((cookie, other)) if sent.split("; ").any(|pair| pair == *cookie) => (**other).clone(),
                    _ => page,
                };
                let (status, body) =
                    if page.etag.is_some() && validator.as_deref() == page.etag { (304, Vec::new()) } else { (page.status, page.body) };
                let mut response = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\n", body.len());
                if let Some(etag) = page.etag {
                    response.push_str(&format!("ETag: {etag}\r\n"));
                }
                for (name, value) in &page.headers {
                    response.push_str(&format!("{name}: {value}\r\n"));
                }
                response.push_str("\r\n");
                let mut bytes = response.into_bytes();
                bytes.extend_from_slice(&body);
                if socket.write_all(&bytes).await.is_err() {
                    return;
                }
            }
        });
    }
}

struct TestResolver {
    addr: SocketAddr,
    known: HashSet<&'static str>,
}

impl Resolve for TestResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let found = self.known.contains(name.as_str()).then_some(self.addr);
        Box::pin(async move {
            match found {
                Some(addr) => Ok(Box::new(std::iter::once(addr)) as Addrs),
                None => Err(Box::new(ResolveError::from("no such host")) as Box<dyn std::error::Error + Send + Sync>),
            }
        })
    }
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn urlset(host: &str, paths: &[String]) -> String {
    let entries: String = paths
        .iter()
        .enumerate()
        .map(|(i, path)| format!("<url><loc>http://{host}{path}</loc><lastmod>2024-01-{:02}T10:00:00Z</lastmod></url>", i % 28 + 1))
        .collect();
    format!("<urlset>{entries}</urlset>")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crawls_a_small_web() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    {
        let mut w = web.lock().unwrap();
        w.add("site.test", "/robots.txt", 200, None, b"User-agent: *\nAllow: /\nDisallow: /private/\nSitemap: http://site.test/sitemap_index.xml\n");
        w.add(
            "site.test",
            "/sitemap_index.xml",
            200,
            None,
            b"<sitemapindex><sitemap><loc>http://site.test/posts.xml.gz</loc><lastmod>2024-01-01</lastmod></sitemap>\
              <sitemap><loc>/pages.xml</loc></sitemap><sitemap><loc>/private/more.xml</loc></sitemap></sitemapindex>",
        );
        let posts: Vec<String> = (1..=12).map(|i| format!("/post/{i}")).collect();
        w.add("site.test", "/posts.xml.gz", 200, None, &gzip(urlset("site.test", &posts).as_bytes()));
        let pages = urlset("site.test", &["/".into(), "/about".into(), "/contact".into(), "/private/b".into()])
            .replace("</urlset>", "<url><loc>http://elsewhere.test/x</loc></url></urlset>");
        w.add("site.test", "/pages.xml", 200, Some("\"v1\""), pages.as_bytes());
        w.add("site.test", "/private/more.xml", 200, None, urlset("site.test", &["/private/a".into()]).as_bytes());
        w.add("site.test", "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p></body></html>").as_bytes());
        w.add("blocked.test", "/robots.txt", 200, None, b"User-agent: *\nDisallow: /\n");
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-world-{}", std::process::id()));
    let hosts = ["site.test", "blocked.test", "empty.test", "gone.test"];
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let resources = Resources::new(8 << 20, 8 << 20, owned.len());
    let buckets = Arc::new(Buckets::open(&root, &owned, &resources).unwrap());
    for host in hosts {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
        buckets.store(host).write(changes).unwrap();
    }
    let resolver = TestResolver { addr, known: ["site.test", "blocked.test", "empty.test"].into() };
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(resolver))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 4);
    assert_eq!(crawl.step().await.unwrap(), 4);

    let domain = |host: &str| buckets.store(host).domain(host).unwrap().unwrap();
    let site = domain("site.test");
    assert_eq!((site["state"].as_str(), site["urls"].as_i64(), site["lang"].as_str()), (Some("active"), Some(15), Some("en")));
    assert_eq!(buckets.store("site.test").sitemaps("site.test").unwrap().len(), 3);
    assert_eq!(site["rules"]["disallow"], serde_json::json!(["/private/"]));
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path.starts_with("/private/")));
    let stored = |url: &str| buckets.store("site.test").url(&urls::parse(url, "site.test").unwrap()).unwrap();
    assert!(stored("http://site.test/post/12").is_some() && stored("http://site.test/about").is_some());
    assert!(stored("http://site.test/private/b").is_none(), "a page robots.txt disallows is never kept");
    assert_eq!(domain("blocked.test")["reason"], "robots_disallow");
    assert_eq!((domain("empty.test")["state"].as_str(), domain("empty.test")["reason"].as_str()), (Some("no_sitemap"), Some("no_sitemap")));
    assert_eq!(domain("gone.test")["reason"], "ClientConnectorDNSError");
    assert!(!web.lock().unwrap().hits.iter().any(|(host, _, _)| host.starts_with("www.")), "a bare host that answers never sends the bot to www");
    assert!(domain("site.test").get("www").is_none());

    let store = buckets.store("site.test");
    let mut changes = Changes::default();
    let mut record = domain("site.test");
    record.insert("due".into(), 0.into());
    changes.domain("site.test", &record);
    for (url, mut sitemap) in store.sitemaps("site.test").unwrap() {
        sitemap.insert("next".into(), 0.into());
        changes.sitemap("site.test", &url, &sitemap);
    }
    store.write(changes).unwrap();
    web.lock().unwrap().hits.clear();
    crawl.load().unwrap();
    assert_eq!(crawl.step().await.unwrap(), 1);
    let hits = web.lock().unwrap().hits.clone();
    assert!(hits.iter().any(|(_, path, validator)| path == "/pages.xml" && validator.as_deref() == Some("\"v1\"")));
    assert!(!hits.iter().any(|(_, path, _)| path == "/robots.txt" || path == "/"));
    let site = domain("site.test");
    assert_eq!((site["state"].as_str(), site["urls"].as_i64()), (Some("active"), Some(15)));
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sitemap_listing_new_urls_on_every_read_is_held_back_while_one_that_settles_is_recorded() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let listing = |host: &str, batch: &str| urlset(host, &(0..600).map(|i| format!("/{batch}/{i}")).collect::<Vec<_>>());
    {
        let mut w = web.lock().unwrap();
        for host in ["hacked.test", "busy.test"] {
            w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nAllow: /\nSitemap: http://{host}/list.xml\n").as_bytes());
            w.add(host, "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p></body></html>").as_bytes());
            w.add(host, "/list.xml", 200, None, listing(host, "first").as_bytes());
        }
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-churn-{}", std::process::id()));
    let hosts = ["hacked.test", "busy.test"];
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let resources = Resources::new(8 << 20, 8 << 20, owned.len());
    let buckets = Arc::new(Buckets::open(&root, &owned, &resources).unwrap());
    for host in hosts {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
        buckets.store(host).write(changes).unwrap();
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: hosts.into() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    let stored = |host: &str, path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    let sitemap = |host: &str| buckets.store(host).sitemaps(host).unwrap().into_iter().next().unwrap().1;
    let read_again = |crawl: &mut Loop| {
        for host in hosts {
            let store = buckets.store(host);
            let mut changes = Changes::default();
            let mut record = store.domain(host).unwrap().unwrap();
            record.insert("due".into(), 0.into());
            changes.domain(host, &record);
            for (url, mut listed) in store.sitemaps(host).unwrap() {
                listed.insert("next".into(), 0.into());
                changes.sitemap(host, &url, &listed);
            }
            store.write(changes).unwrap();
        }
        crawl.load().unwrap();
    };

    assert_eq!(crawl.load().unwrap(), 2);
    assert_eq!(crawl.step().await.unwrap(), 2);
    assert!(stored("hacked.test", "/first/0") && stored("busy.test", "/first/0"), "a first read is taken as it comes");

    {
        let mut w = web.lock().unwrap();
        w.add("hacked.test", "/list.xml", 200, None, listing("hacked.test", "spam-a").as_bytes());
        w.add("busy.test", "/list.xml", 200, None, listing("busy.test", "latest").as_bytes());
    }
    read_again(&mut crawl);
    assert_eq!(crawl.step().await.unwrap(), 2);
    for host in hosts {
        let held = sitemap(host);
        assert!(held["suspect"].as_i64().is_some() && held.contains_key("sketch"), "{host}: a re-read of nothing but new URLs is suspect");
        assert!(!stored(host, "/spam-a/0") && !stored(host, "/latest/0"), "{host}: and its URLs wait for the second read");
    }

    web.lock().unwrap().add("hacked.test", "/list.xml", 200, None, listing("hacked.test", "spam-b").as_bytes());
    read_again(&mut crawl);
    assert_eq!(crawl.step().await.unwrap(), 2);
    assert!(stored("busy.test", "/latest/599") && !sitemap("busy.test").contains_key("suspect"), "the same list again: a real one, recorded now");
    let hacked = sitemap("hacked.test");
    assert!(hacked["generated"].as_i64().is_some() && !hacked.contains_key("suspect"), "new URLs again: generated");
    assert!(!stored("hacked.test", "/spam-a/0") && !stored("hacked.test", "/spam-b/0"));

    web.lock().unwrap().add("hacked.test", "/list.xml", 200, None, listing("hacked.test", "spam-c").as_bytes());
    read_again(&mut crawl);
    crawl.step().await.unwrap();
    assert!(!stored("hacked.test", "/spam-c/0"), "a generated sitemap's URLs stay out for the week");
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subdomain_kept_as_a_domain_of_its_own_is_crawled_on_its_own_host_with_its_own_robots_txt() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let host = "news.site.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nDisallow: /drafts/\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p></body></html>").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..600).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add("site.test", "/robots.txt", 200, None, b"User-agent: *\nDisallow: /\n");
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-subdomain-{}", std::process::id()));
    let owned = vec![bucket_of(host)];
    let buckets = Arc::new(Buckets::open(&root, &owned, &Resources::new(8 << 20, 8 << 20, 1)).unwrap());
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
    buckets.store(host).write(changes).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: [host, "site.test"].into() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 1);
    assert_eq!(crawl.step().await.unwrap(), 1);
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["canonical"].as_str()), (Some("active"), None), "its own host is no redirect to the parent");
    assert_eq!(record["rules"]["disallow"], serde_json::json!(["/drafts/"]), "the rules are its own host's");
    let hits = web.lock().unwrap().hits.clone();
    assert!(hits.iter().any(|(h, path, _)| h == host && path == "/robots.txt") && !hits.iter().any(|(h, _, _)| h == "site.test"));
    assert!(buckets.store(host).url(&urls::parse(&format!("http://{host}/story/599"), host).unwrap()).unwrap().is_some());
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_homepage_that_answers_with_an_error_keeps_its_status_on_the_record() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let hosts: [(&str, u16); 6] = [("missing.test", 404), ("gone-home.test", 410), ("closed.test", 401), ("denied.test", 403), ("busy.test", 429), ("known.test", 403)];
    {
        let mut w = web.lock().unwrap();
        for (host, status) in hosts {
            w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
            w.add(host, "/list.xml", 200, None, urlset(host, &(0..600).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
            w.add(host, "/", status, None, b"no");
        }
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-home-status-{}", std::process::id()));
    let mut owned: Vec<usize> = hosts.iter().map(|(h, _)| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&root, &owned, &Resources::new(8 << 20, 8 << 20, owned.len())).unwrap());
    for (host, _) in hosts {
        let mut changes = Changes::default();
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0));
        if host == "known.test" {
            record.insert("ok".into(), 1_700_000_000.into());
        }
        changes.domain(host, &record);
        buckets.store(host).write(changes).unwrap();
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: hosts.iter().map(|(h, _)| *h).collect() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), hosts.len());
    assert_eq!(crawl.step().await.unwrap(), hosts.len());
    for (host, status) in hosts {
        let record = buckets.store(host).domain(host).unwrap().unwrap();
        let state = if host == "known.test" { "new" } else { "ineligible" };
        assert_eq!((record["state"].as_str(), record["reason"].as_str()), (Some(state), Some("homepage_error")), "{host}: a refusal on a recheck of a site that answered before is tried again soon");
        assert_eq!(record["home_status"].as_u64(), Some(u64::from(status)), "{host}");
    }
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_feed_named_as_a_sitemap_is_read_as_one() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let host = "feed.test";
    {
        let mut w = web.lock().unwrap();
        let items: String = (1..=12).map(|i| format!("<item><link>http://{host}/2026/10/story-{i}</link><pubDate>Tue, 27 Oct 2026 10:{i:02}:00 GMT</pubDate></item>")).collect();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nAllow: /\nSitemap: http://{host}/feed\n").as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p></body></html>").as_bytes());
        w.add(host, "/feed", 200, None, format!("<?xml version=\"1.0\"?><rss version=\"2.0\"><channel>{items}</channel></rss>").as_bytes());
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-feed-{}", std::process::id()));
    let buckets = Arc::new(Buckets::open(&root, &[bucket_of(host)], &Resources::new(8 << 20, 8 << 20, 1)).unwrap());
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
    buckets.store(host).write(changes).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: [host].into() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 1);
    assert_eq!(crawl.step().await.unwrap(), 1);
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(12)), "the feed's items are the site's listed pages");
    let story = buckets.store(host).url(&urls::parse(&format!("http://{host}/2026/10/story-12"), host).unwrap()).unwrap().unwrap();
    assert_eq!(story.lastmod, 1_793_095_920, "dated by the item's pubDate");
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_index_entry_naming_several_sitemaps_reads_each_and_their_cdata_locations() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let host = "sports.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nAllow: /\nSitemap: http://{host}/sitemap.xml\n").as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p></body></html>").as_bytes());
        let index = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sitemapindex xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n    <sitemap>\n        \
             <loc>http://{host}/googlenewssitemap</loc>\n        <loc>http://{host}/sitemap/videos</loc>\n        <loc>http://{host}/where-to-watch.xml</loc>\n    \
             </sitemap> \n</sitemapindex>"
        );
        w.add(host, "/sitemap.xml", 200, None, index.as_bytes());
        let news: String = (1..=12)
            .map(|i| format!("\t<url>\n\t\t<loc><![CDATA[http://{host}/f1/story/_/id/{i}/title]]></loc>\n\t\t<n:news><n:publication_date>2026-11-02T09:{i:02}:00Z</n:publication_date></n:news>\n\t</url>\n"))
            .collect();
        let news = format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\" xmlns:n=\"http://www.google.com/schemas/sitemap-news/0.9\">{news}</urlset>");
        w.add(host, "/googlenewssitemap", 200, None, news.as_bytes());
        w.add(host, "/sitemap/videos", 200, None, urlset(host, &["/video/1".into(), "/video/2".into()]).as_bytes());
        w.add(host, "/where-to-watch.xml", 200, None, urlset(host, &["/where-to-watch".into()]).as_bytes());
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-several-{}", std::process::id()));
    let buckets = Arc::new(Buckets::open(&root, &[bucket_of(host)], &Resources::new(8 << 20, 8 << 20, 1)).unwrap());
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
    buckets.store(host).write(changes).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: [host].into() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Off,
        resolver: Some(Arc::new(TestResolver { addr, known: [host].into() })),
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 1);
    assert_eq!(crawl.step().await.unwrap(), 1);
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(15)), "every sitemap the entry names, and every page of the news sitemap");
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap();
    assert!(stored("/f1/story/_/id/12/title").is_some() && stored("/video/2").is_some() && stored("/where-to-watch").is_some());
    assert_eq!(stored("/f1/story/_/id/12/title").unwrap().lastmod, 1_793_610_720, "dated by its news date under the file's own prefix");
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_landing_site_lists_its_homepage_and_the_pages_it_links_to_beside_its_sitemap() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let web: Shared = Arc::default();
    let host = "landing.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nAllow: /\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        let links = "<a href=\"/about\">About</a><a href='/en-us/pricing'>Pricing</a><a href=team/>Team</a><a href=\"https://other.test/x\">Other</a>";
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><body><p>{ENGLISH}</p>{links}</body></html>").as_bytes());
    }
    tokio::spawn(serve(listener, web.clone()));

    let root = std::env::temp_dir().join(format!("desearch-bot-landing-{}", std::process::id()));
    let buckets = Arc::new(Buckets::open(&root, &[bucket_of(host)], &Resources::new(8 << 20, 8 << 20, 1)).unwrap());
    buckets.allow(&Allowed::with_terms([(host, Terms { send_cap: Some(100), depth: Some(1), ..Terms::default() })]));
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
    buckets.store(host).write(changes).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: [host].into() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 1);
    assert_eq!(crawl.step().await.unwrap(), 1);
    let store = buckets.store(host);
    let record = store.domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(16)), "the sitemap's 12 pages, the homepage and the 3 it links to on its own host");
    for page in ["/", "/about", "/en-us/pricing", "/team/"] {
        assert!(store.url(&urls::parse(&format!("http://{host}{page}"), host).unwrap()).unwrap().is_some(), "{page}");
    }
    assert!(web.lock().unwrap().hits.iter().any(|(h, path, _)| h == host && path == "/"), "the homepage was read");
    drop(crawl);
    std::fs::remove_dir_all(&root).ok();
}

async fn visit_once(web: Shared, hosts: &[&'static str], answered: &[&str], name: &str) -> Arc<Buckets> {
    visit_resolving(web, hosts, hosts, answered, name).await
}

async fn visit_resolving(web: Shared, hosts: &[&'static str], resolves: &[&'static str], answered: &[&str], name: &str) -> Arc<Buckets> {
    visit_seeded(web, hosts, resolves, answered, &[], name).await
}

async fn visit_seeded(web: Shared, hosts: &[&'static str], resolves: &[&'static str], answered: &[&str], www: &[&str], name: &str) -> Arc<Buckets> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, web));
    let root = std::env::temp_dir().join(format!("desearch-bot-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&root, &owned, &Resources::new(8 << 20, 8 << 20, owned.len())).unwrap());
    for host in hosts {
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0));
        if answered.contains(host) {
            record.insert("ok".into(), 1_700_000_000.into());
        }
        if www.contains(host) {
            record.insert("www".into(), true.into());
        }
        let mut changes = Changes::default();
        changes.domain(host, &record);
        buckets.store(host).write(changes).unwrap();
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: resolves.iter().copied().collect() }))
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links: Slots::new(2),
        follow_links: FollowLinks::Thin,
        resolver: None,
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), hosts.len());
    assert_eq!(crawl.step().await.unwrap(), hosts.len());
    drop(crawl);
    buckets
}

/// One crawl step over hosts seeded as new, after `prepare` has set up their list and stored records; the loop is kept for later steps.
async fn crawl_prepared(web: Shared, hosts: &[&'static str], name: &str, prepare: impl FnOnce(&Arc<Buckets>)) -> (Arc<Buckets>, Loop) {
    crawl_following(web, hosts, name, FollowLinks::Thin, Slots::new(2), prepare).await
}

async fn crawl_following(web: Shared, hosts: &[&'static str], name: &str, follow_links: FollowLinks, links: Slots, prepare: impl FnOnce(&Arc<Buckets>)) -> (Arc<Buckets>, Loop) {
    crawl_adjusted(web, hosts, &[], name, follow_links, links, prepare, |_| {}).await
}

/// How long a test request may take before it fails.
const TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// As `crawl_following`, with more hosts that resolve, such as a site's www host.
async fn crawl_resolving(web: Shared, hosts: &[&'static str], also: &[&'static str], name: &str, follow_links: FollowLinks, prepare: impl FnOnce(&Arc<Buckets>)) -> (Arc<Buckets>, Loop) {
    crawl_adjusted(web, hosts, also, name, follow_links, Slots::new(2), prepare, |_| {}).await
}

#[allow(clippy::too_many_arguments)]
async fn crawl_adjusted(
    web: Shared,
    hosts: &[&'static str],
    also: &[&'static str],
    name: &str,
    follow_links: FollowLinks,
    links: Slots,
    prepare: impl FnOnce(&Arc<Buckets>),
    adjust: impl FnOnce(&mut Visitor),
) -> (Arc<Buckets>, Loop) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, web));
    let root = std::env::temp_dir().join(format!("desearch-bot-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&root, &owned, &Resources::new(8 << 20, 8 << 20, owned.len())).unwrap());
    for host in hosts {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0)));
        buckets.store(host).write(changes).unwrap();
    }
    prepare(&buckets);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(TestResolver { addr, known: hosts.iter().chain(also).copied().collect() }))
        .timeout(if name.starts_with("slow-") { Duration::from_millis(800) } else { TEST_TIMEOUT })
        .build()
        .unwrap();
    let mut visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::parse("test\ngov.test\n")),
        signer: None,
        language: Arc::new(|text: &str| Some(if text.contains("the") { "en" } else { "fr" }.to_string())),
        floor: 0.0,
        connect_timeout: Duration::from_secs(10),
        cpu: Slots::new(4),
        bodies: Slots::new(2),
        pause: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heavy: Slots::new(1),
        links,
        follow_links,
        resolver: None,
    };
    adjust(&mut visitor);
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), hosts.len());
    assert_eq!(crawl.step().await.unwrap(), hosts.len());
    (buckets, crawl)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_whose_bare_host_has_no_dns_record_is_read_on_its_www_host() {
    let web: Shared = Arc::default();
    let (host, www) = ("wwwonly.test", "www.wwwonly.test");
    {
        let mut w = web.lock().unwrap();
        w.add(www, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{www}/list.xml\n").as_bytes());
        w.add(www, "/list.xml", 200, None, urlset(www, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(www, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let buckets = visit_resolving(web, &[host], &[www], &[], "wwwonly").await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["lang"].as_str(), record["urls"].as_i64()), (Some("active"), Some("en"), Some(12)));
    assert_eq!(record["www"], true, "the next visit starts on the www host");
    let stored = buckets.store(host).url(&urls::parse(&format!("http://{www}/story/3"), host).unwrap()).unwrap();
    assert!(stored.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bare_host_that_answers_again_clears_the_www_flag_and_is_read_first() {
    let web: Shared = Arc::default();
    let (host, www) = ("flip.test", "www.flip.test");
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let buckets = visit_seeded(web.clone(), &[host], &[host, www], &[], &[host], "flip").await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!(record["state"].as_str(), Some("active"));
    assert!(record.get("www").is_none());
    assert!(!web.lock().unwrap().hits.iter().any(|(h, _, _)| h == www), "the bare host is asked first");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sign_in_redirect_that_sets_a_cookie_reaches_the_homepage() {
    let web: Shared = Arc::default();
    let host = "cookie.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.redirect(host, "/", &format!("http://{host}/login"), Some("session=1; Path=/"));
        w.redirect(host, "/login", &format!("http://{host}/"), None);
        let home = Page { status: 200, etag: None, body: format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").into_bytes(), headers: Vec::new(), with_cookie: None };
        w.with_cookie(host, "/", "session=1", home);
    }
    let buckets = visit_once(web, &[host], &[], "cookie").await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["home_status"].as_u64()), (Some("active"), Some(200)), "without the cookie it loops until too many redirects");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_recheck_whose_sitemap_answers_503_is_tried_again() {
    let web: Shared = Arc::default();
    let hosts = ["busy.test", "fresh.test"];
    {
        let mut w = web.lock().unwrap();
        for host in hosts {
            w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
            w.add(host, "/list.xml", 503, None, b"busy");
        }
    }
    let buckets = visit_once(web, &hosts, &["busy.test"], "sitemap-503").await;
    let state = |host: &str| {
        let record = buckets.store(host).domain(host).unwrap().unwrap();
        (record["state"].as_str().unwrap_or_default().to_string(), record["reason"].as_str().unwrap_or_default().to_string())
    };
    assert_eq!(state("busy.test"), ("new".into(), "sitemap_503".into()), "a recheck of a site that answered before");
    assert_eq!(state("fresh.test"), ("no_sitemap".into(), "no_sitemap".into()), "a site never seen before");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_that_skips_robots_txt_guesses_where_the_homepage_was_last_read() {
    let web: Shared = Arc::default();
    let host = "plainhome.test";
    {
        let mut w = web.lock().unwrap();
        let links: String = (0..12).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
    }
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "home-scheme", |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for (name, at) in [("robots", now), ("checked", now), ("ok", now), ("lang", 0)] {
            record.insert(name.into(), if name == "lang" { "en".into() } else { at.into() });
        }
        let home = format!("http://{host}/");
        let mut changes = Changes::default();
        changes.domain(host, &record);
        changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "kind": "homepage", "depth": 0, "status": "ok", "urls": 13, "next": 0}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!(record["state"].as_str(), Some("active"), "guesses on https would fail three times before the homepage is reached");
    let hits = web.lock().unwrap().hits.clone();
    assert!(hits.iter().any(|(h, p, _)| h == host && p == "/") && !hits.iter().any(|(_, p, _)| p == "/robots.txt"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sitemap_is_guessed_where_robots_txt_answered() {
    let web: Shared = Arc::default();
    let host = "plain.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        w.add(host, "/sitemap.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let buckets = visit_once(web.clone(), &[host], &[], "guess-scheme").await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(12)), "robots.txt answered on http only, and so does the guess");
    assert!(web.lock().unwrap().hits.iter().any(|(h, path, _)| h == host && path == "/sitemap.xml"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stored_sitemap_that_is_gone_leaves_the_site_guessed_at() {
    let web: Shared = Arc::default();
    let host = "stale.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        w.add(host, "/sitemap.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "stale-guess", |buckets| {
        let old = format!("http://{host}/old-sitemap.xml");
        let mut changes = Changes::default();
        changes.sitemap(host, &old, &serde_json::json!({"id": sitemap_id(&old), "depth": 0, "status": "gone", "urls": 0, "next": 0}).as_object().unwrap().clone());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(12)), "the dead record is read, then the guesses");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_recheck_between_two_reads_of_a_working_sitemap_keeps_the_site_active() {
    let web: Shared = Arc::default();
    let host = "between.test";
    let list = format!("http://{host}/list.xml");
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: {list}\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let later = chrono::Utc::now().timestamp() + 86_400;
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "standing", |buckets| {
        let mut changes = Changes::default();
        let record = serde_json::json!({"id": sitemap_id(&list), "kind": "urlset", "depth": 0, "status": "ok", "urls": 12, "next": later});
        changes.sitemap(host, &list, record.as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["urls"].as_i64()), (Some("active"), Some(12)), "the sitemap is not due, and still lists its 12 pages");
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/list.xml"), "a sitemap not due is not read");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_found_without_sitemaps_before_the_list_named_it_is_due_again_once_it_does() {
    let web: Shared = Arc::default();
    let host = "joined.test";
    {
        let mut w = web.lock().unwrap();
        let links: String = (0..12).map(|i| format!("<a href=\"/paper/{i}\">Paper {i}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
    }
    let list = std::env::temp_dir().join(format!("desearch-bot-joined-list-{}.json", std::process::id()));
    std::fs::write(&list, b"[\"other.test\"]").unwrap();
    let allowed = Allowed::from_file(&list).unwrap();
    let (buckets, mut crawl) = crawl_prepared(web.clone(), &[host], "joined", |buckets| buckets.allow(&allowed)).await;
    let record = || buckets.store(host).domain(host).unwrap().unwrap();
    let now = chrono::Utc::now().timestamp();
    assert_eq!((record()["state"].as_str(), record()["listed"].as_bool()), (Some("no_sitemap"), Some(false)));
    assert!(record()["due"].as_i64().unwrap() > now + 86_400, "found without sitemaps, it waits for its monthly recheck");
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/"), "an unlisted site's homepage is no guess");

    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(&list, format!("[\"other.test\", \"{host}\"]")).unwrap();
    assert!(allowed.reload().unwrap());
    assert_eq!(crawl.step().await.unwrap(), 0);
    let due = record()["due"].as_i64().unwrap();
    assert!(due <= chrono::Utc::now().timestamp() + 2, "the list naming it makes it due within seconds, not {due}");

    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(crawl.step().await.unwrap(), 1);
    assert_eq!(record()["listed"].as_bool(), Some(true));
    let stored = buckets.store(host).url(&urls::parse(&format!("http://{host}/paper/7"), host).unwrap()).unwrap();
    assert!(stored.is_some(), "the listed visit reads its homepage as a list");
    std::fs::remove_file(&list).ok();
}

/// A site whose homepage links to sections that link to pages, and to what a run must leave alone.
fn deep_site(web: &Shared, host: &str) {
    let mut w = web.lock().unwrap();
    w.add(host, "/robots.txt", 200, None, b"User-agent: *\nDisallow: /private/\n");
    let page = |links: &[&str]| {
        let links: String = links.iter().map(|href| format!("<a href=\"{href}\">{href}</a>")).collect();
        format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").into_bytes()
    };
    w.add(host, "/", 200, None, &page(&["/a", "/b", "/private/x", "/search?q=1", "/report.pdf", "http://other.test/x"]));
    w.add(host, "/a", 200, None, &page(&["/a/1", "/a/2", "/a/3"]));
    w.add(host, "/a/1", 200, None, &page(&["/a/1/deep"]));
    w.add(host, "/b", 200, None, &page(&["/b/1"]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_listed_site_without_sitemaps_has_its_links_followed_a_few_levels_down_within_its_rules() {
    let web: Shared = Arc::default();
    let host = "deep.test";
    deep_site(&web, host);
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "links", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    for path in ["/", "/a", "/b", "/a/1", "/a/1/deep"] {
        assert!(stored(path), "{path}");
    }
    assert!(!stored("/a/2") && !stored("/b/1"), "pages that answered 404 are never queued");
    assert!(!stored("/private/x") && !stored("/search?q=1"), "robots.txt and queries are respected");
    let hits = web.lock().unwrap().hits.clone();
    for path in ["/private/x", "/a/1/deep", "/report.pdf", "/search?q=1"] {
        assert!(!hits.iter().any(|(_, p, _)| p == path), "{path} is never fetched");
    }
    assert!(!hits.iter().any(|(h, _, _)| h == "other.test"));
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["reason"].as_str(), record["urls"].as_i64()), (Some("no_sitemap"), Some("sitemap_too_small"), Some(4)), "the run's pages are queued, and the verdict is the homepage's");
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().unwrap();
    let next = links["next"].as_i64().unwrap() - chrono::Utc::now().timestamp();
    assert_eq!(links["kind"].as_str(), Some("links"));
    assert!((6 * 86_400..=7 * 86_400).contains(&next), "a run that found new pages comes again in a week, not in {next} s");
}

/// Whether one visit stored a page three links below the homepage, and in how many days its next link run is due.
async fn followed(host: &'static str, mode: FollowLinks, web: Shared, terms: Terms) -> (bool, Option<i64>) {
    let (buckets, _crawl) = crawl_following(web, &[host], host, mode, Slots::new(2), |buckets| buckets.allow(&Allowed::with_terms([(host, terms)]))).await;
    let deep = buckets.store(host).url(&urls::parse(&format!("http://{host}/a/1/deep"), host).unwrap()).unwrap().is_some();
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap();
    (deep, links.and_then(|l| l["next"].as_i64()).map(|next| (next - chrono::Utc::now().timestamp() + 43_200) / 86_400))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn links_are_followed_on_whole_sites_with_big_sitemaps_only_when_switched_to_all_and_never_when_off() {
    let big = |host: &str| {
        let web: Shared = Arc::default();
        deep_site(&web, host);
        let list = format!("http://{host}/list.xml");
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nDisallow: /private/\nSitemap: {list}\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..1200).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        drop(w);
        web
    };
    let whole = Terms { weight: 3, ..Terms::default() };
    assert_eq!(followed("big-thin.test", FollowLinks::Thin, big("big-thin.test"), whole).await, (false, None), "a site whose sitemaps list 1,200 pages");
    assert_eq!(followed("big-all.test", FollowLinks::All, big("big-all.test"), whole).await, (true, Some(30)), "a whole site, monthly");
    assert_eq!(followed("big-node.test", FollowLinks::All, big("big-node.test"), Terms::default()).await, (false, None), "a site the list weights below its own picks");
    let flat: Shared = Arc::default();
    deep_site(&flat, "flat.test");
    assert_eq!(followed("flat.test", FollowLinks::Off, flat, Terms::default()).await, (false, None), "off, even with no sitemap");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_stops_at_a_refusal_follows_no_redirect_off_its_rules_and_no_link_on_a_nofollow_page() {
    let web: Shared = Arc::default();
    let host = "polite.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nDisallow: /private/\n");
        let page = |head: &str, links: &[&str]| {
            let links: String = links.iter().map(|href| format!("<a href=\"{href}\">{href}</a>")).collect();
            format!("<html lang=\"en\"><head>{head}</head><p>{ENGLISH}</p>{links}</html>").into_bytes()
        };
        w.add(host, "/", 200, None, &page("", &["/a", "/go", "/b", "/c"]));
        w.add(host, "/a", 200, None, &page("<meta content=\"noindex, nofollow\" name=\"robots\">", &["/a/1"]));
        w.redirect(host, "/go", &format!("http://{host}/private/x"), None);
        w.add(host, "/b", 403, None, b"no");
        w.add(host, "/c", 200, None, &page("", &["/c/1"]));
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "polite", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    assert!(stored("/c") && !stored("/a/1") && !stored("/c/1"), "the nofollow page's link is never kept, nor what lies past the refusal");
    let hits = web.lock().unwrap().hits.clone();
    for path in ["/private/x", "/c", "/a/1"] {
        assert!(!hits.iter().any(|(_, p, _)| p == path), "{path} is never fetched");
    }
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().unwrap();
    assert_eq!(links["status"].as_str(), Some("ok"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_that_finds_every_slot_taken_waits_hours_to_days_and_nothing_loops() {
    let web: Shared = Arc::default();
    let host = "waiting.test";
    deep_site(&web, host);
    let links = Slots::new(1);
    let _held = links.try_take().unwrap();
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "links-wait", FollowLinks::Thin, links, |buckets| buckets.allow(&Allowed::of([host]))).await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    let wait = record["due"].as_i64().unwrap() - chrono::Utc::now().timestamp();
    assert!((2 * 3_600 - 5..=7 * 86_400).contains(&wait), "due in {wait} s");
    assert!(buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().is_none());

    let web: Shared = Arc::default();
    let host = "grown.test";
    let list = format!("http://{host}/list.xml");
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: {list}\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..1200).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "grown", |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut changes = Changes::default();
        for (url, kind, urls) in [(list.clone(), "urlset", 1200), (format!("http://{host}/"), "homepage", 3), (format!("https://{host}/#links"), "links", 40)] {
            changes.sitemap(host, &url, serde_json::json!({"id": sitemap_id(&url), "kind": kind, "depth": 0, "status": "ok", "urls": urls, "next": 0}).as_object().unwrap());
        }
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    let due = record["due"].as_i64().unwrap() - chrono::Utc::now().timestamp();
    assert_eq!(record["state"].as_str(), Some("active"));
    assert!(due > 60, "a homepage and link listing no longer read set no time, so the site is not due again at once: {due} s");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_whose_homepage_is_not_english_has_no_links_followed() {
    let web: Shared = Arc::default();
    let host = "ailleurs.test";
    {
        let mut w = web.lock().unwrap();
        let french = "Le comité examine le budget annuel pendant que les lecteurs suivent les nouvelles locales chaque jour.";
        let links: String = (0..12).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"fr\"><p>{french} {french} {french}</p>{links}</html>").as_bytes());
        w.add(host, "/page/1", 200, None, format!("<html><p>{french}</p><a href=\"/page/1/deep\">plus</a></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "not-english", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["reason"].as_str()), (Some("ineligible"), Some("not_english")));
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/page/1"), "judged before any link is followed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_homepage_read_before_a_link_run_only_decides_whether_the_run_goes() {
    let web: Shared = Arc::default();
    let (refuses, small) = ("refuses.test", "petit.test");
    {
        let mut w = web.lock().unwrap();
        w.add(refuses, "/", 403, None, b"no");
        let french = "Le comité examine le budget annuel pendant que les lecteurs suivent les nouvelles locales chaque jour.";
        let links: String = (0..3).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(small, "/", 200, None, format!("<html lang=\"fr\"><p>{french} {french}</p>{links}</html>").as_bytes());
        w.add(small, "/page/1", 200, None, format!("<html><p>{french}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[refuses, small], "gate", |buckets| buckets.allow(&Allowed::of([refuses, small]))).await;
    let record = |host: &str| buckets.store(host).domain(host).unwrap().unwrap();
    let verdict = |host: &str| (record(host)["state"].as_str().map(str::to_string), record(host)["reason"].as_str().map(str::to_string));
    assert_eq!(verdict(refuses), (Some("no_sitemap".into()), Some("no_sitemap".into())), "as without link runs: a refusing homepage is no verdict on a site without sitemaps");
    assert_eq!(verdict(small), (Some("no_sitemap".into()), Some("sitemap_too_small".into())));
    for host in [refuses, small] {
        assert!(record(host).get("lang").is_none() && record(host).get("home_status").is_none(), "{host} keeps no homepage verdict");
    }
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/page/1"), "no run past a homepage that isn't English");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_that_gets_no_answer_leaves_an_active_site_active() {
    let web: Shared = Arc::default();
    let host = "quiet.test";
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "quiet", |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for (name, at) in [("robots", now), ("checked", now), ("ok", now - 86_400)] {
            record.insert(name.into(), at.into());
        }
        record.insert("lang".into(), "en".into());
        let mut changes = Changes::default();
        changes.domain(host, &record);
        let list = format!("http://{host}/list.xml");
        let home = format!("http://www.{host}/");
        for (url, kind, urls) in [(list, "urlset", 12), (home, "homepage", 5)] {
            changes.sitemap(host, &url, serde_json::json!({"id": sitemap_id(&url), "kind": kind, "depth": 0, "status": "ok", "urls": urls, "next": now + 86_400}).as_object().unwrap());
        }
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!(record["state"].as_str(), Some("active"), "a run whose fetches all fail says nothing about the site: {record:?}");
    assert_eq!(record["ok"].as_i64(), Some(now - 86_400), "nor that it answered");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rate_limit_a_link_run_draws_is_no_verdict_on_the_site() {
    let web: Shared = Arc::default();
    let host = "limits.test";
    {
        let mut w = web.lock().unwrap();
        let links: String = (0..12).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
        let limited = Page { status: 429, etag: None, body: b"slow down".to_vec(), headers: Vec::new(), with_cookie: None };
        w.later.insert((host.into(), "/".into()), (2, limited));
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "limits", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["lang"].as_str()), (Some("active"), Some("en")), "judged on the homepage as it answered before the run: {record:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_homepage_that_asks_robots_not_to_follow_its_links_lists_only_itself() {
    let web: Shared = Arc::default();
    let (meta, header, preview) = ("meta-nofollow.test", "header-nofollow.test", "preview.test");
    {
        let mut w = web.lock().unwrap();
        let links: String = (0..12).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(meta, "/", 200, None, format!("<html lang=\"en\"><head><meta name=robots content=noindex,nofollow></head><p>{ENGLISH}</p>{links}</html>").as_bytes());
        let body = format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").into_bytes();
        let headers = vec![("X-Robots-Tag", "noarchive".to_string()), ("X-Robots-Tag", "none".to_string())];
        w.pages.insert((header.into(), "/".into()), Page { status: 200, etag: None, body, headers, with_cookie: None });
        let body = format!("<html lang=\"en\"><head><meta name=\"robots\" content=\"index, max-image-preview:none\"></head><p>{ENGLISH}</p>{links}</html>").into_bytes();
        let headers = vec![("X-Robots-Tag", "max-image-preview:none, otherbot: noarchive".to_string())];
        w.pages.insert((preview.into(), "/".into()), Page { status: 200, etag: None, body, headers, with_cookie: None });
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[meta, header, preview], "nofollow", |buckets| buckets.allow(&Allowed::of([meta, header, preview]))).await;
    assert!(buckets.store(preview).url(&urls::parse(&format!("http://{preview}/page/3"), preview).unwrap()).unwrap().is_some(), "an image preview limit is no nofollow");
    for host in [meta, header] {
        let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
        assert!(stored("/") && !stored("/page/3"), "{host}: the homepage only");
    }
    assert!(!web.lock().unwrap().hits.iter().any(|(host, path, _)| host != preview && path.starts_with("/page/")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_listed_with_www_has_its_robots_rules_applied_and_a_link_run_stays_on_its_port() {
    let web: Shared = Arc::default();
    // Under a public suffix of its own, as gov.uk is, the www host is the domain itself.
    let (host, ported) = ("www.gov.test", "ported.test");
    let other = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = other.local_addr().unwrap().port();
    tokio::spawn(serve(other, web.clone()));
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nDisallow: /private/\nSitemap: http://{host}/index.xml\n").as_bytes());
        w.add(host, "/index.xml", 200, None, format!("<sitemapindex><sitemap><loc>http://{host}/private/more.xml</loc></sitemap><sitemap><loc>http://{host}/ok.xml</loc></sitemap></sitemapindex>").as_bytes());
        let pages: Vec<String> = (0..12).map(|i| format!("/a/{i}")).chain(["/private/b".to_string()]).collect();
        w.add(host, "/ok.xml", 200, None, urlset(host, &pages).as_bytes());
        w.add(host, "/private/more.xml", 200, None, urlset(host, &["/private/c".into()]).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
        w.add(ported, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p><a href=\"/go\">go</a></html>").as_bytes());
        w.redirect(ported, "/go", &format!("http://{ported}:{port}/elsewhere"), None);
        w.add(ported, "/elsewhere", 200, None, format!("<html><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "www-robots", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let _ported = crawl_prepared(web.clone(), &[ported], "www-ported", |buckets| buckets.allow(&Allowed::of([ported]))).await;
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    assert!(stored("/a/3") && !stored("/private/b") && !stored("/private/c"), "a page robots.txt disallows is never kept");
    let hits = web.lock().unwrap().hits.clone();
    assert!(!hits.iter().any(|(_, path, _)| path.starts_with("/private/")), "nor fetched");
    assert!(!hits.iter().any(|(_, path, _)| path == "/elsewhere"), "a redirect to another port leaves the site");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_keeps_no_files_or_forms_and_stops_at_a_busy_server() {
    let web: Shared = Arc::default();
    let host = "busy.links.test";
    {
        let mut w = web.lock().unwrap();
        let page = |links: &[&str]| {
            let links: String = links.iter().map(|href| format!("<a href=\"{href}\">{href}</a>")).collect();
            format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").into_bytes()
        };
        w.add(host, "/", 200, None, &page(&["/x", "/y", "/z"]));
        w.add(host, "/x", 200, None, &page(&["/x/1", "/x/form.docx", "/wp-admin/edit", "/cdn-cgi/l/email", "/x/icon.ico"]));
        w.add(host, "/y", 503, None, b"busy");
        w.add(host, "/z", 200, None, &page(&["/z/1"]));
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "links-busy", |buckets| buckets.allow(&Allowed::of([host]))).await;
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    assert!(stored("/x/1"));
    for path in ["/x/form.docx", "/wp-admin/edit", "/cdn-cgi/l/email", "/x/icon.ico"] {
        assert!(!stored(path), "{path} is no page");
    }
    assert!(!web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/z"), "the 503 ends the run");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_homepage_listing_not_read_on_a_recheck_keeps_no_site_active() {
    let web: Shared = Arc::default();
    let host = "homeonly.test";
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "homeonly", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1), Some("new_generic"), &[], State::NoSitemap, Some("sitemap_too_small"), Some(0)));
        let home = format!("http://{host}/");
        changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "kind": "homepage", "depth": 0, "status": "ok", "urls": 13, "next": now + 3_600}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!(record["state"].as_str(), Some("no_sitemap"), "only sitemaps stand in for a read");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn guesses_go_where_the_homepage_was_last_read() {
    let web: Shared = Arc::default();
    let (host, www) = ("lastread.test", "www.lastread.test");
    {
        let mut w = web.lock().unwrap();
        w.add(www, "/sitemap.xml", 200, None, urlset(www, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(www, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_following(web.clone(), &[host, www], "last-read", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for (name, at) in [("robots", now), ("checked", now), ("ok", now)] {
            record.insert(name.into(), at.into());
        }
        record.insert("lang".into(), "en".into());
        let mut changes = Changes::default();
        changes.domain(host, &record);
        for (home, fetched) in [(format!("http://{host}/"), now - 10 * 86_400), (format!("http://{www}/"), now - 86_400)] {
            changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "kind": "homepage", "depth": 0, "status": "ok", "urls": 3, "next": 0, "fetched": fetched}).as_object().unwrap());
        }
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!(record["state"].as_str(), Some("active"), "{record:?}");
    assert!(buckets.store(host).url(&urls::parse(&format!("http://{www}/story/3"), host).unwrap()).unwrap().is_some(), "the guess went to the www host, read last");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sites_on_one_address_take_their_link_runs_one_at_a_time() {
    let web: Shared = Arc::default();
    let hosts = ["first.shared.test", "second.shared.test", "third.shared.test"];
    for host in hosts {
        deep_site(&web, host);
    }
    let (buckets, _crawl) = crawl_prepared(web.clone(), &hosts, "shared-address", |buckets| buckets.allow(&Allowed::of(hosts))).await;
    let ran: Vec<&str> = hosts.iter().copied().filter(|host| buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().is_some()).collect();
    assert_eq!(ran.len(), 1, "they all answer from 127.0.0.1: {ran:?}");
    for host in hosts.iter().filter(|host| !ran.contains(host)) {
        let wait = buckets.store(host).domain(host).unwrap().unwrap()["due"].as_i64().unwrap() - chrono::Utc::now().timestamp();
        assert!((2 * 3_600 - 5..=7 * 86_400).contains(&wait), "{host} waits as for a busy slot: {wait} s");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sitemap_that_refuses_us_is_no_reason_to_guess() {
    let web: Shared = Arc::default();
    let host = "refuses-sitemap.test";
    {
        let mut w = web.lock().unwrap();
        for path in ["/robots.txt", "/list.xml", "/sitemap.xml", "/sitemap_index.xml", "/feed", "/rss.xml", "/"] {
            w.add(host, path, 403, None, b"denied");
        }
    }
    let now = chrono::Utc::now().timestamp();
    let (_buckets, _crawl) = crawl_following(web.clone(), &[host], "refused-sitemap", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for (name, at) in [("robots", now), ("checked", now), ("ok", now)] {
            record.insert(name.into(), at.into());
        }
        record.insert("lang".into(), "en".into());
        let mut changes = Changes::default();
        changes.domain(host, &record);
        let list = format!("http://{host}/list.xml");
        changes.sitemap(host, &list, serde_json::json!({"id": sitemap_id(&list), "kind": "urlset", "depth": 0, "status": "error", "error": "http_403", "urls": 500, "next": 0}).as_object().unwrap());
        let home = format!("http://{host}/");
        changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "kind": "homepage", "depth": 0, "status": "ok", "urls": 3, "next": now + 86_400, "fetched": now - 3_600}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits, ["/list.xml"], "a refusal is an answer: the sitemap is asked again, and nothing is guessed");
}

/// A site's state and reason after one visit in a mode, with every link slot taken when `busy`.
async fn verdict(web: Shared, host: &'static str, name: &str, mode: FollowLinks, busy: bool, prepare: impl FnOnce(&Arc<Buckets>)) -> (Option<String>, Option<String>, bool) {
    let links = Slots::new(1);
    let _held = busy.then(|| links.try_take().unwrap());
    let (buckets, _crawl) = crawl_following(web, &[host], name, mode, links, prepare).await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    let home = buckets.store(host).sitemap(host, &format!("http://{host}/")).unwrap().is_some();
    (record["state"].as_str().map(str::to_string), record["reason"].as_str().map(str::to_string), home)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn following_links_changes_no_verdict_even_when_no_run_happens() {
    let small = |web: &Shared, host: &str| {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..5).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        let links: String = (0..12).map(|i| format!("<a href=\"/page/{i}\">Page {i}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
    };
    let gone = |web: &Shared, host: &str| {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        w.add(host, "/list.xml", 404, None, b"gone");
        w.add(host, "/", 403, None, b"denied");
    };
    let active = |host: &'static str| {
        move |buckets: &Arc<Buckets>| {
            buckets.allow(&Allowed::of([host]));
            let now = chrono::Utc::now().timestamp();
            let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
            for (name, at) in [("robots", now - 10 * 86_400), ("checked", now - 86_400), ("ok", now - 86_400)] {
                record.insert(name.into(), at.into());
            }
            record.insert("lang".into(), "en".into());
            let mut changes = Changes::default();
            changes.domain(host, &record);
            let list = format!("http://{host}/list.xml");
            changes.sitemap(host, &list, serde_json::json!({"id": sitemap_id(&list), "kind": "urlset", "depth": 0, "status": "ok", "urls": 12, "next": 0}).as_object().unwrap());
            buckets.store(host).write(changes).unwrap();
        }
    };
    for (off, thin, site) in [("small-off.test", "small-thin.test", 0), ("gone-off.test", "gone-thin.test", 1)] {
        let (off_web, thin_web): (Shared, Shared) = (Arc::default(), Arc::default());
        if site == 0 {
            small(&off_web, off);
            small(&thin_web, thin);
        } else {
            gone(&off_web, off);
            gone(&thin_web, thin);
        }
        let listed = |host: &'static str| move |buckets: &Arc<Buckets>| buckets.allow(&Allowed::of([host]));
        let (a, b) = if site == 0 {
            (verdict(off_web, off, off, FollowLinks::Off, false, listed(off)).await, verdict(thin_web, thin, thin, FollowLinks::Thin, true, listed(thin)).await)
        } else {
            (verdict(off_web, off, off, FollowLinks::Off, false, active(off)).await, verdict(thin_web, thin, thin, FollowLinks::Thin, true, active(thin)).await)
        };
        assert_eq!(a, b, "{off} and {thin}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_changes_no_verdict() {
    let (off_web, thin_web): (Shared, Shared) = (Arc::default(), Arc::default());
    deep_site(&off_web, "deep-off.test");
    deep_site(&thin_web, "deep-thin.test");
    let listed = |host: &'static str| move |buckets: &Arc<Buckets>| buckets.allow(&Allowed::of([host]));
    let off = verdict(off_web, "deep-off.test", "deep-off", FollowLinks::Off, false, listed("deep-off.test")).await;
    let thin = verdict(thin_web.clone(), "deep-thin.test", "deep-thin", FollowLinks::Thin, false, listed("deep-thin.test")).await;
    assert_eq!(off, thin);
    assert!(thin_web.lock().unwrap().hits.iter().any(|(_, path, _)| path == "/a/1"), "the run did go");

    let gone = |web: &Shared, host: &str| {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        w.add(host, "/list.xml", 404, None, b"gone");
        w.add(host, "/", 403, None, b"denied");
    };
    let active = |host: &'static str| {
        move |buckets: &Arc<Buckets>| {
            buckets.allow(&Allowed::of([host]));
            let now = chrono::Utc::now().timestamp();
            let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
            for (name, at) in [("robots", now - 10 * 86_400), ("checked", now - 86_400), ("ok", now - 86_400)] {
                record.insert(name.into(), at.into());
            }
            record.insert("lang".into(), "en".into());
            let mut changes = Changes::default();
            changes.domain(host, &record);
            let list = format!("http://{host}/list.xml");
            changes.sitemap(host, &list, serde_json::json!({"id": sitemap_id(&list), "kind": "urlset", "depth": 0, "status": "ok", "urls": 12, "next": 0}).as_object().unwrap());
            buckets.store(host).write(changes).unwrap();
        }
    };
    let (off_web, thin_web): (Shared, Shared) = (Arc::default(), Arc::default());
    gone(&off_web, "gone-run-off.test");
    gone(&thin_web, "gone-run-thin.test");
    let off = verdict(off_web, "gone-run-off.test", "gone-run-off", FollowLinks::Off, false, active("gone-run-off.test")).await;
    let thin = verdict(thin_web, "gone-run-thin.test", "gone-run-thin", FollowLinks::Thin, false, active("gone-run-thin.test")).await;
    assert_eq!((off.clone(), off.1.as_deref()), (thin, Some("sitemap_gone")), "a failed run's record never stands in for the gone sitemap");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_queues_no_page_that_failed_or_moved_where_robots_txt_disallows() {
    let web: Shared = Arc::default();
    let host = "keepgo.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nDisallow: /private/\n");
        let links: String = ["/go", "/gone", "/moved", "/b"].iter().map(|href| format!("<a href=\"{href}\">{href}</a>")).collect();
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
        w.redirect(host, "/go", &format!("http://{host}/private/x"), None);
        w.add(host, "/gone", 404, None, b"no");
        w.redirect(host, "/moved", &format!("http://{host}/new-home"), None);
        w.add(host, "/new-home", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
        w.add(host, "/b", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "keepgo", FollowLinks::Thin, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().unwrap();
    assert_eq!(links["urls"].as_i64(), Some(3), "the homepage, /b and /new-home: {links:?}");
    let stored = |path: &str| buckets.store(host).url(&urls::parse(&format!("http://{host}{path}"), host).unwrap()).unwrap().is_some();
    assert!(stored("/new-home") && !stored("/private/x"));
    let ready: Vec<String> = buckets.store(host).ready_list(host, 100).unwrap().iter().map(|e| String::from_utf8_lossy(&e.rest).into_owned()).collect();
    for path in ["/go", "/gone", "/moved"] {
        assert!(!ready.contains(&format!("{host}{path}")), "{path}, listed by the homepage, waits no more once the run found it gone or moved: {ready:?}");
    }
    assert!(ready.contains(&format!("{host}/b")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_finding_mostly_new_pages_comes_back_in_its_interval_not_minutes() {
    let web: Shared = Arc::default();
    let host = "fresh-links.test";
    let list = format!("http://{host}/list.xml");
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p><a href=\"/p1\">1</a><a href=\"/p2\">2</a></html>").as_bytes());
        for part in ["p1", "p2"] {
            let links: String = (0..300).map(|i| format!("<a href=\"/x/x/x/{part}-{i}\">{i}</a>")).collect();
            w.add(host, &format!("/{part}"), 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
        }
    }
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "fresh-links", |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for (name, at) in [("robots", now), ("checked", now), ("ok", now)] {
            record.insert(name.into(), at.into());
        }
        record.insert("lang".into(), "en".into());
        let mut changes = Changes::default();
        changes.domain(host, &record);
        let (home, links) = (format!("http://{host}/"), format!("https://{host}/#links"));
        for (url, kind, urls, next) in [(list.clone(), "urlset", 12, 0), (home, "homepage", 3, now + 86_400), (links, "links", 600, 0)] {
            changes.sitemap(host, &url, serde_json::json!({"id": sitemap_id(&url), "kind": kind, "depth": 0, "status": "ok", "urls": urls, "next": next}).as_object().unwrap());
        }
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().unwrap();
    assert!(links["next"].as_i64().unwrap() - now > 86_400, "no second run minutes later: {links:?}");
    assert!(buckets.store(host).url(&urls::parse(&format!("http://{host}/x/x/x/p2-7"), host).unwrap()).unwrap().is_some(), "and its 600 pages are kept");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_whose_robots_txt_names_sitemaps_that_refuse_us_is_not_guessed_at() {
    let web: Shared = Arc::default();
    let host = "named-refuses.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/news.xml\n").as_bytes());
        w.add(host, "/news.xml", 403, None, b"denied");
        w.add(host, "/sitemap.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "named-refuses", FollowLinks::Off, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits, ["/robots.txt", "/news.xml"], "the sitemap robots.txt names refused us: no way around it");
    assert_eq!(buckets.store(host).domain(host).unwrap().unwrap()["state"].as_str(), Some("no_sitemap"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_that_refuses_a_guess_gets_no_more_guesses() {
    let web: Shared = Arc::default();
    let host = "refuses-guesses.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/feed", "/rss.xml", "/"] {
            w.add(host, path, 403, None, b"denied");
        }
    }
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "refuses-guesses", FollowLinks::Off, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits, ["/robots.txt", "/sitemap.xml"], "the first refusal ends the guessing");
    assert_eq!(buckets.store(host).domain(host).unwrap().unwrap()["state"].as_str(), Some("no_sitemap"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_whose_robots_txt_refuses_us_gets_no_link_work() {
    let (off_web, thin_web): (Shared, Shared) = (Arc::default(), Arc::default());
    for (web, host) in [(&off_web, "refuses-off.test"), (&thin_web, "refuses-thin.test")] {
        deep_site(web, host);
        web.lock().unwrap().add(host, "/robots.txt", 403, None, b"denied");
    }
    let listed = |host: &'static str| move |buckets: &Arc<Buckets>| buckets.allow(&Allowed::of([host]));
    let off = verdict(off_web.clone(), "refuses-off.test", "refuses-off", FollowLinks::Off, false, listed("refuses-off.test")).await;
    let thin = verdict(thin_web.clone(), "refuses-thin.test", "refuses-thin", FollowLinks::Thin, false, listed("refuses-thin.test")).await;
    assert_eq!(off, thin);
    let paths = |web: &Shared| web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect::<Vec<_>>();
    assert_eq!(paths(&off_web), paths(&thin_web), "not one request more");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_that_refuses_a_sitemap_or_a_guess_gets_no_link_work() {
    for (case, robots, refusing) in [("named", "Sitemap: http://{host}/news.xml\n", "/news.xml"), ("guessed", "", "/sitemap.xml")] {
        let (off_web, thin_web): (Shared, Shared) = (Arc::default(), Arc::default());
        let (off_host, thin_host): (&'static str, &'static str) = (format!("{case}-off.test").leak(), format!("{case}-thin.test").leak());
        for (web, host) in [(&off_web, off_host), (&thin_web, thin_host)] {
            deep_site(web, host);
            let mut w = web.lock().unwrap();
            w.add(host, "/robots.txt", 200, None, format!("User-agent: *\n{}", robots.replace("{host}", host)).as_bytes());
            w.add(host, refusing, 403, None, b"denied");
        }
        let listed = |host: &'static str| move |buckets: &Arc<Buckets>| buckets.allow(&Allowed::of([host]));
        let off = verdict(off_web.clone(), off_host, &format!("{case}-off"), FollowLinks::Off, false, listed(off_host)).await;
        let thin = verdict(thin_web.clone(), thin_host, &format!("{case}-thin"), FollowLinks::Thin, false, listed(thin_host)).await;
        assert_eq!(off, thin, "{case}");
        let paths = |web: &Shared| web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(&off_web), ["/robots.txt", refusing], "{case}");
        assert_eq!(paths(&off_web), paths(&thin_web), "{case}: not one request more");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_site_whose_robots_txt_refuses_us_is_not_guessed_at_even_with_a_gone_sitemap_stored() {
    let web: Shared = Arc::default();
    let host = "robots-refuses-gone.test";
    web.lock().unwrap().add(host, "/robots.txt", 403, None, b"denied");
    let (_buckets, crawl) = crawl_following(web.clone(), &[host], "robots-refuses-gone", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let gone = format!("http://{host}/sitemap.xml");
        let mut changes = Changes::default();
        changes.sitemap(host, &gone, serde_json::json!({"id": sitemap_id(&gone), "kind": "urlset", "depth": 0, "status": "gone", "urls": 0, "next": 0}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits, ["/robots.txt", "/sitemap.xml"], "only the stored sitemap is asked for again");
    assert_eq!(crawl.stats.requests, 3, "robots.txt on https then http, and the stored sitemap: no guess on either scheme");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_named_sitemap_that_went_dead_hides_the_guesses_on_a_visit_that_skips_robots_txt() {
    let web: Shared = Arc::default();
    let host = "named-dead.test";
    let now = chrono::Utc::now().timestamp();
    let (_buckets, crawl) = crawl_following(web.clone(), &[host], "named-dead", FollowLinks::Off, Slots::new(2), |buckets| {
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
        for name in ["robots", "checked", "ok"] {
            record.insert(name.into(), now.into());
        }
        record.insert("lang".into(), "en".into());
        let (news, home) = (format!("http://{host}/news.xml"), format!("http://{host}/"));
        let mut changes = Changes::default();
        changes.domain(host, &record);
        changes.sitemap(host, &news, serde_json::json!({"id": sitemap_id(&news), "kind": "urlset", "depth": 0, "status": "error", "error": "SocketTimeoutError", "urls": 0, "next": now + 3600}).as_object().unwrap());
        changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "kind": "homepage", "depth": 0, "status": "ok", "urls": 3, "next": now + 3600}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert!(hits.is_empty(), "robots.txt named /news.xml, read an hour ago: nothing is guessed, got {hits:?}");
    assert_eq!(crawl.stats.requests, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_landing_site_whose_homepage_failed_last_time_is_still_guessed_at() {
    let web: Shared = Arc::default();
    let host = "landing-home-500.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        w.add(host, "/sitemap.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
        w.add(host, "/", 500, None, b"oops");
    }
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "landing-home-500", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::with_terms([(host, Terms { depth: Some(2), ..Terms::default() })]));
        let home = format!("http://{host}/");
        let mut changes = Changes::default();
        changes.sitemap(host, &home, serde_json::json!({"id": sitemap_id(&home), "depth": 0, "status": "error", "error": "http_500", "urls": 0, "next": 0}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert!(hits.contains(&"/sitemap.xml".to_string()), "a failed homepage read is no sitemap: {hits:?}");
    assert_ne!(record["reason"].as_str(), Some("no_sitemap"), "{record:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn link_work_failing_on_our_side_keeps_the_verdict() {
    let web: Shared = Arc::default();
    let host = "links-crash.test";
    deep_site(&web, host);
    let (buckets, crawl) = crawl_adjusted(
        web.clone(),
        &[host],
        &[],
        "links-crash",
        FollowLinks::Thin,
        Slots::new(2),
        |buckets| buckets.allow(&Allowed::of([host])),
        // Only the link run's homepage check asks for the language of a site without sitemaps.
        |visitor| visitor.language = Arc::new(|_: &str| panic!("language detection broke")),
    )
    .await;
    let record = buckets.store(host).domain(host).unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["reason"].as_str()), (Some("no_sitemap"), Some("sitemap_too_small")), "{record:?}");
    assert_eq!(crawl.summary()["link_errors"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_keeps_a_page_that_redirects_to_its_own_store_key() {
    let mut kept = Vec::new();
    for mode in [FollowLinks::Off, FollowLinks::Thin] {
        let host: &'static str = format!("samekey-{mode:?}.test").to_lowercase().leak();
        let www: &'static str = format!("www.{host}").leak();
        let web: Shared = Arc::default();
        {
            let mut w = web.lock().unwrap();
            w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
            let mut paths: Vec<String> = (0..10).map(|i| format!("/story/{i}")).collect();
            paths.extend(["/about".to_string(), "/contact".to_string()]);
            w.add(host, "/list.xml", 200, None, urlset(host, &paths).as_bytes());
            let page = |links: &[&str]| {
                let links: String = links.iter().map(|href| format!("<a href=\"{href}\">{href}</a>")).collect();
                format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").into_bytes()
            };
            w.add(host, "/", 200, None, &page(&["/about?utm_source=nav", &format!("http://{www}/contact")]));
            w.redirect(host, "/about?utm_source=nav", &format!("http://{host}/about"), None);
            w.redirect(www, "/contact", &format!("http://{host}/contact"), None);
            w.add(host, "/about", 200, None, &page(&[]));
            w.add(host, "/contact", 200, None, &page(&[]));
        }
        let (buckets, _crawl) = crawl_resolving(web.clone(), &[host], &[www], &format!("samekey-{mode:?}"), mode, |buckets| buckets.allow(&Allowed::of([host]))).await;
        let ready: Vec<String> = buckets.store(host).ready_list(host, 1000).unwrap().iter().map(|e| String::from_utf8_lossy(&e.rest).into_owned()).collect();
        kept.push((ready.contains(&format!("{host}/about")), ready.contains(&format!("{host}/contact"))));
    }
    assert_eq!(kept, [(true, true), (true, true)], "a redirect to the same page, by tracking query or www host, takes nothing off the ready list");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_takes_no_page_a_sitemap_lists_off_the_ready_list() {
    let web: Shared = Arc::default();
    let host = "listed-gone.test";
    {
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
        let mut paths: Vec<String> = (0..10).map(|i| format!("/story/{i}")).collect();
        paths.push("/gone".into());
        w.add(host, "/list.xml", 200, None, urlset(host, &paths).as_bytes());
        w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p><a href=\"/gone\">x</a><a href=\"/story/1\">y</a></html>").as_bytes());
        w.add(host, "/story/1", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
    }
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "listed-gone", FollowLinks::Thin, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert!(hits.contains(&"/gone".to_string()), "the run asked for it: {hits:?}");
    let ready: Vec<String> = buckets.store(host).ready_list(host, 1000).unwrap().iter().map(|e| String::from_utf8_lossy(&e.rest).into_owned()).collect();
    assert!(ready.contains(&format!("{host}/gone")), "the sitemap lists it, so a 404 to a link run leaves it queued: {ready:?}");
}

/// An active thin site, its language known, whose robots.txt names one sitemap of a dozen pages.
fn thin_active(web: &Shared, host: &str, list: u16) {
    let mut w = web.lock().unwrap();
    w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
    if list == 200 {
        w.add(host, "/list.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
    } else {
        w.add(host, "/list.xml", list, None, b"denied");
    }
}

fn active_record(now: i64) -> serde_json::Map<String, serde_json::Value> {
    let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
    for (name, at) in [("robots", now - 2 * 86_400), ("checked", now - 86_400), ("ok", now - 86_400)] {
        record.insert(name.into(), at.into());
    }
    record.insert("lang".into(), "en".into());
    record
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sitemap_that_refused_us_last_visit_still_bars_link_work() {
    let mut hits_by = Vec::new();
    for mode in [FollowLinks::Off, FollowLinks::Thin] {
        let host: &'static str = format!("refused-before-{mode:?}.test").to_lowercase().leak();
        let web: Shared = Arc::default();
        deep_site(&web, host);
        thin_active(&web, host, 403);
        let now = chrono::Utc::now().timestamp();
        let (_buckets, _crawl) = crawl_following(web.clone(), &[host], &format!("refused-before-{mode:?}"), mode, Slots::new(2), |buckets| {
            buckets.allow(&Allowed::of([host]));
            let list = format!("http://{host}/list.xml");
            let mut changes = Changes::default();
            changes.domain(host, &active_record(now));
            changes.sitemap(host, &list, serde_json::json!({"id": sitemap_id(&list), "kind": "urlset", "depth": 0, "status": "error", "error": "http_403", "urls": 0, "next": now + 2 * 86_400}).as_object().unwrap());
            buckets.store(host).write(changes).unwrap();
        })
        .await;
        hits_by.push(web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect::<Vec<_>>());
    }
    assert_eq!(hits_by[0], ["/robots.txt"]);
    assert_eq!(hits_by[0], hits_by[1], "a site that refused its sitemap gets no more requests with links followed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_its_homepage_refuses_waits_twice_as_long_before_the_next() {
    let web: Shared = Arc::default();
    let host = "home-refuses-run.test";
    thin_active(&web, host, 200);
    web.lock().unwrap().add(host, "/", 403, None, b"denied");
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "home-refuses-run", FollowLinks::Thin, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let mut changes = Changes::default();
        changes.domain(host, &active_record(now));
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().unwrap();
    assert_eq!(links["error"].as_str(), Some("http_403"));
    assert!(links["next"].as_i64().unwrap() - now >= 14 * 86_400 - 60, "a refused run comes back after twice its week: {links:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_homepage_refusing_link_works_language_check_is_kept_as_a_refused_run() {
    let web: Shared = Arc::default();
    let host = "gate-refuses.test";
    {
        // robots.txt names a sitemap too small to list the site, so the visit guesses nothing and reads no homepage.
        let mut w = web.lock().unwrap();
        w.add(host, "/robots.txt", 200, None, format!("User-agent: *\nSitemap: http://{host}/list.xml\n").as_bytes());
        w.add(host, "/list.xml", 200, None, urlset(host, &["/story/1".to_string(), "/story/2".to_string()]).as_bytes());
        w.add(host, "/", 403, None, b"denied");
    }
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "gate-refuses", FollowLinks::Thin, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap();
    let links = links.expect("the refusal is kept, so the next run waits for its retry");
    assert_eq!(links["error"].as_str(), Some("http_403"));
    assert!(links["next"].as_i64().unwrap() - now >= 14 * 86_400 - 60, "{links:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robots_txt_that_refuses_us_stops_all_guessing_while_a_rate_limit_stops_none_on_a_first_visit() {
    let mut seen = Vec::new();
    for status in [403u16, 429] {
        let web: Shared = Arc::default();
        let host: &'static str = format!("robots-{status}.test").leak();
        {
            let mut w = web.lock().unwrap();
            w.add(host, "/robots.txt", status, None, b"no");
            w.add(host, "/sitemap.xml", 200, None, urlset(host, &(0..12).map(|i| format!("/story/{i}")).collect::<Vec<_>>()).as_bytes());
            w.add(host, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p></html>").as_bytes());
        }
        let (buckets, _crawl) = crawl_following(web.clone(), &[host], &format!("robots-{status}"), FollowLinks::Off, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
        let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
        let state = buckets.store(host).domain(host).unwrap().unwrap()["state"].as_str().map(String::from);
        seen.push((hits.contains(&"/sitemap.xml".to_string()), state));
    }
    assert_eq!(seen[0], (false, Some("no_sitemap".into())), "a refusal is a wall");
    assert_eq!(seen[1], (true, Some("active".into())), "a rate limit with nothing stored is guessed at, as production does");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robots_txt_rate_limit_with_a_gone_sitemap_stored_asks_only_for_that_sitemap() {
    let web: Shared = Arc::default();
    let host = "robots-429-gone.test";
    web.lock().unwrap().add(host, "/robots.txt", 429, None, b"slow down");
    let (_buckets, _crawl) = crawl_following(web.clone(), &[host], "robots-429-gone", FollowLinks::Off, Slots::new(2), |buckets| {
        buckets.allow(&Allowed::of([host]));
        let gone = format!("http://{host}/sitemap.xml");
        let mut changes = Changes::default();
        changes.sitemap(host, &gone, serde_json::json!({"id": sitemap_id(&gone), "kind": "urlset", "depth": 0, "status": "gone", "urls": 0, "next": 0}).as_object().unwrap());
        buckets.store(host).write(changes).unwrap();
    })
    .await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits, ["/robots.txt", "/sitemap.xml"], "a stored sitemap hides the guesses, as production has it");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn link_work_failing_on_our_side_waits_its_interval_before_running_again() {
    let web: Shared = Arc::default();
    let host = "links-crash-again.test";
    deep_site(&web, host);
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_adjusted(
        web.clone(),
        &[host],
        &[],
        "links-crash-again",
        FollowLinks::Thin,
        Slots::new(2),
        |buckets| buckets.allow(&Allowed::of([host])),
        |visitor| visitor.language = Arc::new(|_: &str| panic!("language detection broke")),
    )
    .await;
    let links = buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().expect("the failed run is kept");
    assert_eq!(links["status"].as_str(), Some("error"));
    assert!(links["next"].as_i64().unwrap() - now >= 7 * 86_400 - 60, "{links:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pages_slow_to_answer_never_stop_link_runs_at_their_address() {
    let slots = Slots::new(2);
    let (slow, next): (&'static str, &'static str) = ("slow-pages.test", "next-door.test");
    let web: Shared = Arc::default();
    {
        let mut w = web.lock().unwrap();
        w.add(slow, "/robots.txt", 200, None, b"User-agent: *\nAllow: /\n");
        let links: String = (1..=4).map(|i| format!("<a href=\"/s{i}\">{i}</a>")).collect();
        w.add(slow, "/", 200, None, format!("<html lang=\"en\"><p>{ENGLISH}</p>{links}</html>").as_bytes());
        for i in 1..=4 {
            w.silent.insert((slow.into(), format!("/s{i}")));
        }
    }
    deep_site(&web, next);
    // Requests here give up after 800 ms, well past the 200 ms a connection may take.
    let (_buckets, _crawl) = crawl_adjusted(web.clone(), &[slow], &[], "slow-pages", FollowLinks::Thin, slots.clone(), |b| b.allow(&Allowed::of([slow])), |visitor| {
        visitor.connect_timeout = Duration::from_millis(200);
    })
    .await;
    let (_buckets, _crawl) = crawl_following(web.clone(), &[next], "next-door", FollowLinks::Thin, slots.clone(), |b| b.allow(&Allowed::of([next]))).await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().filter(|(host, _, _)| host == next).map(|(_, path, _)| path.clone()).collect();
    assert!(hits.contains(&"/a".to_string()), "the address answered, if slowly, so its neighbour's run goes ahead: {hits:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_run_reads_the_homepage_the_visit_read_without_asking_again() {
    let web: Shared = Arc::default();
    let host = "home-once.test";
    deep_site(&web, host);
    let (buckets, _crawl) = crawl_following(web.clone(), &[host], "home-once", FollowLinks::Thin, Slots::new(2), |buckets| buckets.allow(&Allowed::of([host]))).await;
    let hits: Vec<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(hits.iter().filter(|path| *path == "/").count(), 1, "{hits:?}");
    assert!(hits.contains(&"/a".to_string()), "and the run went on from it: {hits:?}");
    assert!(buckets.store(host).sitemap(host, &format!("https://{host}/#links")).unwrap().is_some());
}

/// An active site read an hour ago, its robots.txt rules disallowing /site, with these sitemaps stored at these next times.
fn active_with(buckets: &Arc<Buckets>, host: &str, now: i64, sitemaps: &[(&str, &str, i64, i64)]) {
    let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::Active, None, Some(0));
    for (name, at) in [("robots", now - 3600), ("checked", now - 3600), ("ok", now - 3600)] {
        record.insert(name.into(), at.into());
    }
    record.insert("lang".into(), "en".into());
    record.insert("allows".into(), true.into());
    record.insert("rules".into(), serde_json::json!({"allow": [], "disallow": ["/site"]}));
    let mut changes = Changes::default();
    changes.domain(host, &record);
    for (path, kind, depth, next) in sitemaps {
        let url = format!("http://{host}{path}");
        changes.sitemap(host, &url, serde_json::json!({"id": sitemap_id(&url), "kind": kind, "depth": depth, "status": "ok", "urls": 12, "next": next, "interval": 86_400}).as_object().unwrap());
    }
    buckets.store(host).write(changes).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_due_sitemap_robots_txt_disallows_sets_no_due_time() {
    let web: Shared = Arc::default();
    let host = "disallowed-sitemap.test";
    let now = chrono::Utc::now().timestamp();
    let (buckets, _crawl) = crawl_prepared(web.clone(), &[host], "disallowed-sitemap", |buckets| {
        active_with(buckets, host, now, &[("/sitemap.xml", "urlset", 0, now - 30 * 86_400), ("/news.xml", "urlset", 0, now + 6 * 3600)]);
    })
    .await;
    assert!(web.lock().unwrap().hits.is_empty(), "/sitemap.xml stays unread while robots.txt disallows it, and /news.xml is not due");
    let due = buckets.store(host).domain(host).unwrap().unwrap()["due"].as_i64().unwrap();
    assert!((due - (now + 6 * 3600)).abs() <= 120, "due when /news.xml is, not at once and not an hour on: due in {} s", due - now);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backlog_of_due_children_is_read_a_visit_at_a_time_until_none_is_due() {
    let web: Shared = Arc::default();
    let host = "big-backlog.test";
    let children: Vec<String> = (0..100).map(|i| format!("/child-{i:03}.xml")).collect();
    {
        let mut w = web.lock().unwrap();
        for (i, child) in children.iter().enumerate() {
            w.add(host, child, 200, None, urlset(host, &[format!("/page-{i}")]).as_bytes());
        }
    }
    let now = chrono::Utc::now().timestamp();
    let mut stored: Vec<(&str, &str, i64, i64)> = vec![("/index.xml", "index", 0, now + 86_400)];
    stored.extend(children.iter().map(|child| (child.as_str(), "urlset", 1, now - 86_400)));
    let (buckets, mut crawl) = crawl_prepared(web.clone(), &[host], "big-backlog", |buckets| active_with(buckets, host, now, &stored)).await;
    let mut read = web.lock().unwrap().hits.len();
    assert_eq!(read, 40, "a visit reads 40 files at most");
    for _ in 0..2 {
        assert!(buckets.store(host).domain(host).unwrap().unwrap()["due"].as_i64().unwrap() <= chrono::Utc::now().timestamp(), "due again at once while children wait");
        crawl.load().unwrap();
        crawl.step().await.unwrap();
        let now_read = web.lock().unwrap().hits.len();
        assert!(now_read > read, "each visit reads more of the backlog");
        read = now_read;
    }
    let paths: HashSet<String> = web.lock().unwrap().hits.iter().map(|(_, path, _)| path.clone()).collect();
    assert_eq!(paths.len(), 100, "every child read once, none twice");
    let due = buckets.store(host).domain(host).unwrap().unwrap()["due"].as_i64().unwrap();
    assert!(due > chrono::Utc::now().timestamp() + 3600, "and the site rests once none is due");
}

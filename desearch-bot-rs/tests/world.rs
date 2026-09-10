//! The crawl loop end to end against a small fake web.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use desearch_bot::buckets::{bucket_of, Buckets, Changes, Resources};
use desearch_bot::crawl::Loop;
use desearch_bot::records;
use desearch_bot::registry::Registry;
use desearch_bot::states::State;
use desearch_bot::suffixes::PublicSuffixList;
use desearch_bot::urls;
use desearch_bot::visit::Visitor;
use hickory_resolver::error::ResolveError;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

const ENGLISH: &str = "The quick brown fox jumps over the lazy dog while the committee reviews the annual budget \
    report. Readers who follow the local news will find stories about schools, transport, weather and the people \
    who make this town a better place to live and work every single day of the year.";

#[derive(Clone)]
struct Page {
    status: u16,
    etag: Option<&'static str>,
    body: Vec<u8>,
}

#[derive(Default)]
struct Web {
    pages: HashMap<(String, String), Page>,
    hits: Vec<(String, String, Option<String>)>,
}

impl Web {
    fn add(&mut self, host: &str, path: &str, status: u16, etag: Option<&'static str>, body: &[u8]) {
        self.pages.insert((host.into(), path.into()), Page { status, etag, body: body.to_vec() });
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
                let page = {
                    let mut web = web.lock().unwrap();
                    web.hits.push((host.clone(), path.to_string(), validator.clone()));
                    web.pages.get(&(host, path.to_string())).cloned()
                };
                let page = page.unwrap_or(Page { status: 404, etag: None, body: b"not found".to_vec() });
                let (status, body) =
                    if page.etag.is_some() && validator.as_deref() == page.etag { (304, Vec::new()) } else { (page.status, page.body) };
                let mut response = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\n", body.len());
                if let Some(etag) = page.etag {
                    response.push_str(&format!("ETag: {etag}\r\n"));
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
        w.add("site.test", "/robots.txt", 200, None, b"User-agent: *\nAllow: /\nSitemap: http://site.test/sitemap_index.xml\n");
        w.add(
            "site.test",
            "/sitemap_index.xml",
            200,
            None,
            b"<sitemapindex><sitemap><loc>http://site.test/posts.xml.gz</loc><lastmod>2024-01-01</lastmod></sitemap>\
              <sitemap><loc>/pages.xml</loc></sitemap></sitemapindex>",
        );
        let posts: Vec<String> = (1..=12).map(|i| format!("/post/{i}")).collect();
        w.add("site.test", "/posts.xml.gz", 200, None, &gzip(urlset("site.test", &posts).as_bytes()));
        let pages = urlset("site.test", &["/".into(), "/about".into(), "/contact".into()])
            .replace("</urlset>", "<url><loc>http://elsewhere.test/x</loc></url></urlset>");
        w.add("site.test", "/pages.xml", 200, Some("\"v1\""), pages.as_bytes());
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
        cpu: Arc::new(Semaphore::new(4)),
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), 16, Registry::offline(), HashSet::new());
    assert_eq!(crawl.load().unwrap(), 4);
    assert_eq!(crawl.step().await.unwrap(), 4);

    let domain = |host: &str| buckets.store(host).domain(host).unwrap().unwrap();
    let site = domain("site.test");
    assert_eq!((site["state"].as_str(), site["urls"].as_i64(), site["lang"].as_str()), (Some("active"), Some(15), Some("en")));
    assert_eq!(buckets.store("site.test").sitemaps("site.test").unwrap().len(), 3);
    let stored = |url: &str| buckets.store("site.test").url(&urls::parse(url, "site.test").unwrap()).unwrap();
    assert!(stored("http://site.test/post/12").is_some() && stored("http://site.test/about").is_some());
    assert_eq!(domain("blocked.test")["reason"], "robots_disallow");
    assert_eq!((domain("empty.test")["state"].as_str(), domain("empty.test")["reason"].as_str()), (Some("no_sitemap"), Some("no_sitemap")));
    assert_eq!(domain("gone.test")["reason"], "ClientConnectorDNSError");

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

//! URLs as the Python crawler's `links` module reads them: normalizing, crawl scope, a page's links and robots meta, sitemap entries.

use std::cell::RefCell;
use std::collections::HashSet;
use std::io::Read;
use std::sync::LazyLock;

use html5ever::tendril::StrTendril;
use html5ever::tokenizer::states::RawKind;
use html5ever::tokenizer::{BufferQueue, Tag, TagKind, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts};
use quick_xml::events::Event;
use quick_xml::name::ResolveResult;
use regex::Regex;

use super::Crash;
use crate::pystr;
use crate::urls;

pub const MAX_SITEMAP_BYTES: usize = 64 << 20;
const SITEMAP_NS: &[u8] = b"http://www.sitemaps.org/schemas/sitemap/0.9";

static TRACKING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:utm_|(?:fbclid|gclid|mc_cid|mc_eid|ref|cmpid|ito)\n?$)").unwrap());
static SKIP_EXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\.(jpe?g|png|gif|webp|svg|ico|bmp|tiff?|pdf|zip|gz|tgz|bz2|xz|7z|rar|tar|mp[34]|m4[av]|mov|avi|wmv|webm|ogg|wav|flac|exe|dmg|msi|apk|iso|bin|css|js|json|xml|rss|atom|woff2?|ttf|eot|docx?|xlsx?|pptx?|odt|csv)\n?$",
    )
    .unwrap()
});
static XML_ENCODING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"^<\?xml[^>]+[\s\x1c-\x1f]+encoding[\s\x1c-\x1f]*=[\s\x1c-\x1f]*["'][^"']*["']"#).unwrap());
static XML_DECLARED: LazyLock<regex::bytes::Regex> =
    LazyLock::new(|| regex::bytes::Regex::new(r#"(?-u)^<\?xml[^>]*\sencoding\s*=\s*["']([A-Za-z0-9._:-]+)["']"#).unwrap());

const USES_PARAMS: [&str; 16] = ["", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu", "sip", "sips", "mms", "sftp", "tel"];

/// `urlparse(url, scheme)` as six parts, empty where Python has empty strings.
pub fn urlparse(url: &str, scheme: &str) -> Result<[String; 6], urls::Invalid> {
    let p = urls::split(url)?;
    let scheme = p.scheme.clone().unwrap_or_else(|| scheme.to_string());
    let (mut path, mut params) = (p.path.clone(), String::new());
    if USES_PARAMS.contains(&scheme.as_str()) && path.contains(';') {
        let at = match path.rfind('/') {
            Some(slash) => path[slash..].find(';').map(|i| slash + i),
            None => path.find(';'),
        };
        if let Some(at) = at {
            params = path[at + 1..].to_string();
            path.truncate(at);
        }
    }
    Ok([scheme, p.netloc.unwrap_or_default(), path, params, p.query.unwrap_or_default(), p.fragment.unwrap_or_default()])
}

/// Python 3.12's `urlunsplit`.
fn urlunsplit(scheme: &str, netloc: &str, path: &str, query: &str, fragment: &str) -> String {
    let mut url = path.to_string();
    if !netloc.is_empty() {
        if !url.is_empty() && !url.starts_with('/') {
            url.insert(0, '/');
        }
        url = format!("//{netloc}{url}");
    } else if url.starts_with("//") || (!scheme.is_empty() && urls::USES_NETLOC.contains(&scheme) && (url.is_empty() || url.starts_with('/'))) {
        url = format!("//{url}");
    }
    if !scheme.is_empty() {
        url = format!("{scheme}:{url}");
    }
    if !query.is_empty() {
        url = format!("{url}?{query}");
    }
    if !fragment.is_empty() {
        url = format!("{url}#{fragment}");
    }
    url
}

fn urlunparse(scheme: &str, netloc: &str, path: &str, params: &str, query: &str, fragment: &str) -> String {
    let path = if params.is_empty() { path.to_string() } else { format!("{path};{params}") };
    urlunsplit(scheme, netloc, &path, query, fragment)
}

/// Python 3.12's `urllib.parse.urljoin`, which parses `;params` off the last segment and drops empty parts.
pub fn urljoin(base: &str, url: &str) -> Result<String, urls::Invalid> {
    if base.is_empty() {
        return Ok(url.to_string());
    }
    if url.is_empty() {
        return Ok(base.to_string());
    }
    let [bscheme, bnetloc, bpath, bparams, bquery, _] = urlparse(base, "")?;
    let [scheme, mut netloc, mut path, mut params, mut query, fragment] = urlparse(url, &bscheme)?;
    if scheme != bscheme || !urls::USES_RELATIVE.contains(&scheme.as_str()) {
        return Ok(url.to_string());
    }
    if urls::USES_NETLOC.contains(&scheme.as_str()) {
        if !netloc.is_empty() {
            return Ok(urlunparse(&scheme, &netloc, &path, &params, &query, &fragment));
        }
        netloc = bnetloc;
    }
    if path.is_empty() && params.is_empty() {
        (path, params) = (bpath, bparams);
        if query.is_empty() {
            query = bquery;
        }
        return Ok(urlunparse(&scheme, &netloc, &path, &params, &query, &fragment));
    }
    let mut base_parts: Vec<&str> = bpath.split('/').collect();
    if base_parts.last() != Some(&"") {
        base_parts.pop();
    }
    let segments: Vec<&str> = if path.starts_with('/') {
        path.split('/').collect()
    } else {
        let all: Vec<&str> = base_parts.into_iter().chain(path.split('/')).collect();
        let last = all.len() - 1;
        all.iter().enumerate().filter(|(i, s)| *i == 0 || *i == last || !s.is_empty()).map(|(_, s)| *s).collect()
    };
    let mut resolved: Vec<&str> = Vec::new();
    for seg in &segments {
        match *seg {
            ".." => {
                resolved.pop();
            }
            "." => {}
            other => resolved.push(other),
        }
    }
    if matches!(segments.last(), Some(&".") | Some(&"..")) {
        resolved.push("");
    }
    let joined = resolved.join("/");
    Ok(urlunparse(&scheme, &netloc, if joined.is_empty() { "/" } else { &joined }, &params, &query, &fragment))
}

/// Python's `urllib.parse.parse_qsl(query, keep_blank_values=True)`.
pub fn parse_qsl(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            (pystr::unquote(&name.replace('+', " ")), pystr::unquote(&value.replace('+', " ")))
        })
        .collect()
}

/// Python's `urllib.parse.quote_plus(text, safe="")`.
pub fn quote_plus(text: &str) -> String {
    if text.contains(' ') {
        super::robots::quote(text, " ").replace(' ', "+")
    } else {
        super::robots::quote(text, "")
    }
}

/// Python's `urlencode` of (name, value) pairs.
pub fn urlencode(pairs: &[(String, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v))).collect::<Vec<_>>().join("&")
}

/// An absolute http(s) URL without fragment, default port or tracking parameters.
pub fn normalize(url: &str, base: &str) -> Option<String> {
    let joined = urljoin(base, pystr::strip(url)).ok()?;
    let parts = urls::split(&joined).ok()?;
    let port = parts.port().ok()?;
    let scheme = parts.scheme.as_deref().unwrap_or("");
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let mut host = parts.hostname().filter(|h| !h.is_empty())?.to_lowercase();
    let default = if scheme == "http" { 80 } else { 443 };
    if let Some(port) = port.filter(|&p| p != 0 && p != default) {
        host = format!("{host}:{port}");
    }
    let kept: Vec<(String, String)> = parse_qsl(parts.query.as_deref().unwrap_or("")).into_iter().filter(|(k, _)| !TRACKING.is_match(k)).collect();
    let query = urlencode(&kept);
    let path = if parts.path.is_empty() { "/" } else { parts.path.as_str() };
    let mut out = format!("{scheme}://{host}");
    if !path.starts_with('/') {
        out.push('/');
    }
    out.push_str(path);
    if !query.is_empty() {
        out.push('?');
        out.push_str(&query);
    }
    Some(out)
}

pub fn host_of(url: &str) -> String {
    urls::split(url).map(|p| p.netloc().to_lowercase()).unwrap_or_default()
}

pub fn in_scope(url: &str, host: &str, prefixes: &[String]) -> bool {
    let Ok(parts) = urls::split(url) else { return false };
    if parts.netloc().to_lowercase() != host || SKIP_EXT.is_match(&parts.path) {
        return false;
    }
    prefixes.is_empty() || prefixes.iter().any(|p| parts.path.starts_with(p.as_str()))
}

/// What a page's markup says, as lxml's tree of it shows: the root's `lang`, the first `<base>`, every `<meta>` and `<a>`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Doc {
    pub lang: Option<String>,
    pub base: Option<Option<String>>,
    pub metas: Vec<(Option<String>, Option<String>)>,
    pub anchors: Vec<(Option<String>, Option<String>)>,
}

#[derive(Default)]
struct Reading {
    doc: Doc,
    rooted: bool,
    done: bool,
}

struct Sink(RefCell<Reading>);

fn attr(tag: &Tag, name: &str) -> Option<String> {
    tag.attrs.iter().find(|a| &*a.name.local == name).map(|a| a.value.to_string())
}

impl TokenSink for Sink {
    type Handle = ();

    fn process_token(&self, token: Token, _line: u64) -> TokenSinkResult<()> {
        let mut r = self.0.borrow_mut();
        if r.done {
            return TokenSinkResult::Continue;
        }
        match token {
            Token::TagToken(tag) if tag.kind == TagKind::StartTag => {
                let name = &*tag.name;
                if name == "html" && !r.rooted {
                    r.doc.lang = attr(&tag, "lang");
                }
                r.rooted = true;
                match name {
                    "base" if r.doc.base.is_none() => r.doc.base = Some(attr(&tag, "href")),
                    "meta" => r.doc.metas.push((attr(&tag, "name"), attr(&tag, "content"))),
                    "a" => r.doc.anchors.push((attr(&tag, "href"), attr(&tag, "rel"))),
                    _ => {}
                }
                if tag.self_closing {
                    return TokenSinkResult::Continue;
                }
                return match name {
                    "script" => TokenSinkResult::RawData(RawKind::ScriptData),
                    "style" | "xmp" | "iframe" | "noembed" | "noframes" => TokenSinkResult::RawData(RawKind::Rawtext),
                    "title" | "textarea" => TokenSinkResult::RawData(RawKind::Rcdata),
                    "plaintext" => TokenSinkResult::Plaintext,
                    _ => TokenSinkResult::Continue,
                };
            }
            Token::TagToken(tag) if &*tag.name == "html" && r.rooted => r.done = true,
            Token::CharacterTokens(text) if text.chars().any(|c| !matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c')) => r.rooted = true,
            Token::NullCharacterToken => r.rooted = true,
            _ => {}
        }
        TokenSinkResult::Continue
    }
}

/// lxml's `document_fromstring`, or None where the crawler gets no document: blank text, an XML encoding declaration, or no element.
pub fn parse_html(text: &str) -> Option<Doc> {
    if pystr::strip(text).is_empty() || XML_ENCODING.is_match(text) {
        return None;
    }
    let tokenizer = Tokenizer::new(Sink(RefCell::new(Reading::default())), TokenizerOpts::default());
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(text));
    let _ = tokenizer.feed(&input);
    tokenizer.end();
    let reading = tokenizer.sink.0.into_inner();
    reading.rooted.then_some(reading.doc)
}

/// The page's robots directives for us: `<meta name=robots|desearchbot>` and X-Robots-Tag.
pub fn robots_meta(doc: &Doc, header: &str) -> HashSet<String> {
    let mut found: HashSet<String> = header.split(',').map(pystr::strip).filter(|d| !d.is_empty()).map(str::to_lowercase).collect();
    for (name, content) in &doc.metas {
        let name = name.as_deref().unwrap_or("").to_lowercase();
        if name == "robots" || name == "desearchbot" {
            found.extend(content.as_deref().unwrap_or("").split(',').map(|d| pystr::strip(d).to_lowercase()));
        }
    }
    if found.contains("none") {
        found.insert("noindex".into());
        found.insert("nofollow".into());
    }
    found
}

/// Links a page offers to follow, in page order, without rel=nofollow ones.
pub fn page_links(doc: &Doc, url: &str) -> Result<Vec<String>, Crash> {
    let base_url = match doc.base.as_ref().and_then(|href| href.as_deref()).filter(|href| !href.is_empty()) {
        Some(href) => urljoin(url, href).map_err(|_| Crash("ValueError"))?,
        None => url.to_string(),
    };
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (href, rel) in &doc.anchors {
        let Some(href) = href.as_deref().filter(|h| !h.is_empty()) else { continue };
        if pystr::split_ws(&rel.as_deref().unwrap_or("").to_lowercase()).any(|r| r == "nofollow") {
            continue;
        }
        if let Some(u) = normalize(href, &base_url) {
            if seen.insert(u.clone()) {
                out.push(u);
            }
        }
    }
    Ok(out)
}

/// The body gunzipped up to the sitemap limit, as `zlib.decompressobj(16 + MAX_WBITS).decompress(body, limit)`.
fn gunzip(body: &[u8]) -> Result<Vec<u8>, Crash> {
    let mut out = Vec::new();
    let mut reader = flate2::read::GzDecoder::new(body).take(MAX_SITEMAP_BYTES as u64);
    match reader.read_to_end(&mut out) {
        Ok(_) => Ok(out),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(out),
        Err(_) => Err(Crash("error")),
    }
}

fn utf8_body(body: &[u8]) -> Option<String> {
    if let Some(rest) = body.strip_prefix(b"\xef\xbb\xbf") {
        return String::from_utf8(rest.to_vec()).ok();
    }
    if body.starts_with(b"\xff\xfe") || body.starts_with(b"\xfe\xff") {
        let (text, _, bad) = encoding_rs::Encoding::for_bom(body)?.0.decode(body);
        return (!bad).then(|| text.into_owned());
    }
    let declared = XML_DECLARED.captures(body).and_then(|c| encoding_rs::Encoding::for_label(&c[1]));
    match declared {
        Some(e) if e != encoding_rs::UTF_8 => {
            let (text, bad) = e.decode_without_bom_handling(body);
            (!bad).then(|| text.into_owned())
        }
        _ => String::from_utf8(body.to_vec()).ok(),
    }
}

fn reference(name: &[u8]) -> Option<char> {
    let name = std::str::from_utf8(name).ok()?;
    let number = |digits: &str, radix| u32::from_str_radix(digits, radix).ok().and_then(char::from_u32);
    let c = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        _ => match name.strip_prefix("#x") {
            Some(hex) if !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()) => number(hex, 16)?,
            _ => {
                let digits = name.strip_prefix('#')?;
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                number(digits, 10)?
            }
        },
    };
    let xml_char = matches!(c, '\t' | '\n' | '\r') || ('\u{20}'..='\u{d7ff}').contains(&c) || ('\u{e000}'..='\u{fffd}').contains(&c) || c >= '\u{10000}';
    xml_char.then_some(c)
}

/// The `<loc>` texts of a well-formed sitemap, every one in document order; None where lxml would refuse the file.
fn sitemap_locs(text: &str) -> Option<(String, Vec<String>)> {
    let mut reader = quick_xml::NsReader::from_str(text);
    reader.config_mut().check_end_names = true;
    let (mut depth, mut root, mut closed) = (0usize, None::<String>, false);
    let mut locs = Vec::new();
    let mut open: Option<String> = None;
    let mut started = false;
    loop {
        let (ns, event) = reader.read_resolved_event().ok()?;
        if matches!(event, Event::Decl(_)) && started {
            return None;
        }
        started = true;
        match event {
            Event::Start(ref e) | Event::Empty(ref e) => {
                if closed || (depth == 0 && root.is_some()) {
                    return None;
                }
                for a in e.attributes().with_checks(true) {
                    a.ok()?.normalized_value(quick_xml::XmlVersion::Implicit1_0).ok()?;
                }
                let local = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                if root.is_none() {
                    root = Some(local.clone());
                }
                if let Some(text) = open.take() {
                    locs.push(text);
                }
                let ours = matches!(ns, ResolveResult::Unbound) || matches!(ns, ResolveResult::Bound(n) if n.as_ref() == SITEMAP_NS);
                let is_loc = ours && local == "loc";
                if matches!(event, Event::Start(_)) {
                    depth += 1;
                    if is_loc {
                        open = Some(String::new());
                    }
                } else if is_loc {
                    locs.push(String::new());
                }
                if depth == 0 {
                    closed = true;
                }
            }
            Event::End(_) => {
                if let Some(text) = open.take() {
                    locs.push(text);
                }
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    closed = true;
                }
            }
            Event::Text(t) => {
                let raw = t.decode().ok()?;
                if depth == 0 && !raw.chars().all(|c| matches!(c, ' ' | '\t' | '\n' | '\r')) {
                    return None;
                }
                if let Some(text) = open.as_mut() {
                    text.push_str(&raw);
                }
            }
            Event::CData(t) => {
                if depth == 0 {
                    return None;
                }
                let raw = t.decode().ok()?;
                if let Some(text) = open.as_mut() {
                    text.push_str(&raw);
                }
            }
            Event::GeneralRef(r) => {
                if depth == 0 {
                    return None;
                }
                let c = reference(r.as_ref())?;
                if let Some(text) = open.as_mut() {
                    text.push(c);
                }
            }
            Event::Comment(_) | Event::PI(_) => {
                if let Some(text) = open.take() {
                    locs.push(text);
                }
            }
            Event::Decl(_) | Event::DocType(_) => {}
            Event::Eof => break,
        }
    }
    if depth != 0 || root.is_none() {
        return None;
    }
    Some((root.unwrap_or_default(), locs))
}

/// (page URLs, child sitemap URLs) of a sitemap or sitemap index, gzipped or not.
pub fn sitemap_entries(body: &[u8]) -> Result<(Vec<String>, Vec<String>), Crash> {
    let body = if body.starts_with(&[0x1f, 0x8b]) { gunzip(body)? } else { body.to_vec() };
    let Some((root, locs)) = utf8_body(&body).and_then(|text| sitemap_locs(&text)) else {
        return Ok((Vec::new(), Vec::new()));
    };
    let locs: Vec<String> = locs.iter().map(|l| pystr::strip(l).to_string()).filter(|l| !l.is_empty()).collect();
    Ok(if root.ends_with("sitemapindex") { (Vec::new(), locs) } else { (locs, Vec::new()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn links(html: &str, url: &str) -> Vec<String> {
        page_links(&parse_html(html).unwrap(), url).unwrap()
    }

    #[test]
    fn normalize_and_scope() {
        assert_eq!(normalize("/a/b?utm_source=x&id=2#top", "https://Ex.test:443/").as_deref(), Some("https://ex.test/a/b?id=2"));
        assert_eq!(normalize("mailto:x@ex.test", ""), None);
        assert_eq!(normalize("http://ex.test:8080/", "").as_deref(), Some("http://ex.test:8080/"));
        assert_eq!(normalize("https://a.test: Managing Supply/x", ""), None);
        assert_eq!(normalize("https://a.test:99999/", ""), None);
        assert_eq!(normalize("https://a.test:8080/x#y", "").as_deref(), Some("https://a.test:8080/x"));
        assert_eq!(normalize(" https://u:p@A.test/x?b=1+2&a=%41&e&ref=z&REF=y&gclid=1&q=a%2Fb ", "").as_deref(), Some("https://a.test/x?b=1+2&a=A&e=&q=a%2Fb"));
        assert_eq!(normalize("?x=é", "https://a.test/p").as_deref(), Some("https://a.test/p?x=%C3%A9"));
        assert_eq!(normalize("https://a.test", "").as_deref(), Some("https://a.test/"));
        let none: [String; 0] = [];
        assert!(in_scope("https://ex.test/doc/a", "ex.test", &["/doc/".into()]));
        assert!(!in_scope("https://ex.test/blog/a", "ex.test", &["/doc/".into()]));
        assert!(!in_scope("https://ex.test/a.PDF", "ex.test", &none) && !in_scope("https://other.test/a", "ex.test", &none));
    }

    #[test]
    fn links_and_robots_meta() {
        let html = r#"<html><head><base href="/docs/"><meta name="robots" content="noindex"></head><body><a href="a">a</a><a href="b" rel="nofollow">b</a><a href="a#x">again</a><a href="https://other.test/c">c</a></body></html>"#;
        assert_eq!(links(html, "https://ex.test/start"), ["https://ex.test/docs/a", "https://other.test/c"]);
        assert_eq!(robots_meta(&parse_html(html).unwrap(), ""), HashSet::from(["noindex".to_string()]));
        let none = robots_meta(&parse_html("<p>x</p>").unwrap(), "none");
        assert!(none.contains("noindex") && none.contains("nofollow"));
    }

    #[test]
    fn markup_reads_as_libxml2_does() {
        for empty in [
            "",
            "  \n",
            "<!-- x -->",
            "<!DOCTYPE html>",
            "</p>",
            "</a href=e>",
            "<!doctype html> </html>",
            "<?php x ?>",
            "<?xml version='1.0' encoding='utf-8'?><a href=x>",
        ] {
            assert_eq!(parse_html(empty), None, "{empty:?}");
        }
        assert!(parse_html(" <?xml version='1.0' encoding='utf-8'?><a href=x>").is_some());
        let lang = |html: &str| parse_html(html).unwrap().lang;
        assert_eq!(lang("<!-- c --><!doctype html>\n<HTML LANG=fr>"), Some("fr".into()));
        assert_eq!(lang("\u{feff}<html lang=en>"), Some("en".into()));
        assert_eq!(lang("<p>x</p><html lang=fr>"), None);
        assert_eq!(lang("hello<html lang=fr>"), None);
        assert_eq!(lang("<html lang=en><html lang=fr>"), Some("en".into()));
        assert_eq!(lang("<html xml:lang=fr>"), None);
        let hrefs = |html: &str| parse_html(html).unwrap().anchors.into_iter().map(|(h, _)| h.unwrap_or_default()).collect::<Vec<_>>();
        assert_eq!(
            hrefs("<script><a href=s></script><noscript><a href=ns></noscript><style><a href=st></style><title><a href=t></title><a href=1>"),
            ["ns", "1"]
        );
        assert_eq!(hrefs("<a href='a&amp;b&nbsp;c&notit;d&#x41;&#0;&ampx&lt'>"), ["a&b\u{a0}c&notit;dA\u{fffd}&ampx<"]);
        assert_eq!(hrefs("<a href=1></body><a href=2></html><a href=3>"), ["1", "2"]);
        assert_eq!(hrefs("</html><a href=2>"), ["2"]);
        assert_eq!(hrefs("<a HREF=X href=y><a/href=z><svg><a href=sv></a></svg><![CDATA[<a href=cd>]]><a href=after>"), ["X", "z", "sv", "after"]);
        assert_eq!(parse_html("<base><base href=/b2>").unwrap().base, Some(None));
    }

    #[test]
    fn sitemaps_plain_index_and_gzip() {
        let urlset = br#"<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc> https://ex.test/a </loc></url><url><loc>https://ex.test/b?x=1&amp;y=2</loc></url></urlset>"#;
        assert_eq!(sitemap_entries(urlset).unwrap(), (vec!["https://ex.test/a".to_string(), "https://ex.test/b?x=1&y=2".to_string()], vec![]));
        let index =
            br#"<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://ex.test/s1.xml.gz</loc></sitemap></sitemapindex>"#;
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gz, index).unwrap();
        assert_eq!(sitemap_entries(&gz.finish().unwrap()).unwrap(), (vec![], vec!["https://ex.test/s1.xml.gz".to_string()]));
        for bad in [
            &b"not xml"[..],
            b"",
            b"<urlset><url><loc>https://ex.test/a?x&y</loc></url></urlset>",
            b"<urlset><loc>a</loc>",
            b"<urlset></urlset><x/>",
            b"<urlset><loc>a</lc></urlset>",
        ] {
            assert_eq!(sitemap_entries(bad).unwrap(), (vec![], vec![]), "{}", String::from_utf8_lossy(bad));
        }
        let mixed = br#"<urlset xmlns:i="urn:img"><url><loc>a<!-- c -->b</loc><i:loc>img</i:loc><loc><![CDATA[c]]>d</loc></url></urlset>"#;
        assert_eq!(sitemap_entries(mixed).unwrap().0, ["a", "cd"]);
        assert!(sitemap_entries(b"\x1f\x8bnot gzip").is_err());
    }
}

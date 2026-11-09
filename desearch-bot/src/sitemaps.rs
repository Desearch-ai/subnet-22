//! Parse sitemap files into the addresses and dates they list.

use std::sync::LazyLock;

use memchr::memchr;
use regex::bytes::Regex;

static SITEMAP_INDEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)<sitemapindex").unwrap());
static NEWS_NAMESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)sitemap-news/0\.9").unwrap());
static NEWS_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i-u)\bxmlns:([a-z_][a-z0-9_.-]*)\s*=\s*["']https?://www\.google\.com/schemas/sitemap-news/0\.9/?["']"#).unwrap());
/// RSS 2.0, RSS 1.0 and Atom, which the sitemap protocol takes as sitemaps.
static FEED_ROOT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)<(rss|rdf:RDF|feed)\b").unwrap());
static FEED_ITEM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is-u)<item\b[^>]*>(.*?)</item>|<entry\b[^>]*>(.*?)</entry>").unwrap());
static RSS_LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is-u)<link>\s*(?:<!\[CDATA\[)?\s*([^<\s\]]+)").unwrap());
static RSS_GUID_LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?is-u)<guid[^>]*isPermaLink=["']true["'][^>]*>\s*([^<\s]+)"#).unwrap());
static RSS_DATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is-u)<(pubDate|dc:date)>\s*([^<]+?)\s*</").unwrap());
static ATOM_LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is-u)<link\b([^>]*)>").unwrap());
static ATOM_ATTR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?is-u)\b(rel|href)\s*=\s*["']([^"']*)["']"#).unwrap());
static ATOM_DATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is-u)<(updated|published)>\s*([^<\s]+)\s*</").unwrap());

const FIELDS: [&[u8]; 4] = [b"loc", b"lastmod", b"changefreq", b"news:publication_date"];
const CDATA: &[u8] = b"<![CDATA[";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub url: String,
    pub lastmod: Option<String>,
    pub changefreq: Option<String>,
    pub published: Option<String>,
}

/// The file kind and its entries, pairing each location with its own dates.
pub fn parse_entries(body: &[u8]) -> (&'static str, Vec<Entry>) {
    let mut entries = Vec::new();
    let mut kind = None;
    // Once a closing tag is missing from some point on, no later opening tag can find one either.
    let mut unclosed = [false; 2];
    let mut names = None;
    let mut at = 0;
    while let Some(found) = memchr(b'<', &body[at..]) {
        let start = at + found;
        at = start + 1;
        let (which, name, closing): (usize, usize, &[u8]) = if starts_with(body, start + 1, b"url") {
            (0, 3, b"</url>")
        } else if starts_with(body, start + 1, b"sitemap") {
            (1, 7, b"</sitemap>")
        } else {
            continue;
        };
        let open_end = start + 1 + name;
        if unclosed[which] || body.get(open_end).is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_') {
            continue;
        }
        let Some(close) = find(body, open_end, closing) else {
            unclosed[which] = true;
            continue;
        };
        kind.get_or_insert(if which == 0 { "urlset" } else { "index" });
        let names = names.get_or_insert_with(|| field_names(body));
        entries.extend(entry(&body[open_end..close], which == 1, names));
        at = close + closing.len();
    }
    if let (Some(kind), false) = (kind, entries.is_empty()) {
        return (kind, entries);
    }
    let locations = bare_locations(body);
    if locations.is_empty() {
        let items = feed_entries(body);
        return if items.is_empty() { ("invalid", Vec::new()) } else { ("urlset", items) };
    }
    let kind = if SITEMAP_INDEX.is_match(&body[..body.len().min(4096)]) { "index" } else { "urlset" };
    (kind, locations)
}

/// The fields an entry is read for: `FIELDS`, then the news date under each other prefix the file binds to the news sitemap namespace.
fn field_names(body: &[u8]) -> Vec<Vec<u8>> {
    let mut names: Vec<Vec<u8>> = FIELDS.iter().map(|name| name.to_vec()).collect();
    for bound in NEWS_PREFIX.captures_iter(&body[..body.len().min(4096)]) {
        let name = [&bound[1], b":publication_date"].concat();
        if !names.iter().any(|known| known.eq_ignore_ascii_case(&name)) {
            names.push(name);
        }
    }
    names
}

/// Python's `<(loc|lastmod|changefreq|news:publication_date)>\s*([^<\s]*)\s*</` over `names`, first value of each kept; with `every`, each location is an entry with the dates.
fn entry(inner: &[u8], every: bool, names: &[Vec<u8>]) -> Vec<Entry> {
    let mut fields: [Option<&[u8]>; 4] = [None; 4];
    let mut locations = Vec::new();
    let mut at = 0;
    while let Some(found) = memchr(b'<', &inner[at..]) {
        let start = at + found;
        at = start + 1;
        let Some(named) = names.iter().position(|name| starts_with(inner, start + 1, name) && inner.get(start + 1 + name.len()) == Some(&b'>')) else {
            continue;
        };
        let slot = named.min(FIELDS.len() - 1);
        let (value, end) = value(inner, start + 2 + names[named].len());
        if !starts_with(inner, end, b"</") {
            continue;
        }
        at = end + 2;
        if !value.is_empty() {
            fields[slot].get_or_insert(value);
            if slot == 0 && (every || locations.is_empty()) {
                locations.push(value);
            }
        }
    }
    locations
        .into_iter()
        .map(|url| Entry { url: location(url), lastmod: fields[1].map(lossy), changefreq: fields[2].map(ascii_lower), published: fields[3].map(lossy) })
        .collect()
}

/// The items of an RSS or Atom feed: each one's page and date, as a sitemap would list them.
fn feed_entries(body: &[u8]) -> Vec<Entry> {
    if !FEED_ROOT.is_match(&body[..body.len().min(4096)]) {
        return Vec::new();
    }
    let mut entries = Vec::new();
    for item in FEED_ITEM.captures_iter(body) {
        let (url, date) = match (item.get(1), item.get(2)) {
            (Some(rss), _) => {
                let rss = rss.as_bytes();
                let link = RSS_LINK.captures(rss).or_else(|| RSS_GUID_LINK.captures(rss)).map(|c| location(&c[1]));
                let date = RSS_DATE.captures(rss).and_then(|c| feed_date(&lossy(&c[2])));
                (link, date)
            }
            (None, Some(atom)) => {
                let atom = atom.as_bytes();
                let link = ATOM_LINK.captures_iter(atom).find_map(|tag| {
                    let attrs: Vec<(String, Vec<u8>)> = ATOM_ATTR.captures_iter(&tag[1]).map(|a| (ascii_lower(&a[1]), a[2].to_vec())).collect();
                    let alternate = attrs.iter().all(|(name, value)| name != "rel" || value.eq_ignore_ascii_case(b"alternate"));
                    attrs.iter().find(|(name, _)| name == "href").filter(|_| alternate).map(|(_, href)| location(href))
                });
                let date = ATOM_DATE.captures(atom).and_then(|c| feed_date(&lossy(&c[2])));
                (link, date)
            }
            (None, None) => continue,
        };
        if let Some(url) = url.filter(|url| !url.is_empty()) {
            entries.push(Entry { url, lastmod: date, ..Entry::default() });
        }
    }
    entries
}

/// A feed's date as ISO 8601: RSS dates are RFC 2822, Atom's and Dublin Core's are ISO already.
fn feed_date(text: &str) -> Option<String> {
    let text = text.trim();
    match chrono::DateTime::parse_from_rfc2822(text) {
        Ok(date) => Some(date.to_rfc3339()),
        Err(_) => text.starts_with(|c: char| c.is_ascii_digit()).then(|| text.to_string()),
    }
}

/// Python's `<loc>\s*([^<\s]+)\s*</loc>` over the whole file, for files without entries.
fn bare_locations(body: &[u8]) -> Vec<Entry> {
    let mut locations = Vec::new();
    let mut at = 0;
    while let Some(found) = memchr(b'<', &body[at..]) {
        let start = at + found;
        at = start + 1;
        if !starts_with(body, start + 1, b"loc>") {
            continue;
        }
        let (value, end) = value(body, start + 5);
        if value.is_empty() || !starts_with(body, end, b"</loc>") {
            continue;
        }
        locations.push(Entry { url: location(value), ..Entry::default() });
        at = end + 6;
    }
    locations
}

/// Whitespace, a run of anything but `<` and whitespace, alone or as all a CDATA section holds besides whitespace, whitespace: the value and where the text after it starts.
fn value(text: &[u8], from: usize) -> (&[u8], usize) {
    let skip = |mut p: usize| {
        while text.get(p).is_some_and(|&b| is_space(b)) {
            p += 1;
        }
        p
    };
    let start = skip(from).min(text.len());
    if starts_with(text, start, CDATA) {
        let open = start + CDATA.len();
        // A value holds no `<`, so its section must close before the next one: the search never runs past it.
        let bound = memchr(b'<', &text[open.min(text.len())..]).map_or(text.len(), |at| open + at);
        let Some(close) = find(&text[..bound], open, b"]]>") else {
            return (&[], start);
        };
        let held = &text[open..close];
        let first = held.iter().position(|&b| !is_space(b)).unwrap_or(held.len());
        let last = held.iter().rposition(|&b| !is_space(b)).map_or(first, |last| last + 1);
        let held = &held[first..last];
        if held.iter().any(|&b| b == b'<' || is_space(b)) {
            return (&[], start);
        }
        return (held, skip(close + 3));
    }
    let mut end = start;
    while text.get(end).is_some_and(|&b| b != b'<' && !is_space(b)) {
        end += 1;
    }
    (&text[start..end], skip(end))
}

/// Python's `\s` in a bytes pattern.
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn starts_with(text: &[u8], at: usize, pattern: &[u8]) -> bool {
    text.get(at..at + pattern.len()).is_some_and(|found| found.eq_ignore_ascii_case(pattern))
}

fn find(text: &[u8], from: usize, pattern: &[u8]) -> Option<usize> {
    let mut at = from;
    while let Some(found) = memchr(pattern[0], text.get(at..)?) {
        if starts_with(text, at + found, pattern) {
            return Some(at + found);
        }
        at += found + 1;
    }
    None
}

/// Whether the file uses Google's news sitemap format, which lists only recent articles.
pub fn is_news(body: &[u8]) -> bool {
    NEWS_NAMESPACE.is_match(&body[..body.len().min(4096)])
}

pub fn has_time(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.contains('T'))
}

fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The address with XML's character references decoded, so `&amp;` in a sitemap reads as `&`.
pub(crate) fn location(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if !text.contains('&') {
        return text.into_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = &*text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at + 1..];
        match rest.find(';').and_then(|end| reference(&rest[..end]).map(|c| (c, end))) {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => out.push('&'),
        }
    }
    out.push_str(rest);
    out
}

fn reference(name: &str) -> Option<char> {
    let number = |digits: &str, radix: u32| {
        let valid = !digits.is_empty() && digits.chars().all(|c| c.is_digit(radix));
        valid.then(|| u32::from_str_radix(digits, radix).ok()).flatten().and_then(char::from_u32).filter(|c| !c.is_control())
    };
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => match name.strip_prefix("#x") {
            Some(hex) => number(hex, 16),
            None => number(name.strip_prefix('#')?, 10),
        },
    }
}

fn ascii_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| if b < 0x80 { b.to_ascii_lowercase() as char } else { '\u{fffd}' }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// A CDATA section's text up to its first `]]>`, between whitespace, with no `<` or whitespace in it.
    const HELD: &str = r"<!\[CDATA\[\s*((?:[^\s<\]]|\][^\s<\]]|\]\]+[^\s<\]>])*\]*)\s*\]\]>";
    static PATTERNS: LazyLock<[Regex; 2]> = LazyLock::new(|| {
        [Regex::new(r"(?is-u)<url\b(.*?)</url>|<sitemap\b(.*?)</sitemap>").unwrap(), Regex::new(&format!(r"(?i-u)<loc>\s*(?:{HELD}|([^<\s]+))\s*</loc>")).unwrap()]
    });

    thread_local! {
        static FIELD_PATTERNS: std::cell::RefCell<std::collections::HashMap<Vec<Vec<u8>>, std::rc::Rc<Regex>>> = Default::default();
    }

    /// `<(loc|lastmod|changefreq|news:publication_date|…)>\s*([^<\s]*)\s*</` over the names given.
    fn field_pattern(names: &[Vec<u8>]) -> std::rc::Rc<Regex> {
        FIELD_PATTERNS.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(found) = cache.get(names) {
                return found.clone();
            }
            let alternatives: Vec<String> = names.iter().map(|name| regex::escape(std::str::from_utf8(name).unwrap())).collect();
            let pattern = std::rc::Rc::new(Regex::new(&format!(r"(?i-u)<({})>\s*(?:{HELD}|([^<\s]*))\s*</", alternatives.join("|"))).unwrap());
            cache.insert(names.to_vec(), pattern.clone());
            pattern
        })
    }

    /// Regular expressions kept as the definition the scanner must match.
    fn oracle(body: &[u8]) -> (&'static str, Vec<Entry>) {
        let [entry, loc] = &*PATTERNS;
        let names = field_names(body);
        let field = field_pattern(&names);
        let mut entries = Vec::new();
        let mut kind = None;
        for c in entry.captures_iter(body) {
            kind.get_or_insert(if c.get(1).is_some() { "urlset" } else { "index" });
            let inner = c.get(1).or(c.get(2)).unwrap().as_bytes();
            let mut fields: [Option<&[u8]>; 4] = [None; 4];
            let mut locations = Vec::new();
            for f in field.captures_iter(inner) {
                let value = f.get(2).or(f.get(3)).unwrap().as_bytes();
                let slot = names.iter().position(|n| n.eq_ignore_ascii_case(f.get(1).unwrap().as_bytes())).unwrap().min(FIELDS.len() - 1);
                if !value.is_empty() {
                    fields[slot].get_or_insert(value);
                    if slot == 0 {
                        locations.push(value);
                    }
                }
            }
            locations.truncate(if c.get(2).is_some() { usize::MAX } else { 1 });
            for location in locations {
                entries.push(Entry {
                    url: super::location(location),
                    lastmod: fields[1].map(lossy),
                    changefreq: fields[2].map(ascii_lower),
                    published: fields[3].map(lossy),
                });
            }
        }
        if let (Some(kind), false) = (kind, entries.is_empty()) {
            return (kind, entries);
        }
        let locations: Vec<Entry> = loc
            .captures_iter(body)
            .map(|c| c.get(1).or(c.get(2)).unwrap().as_bytes())
            .filter(|value| !value.is_empty())
            .map(|value| Entry { url: super::location(value), ..Entry::default() })
            .collect();
        if locations.is_empty() {
            return ("invalid", Vec::new());
        }
        (if SITEMAP_INDEX.is_match(&body[..body.len().min(4096)]) { "index" } else { "urlset" }, locations)
    }

    #[test]
    fn scanner_reads_exactly_what_the_patterns_read() {
        let pieces: [&[u8]; 38] = [
            b"<url>", b"</url>", b"<URL >", b"<urlset>", b"<url_x>", b"<sitemap>", b"</SITEMAP>", b"<sitemapindex>",
            b"<loc>", b"</loc>", b"<LOC>", b"<lastmod>", b"</lastmod>", b"<changefreq>", b"<news:publication_date>",
            b"</", b"<", b">", b" ", b"\n", b"\x0b", b"\t", b"http://a.com/x", b"2024-01-01", b"Daily", b"\xff\xfe", b"</ loc>",
            b"<locale>", b"x", b"<url/>", b"<![CDATA[", b"<![cdata[", b"]]>", b"]", b"]]",
            b"<urlset xmlns:n=\"http://www.google.com/schemas/sitemap-news/0.9\">", b"<n:publication_date>", b"<N:Publication_Date>",
        ];
        let held: [&[u8]; 13] = [b"<url>", b"</url>", b"<sitemap>", b"</sitemap>", b"<loc>", b"</loc>", b"<![CDATA[", b"]]>", b"]", b">", b"<", b" ", b"x"];
        let mut rng = StdRng::seed_from_u64(7);
        for pieces in [&pieces[..], &held[..]] {
            for _ in 0..50_000 {
                let body: Vec<u8> = (0..rng.gen_range(1..40)).flat_map(|_| pieces[rng.gen_range(0..pieces.len())].to_vec()).collect();
                assert_eq!(parse_entries(&body), oracle(&body), "{}", String::from_utf8_lossy(&body));
            }
        }
        if let Ok(path) = std::env::var("SITEMAP_FILE") {
            let body = std::fs::read(path).unwrap();
            assert_eq!(parse_entries(&body), oracle(&body));
        }
    }

    #[test]
    fn pairs_each_location_with_its_dates() {
        let body = b"<urlset><url><loc> https://a.com/1 </loc><lastmod>2024-01-01</lastmod></url>\
            <URL><LOC>https://a.com/2</LOC><changefreq>Daily</changefreq></URL><url><lastmod>x</lastmod></url></urlset>";
        let (kind, entries) = parse_entries(body);
        assert_eq!(kind, "urlset");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lastmod.as_deref(), Some("2024-01-01"));
        assert_eq!(entries[1].changefreq.as_deref(), Some("daily"));
    }

    #[test]
    fn locations_decode_xml_references() {
        let body = b"<urlset><url><loc>https://a.com/?post_type=product&amp;p=2669</loc></url>\
            <url><loc>https://a.com/Q&amp;A&#39;s&#x2F;&amp;amp;&nbsp;&#0;&region=en&amp</loc></url></urlset>";
        let (_, entries) = parse_entries(body);
        assert_eq!(entries[0].url, "https://a.com/?post_type=product&p=2669");
        assert_eq!(entries[1].url, "https://a.com/Q&A's/&amp;&nbsp;&#0;&region=en&amp");
    }

    #[test]
    fn reads_values_held_in_cdata_sections() {
        let body = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\" xmlns:n=\"http://www.google.com/schemas/sitemap-news/0.9\">\t<url>\n\
            \t\t<loc><![CDATA[https://www.example.com/f1/story/_/id/1/title]]></loc>\n\t\t<n:news><n:publication_date>2026-11-02T09:57:23Z</n:publication_date></n:news>\n\t</url>\n\
            \t<url><loc> <![CDATA[ https://www.example.com/nba/story/_/id/2/?a=1&amp;b=2 ]]> </loc><lastmod><![CDATA[2026-11-02]]></lastmod></url>\n\
            \t<url><loc><![CDATA[https://www.example.com/bad link]]></loc></url><url><loc><![CDATA[https://www.example.com/open</loc></url></urlset>";
        let (kind, entries) = parse_entries(body);
        assert_eq!(kind, "urlset");
        assert_eq!(entries.iter().map(|e| (e.url.as_str(), e.lastmod.as_deref())).collect::<Vec<_>>(), [
            ("https://www.example.com/f1/story/_/id/1/title", None),
            ("https://www.example.com/nba/story/_/id/2/?a=1&b=2", Some("2026-11-02")),
        ], "a section holding whitespace, or never closed, is no value");
        let (kind, entries) = parse_entries(b"<sitemapindex><loc><![CDATA[https://a.com/s.xml]]></loc></sitemapindex>");
        assert_eq!((kind, entries[0].url.as_str()), ("index", "https://a.com/s.xml"));
    }

    #[test]
    fn a_large_file_of_unclosed_cdata_sections_reads_in_linear_time() {
        let entry = b"<url><loc><![CDATA[https://www.example.com/a/long/enough/path/to/a/story</loc><lastmod>2026-11-02</lastmod></url>\n";
        let body: Vec<u8> = [&b"<urlset>"[..], &entry.repeat(2_000_000 / entry.len()), b"</urlset>"].concat();
        let started = std::time::Instant::now();
        let (kind, entries) = parse_entries(&body);
        assert_eq!((kind, entries.len()), ("invalid", 0));
        assert!(started.elapsed() < std::time::Duration::from_secs(1), "2 MB of sections never closed took {:?}", started.elapsed());
    }

    #[test]
    fn reads_news_dates_under_any_prefix_bound_to_the_news_namespace() {
        let body = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\" xmlns:n=\"http://www.google.com/schemas/sitemap-news/0.9\">\t<url>\n\
            \t\t<loc><![CDATA[https://www.example.com/f1/story/_/id/1/title]]></loc>\n\t\t<n:news>\n\t\t\t<n:publication>\n\t\t\t\t<n:name>Example</n:name>\n\t\t\t</n:publication>\n\
            \t\t\t<n:publication_date>2026-11-02T09:57:23Z</n:publication_date>\n\t\t</n:news>\n\t</url>\n\
            \t<url><loc>https://www.example.com/2</loc><news:news><news:publication_date>2026-11-01</news:publication_date></news:news></url>\n\
            \t<url><loc>https://www.example.com/3</loc><x:publication_date>2026-10-01</x:publication_date></url></urlset>";
        let (_, entries) = parse_entries(body);
        assert_eq!(entries.iter().map(|e| e.published.as_deref()).collect::<Vec<_>>(), [Some("2026-11-02T09:57:23Z"), Some("2026-11-01"), None], "a prefix bound elsewhere, or not at all, is no news date");
        let both = b"<urlset xmlns:news='https://www.google.com/schemas/sitemap-news/0.9/' xmlns:g=\"http://www.google.com/schemas/sitemap-news/0.9\">\
            <url><loc>https://a.com/1</loc><g:publication_date>2026-10-30</g:publication_date></url></urlset>";
        assert_eq!(parse_entries(both).1[0].published.as_deref(), Some("2026-10-30"));
    }

    #[test]
    fn an_index_entry_naming_several_sitemaps_lists_each_and_a_page_entry_keeps_its_first() {
        let body = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sitemapindex xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n    <sitemap>\n\
            \x20       <loc>https://www.example.com/googlenewssitemap</loc>\n        <loc>https://www.example.com/sitemap/watch-videos/</loc>\n\
            \x20       <loc>https://www.example.com/sitemap/videos</loc>\n        <loc>https://www.example.com/where-to-watch.xml</loc>\n\
            \x20       <loc>https://www.example.com/google-news-posts</loc>\n    </sitemap> \n</sitemapindex>";
        let (kind, entries) = parse_entries(body);
        assert_eq!(kind, "index");
        assert_eq!(entries.iter().map(|e| e.url.rsplit('/').find(|p| !p.is_empty()).unwrap()).collect::<Vec<_>>(), [
            "googlenewssitemap", "watch-videos", "videos", "where-to-watch.xml", "google-news-posts",
        ]);
        let usual = b"<sitemapindex><sitemap><loc>https://a.com/1.xml</loc><lastmod>2026-01-01</lastmod></sitemap>\
            <sitemap><loc>https://a.com/2.xml</loc><loc>https://a.com/3.xml</loc><lastmod>2026-01-02</lastmod></sitemap></sitemapindex>";
        let (_, entries) = parse_entries(usual);
        let found: Vec<(&str, Option<&str>)> = entries.iter().map(|e| (e.url.as_str(), e.lastmod.as_deref())).collect();
        assert_eq!(found,[("https://a.com/1.xml", Some("2026-01-01")), ("https://a.com/2.xml", Some("2026-01-02")), ("https://a.com/3.xml", Some("2026-01-02"))]);
        let page = b"<urlset><url><loc>https://a.com/x</loc><loc>https://a.com/y</loc></url></urlset>";
        assert_eq!(parse_entries(page).1.iter().map(|e| e.url.as_str()).collect::<Vec<_>>(), ["https://a.com/x"], "a page's dates are its first location's");
    }

    #[test]
    fn reads_rss_and_atom_feeds_as_lists_of_pages() {
        let rss = b"<?xml version=\"1.0\"?><rss version=\"2.0\"><channel><link>https://a.com/</link>\
            <item><title>One</title><link>https://a.com/2026/10/one?x=1&amp;y=2</link><pubDate>Tue, 27 Oct 2026 10:00:00 GMT</pubDate></item>\
            <item><link><![CDATA[ https://a.com/two ]]></link><dc:date>2026-10-26T08:00:00Z</dc:date></item>\
            <item><guid isPermaLink=\"true\">https://a.com/three</guid></item><item><title>no link</title></item></channel></rss>";
        let (kind, entries) = parse_entries(rss);
        assert_eq!(kind, "urlset");
        let found: Vec<(&str, Option<&str>)> = entries.iter().map(|e| (e.url.as_str(), e.lastmod.as_deref())).collect();
        assert_eq!(found, [
            ("https://a.com/2026/10/one?x=1&y=2", Some("2026-10-27T10:00:00+00:00")),
            ("https://a.com/two", Some("2026-10-26T08:00:00Z")),
            ("https://a.com/three", None),
        ]);
        let atom = b"<feed xmlns=\"http://www.w3.org/2005/Atom\"><link href=\"https://a.com/\"/>\
            <entry><link rel=\"edit\" href=\"https://a.com/edit/1\"/><link rel=\"alternate\" type=\"text/html\" href=\"https://a.com/post-1\"/><updated>2026-10-27T09:00:00Z</updated></entry>\
            <entry><link href='https://a.com/post-2'/><published>2026-10-25</published></entry></feed>";
        let (kind, entries) = parse_entries(atom);
        assert_eq!(kind, "urlset");
        assert_eq!(entries.iter().map(|e| (e.url.as_str(), e.lastmod.as_deref())).collect::<Vec<_>>(), [
            ("https://a.com/post-1", Some("2026-10-27T09:00:00Z")),
            ("https://a.com/post-2", Some("2026-10-25")),
        ]);
        assert_eq!(parse_entries(b"<html><item><link>https://a.com/x</link></item></html>").0, "invalid", "items outside a feed are no feed");
    }

    #[test]
    fn falls_back_to_bare_locations() {
        let (kind, entries) = parse_entries(b"<sitemapindex><loc>https://a.com/s.xml</loc>");
        assert_eq!((kind, entries.len()), ("index", 1));
        assert_eq!(parse_entries(b"<html></html>").0, "invalid");
    }
}

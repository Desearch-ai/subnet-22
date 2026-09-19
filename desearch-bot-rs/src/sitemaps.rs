//! Parse sitemap files into the addresses and dates they list, exactly as the Python crawler's patterns read them.

use std::sync::LazyLock;

use memchr::memchr;
use regex::bytes::Regex;

static SITEMAP_INDEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)<sitemapindex").unwrap());
static NEWS_NAMESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)sitemap-news/0\.9").unwrap());

const FIELDS: [&[u8]; 4] = [b"loc", b"lastmod", b"changefreq", b"news:publication_date"];

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
        if let Some(entry) = entry(&body[open_end..close]) {
            entries.push(entry);
        }
        at = close + closing.len();
    }
    if let (Some(kind), false) = (kind, entries.is_empty()) {
        return (kind, entries);
    }
    let locations = bare_locations(body);
    if locations.is_empty() {
        return ("invalid", Vec::new());
    }
    let kind = if SITEMAP_INDEX.is_match(&body[..body.len().min(4096)]) { "index" } else { "urlset" };
    (kind, locations)
}

/// Python's `<(loc|lastmod|changefreq|news:publication_date)>\s*([^<\s]*)\s*</`, first value of each kept.
fn entry(inner: &[u8]) -> Option<Entry> {
    let mut fields: [Option<&[u8]>; 4] = [None; 4];
    let mut at = 0;
    while let Some(found) = memchr(b'<', &inner[at..]) {
        let start = at + found;
        at = start + 1;
        let Some(slot) = FIELDS.iter().position(|name| starts_with(inner, start + 1, name) && inner.get(start + 1 + name.len()) == Some(&b'>'))
        else {
            continue;
        };
        let (value, end) = value(inner, start + 2 + FIELDS[slot].len());
        if !starts_with(inner, end, b"</") {
            continue;
        }
        at = end + 2;
        if !value.is_empty() {
            fields[slot].get_or_insert(value);
        }
    }
    Some(Entry {
        url: lossy(fields[0]?),
        lastmod: fields[1].map(lossy),
        changefreq: fields[2].map(ascii_lower),
        published: fields[3].map(lossy),
    })
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
        locations.push(Entry { url: lossy(value), ..Entry::default() });
        at = end + 6;
    }
    locations
}

/// Whitespace, a run of anything but `<` and whitespace, whitespace: the value and where the text after it starts.
fn value(text: &[u8], from: usize) -> (&[u8], usize) {
    let skip = |mut p: usize| {
        while text.get(p).is_some_and(|&b| is_space(b)) {
            p += 1;
        }
        p
    };
    let start = skip(from).min(text.len());
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

fn ascii_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| if b < 0x80 { b.to_ascii_lowercase() as char } else { '\u{fffd}' }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The Python crawler's regular expressions, kept as the definition the scanner must match.
    fn oracle(body: &[u8]) -> (&'static str, Vec<Entry>) {
        let entry = Regex::new(r"(?is-u)<url\b(.*?)</url>|<sitemap\b(.*?)</sitemap>").unwrap();
        let field = Regex::new(r"(?i-u)<(loc|lastmod|changefreq|news:publication_date)>\s*([^<\s]*)\s*</").unwrap();
        let loc = Regex::new(r"(?i-u)<loc>\s*([^<\s]+)\s*</loc>").unwrap();
        let mut entries = Vec::new();
        let mut kind = None;
        for c in entry.captures_iter(body) {
            kind.get_or_insert(if c.get(1).is_some() { "urlset" } else { "index" });
            let inner = c.get(1).or(c.get(2)).unwrap().as_bytes();
            let mut fields: [Option<&[u8]>; 4] = [None; 4];
            for f in field.captures_iter(inner) {
                let value = f.get(2).unwrap().as_bytes();
                let slot = FIELDS.iter().position(|n| n.eq_ignore_ascii_case(f.get(1).unwrap().as_bytes())).unwrap();
                if !value.is_empty() {
                    fields[slot].get_or_insert(value);
                }
            }
            if let Some(location) = fields[0] {
                entries.push(Entry {
                    url: lossy(location),
                    lastmod: fields[1].map(lossy),
                    changefreq: fields[2].map(ascii_lower),
                    published: fields[3].map(lossy),
                });
            }
        }
        if let (Some(kind), false) = (kind, entries.is_empty()) {
            return (kind, entries);
        }
        let locations: Vec<Entry> = loc.captures_iter(body).map(|c| Entry { url: lossy(&c[1]), ..Entry::default() }).collect();
        if locations.is_empty() {
            return ("invalid", Vec::new());
        }
        (if SITEMAP_INDEX.is_match(&body[..body.len().min(4096)]) { "index" } else { "urlset" }, locations)
    }

    #[test]
    fn scanner_reads_exactly_what_the_patterns_read() {
        let pieces: [&[u8]; 30] = [
            b"<url>", b"</url>", b"<URL >", b"<urlset>", b"<url_x>", b"<sitemap>", b"</SITEMAP>", b"<sitemapindex>",
            b"<loc>", b"</loc>", b"<LOC>", b"<lastmod>", b"</lastmod>", b"<changefreq>", b"<news:publication_date>",
            b"</", b"<", b">", b" ", b"\n", b"\x0b", b"\t", b"http://a.com/x", b"2024-01-01", b"Daily", b"\xff\xfe", b"</ loc>",
            b"<locale>", b"x", b"<url/>",
        ];
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50_000 {
            let body: Vec<u8> = (0..rng.gen_range(1..40)).flat_map(|_| pieces[rng.gen_range(0..pieces.len())].to_vec()).collect();
            assert_eq!(parse_entries(&body), oracle(&body), "{}", String::from_utf8_lossy(&body));
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
    fn falls_back_to_bare_locations() {
        let (kind, entries) = parse_entries(b"<sitemapindex><loc>https://a.com/s.xml</loc>");
        assert_eq!((kind, entries.len()), ("index", 1));
        assert_eq!(parse_entries(b"<html></html>").0, "invalid");
    }
}

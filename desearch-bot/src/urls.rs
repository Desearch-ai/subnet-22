//! Every URL we know, normalised once, and split and joined exactly as Python's urllib does.

use std::borrow::Cow;
use std::net::{Ipv4Addr, Ipv6Addr};

use stringprep::tables;
use unicode_normalization::UnicodeNormalization;

use crate::text;

pub const HTTPS: u8 = 1;
pub const WWW: u8 = 2;
pub const TIMED: u8 = 4;

const SAFE_PATH: &[u8] = b"/:@!$&'()*+,;=-._~%";
const SAFE_QUERY: &[u8] = b"/:@!$&'()*+,;=-._~%?";
const TRACKING: [&str; 3] = ["gclid", "fbclid", "msclkid"];
const HEX: &[u8; 16] = b"0123456789ABCDEF";
const DOTS: [char; 4] = ['.', '\u{3002}', '\u{ff0e}', '\u{ff61}'];
const USES_RELATIVE: [&str; 20] = [
    "", "ftp", "http", "gopher", "nntp", "imap", "wais", "file", "https", "shttp", "mms", "prospero", "rtsp",
    "rtsps", "rtspu", "sftp", "svn", "svn+ssh", "ws", "wss",
];
const USES_NETLOC: [&str; 27] = [
    "", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https", "shttp", "snews",
    "prospero", "rtsp", "rtsps", "rtspu", "rsync", "svn", "svn+ssh", "sftp", "nfs", "git", "git+ssh", "ws", "wss",
    "itms-services",
];

/// Text Python's urllib or IDNA codec would reject with a ValueError.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invalid;

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("not a URL or host Python would accept")
    }
}

impl std::error::Error for Invalid {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Url {
    pub key: Vec<u8>,
    pub flags: u8,
}

/// What the store keeps per URL, packed as Python's struct `<qIIIIIB`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub sitemap_id: i64,
    pub lastmod: u32,
    pub first_seen: u32,
    pub last_seen: u32,
    pub crawled_at: u32,
    pub pushed_at: u32,
    pub flags: u8,
}

impl Record {
    pub const SIZE: usize = 29;

    pub fn pack(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[..8].copy_from_slice(&self.sitemap_id.to_le_bytes());
        let words = [self.lastmod, self.first_seen, self.last_seen, self.crawled_at, self.pushed_at];
        for (i, word) in words.into_iter().enumerate() {
            out[8 + i * 4..12 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        out[28] = self.flags;
        out
    }

    pub fn unpack(data: &[u8]) -> Option<Record> {
        if data.len() != Self::SIZE {
            return None;
        }
        let word = |i: usize| u32::from_le_bytes(data[8 + i * 4..12 + i * 4].try_into().unwrap());
        Some(Record {
            sitemap_id: i64::from_le_bytes(data[..8].try_into().unwrap()),
            lastmod: word(0),
            first_seen: word(1),
            last_seen: word(2),
            crawled_at: word(3),
            pushed_at: word(4),
            flags: data[28],
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Listing {
    pub listed: usize,
    pub new: usize,
    pub moved: usize,
}

/// The five parts of Python's `_urlsplit`, with None wherever Python has None.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    pub scheme: Option<String>,
    pub netloc: Option<String>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

impl Parts {
    pub fn netloc(&self) -> &str {
        self.netloc.as_deref().unwrap_or("")
    }

    /// Python's `SplitResult.hostname`.
    pub fn hostname(&self) -> Option<String> {
        let (host, _) = host_info(self.netloc());
        if host.is_empty() {
            return None;
        }
        let (name, zone) = host.find('%').map_or((host, ""), |i| (&host[..i], &host[i..]));
        Some(format!("{}{}", name.to_lowercase(), zone))
    }

    /// Python's `SplitResult.port`.
    pub fn port(&self) -> Result<Option<u16>, Invalid> {
        let Some(port) = host_info(self.netloc()).1 else {
            return Ok(None);
        };
        if !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Invalid);
        }
        let digits = port.trim_start_matches('0');
        if digits.len() > 5 {
            return Err(Invalid);
        }
        let value: u32 = if digits.is_empty() { 0 } else { digits.parse().map_err(|_| Invalid)? };
        u16::try_from(value).map(Some).map_err(|_| Invalid)
    }
}

/// Python's `urllib.parse._urlsplit(url, None)`.
pub fn split(url: &str) -> Result<Parts, Invalid> {
    let url: String = url
        .trim_start_matches(|c: char| c <= ' ')
        .chars()
        .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
        .collect();
    let mut rest = url.as_str();
    let mut scheme = None;
    if let Some(i) = rest.find(':') {
        let bytes = rest.as_bytes();
        if i > 0
            && bytes[0].is_ascii_alphabetic()
            && bytes[..i].iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
        {
            scheme = Some(rest[..i].to_ascii_lowercase());
            rest = &rest[i + 1..];
        }
    }
    let mut netloc = None;
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        let found = &after[..end];
        if found.contains('[') != found.contains(']') {
            return Err(Invalid);
        }
        if found.contains('[') {
            check_bracketed_netloc(found)?;
        }
        netloc = Some(found.to_string());
        rest = &after[end..];
    }
    let (rest, fragment) = match rest.split_once('#') {
        Some((before, fragment)) => (before, Some(fragment.to_string())),
        None => (rest, None),
    };
    let (path, query) = match rest.split_once('?') {
        Some((path, query)) => (path, Some(query.to_string())),
        None => (rest, None),
    };
    check_netloc(netloc.as_deref())?;
    Ok(Parts { scheme, netloc, path: path.to_string(), query, fragment })
}

/// Python's `urllib.parse.urljoin(base, url)`.
pub fn join(base: &str, url: &str) -> Result<String, Invalid> {
    if base.is_empty() {
        return Ok(url.to_string());
    }
    if url.is_empty() {
        return Ok(base.to_string());
    }
    let b = split(base)?;
    let u = split(url)?;
    let scheme = u.scheme.clone().or_else(|| b.scheme.clone());
    let named = scheme.as_deref().unwrap_or("");
    if scheme != b.scheme || (!named.is_empty() && !USES_RELATIVE.contains(&named)) {
        return Ok(url.to_string());
    }
    let mut netloc = u.netloc.as_deref();
    if named.is_empty() || USES_NETLOC.contains(&named) {
        if netloc.is_some_and(|n| !n.is_empty()) {
            return Ok(unsplit(named, netloc, &u.path, u.query.as_deref(), u.fragment.as_deref()));
        }
        netloc = b.netloc.as_deref();
    }
    if u.path.is_empty() {
        let (query, fragment) = match u.query.as_deref() {
            Some(query) => (Some(query), u.fragment.as_deref()),
            None => (b.query.as_deref(), u.fragment.as_deref().or(b.fragment.as_deref())),
        };
        return Ok(unsplit(named, netloc, &b.path, query, fragment));
    }
    let segments: Vec<&str> = if u.path.starts_with('/') {
        u.path.split('/').collect()
    } else {
        let mut base_parts: Vec<&str> = b.path.split('/').collect();
        if base_parts.last() != Some(&"") {
            base_parts.pop();
        }
        base_parts.extend(u.path.split('/'));
        let last = base_parts.len().saturating_sub(1);
        base_parts
            .into_iter()
            .enumerate()
            .filter(|(i, segment)| *i == 0 || *i == last || !segment.is_empty())
            .map(|(_, segment)| segment)
            .collect()
    };
    let mut resolved: Vec<&str> = Vec::with_capacity(segments.len());
    for segment in &segments {
        match *segment {
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
    let path = resolved.join("/");
    let path = if path.is_empty() { "/" } else { path.as_str() };
    Ok(unsplit(named, netloc, path, u.query.as_deref(), u.fragment.as_deref()))
}

fn unsplit(scheme: &str, netloc: Option<&str>, path: &str, query: Option<&str>, fragment: Option<&str>) -> String {
    let mut url = String::with_capacity(path.len() + 64);
    if !scheme.is_empty() {
        url.push_str(scheme);
        url.push(':');
    }
    if let Some(netloc) = netloc {
        url.push_str("//");
        url.push_str(netloc);
        if !path.is_empty() && !path.starts_with('/') {
            url.push('/');
        }
    } else if path.starts_with("//") {
        url.push_str("//");
    }
    url.push_str(path);
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    if let Some(fragment) = fragment {
        url.push('#');
        url.push_str(fragment);
    }
    url
}

fn host_info(netloc: &str) -> (&str, Option<&str>) {
    let info = netloc.rsplit_once('@').map_or(netloc, |(_, info)| info);
    let (host, port) = match info.split_once('[') {
        Some((_, bracketed)) => {
            let (host, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            (host, after.split_once(':').map_or("", |(_, port)| port))
        }
        None => info.split_once(':').unwrap_or((info, "")),
    };
    (host, Some(port).filter(|port| !port.is_empty()))
}

fn check_bracketed_netloc(netloc: &str) -> Result<(), Invalid> {
    let info = netloc.rsplit_once('@').map_or(netloc, |(_, info)| info);
    let host = match info.split_once('[') {
        Some((before, bracketed)) => {
            if !before.is_empty() {
                return Err(Invalid);
            }
            let (host, port) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            if !port.is_empty() && !port.starts_with(':') {
                return Err(Invalid);
            }
            host
        }
        None => info.split_once(':').map_or(info, |(host, _)| host),
    };
    if let Some(rest) = host.strip_prefix('v') {
        let hex = rest.bytes().take_while(u8::is_ascii_hexdigit).count();
        let valid = hex > 0 && rest[hex..].starts_with('.') && rest.len() > hex + 1 && !rest.contains('\n');
        return if valid { Ok(()) } else { Err(Invalid) };
    }
    if host.parse::<Ipv4Addr>().is_ok() {
        return Err(Invalid);
    }
    let address = match host.split_once('%') {
        Some((address, zone)) if !zone.is_empty() && !zone.contains('%') => address,
        Some(_) => return Err(Invalid),
        None => host,
    };
    address.parse::<Ipv6Addr>().map(|_| ()).map_err(|_| Invalid)
}

fn check_netloc(netloc: Option<&str>) -> Result<(), Invalid> {
    let Some(netloc) = netloc.filter(|n| !n.is_ascii()) else {
        return Ok(());
    };
    let bare: String = netloc.chars().filter(|c| !matches!(c, '@' | ':' | '#' | '?')).collect();
    let normalized: String = bare.nfkc().collect();
    if bare != normalized && normalized.contains(['/', '?', '#', '@', ':']) {
        return Err(Invalid);
    }
    Ok(())
}

/// A host as the stores key it: stripped, IDNA-encoded (IDNA 2003) and lowercased.
pub fn ascii(host: &str) -> Result<String, Invalid> {
    let host = text::strip(host).trim_end_matches('.');
    Ok(idna(host)?.to_ascii_lowercase())
}

/// Python's `str.encode("idna")`.
fn idna(input: &str) -> Result<Cow<'_, str>, Invalid> {
    if input.is_empty() {
        return Ok(Cow::Borrowed(""));
    }
    if input.is_ascii() {
        let labels: Vec<&str> = input.split('.').collect();
        let last = labels.len() - 1;
        for (i, label) in labels.iter().enumerate() {
            if (i < last && label.is_empty()) || label.len() >= 64 {
                return Err(Invalid);
            }
        }
        return Ok(Cow::Borrowed(input));
    }
    let mut labels: Vec<&str> = input.split(DOTS).collect();
    let trailing = if labels.last() == Some(&"") {
        labels.pop();
        "."
    } else {
        ""
    };
    let mut out = String::with_capacity(input.len() + 8);
    for label in labels {
        if !out.is_empty() {
            out.push('.');
        }
        out.push_str(&label_to_ascii(label)?);
    }
    out.push_str(trailing);
    Ok(Cow::Owned(out))
}

fn label_to_ascii(label: &str) -> Result<String, Invalid> {
    if label.is_ascii() {
        return if (1..64).contains(&label.len()) { Ok(label.to_string()) } else { Err(Invalid) };
    }
    let label = nameprep(label)?;
    if label.is_ascii() {
        return if (1..64).contains(&label.len()) { Ok(label) } else { Err(Invalid) };
    }
    if label.to_lowercase().starts_with("xn--") {
        return Err(Invalid);
    }
    let encoded = format!("xn--{}", punycode(&label).ok_or(Invalid)?);
    if encoded.len() < 64 {
        Ok(encoded)
    } else {
        Err(Invalid)
    }
}

/// RFC 3491 nameprep as Python's `encodings.idna` applies it, which allows unassigned code points.
fn nameprep(label: &str) -> Result<String, Invalid> {
    let mut mapped = String::with_capacity(label.len());
    for c in label.chars().filter(|&c| !tables::commonly_mapped_to_nothing(c)) {
        mapped.extend(tables::case_fold_for_nfkc(c));
    }
    let normalized: String = mapped.nfkc().collect();
    let prohibited = |c: char| {
        tables::non_ascii_space_character(c)
            || tables::non_ascii_control_character(c)
            || tables::private_use(c)
            || tables::non_character_code_point(c)
            || tables::surrogate_code(c)
            || tables::inappropriate_for_plain_text(c)
            || tables::inappropriate_for_canonical_representation(c)
            || tables::change_display_properties_or_deprecated(c)
            || tables::tagging_character(c)
    };
    if normalized.chars().any(prohibited) {
        return Err(Invalid);
    }
    let right_to_left: Vec<bool> = normalized.chars().map(tables::bidi_r_or_al).collect();
    if right_to_left.iter().any(|&r| r)
        && (normalized.chars().any(tables::bidi_l) || !right_to_left[0] || !right_to_left[right_to_left.len() - 1])
    {
        return Err(Invalid);
    }
    Ok(normalized)
}

/// RFC 3492 punycode, as Python's "punycode" codec writes it.
fn punycode(input: &str) -> Option<String> {
    const BASE: u32 = 36;
    const TMIN: u32 = 1;
    const TMAX: u32 = 26;
    let code_points: Vec<u32> = input.chars().map(u32::from).collect();
    let mut output: String = input.chars().filter(char::is_ascii).collect();
    let basic = output.len() as u32;
    let mut handled = basic;
    if basic > 0 {
        output.push('-');
    }
    let digit = |d: u32| (if d < 26 { b'a' + d as u8 } else { b'0' + (d - 26) as u8 }) as char;
    let (mut n, mut delta, mut bias) = (128u32, 0u32, 72u32);
    while (handled as usize) < code_points.len() {
        let m = *code_points.iter().filter(|&&c| c >= n).min()?;
        delta = delta.checked_add((m - n).checked_mul(handled + 1)?)?;
        n = m;
        for &c in &code_points {
            if c < n {
                delta = delta.checked_add(1)?;
            }
            if c == n {
                let mut q = delta;
                let mut k = BASE;
                loop {
                    let t = if k <= bias { TMIN } else if k >= bias + TMAX { TMAX } else { k - bias };
                    if q < t {
                        break;
                    }
                    output.push(digit(t + (q - t) % (BASE - t)));
                    q = (q - t) / (BASE - t);
                    k += BASE;
                }
                output.push(digit(q));
                bias = adapt(delta, handled + 1, handled == basic);
                delta = 0;
                handled += 1;
            }
        }
        delta = delta.checked_add(1)?;
        n += 1;
    }
    Some(output)
}

fn adapt(delta: u32, points: u32, first: bool) -> u32 {
    let mut delta = if first { delta / 700 } else { delta / 2 };
    delta += delta / points;
    let mut k = 0;
    while delta > 35 * 26 / 2 {
        delta /= 35;
        k += 36;
    }
    k + 36 * delta / (delta + 38)
}

/// Normalises the URLs one domain's sitemaps list.
pub struct Normaliser {
    domain: String,
    suffix: String,
}

impl Normaliser {
    pub fn new(domain: &str) -> Result<Self, Invalid> {
        let domain = ascii(domain)?;
        Ok(Normaliser { suffix: format!(".{domain}"), domain })
    }

    /// A sitemap URL as its store key, or None when it is not a web page on this domain.
    pub fn parse(&self, url: &str) -> Option<Url> {
        let parts = split(text::strip(url)).ok()?;
        let scheme = parts.scheme.as_deref().unwrap_or("");
        let default_port = match scheme {
            "http" => 80,
            "https" => 443,
            _ => return None,
        };
        let netloc = parts.netloc();
        let simple = !netloc.is_empty() && netloc.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-');
        let (host, port) = if simple {
            (ascii(netloc).ok()?, None)
        } else {
            let hostname = parts.hostname()?;
            (ascii(&hostname).ok()?, parts.port().ok()?)
        };
        if host != self.domain && !host.ends_with(&self.suffix) {
            return None;
        }
        let bare = host.strip_prefix("www.").unwrap_or(&host);
        let mut key = String::with_capacity(self.domain.len() + url.len() + 8);
        key.push_str(&self.domain);
        key.push('\0');
        key.push_str(bare);
        if let Some(port) = port.filter(|&port| port != default_port) {
            key.push(':');
            key.push_str(&port.to_string());
        }
        let path = tidy(&parts.path, SAFE_PATH);
        key.push_str(if path.is_empty() { "/" } else { &path });
        let query = tidy_query(parts.query.as_deref().unwrap_or(""));
        if !query.is_empty() {
            key.push('?');
            key.push_str(&query);
        }
        let flags = if scheme == "https" { HTTPS } else { 0 } | if bare.len() != host.len() { WWW } else { 0 };
        Some(Url { key: key.into_bytes(), flags })
    }
}

/// A listed URL, normalised and checked against its domain.
pub fn parse(url: &str, domain: &str) -> Option<Url> {
    Normaliser::new(domain).ok()?.parse(url)
}

fn tidy<'a>(part: &'a str, safe: &[u8]) -> Cow<'a, str> {
    let plain = |b: u8| b.is_ascii_alphanumeric() || b"-._~/:@!$&'()*+,;=".contains(&b);
    if part.bytes().all(plain) {
        return Cow::Borrowed(part);
    }
    Cow::Owned(quote(&unescape(part), safe))
}

fn unescape(part: &str) -> Vec<u8> {
    let bytes = part.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() && bytes[i + 1].is_ascii_hexdigit() && bytes[i + 2].is_ascii_hexdigit() {
            let value = hex_value(bytes[i + 1]) * 16 + hex_value(bytes[i + 2]);
            if value.is_ascii_alphanumeric() || matches!(value, b'-' | b'.' | b'_' | b'~') {
                out.push(value);
            } else {
                out.extend([b'%', bytes[i + 1].to_ascii_uppercase(), bytes[i + 2].to_ascii_uppercase()]);
            }
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

fn hex_value(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        _ => b - b'A' + 10,
    }
}

/// Python's `urllib.parse.quote(text, safe)` over already UTF-8 encoded text.
fn quote(bytes: &[u8], safe: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 16);
    for &b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') || safe.contains(&b) {
            out.push(b as char);
        } else {
            out.extend(['%', HEX[(b >> 4) as usize] as char, HEX[(b & 15) as usize] as char]);
        }
    }
    out
}

fn tidy_query(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut kept: Vec<String> = query
        .split('&')
        .filter(|pair| {
            let name = pair.split('=').next().unwrap_or("");
            let name = if name.is_ascii() { name.to_ascii_lowercase() } else { name.to_lowercase() };
            !pair.is_empty() && !TRACKING.contains(&name.as_str()) && !name.starts_with("utm_")
        })
        .map(|pair| tidy(pair, SAFE_QUERY).into_owned())
        .collect();
    kept.sort_unstable();
    kept.join("&")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(url: &str, domain: &str) -> Option<String> {
        parse(url, domain).map(|u| String::from_utf8(u.key).unwrap().replace('\0', " "))
    }

    #[test]
    fn normalises_like_python() {
        assert_eq!(key("HTTPS://Example.COM:443/Page#top", "example.com").unwrap(), "example.com example.com/Page");
        assert_eq!(key("https://example.com/s?b=2&utm_source=x&a=1&fbclid=z", "example.com").unwrap(), "example.com example.com/s?a=1&b=2");
        assert_eq!(key("https://example.com/caf%c3%a9/%7Euser/a%20b", "example.com").unwrap(), "example.com example.com/caf%C3%A9/~user/a%20b");
        assert_eq!(key("https://münchen.de/stadt", "münchen.de").unwrap(), "xn--mnchen-3ya.de xn--mnchen-3ya.de/stadt");
        assert_eq!(parse("http://www.example.com/a", "example.com").unwrap().flags, WWW);
        assert_eq!(key("https://evil-example.com/a", "example.com"), None);
        assert_eq!(key("https://example.com:99999/", "example.com"), None);
    }

    #[test]
    fn joins_like_python() {
        let base = "https://example.com/sitemaps/index.xml";
        assert_eq!(join(base, "part-1.xml").unwrap(), "https://example.com/sitemaps/part-1.xml");
        assert_eq!(join(base, "/root.xml").unwrap(), "https://example.com/root.xml");
        assert_eq!(join(base, "../up.xml").unwrap(), "https://example.com/up.xml");
        assert_eq!(join(base, "//cdn.example.com/a.xml").unwrap(), "https://cdn.example.com/a.xml");
        assert_eq!(join(base, "HTTP://Other.com/x").unwrap(), "HTTP://Other.com/x");
        assert_eq!(join("https://example.com/", "?page=2").unwrap(), "https://example.com/?page=2");
    }

    #[test]
    fn record_round_trips() {
        let record = Record { sitemap_id: 42, lastmod: 1, first_seen: 2, last_seen: 3, crawled_at: 4, pushed_at: 5, flags: 7 };
        assert_eq!(Record::unpack(&record.pack()), Some(record));
    }
}

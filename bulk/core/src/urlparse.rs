//! Python 3.14's `urllib.parse` on str values: `urlsplit`, `urlunsplit`, `urljoin`, `parse_qsl`, `unquote`, `quote` and `urlencode`.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::py::{repr_str, ValueError};

pub const USES_NETLOC: [&str; 27] = [
    "",
    "ftp",
    "http",
    "gopher",
    "nntp",
    "telnet",
    "imap",
    "wais",
    "file",
    "mms",
    "https",
    "shttp",
    "snews",
    "prospero",
    "rtsp",
    "rtsps",
    "rtspu",
    "rsync",
    "svn",
    "svn+ssh",
    "sftp",
    "nfs",
    "git",
    "git+ssh",
    "ws",
    "wss",
    "itms-services",
];
pub const USES_RELATIVE: [&str; 20] = [
    "", "ftp", "http", "gopher", "nntp", "imap", "wais", "file", "https", "shttp", "mms", "prospero", "rtsp", "rtsps", "rtspu", "sftp", "svn", "svn+ssh", "ws",
    "wss",
];

static IPV_FUTURE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\Av[a-fA-F0-9]+\..+\z").expect("pattern compiles"));

pub type Result<T> = std::result::Result<T, ValueError>;

/// `urllib.parse.SplitResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitResult {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

/// `_urlsplit`: the parts with None where the URL has none, not even an empty one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    pub scheme: Option<String>,
    pub netloc: Option<String>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

fn is_scheme_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
}

fn error(message: impl Into<String>) -> ValueError {
    ValueError(message.into())
}

/// `_urlsplit(url, None)`, with its ValueErrors for bad brackets and NFKC-unsafe hosts.
pub fn split_parts(url: &str) -> Result<Parts> {
    let url: String = url.trim_start_matches(|c: char| c <= ' ').chars().filter(|c| !matches!(c, '\t' | '\r' | '\n')).collect();
    let mut split = Parts::default();
    let mut rest = url.as_str();
    if let Some(i) = url.find(':') {
        if i > 0 && url.starts_with(|c: char| c.is_ascii_alphabetic()) && url[..i].chars().all(is_scheme_char) {
            split.scheme = Some(url[..i].to_ascii_lowercase());
            rest = &url[i + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        let delim = after.find(['/', '?', '#']).unwrap_or(after.len());
        let netloc = &after[..delim];
        rest = &after[delim..];
        let (open, close) = (netloc.contains('['), netloc.contains(']'));
        if open != close {
            return Err(error("Invalid IPv6 URL"));
        }
        if open {
            check_bracketed_netloc(netloc)?;
        }
        split.netloc = Some(netloc.to_string());
    }
    if let Some((before, fragment)) = rest.split_once('#') {
        split.fragment = Some(fragment.to_string());
        rest = before;
    }
    if let Some((before, query)) = rest.split_once('?') {
        split.query = Some(query.to_string());
        rest = before;
    }
    split.path = rest.to_string();
    check_netloc(split.netloc.as_deref().unwrap_or(""))?;
    Ok(split)
}

/// `urlsplit(url)`.
pub fn urlsplit(url: &str) -> Result<SplitResult> {
    let parts = split_parts(url)?;
    Ok(SplitResult {
        scheme: parts.scheme.unwrap_or_default(),
        netloc: parts.netloc.unwrap_or_default(),
        path: parts.path,
        query: parts.query.unwrap_or_default(),
        fragment: parts.fragment.unwrap_or_default(),
    })
}

fn check_bracketed_netloc(netloc: &str) -> Result<()> {
    let host_and_port = netloc.rsplit_once('@').map_or(netloc, |(_, rest)| rest);
    let host = match host_and_port.split_once('[') {
        Some((before, bracketed)) => {
            if !before.is_empty() {
                return Err(error("Invalid IPv6 URL"));
            }
            let (host, port) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            if !port.is_empty() && !port.starts_with(':') {
                return Err(error("Invalid IPv6 URL"));
            }
            host
        }
        None => host_and_port.split_once(':').map_or(host_and_port, |(host, _)| host),
    };
    if host.starts_with('v') {
        if !IPV_FUTURE.is_match(host) {
            return Err(error("IPvFuture address is invalid"));
        }
        return Ok(());
    }
    if host.parse::<Ipv4Addr>().is_ok() {
        return Err(error("An IPv4 address cannot be in brackets"));
    }
    let (address, scope) = host.split_once('%').map_or((host, None), |(address, scope)| (address, Some(scope)));
    if address.parse::<Ipv6Addr>().is_err() || scope.is_some_and(|s| s.is_empty() || s.contains('%')) {
        return Err(error(format!("{} does not appear to be an IPv4 or IPv6 address", repr_str(host))));
    }
    Ok(())
}

fn check_netloc(netloc: &str) -> Result<()> {
    if netloc.is_ascii() {
        return Ok(());
    }
    let n: String = netloc.chars().filter(|c| !matches!(c, '@' | ':' | '#' | '?')).collect();
    let normalized: String = n.nfkc().collect();
    if n != normalized && normalized.contains(['/', '?', '#', '@', ':']) {
        return Err(error(format!("netloc '{netloc}' contains invalid characters under NFKC normalization")));
    }
    Ok(())
}

/// `_urlunsplit`: None parts are left out, empty ones kept.
pub fn unsplit_parts(scheme: Option<&str>, netloc: Option<&str>, path: &str, query: Option<&str>, fragment: Option<&str>) -> String {
    let mut url = path.to_string();
    if let Some(netloc) = netloc {
        if !url.is_empty() && !url.starts_with('/') {
            url.insert(0, '/');
        }
        url = format!("//{netloc}{url}");
    } else if url.starts_with("//") {
        url = format!("//{url}");
    }
    if let Some(scheme) = scheme.filter(|s| !s.is_empty()) {
        url = format!("{scheme}:{url}");
    }
    if let Some(query) = query {
        url = format!("{url}?{query}");
    }
    if let Some(fragment) = fragment {
        url = format!("{url}#{fragment}");
    }
    url
}

/// `urlunsplit(parts)`.
pub fn urlunsplit(parts: &SplitResult) -> String {
    let netloc = if !parts.netloc.is_empty() {
        Some(parts.netloc.as_str())
    } else if !parts.scheme.is_empty() && USES_NETLOC.contains(&parts.scheme.as_str()) && (parts.path.is_empty() || parts.path.starts_with('/')) {
        Some("")
    } else {
        None
    };
    let some = |s: &str| (!s.is_empty()).then_some(s.to_string());
    unsplit_parts(some(&parts.scheme).as_deref(), netloc, &parts.path, some(&parts.query).as_deref(), some(&parts.fragment).as_deref())
}

/// `urljoin(base, url)`.
pub fn urljoin(base: &str, url: &str) -> Result<String> {
    if base.is_empty() {
        return Ok(url.to_string());
    }
    if url.is_empty() {
        return Ok(base.to_string());
    }
    let b = split_parts(base)?;
    let u = split_parts(url)?;
    let scheme = u.scheme.clone().or_else(|| b.scheme.clone());
    if scheme != b.scheme || scheme.as_deref().is_some_and(|s| !s.is_empty() && !USES_RELATIVE.contains(&s)) {
        return Ok(url.to_string());
    }
    let scheme_ref = scheme.as_deref();
    let mut netloc = u.netloc.clone();
    if scheme_ref.is_none_or(|s| s.is_empty() || USES_NETLOC.contains(&s)) {
        if netloc.as_deref().is_some_and(|n| !n.is_empty()) {
            return Ok(unsplit_parts(scheme_ref, netloc.as_deref(), &u.path, u.query.as_deref(), u.fragment.as_deref()));
        }
        netloc = b.netloc.clone();
    }
    if u.path.is_empty() {
        let (query, fragment) = match u.query {
            None => (b.query.clone(), u.fragment.clone().or(b.fragment.clone())),
            Some(query) => (Some(query), u.fragment.clone()),
        };
        return Ok(unsplit_parts(scheme_ref, netloc.as_deref(), &b.path, query.as_deref(), fragment.as_deref()));
    }
    let mut base_parts: Vec<&str> = b.path.split('/').collect();
    if base_parts.last() != Some(&"") {
        base_parts.pop();
    }
    let segments: Vec<&str> = if u.path.starts_with('/') {
        u.path.split('/').collect()
    } else {
        let all: Vec<&str> = base_parts.into_iter().chain(u.path.split('/')).collect();
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
    Ok(unsplit_parts(scheme_ref, netloc.as_deref(), if joined.is_empty() { "/" } else { &joined }, u.query.as_deref(), u.fragment.as_deref()))
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn unquote_to_bytes(ascii: &str) -> Vec<u8> {
    let bytes = ascii.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let (Some(hi), Some(lo)) = (bytes.get(i + 1).copied().and_then(hex_value), bytes.get(i + 2).copied().and_then(hex_value)) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// `unquote(s)`: escapes in each ASCII run decoded as UTF-8 with replacement, other characters kept.
pub fn unquote(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut start = 0;
    let mut ascii = s.starts_with(|c: char| c.is_ascii());
    for (i, c) in s.char_indices() {
        if c.is_ascii() != ascii {
            push_run(&mut out, &s[start..i], ascii);
            start = i;
            ascii = c.is_ascii();
        }
    }
    push_run(&mut out, &s[start..], ascii);
    out
}

fn push_run(out: &mut String, run: &str, ascii: bool) {
    if ascii {
        out.push_str(&String::from_utf8_lossy(&unquote_to_bytes(run)));
    } else {
        out.push_str(run);
    }
}

/// `unquote_plus(s)`.
pub fn unquote_plus(s: &str) -> String {
    unquote(&s.replace('+', " "))
}

/// `parse_qsl(qs, keep_blank_values=True)`.
pub fn parse_qsl_blank(qs: &str) -> Vec<(String, String)> {
    qs.split('&')
        .filter(|field| !field.is_empty())
        .map(|field| {
            let (name, value) = field.split_once('=').unwrap_or((field, ""));
            (unquote_plus(name), unquote_plus(value))
        })
        .collect()
}

/// `quote(s, safe)`: bytes outside the always-safe set and `safe` percent-encoded.
pub fn quote_bytes(bytes: &[u8], safe: &str) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') || (b.is_ascii() && safe.as_bytes().contains(&b)) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `unquote(s, errors="surrogateescape")` encoded back with `surrogateescape`: the bytes the string stands for.
pub fn unquote_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let (Some(hi), Some(lo)) = (bytes.get(i + 1).copied().and_then(hex_value), bytes.get(i + 2).copied().and_then(hex_value)) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// `quote_plus(s, safe="")`.
pub fn quote_plus(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `urlencode(pairs)`.
pub fn urlencode(pairs: &[(String, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v))).collect::<Vec<_>>().join("&")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(url: &str) -> (String, String, String, String, String) {
        let s = urlsplit(url).unwrap();
        (s.scheme, s.netloc, s.path, s.query, s.fragment)
    }

    #[test]
    fn splits_as_python() {
        let owned = |p: [&str; 5]| (p[0].to_string(), p[1].to_string(), p[2].to_string(), p[3].to_string(), p[4].to_string());
        assert_eq!(split("HTTPS://Ex.com/a?b=1#c?d"), owned(["https", "Ex.com", "/a", "b=1", "c?d"]));
        assert_eq!(split(" \x01https://e\t.com/p\n"), owned(["https", "e.com", "/p", "", ""]));
        assert_eq!(split("1http://x"), owned(["", "", "1http://x", "", ""]));
        assert_eq!(split("mailto:a@b"), owned(["mailto", "", "a@b", "", ""]));
        assert_eq!(split("http://[2001:db8::1]:80/x"), owned(["http", "[2001:db8::1]:80", "/x", "", ""]));
        assert!(urlsplit("http://[2001:db8::1/x").is_err());
        assert!(urlsplit("http://[192.0.2.1]/").is_err());
        assert!(urlsplit("http://a\u{2100}b/").is_err());
        assert!(urlsplit("http://[2001:db8::1%]/").is_err() && urlsplit("http://u[x]@example.com/").is_err());
        assert_eq!(split("http://[2001:db8::1%eth0]/").1, "[2001:db8::1%eth0]");
    }

    #[test]
    fn unsplits_as_python() {
        for url in ["https://e.com/a?b#c", "https://e.com", "x//y", "//h/p", "file:///tmp"] {
            assert_eq!(urlunsplit(&urlsplit(url).unwrap()), url, "{url}");
        }
        assert_eq!(urlunsplit(&urlsplit("https://e.com/?#").unwrap()), "https://e.com/");
        assert_eq!(urlunsplit(&urlsplit("https:").unwrap()), "https://");
        assert_eq!(urlunsplit(&urlsplit("http:/x").unwrap()), "http:///x");
    }

    #[test]
    fn queries_as_python() {
        let pairs = parse_qsl_blank("a=1&&b&c=%2Fx+y&%E2%82=%zz&d=%C3%A9é");
        let want = [("a", "1"), ("b", ""), ("c", "/x y"), ("\u{fffd}", "%zz"), ("d", "\u{e9}\u{e9}")];
        assert_eq!(pairs, want.map(|(k, v)| (k.to_string(), v.to_string())));
        assert_eq!(urlencode(&pairs), "a=1&b=&c=%2Fx+y&%EF%BF%BD=%25zz&d=%C3%A9%C3%A9");
        assert_eq!(quote_plus("a b+c~/"), "a+b%2Bc~%2F");
    }
}

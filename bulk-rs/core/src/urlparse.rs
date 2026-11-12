//! Python 3.13's `urllib.parse`: `urlsplit`, `urlunsplit`, `parse_qsl`, `unquote_plus` and `urlencode` of str values.

use std::net::Ipv6Addr;
use std::sync::LazyLock;

use anyhow::{bail, Result};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

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

static IPV_FUTURE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\Av[a-fA-F0-9]+\..+\z").expect("pattern compiles"));

/// `urllib.parse.SplitResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitResult {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

fn is_scheme_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
}

/// `urlsplit(url)`, with its ValueErrors for bad brackets and NFKC-unsafe hosts.
pub fn urlsplit(url: &str) -> Result<SplitResult> {
    let url: String = url.trim_start_matches(|c: char| c <= ' ').chars().filter(|c| !matches!(c, '\t' | '\r' | '\n')).collect();
    let mut split = SplitResult::default();
    let mut rest = url.as_str();
    if let Some(i) = url.find(':') {
        if i > 0 && url.starts_with(|c: char| c.is_ascii_alphabetic()) && url[..i].chars().all(is_scheme_char) {
            split.scheme = url[..i].to_ascii_lowercase();
            rest = &url[i + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        let delim = after.find(['/', '?', '#']).unwrap_or(after.len());
        split.netloc = after[..delim].to_string();
        rest = &after[delim..];
        let (open, close) = (split.netloc.contains('['), split.netloc.contains(']'));
        if open != close {
            bail!("Invalid IPv6 URL");
        }
        if open {
            check_bracketed_netloc(&split.netloc)?;
        }
    }
    if let Some((before, fragment)) = rest.split_once('#') {
        split.fragment = fragment.to_string();
        rest = before;
    }
    if let Some((before, query)) = rest.split_once('?') {
        split.query = query.to_string();
        rest = before;
    }
    split.path = rest.to_string();
    check_netloc(&split.netloc)?;
    Ok(split)
}

fn check_bracketed_netloc(netloc: &str) -> Result<()> {
    let host_and_port = netloc.rsplit_once('@').map_or(netloc, |(_, rest)| rest);
    let host = match host_and_port.split_once('[') {
        Some((before, bracketed)) => {
            if !before.is_empty() {
                bail!("Invalid IPv6 URL");
            }
            let (host, port) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            if !port.is_empty() && !port.starts_with(':') {
                bail!("Invalid IPv6 URL");
            }
            host
        }
        None => host_and_port.split_once(':').map_or(host_and_port, |(host, _)| host),
    };
    if host.starts_with('v') {
        if !IPV_FUTURE.is_match(host) {
            bail!("IPvFuture address is invalid");
        }
        return Ok(());
    }
    let (address, scope) = host.split_once('%').map_or((host, None), |(address, scope)| (address, Some(scope)));
    if address.parse::<Ipv6Addr>().is_err() || scope.is_some_and(|s| s.is_empty() || s.contains('%')) {
        bail!("{host:?} does not appear to be an IPv6 address");
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
        bail!("netloc '{netloc}' contains invalid characters under NFKC normalization");
    }
    Ok(())
}

/// `urlunsplit(parts)`.
pub fn urlunsplit(parts: &SplitResult) -> String {
    let mut url = parts.path.clone();
    if !parts.netloc.is_empty() {
        if !url.is_empty() && !url.starts_with('/') {
            url.insert(0, '/');
        }
        url = format!("//{}{url}", parts.netloc);
    } else if url.starts_with("//") || (!parts.scheme.is_empty() && USES_NETLOC.contains(&parts.scheme.as_str()) && (url.is_empty() || url.starts_with('/'))) {
        url = format!("//{url}");
    }
    if !parts.scheme.is_empty() {
        url = format!("{}:{url}", parts.scheme);
    }
    if !parts.query.is_empty() {
        url = format!("{url}?{}", parts.query);
    }
    if !parts.fragment.is_empty() {
        url = format!("{url}#{}", parts.fragment);
    }
    url
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

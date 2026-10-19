//! Page URLs canonicalised and keyed exactly as `app.canonical` does with Python's urllib.

use std::net::{Ipv4Addr, Ipv6Addr};

use sha1::{Digest, Sha1};
use unicode_normalization::UnicodeNormalization;

use crate::text;

const USES_NETLOC: [&str; 27] = [
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
const TRACKING: [&str; 7] = ["fbclid", "gclid", "mc_cid", "mc_eid", "ref", "cmpid", "ito"];
const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// A URL Python would reject with a ValueError.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invalid;

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a URL Python's urllib rejects")
    }
}

impl std::error::Error for Invalid {}

/// The five parts of Python's `urlsplit`, empty where Python's are.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

pub fn canonicalize(url: &str) -> Result<String, Invalid> {
    let parts = split(text::strip(url))?;
    let mut query = String::with_capacity(parts.query.len());
    for (name, value) in parse_qsl(&parts.query) {
        if tracking(&name) {
            continue;
        }
        if !query.is_empty() {
            query.push('&');
        }
        quote_plus(&name, &mut query);
        query.push('=');
        quote_plus(&value, &mut query);
    }
    Ok(unsplit(&parts.scheme, &parts.netloc.to_lowercase(), &parts.path, &query))
}

pub fn domain_of(url: &str) -> Result<String, Invalid> {
    let host = split(url)?.netloc.to_lowercase();
    Ok(match host.strip_prefix("www.") {
        Some(bare) => bare.to_string(),
        None => host,
    })
}

pub fn url_sha1(url: &str) -> Result<String, Invalid> {
    Ok(sha1_hex(canonicalize(url)?.as_bytes()))
}

pub fn sha1_hex(data: &[u8]) -> String {
    hex(&Sha1::digest(data))
}

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

/// Python's `urllib.parse.urlsplit(url)`.
pub fn split(url: &str) -> Result<Parts, Invalid> {
    let url: String = url.trim_start_matches(|c: char| c <= ' ').chars().filter(|c| !matches!(c, '\t' | '\r' | '\n')).collect();
    let mut rest = url.as_str();
    let mut parts = Parts::default();
    if let Some(i) = rest.find(':') {
        let bytes = rest.as_bytes();
        if i > 0 && bytes[0].is_ascii_alphabetic() && bytes[..i].iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')) {
            parts.scheme = rest[..i].to_ascii_lowercase();
            rest = &rest[i + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        let netloc = &after[..end];
        if netloc.contains('[') != netloc.contains(']') {
            return Err(Invalid);
        }
        if netloc.contains('[') {
            check_bracketed_netloc(netloc)?;
        }
        parts.netloc = netloc.to_string();
        rest = &after[end..];
    }
    if let Some((before, fragment)) = rest.split_once('#') {
        parts.fragment = fragment.to_string();
        rest = before;
    }
    if let Some((before, query)) = rest.split_once('?') {
        parts.query = query.to_string();
        rest = before;
    }
    check_netloc(&parts.netloc)?;
    parts.path = rest.to_string();
    Ok(parts)
}

fn unsplit(scheme: &str, netloc: &str, path: &str, query: &str) -> String {
    let mut url = String::with_capacity(scheme.len() + netloc.len() + path.len() + query.len() + 8);
    if !scheme.is_empty() {
        url.push_str(scheme);
        url.push(':');
    }
    if !netloc.is_empty() {
        url.push_str("//");
        url.push_str(netloc);
        if !path.is_empty() && !path.starts_with('/') {
            url.push('/');
        }
    } else if path.starts_with("//") || (!scheme.is_empty() && USES_NETLOC.contains(&scheme) && (path.is_empty() || path.starts_with('/'))) {
        url.push_str("//");
    }
    url.push_str(path);
    if !query.is_empty() {
        url.push('?');
        url.push_str(query);
    }
    url
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

fn check_netloc(netloc: &str) -> Result<(), Invalid> {
    if netloc.is_ascii() {
        return Ok(());
    }
    let bare: String = netloc.chars().filter(|c| !matches!(c, '@' | ':' | '#' | '?')).collect();
    let normalized: String = bare.nfkc().collect();
    if bare != normalized && normalized.contains(['/', '?', '#', '@', ':']) {
        return Err(Invalid);
    }
    Ok(())
}

/// Python's `parse_qsl(query, keep_blank_values=True)`.
pub fn parse_qsl(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            (unquote_plus(name), unquote_plus(value))
        })
        .collect()
}

/// Python's `unquote_plus`: escapes decoded within each ASCII run as UTF-8 with replacement characters.
pub fn unquote_plus(s: &str) -> String {
    let s = s.replace('+', " ");
    if !s.contains('%') {
        return s;
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s.as_str();
    while !rest.is_empty() {
        let ascii = rest.bytes().take_while(u8::is_ascii).count();
        if ascii > 0 {
            out.push_str(&String::from_utf8_lossy(&percent_decode(&rest.as_bytes()[..ascii])));
            rest = &rest[ascii..];
        }
        let other = rest.char_indices().find(|(_, c)| c.is_ascii()).map_or(rest.len(), |(i, _)| i);
        out.push_str(&rest[..other]);
        rest = &rest[other..];
    }
    out
}

fn percent_decode(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() && bytes[i + 1].is_ascii_hexdigit() && bytes[i + 2].is_ascii_hexdigit() {
            out.push(hex_value(bytes[i + 1]) * 16 + hex_value(bytes[i + 2]));
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

/// Python's `quote_plus(s)`, appended to `out`.
pub fn quote_plus(s: &str, out: &mut String) {
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') {
            out.push(b as char);
        } else if b == b' ' {
            out.push('+');
        } else {
            out.extend(['%', HEX[(b >> 4) as usize] as char, HEX[(b & 15) as usize] as char]);
        }
    }
}

/// The query names `TRACKING` in `app.canonical` matches, with Python's case-insensitive `^` and `$`.
fn tracking(name: &str) -> bool {
    let folded: String = name.chars().map(text::fold).collect();
    if folded.starts_with("utm_") {
        return true;
    }
    let bare = folded.strip_suffix('\n').unwrap_or(&folded);
    TRACKING.contains(&bare)
}

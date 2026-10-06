//! Page URLs canonicalised and keyed exactly as `app.canonical` does with Python's html and urllib.

use std::borrow::Cow;
use std::net::{Ipv4Addr, Ipv6Addr};

use sha1::{Digest, Sha1};
use unicode_normalization::UnicodeNormalization;

use crate::entities::ENTITIES;
use crate::text;

const USES_NETLOC: [&str; 27] = [
    "", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https", "shttp", "snews",
    "prospero", "rtsp", "rtsps", "rtspu", "rsync", "svn", "svn+ssh", "sftp", "nfs", "git", "git+ssh", "ws", "wss",
    "itms-services",
];
const TRACKING: [&str; 7] = ["fbclid", "gclid", "mc_cid", "mc_eid", "ref", "cmpid", "ito"];
const HEX: &[u8; 16] = b"0123456789ABCDEF";
/// Python's `int()` refuses longer decimal strings.
const MAX_INT_DIGITS: usize = 4300;
const LONGEST_NAME: usize = 32;
const WINDOWS_1252: [char; 32] = [
    '\u{20ac}', '\u{81}', '\u{201a}', '\u{192}', '\u{201e}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{2c6}', '\u{2030}',
    '\u{160}', '\u{2039}', '\u{152}', '\u{8d}', '\u{17d}', '\u{8f}', '\u{90}', '\u{2018}', '\u{2019}', '\u{201c}',
    '\u{201d}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{2dc}', '\u{2122}', '\u{161}', '\u{203a}', '\u{153}', '\u{9d}',
    '\u{17e}', '\u{178}',
];

/// A URL Python would reject with a ValueError.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invalid;

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a URL Python's urllib or html module rejects")
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
    let unescaped = unescape(text::strip(url))?;
    let parts = split(&unescaped)?;
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
        if i > 0
            && bytes[0].is_ascii_alphabetic()
            && bytes[..i].iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
        {
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

/// Python's `html.unescape`.
pub fn unescape(s: &str) -> Result<Cow<'_, str>, Invalid> {
    if !s.contains('&') {
        return Ok(Cow::Borrowed(s));
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        match reference(after, &mut out)? {
            Some(used) => rest = &after[used..],
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    Ok(Cow::Owned(out))
}

/// The reference after an `&`, written to `out`, and the bytes it used; None where Python's pattern does not match.
fn reference(after: &str, out: &mut String) -> Result<Option<usize>, Invalid> {
    let bytes = after.as_bytes();
    let semicolon = |end: usize| end + usize::from(bytes.get(end) == Some(&b';'));
    if bytes.first() == Some(&b'#') {
        let digits = bytes[1..].iter().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 {
            if digits > MAX_INT_DIGITS {
                return Err(Invalid);
            }
            let number = after[1..1 + digits].bytes().fold(0u64, |n, d| n.saturating_mul(10).saturating_add(u64::from(d - b'0')));
            push_number(number, out);
            return Ok(Some(semicolon(1 + digits)));
        }
        if matches!(bytes.get(1), Some(b'x' | b'X')) {
            let digits = bytes[2..].iter().take_while(|b| b.is_ascii_hexdigit()).count();
            if digits > 0 {
                let number = after[2..2 + digits].bytes().fold(0u64, |n, d| n.saturating_mul(16).saturating_add(u64::from(hex_value(d))));
                push_number(number, out);
                return Ok(Some(semicolon(2 + digits)));
            }
        }
        return Ok(None);
    }
    let mut end = 0;
    let mut count = 0;
    for (i, c) in after.char_indices() {
        if count == LONGEST_NAME || matches!(c, '\t' | '\n' | '\u{c}' | ' ' | '<' | '&' | '#' | ';') {
            break;
        }
        count += 1;
        end = i + c.len_utf8();
    }
    if count == 0 {
        return Ok(None);
    }
    let used = semicolon(end);
    let name = &after[..used];
    if let Some(value) = entity(name) {
        out.push_str(value);
        return Ok(Some(used));
    }
    let bounds: Vec<usize> = name.char_indices().map(|(i, _)| i).collect();
    for &cut in bounds[2.min(bounds.len())..].iter().rev() {
        if let Some(value) = entity(&name[..cut]) {
            out.push_str(value);
            out.push_str(&name[cut..]);
            return Ok(Some(used));
        }
    }
    out.push('&');
    out.push_str(name);
    Ok(Some(used))
}

fn entity(name: &str) -> Option<&'static str> {
    ENTITIES.binary_search_by(|(known, _)| known.as_bytes().cmp(name.as_bytes())).ok().map(|i| ENTITIES[i].1)
}

fn push_number(number: u64, out: &mut String) {
    match number {
        0 => out.push('\u{fffd}'),
        0x0d => out.push('\r'),
        0x80..=0x9f => out.push(WINDOWS_1252[(number - 0x80) as usize]),
        0xd800..=0xdfff => out.push('\u{fffd}'),
        n if n > 0x10ffff => out.push('\u{fffd}'),
        0x01..=0x08 | 0x0b | 0x0e..=0x1f | 0x7f | 0xfdd0..=0xfdef => {}
        n if n & 0xfffe == 0xfffe => {}
        n => out.extend(char::from_u32(n as u32)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_entity_prefix_wins() {
        assert_eq!(unescape("a&ampx=1&copy=2&notit;&#x41;&#0;&#128;&bogus;").unwrap(), "a&x=1\u{a9}=2\u{ac}it;A\u{fffd}\u{20ac}&bogus;");
    }
}

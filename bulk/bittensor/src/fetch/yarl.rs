//! The URLs aiohttp requests and reports, as yarl builds them: `URL(text)`, `str(url)`, `url.join(other)` and the request target.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

use bulk_core::py::ValueError;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

const SCHEME_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";
const ALLOWED: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$'()*,";
const QS: &str = "+&=;";
const DEFAULT_PORTS: [(&str, u16); 5] = [("http", 80), ("https", 443), ("ws", 80), ("wss", 443), ("ftp", 21)];
const REQUIRES_HOST: [&str; 5] = ["http", "https", "ws", "wss", "ftp"];

static IPV_FUTURE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\Av[a-fA-F0-9]+\..+\z").expect("pattern compiles"));

/// One `yarl._quoting._Quoter`.
struct Quoter {
    safe: String,
    protected: &'static str,
    qs: bool,
}

impl Quoter {
    fn new(safe: &str, protected: &'static str, qs: bool) -> Quoter {
        let mut all = format!("{safe}{ALLOWED}");
        if !qs {
            all.push_str(QS);
        }
        all.push_str(protected);
        Quoter { safe: all, protected, qs }
    }

    fn quote(&self, value: &str) -> String {
        let bytes = value.as_bytes();
        let mut out = String::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let ch = bytes[i];
            i += 1;
            if ch == b'%' {
                if i == bytes.len() {
                    out.push_str("%25");
                    continue;
                }
                if i + 1 == bytes.len() {
                    out.push_str("%25");
                    continue;
                }
                let (a, b) = (bytes[i].to_ascii_uppercase(), bytes[i + 1].to_ascii_uppercase());
                let value = (a as char).to_digit(16).zip((b as char).to_digit(16));
                let Some((hi, lo)) = value.filter(|_| a.is_ascii_alphanumeric() && b.is_ascii_alphanumeric()) else {
                    out.push_str("%25");
                    continue;
                };
                let unquoted = (hi * 16 + lo) as u8;
                if unquoted.is_ascii() && self.protected.as_bytes().contains(&unquoted) {
                    out.push_str(&format!("%{}{}", a as char, b as char));
                } else if unquoted.is_ascii() && self.safe.as_bytes().contains(&unquoted) {
                    out.push(unquoted as char);
                } else {
                    out.push_str(&format!("%{}{}", a as char, b as char));
                }
                i += 2;
                continue;
            }
            if self.qs && ch == b' ' {
                out.push('+');
            } else if ch.is_ascii() && self.safe.as_bytes().contains(&ch) {
                out.push(ch as char);
            } else {
                out.push_str(&format!("%{ch:02X}"));
            }
        }
        out
    }
}

static REQUOTER: LazyLock<Quoter> = LazyLock::new(|| Quoter::new("", "", false));
static PATH_REQUOTER: LazyLock<Quoter> = LazyLock::new(|| Quoter::new("@:", "/+", false));
static QUERY_REQUOTER: LazyLock<Quoter> = LazyLock::new(|| Quoter::new("?/:@", "=+&;", true));
static FRAGMENT_REQUOTER: LazyLock<Quoter> = LazyLock::new(|| Quoter::new("?/:@", "", false));

/// A URL as yarl holds it once encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Url {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
    user: Option<String>,
    password: Option<String>,
    host: String,
    port: Option<u16>,
}

fn error(message: impl Into<String>) -> ValueError {
    ValueError(message.into())
}

/// `normalize_path`: `.` and `..` segments dropped, the root kept.
fn normalize_path(path: &str) -> String {
    let (prefix, rest) = match path.strip_prefix('/') {
        Some(rest) => ("/", rest),
        None => ("", path),
    };
    let segments: Vec<&str> = rest.split('/').collect();
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
    format!("{prefix}{}", resolved.join("/"))
}

fn relative_scheme_colon(path: &str) -> String {
    match path.find(':') {
        Some(at) if at > 0 && path[..at].chars().all(|c| SCHEME_CHARS.contains(c)) => format!("{}%3A{}", &path[..at], &path[at + 1..]),
        _ => path.to_string(),
    }
}

/// Python's `IPv6Address.compressed`: lowercase groups, the longest run of two or more zero groups as `::`.
fn compressed_v6(address: &Ipv6Addr) -> String {
    let groups = address.segments();
    let (mut best, mut best_len, mut at, mut len) = (0usize, 0usize, 0usize, 0usize);
    for (i, group) in groups.iter().enumerate() {
        if *group == 0 {
            if len == 0 {
                at = i;
            }
            len += 1;
            if len > best_len {
                best = at;
                best_len = len;
            }
        } else {
            len = 0;
        }
    }
    let hex: Vec<String> = groups.iter().map(|g| format!("{g:x}")).collect();
    if best_len < 2 {
        return hex.join(":");
    }
    let head = hex[..best].join(":");
    let tail = hex[best + best_len..].join(":");
    format!("{head}::{tail}")
}

/// `_encode_host`: IP addresses in their compressed forms, ASCII names lowercased, others IDNA-encoded.
fn encode_host(host: &str) -> Result<String, ValueError> {
    if host.chars().last().is_some_and(|c| c.is_numeric()) || host.contains(':') {
        let part = if host.contains("%25") { "%25" } else { "%" };
        let (raw_ip, sep, zone) = match host.split_once(part) {
            Some((ip, zone)) => (ip, part, zone),
            None => (host, "", ""),
        };
        if let Ok(v4) = raw_ip.parse::<Ipv4Addr>() {
            return Ok(format!("{v4}{sep}{zone}"));
        }
        if let Ok(v6) = raw_ip.parse::<Ipv6Addr>() {
            return Ok(format!("[{}{sep}{zone}]", compressed_v6(&v6)));
        }
    }
    if host.is_ascii() {
        return Ok(host.to_ascii_lowercase());
    }
    idna::domain_to_ascii(host).map_err(|_| error(format!("Host {} cannot be IDNA-encoded", bulk_core::py::repr_str(host))))
}

impl Url {
    /// `URL(text)`: the text split, then its host, path, query and fragment encoded as yarl encodes them.
    pub fn parse(text: &str) -> Result<Url, ValueError> {
        let text: String = text.trim_start_matches(|c: char| c <= ' ').chars().filter(|c| !matches!(c, '\t' | '\r' | '\n')).collect();
        let mut rest = text.as_str();
        let mut scheme = String::new();
        if let Some(i) = rest.find(':') {
            if i > 0 && rest[..i].chars().all(|c| SCHEME_CHARS.contains(c)) {
                scheme = rest[..i].to_ascii_lowercase();
                rest = &rest[i + 1..];
            }
        }
        let (has_hash, has_question) = (rest.contains('#'), rest.contains('?'));
        let mut netloc = String::new();
        if let Some(after) = rest.strip_prefix("//") {
            let delims: &[char] = match (has_hash, has_question) {
                (true, true) => &['/', '?', '#'],
                (false, true) => &['/', '?'],
                (true, false) => &['/', '#'],
                (false, false) => &['/'],
            };
            let end = after.find(delims).unwrap_or(after.len());
            netloc = after[..end].to_string();
            rest = &after[end..];
            if netloc.contains('\\') {
                return Err(error("Invalid URL: backslash ('\\') is not allowed in the authority component per RFC 3986."));
            }
            let (open, close) = (netloc.contains('['), netloc.contains(']'));
            if open != close {
                return Err(error("Invalid IPv6 URL"));
            }
            if open {
                let hostinfo = netloc.rsplit_once('@').map_or(netloc.as_str(), |(_, h)| h);
                if !hostinfo.starts_with('[') || hostinfo.matches('[').count() > 1 || hostinfo.matches(']').count() > 1 {
                    return Err(error("Invalid IPv6 URL"));
                }
                let (bracketed, after) = hostinfo[1..].split_once(']').unwrap_or((&hostinfo[1..], ""));
                if !after.is_empty() && !after.starts_with(':') {
                    return Err(error("Invalid IPv6 URL"));
                }
                if bracketed.starts_with('v') {
                    if !IPV_FUTURE.is_match(bracketed) {
                        return Err(error("IPvFuture address is invalid"));
                    }
                } else if !bracketed.contains(':') {
                    return Err(error("The IPv6 content between brackets is not valid"));
                }
            }
        }
        let mut fragment = String::new();
        let mut query = String::new();
        let mut path = rest.to_string();
        if has_hash {
            if let Some((before, after)) = path.clone().split_once('#') {
                path = before.to_string();
                fragment = after.to_string();
            }
        }
        if has_question {
            if let Some((before, after)) = path.clone().split_once('?') {
                path = before.to_string();
                query = after.to_string();
            }
        }
        if !netloc.is_empty() && !netloc.is_ascii() {
            let n: String = netloc.chars().filter(|c| !matches!(c, '@' | ':' | '#' | '?')).collect();
            let normalized: String = n.nfkc().collect();
            if n != normalized && normalized.contains(['/', '?', '#', '@', ':', '%']) {
                return Err(error(format!("netloc '{netloc}' contains invalid characters under NFKC normalization")));
            }
        }
        let (mut user, mut password, mut host, mut port) = (None, None, String::new(), None);
        if !netloc.is_empty() {
            let (u, p, h, explicit) = split_netloc(&netloc)?;
            let h = match h {
                Some(h) => h,
                None if REQUIRES_HOST.contains(&scheme.as_str()) => {
                    return Err(error(format!("Invalid URL: host is required for absolute urls with the {scheme} scheme")));
                }
                None => String::new(),
            };
            host = encode_host(&h)?;
            port = explicit;
            user = u.map(|u| if u.is_empty() { u } else { REQUOTER.quote(&u) });
            password = p.map(|p| if p.is_empty() { p } else { REQUOTER.quote(&p) });
            netloc = make_netloc(user.as_deref(), password.as_deref(), &host, port);
        }
        if !path.is_empty() {
            path = PATH_REQUOTER.quote(&path);
            if !netloc.is_empty() && path.contains('.') {
                path = normalize_path(&path);
            } else if scheme.is_empty() && netloc.is_empty() {
                path = relative_scheme_colon(&path);
            }
        }
        if !query.is_empty() {
            query = QUERY_REQUOTER.quote(&query);
        }
        if !fragment.is_empty() {
            fragment = FRAGMENT_REQUOTER.quote(&fragment);
        }
        Ok(Url { scheme, netloc, path, query, fragment, user, password, host, port })
    }

    fn from_parts(base: &Url, scheme: &str, netloc_of: &Url, path: String, query: String, fragment: String) -> Url {
        let path = if !path.is_empty() && scheme.is_empty() && netloc_of.netloc.is_empty() && path.contains(':') { relative_scheme_colon(&path) } else { path };
        let _ = base;
        Url {
            scheme: scheme.to_string(),
            netloc: netloc_of.netloc.clone(),
            path,
            query,
            fragment,
            user: netloc_of.user.clone(),
            password: netloc_of.password.clone(),
            host: netloc_of.host.clone(),
            port: netloc_of.port,
        }
    }

    /// The host, empty for a relative URL.
    pub fn raw_host(&self) -> &str {
        &self.host
    }

    /// `url.join(other)`, as aiohttp resolves a redirect's relative `Location`.
    pub fn join(&self, other: &Url) -> Url {
        let scheme = if other.scheme.is_empty() { self.scheme.clone() } else { other.scheme.clone() };
        if scheme != self.scheme || !bulk_core::urlparse::USES_RELATIVE.contains(&scheme.as_str()) {
            return other.clone();
        }
        if !other.netloc.is_empty() && bulk_core::urlparse::USES_NETLOC.contains(&scheme.as_str()) {
            return Url::from_parts(self, &scheme, other, other.path.clone(), other.query.clone(), other.fragment.clone());
        }
        let orig = &self.path;
        let path = if !other.path.is_empty() {
            let joined = if other.path.starts_with('/') {
                other.path.clone()
            } else if orig.is_empty() {
                format!("/{}", other.path)
            } else if orig.ends_with('/') {
                format!("{orig}{}", other.path)
            } else {
                let parts = self.raw_parts();
                let mut kept: Vec<&str> = parts[..parts.len() - 1].iter().map(String::as_str).collect();
                kept.push("");
                let mut path = format!("{}{}", kept.join("/"), other.path);
                if orig.starts_with('/') {
                    path.remove(0);
                }
                path
            };
            if joined.contains('.') {
                normalize_path(&joined)
            } else {
                joined
            }
        } else {
            orig.clone()
        };
        let keep = !other.path.is_empty();
        let query = if keep || !other.query.is_empty() { other.query.clone() } else { self.query.clone() };
        let fragment = if keep || !other.fragment.is_empty() { other.fragment.clone() } else { self.fragment.clone() };
        Url::from_parts(self, &scheme, self, path, query, fragment)
    }

    fn raw_parts(&self) -> Vec<String> {
        let path = &self.path;
        if !self.netloc.is_empty() {
            if path.is_empty() {
                return vec!["/".into()];
            }
            return std::iter::once("/".to_string()).chain(path[1..].split('/').map(str::to_string)).collect();
        }
        if let Some(rest) = path.strip_prefix('/') {
            return std::iter::once("/".to_string()).chain(rest.split('/').map(str::to_string)).collect();
        }
        path.split('/').map(str::to_string).collect()
    }

    /// The URL without its fragment, as a response reports the URL it answered.
    pub fn without_fragment(&self) -> Url {
        Url { fragment: String::new(), ..self.clone() }
    }

    /// The request target: the encoded path and query.
    pub fn raw_path_qs(&self) -> String {
        let path = if self.path.is_empty() && !self.netloc.is_empty() { "/" } else { self.path.as_str() };
        if self.query.is_empty() {
            path.to_string()
        } else {
            format!("{path}?{}", self.query)
        }
    }
}

impl std::fmt::Display for Url {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path =
            if self.path.is_empty() && !self.netloc.is_empty() && (!self.query.is_empty() || !self.fragment.is_empty()) { "/" } else { self.path.as_str() };
        let default = DEFAULT_PORTS.iter().find(|(s, _)| *s == self.scheme).map(|(_, p)| *p);
        let netloc = if self.port.is_some() && self.port == default {
            make_netloc(self.user.as_deref(), self.password.as_deref(), &self.host, None)
        } else {
            self.netloc.clone()
        };
        f.write_str(&unsplit(&self.scheme, &netloc, path, &self.query, &self.fragment))
    }
}

fn unsplit(scheme: &str, netloc: &str, path: &str, query: &str, fragment: &str) -> String {
    let mut url = if !netloc.is_empty() || (!scheme.is_empty() && bulk_core::urlparse::USES_NETLOC.contains(&scheme)) || path.starts_with("//") {
        if !path.is_empty() && !path.starts_with('/') {
            if scheme.is_empty() {
                format!(":{path}")
            } else {
                format!("{scheme}://{netloc}/{path}")
            }
        } else if scheme.is_empty() {
            format!("//{netloc}{path}")
        } else {
            format!("{scheme}://{netloc}{path}")
        }
    } else if !scheme.is_empty() {
        format!("{scheme}:{path}")
    } else {
        path.to_string()
    };
    if !query.is_empty() {
        url = format!("{url}?{query}");
    }
    if !fragment.is_empty() {
        url = format!("{url}#{fragment}");
    }
    url
}

fn make_netloc(user: Option<&str>, password: Option<&str>, host: &str, port: Option<u16>) -> String {
    let mut ret = host.to_string();
    if let Some(port) = port {
        ret = format!("{ret}:{port}");
    }
    match (user, password) {
        (None, None) => ret,
        (user, Some(password)) => format!("{}:{password}@{ret}", user.unwrap_or("")),
        (Some(user), None) if !user.is_empty() => format!("{user}@{ret}"),
        _ => ret,
    }
}

type Netloc = (Option<String>, Option<String>, Option<String>, Option<u16>);

fn split_netloc(netloc: &str) -> Result<Netloc, ValueError> {
    let (user, password, hostinfo) = match netloc.rsplit_once('@') {
        None => (None, None, netloc),
        Some((userinfo, hostinfo)) => match userinfo.split_once(':') {
            Some((user, password)) => (Some(user.to_string()), Some(password.to_string()), hostinfo),
            None => (Some(userinfo.to_string()), None, hostinfo),
        },
    };
    let (hostname, port) = if hostinfo.contains('[') {
        if !hostinfo.starts_with('[') || hostinfo.matches('[').count() > 1 || hostinfo.matches(']').count() > 1 {
            return Err(error("Invalid IPv6 URL"));
        }
        let (hostname, after) = hostinfo[1..].split_once(']').unwrap_or((&hostinfo[1..], ""));
        if !after.is_empty() && !after.starts_with(':') {
            return Err(error("Invalid IPv6 URL"));
        }
        (hostname, after.split_once(':').map_or("", |(_, p)| p))
    } else {
        hostinfo.split_once(':').unwrap_or((hostinfo, ""))
    };
    let user = user.filter(|u| !u.is_empty());
    let host = (!hostname.is_empty()).then(|| hostname.to_string());
    if port.is_empty() {
        return Ok((user, password, host, None));
    }
    let number: u32 = port.parse().map_err(|_| error("Invalid URL: port can't be converted to integer"))?;
    let number = u16::try_from(number).map_err(|_| error("Port out of range 0-65535"))?;
    Ok((user, password, host, Some(number)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_print_as_yarl() {
        for (given, want) in [
            ("https://A.io:443/x", "https://a.io/x"),
            ("https://a.io/a b?q=1 2#f", "https://a.io/a%20b?q=1+2#f"),
            ("https://a.io/%7e/%7E/~", "https://a.io/~/~/~"),
            ("https://a.io/caf\u{e9}", "https://a.io/caf%C3%A9"),
            ("https://b\u{fc}cher.de/", "https://xn--bcher-kva.de/"),
            ("https://a.io/x/../y/./z", "https://a.io/y/z"),
            ("https://a.io", "https://a.io"),
            ("https://a.io?x", "https://a.io/?x"),
            ("https://a.io/p?a=%2F&b=/&c=%zz", "https://a.io/p?a=/&b=/&c=%25zz"),
            ("https://a.io/%2F%2f", "https://a.io/%2F%2F"),
            ("https://a.io/[x]", "https://a.io/%5Bx%5D"),
            ("https://a.io:8443/", "https://a.io:8443/"),
        ] {
            assert_eq!(Url::parse(given).unwrap().to_string(), want, "{given}");
        }
        let base = Url::parse("https://a.io/dir/page?x=1").unwrap();
        for (location, want) in [
            ("/new", "https://a.io/new"),
            ("other", "https://a.io/dir/other"),
            ("../up", "https://a.io/up"),
            ("//c.io/z", "https://c.io/z"),
            ("?q=2", "https://a.io/dir/page?q=2"),
            ("#frag", "https://a.io/dir/page?x=1#frag"),
        ] {
            assert_eq!(base.join(&Url::parse(location).unwrap()).to_string(), want, "{location}");
        }
    }
}

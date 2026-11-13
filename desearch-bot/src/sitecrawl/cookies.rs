//! A cookie jar shared by every request of a run: `Set-Cookie` headers taken in, and the cookies a request's host and path match sent back.

use std::net::IpAddr;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;

#[derive(Clone, Debug)]
struct Cookie {
    value: String,
    host_only: bool,
    secure: bool,
    expires: Option<DateTime<Utc>>,
}

/// Cookies keyed by (domain, path, name), in the order they were first set.
#[derive(Default)]
pub struct Jar(Mutex<IndexMap<(String, String, String), Cookie>>);

fn is_ip(host: &str) -> bool {
    host.trim_matches(['[', ']']).parse::<IpAddr>().is_ok()
}

fn domain_match(host: &str, domain: &str) -> bool {
    host == domain || (host.ends_with(domain) && host[..host.len() - domain.len()].ends_with('.') && !is_ip(host))
}

fn path_match(request: &str, cookie: &str) -> bool {
    request == cookie || (request.starts_with(cookie) && (cookie.ends_with('/') || request[cookie.len()..].starts_with('/')))
}

fn default_path(path: &str) -> String {
    match path.rfind('/') {
        Some(i) if path.starts_with('/') && i > 0 => path[..i].to_string(),
        _ => "/".to_string(),
    }
}

impl Jar {
    /// Take the response's `Set-Cookie` headers for a request to this host and path.
    pub fn store(&self, host: &str, path: &str, headers: &[String]) {
        if is_ip(host) {
            return;
        }
        let now = Utc::now();
        let mut jar = self.0.lock().unwrap();
        for header in headers {
            let mut parts = header.split(';');
            let Some((name, value)) = parts.next().and_then(|pair| pair.split_once('=')) else { continue };
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let value = value.trim();
            let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value).to_string();
            let (mut domain, mut cookie_path, mut secure, mut max_age, mut expires) = (None, None, false, None, None);
            for attr in parts {
                let (key, val) = attr.split_once('=').map_or((attr.trim(), ""), |(k, v)| (k.trim(), v.trim()));
                match key.to_ascii_lowercase().as_str() {
                    "domain" if !val.is_empty() => domain = Some(val.trim_start_matches('.').to_lowercase()),
                    "path" if val.starts_with('/') => cookie_path = Some(val.to_string()),
                    "secure" => secure = true,
                    "max-age" => max_age = val.parse::<i64>().ok(),
                    "expires" => expires = DateTime::parse_from_rfc2822(val).ok().map(|t| t.with_timezone(&Utc)),
                    _ => {}
                }
            }
            let host_only = domain.is_none();
            let domain = match domain {
                Some(d) if !domain_match(host, &d) => continue,
                Some(d) => d,
                None => host.to_string(),
            };
            let expires = match max_age {
                Some(seconds) => Some(now + chrono::Duration::seconds(seconds.clamp(-1, 400 * 86_400))),
                None => expires,
            };
            let key = (domain, cookie_path.unwrap_or_else(|| default_path(path)), name.to_string());
            if expires.is_some_and(|t| t <= now) {
                jar.shift_remove(&key);
                continue;
            }
            jar.insert(key, Cookie { value, host_only, secure, expires });
        }
    }

    /// The name and value of every cookie the jar sends to this URL.
    pub fn filter(&self, scheme: &str, host: &str, path: &str) -> IndexMap<String, String> {
        let now = Utc::now();
        let jar = self.0.lock().unwrap();
        let path = if path.is_empty() { "/" } else { path };
        let mut out = IndexMap::new();
        for ((domain, cookie_path, name), cookie) in jar.iter() {
            let domain_ok = if cookie.host_only { host == domain } else { domain_match(host, domain) };
            if !domain_ok || !path_match(path, cookie_path) || (cookie.secure && scheme != "https") || cookie.expires.is_some_and(|t| t <= now) {
                continue;
            }
            out.insert(name.clone(), cookie.value.clone());
        }
        out
    }

    /// The `Cookie` header for a request to this URL, if any cookie goes with it.
    pub fn header(&self, scheme: &str, host: &str, path: &str) -> Option<String> {
        let found = self.filter(scheme, host, path);
        (!found.is_empty()).then(|| found.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_domain_cookie_set_by_a_subdomain_goes_to_the_parent() {
        let jar = Jar::default();
        jar.store("idp.pub.test", "/authorize", &["session=1; Domain=pub.test; Path=/".into(), "x=2; Domain=other.test".into()]);
        assert_eq!(jar.filter("https", "pub.test", "/art").get("session").map(String::as_str), Some("1"));
        assert!(jar.filter("https", "other.test", "/").is_empty(), "a domain the host is not under is refused");
        jar.store("a.test", "/x/y", &["h=1".into(), "s=1; Secure".into()]);
        assert!(jar.filter("https", "a.test", "/x/z").contains_key("h") && !jar.filter("https", "a.test", "/").contains_key("h"));
        assert!(!jar.filter("http", "a.test", "/x/").contains_key("s") && !jar.filter("https", "b.a.test", "/x/").contains_key("h"));
        jar.store("a.test", "/x/y", &["h=1; Max-Age=0".into()]);
        assert!(!jar.filter("https", "a.test", "/x/z").contains_key("h"));
        jar.store("127.0.0.1", "/", &["ip=1".into()]);
        assert!(jar.filter("https", "127.0.0.1", "/").is_empty());
    }
}

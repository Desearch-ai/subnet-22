//! The crawler's HTTP client, and the rule that keeps it off private networks.

use std::error::Error;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use hickory_resolver::config::{LookupIpStrategy, NameServerConfigGroup, ResolverConfig, ResolverOpts};
use hickory_resolver::error::ResolveError;
use hickory_resolver::TokioAsyncResolver;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::signing::USER_AGENT;

pub const RESOLVER: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const RESOLVER_PORT: u16 = 5335;
/// A big sitemap may take a while to arrive, but no request runs longer than this.
pub const MAX_REQUEST: Duration = Duration::from_secs(60);

const V4_PRIVATE: [([u8; 4], u32); 14] = [
    ([0, 0, 0, 0], 8),
    ([10, 0, 0, 0], 8),
    ([127, 0, 0, 0], 8),
    ([169, 254, 0, 0], 16),
    ([172, 16, 0, 0], 12),
    ([192, 0, 0, 0], 24),
    ([192, 0, 0, 170], 31),
    ([192, 0, 2, 0], 24),
    ([192, 168, 0, 0], 16),
    ([198, 18, 0, 0], 15),
    ([198, 51, 100, 0], 24),
    ([203, 0, 113, 0], 24),
    ([240, 0, 0, 0], 4),
    ([255, 255, 255, 255], 32),
];
const V4_EXCEPTIONS: [([u8; 4], u32); 2] = [([192, 0, 0, 9], 32), ([192, 0, 0, 10], 32)];
const V4_SHARED: ([u8; 4], u32) = ([100, 64, 0, 0], 10);
const V6_PRIVATE: [(u128, u32); 11] = [
    (1, 128),
    (0, 128),
    (0xffff_0000_0000, 96),
    (0x0064_ff9b_0001 << 80, 48),
    (0x0100 << 112, 64),
    (0x2001 << 112, 23),
    (0x2001_0db8 << 96, 32),
    (0x2002 << 112, 16),
    (0x3fff << 112, 20),
    (0xfc00 << 112, 7),
    (0xfe80 << 112, 10),
];
const V6_EXCEPTIONS: [(u128, u32); 6] = [
    ((0x2001_0001 << 96) | 1, 128),
    ((0x2001_0001 << 96) | 2, 128),
    (0x2001_0003 << 96, 32),
    (0x2001_0004_0112 << 80, 48),
    (0x2001_0020 << 96, 28),
    (0x2001_0030 << 96, 28),
];

pub(crate) fn within4(ip: Ipv4Addr, (net, prefix): ([u8; 4], u32)) -> bool {
    let mask = u32::MAX.checked_shl(32 - prefix).unwrap_or(0);
    u32::from(ip) & mask == u32::from(Ipv4Addr::from(net)) & mask
}

pub(crate) fn within6(ip: Ipv6Addr, (net, prefix): (u128, u32)) -> bool {
    let mask = u128::MAX.checked_shl(128 - prefix).unwrap_or(0);
    u128::from(ip) & mask == net & mask
}

/// Python's `ipaddress.ip_address(ip).is_global`.
pub fn is_global(ip: IpAddr) -> bool {
    let global4 = |v4: Ipv4Addr| {
        let private = V4_PRIVATE.iter().any(|&net| within4(v4, net)) && !V4_EXCEPTIONS.iter().any(|&net| within4(v4, net));
        !within4(v4, V4_SHARED) && !private
    };
    match ip {
        IpAddr::V4(v4) => global4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => global4(v4),
            None => !(V6_PRIVATE.iter().any(|&net| within6(v6, net)) && !V6_EXCEPTIONS.iter().any(|&net| within6(v6, net))),
        },
    }
}

/// False for an address literal inside a private network; names are checked when they resolve.
pub fn public_host(host: &str) -> bool {
    let literal = host.trim_matches(['[', ']']);
    let literal = literal.split_once('%').map_or(literal, |(address, _)| address);
    literal.parse::<IpAddr>().map_or(true, is_global)
}

#[derive(Debug)]
struct NotPublic(String);

impl fmt::Display for NotPublic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} resolves only to non-public addresses", self.0)
    }
}

impl Error for NotPublic {}

#[derive(Debug)]
struct Blocked(String);

impl fmt::Display for Blocked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} resolves into the abuse blocklist", self.0)
    }
}

impl Error for Blocked {}

/// How `failure` names a request the abuse blocklist kept from going out.
pub const BLOCKED: &str = "Blocked";

/// Public answers, or why none may be asked: none public, or one in the abuse blocklist, which then refuses the name outright.
fn usable(name: &str, answers: impl Iterator<Item = IpAddr>) -> Result<Addrs, Box<dyn Error + Send + Sync>> {
    let mut public = Vec::new();
    for ip in answers {
        if crate::blocklist::blocked_ip(ip) {
            return Err(Box::new(Blocked(name.to_string())));
        }
        if is_global(ip) {
            public.push(SocketAddr::new(ip, 0));
        }
    }
    if public.is_empty() {
        return Err(Box::new(NotPublic(name.to_string())));
    }
    Ok(Box::new(public.into_iter()))
}

/// The local caching resolver, with answers that point into a private network dropped.
#[derive(Clone)]
pub struct PublicResolver(TokioAsyncResolver);

impl PublicResolver {
    pub fn local() -> Self {
        let servers = NameServerConfigGroup::from_ips_clear(&[RESOLVER], RESOLVER_PORT, true);
        let mut options = ResolverOpts::default();
        options.ip_strategy = LookupIpStrategy::Ipv4thenIpv6;
        options.cache_size = 65_536;
        options.timeout = Duration::from_secs(5);
        PublicResolver(TokioAsyncResolver::tokio(ResolverConfig::from_parts(None, vec![], servers), options))
    }
}

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let resolver = self.0.clone();
        Box::pin(async move {
            let answers = resolver.lookup_ip(name.as_str()).await?;
            usable(name.as_str(), answers.iter())
        })
    }
}

#[derive(Debug)]
struct NotResolved(String);

impl fmt::Display for NotResolved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} did not resolve", self.0)
    }
}

impl Error for NotResolved {}

/// The system's resolver, which aiohttp asks by default, with answers that point into a private network dropped.
#[derive(Clone, Copy, Default)]
pub struct SystemResolver;

impl Resolve for SystemResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let host = name.as_str().to_string();
            let found = tokio::net::lookup_host((host.as_str(), 0)).await.map_err(|e| Box::new(NotResolved(format!("{host} ({e})"))) as Box<dyn Error + Send + Sync>)?;
            usable(&host, found.map(|a| a.ip()))
        })
    }
}

/// One client for the whole process; the local resolver caches DNS, and redirects are followed by hand.
pub fn client(resolver: PublicResolver, read_timeout: Duration) -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .timeout(MAX_REQUEST)
        .connect_timeout(connect_timeout(read_timeout))
        .read_timeout(read_timeout)
        .pool_idle_timeout(Duration::from_secs(15))
        .pool_max_idle_per_host(4)
        .http1_only()
        .tcp_nodelay(true)
        .dns_resolver(Arc::new(resolver))
        .build()
}

/// Connecting covers the DNS lookup, TCP and the TLS handshake, so it gets a little longer than a read.
pub fn connect_timeout(read_timeout: Duration) -> Duration {
    read_timeout + Duration::from_secs(2)
}

/// The body up to limit bytes.
pub async fn read_body(response: &mut reqwest::Response, limit: usize) -> reqwest::Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        body.extend_from_slice(&chunk);
        if body.len() >= limit {
            break;
        }
    }
    body.truncate(limit);
    Ok(body)
}

/// Whether the connection itself failed: refused, or timed out while connecting; a slow answer, a DNS or a TLS failure never is.
pub fn connect_failed(error: &reqwest::Error) -> bool {
    if !error.is_connect() {
        return false;
    }
    let mut cause = error.source();
    while let Some(current) = cause {
        if current.downcast_ref::<std::io::Error>().is_some_and(|io| matches!(io.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::TimedOut)) {
            return true;
        }
        if current.is::<ResolveError>() || current.is::<NotPublic>() || current.is::<NotResolved>() || current.is::<Blocked>() || current.is::<rustls::Error>() {
            return false;
        }
        cause = current.source();
    }
    error.is_timeout()
}

/// The name aiohttp gives the same failure; `connecting` is the connect timeout while no answer has arrived yet.
pub fn failure(error: &reqwest::Error, elapsed: Duration, connecting: Option<Duration>) -> &'static str {
    if error.is_builder() {
        return "InvalidUrlClientError";
    }
    if error.is_timeout() {
        return if error.is_connect() || connecting.is_some_and(|limit| elapsed >= limit) {
            "ConnectionTimeoutError"
        } else if elapsed >= MAX_REQUEST {
            "TimeoutError"
        } else {
            "SocketTimeoutError"
        };
    }
    let mut cause = error.source();
    while let Some(current) = cause {
        if current.is::<Blocked>() {
            return BLOCKED;
        }
        if current.is::<ResolveError>() || current.is::<NotPublic>() || current.is::<NotResolved>() {
            return "ClientConnectorDNSError";
        }
        let mut inner: &(dyn Error + 'static) = current;
        while let Some(wrapped) = inner.downcast_ref::<std::io::Error>().and_then(|io| io.get_ref()) {
            inner = wrapped;
        }
        if let Some(tls) = inner.downcast_ref::<rustls::Error>() {
            return match tls {
                rustls::Error::InvalidCertificate(_) => "ClientConnectorCertificateError",
                _ => "ClientConnectorSSLError",
            };
        }
        if current.to_string().contains("connection closed before message completed") {
            return "ServerDisconnectedError";
        }
        cause = current.source();
    }
    if error.is_connect() {
        "ClientConnectorError"
    } else if error.is_body() || error.is_decode() {
        "ClientPayloadError"
    } else {
        "ClientOSError"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_a_refused_connection_or_a_connect_timeout_is_a_connect_failure() {
        let client = reqwest::Client::builder().timeout(Duration::from_millis(300)).connect_timeout(Duration::from_millis(200)).build().unwrap();
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let error = client.get(format!("http://{closed}/")).send().await.unwrap_err();
        assert!(connect_failed(&error), "refused: {error:?}");
        let silent = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = silent.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = silent.accept().await {
                held.push(socket);
            }
        });
        let started = std::time::Instant::now();
        let error = client.get(format!("http://{addr}/")).send().await.unwrap_err();
        assert_eq!(failure(&error, started.elapsed(), Some(Duration::from_millis(200))), "ConnectionTimeoutError", "named as aiohttp names it");
        assert!(!connect_failed(&error), "connected, then no answer: {error:?}");
    }

    #[tokio::test]
    async fn the_system_resolver_keeps_private_answers_out() {
        let client = reqwest::Client::builder().dns_resolver(Arc::new(SystemResolver)).build().unwrap();
        let started = std::time::Instant::now();
        let error = client.get("http://localhost:9/").send().await.unwrap_err();
        assert_eq!(failure(&error, started.elapsed(), None), "ClientConnectorDNSError");
        let error = client.get("http://no-such-host.invalid/").send().await.unwrap_err();
        assert_eq!(failure(&error, started.elapsed(), None), "ClientConnectorDNSError");
    }

    #[test]
    fn an_answer_in_the_blocklist_refuses_the_name() {
        let ips = |list: &[&str]| list.iter().map(|ip| ip.parse::<IpAddr>().unwrap()).collect::<Vec<_>>().into_iter();
        assert!(usable("shfg.com", ips(&["8.8.8.8", "106.249.55.48"])).err().unwrap().is::<Blocked>());
        assert!(usable("a.test", ips(&["10.0.0.1"])).err().unwrap().is::<NotPublic>());
        assert_eq!(usable("a.test", ips(&["10.0.0.1", "8.8.8.8"])).unwrap().count(), 1);
    }

    #[test]
    fn global_matches_python() {
        let global = |s: &str| is_global(s.parse().unwrap());
        for public in ["8.8.8.8", "192.0.0.9", "2606:4700::1111", "2001:4:112::1", "224.0.0.1"] {
            assert!(global(public), "{public}");
        }
        for private in ["10.0.0.1", "100.64.1.1", "127.0.0.1", "192.0.0.171", "::1", "2001:db8::1", "::ffff:10.0.0.1", "fe80::1", "fd00::1"] {
            assert!(!global(private), "{private}");
        }
        assert!(!public_host("[::1]") && !public_host("169.254.169.254") && public_host("example.com"));
    }
}

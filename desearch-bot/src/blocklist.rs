//! Operators, domains and networks that asked us to stop: nothing on the bot box sends them a request.

use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::net::{within4, within6};

#[derive(Deserialize)]
struct Raw {
    operators: Vec<String>,
    not_blocked: HashSet<String>,
    domains: Vec<String>,
    networks: Vec<String>,
}

pub struct Blocklist {
    operators: Vec<String>,
    not_blocked: HashSet<String>,
    domains: Vec<String>,
    networks: Vec<(IpAddr, u32)>,
}

pub static BLOCKLIST: LazyLock<Blocklist> =
    LazyLock::new(|| Blocklist::parse(include_str!("../data/blocklist.json")).expect("data/blocklist.json is valid"));

fn network(text: &str) -> Option<(IpAddr, u32)> {
    let (address, prefix) = text.split_once('/').unwrap_or((text, ""));
    let address: IpAddr = address.parse().ok()?;
    let most = if address.is_ipv4() { 32 } else { 128 };
    let prefix = if prefix.is_empty() { most } else { prefix.parse().ok()? };
    (prefix <= most).then_some((address, prefix))
}

fn under(host: &str, domain: &str) -> bool {
    host == domain || host.strip_suffix(domain).is_some_and(|rest| rest.ends_with('.'))
}

impl Blocklist {
    pub fn parse(json: &str) -> anyhow::Result<Self> {
        let raw: Raw = serde_json::from_str(json)?;
        let networks = raw.networks.iter().map(|n| network(n).ok_or_else(|| anyhow::anyhow!("bad network {n}"))).collect::<anyhow::Result<_>>()?;
        Ok(Blocklist { operators: raw.operators, not_blocked: raw.not_blocked, domains: raw.domains, networks })
    }

    pub fn ip(&self, ip: IpAddr) -> bool {
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
            v4 => v4,
        };
        self.networks.iter().any(|&(net, prefix)| match (ip, net) {
            (IpAddr::V4(ip), IpAddr::V4(net)) => within4(ip, (net.octets(), prefix)),
            (IpAddr::V6(ip), IpAddr::V6(net)) => within6(ip, (u128::from(net), prefix)),
            _ => false,
        })
    }

    /// A blocked domain or anything under it, an operator's name in any label but the TLD, or an address literal in a blocked network.
    pub fn host(&self, host: &str) -> bool {
        let host = host.trim_matches(['[', ']']).trim_end_matches('.').to_ascii_lowercase();
        if let Ok(ip) = host.split_once('%').map_or(host.as_str(), |(address, _)| address).parse::<IpAddr>() {
            return self.ip(ip);
        }
        if self.domains.iter().any(|d| under(&host, d)) {
            return true;
        }
        if self.not_blocked.iter().any(|d| under(&host, d)) {
            return false;
        }
        let labels: Vec<&str> = host.split('.').collect();
        let named = &labels[..labels.len().saturating_sub(1)];
        named.iter().any(|label| self.operators.iter().any(|name| label.contains(name.as_str())))
    }

    pub fn url(&self, url: &str) -> bool {
        crate::urls::split(url).ok().and_then(|parts| parts.hostname()).is_some_and(|host| self.host(&host))
    }
}

pub fn blocked_host(host: &str) -> bool {
    BLOCKLIST.host(host)
}

pub fn blocked_ip(ip: IpAddr) -> bool {
    BLOCKLIST.ip(ip)
}

pub fn blocked_url(url: &str) -> bool {
    BLOCKLIST.url(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_the_reported_operator_by_name_domain_and_network() {
        for host in ["shinhan.com", "www.shinhangroup.com", "shinhangroup.co.kr", "shfg.com", "www.shfg.com", "SHFG.COM.", "jejubank.co.kr", "106.249.55.48", "[::ffff:106.249.55.48]"] {
            assert!(blocked_host(host), "{host}");
        }
        for host in ["kakushinhan.org", "www.kakushinhan.org", "marushinhanten.com", "notshfg.com", "example.com", "106.249.56.48"] {
            assert!(!blocked_host(host), "{host}");
        }
        assert!(blocked_ip("106.249.55.115".parse().unwrap()));
        assert!(blocked_ip("::ffff:106.249.55.144".parse().unwrap()));
        assert!(!blocked_ip("106.249.54.255".parse().unwrap()));
        assert!(blocked_url("http://www.shinhangroup.com/robots.txt"));
        assert!(blocked_url("https://106.249.55.48:8443/"));
        assert!(!blocked_url("https://example.com/shinhan-bank-news"));
    }

    #[test]
    fn reads_networks_of_both_families() {
        let list = Blocklist::parse(r#"{"operators": [], "not_blocked": [], "domains": ["a.test"], "networks": ["2001:db8::/32", "192.0.2.7"]}"#).unwrap();
        assert!(list.ip("2001:db8::1".parse().unwrap()) && !list.ip("2001:db9::1".parse().unwrap()));
        assert!(list.ip("192.0.2.7".parse().unwrap()) && !list.ip("192.0.2.8".parse().unwrap()));
        assert!(list.host("x.a.test") && !list.host("xa.test"));
        assert!(Blocklist::parse(r#"{"operators": [], "not_blocked": [], "domains": [], "networks": ["10.0.0.0/33"]}"#).is_err());
    }
}

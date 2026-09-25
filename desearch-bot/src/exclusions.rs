//! Which hosts never reach a miner, as far as the crawl loop has to know.

use std::sync::LazyLock;

use regex::Regex;

use crate::suffixes::RULES;

static INFRASTRUCTURE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!("(?i){}", RULES.infrastructure_name)).expect("infrastructure pattern"));

/// Operators who asked us to stop; matching the label covers their other sites too.
pub fn blocked_operator(host: &str) -> bool {
    if RULES.not_blocked.contains(host) {
        return false;
    }
    let label = host.split('.').next().unwrap_or("");
    RULES.blocked_operators.iter().any(|name| label.contains(name.as_str()))
}

pub fn valid_host(host: &str) -> bool {
    let labels: Vec<&str> = host.split('.').collect();
    host.len() <= 253
        && labels.len() >= 2
        && labels.iter().all(|label| {
            (1..=63).contains(&label.len())
                && label.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                && !label.starts_with('-')
                && !label.ends_with('-')
        })
        && !labels[labels.len() - 1].bytes().all(|b| b.is_ascii_digit())
}

/// Why a domain found through a redirect must not be crawled, judged without category lists.
pub fn exclusion_reason(host: &str, tld_group: &str) -> Option<&'static str> {
    if blocked_operator(host) {
        Some("blocked_operator")
    } else if !valid_host(host) {
        Some("invalid_host")
    } else if RULES.dynamic_platforms.contains(host) {
        Some("dynamic_platform")
    } else if RULES.infrastructure.contains(host) {
        Some("infrastructure")
    } else if INFRASTRUCTURE_NAME.is_match(host) {
        Some("infrastructure_name")
    } else if tld_group == "other_cctld" {
        Some("non_english_tld")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judges_hosts_like_python() {
        assert!(blocked_operator("shinhan.com"));
        assert!(!blocked_operator("kakushinhan.org"));
        assert!(valid_host("example.com"));
        assert!(!valid_host("-a.com") && !valid_host("a.123") && !valid_host("Example.com") && !valid_host("com"));
        assert_eq!(exclusion_reason("cdn.example.com", "big_generic"), Some("infrastructure_name"));
        assert_eq!(exclusion_reason("example.de", "other_cctld"), Some("non_english_tld"));
        assert_eq!(exclusion_reason("desearch-news.com", "big_generic"), None);
    }
}

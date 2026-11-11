//! A source's own section of index-1's hidden-URLs file: exact-path rules for the pages it dropped.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;

pub const HIDDEN: &str = "/data0/index/hidden-urls.txt";

pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit())
        && name.len() <= 41
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// The router's `host/path-glob` line matching exactly this URL's path, or None when a rule can't say it.
pub fn rule(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = url.split_once("://").map(|(_, rest)| rest)?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    let tail = &rest[end..];
    let host_port = authority.rsplit_once('@').map(|(_, h)| h).unwrap_or(authority);
    let host = match host_port.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or_default().to_string(),
        None => host_port.split(':').next().unwrap_or_default().to_string(),
    }
    .to_lowercase();
    let path_end = tail.find(['?', '#']).unwrap_or(tail.len());
    let path = &tail[..path_end];
    if host.is_empty() || tail[path_end..].starts_with('?') || path.chars().any(char::is_whitespace) {
        return None;
    }
    let path = if path.is_empty() { "/" } else { path };
    let mut out = host;
    for c in path.chars() {
        if matches!(c, '*' | '?' | '[') {
            out.push('[');
            out.push(c);
            out.push(']');
        } else {
            out.push(c);
        }
    }
    Some(out)
}

pub fn rewrite(text: &str, source: &str, urls: &[String]) -> String {
    let (begin, end) = (format!("# refresh {source} begin"), format!("# refresh {source} end"));
    let mut kept: Vec<&str> = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if line == begin {
            inside = true;
        } else if line == end {
            inside = false;
        } else if !inside {
            kept.push(line);
        }
    }
    let rules: BTreeSet<String> = urls.iter().filter_map(|u| rule(u)).collect();
    let mut out: Vec<String> = kept.into_iter().map(str::to_string).collect();
    if !rules.is_empty() {
        out.push(begin);
        out.extend(rules);
        out.push(end);
    }
    out.join("\n") + "\n"
}

/// The forced command on index-1: replaces the source's section of the file with the URLs on stdin.
pub fn hide_section(source: &str, path: &Path, input: &str) -> Result<String> {
    let before = std::fs::read_to_string(path).unwrap_or_default();
    let urls: Vec<String> = input.split_whitespace().map(str::to_string).collect();
    let after = rewrite(&before, source, &urls);
    if after != before {
        let tmp = path.with_file_name(format!("{}.refresh-{source}.tmp", path.file_name().unwrap_or_default().to_string_lossy()));
        std::fs::write(&tmp, &after)?;
        std::fs::rename(&tmp, path)?;
    }
    Ok(format!("{source}: {:+} lines", after.lines().count() as i64 - before.lines().count() as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_match_the_exact_path_and_escape_globs() {
        assert_eq!(rule("https://taostats.io/subnets/7").as_deref(), Some("taostats.io/subnets/7"));
        assert_eq!(rule("https://A.io/x[1]*").as_deref(), Some("a.io/x[[]1][*]"));
        assert_eq!(rule("https://a.io/p#frag").as_deref(), Some("a.io/p"));
        assert_eq!(rule("https://user@a.io:8080").as_deref(), Some("a.io/"));
        assert_eq!(rule("https://a.io/q?id=1"), None);
        assert_eq!(rule("not a url"), None);
    }

    #[test]
    fn the_section_is_rewritten_whole() {
        let text = "# old rules\nexample.com/junk/*\n";
        let urls = vec!["https://taostats.io/subnets/7".to_string(), "https://a.io/x".to_string()];
        let once = rewrite(text, "bittensor", &urls);
        assert_eq!(once, "# old rules\nexample.com/junk/*\n# refresh bittensor begin\na.io/x\ntaostats.io/subnets/7\n# refresh bittensor end\n");
        assert_eq!(rewrite(&once, "bittensor", &[]), text);
    }

    #[test]
    fn source_names_are_checked() {
        assert!(valid_name("bittensor") && valid_name("subnet-sites"));
        assert!(!valid_name("ls /") && !valid_name("") && !valid_name("../x") && !valid_name("-a"));
    }
}

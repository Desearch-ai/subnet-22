//! When a sent page goes out again: its domain's re-crawl interval, and the long wait of a page that keeps failing.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub const DAY: u32 = 86_400;
pub const DEFAULT_EVERY: u32 = 7 * DAY;
pub const DEFAULT_GONE: u32 = 30 * DAY;

#[derive(Clone, Debug, PartialEq)]
pub struct Recrawl {
    /// Seconds after a send that a page without a lastmod, or one that failed, goes out again.
    pub every: u32,
    /// Seconds a page that failed on two sends in a row waits before it goes out again.
    pub gone: u32,
    /// Intervals for listed domains and their subdomains.
    overrides: HashMap<String, u32>,
}

impl Default for Recrawl {
    fn default() -> Self {
        Recrawl::new(DEFAULT_EVERY, DEFAULT_GONE)
    }
}

impl Recrawl {
    pub fn new(every: u32, gone: u32) -> Self {
        Recrawl { every, gone, overrides: HashMap::new() }
    }

    /// Adds `domain days` lines, such as `bbc.co.uk 1` or `wikipedia.org 0.5`; `#` starts a comment.
    pub fn with_overrides(mut self, text: &str) -> Result<Self> {
        for (number, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let (Some(domain), Some(days), None) = (fields.next(), fields.next(), fields.next()) else {
                bail!("line {}: expected a domain and a number of days", number + 1);
            };
            let days: f64 = days.parse().ok().filter(|d: &f64| *d > 0.0).with_context(|| format!("line {}: {days} is not a number of days", number + 1))?;
            let domain = domain.trim_start_matches("www.").to_ascii_lowercase();
            self.overrides.insert(domain, (days * f64::from(DAY)).round().clamp(1.0, f64::from(u32::MAX)) as u32);
        }
        Ok(self)
    }

    pub fn with_overrides_file(self, path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        self.with_overrides(&text).with_context(|| path.display().to_string())
    }

    pub fn overridden(&self) -> usize {
        self.overrides.len()
    }

    /// The interval of `domain` from its own line or its nearest listed parent's; None when neither is listed.
    pub fn own_interval(&self, domain: &str) -> Option<u32> {
        let mut name = domain;
        loop {
            if let Some(&every) = self.overrides.get(name) {
                return Some(every);
            }
            match name.split_once('.') {
                Some((_, parent)) if parent.contains('.') => name = parent,
                _ => return None,
            }
        }
    }

    /// The interval of `domain`: its own line, else its nearest listed parent's, else the default.
    pub fn interval(&self, domain: &str) -> u32 {
        let mut name = domain;
        loop {
            if let Some(&every) = self.overrides.get(name) {
                return every;
            }
            match name.split_once('.') {
                Some((_, parent)) if parent.contains('.') => name = parent,
                _ => return self.every,
            }
        }
    }
}

//! The domains the bot may queue for the task API, from a JSON list read again when it changes; with no list, every domain.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::isodate;

/// What the list says about one domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Terms {
    /// Turns a domain gets for each its rank earns, 1 to 4.
    pub weight: u8,
    /// Pages of the domain the bot may ever send from its new-page lane; None for no cap.
    pub send_cap: Option<u64>,
    /// Pages the domain may send in an hour; None for `--per-domain-hourly`.
    pub hourly: Option<u64>,
    /// For a domain without a cap, the most stored pages queued when it joins the list, newest first.
    pub backlog: Option<u64>,
    /// The oldest page date the domain queues, as Unix seconds: a page dated earlier stays unsent, an undated one goes.
    pub since: Option<u32>,
    /// A news site, whose sitemaps are read again before other sites'.
    pub news: bool,
    /// Pages last sent before this time, as Unix seconds, go once more, as retries; a site is walked for them once per value.
    pub resend_before: Option<u32>,
    /// A landing site's: the most path levels its pages may have, a leading English locale not counted; deeper pages, pages with a query and pages of other hosts stay unsent.
    pub depth: Option<u8>,
}

impl Default for Terms {
    fn default() -> Self {
        Terms { weight: 1, send_cap: None, hourly: None, backlog: None, since: None, news: false, resend_before: None, depth: None }
    }
}

pub struct Allowed {
    path: Option<PathBuf>,
    domains: RwLock<Option<HashMap<String, Terms>>>,
    read: RwLock<Option<SystemTime>>,
    /// Bumped on each change, so a reader knows to look again.
    version: AtomicU64,
    /// Listed subdomains of listed domains whose own record speaks for them; only they take their pages from the parent.
    apart: RwLock<HashSet<String>>,
}

impl Allowed {
    pub fn all() -> Arc<Self> {
        Arc::new(Allowed { path: None, domains: RwLock::new(None), read: RwLock::new(None), version: AtomicU64::new(0), apart: RwLock::default() })
    }

    pub fn of<S: Into<String>>(domains: impl IntoIterator<Item = S>) -> Arc<Self> {
        let domains = domains.into_iter().map(|d| (d.into(), Terms::default())).collect();
        Arc::new(Allowed { path: None, domains: RwLock::new(Some(domains)), read: RwLock::new(None), version: AtomicU64::new(1), apart: RwLock::default() })
    }

    pub fn with_terms<S: Into<String>>(domains: impl IntoIterator<Item = (S, Terms)>) -> Arc<Self> {
        let domains = domains.into_iter().map(|(d, l)| (d.into(), l)).collect();
        Arc::new(Allowed { path: None, domains: RwLock::new(Some(domains)), read: RwLock::new(None), version: AtomicU64::new(1), apart: RwLock::default() })
    }

    pub fn from_file(path: &Path) -> Result<Arc<Self>> {
        let allowed = Allowed { path: Some(path.to_path_buf()), domains: RwLock::new(None), read: RwLock::new(None), version: AtomicU64::new(0), apart: RwLock::default() };
        allowed.reload()?;
        Ok(Arc::new(allowed))
    }

    pub fn allows(&self, domain: &str) -> bool {
        self.domains.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_none_or(|d| d.contains_key(domain))
    }

    /// The domain's weight and cap; None when it is not listed or there is no list.
    pub fn get(&self, domain: &str) -> Option<Terms> {
        self.domains.read().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|d| d.get(domain).copied())
    }

    pub fn count(&self) -> Option<usize> {
        self.domains.read().unwrap_or_else(|e| e.into_inner()).as_ref().map(HashMap::len)
    }

    /// Every listed domain with what the list says about it; empty with no list.
    pub fn entries(&self) -> Vec<(String, Terms)> {
        let domains = self.domains.read().unwrap_or_else(|e| e.into_inner());
        domains.as_ref().map_or_else(Vec::new, |d| d.iter().map(|(name, listing)| (name.clone(), *listing)).collect())
    }

    /// Whether the host is a listed subdomain whose own record speaks for it, so its pages go under it or not at all, never under its parent.
    pub fn apart(&self, host: &str) -> bool {
        self.get(host).is_some() && self.apart.read().unwrap_or_else(|e| e.into_inner()).contains(host)
    }

    pub fn set_apart(&self, hosts: HashSet<String>) {
        *self.apart.write().unwrap_or_else(|e| e.into_inner()) = hosts;
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Reads the list again if the file changed since; returns whether it did.
    pub fn reload(&self) -> Result<bool> {
        let Some(path) = &self.path else {
            return Ok(false);
        };
        let modified = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?.modified()?;
        if *self.read.read().unwrap_or_else(|e| e.into_inner()) == Some(modified) {
            return Ok(false);
        }
        let domains = parse(&std::fs::read(path)?).with_context(|| format!("reading {}", path.display()))?;
        *self.domains.write().unwrap_or_else(|e| e.into_inner()) = Some(domains);
        *self.read.write().unwrap_or_else(|e| e.into_inner()) = Some(modified);
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
}

/// A JSON list of domain names, or of objects naming theirs under `host`, with an optional `type`, `weight`, `send_cap`, `hourly`, `backlog`, `since` and `resend_before` (ISO dates), and `depth`.
pub fn parse(raw: &[u8]) -> Result<HashMap<String, Terms>> {
    let listed: Vec<Value> = serde_json::from_slice(raw).context("not a JSON list of domains")?;
    Ok(listed
        .iter()
        .filter_map(|entry| {
            let host = entry.as_str().or_else(|| entry.get("host").and_then(Value::as_str))?;
            let number = |name: &str| entry.get(name).and_then(Value::as_u64);
            let weight = number("weight").map_or(1, |w| w.clamp(1, 4) as u8);
            let date = |name: &str| entry.get(name).and_then(Value::as_str).and_then(isodate::fromisoformat).map(|micros| (micros / 1_000_000).clamp(0, i64::from(u32::MAX)) as u32);
            let news = entry.get("type").and_then(Value::as_str) == Some("news");
            let depth = number("depth").map(|depth| depth.min(u64::from(u8::MAX)) as u8);
            Some((host.to_string(), Terms { weight, send_cap: number("send_cap"), hourly: number("hourly"), backlog: number("backlog"), since: date("since"), news, resend_before: date("resend_before"), depth }))
        })
        .collect())
}

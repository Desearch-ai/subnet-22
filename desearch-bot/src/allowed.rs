//! The domains the bot may queue for the task API, from a JSON list read again when it changes; with no list, every domain.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde_json::Value;

/// A listed domain's kind of site, which sets how often its pages are crawled again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SiteType {
    News,
    Reference,
    Docs,
    Academic,
    Public,
    Media,
    Community,
    Company,
    Organization,
    Personal,
    #[default]
    Other,
}

impl SiteType {
    pub fn parse(name: &str) -> SiteType {
        match name {
            "news" => SiteType::News,
            "reference" => SiteType::Reference,
            "docs" => SiteType::Docs,
            "academic" => SiteType::Academic,
            "public" => SiteType::Public,
            "media" => SiteType::Media,
            "community" => SiteType::Community,
            "company" => SiteType::Company,
            "organization" => SiteType::Organization,
            "personal" => SiteType::Personal,
            _ => SiteType::Other,
        }
    }
}

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
    pub site: SiteType,
}

impl Default for Terms {
    fn default() -> Self {
        Terms { weight: 1, send_cap: None, hourly: None, backlog: None, site: SiteType::Other }
    }
}

pub struct Allowed {
    path: Option<PathBuf>,
    domains: RwLock<Option<HashMap<String, Terms>>>,
    read: RwLock<Option<SystemTime>>,
    /// Bumped on each change, so a reader knows to look again.
    version: AtomicU64,
}

impl Allowed {
    pub fn all() -> Arc<Self> {
        Arc::new(Allowed { path: None, domains: RwLock::new(None), read: RwLock::new(None), version: AtomicU64::new(0) })
    }

    pub fn of<S: Into<String>>(domains: impl IntoIterator<Item = S>) -> Arc<Self> {
        let domains = domains.into_iter().map(|d| (d.into(), Terms::default())).collect();
        Arc::new(Allowed { path: None, domains: RwLock::new(Some(domains)), read: RwLock::new(None), version: AtomicU64::new(1) })
    }

    pub fn with_terms<S: Into<String>>(domains: impl IntoIterator<Item = (S, Terms)>) -> Arc<Self> {
        let domains = domains.into_iter().map(|(d, l)| (d.into(), l)).collect();
        Arc::new(Allowed { path: None, domains: RwLock::new(Some(domains)), read: RwLock::new(None), version: AtomicU64::new(1) })
    }

    pub fn from_file(path: &Path) -> Result<Arc<Self>> {
        let allowed = Allowed { path: Some(path.to_path_buf()), domains: RwLock::new(None), read: RwLock::new(None), version: AtomicU64::new(0) };
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

/// A JSON list of domain names, or of objects naming theirs under `host`, with an optional `type`, `weight`, `send_cap`, `hourly` and `backlog`.
pub fn parse(raw: &[u8]) -> Result<HashMap<String, Terms>> {
    let listed: Vec<Value> = serde_json::from_slice(raw).context("not a JSON list of domains")?;
    Ok(listed
        .iter()
        .filter_map(|entry| {
            let host = entry.as_str().or_else(|| entry.get("host").and_then(Value::as_str))?;
            let number = |name: &str| entry.get(name).and_then(Value::as_u64);
            let weight = number("weight").map_or(1, |w| w.clamp(1, 4) as u8);
            let site = entry.get("type").and_then(Value::as_str).map_or(SiteType::Other, SiteType::parse);
            Some((host.to_string(), Terms { weight, send_cap: number("send_cap"), hourly: number("hourly"), backlog: number("backlog"), site }))
        })
        .collect())
}

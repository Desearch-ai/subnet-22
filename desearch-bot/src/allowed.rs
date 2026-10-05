//! The domains the bot may queue for the task API, from a JSON list read again when it changes; with no list, every domain.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde_json::Value;

pub struct Allowed {
    path: Option<PathBuf>,
    domains: RwLock<Option<HashSet<String>>>,
    read: RwLock<Option<SystemTime>>,
}

impl Allowed {
    pub fn all() -> Arc<Self> {
        Arc::new(Allowed { path: None, domains: RwLock::new(None), read: RwLock::new(None) })
    }

    pub fn of<S: Into<String>>(domains: impl IntoIterator<Item = S>) -> Arc<Self> {
        let domains = domains.into_iter().map(Into::into).collect();
        Arc::new(Allowed { path: None, domains: RwLock::new(Some(domains)), read: RwLock::new(None) })
    }

    pub fn from_file(path: &Path) -> Result<Arc<Self>> {
        let allowed = Allowed { path: Some(path.to_path_buf()), domains: RwLock::new(None), read: RwLock::new(None) };
        allowed.reload()?;
        Ok(Arc::new(allowed))
    }

    pub fn allows(&self, domain: &str) -> bool {
        self.domains.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_none_or(|d| d.contains(domain))
    }

    pub fn count(&self) -> Option<usize> {
        self.domains.read().unwrap_or_else(|e| e.into_inner()).as_ref().map(HashSet::len)
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
        Ok(true)
    }
}

/// A JSON list of domain names, or of objects naming theirs under `host`.
pub fn parse(raw: &[u8]) -> Result<HashSet<String>> {
    let listed: Vec<Value> = serde_json::from_slice(raw).context("not a JSON list of domains")?;
    Ok(listed
        .iter()
        .filter_map(|entry| entry.as_str().or_else(|| entry.get("host").and_then(Value::as_str)))
        .map(str::to_string)
        .collect())
}

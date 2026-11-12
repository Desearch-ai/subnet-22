//! The Python crawler's fetches in Rust: site crawls by links, URL lists and EDGAR filings, under robots.txt and each host's pace, into its record files.

pub mod cookies;
pub mod edgar;
pub mod fetch;
pub mod hosts;
pub mod links;
pub mod lists;
pub mod robots;
pub mod run;
pub mod scope;
pub mod sites;
pub mod store;

use indexmap::IndexMap;

/// A Python exception that ends one host's crawl, by its type name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crash(pub &'static str);

/// A `collections.Counter`: keys in the order first counted, a key added by `+= 0` kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counts(pub IndexMap<String, i64>);

impl Counts {
    pub fn add(&mut self, key: &str, n: i64) {
        *self.0.entry(key.to_string()).or_insert(0) += n;
    }

    pub fn get(&self, key: &str) -> i64 {
        self.0.get(key).copied().unwrap_or(0)
    }

    pub fn update(&mut self, other: &Counts) {
        for (k, v) in &other.0 {
            self.add(k, *v);
        }
    }
}

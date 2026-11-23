//! Pages taken down by URL: change files of `removed` rows for a list of URLs, each saved to a backup first.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use desearch::canonical::domain_of;

use crate::changes::{self, Body, Change, Kind};
use crate::records::page_key;
use crate::worker::ChangeFeed;

/// Removal rows in one change file at most.
pub const PER_FILE: usize = 10_000;

/// One removal row for each page the lines name, in order and once a page, and the lines that name no page.
pub fn removals(lines: &[String], published_at: &str) -> (Vec<Change>, Vec<String>) {
    let mut seen = HashSet::new();
    let (mut rows, mut refused) = (Vec::new(), Vec::new());
    for url in lines.iter().map(|line| line.trim()).filter(|url| !url.is_empty()) {
        let web = url.starts_with("https://") || url.starts_with("http://");
        match (page_key(url), domain_of(url)) {
            (Ok(key), Ok(domain)) if web && !domain.is_empty() => {
                if seen.insert(key.clone()) {
                    let body = Body::Removed { url: url.to_string(), domain };
                    rows.push(Change { key, kind: Kind::Removed, previous_content_sha1: String::new(), published_at: published_at.to_string(), body });
                }
            }
            _ => refused.push(url.to_string()),
        }
    }
    (rows, refused)
}

/// Saves every row to `backup` as a `key<TAB>url` line, then appends the rows to `feed` in change files of at most `PER_FILE`; the numbers of those files.
pub fn take_down(rows: &[Change], feed: &dyn ChangeFeed, backup: &Path) -> Result<Vec<u64>> {
    let file = OpenOptions::new().write(true).create_new(true).open(backup).with_context(|| format!("creating {}", backup.display()))?;
    let mut saved = BufWriter::new(file);
    for row in rows {
        writeln!(saved, "{}\t{}", row.key, row.url().unwrap_or_default())?;
    }
    saved.into_inner()?.sync_all()?;
    rows.chunks(PER_FILE).map(|chunk| feed.append(changes::encode(chunk)?, chunk.len())).collect()
}

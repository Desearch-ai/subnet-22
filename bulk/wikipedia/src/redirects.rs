//! `wikipedia-redirects`: the redirect URLs that lead to a run's pages, from the snapshots the run read.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{dict, field, read, text, truthy};

/// The file a run's redirects go to by default, `redirect url<TAB>page url` a line, sorted by redirect URL.
pub const FILE: &str = "redirects.tsv";
const BATCH: usize = 256;
const ARTICLE_PATH: &str = "/wiki/";
/// Bytes besides letters and digits that MediaWiki's `wfUrlencode` leaves as they are.
const PLAIN: &[u8] = b"-_.;@$!*(),/~:";
const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// The URL MediaWiki gives title `name` on the page's site; the snapshot's own redirect URL is left unescaped.
fn redirect_url(page_url: &str, name: Option<&str>, url: Option<&str>) -> Option<String> {
    let (Some(name), Some(at)) = (name.filter(|n| !n.is_empty()), page_url.find(ARTICLE_PATH)) else { return url.map(str::to_string) };
    let mut out = String::with_capacity(at + ARTICLE_PATH.len() + name.len() * 3);
    out.push_str(&page_url[..at + ARTICLE_PATH.len()]);
    for &b in name.replace(' ', "_").as_bytes() {
        if b.is_ascii_alphanumeric() || PLAIN.contains(&b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 15) as usize] as char);
        }
    }
    Some(out)
}

#[derive(clap::Args, Clone, Debug)]
pub struct RedirectsArgs {
    /// The run folder whose pages the redirects lead to.
    #[arg(long)]
    pub run: PathBuf,
    /// Where the lines go; `redirects.tsv` in the run folder by default.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// The snapshots the run read.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct Numbered {
    identifier: Option<Value>,
}

#[derive(Deserialize)]
struct Event<'a> {
    #[serde(borrow, rename = "type")]
    kind: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Redirect<'a> {
    #[serde(borrow)]
    name: Option<Cow<'a, str>>,
    #[serde(borrow)]
    url: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Article<'a> {
    identifier: Option<Value>,
    #[serde(borrow)]
    url: Option<Cow<'a, str>>,
    namespace: Option<Numbered>,
    version: Option<Numbered>,
    #[serde(borrow)]
    event: Option<Event<'a>>,
    #[serde(borrow)]
    redirects: Option<Vec<Redirect<'a>>>,
}

/// One main-namespace article that is not a deletion: its id, revision, URL and redirect URLs.
#[derive(Debug, PartialEq, Eq)]
struct Entry {
    id: i64,
    revision: i64,
    url: String,
    redirects: Vec<String>,
}

fn main_namespace(identifier: Option<&Value>) -> bool {
    match identifier {
        Some(Value::Number(n)) => n.as_f64() == Some(0.0),
        Some(Value::Bool(b)) => !b,
        _ => false,
    }
}

fn revision_of(identifier: Option<&Value>) -> Result<i64> {
    match identifier {
        v if !truthy(v) => Ok(0),
        Some(v) => v.as_i64().ok_or_else(|| anyhow!("revision {v} is not an integer")),
        None => Ok(0),
    }
}

fn id_of(identifier: Option<&Value>) -> Result<i64> {
    let id = identifier.ok_or_else(|| anyhow!("article has no identifier"))?;
    id.as_i64().ok_or_else(|| anyhow!("identifier {id} is not an integer"))
}

fn entry(line: &[u8]) -> Result<Option<Entry>> {
    let Ok(a) = serde_json::from_slice::<Article>(line.strip_prefix(b"\xef\xbb\xbf").unwrap_or(line)) else {
        return entry_of_value(&read::parse(line)?);
    };
    if !main_namespace(a.namespace.as_ref().and_then(|n| n.identifier.as_ref())) {
        return Ok(None);
    }
    if a.event.as_ref().and_then(|e| e.kind.as_deref()) == Some("delete") {
        return Ok(None);
    }
    let url = a.url.as_deref().unwrap_or("").to_string();
    let redirects = a.redirects.iter().flatten().filter_map(|r| redirect_url(&url, r.name.as_deref(), r.url.as_deref())).collect();
    Ok(Some(Entry { id: id_of(a.identifier.as_ref())?, revision: revision_of(a.version.as_ref().and_then(|v| v.identifier.as_ref()))?, url, redirects }))
}

fn entry_of_value(article: &Value) -> Result<Option<Entry>> {
    if !main_namespace(field(dict(article, "namespace")?, "identifier")) {
        return Ok(None);
    }
    if field(dict(article, "event")?, "type").and_then(Value::as_str) == Some("delete") {
        return Ok(None);
    }
    let url = text(article, "url").unwrap_or("").to_string();
    let redirects = match article.get("redirects") {
        Some(Value::Array(list)) => {
            list.iter().filter_map(|r| redirect_url(&url, r.get("name").and_then(Value::as_str), r.get("url").and_then(Value::as_str))).collect()
        }
        _ => Vec::new(),
    };
    Ok(Some(Entry { id: id_of(article.get("identifier"))?, revision: revision_of(field(dict(article, "version")?, "identifier"))?, url, redirects }))
}

/// What `redirects` found, for the summary line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Found {
    pub articles: u64,
    pub older_copies: u64,
    pub redirects: u64,
    pub not_in_run: u64,
    pub page_urls: u64,
    pub ambiguous: u64,
    pub kept: u64,
}

/// The latest copy of each article in `paths`, the same copy a run keeps: the highest revision, the first of equals.
fn latest(paths: &[PathBuf], found: &mut Found) -> Result<HashMap<i64, Entry>> {
    let mut latest: HashMap<i64, Entry> = HashMap::new();
    let mut keep = |entries: Vec<Result<Option<Entry>>>| -> Result<()> {
        for entry in entries {
            let Some(entry) = entry? else { continue };
            match latest.get(&entry.id) {
                Some(seen) if entry.revision <= seen.revision => found.older_copies += 1,
                _ => {
                    found.articles += u64::from(!latest.contains_key(&entry.id));
                    latest.insert(entry.id, entry);
                }
            }
        }
        Ok(())
    };
    for path in paths {
        let (tx, rx) = sync_channel::<Vec<Vec<u8>>>(64);
        let path = path.clone();
        let reader = std::thread::spawn(move || {
            let mut batch = Vec::with_capacity(BATCH);
            read::each_line(&path, |line| {
                batch.push(line);
                if batch.len() == BATCH && tx.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH))).is_err() {
                    return Ok(false);
                }
                Ok(true)
            })?;
            if !batch.is_empty() {
                tx.send(batch).ok();
            }
            anyhow::Ok(())
        });
        for batch in rx {
            keep(batch.par_iter().map(|line| entry(line)).collect())?;
        }
        reader.join().map_err(|_| anyhow!("the snapshot reader panicked"))??;
    }
    Ok(latest)
}

/// Redirect URL to page URL for each redirect that leads to exactly one of `pages` and is not a page itself.
pub fn redirects(paths: &[PathBuf], pages: &HashSet<String>, found: &mut Found) -> Result<BTreeMap<String, String>> {
    let latest = latest(paths, found)?;
    let mut to: BTreeMap<String, Option<String>> = BTreeMap::new();
    for entry in latest.into_values() {
        for redirect in entry.redirects {
            found.redirects += 1;
            if !pages.contains(&entry.url) {
                found.not_in_run += 1;
            } else if pages.contains(&redirect) {
                found.page_urls += 1;
            } else if redirect != entry.url && !redirect.contains(['\t', '\n', '\r']) {
                match to.get_mut(&redirect) {
                    None => {
                        to.insert(redirect, Some(entry.url.clone()));
                    }
                    Some(page) => {
                        if page.as_deref() != Some(entry.url.as_str()) {
                            *page = None;
                        }
                    }
                }
            }
        }
    }
    found.ambiguous = to.values().filter(|p| p.is_none()).count() as u64;
    let kept: BTreeMap<String, String> = to.into_iter().filter_map(|(r, p)| p.map(|p| (r, p))).collect();
    found.kept = kept.len() as u64;
    Ok(kept)
}

pub fn write(path: &Path, kept: &BTreeMap<String, String>) -> Result<()> {
    let partial = path.with_extension("tsv.partial");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&partial).with_context(|| format!("creating {}", partial.display()))?);
    for (redirect, page) in kept {
        writeln!(out, "{redirect}\t{page}")?;
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&partial, path)?;
    Ok(())
}

pub fn main(args: RedirectsArgs) -> Result<()> {
    let pages: HashSet<String> = bulk_core::runs::page_urls(&args.run)?.into_iter().collect();
    let mut found = Found::default();
    let kept = redirects(&args.files, &pages, &mut found)?;
    let out = args.out.clone().unwrap_or_else(|| args.run.join(FILE));
    write(&out, &kept)?;
    let line = json!({
        "out": out.to_string_lossy(),
        "pages": pages.len(),
        "articles": found.articles,
        "older_copies": found.older_copies,
        "redirects": found.redirects,
        "not_in_run": found.not_in_run,
        "page_urls": found.page_urls,
        "ambiguous": found.ambiguous,
        "kept": found.kept,
    });
    println!("{}", bulk_core::pyjson::dumps(&line, None));
    Ok(())
}

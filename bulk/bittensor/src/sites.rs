//! `site_pages` and `run_sites`: the subnets' fetched website pages, each read by the shared v2 extraction.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{anyhow, Result};
use bulk_core::py::{self, when_truthy};
use bulk_core::worker::{map_in_order, page_from_json, Worker, BLOCKED, EXTRACT};
use bulk_core::{upload_row, urlparse, Batches, Counts, Page};
use indexmap::IndexMap;
use regex::Regex;
use serde_json::{json, Value};

use crate::{remove_named, without_tracking};

pub const SITES_SOURCE: &str = "subnet-sites";
pub const MIN_TEXT_CHARS: usize = 200;
pub const DEFAULT_CONTENT_TYPE: &str = "text/html";

static APP_PATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)/(?:app|console|dashboard|account|me)(?:/|\n?\z)|/5[1-9A-HJ-NP-Za-km-z]{47}(?:/|\n?\z)").expect("pattern compiles"));

/// A fetched page by its file: the cleaned URL, the content type as fetched and the file's bytes.
struct Fetched {
    url: String,
    content_type: Value,
    html: Vec<u8>,
}

struct Read {
    page: Page,
    app_page: bool,
    blocked: bool,
}

/// Whether the URL's path is an app, console, dashboard, account or wallet page.
pub fn app_page(url: &str) -> Result<bool> {
    Ok(APP_PATH.is_match(&desearch::text::fold_turkish_i(&urlparse::urlsplit(url)?.path)))
}

/// `site_pages` before extraction: each page by file name in first-listed order, the last listing's URL kept.
fn fetched_pages(folder: &Path) -> Result<IndexMap<String, Fetched>> {
    let mut found = IndexMap::new();
    let sites = py::load_json(&folder.join("sites.json"))?;
    for site in py::list(&sites, "sites.json")? {
        for fetched in py::list(py::item(site, "pages")?, "a site's pages")? {
            let Some(name) = when_truthy(py::get(fetched, "file")?) else {
                continue;
            };
            let name = py::str_of(name, "a page file")?;
            let path = folder.join("html").join(name);
            let html = std::fs::read(&path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
            let url = without_tracking(py::str_of(py::item(fetched, "final_url")?, "a final_url")?)?;
            found.insert(name.to_string(), Fetched { url, content_type: py::item(fetched, "content_type")?.clone(), html });
        }
    }
    Ok(found)
}

fn read(worker: &mut Worker, fetched: &Fetched) -> Result<Read> {
    let size = u32::try_from(fetched.url.len())?.to_be_bytes();
    let html = py::decode_replace(&fetched.html);
    let answer = worker.ask_json(&[EXTRACT, &size, fetched.url.as_bytes(), html.as_bytes()])?;
    let page = page_from_json(answer.get("page").ok_or_else(|| anyhow!("an extract answer without a page"))?)?;
    let app_page = app_page(&fetched.url)?;
    let blocked = !app_page && {
        let request = json!({"status": 200, "html": py::text_replace(&fetched.html), "text": &page.text, "title": &page.title});
        let answer = worker.ask_json(&[BLOCKED, serde_json::to_string(&request)?.as_bytes()])?;
        answer.get("blocked").and_then(Value::as_bool).ok_or_else(|| anyhow!("a blocked answer without a verdict"))?
    };
    Ok(Read { page, app_page, blocked })
}

/// `run_sites`: each fetched page once, unless it is an app page, blocked or thin, then the pages the fetch named removed.
pub fn run_sites(folder: &Path, batches: &mut Batches, as_of_us: i64, workers: usize, spawn: &(dyn Fn() -> Result<Worker> + Sync)) -> Result<Counts> {
    let mut counts = Counts::default();
    let mut seen: HashSet<String> = HashSet::new();
    let pages: Vec<Fetched> = fetched_pages(folder)?.into_values().collect();
    let reads = map_in_order(&pages, workers, spawn, &read)?;
    for (fetched, read) in pages.into_iter().zip(reads) {
        if seen.contains(&fetched.url) {
            counts.add("duplicate", 1);
        } else if read.app_page {
            counts.add("app_page", 1);
        } else if read.blocked {
            counts.add("blocked", 1);
        } else if read.page.text.chars().count() < MIN_TEXT_CHARS {
            counts.add("thin", 1);
        } else {
            let content_type = match when_truthy(Some(&fetched.content_type)) {
                Some(v) => py::str_of(v, "a content type")?,
                None => DEFAULT_CONTENT_TYPE,
            };
            batches.add(upload_row(&fetched.url, read.page, as_of_us, content_type))?;
            seen.insert(fetched.url);
            counts.add("pages", 1);
        }
    }
    remove_named(folder, batches, &seen, &mut counts)?;
    Ok(counts)
}

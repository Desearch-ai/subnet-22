//! The anonymous `pmc-oa-opendata` bucket: which versions are taken, their inventory, and reads of their metadata and XML.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::http::Client;
use flate2::read::MultiGzDecoder;
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::ahead::bounded_map;
use crate::csv::CsvRows;
use crate::py::{cmp_decimal, decimal, str_of, truthy};

pub const BUCKET: &str = "https://pmc-oa-opendata.s3.amazonaws.com";
/// Reads in flight per worker, so the reader holds a few articles ahead rather than every name.
pub const AHEAD_PER_WORKER: usize = 4;
pub const COMMERCIAL: [&str; 4] = ["CC0", "CC BY", "CC BY-SA", "CC BY-ND"];
pub const INVENTORY_LISTING: &str = "?list-type=2&delimiter=/&prefix=inventory-reports/pmc-oa-opendata/metadata/";

/// One version's metadata, with its XML only when the version is taken.
pub type Article = (Value, Option<Vec<u8>>);

/// Why an article version stays out, or None for one we take.
pub fn left_out(meta: &Value) -> Option<&'static str> {
    if truthy(meta.get("gone")) {
        return Some("gone");
    }
    if !truthy(meta.get("is_pmc_openaccess")) {
        return Some("not_open_access");
    }
    if !matches!(meta.get("license_code"), Some(Value::String(code)) if COMMERCIAL.contains(&code.as_str())) {
        return Some("license");
    }
    if truthy(meta.get("is_retracted")) {
        return Some("retracted");
    }
    None
}

/// The newest version of each PMCID, in the order PMCIDs are first seen.
#[derive(Default)]
pub struct Newest(IndexMap<String, (String, String)>);

impl Newest {
    pub fn add(&mut self, key: &str) {
        let file = key.rsplit('/').next().unwrap_or(key);
        let name = file.strip_suffix(".json").unwrap_or(file);
        let (pmcid, version) = name.split_once('.').unwrap_or((name, ""));
        let Some(version) = decimal(version).filter(|_| pmcid.starts_with("PMC")) else { return };
        match self.0.get_mut(pmcid) {
            Some(entry) if cmp_decimal(&version, &entry.0).is_gt() => *entry = (version, name.to_string()),
            Some(_) => {}
            None => {
                self.0.insert(pmcid.to_string(), (version, name.to_string()));
            }
        }
    }

    pub fn finish(self) -> IndexMap<String, String> {
        self.0.into_iter().map(|(pmcid, (_, name))| (pmcid, name)).collect()
    }
}

/// `bulk.pmc.latest_versions`.
pub fn latest_versions<S: AsRef<str>>(keys: impl IntoIterator<Item = S>) -> IndexMap<String, String> {
    let mut newest = Newest::default();
    for key in keys {
        newest.add(key.as_ref());
    }
    newest.finish()
}

fn open_text(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(if name.len() > ".gz".len() && name.ends_with(".gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(BufReader::new(file))))
    } else {
        Box::new(BufReader::new(file))
    })
}

/// `bulk.pmc.inventory_keys`: each metadata key in the S3 inventory CSV files, gzipped or not.
pub fn inventory_keys(paths: &[PathBuf], mut each: impl FnMut(&str)) -> Result<()> {
    for path in paths {
        for row in CsvRows::new(open_text(path)?) {
            let row = row.with_context(|| format!("reading {}", path.display()))?;
            if row.len() > 1 && row[1].starts_with("metadata/") {
                each(&row[1]);
            }
        }
    }
    Ok(())
}

/// `latest_versions(inventory_keys(paths))`.
pub fn latest_from_inventory(paths: &[PathBuf]) -> Result<IndexMap<String, String>> {
    let mut newest = Newest::default();
    inventory_keys(paths, |key| newest.add(key))?;
    Ok(newest.finish())
}

/// The newest version names of an inventory, sorted, as the bucket runs read them.
pub fn sorted_names(paths: &[PathBuf]) -> Result<Vec<String>> {
    let mut names: Vec<String> = latest_from_inventory(paths)?.into_values().collect();
    names.sort();
    Ok(names)
}

/// `bulk.pmc.fetch`: None when the bucket answers 404.
pub fn fetch(client: &Client, base: &str, key: &str) -> Result<Option<Vec<u8>>> {
    client.get_retrying(&format!("{base}/{key}")).with_context(|| format!("reading {key}"))
}

fn fetch_found(client: &Client, base: &str, key: &str) -> Result<Vec<u8>> {
    fetch(client, base, key)?.ok_or_else(|| anyhow!("HTTP Error 404: Not Found: {base}/{key}"))
}

/// `json.loads` of a metadata file.
pub fn parse_meta(data: &[u8]) -> Result<Value> {
    match serde_json::from_slice(data) {
        Ok(meta) => Ok(meta),
        Err(error) => {
            let text = std::str::from_utf8(data).map_err(|_| error)?;
            Ok(serde_json::from_str(&bulk_core::text::drop_lone_surrogates(text))?)
        }
    }
}

fn gone(name: &str) -> Value {
    json!({"pmcid": name.split('.').next().unwrap_or(name), "gone": true})
}

fn one(client: &Client, base: &str, name: &str) -> Result<Article> {
    let Some(data) = fetch(client, base, &format!("metadata/{name}.json"))? else { return Ok((gone(name), None)) };
    let meta = parse_meta(&data).with_context(|| format!("metadata of {name}"))?;
    if left_out(&meta).is_some() {
        return Ok((meta, None));
    }
    match fetch(client, base, &format!("{name}/{name}.xml"))? {
        Some(xml) => Ok((meta, Some(xml))),
        None => Ok((gone(name), None)),
    }
}

/// `bulk.pmc.from_bucket`: in order, at most `workers * AHEAD_PER_WORKER` reads in flight; a 404 means the version is gone.
pub fn from_bucket<I>(names: I, client: Arc<Client>, base: String, workers: usize) -> impl Iterator<Item = Result<Article>>
where
    I: Iterator<Item = String>,
{
    bounded_map(names, workers, workers.max(1) * AHEAD_PER_WORKER, move |name: String| one(&client, &base, &name))
}

/// `bulk.pmc.taken_pmids`: the PMIDs of the versions taken, from their metadata alone.
pub fn taken_pmids<I>(names: I, client: Arc<Client>, base: String, workers: usize) -> impl Iterator<Item = Result<String>>
where
    I: Iterator<Item = String>,
{
    let pmids = bounded_map(names, workers, workers.max(1) * AHEAD_PER_WORKER, move |name: String| -> Result<String> {
        let meta = parse_meta(&fetch_found(&client, &base, &format!("metadata/{name}.json"))?)?;
        if left_out(&meta).is_some() || !truthy(meta.get("pmid")) {
            return Ok(String::new());
        }
        Ok(meta.get("pmid").map(str_of).unwrap_or_default())
    });
    pmids.filter(|pmid| !matches!(pmid, Ok(p) if p.is_empty()))
}

/// `bulk.pmc.from_folder`: the `PMC*.json` files in name order, each with its `.xml` when taken.
pub fn from_folder(folder: &Path) -> Result<impl Iterator<Item = Result<Article>>> {
    let mut paths: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(folder).with_context(|| format!("reading --folder {}", folder.display()))? {
        let entry = entry?;
        let Ok(name) = entry.file_name().into_string() else { continue };
        if name.starts_with("PMC") && name.ends_with(".json") && name.len() >= "PMC.json".len() {
            paths.push((name, entry.path()));
        }
    }
    if paths.is_empty() {
        bail!("--folder {} holds no PMC*.json metadata", folder.display());
    }
    paths.sort();
    Ok(paths.into_iter().map(|(_, path)| -> Result<Article> {
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let meta = parse_meta(text.as_bytes()).with_context(|| format!("reading {}", path.display()))?;
        let xml = path.with_extension("xml");
        let xml = if xml.exists() && left_out(&meta).is_none() { Some(fs::read(&xml)?) } else { None };
        Ok((meta, xml))
    }))
}

/// `bulk.pmc.latest_inventory`: the newest daily inventory's CSV files, downloaded into `dest` unless there already.
pub fn latest_inventory(dest: &Path, client: &Client, base: &str) -> Result<Vec<PathBuf>> {
    let listing = fetch_found(client, base, INVENTORY_LISTING)?;
    let listing = String::from_utf8(listing).context("the inventory listing")?;
    let doc = roxmltree::Document::parse(&listing).context("the inventory listing")?;
    let mut days: Vec<&str> = doc
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "Prefix")
        .filter_map(|n| n.text())
        .filter(|t| !t.is_empty() && t.trim_end_matches('/').ends_with('Z'))
        .collect();
    days.sort();
    let Some(day) = days.last() else { bail!("no daily inventory in the listing") };
    let manifest: Value = serde_json::from_slice(&fetch_found(client, base, &format!("{day}manifest.json"))?)?;
    fs::create_dir_all(dest)?;
    let mut paths = Vec::new();
    for file in manifest["files"].as_array().ok_or_else(|| anyhow!("an inventory manifest without files"))? {
        let key = file["key"].as_str().ok_or_else(|| anyhow!("an inventory file without a key"))?;
        let path = dest.join(key.rsplit('/').next().unwrap_or(key));
        if !path.exists() {
            let partial = path.with_extension("partial");
            fs::write(&partial, fetch_found(client, base, key)?)?;
            fs::rename(&partial, &path)?;
        }
        paths.push(path);
    }
    Ok(paths)
}

/// Whitespace-separated names, as `read_text().split()` gives them.
pub fn read_names(path: &Path) -> Result<Vec<String>> {
    let mut text = String::new();
    File::open(path).with_context(|| format!("opening {}", path.display()))?.read_to_string(&mut text)?;
    Ok(desearch::text::words(&text).map(str::to_string).collect())
}

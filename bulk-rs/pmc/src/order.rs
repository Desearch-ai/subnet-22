//! `pmc-order`: version names by OpenAlex citations per year, read from the works Parquet snapshot by HTTP range reads.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use bulk_core::http::Client;
pub use bulk_core::ranged::RangeClient;
use bulk_core::ranged::{column_chunks, int_at, leaves, metadata, Strings};
use flate2::read::MultiGzDecoder;
use indexmap::IndexMap;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;

use crate::ahead::bounded_map;
use crate::csv::CsvRows;
use crate::py::{decimal, decimal_value};

pub const PMC_IDS: &str = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz";
pub const OPENALEX_WORKS: &str = "https://openalex.s3.amazonaws.com/data/parquet/works/manifest.json";
pub const PUBMED_URL: &str = "https://pubmed.ncbi.nlm.nih.gov/";
pub const COLUMNS: [&str; 2] = ["ids", "cited_by_count"];

/// `bulk.pmc.pmc_ids`: PMCID to its PMID and year, from NCBI's PMC-ids.csv.gz.
pub fn pmc_ids(path: &Path) -> Result<HashMap<String, (String, i64)>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut rows = CsvRows::new(std::io::BufReader::new(MultiGzDecoder::new(std::io::BufReader::new(file))));
    let header = rows.next().transpose()?.unwrap_or_default();
    let column = |name: &str| header.iter().rposition(|h| h == name).ok_or_else(|| anyhow!("{} has no {name} column", path.display()));
    let (pmcid, pmid, year) = (column("PMCID")?, column("PMID")?, column("Year")?);
    let mut ids = HashMap::new();
    for row in rows {
        let row = row?;
        if row.is_empty() {
            continue;
        }
        let field = |i: usize| row.get(i).map(String::as_str).unwrap_or("");
        if field(pmcid).is_empty() {
            continue;
        }
        let published = decimal(field(year)).map_or(0, |y| decimal_value(&y));
        ids.insert(field(pmcid).to_string(), (field(pmid).to_string(), published));
    }
    Ok(ids)
}

/// `bulk.pmc.order`: most cited per year since publication first, the newest PMCID first among equals.
pub fn order(names: &IndexMap<String, String>, ids: &HashMap<String, (String, i64)>, cited: &HashMap<String, i64>, year: i64) -> Vec<String> {
    let rank = |pmcid: &str| -> (f64, i64) {
        let per_year = match ids.get(pmcid) {
            Some((pmid, published)) if *published != 0 => *cited.get(pmid).unwrap_or(&0) as f64 / (year - published + 1).max(1) as f64,
            _ => 0.0,
        };
        let number = pmcid.strip_prefix("PMC").unwrap_or(pmcid);
        (-per_year, -decimal(number).map_or(0, |n| decimal_value(&n)))
    };
    let mut ranked: Vec<((f64, i64), &String)> = names.iter().map(|(pmcid, name)| (rank(pmcid), name)).collect();
    ranked.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap_or(std::cmp::Ordering::Equal).then(a.0 .1.cmp(&b.0 .1)));
    ranked.into_iter().map(|(_, name)| name.clone()).collect()
}

/// The works files the snapshot's manifest lists, as HTTPS URLs of the bucket's objects.
pub fn openalex_files(client: &Client) -> Result<Vec<String>> {
    files_listed(client, OPENALEX_WORKS)
}

/// The files a works manifest at `manifest` lists: `s3://` ones as HTTPS URLs of the bucket's objects, `http(s)://`
/// ones as they are.
pub fn files_listed(client: &Client, manifest: &str) -> Result<Vec<String>> {
    let listed = client.get_retrying(manifest)?.ok_or_else(|| anyhow!("HTTP Error 404: {manifest}"))?;
    let listed: serde_json::Value = serde_json::from_slice(&listed)?;
    let files = listed["files"].as_array().ok_or_else(|| anyhow!("a works manifest without files"))?;
    files
        .iter()
        .map(|entry| {
            let url = entry["url"].as_str().ok_or_else(|| anyhow!("a works file without a url"))?;
            if url.starts_with("http://") || url.starts_with("https://") {
                return Ok(url.to_string());
            }
            let path = url.strip_prefix("s3://").unwrap_or(url);
            let (bucket, key) = path.split_once('/').ok_or_else(|| anyhow!("a works file outside a bucket: {url}"))?;
            Ok(format!("https://{bucket}.s3.amazonaws.com/{key}"))
        })
        .collect()
}

/// `pmc.map_lookup(ids, "pmid", "first")` with its counts, for the rows whose PMID is wanted.
fn batch_citations(batch: &RecordBatch, wanted: &HashSet<String>, found: &mut HashMap<String, i64>) -> Result<()> {
    let ids = batch.column_by_name("ids").ok_or_else(|| anyhow!("no ids column"))?;
    let cited = batch.column_by_name("cited_by_count").ok_or_else(|| anyhow!("no cited_by_count column"))?;
    let ids = ids.as_map_opt().ok_or_else(|| anyhow!("ids is {} rather than a map", ids.data_type()))?;
    let (keys, values) = (ids.keys(), ids.values());
    let (keys, values) = (Strings::of(keys)?, Strings::of(values)?);
    let offsets = ids.value_offsets();
    for row in 0..ids.len() {
        if ids.is_null(row) {
            continue;
        }
        let entries = offsets[row] as usize..offsets[row + 1] as usize;
        let Some(entry) = entries.into_iter().find(|&j| keys.get(j) == Some("pmid")) else { continue };
        let Some(url) = values.get(entry) else { continue };
        let key = url.strip_prefix(PUBMED_URL).unwrap_or(url);
        if wanted.contains(key) {
            let count = int_at(cited, row)?.unwrap_or(0);
            let best = found.entry(key.to_string()).or_insert(0);
            *best = (*best).max(count);
        }
    }
    Ok(())
}

/// The citation counts of the wanted PMIDs in one works file, reading only its `ids` and `cited_by_count` column chunks.
pub fn file_citations(client: &RangeClient, url: &str, wanted: &HashSet<String>) -> Result<HashMap<String, i64>> {
    let (meta, len) = metadata(client, url)?;
    let meta = ArrowReaderMetadata::try_new(Arc::new(meta), ArrowReaderOptions::new())?;
    let schema = meta.metadata().file_metadata().schema_descr();
    let mask = ProjectionMask::columns(schema, COLUMNS);
    let leaves = leaves(schema, &COLUMNS);
    let mut found = HashMap::new();
    for group in 0..meta.metadata().num_row_groups() {
        let chunks = column_chunks(client, url, len, meta.metadata().row_group(group), &leaves)?;
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(chunks, meta.clone()).with_projection(mask.clone()).with_row_groups(vec![group]).build()?;
        for batch in reader {
            batch_citations(&batch?, wanted, &mut found)?;
        }
    }
    Ok(found)
}

/// `bulk.pmc.citations` over the works files' URLs.
pub fn citations(urls: &[String], client: Arc<RangeClient>, wanted: Arc<HashSet<String>>, workers: usize) -> Result<HashMap<String, i64>> {
    let mut counts: HashMap<String, i64> = HashMap::new();
    let files = bounded_map(urls.iter().cloned(), workers, workers.max(1) * 2, move |url: String| {
        file_citations(&client, &url, &wanted).with_context(|| format!("reading {url}"))
    });
    for found in files {
        for (key, cited) in found? {
            let best = counts.entry(key).or_insert(0);
            *best = (*best).max(cited);
        }
    }
    Ok(counts)
}

/// The current UTC year.
pub fn this_year() -> i64 {
    year_of(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64)
}

/// The UTC year of a Unix time.
pub fn year_of(secs: i64) -> i64 {
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    yoe + era * 400 + i64::from(mp >= 10)
}

#[cfg(test)]
mod tests {
    #[test]
    fn years_change_at_utc_midnight_on_new_year() {
        assert_eq!(super::year_of(0), 1970);
        assert_eq!(super::year_of(1_767_225_599), 2025);
        assert_eq!(super::year_of(1_767_225_600), 2026);
        assert_eq!(super::year_of(951_782_400), 2000);
    }
}

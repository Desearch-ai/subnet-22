//! Time one sitemap file through what a visit does with it: decompress, hash, parse, date, normalise, store.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use desearch_bot::buckets::{bucket_of, Buckets, Resources};
use desearch_bot::urls::Normaliser;
use desearch_bot::{isodate, schedule, sitemaps, visit};
use sha2::{Digest, Sha256};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, file, host, dir] = &args[..] else {
        anyhow::bail!("usage: digest FILE HOST STORE_DIR");
    };
    let raw = std::fs::read(file).with_context(|| format!("reading {file}"))?;
    let resources = Resources::new(256 << 20, 256 << 20, 1);
    let buckets = Buckets::open(Path::new(dir), &[bucket_of(host)], &resources)?;
    let store = buckets.store(host);
    for round in 0..5 {
        let started = Instant::now();
        let body = visit::gunzip(&raw);
        let hash: String = Sha256::digest(&body).iter().take(16).map(|b| format!("{b:02x}")).collect();
        let (kind, entries) = sitemaps::parse_entries(&body);
        let parsed = started.elapsed();
        let now = chrono::Utc::now().timestamp_micros();
        let dates: Vec<Option<i64>> = entries
            .iter()
            .map(|e| schedule::plausible(isodate::parse_lastmod(e.lastmod.as_deref().or(e.published.as_deref())), now))
            .collect();
        let normaliser = Normaliser::new(host)?;
        let rows: Vec<_> = entries
            .iter()
            .zip(&dates)
            .filter_map(|(e, date)| {
                let timed = sitemaps::has_time(e.lastmod.as_deref().or(e.published.as_deref()));
                Some((normaliser.parse(&e.url)?, visit::epoch(*date) as u32, timed))
            })
            .collect();
        let normalised = started.elapsed();
        let listing = store.record_listing(1, rows, visit::epoch(Some(now)) as u32)?;
        let total = started.elapsed();
        println!(
            "{}",
            serde_json::json!({
                "round": round, "kind": kind, "hash": hash, "entries": entries.len(), "listed": listing.listed, "new": listing.new,
                "parse_ms": parsed.as_secs_f64() * 1e3,
                "normalise_ms": (normalised - parsed).as_secs_f64() * 1e3,
                "store_ms": (total - normalised).as_secs_f64() * 1e3,
                "total_ms": total.as_secs_f64() * 1e3,
            })
        );
    }
    Ok(())
}

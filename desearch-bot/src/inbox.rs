//! URL lists dropped in a directory, one file per host, taken in as that host's listing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;

use crate::buckets::{sitemap_id, Buckets};
use crate::{isodate, urls};

/// URLs taken in at once; each part keeps a listing of its own, so a file read again only queues what changed.
const PART: usize = 50_000;
const LOOK_EVERY: Duration = Duration::from_secs(300);

/// What one look at the inbox took in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Taken {
    pub files: usize,
    pub urls: usize,
    pub queued: usize,
    /// Files for hosts this process holds no domain record of, left until the host has its registry row.
    pub unknown: usize,
}

/// Takes in every `<host>.txt` changed since `seen` the way a sitemap's pages come in, so the host's list entry, cap, date floor and robots.txt hold.
pub fn take_in(buckets: &Buckets, dir: &Path, seen: &mut HashMap<String, SystemTime>, now: u32) -> Result<Taken> {
    let mut taken = Taken::default();
    for item in std::fs::read_dir(dir)? {
        let path = item?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()).and_then(|name| name.strip_suffix(".txt")) else {
            continue;
        };
        let name = name.to_lowercase();
        let host = name.strip_prefix("www.").unwrap_or(&name).to_string();
        let Some(store) = buckets.store_for(&host) else {
            continue;
        };
        if !store.has_domain(&host)? {
            taken.unknown += 1;
            continue;
        }
        let modified = std::fs::metadata(&path)?.modified()?;
        if seen.get(&host) == Some(&modified) {
            continue;
        }
        let rows: Vec<(urls::Url, u32, bool)> = std::fs::read_to_string(&path)?
            .lines()
            .filter_map(|line| {
                let (url, date) = line.split_once('\t').unwrap_or((line, ""));
                let url = urls::parse(url.trim(), &host)?;
                let lastmod = isodate::parse_lastmod(Some(date.trim()).filter(|date| !date.is_empty())).map_or(0, |micros| (micros / 1_000_000).clamp(0, i64::from(u32::MAX)) as u32);
                Some((url, lastmod, false))
            })
            .collect();
        for (part, chunk) in rows.chunks(PART).enumerate() {
            taken.queued += store.record_listing(&host, sitemap_id(&format!("inbox://{host}/{part}")), chunk.to_vec(), now)?.ready;
        }
        taken.files += 1;
        taken.urls += rows.len();
        seen.insert(host, modified);
    }
    Ok(taken)
}

/// Looks at the inbox every few minutes until told to stop.
pub fn keep_taking(buckets: &Buckets, dir: &Path, stop: &AtomicBool) {
    let mut seen = HashMap::new();
    let mut looked: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        if looked.is_none_or(|at| at.elapsed() >= LOOK_EVERY) {
            looked = Some(Instant::now());
            match take_in(buckets, dir, &mut seen, crate::dispatch::now()) {
                Ok(taken) if taken.files > 0 => println!(
                    "[rs] inbox took {} files, {} URLs, {} queued; {} files wait for their host's registry row",
                    taken.files, taken.urls, taken.queued, taken.unknown
                ),
                Ok(_) => {}
                Err(error) => eprintln!("[rs] reading the inbox {} failed: {error:#}", dir.display()),
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

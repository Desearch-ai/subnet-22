//! URL lists dropped in a directory, one file per host, taken in as that host's listing.

use std::io::BufRead;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::Result;
use serde_json::json;

use crate::buckets::{sitemap_id, BucketStore, Buckets};
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
    /// Files that could not be read; the rest of the inbox goes on without them.
    pub failed: usize,
    pub stopped: bool,
}

/// Takes in every `<host>.txt` changed since it was last taken, a URL and optionally a tab and its date a line, the way a sitemap's pages come in.
pub fn take_in(buckets: &Buckets, dir: &Path, now: u32, stop: &AtomicBool) -> Result<Taken> {
    let mut taken = Taken::default();
    for item in std::fs::read_dir(dir)? {
        if stop.load(Ordering::Relaxed) {
            taken.stopped = true;
            break;
        }
        let Ok(item) = item else {
            taken.failed += 1;
            continue;
        };
        let path = item.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()).and_then(|name| name.strip_suffix(".txt")) else {
            continue;
        };
        let name = name.to_lowercase();
        let host = name.strip_prefix("www.").unwrap_or(&name).to_string();
        let Some(store) = buckets.store_for(&host) else {
            continue;
        };
        match take_file(store, &host, &path, now, stop) {
            Ok(None) => {}
            Ok(Some((urls, queued, finished))) => {
                taken.files += 1;
                taken.urls += urls;
                taken.queued += queued;
                taken.stopped |= !finished;
            }
            Err(error) => {
                eprintln!("[rs] inbox file {} skipped: {error:#}", path.display());
                taken.failed += 1;
            }
        }
        if !store.has_domain(&host).unwrap_or(false) {
            taken.unknown += 1;
        }
    }
    Ok(taken)
}

/// One host's file, in parts: the URLs read, how many went on the list, and whether it was read to the end; None when unchanged or the host is unknown.
fn take_file(store: &BucketStore, host: &str, path: &Path, now: u32, stop: &AtomicBool) -> Result<Option<(usize, usize, bool)>> {
    if !store.has_domain(host)? {
        return Ok(None);
    }
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs();
    let seen = json!({"modified": modified, "size": metadata.len()});
    let mark = format!("inbox:{host}");
    if store.meta(&mark)?.as_ref() == Some(&seen) {
        return Ok(None);
    }
    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let (mut line, mut rows, mut part) = (Vec::new(), Vec::with_capacity(PART), 0usize);
    let (mut urls, mut queued) = (0, 0);
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read > 0 {
            let text = String::from_utf8_lossy(&line);
            let (url, date) = text.trim_end_matches(['\r', '\n']).split_once('\t').unwrap_or((text.trim_end_matches(['\r', '\n']), ""));
            if let Some(url) = urls::parse(url.trim(), host) {
                let lastmod = isodate::parse_lastmod(Some(date.trim()).filter(|date| !date.is_empty())).map_or(0, |micros| (micros / 1_000_000).clamp(0, i64::from(u32::MAX)) as u32);
                rows.push((url, lastmod, false));
            }
        }
        if rows.len() >= PART || (read == 0 && !rows.is_empty()) {
            urls += rows.len();
            queued += store.record_listing(host, sitemap_id(&format!("inbox://{host}/{part}")), std::mem::take(&mut rows), now)?.ready;
            part += 1;
            if stop.load(Ordering::Relaxed) && read > 0 {
                return Ok(Some((urls, queued, false)));
            }
        }
        if read == 0 {
            break;
        }
    }
    store.set_meta(&mark, &seen)?;
    Ok(Some((urls, queued, true)))
}

/// Looks at the inbox every few minutes until told to stop.
pub fn keep_taking(buckets: &Buckets, dir: &Path, stop: &AtomicBool) {
    let mut looked: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        if looked.is_none_or(|at| at.elapsed() >= LOOK_EVERY) {
            looked = Some(Instant::now());
            match take_in(buckets, dir, crate::dispatch::now(), stop) {
                Ok(taken) if taken.files > 0 || taken.failed > 0 => println!(
                    "[rs] inbox took {} files, {} URLs, {} queued; {} files skipped as unreadable, {} wait for their host's registry row",
                    taken.files, taken.urls, taken.queued, taken.failed, taken.unknown
                ),
                Ok(_) => {}
                Err(error) => eprintln!("[rs] reading the inbox {} failed: {error:#}", dir.display()),
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

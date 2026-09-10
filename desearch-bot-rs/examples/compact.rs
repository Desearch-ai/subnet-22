//! Compact one bucket store with the crawler's current options and report its size before and after.

use std::path::Path;
use std::time::Instant;

use desearch_bot::buckets::{BucketStore, Resources};

fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum()
}

fn main() {
    let dir = std::env::args().nth(1).expect("store directory");
    let dir = Path::new(&dir);
    let before = size(dir);
    let resources = Resources::new(256 << 20, 256 << 20, 1);
    let store = BucketStore::open(dir, &resources).unwrap();
    let started = Instant::now();
    store.compact();
    drop(store);
    println!("{:.0} MB -> {:.0} MB in {:.0} s", before as f64 / 1e6, size(dir) as f64 / 1e6, started.elapsed().as_secs_f64());
}

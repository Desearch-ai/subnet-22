//! Read real outcome files the way the bot does and report rows per minute: fetching and parsing, then tallying and scheduling into throwaway stores.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use desearch_bot::buckets::{Buckets, Changes, Resources};
use desearch_bot::outcomes::{Next, OutcomeFeed, OutcomeReader, OutcomeRow};
use desearch_bot::records;
use desearch_bot::recrawl::Recrawl;
use desearch_bot::states::State;

const AHEAD: usize = 8;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let (base, files, dir) = (args.next().expect("feed base URL"), args.next().expect("files to read").parse::<u64>().unwrap(), PathBuf::from(args.next().expect("scratch directory")));
    let feed = OutcomeFeed::new(&base).unwrap();
    let latest = feed.latest().await.unwrap().expect("a feed with files");
    let first = latest.saturating_sub(files - 1);

    let started = Instant::now();
    let mut parsed: Vec<Vec<OutcomeRow>> = Vec::new();
    let seqs: Vec<u64> = (first..=latest).collect();
    for chunk in seqs.chunks(AHEAD) {
        let fetching: Vec<_> = chunk.iter().map(|&seq| (seq, tokio::spawn({
            let feed = feed.clone();
            async move { feed.fetch_below(seq, Some(latest)).await }
        }))).collect();
        for (seq, handle) in fetching {
            match handle.await.unwrap() {
                Ok(Next::File { rows, .. }) => parsed.push(rows),
                Ok(_) => eprintln!("file {seq} not there"),
                Err(error) => eprintln!("file {seq}: {error:#}"),
            }
        }
    }
    let fetched = started.elapsed().as_secs_f64();
    let rows: usize = parsed.iter().map(Vec::len).sum();
    println!("fetched and parsed {} files, {rows} rows in {fetched:.1} s: {:.0} rows/min, {AHEAD} files at once", parsed.len(), rows as f64 / fetched * 60.0);

    let _ = std::fs::remove_dir_all(&dir);
    let resources = Resources::new(512 << 20, 512 << 20, 256);
    let owned: Vec<usize> = (0..256).collect();
    let buckets = Buckets::open(&dir, &owned, &resources).unwrap();
    let hosts: HashSet<&str> = parsed.iter().flatten().map(|row| row.host.as_str()).collect();
    for host in &hosts {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1000), Some("com"), &[], State::Active, None, None));
        buckets.store(host).write(changes).unwrap();
    }
    println!("{} hosts made domains", hosts.len());

    let mut reader = OutcomeReader::new(Recrawl::default(), 200_000);
    let now = desearch_bot::dispatch::now();
    let started = Instant::now();
    let (mut failed, mut dropped) = (0, 0);
    for file in &parsed {
        let counted = reader.apply(&buckets, file, now).unwrap();
        failed += counted.failed;
        dropped += counted.dropped;
    }
    let applied = started.elapsed().as_secs_f64();
    println!("tallied and scheduled {rows} rows ({failed} failed, {dropped} dropped) in {applied:.1} s: {:.0} rows/min on one thread", rows as f64 / applied * 60.0);
    drop(buckets);
    std::fs::remove_dir_all(&dir).ok();
}

//! The current state built once from an index snapshot: each current page is read from the change file and row the index points at.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::Array;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::compactor::{self, each, Spill, Store};
use crate::current::{self, shard_of, ChangeRow, FileRef, Manifest};

/// Wanted rows are sorted into this many files by change file, so one group fits in memory.
const GROUPS: usize = 512;
/// Shards written at once; each holds its whole shard in memory, about 2 GB at full size.
const WRITERS: usize = 4;

pub struct Settings {
    pub prefix: String,
    pub work: PathBuf,
    pub threads: usize,
    /// Only these shards, for a trial on part of the data.
    pub shards: Range<usize>,
    /// Only pages whose current row is in these change files, for a trial on part of the data.
    pub seqs: Range<i64>,
    pub spill_bytes: usize,
    /// Build without the snapshot's pages that name no change file row, instead of refusing.
    pub allow_unlocated: bool,
}

#[derive(Debug, Default)]
pub struct Report {
    pub wanted: u64,
    /// Snapshot pages with no change file row to read them from.
    pub unlocated: u64,
    pub written: u64,
    pub files_read: u64,
    /// The row the index points at holds another key, or no page.
    pub misplaced: u64,
    /// The row's content hashes to another version than the index holds.
    pub other_version: u64,
    pub through_seq: i64,
    pub shards: usize,
    pub fetch_s: f64,
    pub decode_s: f64,
}

struct Want {
    row: usize,
    key: String,
    version: String,
}

/// Sorts the snapshot's rows by the change file they live in, into `GROUPS` files; returns how many, the newest change file, and how many name none.
fn sort_wants(snapshot: &Path, settings: &Settings, groups: &Path) -> Result<(u64, i64, u64)> {
    std::fs::create_dir_all(groups)?;
    let mut writers: Vec<BufWriter<File>> =
        (0..GROUPS).map(|g| Ok(BufWriter::new(File::create(groups.join(format!("{g:04}.tsv")))?))).collect::<Result<_>>()?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(snapshot).with_context(|| format!("opening {}", snapshot.display()))?)?
        .with_batch_size(65_536)
        .build()?;
    let (mut wanted, mut through, mut unlocated) = (0u64, 0i64, 0u64);
    for batch in reader {
        let batch = batch?;
        let column = |name: &str| batch.column(batch.schema().index_of(name).expect("a snapshot column")).clone();
        let (keys, versions, seqs, rows) = (column("key"), column("version"), column("change_seq"), column("change_row"));
        let (keys, versions) = (keys.as_string::<i32>(), versions.as_string::<i32>());
        let (seqs, rows) = (seqs.as_primitive::<Int64Type>(), rows.as_primitive::<Int64Type>());
        for i in 0..batch.num_rows() {
            let key = keys.value(i);
            let Some(shard) = shard_of(key) else { continue };
            if seqs.is_null(i) || rows.is_null(i) {
                unlocated += u64::from(settings.shards.contains(&shard));
                continue;
            }
            let seq = seqs.value(i);
            through = through.max(seq);
            if !settings.shards.contains(&shard) || !settings.seqs.contains(&seq) {
                continue;
            }
            writeln!(writers[(seq as usize) % GROUPS], "{seq}\t{}\t{key}\t{}", rows.value(i), versions.value(i))?;
            wanted += 1;
        }
    }
    for writer in &mut writers {
        writer.flush()?;
    }
    Ok((wanted, through, unlocated))
}

fn read_group(path: &Path) -> Result<HashMap<i64, Vec<Want>>> {
    let mut wants: HashMap<i64, Vec<Want>> = HashMap::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let mut fields = line.splitn(4, '\t');
        let mut next = || fields.next().ok_or_else(|| anyhow!("a short line in {}", path.display()));
        let seq: i64 = next()?.parse()?;
        let row: usize = next()?.parse()?;
        let (key, version) = (next()?.to_string(), next()?.to_string());
        wants.entry(seq).or_default().push(Want { row, key, version });
    }
    Ok(wants)
}

/// Builds every shard in `settings.shards` from `snapshot`, a local copy of `index/snapshots/<day>.parquet`.
pub fn build(store: &dyn Store, snapshot: &Path, settings: &Settings) -> Result<Report> {
    let groups = settings.work.join("wants");
    let (wanted, through, unlocated) = sort_wants(snapshot, settings, &groups)?;
    if unlocated > 0 && !settings.allow_unlocated {
        bail!("{unlocated} pages of the snapshot name no change file row to build them from; --allow-unlocated builds without them");
    }
    let mut report = Report { wanted, unlocated, through_seq: through, ..Report::default() };
    println!("{wanted} pages to read, the snapshot reaches change file {through}");
    let spill = Mutex::new(Spill::new(&settings.work.join("spill"), settings.spill_bytes)?);
    let counted = Mutex::new((0u64, 0u64, 0u64));
    let timed = Mutex::new((0.0f64, 0.0f64));
    for group in 0..GROUPS {
        let wants = read_group(&groups.join(format!("{group:04}.tsv")))?;
        let seqs: Vec<i64> = wants.keys().copied().collect();
        each(&seqs, settings.threads, |&seq| -> Result<()> {
            let started = Instant::now();
            let key = compactor::change_key(store, seq)?.ok_or_else(|| anyhow!("change file {seq} has no number"))?;
            let file = store.get(&key)?.ok_or_else(|| anyhow!("change file {seq} ({key}) is missing"))?;
            let fetched = started.elapsed().as_secs_f64();
            let rows_wanted: HashSet<usize> = wants[&seq].iter().map(|want| want.row).collect();
            let rows: HashMap<usize, ChangeRow> = current::decode_changes_where(file, seq, |row| rows_wanted.contains(&row))
                .with_context(|| format!("change file {seq} ({key})"))?
                .into_iter()
                .collect();
            {
                let mut timed = timed.lock().unwrap();
                timed.0 += fetched;
                timed.1 += started.elapsed().as_secs_f64() - fetched;
            }
            let (mut misplaced, mut other_version, mut found) = (0, 0, Vec::new());
            for want in &wants[&seq] {
                match rows.get(&want.row) {
                    Some(ChangeRow::Page(page)) if page.key == want.key => {
                        if page.version != want.version {
                            other_version += 1;
                        }
                        found.push(*page.clone());
                    }
                    _ => misplaced += 1,
                }
            }
            let mut spill = spill.lock().unwrap();
            for page in found {
                if let Some(shard) = shard_of(&page.key) {
                    spill.push(shard, page)?;
                }
            }
            drop(spill);
            let mut counted = counted.lock().unwrap();
            counted.0 += 1;
            counted.1 += misplaced;
            counted.2 += other_version;
            Ok(())
        })?;
        if group % 32 == 31 {
            let counted = counted.lock().unwrap();
            println!("{} of {GROUPS} groups read: {} change files", group + 1, counted.0);
        }
    }
    let (files, misplaced, other_version) = *counted.lock().unwrap();
    (report.files_read, report.misplaced, report.other_version) = (files, misplaced, other_version);
    (report.fetch_s, report.decode_s) = *timed.lock().unwrap();
    let spill = spill.into_inner().unwrap();
    let shards: Vec<usize> = settings.shards.clone().collect();
    let written = each(&shards, settings.threads.min(WRITERS), |&shard| -> Result<u64> {
        let rows = current::merge(spill.take(shard)?, &HashSet::new(), false);
        let manifest = if rows.is_empty() {
            Manifest { shard, through_seq: through, merged_through: through, ..Manifest::default() }
        } else {
            let key = format!("{}{}/base-{through:012}.parquet", settings.prefix, current::shard_dir(shard));
            let file = current::encode(&rows)?;
            let bytes = file.len() as u64;
            store.put(&key, Bytes::from(file))?;
            Manifest {
                shard,
                through_seq: through,
                merged_through: through,
                base: Some(FileRef { key, rows: rows.len() as u64, bytes, through_seq: through }),
                ..Manifest::default()
            }
        };
        store.put(&compactor::manifest_key(&settings.prefix, shard), Bytes::from(serde_json::to_vec(&manifest)?))?;
        Ok(rows.len() as u64)
    })?;
    report.written = written.iter().sum();
    report.shards = shards.len();
    Ok(report)
}

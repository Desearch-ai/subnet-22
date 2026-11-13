//! OpenAlex: a paper page at doi.org for each English work with an abstract and at least 20 citations that our arXiv
//! and PMC pages do not already hold, read from the works Parquet snapshot.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use bulk_core::http::{Client, USER_AGENT};
use bulk_core::ranged::RangeClient;
use bulk_core::rows::utc_format;
use bulk_core::{upload_row, Batches, Counts, Row, RunArgs};
use bulk_pmc::bounded_map;
use bulk_pmc::order::openalex_files;

pub mod snapshot;
pub mod work;

pub use work::{class, clean_abstract, flags, page, rebuild, Work, MIN_ABSTRACT_CHARS};

pub const CONTENT_TYPE: &str = "application/json";
pub const SOURCE: &str = "openalex";

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// Works as the OpenAlex API returns them, one JSON object per line, instead of the snapshot.
    #[arg(long)]
    pub works: Option<PathBuf>,
    /// Start at this file of the snapshot's manifest, counting from 0.
    #[arg(long = "first-file", default_value_t = 0)]
    pub first_file: usize,
    /// Read at most this many of the manifest's files; all by default.
    #[arg(long, default_value_t = 0)]
    pub files: usize,
    /// The PMC versions our PMC pages hold, as `pmc-pmids --ids-out` writes them; works with one of their PMCIDs,
    /// PMIDs or DOIs stay out.
    #[arg(long)]
    pub held: Option<PathBuf>,
    #[arg(long = "min-cited", default_value_t = 20)]
    pub min_cited: i64,
    /// Close the run folder after the file that takes it to this many pages and go on in `<out>-02`, `<out>-03`, ...,
    /// run `<run>-02`, ...; 0 keeps one folder.
    #[arg(long = "chunk-pages", default_value_t = 0)]
    pub chunk_pages: u64,
    /// Snapshot files read at once.
    #[arg(long, default_value_t = 8)]
    pub workers: usize,
    /// Also write each work read with `--works`, one JSON object per line, as the pilot's quality script flags it.
    #[arg(long = "flags-out")]
    pub flags_out: Option<PathBuf>,
}

fn key(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// The PMCIDs (digits only), PMIDs and lowercased DOIs our PMC pages hold.
#[derive(Default)]
pub struct Held {
    pmcids: HashSet<u64>,
    pmids: HashSet<u64>,
    dois: HashSet<u64>,
}

fn pmcid_digits(id: &str) -> &str {
    let last = id.trim_end_matches('/').rsplit('/').next().unwrap_or(id);
    last.strip_prefix("PMC").unwrap_or(last)
}

impl Held {
    /// `pmc-pmids --ids-out`'s lines: PMCID, PMID and DOI, tab-separated, any of them empty.
    pub fn read(path: &Path) -> Result<Held> {
        let mut held = Held::default();
        for line in BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?).lines() {
            let line = line?;
            let mut fields = line.split('\t');
            let (pmcid, pmid, doi) = (fields.next().unwrap_or(""), fields.next().unwrap_or(""), fields.next().unwrap_or(""));
            if !pmcid.is_empty() {
                held.pmcids.insert(key(pmcid_digits(pmcid)));
            }
            if !pmid.is_empty() {
                held.pmids.insert(key(pmid));
            }
            if !doi.is_empty() {
                held.dois.insert(key(&doi.to_lowercase()));
            }
        }
        Ok(held)
    }

    pub fn holds(&self, work: &Work) -> bool {
        let pmid = work.pmid.trim_end_matches('/').rsplit('/').next().unwrap_or("");
        (!work.pmcid.is_empty() && self.pmcids.contains(&key(pmcid_digits(&work.pmcid))))
            || (!pmid.is_empty() && self.pmids.contains(&key(pmid)))
            || (!work.doi.is_empty() && self.dois.contains(&key(&work.bare_doi().to_lowercase())))
    }
}

/// What one work comes to: a page, or the counts key naming why it has none.
pub enum Outcome {
    Page { doi: u64, row: Box<Row> },
    Out(&'static str),
}

/// A work of the cut as a page, or why it stays out.
pub fn outcome(work: &Work, held: &Held, as_of_us: i64) -> Outcome {
    if work.retracted {
        return Outcome::Out("retracted");
    }
    if work.paratext {
        return Outcome::Out("paratext");
    }
    if work.bare_doi().is_empty() {
        return Outcome::Out("no_doi");
    }
    if work.arxiv() {
        return Outcome::Out("arxiv");
    }
    if held.holds(work) {
        return Outcome::Out("pmc");
    }
    let (text, gaps, dups) = rebuild(work.inverted.as_deref());
    match class(&text, gaps, dups, work.title.as_deref()) {
        "real" | "no_end_punct" => {}
        other => return Outcome::Out(other),
    }
    let cleaned = clean_abstract(&text);
    if cleaned.chars().count() < MIN_ABSTRACT_CHARS {
        return Outcome::Out("short");
    }
    if desearch::text::strip(work.title.as_deref().unwrap_or("")).is_empty() {
        return Outcome::Out("no_title");
    }
    let (url, page) = page(work, &cleaned, &utc_format(as_of_us, "%Y-%m-%d"));
    Outcome::Page { doi: key(&work.bare_doi().to_lowercase()), row: Box::new(upload_row(&url, page, as_of_us, CONTENT_TYPE)) }
}

/// Run folders closed after the file that fills them, as `--chunk-pages` asks.
struct Chunks<'a> {
    args: &'a RunArgs,
    every: u64,
    number: usize,
    batches: Option<Batches>,
    counts: Counts,
    pages: u64,
    total: u64,
    dois: HashSet<u64>,
    as_of_us: i64,
}

impl Chunks<'_> {
    fn folder(&self) -> (PathBuf, String) {
        let run = if self.args.run.is_empty() { utc_format(self.as_of_us, "%Y%m%d") } else { self.args.run.clone() };
        if self.every == 0 || self.number == 1 {
            return (self.args.out.clone(), if self.every == 0 { run } else { format!("{run}-01") });
        }
        (PathBuf::from(format!("{}-{:02}", self.args.out.display(), self.number)), format!("{run}-{:02}", self.number))
    }

    fn batches(&mut self) -> Result<&mut Batches> {
        if self.batches.is_none() {
            let (out, run) = self.folder();
            self.batches = Some(Batches::new(&out, SOURCE, &run, self.as_of_us)?);
        }
        Ok(self.batches.as_mut().expect("opened above"))
    }

    fn close(&mut self) -> Result<()> {
        let batches = match self.batches.take() {
            Some(batches) => batches,
            None => Batches::new(&self.folder().0, SOURCE, &self.folder().1, self.as_of_us)?,
        };
        let out = self.folder().0;
        let summary = batches.close(&self.counts)?;
        println!("{}", bulk_core::run::summary_line(&out, &summary, &self.counts));
        self.counts = Counts::default();
        self.pages = 0;
        self.number += 1;
        Ok(())
    }

    /// Adds the works' pages and counts; false once `--limit` pages are in.
    fn take(&mut self, works: &[Work], held: &Held, limit: u64) -> Result<bool> {
        for work in works {
            match outcome(work, held, self.as_of_us) {
                Outcome::Out(why) => self.counts.add(why, 1),
                Outcome::Page { doi, .. } if !self.dois.insert(doi) => self.counts.add("duplicate_doi", 1),
                Outcome::Page { row, .. } => {
                    self.batches()?.add(*row)?;
                    self.counts.add("pages", 1);
                    self.pages += 1;
                    self.total += 1;
                    if limit > 0 && self.total >= limit {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    /// Ends the file just added: closes the folder when it is full.
    fn file_done(&mut self) -> Result<()> {
        if self.every > 0 && self.pages >= self.every {
            self.close()?;
        }
        Ok(())
    }
}

/// `bulk openalex`.
pub fn main(args: Args) -> Result<()> {
    let as_of_us = bulk_core::rows::parse_as_of(&args.run.as_of)?;
    let held = match &args.held {
        Some(path) => Held::read(path)?,
        None => Held::default(),
    };
    let mut chunks = Chunks {
        args: &args.run,
        every: args.chunk_pages,
        number: 1,
        batches: None,
        counts: Counts::default(),
        pages: 0,
        total: 0,
        dois: HashSet::new(),
        as_of_us,
    };
    let limit = args.run.limit;
    if let Some(path) = &args.works {
        let mut flags_out = args.flags_out.as_ref().map(|p| File::create(p).map(BufWriter::new)).transpose()?;
        let mut works = Vec::new();
        for line in BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let work = Work::from_json(&serde_json::from_str(&line)?)?;
            if let Some(out) = flags_out.as_mut() {
                writeln!(out, "{}", flags(&work))?;
            }
            if work.language == "en" && work.cited >= args.min_cited {
                works.push(work);
            } else {
                chunks.counts.add("outside_cut", 1);
            }
        }
        if let Some(mut out) = flags_out {
            out.flush()?;
        }
        chunks.take(&works, &held, limit)?;
        return chunks.close();
    }
    let client = Client::new(USER_AGENT)?;
    let mut urls = openalex_files(&client)?.split_off(args.first_file);
    if args.files > 0 {
        urls.truncate(args.files);
    }
    let count = urls.len();
    let ranged = Arc::new(RangeClient::new(USER_AGENT)?);
    let min_cited = args.min_cited;
    let files = bounded_map(urls.into_iter(), args.workers, args.workers.max(1) * 2, move |url: String| {
        snapshot::file_works(&ranged, &url, min_cited).with_context(|| format!("reading {url}"))
    });
    for (n, works) in files.enumerate() {
        let going = chunks.take(&works?, &held, limit)?;
        if (n + 1) % 50 == 0 || n + 1 == count {
            eprintln!("{} of {count} files, {} pages", n + 1, chunks.total);
        }
        if !going {
            break;
        }
        chunks.file_done()?;
    }
    if chunks.batches.is_some() || chunks.number == 1 {
        chunks.close()?;
    }
    Ok(())
}

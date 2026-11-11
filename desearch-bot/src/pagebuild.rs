//! `crawler build` in Rust: fetch records into a bulk run folder, with the shared Python extraction in worker processes.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, FixedOffset, NaiveDateTime, Utc};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::pagecheck::{self, MAX_TEXT_CHARS, MIRROR_CHARS, SECTIONED};
use crate::pystr;
use crate::setrule::{self, SetRule};
use crate::uploads::{Batches, Page, Row};

pub const SOURCE: &str = "crawl";

pub struct Options {
    pub records: Vec<PathBuf>,
    pub out: PathBuf,
    pub run: String,
    pub as_of: DateTime<FixedOffset>,
    pub set: Option<PathBuf>,
    pub skip: HashSet<String>,
    pub drop_mirrors: bool,
    pub python: String,
    pub worker: PathBuf,
    pub subnet22: PathBuf,
    pub workers: usize,
    pub worker_memory_mb: u64,
    pub fetched_later_s: i64,
}

/// The record files of record folders (`pages-*.jsonl.zst`, sorted) or the files given.
pub fn record_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("pages-") && n.ends_with(".jsonl.zst")))
                .collect();
            files.sort();
            out.extend(files);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// URLs to leave out, the first word of each line.
pub fn read_urls(path: &Path) -> Result<HashSet<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(pystr::splitlines(&text).into_iter().filter_map(|l| pystr::split_ws(l).next().map(str::to_string)).collect())
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

/// Python's `datetime.fromisoformat` for the times records carry, as UTC; a time with no zone is read as UTC.
pub fn fetched_time(s: &str) -> Result<DateTime<Utc>> {
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.with_timezone(&Utc));
    }
    if let Ok(t) = DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%:z") {
        return Ok(t.with_timezone(&Utc));
    }
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").map(|t| t.and_utc()).map_err(|_| anyhow!("fetched_at {s:?} not understood"))
}

enum Job {
    Counted(&'static str),
    Extract(Value),
}

#[derive(Deserialize)]
struct Answer {
    page: Option<Page>,
    #[serde(default)]
    blocked: bool,
    #[serde(default)]
    which: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

enum Done {
    Counted(&'static str),
    Extracted(Value, Answer),
}

struct Worker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    fn start(o: &Options) -> Result<Self> {
        let mut cmd = Command::new(&o.python);
        cmd.arg(&o.worker).arg("--subnet22").arg(&o.subnet22).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        if o.worker_memory_mb > 0 {
            let bytes = o.worker_memory_mb << 20;
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(move || {
                    let limit = libc::rlimit { rlim_cur: bytes as libc::rlim_t, rlim_max: bytes as libc::rlim_t };
                    if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        let mut child = cmd.spawn().with_context(|| format!("starting {} {}", o.python, o.worker.display()))?;
        let stdin = BufWriter::new(child.stdin.take().expect("piped stdin"));
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Self { child, stdin, stdout })
    }

    fn ask(&mut self, request: &Value) -> Result<Answer> {
        serde_json::to_writer(&mut self.stdin, request)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            bail!("the extraction worker exited");
        }
        Ok(serde_json::from_str(&line)?)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn request(rec: &Value) -> Value {
    let url = rec["url"].as_str().unwrap_or_default();
    if truthy(rec.get("body_b64")) {
        return json!({
            "op": "doc",
            "url": url,
            "body_b64": rec["body_b64"],
            "document": rec.get("document").cloned().unwrap_or(Value::Null),
            "content_type": rec.get("content_type").cloned().unwrap_or(Value::String(String::new())),
        });
    }
    let html = rec["html"].as_str().unwrap_or_default();
    let whole = rec.get("source").and_then(Value::as_str) == Some("edgar");
    if pagecheck::is_fred_data(url) {
        let rewritten = pagecheck::fred_newest_first(html);
        if rewritten != html {
            return json!({"op": "html", "url": url, "html": rewritten, "orig": html, "whole": whole});
        }
    }
    json!({"op": "html", "url": url, "html": html, "whole": whole})
}

fn read_records(o: &Options, skip: &HashSet<String>, jobs: SyncSender<(u64, Job)>) -> Result<()> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut seq = 0u64;
    for file in record_files(&o.records)? {
        let raw = std::fs::File::open(&file).with_context(|| format!("opening {}", file.display()))?;
        let reader = BufReader::new(zstd::stream::read::Decoder::new(raw)?);
        for line in reader.lines() {
            let line = line?;
            if pystr::strip(&line).is_empty() {
                continue;
            }
            let rec: Value = serde_json::from_str(&line).with_context(|| format!("a record in {}", file.display()))?;
            let url = rec.get("url").and_then(Value::as_str).unwrap_or_default().to_string();
            let status_ok = rec.get("status").and_then(Value::as_f64) == Some(200.0) && !rec.get("status").is_some_and(Value::is_boolean);
            let job = if !status_ok || !(truthy(rec.get("html")) || truthy(rec.get("body_b64"))) {
                Job::Counted("not_a_page")
            } else if !seen.insert(url.clone()) {
                Job::Counted("repeat")
            } else if skip.contains(&url) {
                Job::Counted("held")
            } else {
                Job::Extract(rec)
            };
            if jobs.send((seq, job)).is_err() {
                return Ok(());
            }
            seq += 1;
        }
    }
    Ok(())
}

fn work(o: &Options, jobs: Arc<Mutex<Receiver<(u64, Job)>>>, done: SyncSender<(u64, Result<Done>)>) -> Result<()> {
    let mut worker = Worker::start(o)?;
    loop {
        let next = jobs.lock().map_err(|_| anyhow!("job queue poisoned"))?.recv();
        let Ok((seq, job)) = next else { return Ok(()) };
        let result = match job {
            Job::Counted(key) => Ok(Done::Counted(key)),
            Job::Extract(rec) => worker.ask(&request(&rec)).map(|a| Done::Extracted(rec, a)),
        };
        let failed = result.is_err();
        if done.send((seq, result)).is_err() || failed {
            return Ok(());
        }
    }
}

fn meta_str(meta: &Value, key: &str) -> String {
    match meta.get(key) {
        None => String::new(),
        v => pystr::py_str(v),
    }
}

/// A 10-K, 10-Q or 20-F as one page per item, each under the document URL with `?item=`, when its items can be found.
pub fn edgar_sections(rec: &Value, page: &Page) -> Vec<(String, Page)> {
    let meta = rec.get("meta").filter(|m| truthy(Some(m))).cloned().unwrap_or(json!({}));
    let form = meta.get("form").and_then(Value::as_str);
    if rec.get("source").and_then(Value::as_str) != Some("edgar") || !form.is_some_and(|f| SECTIONED.contains(&f)) {
        return Vec::new();
    }
    let url = rec["url"].as_str().unwrap_or_default();
    pagecheck::item_sections(&page.text)
        .into_iter()
        .map(|(item, body)| {
            let title = format!("{} {} {}, Item {item}", meta_str(&meta, "company"), pystr::py_str(meta.get("form")), meta_str(&meta, "filed"));
            let title = pystr::strip(&title).to_string();
            let text = pystr::cut(&format!("{title}\n{body}"), MAX_TEXT_CHARS).to_string();
            let headings = pagecheck::text_headings(&text);
            (format!("{url}?item={item}"), Page { title, text, headings, ..page.clone() })
        })
        .collect()
}

fn bump(counts: &mut IndexMap<String, u64>, key: impl Into<String>) {
    *counts.entry(key.into()).or_insert(0) += 1;
}

fn eligible(page: &Page) -> bool {
    setrule::page_eligible(&page.lang, crate::uploads::EXTRACTION_V2, &page.text)
}

pub fn build(o: &Options) -> Result<IndexMap<String, u64>> {
    let rule = o.set.as_deref().map(SetRule::load).transpose()?;
    let mut batches = Batches::new(&o.out, SOURCE, &o.run, o.as_of)?;
    let mut counts: IndexMap<String, u64> = IndexMap::new();
    let workers = o.workers.max(1);
    let (job_tx, job_rx) = sync_channel::<(u64, Job)>(workers * 4);
    let (done_tx, done_rx) = sync_channel::<(u64, Result<Done>)>(workers * 4);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let later = chrono::Duration::seconds(o.fetched_later_s);
    thread::scope(|scope| -> Result<()> {
        let (job_rx, done_rx) = (job_rx, done_rx);
        let reader = scope.spawn(|| read_records(o, &o.skip, job_tx));
        let mut pool = Vec::new();
        for _ in 0..workers {
            let (jobs, done) = (job_rx.clone(), done_tx.clone());
            pool.push(scope.spawn(move || work(o, jobs, done)));
        }
        drop((job_rx, done_tx));
        let mut waiting: BTreeMap<u64, Result<Done>> = BTreeMap::new();
        let mut next = 0u64;
        for (seq, result) in done_rx.iter() {
            waiting.insert(seq, result);
            while let Some(result) = waiting.remove(&next) {
                next += 1;
                match result? {
                    Done::Counted(key) => bump(&mut counts, key),
                    Done::Extracted(rec, answer) => take(o, rule.as_ref(), &rec, answer, later, &mut batches, &mut counts)?,
                }
            }
        }
        if !waiting.is_empty() {
            bail!("records {next} onward were never answered");
        }
        reader.join().map_err(|_| anyhow!("the record reader panicked"))??;
        for w in pool {
            w.join().map_err(|_| anyhow!("a worker thread panicked"))??;
        }
        Ok(())
    })?;
    batches.close(&counts)?;
    Ok(counts)
}

fn take(
    o: &Options,
    rule: Option<&SetRule>,
    rec: &Value,
    answer: Answer,
    later: chrono::Duration,
    batches: &mut Batches,
    counts: &mut IndexMap<String, u64>,
) -> Result<()> {
    let url = rec["url"].as_str().unwrap_or_default();
    if let Some(error) = answer.error {
        bail!("extraction of {url} failed: {error}");
    }
    let document = truthy(rec.get("body_b64"));
    let Some(mut page) = answer.page else {
        bump(counts, format!("document_{}", answer.which.unwrap_or_default()));
        return Ok(());
    };
    if !document && answer.blocked {
        bump(counts, "blocked");
        return Ok(());
    }
    if !document && pagecheck::is_wall(&page.title, &page.text) {
        bump(counts, "wall");
        return Ok(());
    }
    if o.drop_mirrors && pagecheck::is_mirror(&format!("{} {} {}", pystr::unquote(url), page.title, pystr::cut(&page.text, MIRROR_CHARS))) {
        bump(counts, "mirror");
        return Ok(());
    }
    if rule.is_some_and(|r| !r.admits(url, Some(&page.published), Some(&page.page_type))) {
        bump(counts, "out_of_set");
        return Ok(());
    }
    let fetched = fetched_time(rec.get("fetched_at").and_then(Value::as_str).unwrap_or_default())? + later;
    let content_type = rec.get("content_type").and_then(Value::as_str).filter(|c| !c.is_empty()).unwrap_or("text/html").to_string();
    let sections = edgar_sections(rec, &page);
    if !sections.is_empty() {
        for (item_url, item_page) in sections {
            if o.skip.contains(&item_url) {
                continue;
            }
            if eligible(&item_page) {
                batches.add(Row { url: item_url, fetched_at: fetched, content_type: content_type.clone(), page: item_page })?;
                bump(counts, "page_edgar_item");
            }
        }
        return Ok(());
    }
    if pystr::len(&page.text) > MAX_TEXT_CHARS {
        page.text = pystr::cut(&page.text, MAX_TEXT_CHARS).to_string();
    }
    if !eligible(&page) {
        bump(counts, "ineligible");
        return Ok(());
    }
    let source = match rec.get("source") {
        None => "site".to_string(),
        v => pystr::py_str(v),
    };
    batches.add(Row { url: url.to_string(), fetched_at: fetched, content_type, page })?;
    bump(counts, format!("page_{source}"));
    Ok(())
}

//! Unattended refresh runs as `python -m crawler refresh` makes them: fetch a source again and build a finished run folder, failing loudly on refusals.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::{Datelike, NaiveDate, Timelike, Utc};
use indexmap::{IndexMap, IndexSet};

use crate::pagebuild;
use crate::pystr;
use crate::sitecrawl::edgar::{self, Pace, FORMS};
use crate::sitecrawl::fetch::{Connect, Http};
use crate::sitecrawl::hosts::MIN_DELAY;
use crate::sitecrawl::lists::crawl_urls;
use crate::sitecrawl::robots::{Robots, Rules};
use crate::sitecrawl::sites::Ctx;
use crate::sitecrawl::store::{counts_json, Writer};
use crate::sitecrawl::Counts;

pub const SEC_AGENT: &str = "Desearch support@desearch.ai";
/// SEC answers 403, not 404, for a day with no index (weekends, holidays).
pub const NO_INDEX: [u16; 2] = [403, 404];
pub const MAX_ERROR_SHARE: f64 = 0.05;
pub const MIN_FRED_SHARE: f64 = 0.9;
pub const REFUSALS: [&str; 4] = ["refused_", "rate_limited", "robots_", "server_errors_"];
const INDEX_BYTES: usize = 64 << 20;

/// A run that must not be published: printed as `{"failed": ...}` with exit status 1.
#[derive(Debug)]
pub struct RefreshFailed(pub String);

impl std::fmt::Display for RefreshFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RefreshFailed {}

fn failed(message: String) -> anyhow::Error {
    RefreshFailed(message).into()
}

/// Python's `repr(text)` for the ASCII keys counts have.
pub fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `str(dict(counts))`.
pub fn counts_repr(counts: &Counts) -> String {
    let items: Vec<String> = counts.0.iter().map(|(k, v)| format!("{}: {v}", py_repr(k))).collect();
    format!("{{{}}}", items.join(", "))
}

/// Python's `str(Path(text))`: repeated and trailing slashes and `.` parts dropped.
pub fn py_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let root = if text.starts_with("//") && !text.starts_with("///") {
        "//"
    } else if text.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = text.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    match (root, parts.is_empty()) {
        ("", true) => ".".into(),
        _ => format!("{root}{}", parts.join("/")),
    }
}

/// Python's `Path(text).name`.
pub fn py_name(path: &Path) -> String {
    let text = py_path(path);
    if text == "." || text.ends_with('/') {
        return String::new();
    }
    text.rsplit('/').next().unwrap_or_default().to_string()
}

pub fn days(since: NaiveDate, until: NaiveDate) -> Vec<NaiveDate> {
    since.iter_days().take_while(|d| *d <= until).collect()
}

pub fn daily_index_url(d: NaiveDate) -> String {
    format!("https://www.sec.gov/Archives/edgar/daily-index/{}/QTR{}/form.{}.idx", d.year(), (d.month() - 1) / 3 + 1, d.format("%Y%m%d"))
}

/// CIKs with a chosen form in SEC's daily indexes for each day in the range.
pub async fn filed_ciks(connect: &Connect, user_agent: &str, rules: &Rules, since: NaiveDate, until: NaiveDate, per_second: f64) -> Result<(BTreeSet<i64>, Counts)> {
    let (mut counts, mut ciks, pace) = (Counts::default(), BTreeSet::new(), Pace::new(per_second)?);
    if let Some(why) = &rules.refused {
        counts.add(why, 1);
        return Ok((ciks, counts));
    }
    let http = edgar::sec_http(connect, user_agent)?;
    for d in days(since, until) {
        let url = daily_index_url(d);
        if !rules.allowed(&url) {
            counts.add("robots_disallowed", 1);
            continue;
        }
        pace.turn().await;
        let r = http.get(&url, INDEX_BYTES, false).await;
        if r.status == 200 {
            ciks.extend(edgar::index_ciks(&r.text(), &FORMS));
            counts.add("index_days", 1);
        } else if NO_INDEX.contains(&r.status) {
            counts.add(if d.weekday().num_days_from_monday() < 5 { "no_index_weekday" } else { "no_index_weekend" }, 1);
        } else {
            return Err(failed(format!("daily index {d}: {}", edgar::status_or_error(&r))));
        }
    }
    Ok((ciks, counts))
}

pub fn check_edgar(counts: &Counts) -> Result<()> {
    if counts.0.keys().any(|k| REFUSALS.iter().any(|p| k.starts_with(p))) || counts.get("submissions_403") != 0 || counts.get("document_403") != 0 {
        return Err(failed(format!("refused: {}", counts_repr(counts))));
    }
    let ok: i64 = counts.0.iter().filter(|(k, _)| k.starts_with("form_")).map(|(_, v)| v).sum();
    let bad: i64 = counts.0.iter().filter(|(k, _)| k.starts_with("document_") || k.starts_with("submissions_")).map(|(_, v)| v).sum();
    if bad != 0 && bad as f64 > MAX_ERROR_SHARE * (ok + bad) as f64 {
        return Err(failed(format!("too many fetch errors: {}", counts_repr(counts))));
    }
    Ok(())
}

pub fn check_fred(counts: &Counts, asked: usize) -> Result<()> {
    if counts.0.keys().any(|k| REFUSALS.iter().any(|p| k.starts_with(p))) {
        return Err(failed(format!("refused: {}", counts_repr(counts))));
    }
    if (counts.get("page") as f64) < MIN_FRED_SHARE * asked as f64 {
        return Err(failed(format!("only {} of {asked} pages: {}", counts.get("page"), counts_repr(counts))));
    }
    Ok(())
}

/// The records folder beside a run folder that must not exist yet, emptied.
pub fn fresh(out: &Path) -> Result<PathBuf> {
    if out.exists() {
        return Err(failed(format!("{} exists", py_path(out))));
    }
    let work = out.with_file_name(format!("{}.records", py_name(out)));
    let _ = std::fs::remove_dir_all(&work);
    Ok(work)
}

/// Where a refresh builds and with what: the set file, the subnet-22 checkout the extraction comes from, and its worker processes.
pub struct Target {
    pub out: PathBuf,
    pub run: String,
    pub set: PathBuf,
    pub subnet22: PathBuf,
    pub python: String,
    pub worker: Option<PathBuf>,
    pub workers: usize,
    pub worker_memory_mb: u64,
    pub connect: Connect,
}

fn build_run(records: &Path, t: &Target) -> Result<IndexMap<String, u64>> {
    let as_of = Utc::now().with_nanosecond(0).expect("zero nanoseconds").fixed_offset();
    pagebuild::build(&pagebuild::Options {
        records: vec![records.to_path_buf()],
        out: t.out.clone(),
        run: t.run.clone(),
        as_of,
        set: Some(t.set.clone()),
        skip: HashSet::new(),
        drop_mirrors: true,
        python: t.python.clone(),
        worker: t.worker.clone(),
        subnet22: t.subnet22.clone(),
        workers: t.workers,
        worker_memory_mb: t.worker_memory_mb,
        fetched_later_s: 0,
    })
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}

/// Filings filed from `since` through `until` into a finished run folder; the stdout line.
pub fn refresh_edgar(since: NaiveDate, until: NaiveDate, t: &Target, user_agent: &str, per_second: f64) -> Result<String> {
    let work = fresh(&t.out)?;
    let rt = runtime()?;
    let rules = rt.block_on(edgar::sec_rules(&t.connect, user_agent))?;
    let (ciks, mut counts) = rt.block_on(filed_ciks(&t.connect, user_agent, &rules, since, until, per_second))?;
    let writer = Arc::new(Mutex::new(Writer::new(&work)?));
    let ciks: Vec<i64> = ciks.into_iter().collect();
    let fetched = rt.block_on(edgar::fetch_filings(&t.connect, user_agent, rules, &ciks, since, Some(until), writer.clone(), per_second))?;
    counts.update(&fetched);
    writer.lock().unwrap().close()?;
    counts.0.insert("filers".into(), ciks.len() as i64);
    check_edgar(&counts)?;
    let built = build_run(&work, t)?;
    let _ = std::fs::remove_dir_all(&work);
    Ok(pystr::ascii_json(&format!("{{\"fetch\": {}, \"build\": {}}}", counts_json(&counts.0), crate::uploads::counts_json(&built))))
}

/// The FRED pages of `urls` into a finished run folder; the stdout line.
pub fn refresh_fred(urls: &[String], t: &Target) -> Result<String> {
    let work = fresh(&t.out)?;
    let keep: IndexSet<&String> = urls.iter().collect();
    let writer = Arc::new(Mutex::new(Writer::new(&work)?));
    let counts = runtime()?.block_on(async {
        let http = Arc::new(Http::new(&t.connect)?);
        let robots = Arc::new(Robots::new(http.clone(), "https"));
        let ctx = Arc::new(Ctx { http, robots, writer: writer.clone(), skip: Arc::new(HashSet::new()), set: None, min_delay: MIN_DELAY });
        crawl_urls(&ctx, keep.iter(), 1, "list").await
    })?;
    writer.lock().unwrap().close()?;
    check_fred(&counts, keep.len())?;
    let built = build_run(&work, t)?;
    let _ = std::fs::remove_dir_all(&work);
    Ok(pystr::ascii_json(&format!("{{\"asked\": {}, \"fetch\": {}, \"build\": {}}}", keep.len(), counts_json(&counts.0), crate::uploads::counts_json(&built))))
}

/// The line a failed run prints.
pub fn failed_line(e: &RefreshFailed) -> String {
    pystr::ascii_json(&format!("{{\"failed\": {}}}", serde_json::to_string(&e.0).expect("a string serializes")))
}

/// `date.fromisoformat`: `YYYY-MM-DD`, `YYYYMMDD` and the ISO week forms.
pub fn iso_date(text: &str) -> Result<NaiveDate, String> {
    let bad = || format!("invalid fromisoformat value: {}", py_repr(text));
    let b = text.as_bytes();
    if !text.is_ascii() || b.len() < 7 {
        return Err(bad());
    }
    let num = |s: &str| if s.bytes().all(|c| c.is_ascii_digit()) { s.parse::<u32>().ok() } else { None };
    let year = num(&text[..4]).ok_or_else(bad)? as i32;
    let rest = &text[4..];
    let dashed = rest.starts_with('-');
    let rest = if dashed { &rest[1..] } else { rest };
    let date = if let Some(week) = rest.strip_prefix('W') {
        let (w, d) = match (dashed, week.len()) {
            (true, 2) | (false, 2) => (num(week), Some(1)),
            (true, 4) if week.as_bytes()[2] == b'-' => (num(&week[..2]), num(&week[3..])),
            (false, 3) => (num(&week[..2]), num(&week[2..])),
            _ => (None, None),
        };
        let weekday = d.and_then(|d| chrono::Weekday::try_from(u8::try_from(d.checked_sub(1)?).ok()?).ok());
        w.zip(weekday).and_then(|(w, day)| NaiveDate::from_isoywd_opt(year, w, day))
    } else {
        let (m, d) = match (dashed, rest.len()) {
            (true, 5) if rest.as_bytes()[2] == b'-' => (num(&rest[..2]), num(&rest[3..])),
            (false, 4) => (num(&rest[..2]), num(&rest[2..])),
            _ => (None, None),
        };
        m.zip(d).and_then(|(m, d)| NaiveDate::from_ymd_opt(year, m, d))
    };
    date.filter(|_| year >= 1).ok_or_else(bad)
}

/// The request rate given to `--per-second`, read as Python's `float`.
pub fn per_second(text: &str) -> Result<f64, String> {
    crate::sitecrawl::robots::py_float(text).ok_or_else(|| format!("invalid float value: {}", py_repr(text)))
}

/// A file's stripped non-empty lines in order, duplicates dropped.
pub fn read_lines(path: &Path) -> Result<Vec<String>> {
    use anyhow::Context;
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let lines: IndexSet<String> = pystr::splitlines(&text).into_iter().map(|l| pystr::strip(l).to_string()).filter(|l| !l.is_empty()).collect();
    Ok(lines.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_reprs_paths_and_dates() {
        assert_eq!(py_repr("form_DEF 14A"), "'form_DEF 14A'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'\"\\\n\x01"), "'a\\'\"\\\\\\n\\x01'");
        let mut counts = Counts::default();
        assert_eq!(counts_repr(&counts), "{}");
        counts.add("index_days", 1);
        counts.add("form_10-K", 3);
        assert_eq!(counts_repr(&counts), "{'index_days': 1, 'form_10-K': 3}");
        for (given, text, name) in
            [("out/", "out", "out"), ("./a//b/.", "a/b", "b"), ("/x/y", "/x/y", "y"), ("//x", "//x", "x"), ("///x", "/x", "x"), ("", ".", ""), ("/", "/", "")]
        {
            assert_eq!((py_path(Path::new(given)), py_name(Path::new(given))), (text.to_string(), name.to_string()), "{given}");
        }
        let d = |y, m, d| NaiveDate::from_ymd_opt(y, m, d).unwrap();
        assert_eq!(iso_date("2026-11-10"), Ok(d(2026, 11, 10)));
        assert_eq!(iso_date("20261110"), Ok(d(2026, 11, 10)));
        assert_eq!(iso_date("2026-W46-2"), Ok(d(2026, 11, 10)));
        assert_eq!(iso_date("2026W462"), Ok(d(2026, 11, 10)));
        assert_eq!(iso_date("2026-W46"), Ok(d(2026, 11, 9)));
        for bad in ["2026-1-10", "2026-11-31", "2026/11/10", "20261110 ", "2026-W46-8", "0000-01-01", "２０２６-11-10"] {
            assert!(iso_date(bad).is_err(), "{bad}");
        }
        assert_eq!(days(d(2026, 11, 8), d(2026, 11, 10)).len(), 3);
        assert!(days(d(2026, 11, 10), d(2026, 11, 9)).is_empty());
        assert_eq!(daily_index_url(d(2026, 11, 10)), "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR4/form.20261110.idx");
    }
}

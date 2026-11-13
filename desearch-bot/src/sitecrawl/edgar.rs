//! SEC EDGAR filings as the Python crawler's `edgar` module takes them: filers from an index, their primary documents from their submissions, at SEC's fair-access pace.

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use chrono::NaiveDate;
use regex::Regex;
use serde_json::{Map, Value};
use tokio::sync::Semaphore;
use tokio::time::Instant;

use super::fetch::{Connect, Http, Response};
use super::robots::{Robots, Rules};
use super::store::{self, py_json, Writer};
use super::{Counts, Crash};
use crate::pystr;

pub const ARCHIVES: &str = "https://www.sec.gov/Archives/edgar/data";
pub const FORMS: [&str; 6] = ["10-K", "10-Q", "8-K", "S-1", "20-F", "DEF 14A"];
/// SEC asks for at most 10 requests a second; we stay under it.
pub const MAX_PER_SECOND: f64 = 8.0;
pub const MAX_DOCUMENT_BYTES: usize = 40 << 20;
pub const IN_FLIGHT: usize = 4;
pub const COMPANIES_AT_ONCE: usize = 8;
const ROBOTS_HOST: &str = "www.sec.gov";

// Python's `$` also matches before a final newline.
static XBRL_FRAGMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^R\d+\.htm\n?$").unwrap());
static DOCUMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\.(htm|html|txt)\n?$").unwrap());
static COLUMN_GAP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\x1c-\x1f]{2,}").unwrap());

pub fn submissions_url(cik: i64) -> String {
    format!("https://data.sec.gov/submissions/CIK{cik:010}.json")
}

#[derive(Clone, Debug, PartialEq)]
pub struct Filing {
    pub cik: i64,
    pub company: Value,
    pub form: String,
    pub filed: String,
    pub accession: String,
    pub document: String,
}

impl Filing {
    pub fn url(&self) -> String {
        format!("{ARCHIVES}/{}/{}/{}", self.cik, self.accession.replace('-', ""), self.document)
    }
}

/// CIKs that filed one of `forms` in a `form.idx`, full or daily.
pub fn index_ciks(form_idx: &str, forms: &[&str]) -> BTreeSet<i64> {
    let mut ciks = BTreeSet::new();
    for line in pystr::splitlines(form_idx) {
        let parts: Vec<&str> = COLUMN_GAP.split(pystr::strip(line)).collect();
        if parts.len() >= 5 && forms.contains(&parts[0]) && !parts[2].is_empty() && parts[2].bytes().all(|b| b.is_ascii_digit()) {
            if let Ok(cik) = parts[2].parse() {
                ciks.insert(cik);
            }
        }
    }
    ciks
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => false,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Bool(true) => true,
    }
}

/// `(value or {})` read as a dict: none for a falsy value, an AttributeError for anything else but an object.
fn object_or_empty(v: Option<&Value>) -> Result<Option<&Map<String, Value>>, Crash> {
    match v {
        Some(v) if truthy(v) => v.as_object().map(Some).ok_or(Crash("AttributeError")),
        _ => Ok(None),
    }
}

/// Python's `int(text)` for decimal text; a ValueError otherwise.
fn int_text(text: &str) -> Result<i64, Crash> {
    let t = pystr::strip(text);
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    let b = digits.as_bytes();
    let ok = !b.is_empty()
        && b[0].is_ascii_digit()
        && b[b.len() - 1].is_ascii_digit()
        && b.iter().all(|c| c.is_ascii_digit() || *c == b'_')
        && !digits.contains("__");
    if !ok {
        return Err(Crash("ValueError"));
    }
    let value: i64 = digits.replace('_', "").parse().map_err(|_| Crash("ValueError"))?;
    Ok(if t.starts_with('-') { -value } else { value })
}

/// `int(submissions.get("cik") or 0)`.
fn cik_of(v: Option<&Value>) -> Result<i64, Crash> {
    match v {
        Some(v) if truthy(v) => match v {
            Value::String(s) => int_text(s),
            Value::Bool(_) => Ok(1),
            Value::Number(n) => match (n.as_i64(), n.as_f64()) {
                (Some(i), _) => Ok(i),
                (None, Some(f)) if f.is_finite() => Ok(f.trunc() as i64),
                _ => Err(Crash("OverflowError")),
            },
            _ => Err(Crash("TypeError")),
        },
        _ => Ok(0),
    }
}

/// What `zip` walks for one column: a list's items, a string's characters, an object's keys.
fn column(recent: Option<&Map<String, Value>>, key: &str) -> Result<Vec<Value>, Crash> {
    match recent.and_then(|r| r.get(key)) {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(Value::String(s)) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        Some(Value::Object(m)) => Ok(m.keys().map(|k| Value::String(k.clone())).collect()),
        Some(_) => Err(Crash("TypeError")),
    }
}

/// A company's primary documents of `forms` filed from `since` through `until`, from its submissions JSON; a ValueError is an unreadable answer, any other error ends the run.
pub fn filings(submissions: &Value, since: NaiveDate, forms: &[&str], until: Option<NaiveDate>) -> Result<Vec<Filing>, Crash> {
    let top = submissions.as_object().ok_or(Crash("AttributeError"))?;
    let recent = match object_or_empty(top.get("filings"))? {
        Some(f) => object_or_empty(f.get("recent"))?,
        None => None,
    };
    let cik = cik_of(top.get("cik"))?;
    let company = top.get("name").filter(|v| truthy(v)).cloned().unwrap_or_else(|| Value::String(String::new()));
    let (since, until) = (since.to_string(), until.map(|u| u.to_string()));
    let (forms_of, filed_on) = (column(recent, "form")?, column(recent, "filingDate")?);
    let (accessions, documents) = (column(recent, "accessionNumber")?, column(recent, "primaryDocument")?);
    let mut out = Vec::new();
    for (((form, filed), accession), document) in forms_of.iter().zip(&filed_on).zip(&accessions).zip(&documents) {
        let form = match form {
            Value::Array(_) | Value::Object(_) => return Err(Crash("TypeError")),
            Value::String(f) if forms.contains(&f.as_str()) => f,
            _ => continue,
        };
        let filed = filed.as_str().ok_or(Crash("TypeError"))?;
        if filed < since.as_str() || until.as_deref().is_some_and(|u| filed > u) || !truthy(document) {
            continue;
        }
        let document = document.as_str().ok_or(Crash("AttributeError"))?;
        let name = document.rsplit('/').next().unwrap_or(document);
        if XBRL_FRAGMENT.is_match(name) || !DOCUMENT.is_match(name) {
            continue;
        }
        let accession = accession.as_str().ok_or(Crash("AttributeError"))?;
        out.push(Filing {
            cik,
            company: company.clone(),
            form: form.clone(),
            filed: filed.to_string(),
            accession: accession.to_string(),
            document: document.to_string(),
        });
    }
    Ok(out)
}

/// Starts at most `per_second` requests a second, whoever asks, in the order they ask.
pub struct Pace {
    gap: Duration,
    next_at: tokio::sync::Mutex<Option<Instant>>,
}

impl Pace {
    pub fn new(per_second: f64) -> Result<Self> {
        if per_second.is_nan() || per_second <= 0.0 {
            bail!("requests a second must be above 0, not {per_second}");
        }
        Ok(Pace { gap: Duration::from_secs_f64(1.0 / per_second), next_at: tokio::sync::Mutex::new(None) })
    }

    pub async fn turn(&self) {
        let mut next_at = self.next_at.lock().await;
        if let Some(at) = *next_at {
            if at > Instant::now() {
                tokio::time::sleep_until(at).await;
            }
        }
        let now = Instant::now();
        *next_at = Some(next_at.map_or(now, |at| at.max(now)) + self.gap);
    }
}

/// Python's `str(r.status or r.error)`.
pub fn status_or_error(r: &Response) -> String {
    if r.status != 0 {
        r.status.to_string()
    } else {
        r.error.clone().unwrap_or_else(|| "None".into())
    }
}

/// Python's `str(r.error or r.status)`.
fn error_or_status(r: &Response) -> String {
    r.error.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| r.status.to_string())
}

/// The run's EDGAR requests under one pace and at most `IN_FLIGHT` at once.
struct Lane {
    http: Arc<Http>,
    pace: Pace,
    gate: Semaphore,
    rules: Arc<Rules>,
    since: NaiveDate,
    until: Option<NaiveDate>,
    forms: Vec<String>,
    writer: Arc<Mutex<Writer>>,
    counts: Mutex<Counts>,
}

impl Lane {
    fn count(&self, key: &str) {
        self.counts.lock().unwrap().add(key, 1);
    }

    async fn get(&self, url: &str) -> Response {
        let _slot = self.gate.acquire().await.expect("the gate stays open");
        self.pace.turn().await;
        self.http.get(url, MAX_DOCUMENT_BYTES, false).await
    }

    async fn company(&self, cik: i64) -> Result<()> {
        let r = self.get(&submissions_url(cik)).await;
        if r.status != 200 {
            self.count(&format!("submissions_{}", status_or_error(&r)));
            return Ok(());
        }
        let body = r.body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&r.body);
        let forms: Vec<&str> = self.forms.iter().map(String::as_str).collect();
        let chosen = match serde_json::from_slice::<Value>(body).map(|v| filings(&v, self.since, &forms, self.until)) {
            Ok(Ok(chosen)) => chosen,
            Err(_) | Ok(Err(Crash("ValueError"))) => {
                self.count("submissions_unreadable");
                return Ok(());
            }
            Ok(Err(crash)) => bail!("{} reading the submissions of CIK {cik}", crash.0),
        };
        for f in chosen {
            let url = f.url();
            if !self.rules.allowed(&url) {
                self.count("skipped");
                continue;
            }
            let d = self.get(&url).await;
            if d.status != 200 || d.error.is_some() {
                self.count(&format!("document_{}", error_or_status(&d)));
                continue;
            }
            let meta = format!(
                "{{\"form\": {}, \"filed\": {}, \"company\": {}, \"cik\": {}}}",
                py_json(&Value::String(f.form.clone())),
                py_json(&Value::String(f.filed.clone())),
                py_json(&f.company),
                f.cik
            );
            let line = store::record_line_with(&d, &d.text(), "edgar", &meta);
            self.writer.lock().unwrap().add(&line)?;
            self.count(&format!("form_{}", f.form));
        }
        Ok(())
    }
}

pub fn sec_http(connect: &Connect, user_agent: &str) -> Result<Arc<Http>> {
    Ok(Arc::new(Http::with_headers(connect, &[("User-Agent", user_agent), ("Accept-Encoding", "gzip, deflate")])?))
}

/// SEC's robots.txt, read once for a whole run: the daily indexes and the filings both go by it.
pub async fn sec_rules(connect: &Connect, user_agent: &str) -> Result<Arc<Rules>> {
    Robots::new(sec_http(connect, user_agent)?, "https").rules(ROBOTS_HOST).await.map_err(|crash| anyhow!("{} reading {ROBOTS_HOST}'s robots.txt", crash.0))
}

/// Each company's submissions, then its chosen filings' primary documents, under SEC's pace; companies open `COMPANIES_AT_ONCE` at a time in CIK order.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_filings(
    connect: &Connect,
    user_agent: &str,
    rules: Arc<Rules>,
    ciks: &[i64],
    since: NaiveDate,
    until: Option<NaiveDate>,
    forms: &[String],
    writer: Arc<Mutex<Writer>>,
    per_second: f64,
) -> Result<Counts> {
    let http = sec_http(connect, user_agent)?;
    if let Some(why) = &rules.refused {
        let mut counts = Counts::default();
        counts.add(why, 1);
        return Ok(counts);
    }
    let lane = Arc::new(Lane {
        http,
        pace: Pace::new(per_second)?,
        gate: Semaphore::new(IN_FLIGHT),
        rules,
        since,
        until,
        forms: forms.to_vec(),
        writer,
        counts: Mutex::new(Counts::default()),
    });
    let open = Arc::new(Semaphore::new(COMPANIES_AT_ONCE));
    let mut tasks = tokio::task::JoinSet::new();
    for &cik in ciks {
        let turn = open.clone().acquire_owned().await?;
        while let Some(done) = tasks.try_join_next() {
            done??;
        }
        let lane = lane.clone();
        tasks.spawn(async move {
            let _turn = turn;
            lane.company(cik).await
        });
    }
    while let Some(done) = tasks.join_next().await {
        done??;
    }
    let counts = lane.counts.lock().unwrap().clone();
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn pace_spaces_starts_and_lets_an_idle_lane_start_at_once() {
        let pace = Pace::new(8.0).unwrap();
        let start = Instant::now();
        let mut starts = Vec::new();
        for _ in 0..17 {
            pace.turn().await;
            starts.push((Instant::now() - start).as_secs_f64());
        }
        assert!((starts[16] - 2.0).abs() < 1e-9);
        tokio::time::sleep(Duration::from_secs(5)).await;
        let before = Instant::now();
        pace.turn().await;
        assert_eq!(Instant::now(), before);
        assert!(Pace::new(0.0).is_err() && Pace::new(f64::NAN).is_err());
    }
}

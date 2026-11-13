//! One OpenAlex work: its abstract rebuilt from the inverted index, the checks that keep it out, and its page.

use std::sync::LazyLock;

use anyhow::{anyhow, Result};
use bulk_core::{authors_line, paper_page, Line, Page, PaperMeta};
use desearch::text::{strip, words};
use regex::Regex;
use serde_json::{json, Map, Value};

pub const SITE: &str = "OpenAlex";
pub const DOI_URL: &str = "https://doi.org/";
pub const MOST_AUTHORS: usize = 10;
pub const MIN_ABSTRACT_CHARS: usize = 200;
/// A copyright line is cut when it starts this far into the abstract and leaves at most this many characters.
const COPYRIGHT_FROM: f64 = 0.6;
const COPYRIGHT_MAX_CHARS: usize = 200;

static BOILER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)(no abstract|abstract (is )?not available|an abstract is not available|abstract unavailable|not available for this|",
        r"this article has been retracted|preview of|click here|full[- ]text (is )?available|for the full text|",
        r"this content is only available|please (refer|see|log ?in)|sign in|javascript|cookies?|first page of|",
        r"access (to )?this (article|content)|purchase|subscribe|download (the )?pdf|^summary$|^abstract$|",
        r"publisher summary|this chapter|this book)"
    ))
    .expect("BOILER")
});
static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)</?(jats:|p>|sup|sub|italic|bold|sec|title)|&lt;|&gt;|&amp;").expect("MARKUP"));
static LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:Abstract|ABSTRACT|Summary|SUMMARY)(?:\s+and\s+(?:Summary|Abstract))?(?:\s*[:.\-–—]\s*|\s+)?").expect("LABEL"));
static COPYRIGHT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)©|\(c\)\s*(?:19|20)\d\d|\bcopyright\b").expect("COPYRIGHT"));

/// The fields of a work that its checks and page read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Work {
    pub id: String,
    /// As OpenAlex writes it, `https://doi.org/<doi>`, or empty.
    pub doi: String,
    pub title: Option<String>,
    pub year: Option<i64>,
    /// `YYYY-MM-DD`, or empty.
    pub date: String,
    pub language: String,
    pub kind: String,
    pub cited: i64,
    pub retracted: bool,
    pub paratext: bool,
    /// Each word with its positions, in the index's order; None when the work has no abstract.
    pub inverted: Option<Vec<(String, Vec<i64>)>>,
    pub pmid: String,
    pub pmcid: String,
    /// Each author's name with their institutions.
    pub authors: Vec<(String, Vec<String>)>,
    /// Every location's landing page and PDF URL, empty where missing.
    pub location_urls: Vec<String>,
    pub landing: String,
    pub source: String,
    pub oa_license: String,
    pub volume: String,
    pub issue: String,
    pub first_page: String,
    pub last_page: String,
}

fn text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(v, |v, key| v.get(key)).filter(|v| !v.is_null())
}

/// An `abstract_inverted_index` object as a list of words and positions.
pub fn inverted_of(index: &Map<String, Value>) -> Result<Vec<(String, Vec<i64>)>> {
    index
        .iter()
        .map(|(word, positions)| {
            let positions = positions.as_array().ok_or_else(|| anyhow!("an inverted index entry without positions"))?;
            let positions = positions.iter().map(|p| p.as_i64().ok_or_else(|| anyhow!("a position that is not an integer"))).collect::<Result<_>>()?;
            Ok((word.clone(), positions))
        })
        .collect()
}

impl Work {
    /// A work as the OpenAlex API returns it.
    pub fn from_json(w: &Value) -> Result<Work> {
        let inverted = match w.get("abstract_inverted_index") {
            Some(Value::Object(index)) => Some(inverted_of(index)?),
            Some(Value::String(json)) => {
                Some(inverted_of(serde_json::from_str::<Value>(json)?.as_object().ok_or_else(|| anyhow!("an inverted index that is not an object"))?)?)
            }
            _ => None,
        };
        let list = |key: &str| w.get(key).and_then(Value::as_array).cloned().unwrap_or_default();
        let authors = list("authorships")
            .iter()
            .map(|a| {
                let institutions =
                    a.get("institutions").and_then(Value::as_array).map(|i| i.iter().map(|i| text(i.get("display_name"))).collect()).unwrap_or_default();
                (text(at(a, &["author", "display_name"])), institutions)
            })
            .collect();
        let location_urls = list("locations").iter().flat_map(|l| [text(l.get("landing_page_url")), text(l.get("pdf_url"))]).collect();
        Ok(Work {
            id: text(w.get("id")),
            doi: text(w.get("doi")),
            title: w.get("title").and_then(Value::as_str).map(str::to_string),
            year: w.get("publication_year").and_then(Value::as_i64),
            date: text(w.get("publication_date")),
            language: text(w.get("language")),
            kind: text(w.get("type")),
            cited: w.get("cited_by_count").and_then(Value::as_i64).unwrap_or(0),
            retracted: w.get("is_retracted").and_then(Value::as_bool).unwrap_or(false),
            paratext: w.get("is_paratext").and_then(Value::as_bool).unwrap_or(false),
            inverted,
            pmid: text(at(w, &["ids", "pmid"])),
            pmcid: text(at(w, &["ids", "pmcid"])),
            authors,
            location_urls,
            landing: text(at(w, &["primary_location", "landing_page_url"])),
            source: text(at(w, &["primary_location", "source", "display_name"])),
            oa_license: text(at(w, &["best_oa_location", "license"])),
            volume: text(at(w, &["biblio", "volume"])),
            issue: text(at(w, &["biblio", "issue"])),
            first_page: text(at(w, &["biblio", "first_page"])),
            last_page: text(at(w, &["biblio", "last_page"])),
        })
    }

    /// The DOI without its `https://doi.org/` prefix.
    pub fn bare_doi(&self) -> &str {
        self.doi.strip_prefix(DOI_URL).unwrap_or(&self.doi)
    }

    fn hosts(&self) -> String {
        self.location_urls.join(" ").to_lowercase()
    }

    /// Whether arXiv holds a version: a location on arxiv.org, or an arXiv DOI.
    pub fn arxiv(&self) -> bool {
        self.hosts().contains("arxiv.org") || self.bare_doi().to_lowercase().starts_with("10.48550")
    }

    /// The PMCID OpenAlex gives, `loc` for a PMC location without one, or empty.
    pub fn pmc_flag(&self) -> String {
        if !self.pmcid.is_empty() {
            return self.pmcid.clone();
        }
        let hosts = self.hosts();
        if hosts.contains("ncbi.nlm.nih.gov/pmc") || hosts.contains("pmc.ncbi") {
            "loc".into()
        } else {
            String::new()
        }
    }
}

/// The abstract's words in position order, one word per position (the smallest where two share one), with the count
/// of positions missing below the highest and of positions given twice.
pub fn rebuild(inverted: Option<&[(String, Vec<i64>)]>) -> (String, i64, i64) {
    let mut positions: Vec<(i64, &str)> = inverted.unwrap_or_default().iter().flat_map(|(w, ps)| ps.iter().map(move |&p| (p, w.as_str()))).collect();
    positions.sort_unstable();
    let mut words: Vec<&str> = Vec::with_capacity(positions.len());
    let mut last = None;
    for &(p, w) in &positions {
        if last != Some(p) {
            words.push(w);
            last = Some(p);
        }
    }
    let seen = words.len() as i64;
    let top = positions.last().map_or(-1, |&(p, _)| p);
    (words.join(" "), (top + 1) - seen, positions.len() as i64 - seen)
}

/// The abstract's class: `real`, `no_end_punct`, `short`, `boilerplate`, `garbled` or `empty`.
pub fn class(text: &str, gaps: i64, dups: i64, title: Option<&str>) -> &'static str {
    if strip(text).is_empty() {
        return "empty";
    }
    let n = words(text).count() as f64;
    if gaps as f64 > 2f64.max(0.02 * n) || dups > 0 {
        return "garbled";
    }
    let chars = text.chars().count();
    if chars < 600 && BOILER.is_match(text) {
        return "boilerplate";
    }
    if strip(text).to_lowercase() == strip(title.unwrap_or("")).to_lowercase() {
        return "boilerplate";
    }
    if chars < 200 {
        return "short";
    }
    let end = text.trim_end_matches(desearch::text::is_space);
    if end.ends_with("...") || end.ends_with('…') || !end.chars().last().is_some_and(|c| ".?!)\"'”’]".contains(c)) {
        return "no_end_punct";
    }
    "real"
}

pub fn has_markup(text: &str) -> bool {
    MARKUP.is_match(text)
}

/// The abstract without a leading "Abstract" or "Summary" label and without a closing copyright line.
pub fn clean_abstract(text: &str) -> String {
    let mut text = text;
    if let Some(label) = LABEL.find(text) {
        let rest = &text[label.end()..];
        if rest.chars().next().is_some_and(|c| !c.is_lowercase()) {
            text = rest;
        }
    }
    let chars = text.chars().count();
    let mut end = text.len();
    for found in COPYRIGHT.find_iter(text) {
        let from = text[..found.start()].chars().count();
        if from as f64 >= COPYRIGHT_FROM * chars as f64 && chars - from <= COPYRIGHT_MAX_CHARS {
            end = found.start();
            if text[..end].ends_with('(') {
                end -= 1;
            }
            break;
        }
    }
    strip(&text[..end]).to_string()
}

/// The pilot's per-work flags, as its quality script writes them, less the sample's stratum and the language guess.
pub fn flags(work: &Work) -> Value {
    let (text, gaps, dups) = rebuild(work.inverted.as_deref());
    json!({
        "id": work.id,
        "doi": work.bare_doi(),
        "title": work.title,
        "year": work.year,
        "type": work.kind,
        "cited": work.cited,
        "class": class(&text, gaps, dups, work.title.as_deref()),
        "markup": has_markup(&text),
        "chars": text.chars().count(),
        "gaps": gaps,
        "dups": dups,
        "arxiv": work.arxiv(),
        "pmcid": work.pmc_flag(),
        "pmid": work.pmid,
        "landing": work.landing,
        "source": work.source,
        "oa_license": work.oa_license,
        "abstract": text,
    })
}

/// Where the work was published: venue, year, volume, issue and pages, then the DOI.
pub fn source_line(work: &Work) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !work.source.is_empty() {
        parts.push(work.source.clone());
    }
    if let Some(year) = work.year {
        parts.push(year.to_string());
    }
    if !work.volume.is_empty() {
        parts.push(format!("volume {}", work.volume));
    }
    if !work.issue.is_empty() {
        parts.push(format!("issue {}", work.issue));
    }
    match (work.first_page.as_str(), work.last_page.as_str()) {
        ("", _) => {}
        (first, last) if last.is_empty() || last == first => parts.push(format!("page {first}")),
        (first, last) => parts.push(format!("pages {first}–{last}")),
    }
    let doi = format!("doi: {}", work.bare_doi());
    if parts.is_empty() {
        doi
    } else {
        format!("{}. {doi}", parts.join(", "))
    }
}

/// The work's page: title, authors with their institutions, where it was published, its citation count and the
/// cleaned abstract.
pub fn page(work: &Work, abstract_: &str, as_of_date: &str) -> (String, Page) {
    let url = format!("{DOI_URL}{}", work.bare_doi());
    let names: Vec<String> = work.authors.iter().map(|(name, _)| name.clone()).collect();
    let credited: Vec<String> = work
        .authors
        .iter()
        .map(|(name, institutions)| {
            let mut kept: Vec<&str> = Vec::new();
            for i in institutions.iter().map(|i| strip(i)).filter(|i| !i.is_empty()) {
                if !kept.contains(&i) {
                    kept.push(i);
                }
            }
            if kept.is_empty() {
                name.clone()
            } else {
                format!("{name} ({})", kept.join("; "))
            }
        })
        .collect();
    let authors = authors_line(&credited, MOST_AUTHORS);
    let cited = format!("Cited by {} works in OpenAlex as of {as_of_date}.", work.cited);
    let lines: Vec<Line> = [authors.as_str(), &source_line(work), &cited, abstract_].into_iter().filter(|t| !t.is_empty()).map(Line::new).collect();
    let meta = PaperMeta {
        lang: "en".into(),
        url: url.clone(),
        published: work.date.clone(),
        modified: String::new(),
        author: authors_line(&names, MOST_AUTHORS),
        site_name: SITE.into(),
        tags: Vec::new(),
    };
    (url, paper_page(strip(work.title.as_deref().unwrap_or("")), lines, meta))
}

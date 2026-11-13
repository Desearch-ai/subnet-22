//! `readme_page` and `run_github`: each subnet's GitHub repository with its README, rendered by the shared extraction.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use bulk_core::py::{self, when_truthy};
use bulk_core::worker::{map_in_order, Worker};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta};
use desearch::text::head;
use serde_json::Value;

use crate::{remove_named, DOCS};

pub const GITHUB_SOURCE: &str = "subnet-github";
pub const SITE: &str = "GitHub";
pub const CONTENT_TYPE: &str = "text/html";
const BODY_REQUEST: &[u8] = b"r";

enum Step<'a> {
    Skip(&'static str),
    Page { repo: &'a Value, readme: PathBuf },
}

fn topics(repo: &Value) -> Result<Vec<String>> {
    match when_truthy(py::get(repo, "topics")?) {
        Some(v) => py::list(v, "topics")?.iter().map(|t| py::str_of(t, "a topic").map(str::to_string)).collect(),
        None => Ok(Vec::new()),
    }
}

/// `readme_page`: a repository's page as GitHub shows it: name, description, topics, homepage, then the README's lines.
pub fn readme_page(repo: &Value, readme: Option<Vec<Line>>) -> Result<(String, Page)> {
    let url = py::str_of(py::item(repo, "html_url")?, "html_url")?.to_string();
    let name = py::format(py::item(repo, "full_name")?)?;
    let shown = |key: &str| -> Result<Option<String>> { when_truthy(py::get(repo, key)?).map(py::format).transpose() };
    let description = shown("description")?;
    let title = match &description {
        Some(about) => format!("GitHub - {name}: {about}"),
        None => format!("GitHub - {name}"),
    };
    let mut lines = vec![Line::new(format!("{name} is a GitHub repository."))];
    if let Some(about) = &description {
        lines.push(Line::new(format!("About: {about}")));
    }
    if let Some(homepage) = shown("homepage")? {
        lines.push(Line::new(format!("Homepage: {homepage}")));
    }
    let topics = topics(repo)?;
    if !topics.is_empty() {
        lines.push(Line::new(format!("Topics: {}", topics.join(", "))));
    }
    if let Some(language) = shown("language")? {
        lines.push(Line::new(format!("Language: {language}")));
    }
    let license = py::dict_or_empty(py::get(repo, "license")?, "license")?;
    if let Some(license) = when_truthy(license.get("name")) {
        lines.push(Line::new(format!("License: {}", py::format(license)?)));
    }
    if let Some(readme) = readme {
        lines.push(Line::new("## README"));
        lines.extend(readme);
    }
    let modified = match when_truthy(py::get(repo, "pushed_at")?) {
        Some(v) => head(py::str_of(v, "pushed_at")?, 10).to_string(),
        None => String::new(),
    };
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), modified, site_name: SITE.into(), tags: topics, ..PaperMeta::default() };
    Ok((url, typed_page(DOCS, &title, lines, meta)))
}

fn wire_line(line: &Value) -> Result<Line> {
    let flag = |i: usize| line.get(i).and_then(Value::as_bool).ok_or_else(|| anyhow!("a rendered line {line} without flag {i}"));
    let text = line.get(0).and_then(Value::as_str).ok_or_else(|| anyhow!("a rendered line {line} without text"))?;
    Ok(Line { text: text.to_string(), linked: flag(1)?, verbatim: flag(2)?, row: flag(3)? })
}

/// The README's lines as `render` gives them for its HTML in a body, or None when lxml finds no body.
fn rendered(worker: &mut Worker, html: &str) -> Result<Option<Vec<Line>>> {
    let answer = worker.ask_json(&[BODY_REQUEST, html.as_bytes()])?;
    match answer.get("lines") {
        Some(Value::Null) => Ok(None),
        Some(Value::Array(lines)) => lines.iter().map(wire_line).collect::<Result<Vec<_>>>().map(Some),
        other => bail!("a render answer with lines {other:?}"),
    }
}

/// `run_github`: a page per repository with a README, each repository once, then the pages the fetch named removed.
pub fn run_github(folder: &Path, batches: &mut Batches, as_of_us: i64, workers: usize, spawn: &(dyn Fn() -> Result<Worker> + Sync)) -> Result<Counts> {
    let found = py::load_json(&folder.join("github.json"))?;
    let mut steps = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for record in py::list(&found, "github.json")? {
        let Some(repo) = when_truthy(py::get(record, "repo")?) else {
            steps.push(Step::Skip("no_repo"));
            continue;
        };
        let url = py::str_of(py::item(repo, "html_url")?, "html_url")?;
        if seen.contains(url) {
            steps.push(Step::Skip("duplicate"));
            continue;
        }
        let Some(readme) = when_truthy(py::get(record, "readme_file")?) else {
            steps.push(Step::Skip("no_readme"));
            continue;
        };
        seen.insert(url.to_string());
        steps.push(Step::Page { repo, readme: folder.join(py::str_of(readme, "readme_file")?) });
    }
    let readmes: Vec<&Path> = steps.iter().filter_map(|step| if let Step::Page { readme, .. } = step { Some(readme.as_path()) } else { None }).collect();
    let mut lines = map_in_order(&readmes, workers, spawn, &|worker, path| rendered(worker, &py::read_text(path)?))?.into_iter();
    let mut counts = Counts::default();
    for step in steps {
        match step {
            Step::Skip(why) => counts.add(why, 1),
            Step::Page { repo, .. } => {
                let (url, page) = readme_page(repo, lines.next().ok_or_else(|| anyhow!("a README without its lines"))?)?;
                batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
                counts.add("pages", 1);
            }
        }
    }
    remove_named(folder, batches, &seen, &mut counts)?;
    Ok(counts)
}

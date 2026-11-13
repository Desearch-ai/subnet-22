//! Wikimedia Enterprise snapshots: each article's Parsoid HTML framed like the live page here, then read by the shared
//! v2 extraction in `python -m bulk.page_worker` processes.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::{channel, sync_channel, Receiver};
use std::sync::{Condvar, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::worker::{page_from_json, PythonArgs, Worker, PAGE_WORKER};
use bulk_core::{pyjson, upload_row, Batches, Counts, Page, RunArgs};
use rayon::prelude::*;
use serde_json::{json, Map, Value};

pub mod html;
pub mod read;

pub const SOURCE: &str = "wikipedia";
pub const CONTENT_TYPE: &str = "text/html";
pub const PROJECTS: &[(&str, &str)] = &[
    ("enwiki", "Wikipedia"),
    ("enwikisource", "Wikisource"),
    ("enwikibooks", "Wikibooks"),
    ("enwikiquote", "Wikiquote"),
    ("enwikivoyage", "Wikivoyage"),
    ("enwikinews", "Wikinews"),
];
pub const CATEGORY_LINK: &str = "mw:PageProp/Category";
pub const HIDDEN_MARK: &str = "mw:PageProp/hiddencat";
pub const DISAMBIGUATION_MARK: &str = "mw:PageProp/disambiguation";
pub const OG_IMAGE_MAX: f64 = 1200.0;
pub const THUMB_STEPS: [i64; 9] = [20, 40, 60, 120, 250, 330, 500, 960, 1280];
/// Articles framed longer than this go to the worker's frame, so the extraction's 8M-character cut falls where it does there.
pub const RUST_FRAME_MAX_BYTES: usize = 4_000_000;
/// A run reads `--workers` × 32 articles a window, which fixes the order its counts first appear in.
const WINDOW_PER_WORKER: usize = 32;
/// Results kept ahead of the next one written, so a slow article cannot let the rest pile up without end.
const AHEAD_PER_WORKER: u64 = 256;
const WIKIPEDIA_REQUEST: &[u8] = b"w";
const EXTRACT_REQUEST: &[u8] = b"e";

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
    /// Hidden category names, one per line, from `hidden-categories`.
    #[arg(long)]
    pub hidden: PathBuf,
    /// Sets the window, and so the order of the counts; the CPU count by default.
    #[arg(long)]
    pub workers: Option<usize>,
    /// Processes that extract articles; `--workers` by default.
    #[arg(long = "page-workers")]
    pub page_workers: Option<usize>,
    #[command(flatten)]
    pub python: PythonArgs,
}

#[derive(clap::Args, Clone, Debug)]
pub struct HiddenArgs {
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `(value.get(key) or {})`.
fn dict<'a>(value: &'a Value, key: &str) -> Result<Option<&'a Map<String, Value>>> {
    let found = value.get(key);
    if !truthy(found) {
        return Ok(None);
    }
    found.and_then(Value::as_object).map(Some).ok_or_else(|| anyhow!("{key} is not an object"))
}

fn field<'a>(map: Option<&'a Map<String, Value>>, key: &str) -> Option<&'a Value> {
    map.and_then(|m| m.get(key))
}

/// `str(value)` as an f-string writes it.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => n.as_u64().map_or_else(|| pyjson::float_repr(n.as_f64().unwrap_or(f64::NAN)), |u| u.to_string()),
        },
        other => pyjson::dumps(other, None),
    }
}

fn number(value: Option<&Value>) -> Result<f64> {
    match value {
        v if !truthy(v) => Ok(0.0),
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| anyhow!("{n} is not a number")),
        Some(other) => bail!("image size {other} is not a number"),
        None => Ok(0.0),
    }
}

fn text<'a>(article: &'a Value, key: &str) -> Result<&'a str> {
    article.get(key).and_then(Value::as_str).ok_or_else(|| anyhow!("article {key} missing or not a string"))
}

/// `og_image`: the page image as MediaWiki's og:image shows it.
pub fn og_image(image: Option<&Map<String, Value>>, host: &str) -> Result<String> {
    let url = match field(image, "content_url") {
        Some(v) if truthy(Some(v)) => py_str(v),
        _ => String::new(),
    };
    let (width, height) = (number(field(image, "width"))?, number(field(image, "height"))?);
    if !url.contains("/wikipedia/") || width == 0.0 || height == 0.0 {
        return Ok(url);
    }
    let tracking = format!("?utm_source={host}&utm_campaign=index&utm_content=");
    let vector = url.to_lowercase().ends_with(".svg");
    if !vector && width <= OG_IMAGE_MAX && height <= OG_IMAGE_MAX {
        return Ok(format!("{url}{tracking}thumbnail_unscaled"));
    }
    let scaled = (width * (OG_IMAGE_MAX / width).min(OG_IMAGE_MAX / height)).round_ties_even() as i64;
    let step = THUMB_STEPS.iter().copied().find(|&s| s >= scaled).unwrap_or(THUMB_STEPS[THUMB_STEPS.len() - 1]);
    let after = url.split_once("/wikipedia/").map(|(_, rest)| rest).unwrap_or_default();
    let (project, file_path) = after.split_once('/').ok_or_else(|| anyhow!("image url {url} has no file path"))?;
    let name = file_path.rsplit_once('/').map(|(_, name)| name).ok_or_else(|| anyhow!("image url {url} has no file name"))?;
    Ok(format!("https://thumb.wikimedia.org/wikipedia/{project}/thumb/{file_path}/{step}px-{name}{}{tracking}thumbnail", if vector { ".png" } else { "" }))
}

fn category_name(href: &str) -> String {
    let after = href.split_once("Category:").map_or("", |(_, rest)| rest);
    let before_hash = after.split_once('#').map_or(after, |(name, _)| name);
    html::unquote(before_hash).replace('_', " ")
}

/// The page as the live site frames this revision's Parsoid HTML, and whether it is a disambiguation page;
/// None when the body cannot be cut out as lxml would serialise it, so the worker's `page_html` must do.
pub fn frame(article: &Value, hidden: &HashSet<String>) -> Result<Option<(String, bool)>> {
    let body_html = field(dict(article, "article_body")?, "html").and_then(Value::as_str).unwrap_or("");
    let Some(inner) = html::body_inner(body_html) else {
        return Ok(None);
    };
    let tags = html::start_tags(body_html);
    let inner = html::as_lxml_writes(inner, &tags, inner.as_ptr() as usize - body_html.as_ptr() as usize);
    let disambiguation = tags.iter().any(|t| t.name == "meta" && t.get("property") == Some(DISAMBIGUATION_MARK));
    let mut categories: Vec<String> = Vec::new();
    for tag in &tags {
        let href = tag.get("href").unwrap_or("");
        if tag.name == "link" && tag.get("rel") == Some(CATEGORY_LINK) && href.contains("Category:") {
            let name = category_name(href);
            if !categories.contains(&name) {
                categories.push(name);
            }
        }
    }
    let name = article.get("name").ok_or_else(|| anyhow!("article has no name"))?;
    let url = text(article, "url")?;
    let project = field(dict(article, "is_part_of")?, "identifier").and_then(Value::as_str).unwrap_or("");
    let site = PROJECTS.iter().find(|(id, _)| *id == project).map_or("Wikipedia", |(_, site)| site);
    let entity = field(dict(article, "main_entity")?, "identifier");
    let host = url.split('/').nth(2).ok_or_else(|| anyhow!("article url {url} has no host"))?;
    let image = dict(article, "image")?;
    let original = match field(image, "content_url") {
        Some(v) if truthy(Some(v)) => Some(format!("{}?utm_source={host}&utm_campaign=index&utm_content=original", py_str(v))),
        _ => None,
    };
    let shown_image = og_image(image, host)?;
    let mut ld = Map::new();
    ld.insert("@context".into(), json!("https://schema.org"));
    ld.insert("@type".into(), json!("Article"));
    ld.insert("name".into(), name.clone());
    ld.insert("url".into(), json!(url));
    if truthy(entity) {
        let wikidata = json!(format!("http://www.wikidata.org/entity/{}", py_str(entity.expect("truthy"))));
        ld.insert("sameAs".into(), wikidata.clone());
        ld.insert("mainEntity".into(), wikidata);
    }
    ld.insert("author".into(), json!({"@type": "Organization", "name": "Contributors to Wikimedia projects"}));
    ld.insert(
        "publisher".into(),
        json!({
            "@type": "Organization",
            "name": "Wikimedia Foundation, Inc.",
            "logo": {"@type": "ImageObject", "url": "https://www.wikimedia.org/static/images/wmf-hor-googpub.png"},
        }),
    );
    for (key, value) in [
        ("datePublished", article.get("date_created").cloned()),
        ("dateModified", article.get("date_modified").cloned()),
        ("image", original.map(Value::String)),
        ("headline", article.get("description").cloned()),
    ] {
        if truthy(value.as_ref()) {
            ld.insert(key.into(), value.expect("truthy"));
        }
    }
    let noindex = truthy(field(dict(article, "version")?, "noindex"));
    let robots = if noindex { "noindex,nofollow,max-image-preview:standard" } else { "max-image-preview:standard" };
    let title = html::escape(&format!("{} - {site}", py_str(name)));
    let og = if shown_image.is_empty() { String::new() } else { format!(r#"<meta property="og:image" content="{}">"#, html::escape(&shown_image)) };
    let head = [
        r#"<meta charset="UTF-8">"#.to_string(),
        format!("<title>{title}</title>"),
        r#"<meta name="generator" content="MediaWiki">"#.to_string(),
        format!(r#"<meta name="robots" content="{robots}">"#),
        og,
        format!(r#"<meta property="og:title" content="{title}">"#),
        r#"<meta property="og:type" content="website">"#.to_string(),
        format!(r#"<link rel="canonical" href="{}">"#, html::escape(url)),
        format!(r#"<script type="application/ld+json">{}</script>"#, pyjson::dumps_unicode(&Value::Object(ld))),
    ]
    .concat();
    let shown: String = categories
        .iter()
        .filter(|c| !hidden.contains(c.as_str()))
        .map(|c| {
            let c = html::escape(c);
            format!(r#"<li><a href="/wiki/Category:{c}">{c}</a></li>"#)
        })
        .collect();
    let lang = match dict(article, "in_language")? {
        Some(map) => map.get("identifier").map_or_else(|| "en".to_string(), py_str),
        None => "en".to_string(),
    };
    let page = format!(
        concat!(
            r#"<!DOCTYPE html><html lang="{lang}" dir="ltr"><head>{head}</head><body>"#,
            r#"<div id="mw-content-text" class="mw-body-content"><div class="mw-content-ltr mw-parser-output" lang="{lang}" dir="ltr">{inner}</div></div>"#,
            r#"<div id="catlinks"><div id="mw-normal-catlinks" class="mw-normal-catlinks"><ul>{shown}</ul></div></div></body></html>"#,
        ),
        lang = lang,
        head = head,
        inner = inner,
        shown = shown
    );
    Ok(Some((page, disambiguation)))
}

/// Why an article has no page, or its page.
#[derive(Debug)]
pub enum Outcome {
    Disambiguation,
    NoText,
    Page(Box<Page>),
}

impl Outcome {
    pub fn why(&self) -> &'static str {
        match self {
            Outcome::Disambiguation => "disambiguation",
            Outcome::NoText => "no_text",
            Outcome::Page(_) => "pages",
        }
    }
}

/// One article's page through a worker: framed here when it can be, else framed and read in Python.
pub fn read_article(worker: &mut Worker, article: &Value, line: &[u8], hidden: &HashSet<String>, python_frames: &mut u64) -> Result<Outcome> {
    let url = text(article, "url")?;
    let framed = frame(article, hidden)?.filter(|(page, _)| page.len() <= RUST_FRAME_MAX_BYTES);
    let answer = match &framed {
        Some((_, true)) => return Ok(Outcome::Disambiguation),
        Some((page, false)) => {
            let size = u32::try_from(url.len()).context("url over 4 GiB")?.to_be_bytes();
            worker.ask(&[EXTRACT_REQUEST, &size, url.as_bytes(), page.as_bytes()])?
        }
        None => {
            *python_frames += 1;
            worker.ask(&[WIKIPEDIA_REQUEST, line])?
        }
    };
    let answer: Value = serde_json::from_slice(&answer).context("a worker answer")?;
    if let Some(error) = answer.get("error") {
        bail!("{PAGE_WORKER} failed on {url}: {error}\n{}", answer.get("detail").and_then(Value::as_str).unwrap_or(""));
    }
    if answer.get("disambiguation").is_some_and(|d| d == &Value::Bool(true)) {
        return Ok(Outcome::Disambiguation);
    }
    let page = page_from_json(answer.get("page").ok_or_else(|| anyhow!("a worker answer without a page"))?)?;
    if desearch::text::strip(&page.text).is_empty() {
        return Ok(Outcome::NoText);
    }
    Ok(Outcome::Page(Box::new(page)))
}

enum Event {
    Skip(&'static str),
    Job,
}

struct Job {
    seq: u64,
    bump: i64,
    article: Value,
    line: Vec<u8>,
}

struct Done {
    url: String,
    bump: i64,
    outcome: Outcome,
}

#[derive(Default)]
struct Shared {
    done: BTreeMap<u64, Done>,
    next: u64,
    error: Option<anyhow::Error>,
    stop: bool,
    python_frames: u64,
}

struct State {
    shared: Mutex<Shared>,
    changed: Condvar,
}

impl State {
    fn fail(&self, error: anyhow::Error) {
        let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.error.get_or_insert(error);
        shared.stop = true;
        self.changed.notify_all();
    }
}

/// The filter a run applies as it reads: namespace, deletions, older copies and empty bodies, each job with
/// its revision bump, in reading order.
struct Filter {
    revisions: HashMap<i64, (i64, i64)>,
}

impl Filter {
    fn take(&mut self, article: &Value) -> Result<std::result::Result<i64, &'static str>> {
        let main = match field(dict(article, "namespace")?, "identifier") {
            Some(Value::Number(n)) => n.as_f64() == Some(0.0),
            Some(Value::Bool(b)) => !b,
            _ => false,
        };
        if !main {
            return Ok(Err("other_namespace"));
        }
        if field(dict(article, "event")?, "type").and_then(Value::as_str) == Some("delete") {
            return Ok(Err("deleted"));
        }
        let revision = match field(dict(article, "version")?, "identifier") {
            v if !truthy(v) => 0,
            Some(v) => v.as_i64().ok_or_else(|| anyhow!("revision {v} is not an integer"))?,
            None => 0,
        };
        let id = article.get("identifier").ok_or_else(|| anyhow!("article has no identifier"))?;
        let id = id.as_i64().ok_or_else(|| anyhow!("identifier {id} is not an integer"))?;
        let seen = self.revisions.get(&id).copied();
        if let Some((seen_revision, _)) = seen {
            if revision <= seen_revision {
                return Ok(Err("older_copy"));
            }
        }
        let bump = seen.map_or(0, |(_, bump)| bump + 1);
        self.revisions.insert(id, (revision, bump));
        if !truthy(field(dict(article, "article_body")?, "html")) {
            return Ok(Err("no_html"));
        }
        Ok(Ok(bump))
    }
}

/// The pages of `paths` in reading order, counted in windows of `window_workers` × 32 articles, read by `workers` page
/// workers.
#[allow(clippy::too_many_arguments)]
pub fn run(
    paths: &[PathBuf],
    batches: &mut Batches,
    as_of_us: i64,
    hidden: &HashSet<String>,
    limit: u64,
    window_workers: usize,
    workers: usize,
    spawn: &(dyn Fn() -> Result<Worker> + Sync),
) -> Result<Counts> {
    let workers = workers.max(1);
    let window = window_workers.max(1) * WINDOW_PER_WORKER;
    let ahead = workers as u64 * AHEAD_PER_WORKER;
    let state = State { shared: Mutex::new(Shared::default()), changed: Condvar::new() };
    let (lines_tx, lines_rx) = sync_channel::<Vec<u8>>(1024);
    let (jobs_tx, jobs_rx) = sync_channel::<Job>(workers * 4);
    let jobs_rx = Mutex::new(jobs_rx);
    let (events_tx, events_rx) = channel::<Event>();
    let mut counts = Counts::default();
    let result = std::thread::scope(|scope| -> Result<()> {
        let state = &state;
        scope.spawn(move || {
            let read = || -> Result<()> {
                for path in paths {
                    read::each_line(path, |line| Ok(lines_tx.send(line).is_ok()))?;
                }
                Ok(())
            };
            if let Err(error) = read() {
                state.fail(error);
            }
        });
        scope.spawn(move || {
            if let Err(error) = filter(lines_rx, &jobs_tx, &events_tx, state) {
                state.fail(error);
            }
        });
        for _ in 0..workers {
            let jobs_rx = &jobs_rx;
            scope.spawn(move || {
                if let Err(error) = extract(jobs_rx, hidden, ahead, spawn, state) {
                    state.fail(error);
                }
            });
        }
        let collected = collect(&events_rx, batches, as_of_us, limit, window, state, &mut counts);
        let mut shared = state.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.stop = true;
        state.changed.notify_all();
        drop(shared);
        drop(events_rx);
        // A filter blocked on a full queue sees the stop only once a job is taken off it.
        while jobs_rx.lock().unwrap_or_else(|e| e.into_inner()).recv().is_ok() {}
        collected
    });
    let mut shared = state.shared.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some(error) = shared.error.take() {
        return Err(error);
    }
    result?;
    if shared.python_frames > 0 {
        eprintln!("{} articles framed in Python", shared.python_frames);
    }
    Ok(counts)
}

fn filter(lines: Receiver<Vec<u8>>, jobs: &std::sync::mpsc::SyncSender<Job>, events: &std::sync::mpsc::Sender<Event>, state: &State) -> Result<()> {
    let mut rules = Filter { revisions: HashMap::new() };
    let mut seq = 0u64;
    while let Ok(first) = lines.recv() {
        let mut batch = vec![first];
        while batch.len() < 64 {
            match lines.try_recv() {
                Ok(line) => batch.push(line),
                Err(_) => break,
            }
        }
        let parsed: Vec<Result<Value>> = batch.par_iter().map(|line| read::parse(line)).collect();
        for (line, article) in batch.into_iter().zip(parsed) {
            if state.shared.lock().unwrap_or_else(|e| e.into_inner()).stop {
                return Ok(());
            }
            let article = article?;
            match rules.take(&article)? {
                Err(why) => {
                    if events.send(Event::Skip(why)).is_err() {
                        return Ok(());
                    }
                }
                Ok(bump) => {
                    if events.send(Event::Job).is_err() || jobs.send(Job { seq, bump, article, line }).is_err() {
                        return Ok(());
                    }
                    seq += 1;
                }
            }
        }
    }
    Ok(())
}

fn extract(jobs: &Mutex<Receiver<Job>>, hidden: &HashSet<String>, ahead: u64, spawn: &(dyn Fn() -> Result<Worker> + Sync), state: &State) -> Result<()> {
    let mut worker = spawn()?;
    let mut python_frames = 0;
    loop {
        let job = match jobs.lock().unwrap_or_else(|e| e.into_inner()).recv() {
            Ok(job) => job,
            Err(_) => break,
        };
        {
            let mut shared = state.shared.lock().unwrap_or_else(|e| e.into_inner());
            while !shared.stop && job.seq >= shared.next + ahead {
                shared = state.changed.wait(shared).unwrap_or_else(|e| e.into_inner());
            }
            if shared.stop {
                break;
            }
        }
        let url = text(&job.article, "url")?.to_string();
        let outcome = read_article(&mut worker, &job.article, &job.line, hidden, &mut python_frames)?;
        let mut shared = state.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.done.insert(job.seq, Done { url, bump: job.bump, outcome });
        state.changed.notify_all();
    }
    state.shared.lock().unwrap_or_else(|e| e.into_inner()).python_frames += python_frames;
    Ok(())
}

/// Applies the counts and rows window by window: each window's skips as it was read, then its results.
fn collect(events: &Receiver<Event>, batches: &mut Batches, as_of_us: i64, limit: u64, window: usize, state: &State, counts: &mut Counts) -> Result<()> {
    let mut next = 0u64;
    loop {
        let mut jobs = 0;
        let mut ended = false;
        while jobs < window {
            match events.recv() {
                Ok(Event::Skip(why)) => counts.add(why, 1),
                Ok(Event::Job) => jobs += 1,
                Err(_) => {
                    ended = true;
                    break;
                }
            }
        }
        for _ in 0..jobs {
            let done = {
                let mut shared = state.shared.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(error) = shared.error.take() {
                        return Err(error);
                    }
                    if let Some(done) = shared.done.remove(&next) {
                        next += 1;
                        shared.next = next;
                        state.changed.notify_all();
                        break done;
                    }
                    if shared.stop {
                        bail!("the page workers stopped");
                    }
                    shared = state.changed.wait(shared).unwrap_or_else(|e| e.into_inner());
                }
            };
            counts.add(done.outcome.why(), 1);
            if let Outcome::Page(page) = done.outcome {
                batches.add(upload_row(&done.url, *page, as_of_us + done.bump * 1_000_000, CONTENT_TYPE))?;
            }
            if limit > 0 && counts.get("pages") >= limit {
                return Ok(());
            }
        }
        if ended {
            if let Some(error) = state.shared.lock().unwrap_or_else(|e| e.into_inner()).error.take() {
                return Err(error);
            }
            return Ok(());
        }
    }
}

/// `hidden_categories`: names of the hidden categories in category-namespace snapshots, in file order.
pub fn hidden_categories(paths: &[PathBuf], out: &mut impl Write) -> Result<()> {
    for path in paths {
        read::each_line(path, |line| {
            let article = read::parse(&line)?;
            let html = field(dict(&article, "article_body")?, "html").and_then(Value::as_str).unwrap_or("");
            if html.contains(HIDDEN_MARK) {
                let name = text(&article, "name")?;
                writeln!(out, "{}", name.split_once(':').map_or(name, |(_, rest)| rest))?;
            }
            Ok(true)
        })?;
    }
    Ok(())
}

pub fn hidden_main(args: HiddenArgs) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    hidden_categories(&args.files, &mut out)?;
    out.flush()?;
    Ok(())
}

pub fn main(args: Args) -> Result<()> {
    let hidden_text = std::fs::read_to_string(&args.hidden).with_context(|| format!("reading {}", args.hidden.display()))?;
    let hidden: HashSet<String> = bulk_core::text::splitlines(&hidden_text).into_iter().map(str::to_string).collect();
    let workers = args.workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let python_path = args.python.python_path(PAGE_WORKER);
    let worker_args: Vec<OsString> = vec!["--hidden".into(), args.hidden.clone().into_os_string()];
    let spawn = || Worker::spawn(&args.python.python, &python_path, PAGE_WORKER, &worker_args);
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args.files, &mut batches, as_of_us, &hidden, args.run.limit, workers, args.page_workers.unwrap_or(workers), &spawn)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

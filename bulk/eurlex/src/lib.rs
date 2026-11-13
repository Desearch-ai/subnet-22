//! EUR-Lex (reuse with acknowledgement, Decision 2011/833/EU): English acts in force and CJEU judgments listed through Cellar's SPARQL, their texts fetched from Cellar and read by the shared extraction.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::http::USER_AGENT;
use bulk_core::worker::{map_in_order, page_from_json, PythonArgs, Worker, PAGE_WORKER};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SOURCE: &str = "eurlex";
pub const SITE: &str = "EUR-Lex";
pub const LAW: &str = "article";
pub const SPARQL: &str = "https://publications.europa.eu/webapi/rdf/sparql";
pub const CELLAR: &str = "http://publications.europa.eu/resource/celex";
pub const WORKS: &str = "works.jsonl";
const DOCS: &str = "docs";
const MISSING: &str = "missing.txt";
const PAGE_SIZE: usize = 10_000;
const EXTRACT: &[u8] = b"e";
const ACT: &str = "act";
const JUDGMENT: &str = "judgment";
const ACCEPT: &str = "application/xhtml+xml, text/html;q=0.9";
const FAILING_IN_ROW: usize = 10;
const ATTEMPTS: u64 = 8;
static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"__\[unreferenced:[^\]]*\]__").expect("a valid pattern"));

const PREFIX: &str = "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n";
const ENGLISH: &str = "?expr cdm:expression_belongs_to_work ?work ; cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ; cdm:expression_title ?title .";
const ACTS: &str = "SELECT ?celex ?eli ?date ?title WHERE { ?work cdm:resource_legal_id_celex ?celex ; cdm:resource_legal_id_sector \"3\"^^xsd:string ; cdm:resource_legal_in-force \"true\"^^xsd:boolean . OPTIONAL { ?work cdm:resource_legal_eli ?eli } OPTIONAL { ?work cdm:work_date_document ?date } ";
const JUDGMENTS: &str = "SELECT ?celex ?ecli ?date ?title WHERE { ?work cdm:work_has_resource-type <http://publications.europa.eu/resource/authority/resource-type/JUDG> ; cdm:resource_legal_id_celex ?celex . OPTIONAL { ?work cdm:case-law_ecli ?ecli } OPTIONAL { ?work cdm:work_date_document ?date } ";
/// CELEX years listed, one query each: Cellar refuses to sort or skip past 10,000 rows, so a listing goes year by year.
const FIRST_YEAR: u32 = 1950;
const LAST_YEAR: u32 = 2030;

/// One act or judgment as the listing gives it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub celex: String,
    pub kind: String,
    #[serde(default)]
    pub eli: String,
    #[serde(default)]
    pub ecli: String,
    #[serde(default)]
    pub date: String,
    pub title: String,
}

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// The folder for the listing and the fetched texts; a fetch picks up where an earlier one stopped.
    #[arg(long)]
    pub out: PathBuf,
    /// Cellar requests a second.
    #[arg(long, default_value_t = 2.0)]
    pub per_second: f64,
    /// List the works again even when a listing is there.
    #[arg(long)]
    pub relist: bool,
    /// Stop after this many texts fetched, for trials.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// Requests waiting on Cellar at once, all under --per-second.
    #[arg(long, default_value_t = 8)]
    pub parallel: usize,
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    #[command(flatten)]
    pub python: PythonArgs,
    #[arg(long, default_value_t = 4)]
    pub workers: usize,
    /// The folder `eurlex-fetch` filled.
    pub folder: PathBuf,
}

fn binding(row: &Value, key: &str) -> String {
    row.get(key).and_then(|b| b.get("value")).and_then(Value::as_str).unwrap_or("").to_string()
}

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(120)).build().context("building the HTTP client")
}

fn sparql(client: &reqwest::blocking::Client, query: &str) -> Result<Vec<Value>> {
    for attempt in 0..5u32 {
        let response = client.post(SPARQL).header("Accept", "application/sparql-results+json").form(&[("query", query)]).send();
        match response {
            Ok(r) if r.status().is_success() => {
                let body: Value = serde_json::from_slice(&r.bytes()?)?;
                return Ok(body.pointer("/results/bindings").and_then(Value::as_array).cloned().unwrap_or_default());
            }
            Ok(r) if attempt == 4 => bail!("SPARQL answered {}", r.status()),
            Err(e) if attempt == 4 => return Err(e.into()),
            _ => std::thread::sleep(Duration::from_secs(10 * (attempt as u64 + 1))),
        }
    }
    unreachable!("the last attempt returns")
}

/// How many works a listing's query matches, so a cut-off listing can be told from a short year.
fn count(client: &reqwest::blocking::Client, select: &str, filter: &str) -> Result<usize> {
    let wheres = select.split_once("WHERE").map(|(_, w)| w).context("a listing query without WHERE")?;
    let query = format!("{PREFIX}SELECT (COUNT(DISTINCT ?celex) AS ?n) WHERE{wheres}{filter}");
    let rows = sparql(client, &query)?;
    rows.first().map(|r| binding(r, "n")).and_then(|n| n.parse().ok()).context("a count query without a count")
}

/// Every English act in force and every English CJEU judgment, once each by CELEX, in CELEX order.
pub fn list(client: &reqwest::blocking::Client) -> Result<Vec<Work>> {
    let mut works = Vec::new();
    let mut seen = HashSet::new();
    for (kind, sector, select) in [(ACT, '3', ACTS), (JUDGMENT, '6', JUDGMENTS)] {
        for year in FIRST_YEAR..=LAST_YEAR {
            let filter = format!("FILTER(STRSTARTS(STR(?celex), \"{sector}{year}\")) {ENGLISH} }}");
            let query = format!("{PREFIX}{select}{filter} LIMIT {PAGE_SIZE}");
            let rows = sparql(client, &query)?;
            if rows.len() >= PAGE_SIZE {
                bail!("CELEX {sector}{year} lists {} rows, the most one query returns", rows.len());
            }
            let counted = count(client, select, &filter)?;
            let listed: HashSet<String> = rows.iter().map(|r| binding(r, "celex")).filter(|c| !c.is_empty()).collect();
            if listed.len() < counted {
                bail!("CELEX {sector}{year}: the listing came back with {} of {counted} works", listed.len());
            }
            for row in &rows {
                let celex = binding(row, "celex");
                if !celex.is_empty() && seen.insert(celex.clone()) {
                    works.push(Work { celex, kind: kind.into(), eli: binding(row, "eli"), ecli: binding(row, "ecli"), date: binding(row, "date"), title: binding(row, "title") });
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    works.sort_by(|a, b| a.celex.cmp(&b.celex));
    Ok(works)
}

pub fn read_works(folder: &Path) -> Result<Vec<Work>> {
    let file = File::open(folder.join(WORKS)).with_context(|| format!("reading {}", folder.join(WORKS).display()))?;
    BufReader::new(file).lines().map(|line| Ok(serde_json::from_str(&line?)?)).collect()
}

fn doc_path(folder: &Path, celex: &str) -> PathBuf {
    folder.join(DOCS).join(format!("{}.html", celex.replace(['/', '(', ')'], "_")))
}

/// The listing, then each work's English text from Cellar under the pace, skipping texts already saved or known missing.
pub fn fetch_main(args: FetchArgs) -> Result<()> {
    std::fs::create_dir_all(args.out.join(DOCS))?;
    let client = client()?;
    if args.relist || !args.out.join(WORKS).exists() {
        let works = list(&client)?;
        let mut file = File::create(args.out.join(format!("{WORKS}.part")))?;
        for w in &works {
            writeln!(file, "{}", serde_json::to_string(w)?)?;
        }
        std::fs::rename(args.out.join(format!("{WORKS}.part")), args.out.join(WORKS))?;
        println!("{}", serde_json::json!({"listed": works.len()}));
    }
    let works = read_works(&args.out)?;
    let week_ago = (std::time::SystemTime::now() - Duration::from_secs(7 * 86_400)).duration_since(std::time::UNIX_EPOCH)?.as_secs();
    let missing: HashSet<String> = std::fs::read_to_string(args.out.join(MISSING))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (celex, at) = l.split_once('\t').unwrap_or((l, "0"));
            (at.parse::<u64>().unwrap_or(0) > week_ago).then(|| celex.to_string())
        })
        .collect();
    let mut missing_file = OpenOptions::new().create(true).append(true).open(args.out.join(MISSING))?;
    let gap = Duration::from_secs_f64(1.0 / args.per_second.max(0.01));
    let todo: Vec<&Work> = works.iter().filter(|w| !missing.contains(&w.celex) && !doc_path(&args.out, &w.celex).exists()).collect();
    let (next, pace) = (AtomicUsize::new(0), Mutex::new(Instant::now() - gap));
    let (mut fetched, mut absent, mut failed, mut failing_in_row) = (0usize, 0usize, 0usize, 0usize);
    std::thread::scope(|scope| -> Result<()> {
        let (sender, texts) = mpsc::channel();
        for _ in 0..args.parallel.max(1) {
            let (sender, client, todo, next, pace) = (sender.clone(), &client, &todo, &next, &pace);
            scope.spawn(move || {
                while let Some(w) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                    {
                        let mut last = pace.lock().expect("pace lock");
                        std::thread::sleep(gap.saturating_sub(last.elapsed()));
                        *last = Instant::now();
                    }
                    if sender.send((*w, get_text(client, &w.celex))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for (w, text) in texts {
            let path = doc_path(&args.out, &w.celex);
            failing_in_row = if matches!(text, Text::Failing(_)) { failing_in_row + 1 } else { 0 };
            match text {
                Text::Found(body) => {
                    std::fs::write(path.with_extension("part"), &body)?;
                    std::fs::rename(path.with_extension("part"), &path)?;
                    fetched += 1;
                }
                Text::Failing(why) if failing_in_row >= FAILING_IN_ROW => bail!("{FAILING_IN_ROW} works in a row failed, the last: {why}"),
                text => {
                    if let Text::Failing(why) = text {
                        eprintln!("{why}; tried again after a week");
                        failed += 1;
                    } else {
                        absent += 1;
                    }
                    writeln!(missing_file, "{}\t{}", w.celex, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs())?;
                }
            }
            if (fetched + absent + failed) % 1000 == 0 {
                println!("{}", serde_json::json!({"fetched": fetched, "missing": absent, "failed": failed}));
            }
            if args.limit > 0 && fetched >= args.limit {
                break;
            }
        }
        Ok(())
    })?;
    println!("{}", serde_json::json!({"fetched": fetched, "missing": absent, "failed": failed, "works": works.len()}));
    Ok(())
}

enum Text {
    Found(Vec<u8>),
    Absent,
    Failing(String),
}

/// A work's English XHTML or HTML; a work whose XHTML keeps failing is asked for its HTML alone, and stays failing without it.
fn get_text(client: &reqwest::blocking::Client, celex: &str) -> Text {
    match get_as(client, celex, ACCEPT) {
        Text::Failing(why) => match get_as(client, celex, "text/html") {
            Text::Found(body) => Text::Found(body),
            _ => Text::Failing(why),
        },
        text => text,
    }
}

fn get_as(client: &reqwest::blocking::Client, celex: &str, accept: &str) -> Text {
    let url = format!("{CELLAR}/{celex}");
    let mut why = String::new();
    for attempt in 0..ATTEMPTS {
        match client.get(&url).header("Accept", accept).header("Accept-Language", "eng").send() {
            Ok(r) if r.status().is_success() => match r.bytes() {
                Ok(body) => return Text::Found(body.to_vec()),
                Err(e) => why = format!("GET {url}: {e}"),
            },
            Ok(r) if matches!(r.status().as_u16(), 300 | 404 | 406 | 410) => return Text::Absent,
            Ok(r) => why = format!("Cellar answered {} for {celex} as {accept}", r.status()),
            Err(e) => why = format!("GET {url}: {e}"),
        }
        if attempt + 1 < ATTEMPTS {
            std::thread::sleep(Duration::from_secs(2 * (attempt + 1)));
        }
    }
    Text::Failing(why)
}

/// The page address: an act's ELI page, else its CELEX page; a judgment's CELEX page.
pub fn url_of(w: &Work) -> String {
    if w.kind == ACT {
        if let Some(path) = w.eli.strip_prefix("http://data.europa.eu/eli/") {
            return format!("https://eur-lex.europa.eu/eli/{}/eng", path.trim_end_matches('/'));
        }
        return format!("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{}", w.celex);
    }
    format!("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{}", w.celex)
}

/// A judgment's Cellar title, `<form> of <date>.#<parties>.#<subject>.#<case>.`, as a title and a subject line.
pub fn judgment_title(raw: &str) -> (String, String) {
    let raw = PLACEHOLDER.replace_all(raw, "Judgment");
    let parts: Vec<&str> = raw.split('#').map(|p| p.trim().trim_end_matches('.')).collect();
    match parts.as_slice() {
        [form, parties, subject, case, ..] => (format!("{parties} ({case}), {form}"), subject.to_string()),
        [form, parties, case] => (format!("{parties} ({case}), {form}"), String::new()),
        _ => (parts.join(" "), String::new()),
    }
}

pub fn page_of(w: &Work, body: Vec<Line>) -> (String, Page) {
    let url = url_of(w);
    let (title, subject) = if w.kind == JUDGMENT { judgment_title(&w.title) } else { (w.title.split_whitespace().collect::<Vec<_>>().join(" "), String::new()) };
    let credit = if w.kind == JUDGMENT {
        "Source: EUR-Lex, case law of the Court of Justice of the European Union, © European Union. Only the European Court Reports are authentic."
    } else {
        "Source: EUR-Lex, © European Union. The text as published in the Official Journal of the European Union, which alone is authentic; later amendments are not shown."
    };
    let mut lines = vec![Line::new(credit)];
    let ids: Vec<String> = [("CELEX", &w.celex), ("ELI", &w.eli), ("ECLI", &w.ecli), ("Date", &w.date)]
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    lines.push(Line::new(ids.join(". ") + "."));
    if !subject.is_empty() {
        lines.push(Line::new(format!("Subject: {subject}.")));
    }
    lines.extend(body);
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), published: w.date.clone(), site_name: SITE.into(), ..PaperMeta::default() };
    (url, typed_page(LAW, &title, lines, meta))
}

fn body_lines(text: &str) -> Vec<Line> {
    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.first().is_some_and(|l| l.starts_with("# ")) {
        lines.remove(0);
    }
    lines.into_iter().map(Line::new).collect()
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64, spawn: &(dyn Fn() -> Result<Worker> + Sync)) -> Result<Counts> {
    let works: Vec<Work> = read_works(&args.folder)?.into_iter().filter(|w| doc_path(&args.folder, &w.celex).exists()).collect();
    let mut counts = Counts::default();
    let read = |worker: &mut Worker, w: &Work| -> Result<String> {
        let html = String::from_utf8_lossy(&std::fs::read(doc_path(&args.folder, &w.celex))?).into_owned();
        let url = url_of(w);
        let size = u32::try_from(url.len())?.to_be_bytes();
        let answer = worker.ask_json(&[EXTRACT, &size, url.as_bytes(), html.as_bytes()])?;
        Ok(page_from_json(answer.get("page").ok_or_else(|| anyhow!("an extract answer without a page"))?)?.text)
    };
    let mut seen = HashSet::new();
    for chunk in works.chunks(2_000) {
        let texts = map_in_order(chunk, args.workers, spawn, &read)?;
        for (w, text) in chunk.iter().zip(texts) {
            let body = body_lines(&text);
            if body.is_empty() {
                counts.add("empty", 1);
                continue;
            }
            let (url, page) = page_of(w, body);
            if !seen.insert(url.clone()) {
                counts.add("duplicate", 1);
                continue;
            }
            batches.add(upload_row(&url, page, as_of_us, "application/xhtml+xml"))?;
            counts.add(if w.kind == JUDGMENT { "judgments" } else { "acts" }, 1);
            counts.add("pages", 1);
            if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
                return Ok(counts);
            }
        }
    }
    Ok(counts)
}

pub fn main(args: Args) -> Result<()> {
    let python_path = args.python.python_path(PAGE_WORKER);
    let spawn = || Worker::spawn(&args.python.python, &python_path, PAGE_WORKER, &[]);
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us, &spawn)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(kind: &str, celex: &str, eli: &str) -> Work {
        Work { celex: celex.into(), kind: kind.into(), eli: eli.into(), title: "T".into(), ..Work::default() }
    }

    #[test]
    fn addresses_by_eli_or_celex() {
        assert_eq!(url_of(&work(ACT, "32016R0679", "http://data.europa.eu/eli/reg/2016/679/oj")), "https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng");
        assert_eq!(url_of(&work(ACT, "31953D0030", "")), "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:31953D0030");
        assert_eq!(url_of(&work(JUDGMENT, "62014CJ0362", "")), "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:62014CJ0362");
    }

    #[test]
    fn judgment_titles_read() {
        let raw = "__[unreferenced:arret-c-vi-ext]__ of 12 November 2026.#Council of the European Union v Vladimir Gheorghe Plahotniuc.#Appeal – Restrictive measures.#Case C-897/24 P.";
        let (title, subject) = judgment_title(raw);
        assert_eq!(title, "Council of the European Union v Vladimir Gheorghe Plahotniuc (Case C-897/24 P), Judgment of 12 November 2026");
        assert_eq!(subject, "Appeal – Restrictive measures");
    }

    #[test]
    fn act_page_credits_eur_lex() {
        let mut w = work(ACT, "32016R0679", "http://data.europa.eu/eli/reg/2016/679/oj");
        w.date = "2016-04-27".into();
        let (url, page) = page_of(&w, vec![Line::new("Article 1"), Line::new("Subject-matter and objectives")]);
        assert_eq!(url, "https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng");
        let lines: Vec<&str> = page.text.lines().collect();
        assert!(lines[1].starts_with("Source: EUR-Lex, © European Union."));
        assert_eq!(lines[2], "CELEX: 32016R0679. ELI: http://data.europa.eu/eli/reg/2016/679/oj. Date: 2016-04-27.");
        assert_eq!(&lines[3..], ["Article 1", "Subject-matter and objectives"]);
    }
}

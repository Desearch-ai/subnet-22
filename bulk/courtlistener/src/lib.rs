//! CourtListener's bulk data (Free Law Project, public domain): a page per opinion cluster at its courtlistener.com address.

pub mod markup;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use serde::{Deserialize, Serialize};

pub const SOURCE: &str = "courtlistener";
pub const SITE: &str = "CourtListener";
pub const HOST: &str = "https://www.courtlistener.com";
const CONTENT_TYPE: &str = "text/html";
/// The text columns of an opinion, best first: `html_with_citations` is what courtlistener.com shows.
const TEXTS: [&str; 7] = ["html_with_citations", "xml_harvard", "html_columbia", "html_lawbox", "html", "html_anon_2020", "plain_text"];
const MAX_SYLLABUS_CHARS: usize = 6_000;
const MAX_AUTHOR_CHARS: usize = 80;
/// Tiers by value: the must-have list, the Supreme Court, published federal appellate and state supreme court opinions, then by citations.
pub const TIER_MUST: u8 = 0;
pub const TIER_SCOTUS: u8 = 1;
pub const TIER_FEDERAL_APPEALS: u8 = 2;
pub const TIER_STATE_SUPREME: u8 = 3;
pub const TIER_CITED_10: u8 = 4;
pub const TIER_CITED: u8 = 5;
pub const TIER_REST: u8 = 6;

fn csv_reader(path: &Path) -> Result<csv::Reader<Box<dyn Read>>> {
    let file: Box<dyn Read> =
        if path.as_os_str() == "-" { Box::new(std::io::stdin()) } else { Box::new(File::open(path).with_context(|| format!("opening {}", path.display()))?) };
    Ok(csv::ReaderBuilder::new().escape(Some(b'\\')).double_quote(false).flexible(true).buffer_capacity(1 << 20).from_reader(file))
}

struct Columns(HashMap<String, usize>);

impl Columns {
    fn of<R: Read>(reader: &mut csv::Reader<R>) -> Result<Columns> {
        Ok(Columns(reader.byte_headers()?.iter().enumerate().map(|(i, h)| (String::from_utf8_lossy(h).into_owned(), i)).collect()))
    }

    fn at(&self, name: &str) -> Result<usize> {
        self.0.get(name).copied().with_context(|| format!("no {name} column"))
    }
}

fn field<'a>(record: &'a csv::ByteRecord, at: usize) -> std::borrow::Cow<'a, str> {
    String::from_utf8_lossy(record.get(at).unwrap_or_default())
}

fn number(record: &csv::ByteRecord, at: usize) -> Option<u64> {
    std::str::from_utf8(record.get(at)?).ok()?.trim().parse().ok()
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cut_chars(text: &str, most: usize) -> &str {
    text.char_indices().nth(most).map_or(text, |(end, _)| &text[..end])
}

/// One opinion cluster, its page's header and what the opinions dump holds for it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub id: u64,
    pub tier: u8,
    pub slug: String,
    pub name: String,
    pub court_id: String,
    pub court: String,
    pub date: String,
    pub judges: String,
    pub citations: Vec<String>,
    pub docket: String,
    pub status: String,
    pub cited: u64,
    pub syllabus: Vec<String>,
    pub opinions: u32,
    pub chars: u64,
}

impl Case {
    pub fn url(&self) -> String {
        let slug = if self.slug.is_empty() { "case" } else { self.slug.as_str() };
        format!("{HOST}/opinion/{}/{slug}/", self.id)
    }

    pub fn title(&self) -> String {
        let year = self.date.get(..4).unwrap_or("");
        match (self.court.is_empty(), year.is_empty()) {
            (false, false) => format!("{} ({}, {year})", self.name, self.court),
            (false, true) => format!("{} ({})", self.name, self.court),
            (true, false) => format!("{} ({year})", self.name),
            (true, true) => self.name.clone(),
        }
    }
}

#[derive(clap::Args, Clone, Debug)]
pub struct OpinionsArgs {
    /// The opinions dump, decompressed (`-` for stdin).
    pub opinions: PathBuf,
    /// The index to write: each opinion's id, cluster, type and text length.
    #[arg(long)]
    pub out: PathBuf,
}

/// Indexes the opinions dump, so `courtlistener-cases` knows how many opinions each cluster has.
pub fn opinions_main(args: OpinionsArgs) -> Result<()> {
    let mut reader = csv_reader(&args.opinions)?;
    let columns = Columns::of(&mut reader)?;
    let (id, cluster, kind) = (columns.at("id")?, columns.at("cluster_id")?, columns.at("type")?);
    let texts: Vec<usize> = TEXTS.iter().filter_map(|name| columns.0.get(*name).copied()).collect();
    let mut out = BufWriter::new(File::create(&args.out)?);
    writeln!(out, "id\tcluster_id\ttype\tchars")?;
    let mut record = csv::ByteRecord::new();
    let mut n = 0u64;
    while reader.read_byte_record(&mut record)? {
        let chars = texts.iter().map(|&at| record.get(at).map_or(0, <[u8]>::len)).find(|&len| len > 0).unwrap_or(0);
        writeln!(out, "{}\t{}\t{}\t{chars}", field(&record, id), field(&record, cluster), field(&record, kind))?;
        n += 1;
    }
    out.flush()?;
    println!("{}", serde_json::json!({"opinions": n}));
    Ok(())
}

#[derive(clap::Args, Clone, Debug)]
pub struct CasesArgs {
    /// The courts dump, decompressed.
    #[arg(long)]
    pub courts: PathBuf,
    /// The opinion clusters dump, decompressed.
    #[arg(long)]
    pub clusters: PathBuf,
    /// The dockets dump, decompressed.
    #[arg(long)]
    pub dockets: PathBuf,
    /// The citations dump, decompressed.
    #[arg(long)]
    pub citations: PathBuf,
    /// The index `courtlistener-opinions` wrote; without it every case is kept with its opinions uncounted, and the reader holds them to the end.
    #[arg(long)]
    pub opinions: Option<PathBuf>,
    /// Cluster ids to put first, one a line.
    #[arg(long)]
    pub must: Option<PathBuf>,
    /// The cases file to write, a JSON line a case, best tier first and most cited first within one.
    #[arg(long)]
    pub out: PathBuf,
}

/// The tier of a case by its court's id and jurisdiction, status and citations.
pub fn tier_of(court_id: &str, jurisdiction: &str, status: &str, cited: u64) -> u8 {
    let published = status == "Published";
    if court_id == "scotus" {
        TIER_SCOTUS
    } else if published && jurisdiction == "F" {
        TIER_FEDERAL_APPEALS
    } else if published && jurisdiction == "S" {
        TIER_STATE_SUPREME
    } else if cited >= 10 {
        TIER_CITED_10
    } else if cited >= 1 {
        TIER_CITED
    } else {
        TIER_REST
    }
}

/// `347 U.S. 483` from a citations row's volume, reporter and page.
pub fn citation_text(volume: &str, reporter: &str, page: &str) -> String {
    squash(&format!("{volume} {reporter} {page}"))
}

pub fn cases_main(args: CasesArgs) -> Result<()> {
    let mut counts = Counts::default();
    let mut courts: HashMap<String, (String, String)> = HashMap::new();
    {
        let mut reader = csv_reader(&args.courts)?;
        let c = Columns::of(&mut reader)?;
        let (id, jurisdiction, full) = (c.at("id")?, c.at("jurisdiction")?, c.at("full_name")?);
        let mut record = csv::ByteRecord::new();
        while reader.read_byte_record(&mut record)? {
            courts.insert(field(&record, id).into_owned(), (field(&record, jurisdiction).into_owned(), squash(&field(&record, full))));
        }
    }
    let mut opinions: HashMap<u64, (u32, u64)> = HashMap::new();
    let lines = match &args.opinions {
        Some(path) => Some(BufReader::new(File::open(path)?).lines().skip(1)),
        None => None,
    };
    for line in lines.into_iter().flatten() {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();
        let (Some(cluster), Some(chars)) = (parts.get(1).and_then(|c| c.parse::<u64>().ok()), parts.get(3).and_then(|c| c.parse::<u64>().ok())) else {
            continue;
        };
        let held = opinions.entry(cluster).or_default();
        held.0 += 1;
        held.1 += chars;
    }
    let must: HashSet<u64> = match &args.must {
        Some(path) => std::fs::read_to_string(path)?.lines().filter_map(|l| l.trim().parse().ok()).collect(),
        None => HashSet::new(),
    };
    let mut cases: Vec<Case> = Vec::new();
    let mut by_docket: HashMap<u64, Vec<u32>> = HashMap::new();
    {
        let mut reader = csv_reader(&args.clusters)?;
        let c = Columns::of(&mut reader)?;
        let (id, docket, date, status, cited, blocked, slug) =
            (c.at("id")?, c.at("docket_id")?, c.at("date_filed")?, c.at("precedential_status")?, c.at("citation_count")?, c.at("blocked")?, c.at("slug")?);
        let (name, name_full, name_short, judges, syllabus) =
            (c.at("case_name")?, c.at("case_name_full")?, c.at("case_name_short")?, c.at("judges")?, c.at("syllabus")?);
        let mut record = csv::ByteRecord::new();
        while reader.read_byte_record(&mut record)? {
            counts.add("clusters", 1);
            let Some(cluster) = number(&record, id) else { continue };
            if field(&record, blocked) == "t" {
                counts.add("blocked", 1);
                continue;
            }
            let (n, chars) = match opinions.get(&cluster) {
                Some(&held) => held,
                None if args.opinions.is_none() => (0, 0),
                None => {
                    counts.add("no_opinions", 1);
                    continue;
                }
            };
            let name =
                [name, name_full, name_short].iter().map(|&at| squash(&field(&record, at))).find(|n| !n.is_empty()).unwrap_or_else(|| "Unnamed case".into());
            let syllabus_text = field(&record, syllabus);
            let syllabus = if syllabus_text.trim().is_empty() { Vec::new() } else { markup::paragraphs(cut_chars(&syllabus_text, MAX_SYLLABUS_CHARS * 2)) };
            if let Some(d) = number(&record, docket) {
                by_docket.entry(d).or_default().push(cases.len() as u32);
            }
            cases.push(Case {
                id: cluster,
                slug: field(&record, slug).into_owned(),
                name,
                date: field(&record, date).into_owned(),
                judges: squash(&field(&record, judges)),
                status: field(&record, status).into_owned(),
                cited: number(&record, cited).unwrap_or(0),
                syllabus,
                opinions: n,
                chars,
                ..Case::default()
            });
        }
    }
    let mut blocked_docket = vec![false; cases.len()];
    {
        let mut reader = csv_reader(&args.dockets)?;
        let c = Columns::of(&mut reader)?;
        let (id, court, number_at, blocked) = (c.at("id")?, c.at("court_id")?, c.at("docket_number")?, c.at("blocked")?);
        let mut record = csv::ByteRecord::new();
        while reader.read_byte_record(&mut record)? {
            let Some(held) = number(&record, id).and_then(|d| by_docket.get(&d)) else { continue };
            let court_id = field(&record, court).into_owned();
            let docket_number = squash(&field(&record, number_at));
            let is_blocked = field(&record, blocked) == "t";
            for &at in held {
                let case = &mut cases[at as usize];
                case.court_id = court_id.clone();
                case.docket = docket_number.clone();
                blocked_docket[at as usize] = is_blocked;
            }
        }
    }
    drop(by_docket);
    let at_of: HashMap<u64, u32> = cases.iter().enumerate().map(|(i, c)| (c.id, i as u32)).collect();
    let mut cites: HashMap<u32, Vec<(u32, String)>> = HashMap::new();
    {
        let mut reader = csv_reader(&args.citations)?;
        let c = Columns::of(&mut reader)?;
        let (volume, reporter, page, kind, cluster) = (c.at("volume")?, c.at("reporter")?, c.at("page")?, c.at("type")?, c.at("cluster_id")?);
        let mut record = csv::ByteRecord::new();
        while reader.read_byte_record(&mut record)? {
            let Some(&at) = number(&record, cluster).and_then(|c| at_of.get(&c)) else { continue };
            let text = citation_text(&field(&record, volume), &field(&record, reporter), &field(&record, page));
            cites.entry(at).or_default().push((number(&record, kind).unwrap_or(99) as u32, text));
        }
    }
    let mut kept = Vec::with_capacity(cases.len());
    for (i, mut case) in cases.into_iter().enumerate() {
        if blocked_docket[i] {
            counts.add("blocked", 1);
            continue;
        }
        if let Some(mut held) = cites.remove(&(i as u32)) {
            held.sort();
            case.citations = held.into_iter().map(|(_, t)| t).collect();
        }
        let (jurisdiction, court) = courts.get(&case.court_id).cloned().unwrap_or_default();
        case.court = court;
        case.tier = if must.contains(&case.id) { TIER_MUST } else { tier_of(&case.court_id, &jurisdiction, &case.status, case.cited) };
        kept.push(case);
    }
    kept.sort_by(|a, b| (a.tier, std::cmp::Reverse(a.cited), a.id).cmp(&(b.tier, std::cmp::Reverse(b.cited), b.id)));
    let mut out = BufWriter::new(File::create(&args.out)?);
    let mut tiers: Vec<(u64, u64, u64)> = vec![(0, 0, 0); TIER_REST as usize + 1];
    for case in &kept {
        serde_json::to_writer(&mut out, case)?;
        out.write_all(b"\n")?;
        let t = &mut tiers[case.tier as usize];
        t.0 += 1;
        t.1 += case.opinions as u64;
        t.2 += case.chars;
    }
    out.flush()?;
    let mut line = serde_json::Map::new();
    for (key, n) in counts.iter() {
        line.insert(key.into(), n.into());
    }
    line.insert("cases".into(), kept.len().into());
    line.insert("tiers".into(), tiers.iter().map(|(c, o, ch)| serde_json::json!({"cases": c, "opinions": o, "markup_chars": ch})).collect());
    println!("{}", serde_json::Value::Object(line));
    Ok(())
}

/// One opinion of a cluster: its type code, author and paragraphs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Opinion {
    pub id: u64,
    pub kind: String,
    pub author: String,
    pub paragraphs: Vec<String>,
}

/// The heading an opinion's type code gets, as `020lead` for a lead opinion.
pub fn kind_name(kind: &str) -> &'static str {
    match kind.trim_start_matches(|c: char| c.is_ascii_digit()) {
        "combined" => "Opinion",
        "unamimous" | "unanimous" => "Unanimous opinion",
        "lead" => "Opinion of the court",
        "plurality" => "Plurality opinion",
        "concurrence" => "Concurrence",
        "concurrenceinpart" => "Concurrence in part and dissent in part",
        "dissent" => "Dissent",
        "addendum" => "Addendum",
        "remittitur" => "Remittitur",
        "rehearing" => "On rehearing",
        "onthemerits" => "On the merits",
        "onmotiontostrike" => "On motion to strike",
        "trialcourt" => "Trial court document",
        _ => "Opinion",
    }
}

/// A case's page: the title line, its citation, court, date, docket and judges, the syllabus, then each opinion under its heading.
pub fn page_of(case: &Case, mut opinions: Vec<Opinion>) -> Option<Page> {
    opinions.retain(|o| !o.paragraphs.is_empty());
    if opinions.is_empty() {
        return None;
    }
    opinions.sort_by(|a, b| (&a.kind, a.id).cmp(&(&b.kind, b.id)));
    let title = case.title();
    let mut lines = Vec::new();
    if !case.citations.is_empty() {
        lines.push(Line::new(format!("Citation: {}", case.citations.join(", "))));
    }
    if !case.court.is_empty() {
        lines.push(Line::new(format!("Court: {}", case.court)));
    }
    if !case.date.is_empty() {
        lines.push(Line::new(format!("Decided: {}", case.date)));
    }
    if !case.docket.is_empty() {
        lines.push(Line::new(format!("Docket number: {}", case.docket)));
    }
    if !case.judges.is_empty() {
        lines.push(Line::new(format!("Judges: {}", case.judges)));
    }
    if !case.status.is_empty() && case.status != "Published" {
        lines.push(Line::new(format!("Precedential status: {}", case.status)));
    }
    if !case.syllabus.is_empty() {
        lines.push(Line::new("## Syllabus"));
        let mut held = 0;
        for p in &case.syllabus {
            if held >= MAX_SYLLABUS_CHARS {
                break;
            }
            held += p.chars().count();
            lines.push(Line::new(p.clone()));
        }
    }
    let mut headings: HashMap<String, u32> = HashMap::new();
    for opinion in &opinions {
        let author = squash(cut_chars(&opinion.author, MAX_AUTHOR_CHARS));
        let mut heading = if author.is_empty() { kind_name(&opinion.kind).to_string() } else { format!("{} ({author})", kind_name(&opinion.kind)) };
        let seen = headings.entry(heading.clone()).or_default();
        *seen += 1;
        if *seen > 1 {
            heading = format!("{heading} {seen}");
        }
        lines.push(Line::new(format!("## {heading}")));
        lines.extend(opinion.paragraphs.iter().cloned().map(Line::new));
    }
    let author = opinions.first().map(|o| squash(cut_chars(&o.author, MAX_AUTHOR_CHARS))).unwrap_or_default();
    let meta = PaperMeta { lang: "en".into(), url: case.url(), published: case.date.clone(), author, site_name: SITE.into(), ..PaperMeta::default() };
    Some(typed_page(bulk_core::rows::PAPER, &title, lines, meta))
}

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The cases file `courtlistener-cases` wrote.
    #[arg(long)]
    pub cases: PathBuf,
    /// The tiers to take, as 0,1,2.
    #[arg(long, value_delimiter = ',', default_value = "0,1,2,3")]
    pub tiers: Vec<u8>,
    /// Cases of those tiers to pass over first, in the cases file's order.
    #[arg(long, default_value_t = 0)]
    pub skip: u64,
    /// Cases of those tiers to take at most after the skipped ones; 0 for all.
    #[arg(long, default_value_t = 0)]
    pub take: u64,
    /// Close a run every this many pages and start the next in its own subfolder, `<out>/0001` as run `<run>-0001`; 0 for one run.
    #[arg(long = "part-pages", default_value_t = 0)]
    pub part_pages: u64,
    /// Opinion text a case needs at least, so one-line orders make no page; the must-have tier is taken whatever its length.
    #[arg(long = "min-chars", default_value_t = 500)]
    pub min_chars: usize,
    /// The opinions dump the index was made from, decompressed (`-` for stdin).
    pub opinions: PathBuf,
}

/// The run's batches, or a part of it closed every `part_pages` pages so each can be published while the rest is read.
pub struct Parts {
    out: PathBuf,
    run: String,
    as_of_us: i64,
    part_pages: u64,
    part: u32,
    batches: Option<Batches>,
}

impl Parts {
    pub fn new(args: &RunArgs, part_pages: u64) -> Result<Parts> {
        let as_of_us = bulk_core::rows::parse_as_of(&args.as_of)?;
        let run = if args.run.is_empty() { bulk_core::rows::utc_format(as_of_us, "%Y%m%d") } else { args.run.clone() };
        Ok(Parts { out: args.out.clone(), run, as_of_us, part_pages, part: 0, batches: None })
    }

    pub fn as_of_us(&self) -> i64 {
        self.as_of_us
    }

    fn open(&mut self) -> Result<&mut Batches> {
        if self.batches.is_none() {
            let (out, run) = if self.part_pages == 0 {
                (self.out.clone(), self.run.clone())
            } else {
                self.part += 1;
                (self.out.join(format!("{:04}", self.part)), format!("{}-{:04}", self.run, self.part))
            };
            self.batches = Some(Batches::new(&out, SOURCE, &run, self.as_of_us)?);
        }
        Ok(self.batches.as_mut().expect("open batches"))
    }

    pub fn add(&mut self, row: bulk_core::Row) -> Result<()> {
        let part_pages = self.part_pages;
        let batches = self.open()?;
        batches.add(row)?;
        if part_pages > 0 && batches.rows() >= part_pages {
            let rows = batches.rows();
            let mut counts = Counts::default();
            counts.add("part_pages", rows);
            self.close(&counts)?;
        }
        Ok(())
    }

    fn close(&mut self, counts: &Counts) -> Result<()> {
        if let Some(batches) = self.batches.take() {
            let summary = batches.close(counts)?;
            println!("{}", bulk_core::run::summary_line(&summary.out, &summary, counts));
        }
        Ok(())
    }

    /// Closes the last run with the whole read's counts, opening it first when no page came at all.
    pub fn finish(mut self, counts: &Counts) -> Result<()> {
        if self.batches.is_none() && self.part == 0 {
            self.open()?;
        }
        let mut last = counts.clone();
        if let Some(batches) = self.batches.as_ref().filter(|_| self.part_pages > 0) {
            last.add("part_pages", batches.rows());
        }
        if self.batches.is_none() {
            println!("{}", serde_json::to_string(&counts.iter().collect::<std::collections::BTreeMap<_, _>>())?);
        }
        self.close(&last)
    }
}

/// The cases a run takes, by cluster id.
pub fn chosen(path: &Path, tiers: &[u8], skip: u64, take: u64) -> Result<HashMap<u64, Case>> {
    let mut cases = HashMap::new();
    let mut seen = 0u64;
    for line in BufReader::new(File::open(path)?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        if !tiers.contains(&case.tier) {
            continue;
        }
        seen += 1;
        if seen <= skip {
            continue;
        }
        if take > 0 && seen > skip + take {
            break;
        }
        cases.insert(case.id, case);
    }
    Ok(cases)
}

pub fn run(args: &Args, batches: &mut Parts) -> Result<Counts> {
    let as_of_us = batches.as_of_us();
    let mut cases = chosen(&args.cases, &args.tiers, args.skip, args.take)?;
    let mut counts = Counts::default();
    counts.add("cases", cases.len() as u64);
    let mut pending: HashMap<u64, Vec<Opinion>> = HashMap::new();
    let mut reader = csv_reader(&args.opinions)?;
    let c = Columns::of(&mut reader)?;
    let (id, cluster, kind, author) = (c.at("id")?, c.at("cluster_id")?, c.at("type")?, c.at("author_str")?);
    let texts: Vec<usize> = TEXTS.iter().filter_map(|name| c.0.get(*name).copied()).collect();
    if texts.is_empty() {
        bail!("the opinions dump has none of the text columns");
    }
    let mut record = csv::ByteRecord::new();
    while reader.read_byte_record(&mut record)? {
        counts.add("opinions", 1);
        let Some(cluster_id) = number(&record, cluster) else { continue };
        let Some(case) = cases.get(&cluster_id) else { continue };
        let markup = texts.iter().filter_map(|&at| record.get(at)).find(|t| !t.is_empty()).map(String::from_utf8_lossy).unwrap_or_default();
        let opinion = Opinion {
            id: number(&record, id).unwrap_or(0),
            kind: field(&record, kind).into_owned(),
            author: field(&record, author).into_owned(),
            paragraphs: markup::paragraphs(&markup),
        };
        let held = pending.entry(cluster_id).or_default();
        held.push(opinion);
        if case.opinions == 0 || held.len() < case.opinions as usize {
            continue;
        }
        let held = pending.remove(&cluster_id).unwrap_or_default();
        let case = cases.remove(&cluster_id).expect("a pending case");
        emit(&case, held, args.min_chars, batches, as_of_us, &mut counts)?;
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            return Ok(counts);
        }
    }
    let mut rest: Vec<u64> = pending.keys().copied().collect();
    rest.sort();
    for cluster_id in rest {
        let held = pending.remove(&cluster_id).unwrap_or_default();
        if let Some(case) = cases.remove(&cluster_id) {
            counts.add(if case.opinions == 0 { "uncounted" } else { "incomplete" }, 1);
            emit(&case, held, args.min_chars, batches, as_of_us, &mut counts)?;
        }
    }
    counts.add("missing", cases.len() as u64);
    Ok(counts)
}

fn emit(case: &Case, opinions: Vec<Opinion>, min_chars: usize, batches: &mut Parts, as_of_us: i64, counts: &mut Counts) -> Result<()> {
    let chars: usize = opinions.iter().flat_map(|o| &o.paragraphs).map(|p| p.chars().count()).sum();
    if chars < min_chars && case.tier != TIER_MUST {
        counts.add(if chars == 0 { "empty" } else { "short" }, 1);
        return Ok(());
    }
    match page_of(case, opinions) {
        Some(page) => {
            batches.add(upload_row(&case.url(), page, as_of_us, CONTENT_TYPE))?;
            counts.add("pages", 1);
        }
        None => counts.add("empty", 1),
    }
    Ok(())
}

pub fn main(args: Args) -> Result<()> {
    let mut parts = Parts::new(&args.run, args.part_pages)?;
    let counts = run(&args, &mut parts)?;
    parts.finish(&counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> Case {
        Case {
            id: 105221,
            tier: TIER_SCOTUS,
            slug: "brown-v-board-of-education".into(),
            name: "Brown v. Board of Education".into(),
            court_id: "scotus".into(),
            court: "Supreme Court of the United States".into(),
            date: "1954-05-17".into(),
            judges: "Warren".into(),
            citations: vec!["347 U.S. 483".into(), "74 S. Ct. 686".into()],
            docket: "1".into(),
            status: "Published".into(),
            cited: 1200,
            syllabus: vec!["Segregation of children in public schools solely on the basis of race deprives them of equal protection.".into()],
            opinions: 2,
            chars: 100,
        }
    }

    #[test]
    fn page_of_a_case() {
        let opinions = vec![
            Opinion { id: 2, kind: "040dissent".into(), author: "Doe".into(), paragraphs: vec!["I dissent.".into()] },
            Opinion {
                id: 1,
                kind: "020lead".into(),
                author: "Warren".into(),
                paragraphs: vec!["These cases come to us from the States.".into(), "It is so ordered.".into()],
            },
        ];
        let page = page_of(&case(), opinions).unwrap();
        assert_eq!(case().url(), "https://www.courtlistener.com/opinion/105221/brown-v-board-of-education/");
        assert_eq!((page.page_type.as_str(), page.published.as_str(), page.site_name.as_str()), ("paper", "1954-05-17", "CourtListener"));
        assert_eq!(
            page.text.lines().collect::<Vec<_>>(),
            [
                "Brown v. Board of Education (Supreme Court of the United States, 1954)",
                "Citation: 347 U.S. 483, 74 S. Ct. 686",
                "Court: Supreme Court of the United States",
                "Decided: 1954-05-17",
                "Docket number: 1",
                "Judges: Warren",
                "## Syllabus",
                "Segregation of children in public schools solely on the basis of race deprives them of equal protection.",
                "## Opinion of the court (Warren)",
                "These cases come to us from the States.",
                "It is so ordered.",
                "## Dissent (Doe)",
                "I dissent.",
            ]
        );
    }

    #[test]
    fn empty_opinions_make_no_page() {
        assert!(page_of(&case(), vec![Opinion { id: 1, kind: "010combined".into(), ..Opinion::default() }]).is_none());
    }

    #[test]
    fn tiers() {
        assert_eq!(tier_of("scotus", "F", "Unknown", 0), TIER_SCOTUS);
        assert_eq!(tier_of("ca9", "F", "Published", 0), TIER_FEDERAL_APPEALS);
        assert_eq!(tier_of("ca9", "F", "Unpublished", 12), TIER_CITED_10);
        assert_eq!(tier_of("cal", "S", "Published", 0), TIER_STATE_SUPREME);
        assert_eq!(tier_of("calctapp", "SA", "Published", 3), TIER_CITED);
        assert_eq!(tier_of("calctapp", "SA", "Published", 0), TIER_REST);
    }

    #[test]
    fn run_over_a_dump() {
        let dir = tempfile::tempdir().unwrap();
        let cases = dir.path().join("cases.jsonl");
        let mut other = case();
        other.id = 7;
        other.opinions = 1;
        other.tier = TIER_REST;
        std::fs::write(&cases, format!("{}\n{}\n", serde_json::to_string(&case()).unwrap(), serde_json::to_string(&other).unwrap())).unwrap();
        let dump = dir.path().join("opinions.csv");
        std::fs::write(
            &dump,
            concat!(
                "id,type,author_str,plain_text,html_with_citations,cluster_id\n",
                "1,\"020lead\",\"Warren\",\"\",\"<p>It is \\\"so\\\" ordered.</p>\",\"105221\"\n",
                "3,\"010combined\",\"\",\"x\",\"\",\"7\"\n",
                "2,\"040dissent\",\"Doe\",\"I dissent.\",\"\",\"105221\"\n",
            ),
        )
        .unwrap();
        let out = dir.path().join("run");
        let args = Args {
            run: RunArgs { out: out.clone(), as_of: "2026-09-30T00:00:00Z".into(), run: String::new(), limit: 0 },
            cases,
            tiers: vec![TIER_SCOTUS],
            skip: 0,
            take: 0,
            part_pages: 0,
            min_chars: 5,
            opinions: dump,
        };
        let mut parts = Parts::new(&args.run, 0).unwrap();
        let counts = run(&args, &mut parts).unwrap();
        assert_eq!((counts.get("cases"), counts.get("opinions"), counts.get("pages"), counts.get("incomplete")), (1, 3, 1, 0));
        parts.finish(&counts).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!((manifest["rows"].as_u64(), manifest["run"].as_str()), (Some(1), Some("20260930")));
    }

    #[test]
    fn parts_close_every_so_many_pages() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunArgs { out: dir.path().join("run"), as_of: "2026-09-30T00:00:00Z".into(), run: "cl".into(), limit: 0 };
        let mut parts = Parts::new(&run, 2).unwrap();
        for id in 1..=3 {
            let mut c = case();
            c.id = id;
            let page = page_of(&c, vec![Opinion { id, kind: "010combined".into(), author: String::new(), paragraphs: vec!["Text.".into()] }]).unwrap();
            parts.add(upload_row(&c.url(), page, parts.as_of_us(), CONTENT_TYPE)).unwrap();
        }
        let mut counts = Counts::default();
        counts.add("pages", 3);
        parts.finish(&counts).unwrap();
        let read = |part: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("run").join(part).join("manifest.json")).unwrap()).unwrap()
        };
        assert_eq!((read("0001")["rows"].as_u64(), read("0001")["run"].as_str()), (Some(2), Some("cl-0001")));
        assert_eq!((read("0002")["rows"].as_u64(), read("0002")["counts"]["pages"].as_u64()), (Some(1), Some(3)));
    }
}

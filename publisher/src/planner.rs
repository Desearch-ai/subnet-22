//! The planner: follows the change log and turns published pages that need a vector into embed inputs for the task API, within an hourly page budget and the room the API gives.

use std::collections::{btree_map, BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, Int32Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use desearch::feeds::CHANGES;
use desearch::r2::{self, Bucket, PARQUET};
use desearch::time::utc_day;
use futures::{stream, FutureExt, StreamExt};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use rocksdb::{DBCompressionType, IteratorMode, Options, WriteBatch, DB};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};

use crate::backfill::ShardReader;
use crate::current::{decode_changes, ChangeRow, PageRow, SHARDS};
use crate::records::{Record, EXTRACTION_V2};
use crate::service::now_us;
use crate::set::{page_date, Refusal, Set};

/// How a page becomes texts; any change to what a recipe's texts hold is a new recipe, so every page is embedded again.
pub const RECIPE_V1: &str = "page-v1";
/// For v2 texts, which are markdown.
pub const RECIPE_V2: &str = "page-v2";
pub const RECIPES: [&str; 2] = [RECIPE_V1, RECIPE_V2];
/// The recipes planned; pages of any other keep the vectors they have.
pub const PLANNED_RECIPES: [&str; 1] = [RECIPE_V2];
pub const PAGE: &str = "page";
pub const PASSAGE: &str = "passage";
/// Enough for the model's 2,048 tokens; the model server cuts the rest.
const PAGE_CHARS: usize = 10_000;
/// A page longer than this also gets passages, when its type is one that has them.
const PASSAGES_FROM_CHARS: usize = 2_000;
/// The page types that get passages; None for every type.
const PASSAGE_TYPES_V1: Option<&[&str]> = Some(&["article"]);
const PASSAGE_TYPES_V2: Option<&[&str]> = None;
const PASSAGE_MIN_CHARS: usize = 600;
const PASSAGE_MAX_CHARS: usize = 1_500;
const MAX_PASSAGES_V1: usize = 16;
const MAX_PASSAGES_V2: usize = 32;
const FENCE: &str = "```";
/// The task API's list of inputs waiting for a round.
pub const EMBED_INPUTS: &str = "embed:inputs";
/// Inputs whose task the API dropped after its last attempt.
pub const EMBED_DROPPED: &str = "embed:dropped";
const CURSOR: &[u8] = b"cursor";
/// The backfill's shard and the last key of it planned or passed by, `<shard>\t<key>`.
const BACKFILL: &[u8] = b"backfill";
/// Pages read ahead from `current/` waiting for budget and room.
const BACKFILL_AHEAD: usize = 4_096;
/// How long a pass waits for the backfill's reader before going on.
const BACKFILL_WAIT: Duration = Duration::from_millis(500);
const RANGES_AT_ONCE: usize = 16;
/// Set once every domain's count of planned pages has been built from the planned keys.
const SITES_COUNTED: &[u8] = b"sites-counted";
/// The crawl list's caps are crawl caps, three pages crawled for each one the index keeps.
const CRAWL_PER_INDEX: u64 = 3;
const ROOM_TIMEOUT: Duration = Duration::from_secs(20);
const FILES_PER_PASS: usize = 64;

pub static INPUT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let text = |name: &str| Field::new(name, DataType::Utf8, false);
    Arc::new(Schema::new(vec![
        text("text_id"),
        text("page_key"),
        text("url"),
        text("content_sha1"),
        text("kind"),
        Field::new("index", DataType::Int32, false),
        Field::new("text", DataType::LargeUtf8, false),
        text("recipe"),
    ]))
});

/// One text to embed and where it sits in its page.
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub text_id: String,
    pub kind: &'static str,
    pub index: i32,
    pub text: String,
}

/// The recipe that reads a page's text, by the extraction that wrote it; None for an extraction no recipe reads.
pub fn recipe_of(record: &Record) -> Option<&'static str> {
    match record.extraction.as_deref() {
        None => Some(RECIPE_V1),
        Some(EXTRACTION_V2) => Some(RECIPE_V2),
        Some(_) => None,
    }
}

/// The texts of one page by its recipe: its page text, and the passages of a long page of a type that has them, each led by the title and its section's heading.
pub fn texts(key: &str, record: &Record) -> Vec<Text> {
    let v2 = match recipe_of(record) {
        Some(RECIPE_V1) => false,
        Some(RECIPE_V2) => true,
        _ => return Vec::new(),
    };
    let page = if v2 { page_text_v2(record) } else { page_text(record) };
    let mut texts = vec![Text { text_id: format!("{key}#page"), kind: PAGE, index: 0, text: page }];
    let types = if v2 { PASSAGE_TYPES_V2 } else { PASSAGE_TYPES_V1 };
    let typed = types.is_none_or(|types| record.page_type.as_deref().is_some_and(|t| types.contains(&t)));
    if !typed || record.text.chars().count() <= PASSAGES_FROM_CHARS {
        return texts;
    }
    let sections = if v2 { passages_v2(body_v2(record)) } else { passages(&record.text, &record.headings) };
    let title = record.title.trim();
    for (i, passage) in sections.into_iter().enumerate() {
        let lead = [Some(title).filter(|t| !t.is_empty()), passage.heading.as_deref()].into_iter().flatten().collect::<Vec<_>>().join(" › ");
        let text = if lead.is_empty() { passage.body } else { format!("{lead}\n{}", passage.body) };
        texts.push(Text { text_id: format!("{key}#p{}", i + 1), kind: PASSAGE, index: i as i32 + 1, text });
    }
    texts
}

/// Title, description unless the text opens with it, then the text, one per line, cut to what the model reads.
pub fn page_text(record: &Record) -> String {
    joined(record, record.text.trim())
}

/// `page_text` over the markdown without the title line a v2 text opens with.
pub fn page_text_v2(record: &Record) -> String {
    joined(record, body_v2(record))
}

fn joined(record: &Record, text: &str) -> String {
    let mut parts = Vec::new();
    if !record.title.trim().is_empty() {
        parts.push(record.title.trim());
    }
    if let Some(description) = record.description.as_deref().map(str::trim).filter(|d| !d.is_empty() && !text.starts_with(d)) {
        parts.push(description);
    }
    parts.push(text);
    cut(&parts.join("\n"), PAGE_CHARS)
}

/// A v2 text after its first line when that line is the title, as a heading or not and in any case, as v2 extraction compares them.
fn body_v2(record: &Record) -> &str {
    let (text, title) = (record.text.trim(), record.title.trim());
    let (first, rest) = text.split_once('\n').unwrap_or((text, ""));
    let first = first.trim_start_matches('#').trim();
    if !title.is_empty() && first.to_lowercase() == title.to_lowercase() {
        rest.trim_start()
    } else {
        text
    }
}

/// The text of a markdown heading line: one to six `#`, a space, then the heading.
fn markdown_heading(line: &str) -> Option<&str> {
    let level = line.bytes().take_while(|&b| b == b'#').count();
    let heading = line[level..].strip_prefix([' ', '\t'])?.trim();
    ((1..=6).contains(&level) && !heading.is_empty()).then_some(heading)
}

/// A passage of a page and the heading of the section it is in.
#[derive(Clone, Debug, PartialEq)]
pub struct Passage {
    pub heading: Option<String>,
    pub body: String,
}

/// A piece of a page's text as the splitter places it.
enum Block<'a> {
    /// Starts a new section.
    Heading(&'a str),
    /// A line split at its sentences when it is too long.
    Prose(&'a str),
    /// A table row, a list item with its sub-items, or a fenced code block, in pieces cut only between lines; a table row carries its table's header for a passage that continues the table.
    Whole { pieces: Vec<String>, header: Option<String> },
}

/// Consecutive lines grouped into passages of about `PASSAGE_MIN_CHARS` to `PASSAGE_MAX_CHARS` characters; a line that is one of the page's headings starts a new section, and a long line is split at its sentences.
pub fn passages(text: &str, headings: &[Option<String>]) -> Vec<Passage> {
    let headings: HashSet<&str> = headings.iter().flatten().map(|h| h.trim()).filter(|h| !h.is_empty()).collect();
    let blocks =
        text.lines().map(str::trim).filter(|line| !line.is_empty()).map(|line| if headings.contains(line) { Block::Heading(line) } else { Block::Prose(line) });
    sections(blocks, MAX_PASSAGES_V1)
}

/// `passages` for markdown, up to `MAX_PASSAGES_V2`: each heading line starts a section, and table rows, list items and fenced code blocks are cut only between lines.
pub fn passages_v2(text: &str) -> Vec<Passage> {
    sections(markdown_blocks(text).into_iter(), MAX_PASSAGES_V2)
}

fn markdown_blocks(text: &str) -> Vec<Block<'_>> {
    let lines: Vec<&str> = text.lines().map(str::trim_end).filter(|line| !line.is_empty()).collect();
    let indent = |line: &str| line.len() - line.trim_start().len();
    let mut blocks = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        let (raw, line) = (lines[at], lines[at].trim_start());
        at += 1;
        if line.starts_with(FENCE) {
            if let Some(close) = lines[at..].iter().position(|l| l.trim_start().starts_with(FENCE)) {
                blocks.push(Block::Whole { pieces: cut_between_lines(raw, &lines[at..at + close], Some(lines[at + close])), header: None });
                at += close + 1;
                continue;
            }
        } else if let Some(heading) = markdown_heading(line) {
            blocks.push(Block::Heading(heading));
            continue;
        } else if line.starts_with('|') {
            let end = at + lines[at..].iter().take_while(|l| l.trim_start().starts_with('|')).count();
            // Only a row of header cells is followed by a separator; without one the first row is data.
            let header = (at < end && table_separator(lines[at].trim_start())).then(|| format!("{line}\n{}", lines[at].trim_start()));
            at += usize::from(header.is_some());
            blocks.push(Block::Whole { pieces: vec![header.clone().unwrap_or_else(|| line.to_string())], header: None });
            blocks.extend(lines[at..end].iter().map(|row| Block::Whole { pieces: vec![row.trim_start().to_string()], header: header.clone() }));
            at = end;
            continue;
        } else if list_item(line) {
            let end = at + lines[at..].iter().take_while(|l| indent(l) > indent(raw)).count();
            blocks.push(Block::Whole { pieces: cut_between_lines(line, &lines[at..end], None), header: None });
            at = end;
            continue;
        }
        blocks.push(Block::Prose(line));
    }
    blocks
}

/// `lead`, `lines` and `close` as one piece when it fits a passage, else in pieces of whole lines that each open with `lead` and end with `close`.
fn cut_between_lines(lead: &str, lines: &[&str], close: Option<&str>) -> Vec<String> {
    let size = |line: &str| line.chars().count() + 1;
    let ending = close.map(|close| format!("\n{close}")).unwrap_or_default();
    let fits = |held: usize| held + ending.chars().count() <= PASSAGE_MAX_CHARS;
    let whole = lead.chars().count() + lines.iter().map(|line| size(line)).sum::<usize>();
    let mut pieces = Vec::new();
    let (mut piece, mut held, mut placed) = (lead.to_string(), lead.chars().count(), 0);
    for line in lines {
        if !fits(whole) && placed > 0 && !fits(held + size(line)) {
            pieces.push(std::mem::replace(&mut piece, lead.to_string()) + &ending);
            (held, placed) = (lead.chars().count(), 0);
        }
        piece.push('\n');
        piece.push_str(line);
        held += size(line);
        placed += 1;
    }
    pieces.push(piece + &ending);
    pieces
}

/// A table's `|---|` line, between its header row and its body.
fn table_separator(line: &str) -> bool {
    line.starts_with('|') && line.contains('-') && line.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// `- item` or `N. item`.
fn list_item(line: &str) -> bool {
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    line.starts_with("- ") || (digits > 0 && line[digits..].starts_with(". "))
}

fn sections<'a>(blocks: impl Iterator<Item = Block<'a>>, cap: usize) -> Vec<Passage> {
    let mut passages: Vec<Passage> = Vec::new();
    let mut heading: Option<String> = None;
    let (mut current, mut held) = (String::new(), 0);
    // A tail continuing a table or a cut block is never merged back, so its repeated lead never lands mid-passage.
    let mut continued = false;
    for block in blocks {
        let (pieces, header) = match block {
            Block::Heading(line) => {
                // A section's short tail stays with the next one rather than standing alone.
                if held >= PASSAGE_MIN_CHARS / 3 {
                    passages.push(Passage { heading: heading.clone(), body: std::mem::take(&mut current) });
                    held = 0;
                    if passages.len() >= cap {
                        return passages;
                    }
                }
                heading = Some(line.to_string());
                continue;
            }
            Block::Prose(line) => (sentences_within(line, PASSAGE_MAX_CHARS), None),
            Block::Whole { pieces, header } => (pieces, header),
        };
        for (at, piece) in pieces.into_iter().enumerate() {
            let size = piece.chars().count();
            if held > 0 && held + 1 + size > PASSAGE_MAX_CHARS {
                passages.push(Passage { heading: heading.clone(), body: std::mem::take(&mut current) });
                held = 0;
                if passages.len() >= cap {
                    return passages;
                }
            }
            if held == 0 {
                continued = at > 0 || header.is_some();
                if let Some(header) = &header {
                    current.push_str(header);
                    held = header.chars().count();
                }
            }
            if held > 0 {
                current.push('\n');
                held += 1;
            }
            current.push_str(&piece);
            held += size;
            if held >= PASSAGE_MIN_CHARS {
                passages.push(Passage { heading: heading.clone(), body: std::mem::take(&mut current) });
                held = 0;
            }
            if passages.len() >= cap {
                return passages;
            }
        }
    }
    let room = passages.len() < cap;
    match passages.last_mut() {
        Some(last) if held > 0 && held < PASSAGE_MIN_CHARS / 2 && last.heading == heading && !continued => {
            last.body.push('\n');
            last.body.push_str(&current);
        }
        _ if held > 0 && room => passages.push(Passage { heading, body: current }),
        _ => {}
    }
    passages
}

/// A line in pieces of at most `max` characters, split after sentence ends where it can be.
fn sentences_within(line: &str, max: usize) -> Vec<String> {
    if line.chars().count() <= max {
        return vec![line.to_string()];
    }
    let mut pieces = Vec::new();
    let (mut piece, mut held) = (String::new(), 0);
    for sentence in line.split_inclusive(['.', '?', '!']) {
        let size = sentence.chars().count();
        if held > 0 && held + size > max {
            pieces.push(std::mem::take(&mut piece).trim().to_string());
            held = 0;
        }
        if size > max {
            pieces.extend(sentence.chars().collect::<Vec<_>>().chunks(max).map(|chunk| chunk.iter().collect::<String>().trim().to_string()));
            continue;
        }
        piece.push_str(sentence);
        held += size;
    }
    if held > 0 {
        pieces.push(piece.trim().to_string());
    }
    pieces.retain(|piece| !piece.is_empty());
    pieces
}

fn cut(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => text[..at].to_string(),
        None => text.to_string(),
    }
}

pub struct Settings {
    pub dir: PathBuf,
    /// Pages planned an hour at most; the rest of the log's pages are left for a later backfill.
    pub pages_per_hour: f64,
    pub pages_per_input: usize,
    /// An input is written once its texts reach this many characters, so a task of long articles stays one a miner finishes in its claim time.
    pub chars_per_input: usize,
    /// A part-filled input is written once its first page waited this long.
    pub flush_after: Duration,
    /// Inputs of one lane written but not yet handed over; past this, new pages of the lane are left out.
    pub max_pending: usize,
    /// Pages dated within this long before now go in the fresh lane, which is handed over first and spends no budget; None for no fresh lane.
    pub fresh: Option<Duration>,
    /// The temporary bucket keeps an input a day; one waiting longer than this is dropped, so no miner is sent a file already gone.
    pub pending_max_age: Duration,
    pub min_chars: usize,
    pub langs: Vec<String>,
    pub excluded: HashSet<String>,
    /// The bot's crawl list; when set, only its sites are planned, each up to its index cap. Read again when it changes.
    pub list_file: Option<PathBuf>,
    /// The v1 set; when set, only pages its rules take are planned, and the crawl list is not read. Read again when it changes.
    pub set_file: Option<PathBuf>,
    /// The change file to start after when nothing was planned before; the newest one when None.
    pub start_after: Option<i64>,
    /// Change files fetched and decoded at once, ahead of the one being planned.
    pub reads: usize,
    /// Pages outside the fresh lane are planned with their page text alone; their passages wait until this is off.
    pub page_first: bool,
    /// Read `current/` for pages to embed, with what budget and room the log leaves.
    pub backfill: bool,
    /// The shards the backfill reads, in order.
    pub backfill_shards: Range<usize>,
    pub current_prefix: String,
    pub poll: Duration,
    /// How long a change file number not indexed yet is looked for again before it is given up.
    pub hole_wait: Duration,
}

impl Settings {
    pub fn from_env() -> Result<Settings> {
        let text = |name: &str, default: &str| std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string());
        let excluded = match std::env::var("PLANNER_EXCLUDE_FILE").ok().filter(|v| !v.is_empty()) {
            Some(path) => std::fs::read_to_string(&path)
                .with_context(|| format!("reading PLANNER_EXCLUDE_FILE {path}"))?
                .lines()
                .map(|line| line.trim().to_lowercase())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .collect(),
            None => HashSet::new(),
        };
        Ok(Settings {
            dir: PathBuf::from(text("PLANNER_DIR", "/home/publisher/planner")),
            pages_per_hour: number("PLANNER_PAGES_PER_HOUR", 100_000.0)?,
            pages_per_input: number("PLANNER_PAGES_PER_INPUT", 1_000)?,
            chars_per_input: number("PLANNER_CHARS_PER_INPUT", 3_000_000)?,
            flush_after: Duration::from_secs_f64(number("PLANNER_FLUSH_S", 600.0)?),
            max_pending: number("PLANNER_MAX_PENDING", 50)?,
            fresh: Some(Duration::from_secs_f64(number("PLANNER_FRESH_S", 48.0 * 3600.0)?)).filter(|window| !window.is_zero()),
            pending_max_age: Duration::from_secs_f64(number("PLANNER_PENDING_MAX_AGE_S", 12.0 * 3600.0)?),
            min_chars: number("PLANNER_MIN_CHARS", 300)?,
            langs: text("PLANNER_LANGS", "en").split(',').map(|lang| lang.trim().to_lowercase()).filter(|lang| !lang.is_empty()).collect(),
            excluded,
            list_file: std::env::var("PLANNER_LIST_FILE").ok().filter(|v| !v.is_empty()).map(PathBuf::from),
            set_file: std::env::var("PLANNER_SET_FILE").ok().filter(|v| !v.is_empty()).map(PathBuf::from),
            start_after: std::env::var("PLANNER_START_AFTER").ok().filter(|v| !v.is_empty()).map(|v| v.parse()).transpose()?,
            reads: number("PLANNER_READS", 8)?,
            page_first: text("PLANNER_PAGE_FIRST", "0") == "1",
            backfill: text("PLANNER_BACKFILL", "0") == "1",
            backfill_shards: match std::env::var("PLANNER_BACKFILL_SHARDS").ok().filter(|v| !v.is_empty()) {
                Some(range) => {
                    let (start, end) = range.split_once("..").ok_or_else(|| anyhow!("PLANNER_BACKFILL_SHARDS={range:?} is not a range a..b"))?;
                    start.trim().parse()?..end.trim().parse::<usize>()?.min(SHARDS)
                }
                None => 0..SHARDS,
            },
            current_prefix: text("PLANNER_CURRENT_PREFIX", "current/"),
            poll: Duration::from_secs_f64(number("PLANNER_POLL_S", 10.0)?),
            hole_wait: Duration::from_secs_f64(number("PLANNER_HOLE_WAIT_S", 600.0)?),
        })
    }
}

/// The sites of the bot's crawl list and how many pages of each the index keeps.
#[derive(Debug, Default)]
pub struct Listed {
    caps: HashMap<String, u64>,
}

impl Listed {
    /// The crawl list as the bot reads it: entries of `host` with a `send_cap`, three times the index cap, or the `backlog` of a news set.
    pub fn parse(raw: &[u8]) -> Result<Listed> {
        let entries: Vec<Value> = serde_json::from_slice(raw)?;
        let mut caps = HashMap::new();
        for entry in entries {
            let Some(host) = entry["host"].as_str() else { continue };
            // A news set rolls to its newest articles, so a cap on what the index holds would shut out fresh news.
            let cap = entry["send_cap"].as_u64().map_or(u64::MAX, |crawl| crawl / CRAWL_PER_INDEX);
            caps.insert(site_of(host), cap);
        }
        Ok(Listed { caps })
    }

    /// The index cap of the listed site a page's domain is, or is under.
    pub fn cap(&self, domain: &str) -> Option<u64> {
        let site = site_of(domain);
        let mut rest = site.as_str();
        loop {
            if let Some(&cap) = self.caps.get(rest) {
                return Some(cap);
            }
            rest = rest.split_once('.')?.1;
        }
    }

    pub fn len(&self) -> usize {
        self.caps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.caps.is_empty()
    }
}

/// A host as page keys name it: lowercase, without `www.`.
fn site_of(host: &str) -> String {
    let host = host.trim().to_lowercase();
    host.strip_prefix("www.").map(str::to_string).unwrap_or(host)
}

/// The domain a page key names, `pages/<domain>/<sha1>`.
fn domain_of_key(key: &str) -> &str {
    key.split('/').nth(1).unwrap_or("")
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name).ok().filter(|v| !v.is_empty()) {
        Some(value) => value.trim().parse().map_err(|_| anyhow!("{name}={value:?} is not a number")),
        None => Ok(default),
    }
}

/// Where the planner learns how many inputs the task API takes now.
pub enum Room {
    Api { url: String, http: reqwest::Client },
    Fixed(i64),
}

impl Room {
    pub fn api(task_api: &str) -> Result<Room> {
        let http = reqwest::Client::builder().timeout(ROOM_TIMEOUT).user_agent("desearch-planner").build()?;
        Ok(Room::Api { url: format!("{}/v1/room?kind=embed", task_api.trim_end_matches('/')), http })
    }

    async fn tasks(&self) -> Result<i64> {
        match self {
            Room::Fixed(room) => Ok(*room),
            Room::Api { url, http } => {
                let answer: Value = serde_json::from_slice(&http.get(url).send().await?.error_for_status()?.bytes().await?)?;
                answer["room_tasks"].as_i64().ok_or_else(|| anyhow!("{url} answered without room_tasks"))
            }
        }
    }
}

/// What the planner has planned, its place in the log and the inputs not yet handed over, kept on local disk.
pub struct State {
    db: DB,
}

impl State {
    pub fn open(dir: &Path) -> Result<State> {
        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_compression_type(DBCompressionType::Zstd);
        Ok(State { db: DB::open(&options, dir).with_context(|| format!("opening the planner state at {}", dir.display()))? })
    }

    pub fn cursor(&self) -> Result<Option<i64>> {
        Ok(self.db.get(CURSOR)?.map(|raw| String::from_utf8_lossy(&raw).parse()).transpose()?)
    }

    /// The content hash of the version of `key` last planned with `recipe`.
    pub fn planned(&self, recipe: &str, key: &str) -> Result<Option<String>> {
        Ok(self.planned_version(recipe, key)?.map(|(content_sha1, ..)| content_sha1))
    }

    /// The content hash and fetch time of the version of `key` last planned with `recipe`, and whether only its page text was; the time is empty for versions planned before it was kept.
    pub fn planned_version(&self, recipe: &str, key: &str) -> Result<Option<(String, String, bool)>> {
        Ok(self.db.get(planned_key(recipe, key))?.map(|raw| {
            let value = String::from_utf8_lossy(&raw);
            let mut fields = value.split('\t');
            let (content_sha1, fetched_at) = (fields.next().unwrap_or_default(), fields.next().unwrap_or_default());
            (content_sha1.to_string(), fetched_at.to_string(), fields.next() == Some(PAGE))
        }))
    }

    /// Whether no recipe has planned any version of `key`.
    pub fn unplanned(&self, key: &str) -> Result<bool> {
        for recipe in RECIPES {
            if self.db.get_pinned(planned_key(recipe, key))?.is_some() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// How many pages of `domain` some recipe has planned.
    pub fn site_pages(&self, domain: &str) -> Result<u64> {
        Ok(self.db.get(site_key(domain))?.map(|raw| String::from_utf8_lossy(&raw).parse()).transpose()?.unwrap_or(0))
    }

    /// Builds every domain's count of planned pages from the planned keys, once; later the counts are kept as pages are planned and forgotten.
    pub fn count_sites(&self) -> Result<()> {
        if self.db.get_pinned(SITES_COUNTED)?.is_some() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for (domain, count) in &self.sites_from_planned()? {
            batch.put(site_key(domain), count.to_string());
        }
        batch.put(SITES_COUNTED, "1");
        Ok(self.db.write(batch)?)
    }

    /// Every domain's count of pages some recipe has planned, from the planned keys.
    pub fn sites_from_planned(&self) -> Result<HashMap<String, u64>> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for (at, recipe) in RECIPES.iter().enumerate() {
            let prefix = format!("planned/{recipe}/");
            for item in self.db.iterator(IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward)) {
                let (raw, _) = item?;
                let Some(key) = raw.strip_prefix(prefix.as_bytes()) else { break };
                let key = String::from_utf8_lossy(key);
                let mut earlier = false;
                for before in &RECIPES[..at] {
                    if self.db.get_pinned(planned_key(before, &key))?.is_some() {
                        earlier = true;
                        break;
                    }
                }
                if !earlier {
                    *counts.entry(domain_of_key(&key).to_string()).or_default() += 1;
                }
            }
        }
        Ok(counts)
    }

    fn change_sites(&self, batch: &mut WriteBatch, changes: &HashMap<String, i64>) -> Result<()> {
        for (domain, change) in changes {
            let count = self.site_pages(domain)?.saturating_add_signed(*change);
            batch.put(site_key(domain), count.to_string());
        }
        Ok(())
    }

    /// Change files skipped because they could not be read yet, with the Unix time each was first missed.
    pub fn missing(&self) -> Result<Vec<(i64, f64)>> {
        let mut found = Vec::new();
        for item in self.db.iterator(IteratorMode::From(b"missing/", rocksdb::Direction::Forward)) {
            let (key, value) = item?;
            let Some(seq) = key.strip_prefix(b"missing/") else { break };
            let (Ok(seq), Ok(since)) = (String::from_utf8_lossy(seq).parse(), String::from_utf8_lossy(&value).parse()) else { continue };
            found.push((seq, since));
        }
        Ok(found)
    }

    fn note_missing(&self, seq: i64, since: f64) -> Result<()> {
        Ok(self.db.put(missing_key(seq), since.to_string())?)
    }

    fn settle_missing(&self, seqs: &[i64]) -> Result<()> {
        let mut batch = WriteBatch::default();
        for seq in seqs {
            batch.delete(missing_key(*seq));
        }
        Ok(self.db.write(batch)?)
    }

    pub fn pending(&self) -> Result<Vec<(String, String)>> {
        let mut found = Vec::new();
        for item in self.db.iterator(IteratorMode::From(b"pending/", rocksdb::Direction::Forward)) {
            let (key, value) = item?;
            let Some(input_key) = key.strip_prefix(b"pending/") else { break };
            found.push((String::from_utf8_lossy(input_key).into_owned(), String::from_utf8_lossy(&value).into_owned()));
        }
        Ok(found)
    }

    fn handed_over(&self, input_key: &str) -> Result<()> {
        Ok(self.db.delete(format!("pending/{input_key}"))?)
    }

    /// Drops an input never handed over; its pages are planned again the next time the log shows them.
    fn expire(&self, input_key: &str, entry: &Value) -> Result<()> {
        let mut batch = self.forgetting(&entry["pages"])?;
        batch.delete(format!("pending/{input_key}"));
        Ok(self.db.write(batch)?)
    }

    /// Forgets the versions of an input's pages that never got vectors, so the log's next showing plans them again.
    fn forget(&self, pages: &Value) -> Result<()> {
        Ok(self.db.write(self.forgetting(pages)?)?)
    }

    fn forgetting(&self, pages: &Value) -> Result<WriteBatch> {
        let mut batch = WriteBatch::default();
        let mut gone: HashMap<String, i64> = HashMap::new();
        for page in pages.as_array().into_iter().flatten() {
            let (Some(key), Some(content_sha1)) = (page["page_key"].as_str(), page["content_sha1"].as_str()) else { continue };
            let (mut forgotten, mut kept) = (false, false);
            for recipe in RECIPES {
                match self.planned(recipe, key)? {
                    Some(planned) if planned == content_sha1 => {
                        batch.delete(planned_key(recipe, key));
                        forgotten = true;
                    }
                    Some(_) => kept = true,
                    None => {}
                }
            }
            if forgotten && !kept {
                *gone.entry(domain_of_key(key).to_string()).or_default() -= 1;
            }
        }
        self.change_sites(&mut batch, &gone)?;
        Ok(batch)
    }

    /// The shard the backfill reads and the last key of it planned or passed by; None before it starts.
    pub fn backfill(&self) -> Result<Option<(usize, Option<String>)>> {
        let Some(raw) = self.db.get(BACKFILL)? else { return Ok(None) };
        let raw = String::from_utf8_lossy(&raw).into_owned();
        let (shard, after) = raw.split_once('\t').unwrap_or((&raw, ""));
        Ok(Some((shard.parse()?, Some(after.to_string()).filter(|after| !after.is_empty()))))
    }

    fn keep_backfill(&self, shard: usize, after: Option<&str>) -> Result<()> {
        Ok(self.db.put(BACKFILL, format!("{shard}\t{}", after.unwrap_or("")))?)
    }

    fn keep_cursor(&self, seq: i64) -> Result<()> {
        Ok(self.db.put(CURSOR, seq.to_string())?)
    }

    /// One written input: its pages planned, the input pending, the cursor where the log is read up to and the late files read, together.
    fn commit(&self, pages: &[Planned], input_key: &str, entry: &str, cursor: Option<i64>, settled: &[i64]) -> Result<()> {
        let mut batch = WriteBatch::default();
        let mut new: HashMap<String, i64> = HashMap::new();
        for page in pages {
            if self.unplanned(&page.key)? {
                *new.entry(domain_of_key(&page.key).to_string()).or_default() += 1;
            }
            let page_only = if page.page_only { "\tpage" } else { "" };
            batch.put(planned_key(page.recipe, &page.key), format!("{}\t{}{page_only}", page.content_sha1, page.fetched_at));
        }
        self.change_sites(&mut batch, &new)?;
        batch.put(format!("pending/{input_key}"), entry);
        if let Some(seq) = cursor {
            batch.put(CURSOR, seq.to_string());
        }
        for seq in settled {
            batch.delete(missing_key(*seq));
        }
        Ok(self.db.write(batch)?)
    }
}

fn planned_key(recipe: &str, key: &str) -> String {
    format!("planned/{recipe}/{key}")
}

fn site_key(domain: &str) -> String {
    format!("sites/{domain}")
}

fn missing_key(seq: i64) -> String {
    format!("missing/{seq:020}")
}

/// Why a change file was passed by: its number is not indexed yet, or the file could not be read or decoded.
enum Unread {
    NotNumbered,
    Failed(String),
}

/// One page waiting in the input being filled.
#[derive(Clone, Debug)]
pub struct Planned {
    pub key: String,
    pub url: String,
    pub host: String,
    pub content_sha1: String,
    pub recipe: &'static str,
    pub fetched_at: String,
    pub texts: Vec<Text>,
    /// Only the page text of a page that has passages; they are planned when the page is next seen with `page_first` off.
    pub page_only: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Counts {
    pub files: u64,
    pub pages: u64,
    pub planned: u64,
    /// Pages whose text no planned recipe reads; they keep the vectors they have.
    pub other_recipe: u64,
    pub refused: u64,
    /// Pages of sites the crawl list does not hold.
    pub unlisted: u64,
    /// Pages the set has no rule taking: an unlisted or shut host, or off a whole site's paths.
    pub outside_set: u64,
    /// Pages of a news site with no date in `published` or the URL path.
    pub undated: u64,
    /// Pages of a news site dated before its `since`.
    pub before_since: u64,
    /// New pages of sites whose index cap is reached.
    pub over_cap: u64,
    pub unchanged: u64,
    /// Versions older than the one already planned, from change files read late.
    pub older: u64,
    /// Change files read after later ones, because they were numbered late.
    pub late: u64,
    pub over_budget: u64,
    pub no_room: u64,
    /// Pages planned in the fresh lane, of `planned`.
    pub fresh: u64,
    /// Pages planned with their page text alone, of `planned`.
    pub page_only: u64,
    /// Pages planned for their passages alone, their page text planned before, of `planned`.
    pub parts: u64,
    /// Pages read from `current/` by the backfill, of `pages`.
    pub backfill: u64,
    pub shards: u64,
    pub inputs: u64,
    pub handed_over: u64,
    pub expired: u64,
    pub dropped: u64,
}

/// Pages dated within `Settings::fresh` go ahead of the rest and outside the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Fresh,
    Rest,
}

impl Lane {
    const ALL: [Lane; 2] = [Lane::Fresh, Lane::Rest];

    fn of(entry: &Value) -> Lane {
        if entry["lane"] == "fresh" {
            Lane::Fresh
        } else {
            Lane::Rest
        }
    }
}

/// A shard of `current/` being read for the backfill, its pages arriving as they are read.
struct Reading {
    shard: usize,
    pages: mpsc::Receiver<PageRow>,
    done: tokio::task::JoinHandle<Result<usize>>,
    /// The last key taken from it.
    last: Option<String>,
}

/// The pages of one lane waiting to be written as an input.
#[derive(Default)]
struct Buffer {
    pages: Vec<Planned>,
    at: HashMap<String, usize>,
    /// Pages no recipe has planned before, by domain, so the buffer counts toward index caps.
    new: HashMap<String, u64>,
    chars: usize,
    since: Option<Instant>,
}

pub struct Planner {
    pub settings: Settings,
    pub state: State,
    pub pages: Bucket,
    pub temp: Bucket,
    pub redis: ConnectionManager,
    pub room: Room,
    /// Where the backfill reads `current/`.
    pub current: Option<Bucket>,
    pub counts: Counts,
    listed: Option<Listed>,
    listed_modified: Option<std::time::SystemTime>,
    set: Option<Arc<Set>>,
    reading: Option<Reading>,
    set_modified: Option<std::time::SystemTime>,
    fresh: Buffer,
    rest: Buffer,
    read_through: i64,
    tokens: f64,
    refilled: Instant,
    /// Inputs waiting for room, by lane.
    pending: [usize; 2],
    missing: BTreeMap<i64, Instant>,
    /// Late files read or given up since the buffer was last empty; forgotten with the next commit.
    settled: Vec<i64>,
}

impl Planner {
    pub async fn new(settings: Settings, pages: Bucket, temp: Bucket, redis: ConnectionManager, room: Room) -> Result<Planner> {
        std::fs::create_dir_all(&settings.dir)?;
        let state = State::open(&settings.dir)?;
        let started = Instant::now();
        state.count_sites()?;
        if started.elapsed() > Duration::from_secs(1) {
            println!("planner: counted every domain's planned pages in {:?}", started.elapsed());
        }
        let read_through = match (state.cursor()?, settings.start_after) {
            (Some(seq), _) | (None, Some(seq)) => seq,
            (None, None) => latest(&pages).await?,
        };
        state.keep_cursor(read_through)?;
        let pending = pending_by_lane(&state.pending()?);
        let missing = state
            .missing()?
            .into_iter()
            .map(|(seq, since)| {
                (seq, Instant::now().checked_sub(Duration::from_secs_f64((now_us() as f64 / 1e6 - since).max(0.0))).unwrap_or_else(Instant::now))
            })
            .collect();
        let tokens = budget_cap(settings.pages_per_hour);
        let mut planner = Planner {
            settings,
            state,
            pages,
            temp,
            redis,
            room,
            current: None,
            counts: Counts::default(),
            listed: None,
            listed_modified: None,
            set: None,
            reading: None,
            set_modified: None,
            fresh: Buffer::default(),
            rest: Buffer::default(),
            read_through,
            tokens,
            refilled: Instant::now(),
            pending,
            missing,
            settled: Vec::new(),
        };
        if planner.settings.set_file.is_some() {
            if !planner.reload_set()? {
                return Err(anyhow!("the set in PLANNER_SET_FILE is empty"));
            }
        } else if planner.settings.list_file.is_some() && !planner.reload_list()? {
            return Err(anyhow!("the crawl list in PLANNER_LIST_FILE is empty"));
        }
        Ok(planner)
    }

    /// Reads the set again when its file changed; whether a set is held. A set that cannot be read leaves the one held.
    pub fn reload_set(&mut self) -> Result<bool> {
        let Some(path) = self.settings.set_file.clone() else { return Ok(false) };
        let modified = std::fs::metadata(&path).and_then(|meta| meta.modified()).with_context(|| format!("reading {}", path.display()))?;
        if self.set.is_some() && self.set_modified == Some(modified) {
            return Ok(true);
        }
        match std::fs::read(&path).map_err(anyhow::Error::from).and_then(|raw| Set::parse(&raw)) {
            Ok(set) if !set.is_empty() => {
                println!("planner: set of {} sites read from {}", set.len(), path.display());
                self.set = Some(Arc::new(set));
                self.set_modified = Some(modified);
            }
            Ok(_) => eprintln!("{} holds no sites; keeping the set held", path.display()),
            Err(error) => eprintln!("reading the set {}: {error:#}; keeping the set held", path.display()),
        }
        Ok(self.set.is_some())
    }

    /// Reads the crawl list again when its file changed; whether a list is held. A list that cannot be read leaves the one held.
    pub fn reload_list(&mut self) -> Result<bool> {
        let Some(path) = self.settings.list_file.clone() else { return Ok(false) };
        let modified = std::fs::metadata(&path).and_then(|meta| meta.modified()).with_context(|| format!("reading {}", path.display()))?;
        if self.listed.is_some() && self.listed_modified == Some(modified) {
            return Ok(true);
        }
        match std::fs::read(&path).map_err(anyhow::Error::from).and_then(|raw| Listed::parse(&raw)) {
            Ok(listed) if !listed.is_empty() => {
                println!("planner: crawl list of {} sites read from {}", listed.len(), path.display());
                self.listed = Some(listed);
                self.listed_modified = Some(modified);
            }
            Ok(_) => eprintln!("{} holds no sites; keeping the list held", path.display()),
            Err(error) => eprintln!("reading the crawl list {}: {error:#}; keeping the list held", path.display()),
        }
        Ok(self.listed.is_some())
    }

    /// Reads what the log added since the last pass, writes the inputs that are due and hands over what the API has room for; whether it read anything.
    pub async fn pass(&mut self) -> Result<bool> {
        let reloaded = if self.settings.set_file.is_some() { self.reload_set() } else { self.reload_list() };
        if let Err(error) = reloaded {
            eprintln!("{error:#}");
        }
        self.forget_dropped().await?;
        let latest = latest(&self.pages).await?;
        let mut read = self.read_late().await?;
        let ahead: Vec<i64> = (self.read_through + 1..=latest).take(FILES_PER_PASS.saturating_sub(read)).collect();
        let pages = self.pages.clone();
        let mut fetched = stream::iter(ahead).map(|seq| fetch(pages.clone(), seq).map(move |rows| (seq, rows))).buffered(self.settings.reads.max(1));
        while let Some((seq, rows)) = fetched.next().await {
            match rows? {
                Ok(rows) => {
                    self.take(rows).await?;
                    read += 1;
                }
                Err(why) => self.note_missing(seq, &why)?,
            }
            self.advance(seq)?;
        }
        let backfilled = self.backfill().await?;
        for lane in Lane::ALL {
            if self.buffer(lane).since.is_some_and(|since| since.elapsed() >= self.settings.flush_after) {
                self.flush_lane(lane).await?;
            }
        }
        self.hand_over().await?;
        Ok(read > 0 || backfilled)
    }

    /// Plans pages of `current/` while the budget and the rest lane's room last; whether it took any.
    async fn backfill(&mut self) -> Result<bool> {
        if !self.settings.backfill || self.current.is_none() {
            return Ok(false);
        }
        let mut took = false;
        loop {
            if self.pending[Lane::Rest as usize] >= self.settings.max_pending || !self.has_token() {
                return Ok(took);
            }
            if self.reading.is_none() && !self.start_reading()? {
                return Ok(took);
            }
            let reading = self.reading.as_mut().expect("a shard being read");
            match tokio::time::timeout(BACKFILL_WAIT, reading.pages.recv()).await {
                Err(_) => return Ok(took),
                Ok(Some(page)) => {
                    reading.last = Some(page.key.clone());
                    took = true;
                    self.counts.backfill += 1;
                    if let Some(lane) = self.consider(page) {
                        let buffer = self.buffer(lane);
                        if buffer.pages.len() >= self.settings.pages_per_input || buffer.chars >= self.settings.chars_per_input {
                            self.flush_lane(lane).await?;
                        }
                    }
                }
                Ok(None) => {
                    let reading = self.reading.take().expect("a shard being read");
                    let shard = reading.shard;
                    // A shard that failed is read again from the last place kept.
                    reading.done.await?.with_context(|| format!("reading shard {shard} of {}", self.settings.current_prefix))?;
                    self.flush().await?;
                    self.state.keep_backfill(shard + 1, None)?;
                    self.counts.shards += 1;
                    if shard + 1 == self.settings.backfill_shards.end {
                        println!("planner: the backfill read its last shard, {shard}");
                    }
                }
            }
        }
    }

    /// Starts reading the backfill's next shard from where it was left; false once every shard is read.
    fn start_reading(&mut self) -> Result<bool> {
        let shards = self.settings.backfill_shards.clone();
        let (shard, after) = self.state.backfill()?.unwrap_or((shards.start, None));
        let (shard, after) = if shard < shards.start { (shards.start, None) } else { (shard, after) };
        if shard >= shards.end {
            return Ok(false);
        }
        let reader = ShardReader {
            bucket: self.current.clone().expect("a current/ bucket"),
            prefix: self.settings.current_prefix.clone(),
            handle: Handle::current(),
            ranges_at_once: RANGES_AT_ONCE,
        };
        let (send, pages) = mpsc::channel(BACKFILL_AHEAD);
        let (set, langs, from) = (self.set.clone(), self.settings.langs.clone(), after.clone());
        let done = tokio::task::spawn_blocking(move || {
            let keep = |row: &PageRow| {
                let record = &row.record;
                recipe_of(record).is_some_and(|recipe| PLANNED_RECIPES.contains(&recipe))
                    && lang_wanted(&langs, &record.lang)
                    && set.as_ref().is_none_or(|set| set.admit(&record.url, &record.published).is_ok())
            };
            reader.read(shard, from.as_deref(), keep, |page| send.blocking_send(page).is_ok())
        });
        self.reading = Some(Reading { shard, pages, done, last: after });
        Ok(true)
    }

    /// Change files missed on the way past: read once they can be; a number never indexed is given up after `hole_wait`, a file that cannot be read is tried until it can.
    async fn read_late(&mut self) -> Result<usize> {
        let mut read = 0;
        for (seq, since) in self.missing.clone() {
            match self.read(seq).await? {
                Ok(()) => {
                    self.counts.late += 1;
                    read += 1;
                }
                Err(Unread::NotNumbered) if since.elapsed() >= self.settings.hole_wait => {
                    eprintln!("change file {seq} was never numbered; going on without it")
                }
                Err(_) => continue,
            }
            self.missing.remove(&seq);
            self.settled.push(seq);
        }
        if self.buffered_nothing() && !self.settled.is_empty() {
            self.state.settle_missing(&self.settled)?;
            self.settled.clear();
        }
        Ok(read)
    }

    /// Considers every page of change file `seq`; the inner error says why the file cannot be read yet, the outer one fails the pass.
    async fn read(&mut self, seq: i64) -> Result<std::result::Result<(), Unread>> {
        match fetch(self.pages.clone(), seq).await? {
            Ok(rows) => self.take(rows).await.map(Ok),
            Err(why) => Ok(Err(why)),
        }
    }

    /// Considers every page of one change file, in file order.
    async fn take(&mut self, rows: Vec<ChangeRow>) -> Result<()> {
        for row in rows {
            if let ChangeRow::Page(page) = row {
                if let Some(lane) = self.consider(*page) {
                    let buffer = self.buffer(lane);
                    if buffer.pages.len() >= self.settings.pages_per_input || buffer.chars >= self.settings.chars_per_input {
                        self.flush_lane(lane).await?;
                    }
                }
            }
        }
        self.counts.files += 1;
        Ok(())
    }

    fn note_missing(&mut self, seq: i64, why: &Unread) -> Result<()> {
        if let Unread::Failed(why) = why {
            eprintln!("change file {seq} skipped for now: {why}");
        }
        if let btree_map::Entry::Vacant(slot) = self.missing.entry(seq) {
            slot.insert(Instant::now());
            self.state.note_missing(seq, now_us() as f64 / 1e6)?;
        }
        Ok(())
    }

    /// Takes back the pages of inputs the task API dropped after their last attempt.
    pub async fn forget_dropped(&mut self) -> Result<usize> {
        let dropped: Option<Vec<String>> = redis::cmd("LPOP").arg(EMBED_DROPPED).arg(FILES_PER_PASS * 10).query_async(&mut self.redis).await?;
        let dropped = dropped.unwrap_or_default();
        for raw in &dropped {
            let entry: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
            self.state.forget(&entry["pages"])?;
        }
        self.counts.dropped += dropped.len() as u64;
        Ok(dropped.len())
    }

    /// Plans until `stop` turns true, writing what is buffered first.
    pub async fn run(&mut self, mut stop: watch::Receiver<bool>) -> Result<()> {
        let mut reported = Instant::now();
        while !*stop.borrow() {
            let busy = match self.pass().await {
                Ok(busy) => busy,
                Err(error) => {
                    eprintln!("planner pass failed: {error:#}");
                    false
                }
            };
            if reported.elapsed() >= Duration::from_secs(60) {
                println!("planner: through change file {}, {:?}", self.read_through, self.counts);
                reported = Instant::now();
            }
            if !busy {
                let _ = tokio::time::timeout(self.settings.poll, stop.changed()).await;
            }
        }
        self.flush().await?;
        Ok(())
    }

    /// Marks change file `seq` read; the kept cursor moves only while nothing is buffered, so a restart reads the buffered pages' files again.
    fn advance(&mut self, seq: i64) -> Result<()> {
        self.read_through = seq;
        if self.buffered_nothing() {
            self.state.keep_cursor(seq)?;
        }
        Ok(())
    }

    /// Takes the page into its lane's buffer if a planned recipe reads it, it is wanted, changed, and the budget and the API's room allow it; the lane it went in.
    pub fn consider(&mut self, page: PageRow) -> Option<Lane> {
        self.counts.pages += 1;
        let record = &page.record;
        let Some(recipe) = recipe_of(record).filter(|recipe| PLANNED_RECIPES.contains(recipe)) else {
            self.counts.other_recipe += 1;
            return None;
        };
        if !self.admitted(record) {
            self.counts.refused += 1;
            return None;
        }
        if let Some(Err(refusal)) = self.set.as_ref().map(|set| set.admit(&record.url, &record.published)) {
            match refusal {
                Refusal::Undated => self.counts.undated += 1,
                Refusal::BeforeSince => self.counts.before_since += 1,
                Refusal::Unlisted | Refusal::OffPaths => self.counts.outside_set += 1,
            }
            return None;
        }
        let cap = match self.listed.as_ref().filter(|_| self.set.is_none()) {
            None => None,
            Some(listed) => match listed.cap(&record.domain) {
                Some(cap) => Some(cap),
                None => {
                    self.counts.unlisted += 1;
                    return None;
                }
            },
        };
        let buffered = Lane::ALL.into_iter().find_map(|lane| self.buffer(lane).at.get(&page.key).map(|&at| (lane, at)));
        let already = match buffered {
            Some((lane, at)) => {
                let held = &self.buffer(lane).pages[at];
                Some((held.content_sha1.clone(), held.fetched_at.clone(), held.page_only))
            }
            None => self.state.planned_version(recipe, &page.key).ok().flatten(),
        };
        let mut parts_only = false;
        if let Some((content_sha1, fetched_at, page_only)) = &already {
            if *content_sha1 == record.content_sha1 {
                if !page_only || self.settings.page_first || buffered.is_some() {
                    self.counts.unchanged += 1;
                    return None;
                }
                parts_only = true;
            }
            // Times are `iso` text of one width, so they order as text; a late file can hold an older version than the one planned.
            if record.fetched_at < *fetched_at {
                self.counts.older += 1;
                return None;
            }
        }
        if let Some((lane, at)) = buffered {
            let planned = planned_of(&page, recipe, self.settings.page_first && lane == Lane::Rest, false);
            let buffer = self.buffer_mut(lane);
            buffer.chars = buffer.chars - chars_of(&buffer.pages[at]) + chars_of(&planned);
            buffer.pages[at] = planned;
            return Some(lane);
        }
        let new_in = match cap {
            Some(cap) if self.state.unplanned(&page.key).unwrap_or(false) => {
                let domain = domain_of_key(&page.key).to_string();
                let buffered: u64 = Lane::ALL.into_iter().map(|lane| self.buffer(lane).new.get(&domain).copied().unwrap_or(0)).sum();
                let held = self.state.site_pages(&domain).unwrap_or(0) + buffered;
                if held >= cap {
                    self.counts.over_cap += 1;
                    return None;
                }
                Some(domain)
            }
            _ => None,
        };
        let lane = if self.is_fresh(record) { Lane::Fresh } else { Lane::Rest };
        if self.pending[lane as usize] >= self.settings.max_pending {
            self.counts.no_room += 1;
            return None;
        }
        if lane == Lane::Rest && !self.take_token() {
            self.counts.over_budget += 1;
            return None;
        }
        let planned = planned_of(&page, recipe, self.settings.page_first && lane == Lane::Rest, parts_only);
        if planned.texts.is_empty() {
            self.counts.unchanged += 1;
            return None;
        }
        self.counts.page_only += u64::from(planned.page_only);
        self.counts.parts += u64::from(parts_only);
        let buffer = self.buffer_mut(lane);
        if let Some(domain) = new_in {
            *buffer.new.entry(domain).or_default() += 1;
        }
        buffer.at.insert(page.key, buffer.pages.len());
        buffer.chars += chars_of(&planned);
        buffer.pages.push(planned);
        buffer.since.get_or_insert_with(Instant::now);
        self.counts.planned += 1;
        self.counts.fresh += u64::from(lane == Lane::Fresh);
        Some(lane)
    }

    /// Whether the page is dated within the fresh window, by `published` or its URL, and not past tomorrow.
    fn is_fresh(&self, record: &Record) -> bool {
        let Some(window) = self.settings.fresh else { return false };
        let now = now_us() as f64 / 1e6;
        page_date(&record.published, &record.url).is_some_and(|date| date >= utc_day(now - window.as_secs_f64()) && date <= utc_day(now + 86_400.0))
    }

    fn buffer(&self, lane: Lane) -> &Buffer {
        match lane {
            Lane::Fresh => &self.fresh,
            Lane::Rest => &self.rest,
        }
    }

    fn buffer_mut(&mut self, lane: Lane) -> &mut Buffer {
        match lane {
            Lane::Fresh => &mut self.fresh,
            Lane::Rest => &mut self.rest,
        }
    }

    fn buffered_nothing(&self) -> bool {
        self.fresh.pages.is_empty() && self.rest.pages.is_empty()
    }

    fn admitted(&self, record: &Record) -> bool {
        lang_wanted(&self.settings.langs, &record.lang)
            && record.text.chars().count() >= self.settings.min_chars
            && !self.excluded(&record.domain.to_lowercase())
    }

    /// Whether the host or a domain it is under is on the exclusion list.
    fn excluded(&self, host: &str) -> bool {
        let mut rest = host;
        loop {
            if self.settings.excluded.contains(rest) {
                return true;
            }
            match rest.split_once('.') {
                Some((_, parent)) => rest = parent,
                None => return false,
            }
        }
    }

    fn take_token(&mut self) -> bool {
        if !self.has_token() {
            return false;
        }
        self.tokens -= 1.0;
        true
    }

    fn has_token(&mut self) -> bool {
        let cap = budget_cap(self.settings.pages_per_hour);
        self.tokens = (self.tokens + self.refilled.elapsed().as_secs_f64() * self.settings.pages_per_hour / 3600.0).min(cap);
        self.refilled = Instant::now();
        self.tokens >= 1.0
    }

    /// Writes every lane's buffer.
    pub async fn flush(&mut self) -> Result<()> {
        for lane in Lane::ALL {
            self.flush_lane(lane).await?;
        }
        Ok(())
    }

    /// Writes a lane's buffer as one input in the temp bucket and keeps it pending until the API has room.
    async fn flush_lane(&mut self, lane: Lane) -> Result<()> {
        let buffer = self.buffer(lane);
        if buffer.pages.is_empty() {
            return Ok(());
        }
        let rows: Vec<(&Planned, &Text)> = buffer.pages.iter().flat_map(|page| page.texts.iter().map(move |text| (page, text))).collect();
        let file = encode_input(&rows)?;
        let input_key = format!("embed/inputs/{}.parquet", uuid::Uuid::new_v4().simple());
        let mut entry = json!({
            "input_key": input_key,
            "input_sha256": hex(&Sha256::digest(&file)),
            "written_at": now_us() as f64 / 1e6,
            "texts": rows.len(),
            "chars": rows.iter().map(|(_, text)| text.text.chars().count()).sum::<usize>(),
            "pages": buffer.pages.iter().map(|p| json!({"page_key": p.key, "content_sha1": p.content_sha1, "url": p.url, "host": p.host})).collect::<Vec<_>>(),
        });
        if lane == Lane::Fresh {
            entry["lane"] = "fresh".into();
        }
        // Kept until the input is stored, so a failed write is tried again with the same pages.
        self.temp.put(&input_key, Bytes::from(file), PARQUET, None).await.map_err(|error| anyhow!("writing {input_key}: {error}"))?;
        // The cursor and late files move only with the last lane written, so a restart reads the other lane's pages again.
        let alone = Lane::ALL.into_iter().all(|other| other == lane || self.buffer(other).pages.is_empty());
        let (cursor, settled) = if alone { (Some(self.read_through), std::mem::take(&mut self.settled)) } else { (None, Vec::new()) };
        self.state.commit(&self.buffer(lane).pages, &input_key, &entry.to_string(), cursor, &settled)?;
        if let Some(reading) = self.reading.as_ref().filter(|_| alone) {
            self.state.keep_backfill(reading.shard, reading.last.as_deref())?;
        }
        *self.buffer_mut(lane) = Buffer::default();
        self.pending[lane as usize] += 1;
        self.counts.inputs += 1;
        Ok(())
    }

    /// Hands pending inputs to the task API, oldest first, as many as it has room for.
    pub async fn hand_over(&mut self) -> Result<usize> {
        let pending = self.state.pending()?;
        self.pending = pending_by_lane(&pending);
        if pending.is_empty() {
            return Ok(0);
        }
        let room = match self.room.tasks().await {
            Ok(room) => room.max(0) as usize,
            Err(error) => {
                eprintln!("the task API's embed room is unavailable: {error:#}");
                return Ok(0);
            }
        };
        let now = now_us() as f64 / 1e6;
        let mut pending: Vec<(String, String, Value)> = pending
            .into_iter()
            .map(|(input_key, entry)| {
                let parsed = serde_json::from_str::<Value>(&entry).unwrap_or(Value::Null);
                (input_key, entry, parsed)
            })
            .collect();
        let written = |entry: &Value| entry["written_at"].as_f64().unwrap_or(now);
        pending.sort_by(|a, b| (Lane::of(&a.2) as usize).cmp(&(Lane::of(&b.2) as usize)).then(written(&a.2).total_cmp(&written(&b.2))));
        let (mut handed, mut expired) = (0, 0);
        for (input_key, entry, parsed) in pending {
            let lane = Lane::of(&parsed) as usize;
            if now - parsed["written_at"].as_f64().unwrap_or(now) > self.settings.pending_max_age.as_secs_f64() {
                eprintln!("{input_key} waited longer than {:?} for room; dropped", self.settings.pending_max_age);
                self.state.expire(&input_key, &parsed)?;
                self.pending[lane] -= 1;
                expired += 1;
                continue;
            }
            if handed == room {
                break;
            }
            let _: i64 = self.redis.rpush(EMBED_INPUTS, &entry).await?;
            self.state.handed_over(&input_key)?;
            self.pending[lane] -= 1;
            handed += 1;
        }
        self.counts.handed_over += handed as u64;
        self.counts.expired += expired as u64;
        Ok(handed)
    }

    pub fn read_through(&self) -> i64 {
        self.read_through
    }
}

/// The page as the buffer holds it, its texts made only once it is taken: all of them, its page text alone, or its passages alone.
fn planned_of(page: &PageRow, recipe: &'static str, page_first: bool, parts_only: bool) -> Planned {
    let record = &page.record;
    let mut texts = texts(&page.key, record);
    let page_only = page_first && !parts_only && texts.len() > 1;
    if page_only {
        texts.truncate(1);
    } else if parts_only {
        texts.remove(0);
    }
    Planned {
        key: page.key.clone(),
        url: record.url.clone(),
        host: record.domain.clone(),
        content_sha1: record.content_sha1.clone(),
        recipe,
        fetched_at: record.fetched_at.clone(),
        texts,
        page_only,
    }
}

/// Change file `seq`'s rows, fetched and decoded; the inner error says why the file cannot be read yet.
async fn fetch(pages: Bucket, seq: i64) -> Result<std::result::Result<Vec<ChangeRow>, Unread>> {
    let key = match change_key(&pages, seq).await {
        Ok(Some(key)) => key,
        Ok(None) => return Ok(Err(Unread::NotNumbered)),
        Err(error) => return Ok(Err(Unread::Failed(format!("{error:#}")))),
    };
    match pages.get(&key).await {
        Ok(file) => match tokio::task::spawn_blocking(move || decode_changes(file, seq)).await? {
            Ok(rows) => Ok(Ok(rows)),
            Err(error) => Ok(Err(Unread::Failed(format!("decoding {key}: {error:#}")))),
        },
        Err(r2::Error::Missing) => {
            eprintln!("change file {seq} ({key}) is gone; going on without it");
            Ok(Ok(Vec::new()))
        }
        Err(error) => Ok(Err(Unread::Failed(format!("reading {key}: {error}")))),
    }
}

fn lang_wanted(langs: &[String], lang: &str) -> bool {
    let lang = lang.to_lowercase();
    langs.iter().any(|want| lang == *want || lang.starts_with(&format!("{want}-")) || lang.starts_with(&format!("{want}_")))
}

fn pending_by_lane(pending: &[(String, String)]) -> [usize; 2] {
    let mut counts = [0; 2];
    for (_, entry) in pending {
        counts[Lane::of(&serde_json::from_str(entry).unwrap_or(Value::Null)) as usize] += 1;
    }
    counts
}

fn chars_of(page: &Planned) -> usize {
    page.texts.iter().map(|text| text.text.chars().count()).sum()
}

fn budget_cap(pages_per_hour: f64) -> f64 {
    (pages_per_hour / 4.0).max(1.0)
}

async fn latest(pages: &Bucket) -> Result<i64> {
    let raw = pages.get(&CHANGES.latest_key()).await.map_err(|error| anyhow!("reading {}: {error}", CHANGES.latest_key()))?;
    let latest: Value = serde_json::from_slice(&raw)?;
    latest["seq"].as_i64().ok_or_else(|| anyhow!("{} has no seq", CHANGES.latest_key()))
}

async fn change_key(pages: &Bucket, seq: i64) -> Result<Option<String>> {
    match pages.get(&CHANGES.seq_key(seq as u64)).await {
        Ok(raw) => Ok(serde_json::from_slice::<Value>(&raw)?["key"].as_str().map(String::from)),
        Err(r2::Error::Missing) => Ok(None),
        Err(error) => Err(anyhow!("reading the index of change file {seq}: {error}")),
    }
}

pub fn encode_input(rows: &[(&Planned, &Text)]) -> Result<Vec<u8>> {
    let strings = |pick: &dyn Fn(&(&Planned, &Text)) -> String| -> ArrayRef { Arc::new(StringArray::from(rows.iter().map(pick).collect::<Vec<_>>())) };
    let columns = vec![
        strings(&|(_, text)| text.text_id.clone()),
        strings(&|(page, _)| page.key.clone()),
        strings(&|(page, _)| page.url.clone()),
        strings(&|(page, _)| page.content_sha1.clone()),
        strings(&|(_, text)| text.kind.to_string()),
        Arc::new(Int32Array::from(rows.iter().map(|(_, text)| text.index).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(LargeStringArray::from(rows.iter().map(|(_, text)| text.text.as_str()).collect::<Vec<_>>())) as ArrayRef,
        strings(&|(page, _)| page.recipe.to_string()),
    ];
    let batch = RecordBatch::try_new(INPUT_SCHEMA.clone(), columns)?;
    let properties = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, INPUT_SCHEMA.clone(), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

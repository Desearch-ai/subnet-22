//! The planner: follows the change log and turns published pages that need a vector into embed inputs for the task API, within an hourly page budget and the room the API gives.

use std::collections::{btree_map, BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
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
use redis::{AsyncCommands, Script};
use rocksdb::{DBCompressionType, IteratorMode, Options, WriteBatch, DB};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};

use crate::backfill::ShardReader;
use crate::current::{decode_changes, ChangeRow, PageRow, SHARDS};
use crate::records::{iso, parse_iso, Record, EXTRACTION_V2};
use crate::service::now_us;
use crate::set::{iso_date, url_date, Admitted, Refusal, Set};

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
/// How many pages wait, kept with every change to them.
const WAITING_COUNT: &[u8] = b"waiting-count";
/// The backfill's shard and the last key of it planned or passed by, `<shard>\t<key>`.
const BACKFILL: &[u8] = b"backfill";
/// Pages read ahead from `current/` waiting for budget and room.
const BACKFILL_AHEAD: usize = 4_096;
/// How long a pass waits for the backfill's reader before going on.
const BACKFILL_WAIT: Duration = Duration::from_millis(500);
const RANGES_AT_ONCE: usize = 16;
/// Kept past any input's wait for room, so a push tried again after a crash is not pushed twice.
const PUSHED_TTL_S: u64 = 2 * 86_400;

/// Pushes an input once, however often it is tried.
static PUSH: LazyLock<Script> =
    LazyLock::new(|| Script::new("if redis.call('SET', KEYS[2], '1', 'NX', 'EX', ARGV[2]) then redis.call('RPUSH', KEYS[1], ARGV[1]) return 1 end return 0"));
/// Set once every domain's count of planned pages has been built from the planned keys.
const SITES_COUNTED: &[u8] = b"sites-counted";
/// The crawl list's caps are crawl caps, three pages crawled for each one the index keeps.
const CRAWL_PER_INDEX: u64 = 3;
const ROOM_TIMEOUT: Duration = Duration::from_secs(20);
/// After an answer that left inputs waiting, or none, inputs are held this long before the API is asked again mid-pass.
const ROOM_RETRY: Duration = Duration::from_secs(30);
const FILES_PER_PASS: usize = 64;
/// Set when no page waits without a change file to plan it from.
const WAITING_NUMBERED: &[u8] = b"waiting-numbered";
const UNWAIT_BATCH: i64 = 100_000;
/// A set older than this is warned about, as its pull may have stopped.
const SET_STALE: Duration = Duration::from_secs(2 * 86_400);
const SET_WARN_EVERY: Duration = Duration::from_secs(3600);
/// Change files of waiting pages read a pass, so a backlog of them cannot hold up the log.
const REPLAN_FILES_PER_PASS: usize = 64;

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
    /// Pages kept waiting at most, so the state stays bounded while the log brings more than the budget.
    pub max_waiting: i64,
    /// New pages dated within this long before now go in the fresh lane, which is handed over first and spends no budget; None for no fresh lane.
    pub fresh: Option<Duration>,
    /// Pages the fresh lane takes an hour at most; the rest go with the other pages.
    pub fresh_per_hour: f64,
    /// Pages of one site the fresh lane takes an hour at most, so a site dating every page today cannot fill it.
    pub fresh_per_site: u32,
    /// A part-filled fresh input is written once its first page waited this long.
    pub fresh_flush_after: Duration,
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
    /// How long after reading its last shard the backfill sweeps `current/` again.
    pub backfill_every: Duration,
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
            max_waiting: number("PLANNER_MAX_WAITING", 30_000_000)?,
            fresh: Some(Duration::from_secs_f64(number("PLANNER_FRESH_S", 48.0 * 3600.0)?)).filter(|window| !window.is_zero()),
            fresh_per_hour: number("PLANNER_FRESH_PER_HOUR", 300_000.0)?,
            fresh_per_site: number("PLANNER_FRESH_PER_SITE", 2_000)?,
            fresh_flush_after: Duration::from_secs_f64(number("PLANNER_FRESH_FLUSH_S", 60.0)?),
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
            backfill_every: Duration::from_secs_f64(number("PLANNER_BACKFILL_EVERY_S", 86_400.0)?),
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

/// A version of a page: its text, when it was fetched, and the change file that holds its row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Version {
    pub content_sha1: String,
    pub fetched_at: String,
    /// Only its page text was planned.
    pub page_only: bool,
    pub seq: Option<i64>,
    /// The crawl task that published it; empty for versions planned before it was kept.
    pub task_id: String,
}

impl Version {
    fn parse(raw: &[u8]) -> Version {
        let value = String::from_utf8_lossy(raw);
        let mut fields = value.split('\t');
        let (content_sha1, fetched_at) = (fields.next().unwrap_or_default().to_string(), fields.next().unwrap_or_default().to_string());
        let page_only = fields.next() == Some(PAGE);
        let seq = fields.next().and_then(|seq| seq.parse().ok());
        Version { content_sha1, fetched_at, page_only, seq, task_id: fields.next().unwrap_or_default().to_string() }
    }

    fn value(&self) -> String {
        let page_only = if self.page_only { PAGE } else { "" };
        let seq = self.seq.map(|seq| seq.to_string()).unwrap_or_default();
        format!("{}\t{}\t{page_only}\t{seq}\t{}", self.content_sha1, self.fetched_at, self.task_id)
    }
}

/// What the planner has planned, its place in the log and the inputs not yet handed over, kept on local disk.
pub struct State {
    db: DB,
    waiting: AtomicI64,
}

impl State {
    pub fn open(dir: &Path) -> Result<State> {
        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_compression_type(DBCompressionType::Zstd);
        let db = DB::open(&options, dir).with_context(|| format!("opening the planner state at {}", dir.display()))?;
        let waiting = match db.get(WAITING_COUNT)? {
            Some(raw) => String::from_utf8_lossy(&raw).parse()?,
            None => db.prefix_iterator(b"waiting/").take_while(|item| item.as_ref().is_ok_and(|(key, _)| key.starts_with(b"waiting/"))).count() as i64,
        };
        Ok(State { db, waiting: AtomicI64::new(waiting) })
    }

    /// How many pages wait.
    pub fn waiting_count(&self) -> i64 {
        self.waiting.load(Ordering::Relaxed)
    }

    fn count_waiting(&self, batch: &mut WriteBatch, change: i64) {
        let count = self.waiting.fetch_add(change, Ordering::Relaxed) + change;
        batch.put(WAITING_COUNT, count.max(0).to_string());
    }

    pub fn cursor(&self) -> Result<Option<i64>> {
        Ok(self.db.get(CURSOR)?.map(|raw| String::from_utf8_lossy(&raw).parse()).transpose()?)
    }

    /// The content hash of the version of `key` last planned with `recipe`.
    pub fn planned(&self, recipe: &str, key: &str) -> Result<Option<String>> {
        Ok(self.planned_version(recipe, key)?.map(|version| version.content_sha1))
    }

    /// The version of `key` last planned with `recipe`; its time is empty for versions planned before it was kept.
    pub fn planned_version(&self, recipe: &str, key: &str) -> Result<Option<Version>> {
        Ok(self.db.get(planned_key(recipe, key))?.map(|raw| Version::parse(&raw)))
    }

    /// The newest version of `key` seen but not planned, waiting for budget or room or forgotten after it was planned.
    pub fn waiting(&self, key: &str) -> Result<Option<Version>> {
        Ok(self.db.get(waiting_key(key))?.map(|raw| Version::parse(&raw)))
    }

    /// Keeps `version` waiting unless a newer one of the page already is, or `most` pages already wait; whether it waits.
    fn wait(&self, key: &str, version: &Version, most: i64) -> Result<bool> {
        if self.waiting_count() >= most && self.db.get_pinned(waiting_key(key))?.is_none() {
            return Ok(false);
        }
        let mut batch = WriteBatch::default();
        self.waiting_into(&mut batch, key, version)?;
        self.db.write(batch)?;
        Ok(true)
    }

    fn waiting_into(&self, batch: &mut WriteBatch, key: &str, version: &Version) -> Result<()> {
        let held = self.waiting(key)?;
        if held.as_ref().is_some_and(|held| (held.fetched_at.as_str(), held.seq) >= (version.fetched_at.as_str(), version.seq)) {
            return Ok(());
        }
        match held.map(|held| held.seq) {
            Some(Some(seq)) => batch.delete(replan_key(seq, key)),
            Some(None) => {}
            None => self.count_waiting(batch, 1),
        }
        batch.put(waiting_key(key), version.value());
        if let Some(seq) = version.seq {
            batch.put(replan_key(seq, key), "");
        }
        Ok(())
    }

    /// Stops `key` waiting, once a version as new is planned or its change file is gone.
    fn unwait_into(&self, batch: &mut WriteBatch, key: &str, planned_at: Option<&str>) -> Result<()> {
        let Some(held) = self.waiting(key)? else { return Ok(()) };
        if planned_at.is_some_and(|planned_at| held.fetched_at.as_str() > planned_at) {
            return Ok(());
        }
        batch.delete(waiting_key(key));
        if let Some(seq) = held.seq {
            batch.delete(replan_key(seq, key));
        }
        self.count_waiting(batch, -1);
        Ok(())
    }

    fn unwait(&self, key: &str) -> Result<()> {
        let mut batch = WriteBatch::default();
        self.unwait_into(&mut batch, key, None)?;
        Ok(self.db.write(batch)?)
    }

    /// Stops every page waiting without a change file, once; nothing re-plans such a page, so it is left to the backfill. How many stopped.
    pub fn drop_unnumbered(&self) -> Result<i64> {
        if self.db.get_pinned(WAITING_NUMBERED)?.is_some() {
            return Ok(0);
        }
        let (mut batch, mut held, mut dropped) = (WriteBatch::default(), 0, 0);
        for item in self.db.iterator(IteratorMode::From(b"waiting/", rocksdb::Direction::Forward)) {
            let (key, value) = item?;
            if !key.starts_with(b"waiting/") {
                break;
            }
            if Version::parse(&value).seq.is_none() {
                batch.delete(&key);
                held += 1;
                dropped += 1;
            }
            if held == UNWAIT_BATCH {
                self.count_waiting(&mut batch, -held);
                self.db.write(std::mem::take(&mut batch))?;
                held = 0;
            }
        }
        self.count_waiting(&mut batch, -held);
        batch.put(WAITING_NUMBERED, "1");
        self.db.write(batch)?;
        Ok(dropped)
    }

    /// The first change file after `after` that holds waiting pages, and their keys.
    pub fn replan_group(&self, after: i64) -> Result<Option<(i64, HashSet<String>)>> {
        let start = format!("replan/{:020}/", after + 1);
        let mut group: Option<(i64, HashSet<String>)> = None;
        for item in self.db.iterator(IteratorMode::From(start.as_bytes(), rocksdb::Direction::Forward)) {
            let (raw, _) = item?;
            let Some(rest) = raw.strip_prefix(b"replan/") else { break };
            let rest = String::from_utf8_lossy(rest);
            let Some((seq, key)) = rest.split_once('/') else { continue };
            let seq: i64 = seq.parse()?;
            match &mut group {
                Some((held, keys)) if *held == seq => {
                    keys.insert(key.to_string());
                }
                Some(_) => break,
                None => group = Some((seq, HashSet::from([key.to_string()]))),
            }
        }
        Ok(group)
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

    /// Forgets every planned version of a page gone from its site, so a page that comes back is planned again whatever its text; rows logged before change file `seq`, read late, are refused.
    fn forget_key(&self, key: &str, seq: i64) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put(removed_key(key), seq.to_string());
        let mut planned = false;
        for recipe in RECIPES {
            if self.db.get_pinned(planned_key(recipe, key))?.is_some() {
                batch.delete(planned_key(recipe, key));
                planned = true;
            }
        }
        if planned {
            self.change_sites(&mut batch, &HashMap::from([(domain_of_key(key).to_string(), -1)]))?;
        }
        self.unwait_into(&mut batch, key, None)?;
        Ok(self.db.write(batch)?)
    }

    /// Every task taken back, with the change file that took it back.
    pub fn withdrawn(&self) -> Result<HashMap<String, i64>> {
        let mut found = HashMap::new();
        for item in self.db.iterator(IteratorMode::From(b"withdrawn/", rocksdb::Direction::Forward)) {
            let (key, value) = item?;
            let Some(task_id) = key.strip_prefix(b"withdrawn/") else { break };
            found.insert(String::from_utf8_lossy(task_id).into_owned(), String::from_utf8_lossy(&value).parse()?);
        }
        Ok(found)
    }

    fn withdraw(&self, task_id: &str, seq: i64) -> Result<()> {
        Ok(self.db.put(format!("withdrawn/{task_id}"), seq.to_string())?)
    }

    /// The change file that removed `key`, while no later row has brought it back.
    pub fn removed(&self, key: &str) -> Result<Option<i64>> {
        Ok(self.db.get(removed_key(key))?.map(|raw| String::from_utf8_lossy(&raw).parse()).transpose()?)
    }

    fn returned(&self, key: &str) -> Result<()> {
        Ok(self.db.delete(removed_key(key))?)
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
                match self.planned_version(recipe, key)? {
                    Some(planned) if planned.content_sha1 == content_sha1 => {
                        batch.delete(planned_key(recipe, key));
                        // Planned again from its change file, not left for a re-crawl that may never come; one planned before files were kept is the backfill's.
                        if planned.seq.is_some() {
                            self.waiting_into(&mut batch, key, &Version { page_only: false, ..planned })?;
                        }
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
            let version = Version {
                content_sha1: page.content_sha1.clone(),
                fetched_at: page.fetched_at.clone(),
                page_only: page.page_only,
                seq: Some(page.seq),
                task_id: page.task_id.clone(),
            };
            batch.put(planned_key(page.recipe, &page.key), version.value());
            self.unwait_into(&mut batch, &page.key, Some(&page.fetched_at))?;
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

fn removed_key(key: &str) -> String {
    format!("removed/{key}")
}

fn waiting_key(key: &str) -> String {
    format!("waiting/{key}")
}

fn replan_key(seq: i64, key: &str) -> String {
    format!("replan/{seq:020}/{key}")
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

/// Where a buffered page was read, which says what a restart reads again for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A change file read in order: the kept cursor stays before it.
    Log(i64),
    /// A change file read late: it stays on the missing list.
    Late(i64),
    /// `current/`: the backfill's kept place stays before it.
    Backfill,
    /// A waiting page's change file: the page stays waiting.
    Replan,
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
    /// The change file that holds the version's row.
    pub seq: i64,
    pub task_id: String,
    pub source: Source,
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
    /// Pages of a news site with no date in `published` or the URL path, left out.
    pub undated: u64,
    /// Such pages typed `article`, taken because the set's `undated` is `article`.
    pub undated_article_in: u64,
    /// Pages of a news site dated before its `since`.
    pub before_since: u64,
    /// New pages of sites whose index cap is reached.
    pub over_cap: u64,
    pub unchanged: u64,
    /// Versions older than the one already planned, from change files read late.
    pub older: u64,
    /// Pages the log removed, whose planned versions are forgotten.
    pub removed: u64,
    /// Rows of tasks taken back, which every reader drops.
    pub withdrawn: u64,
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
    /// Pages left for budget or room, kept waiting to be planned from their change file.
    pub waiting: u64,
    /// Pages left for budget or room while `max_waiting` pages already waited, left to the backfill.
    pub not_kept: u64,
    /// Waiting pages read again from their change file, of `pages`.
    pub replanned: u64,
    /// Pages read from `current/` by the backfill, of `pages`.
    pub backfill: u64,
    pub shards: u64,
    pub inputs: u64,
    pub handed_over: u64,
    pub expired: u64,
    pub dropped: u64,
}

/// What became of a page the planner considered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Considered {
    Taken(Lane),
    /// Refused for budget or room, and kept waiting.
    Waiting,
    /// Not to be planned at this version.
    Passed,
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
    /// Pages by the change file read in order, and read late, that they came from.
    log: BTreeMap<i64, usize>,
    late: BTreeMap<i64, usize>,
    backfill: usize,
}

impl Buffer {
    fn count(&mut self, source: Source, add: bool) {
        let tally = |counts: &mut BTreeMap<i64, usize>, seq: i64| {
            let count = counts.entry(seq).or_default();
            *count = if add { *count + 1 } else { count.saturating_sub(1) };
            if *count == 0 {
                counts.remove(&seq);
            }
        };
        match source {
            Source::Log(seq) => tally(&mut self.log, seq),
            Source::Late(seq) => tally(&mut self.late, seq),
            Source::Backfill if add => self.backfill += 1,
            Source::Backfill => self.backfill = self.backfill.saturating_sub(1),
            Source::Replan => {}
        }
    }

    fn push(&mut self, planned: Planned) {
        self.count(planned.source, true);
        self.at.insert(planned.key.clone(), self.pages.len());
        self.chars += chars_of(&planned);
        self.pages.push(planned);
        self.since.get_or_insert_with(Instant::now);
    }

    /// Takes a page out, as a removal in the log does.
    fn remove(&mut self, key: &str) -> bool {
        let Some(at) = self.at.remove(key) else { return false };
        let gone = self.pages.swap_remove(at);
        if let Some(moved) = self.pages.get(at) {
            self.at.insert(moved.key.clone(), at);
        }
        self.count(gone.source, false);
        self.chars -= chars_of(&gone);
        if self.pages.is_empty() {
            self.since = None;
        }
        true
    }

    fn replace(&mut self, at: usize, planned: Planned) {
        self.count(self.pages[at].source, false);
        self.count(planned.source, true);
        self.chars = self.chars - chars_of(&self.pages[at]) + chars_of(&planned);
        self.pages[at] = planned;
    }
}

pub struct Planner {
    pub settings: Settings,
    pub state: Arc<State>,
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
    /// When the backfill sweeps `current/` again, once it has read every shard.
    next_sweep: Option<Instant>,
    /// The last change file the re-plan pass read this sweep.
    replan_after: i64,
    set_modified: Option<std::time::SystemTime>,
    /// The Unix time the held set was built, or its file written when it does not say.
    set_built: Option<f64>,
    set_warned: Option<Instant>,
    /// When the API may be asked for room again mid-pass.
    ask_room_at: Option<Instant>,
    fresh: Buffer,
    rest: Buffer,
    read_through: i64,
    budget: Budget,
    fresh_budget: Budget,
    /// Inputs waiting for room, by lane.
    pending: [usize; 2],
    missing: BTreeMap<i64, Instant>,
    /// Late files read or given up whose pages may still be buffered; forgotten once none are.
    settled: Vec<i64>,
    /// The cursor last kept on disk.
    kept_cursor: i64,
    /// Tasks taken back, with the change file that took each back.
    withdrawn: HashMap<String, i64>,
    /// The hour and the pages each site put in the fresh lane in it.
    fresh_sites: (i64, HashMap<String, u32>),
}

impl Planner {
    pub async fn new(settings: Settings, pages: Bucket, temp: Bucket, redis: ConnectionManager, room: Room) -> Result<Planner> {
        std::fs::create_dir_all(&settings.dir)?;
        let state = Arc::new(State::open(&settings.dir)?);
        let started = Instant::now();
        state.count_sites()?;
        if started.elapsed() > Duration::from_secs(1) {
            println!("planner: counted every domain's planned pages in {:?}", started.elapsed());
        }
        let dropped = state.drop_unnumbered()?;
        if dropped > 0 {
            println!("planner: {dropped} pages waiting without a change file are left to the backfill");
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
        let (budget, fresh_budget) = (Budget::new(settings.pages_per_hour), Budget::new(settings.fresh_per_hour));
        let withdrawn = state.withdrawn()?;
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
            next_sweep: None,
            replan_after: 0,
            set_modified: None,
            set_built: None,
            set_warned: None,
            ask_room_at: None,
            fresh: Buffer::default(),
            rest: Buffer::default(),
            read_through,
            budget,
            fresh_budget,
            pending,
            missing,
            settled: Vec::new(),
            kept_cursor: read_through,
            withdrawn,
            fresh_sites: (0, HashMap::new()),
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
                let written = modified.duration_since(std::time::UNIX_EPOCH).map_or(0.0, |since| since.as_secs_f64());
                let (built, when) = match set.built.as_deref().and_then(|built| parse_iso(&built.replace('Z', "+00:00"))) {
                    Some(built) => (built as f64, "built"),
                    None => (written, "with no build time, its file written"),
                };
                println!("planner: set of {} sites {when} {} read from {}", set.len(), iso((built * 1e6) as i64), path.display());
                self.set = Some(Arc::new(set));
                self.set_modified = Some(modified);
                self.set_built = Some(built);
                self.set_warned = None;
                if let Some(warning) = self.stale_set_warning(Instant::now()) {
                    eprintln!("{warning}");
                }
            }
            Ok(_) => eprintln!("{} holds no sites; keeping the set held", path.display()),
            Err(error) => eprintln!("reading the set {}: {error:#}; keeping the set held", path.display()),
        }
        Ok(self.set.is_some())
    }

    /// How long ago the held set was built.
    pub fn set_age(&self) -> Option<Duration> {
        self.set_built.map(|built| Duration::from_secs_f64((now_us() as f64 / 1e6 - built).max(0.0)))
    }

    /// The warning due at `now` for a set older than two days: when it is read, then once an hour.
    pub fn stale_set_warning(&mut self, now: Instant) -> Option<String> {
        let age = self.set_age().filter(|age| *age > SET_STALE)?;
        if self.set_warned.is_some_and(|warned| now.saturating_duration_since(warned) < SET_WARN_EVERY) {
            return None;
        }
        self.set_warned = Some(now);
        Some(format!("planner: the set was built {:.1} days ago; check that it is still pulled", age.as_secs_f64() / 86_400.0))
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
        // Inputs written before a restart go out before the log is read.
        self.hand_over_unless_held().await?;
        self.forget_dropped().await?;
        let latest = latest(&self.pages).await?;
        let mut read = self.read_late().await?;
        let ahead: Vec<i64> = (self.read_through + 1..=latest).take(FILES_PER_PASS.saturating_sub(read)).collect();
        let pages = self.pages.clone();
        let mut fetched = stream::iter(ahead).map(|seq| fetch(pages.clone(), seq).map(move |rows| (seq, rows))).buffered(self.settings.reads.max(1));
        while let Some((seq, rows)) = fetched.next().await {
            match rows? {
                Ok(rows) => {
                    self.take(rows, seq, false).await?;
                    read += 1;
                }
                Err(why) => self.note_missing(seq, &why)?,
            }
            self.advance(seq)?;
        }
        let replanned = self.replan().await?;
        let backfilled = self.backfill().await?;
        for lane in Lane::ALL {
            let after = if lane == Lane::Fresh { self.settings.fresh_flush_after } else { self.settings.flush_after };
            if self.buffer(lane).since.is_some_and(|since| since.elapsed() >= after) {
                self.flush_lane(lane).await?;
            }
        }
        self.hand_over().await?;
        Ok(read > 0 || replanned || backfilled)
    }

    /// Plans waiting pages from the change files that hold them, oldest first, while the budget holds an input's worth and the rest lane has room; whether it took any.
    async fn replan(&mut self) -> Result<bool> {
        let mut took = false;
        for _ in 0..REPLAN_FILES_PER_PASS {
            if self.pending[Lane::Rest as usize] >= self.settings.max_pending || !self.budget.holds(self.settings.pages_per_input) {
                break;
            }
            let Some((seq, keys)) = self.state.replan_group(self.replan_after)? else {
                self.replan_after = 0;
                break;
            };
            self.replan_after = seq;
            let rows = match fetch(self.pages.clone(), seq).await? {
                Ok(rows) => rows,
                // Past the cursor an unnumbered file is one the log expired, index and all.
                Err(Unread::NotNumbered) if seq <= self.read_through => Vec::new(),
                Err(_) => continue,
            };
            let mut found = HashSet::new();
            for row in rows {
                let ChangeRow::Page(page) = row else { continue };
                if !keys.contains(&page.key) || self.state.waiting(&page.key)?.is_none_or(|held| held.content_sha1 != page.record.content_sha1) {
                    continue;
                }
                let key = page.key.clone();
                found.insert(key.clone());
                took = true;
                self.counts.replanned += 1;
                // A page taken stops waiting when its input is written, so a crash before that loses nothing.
                match self.consider(*page, Source::Replan) {
                    Considered::Taken(lane) => self.write_when_full(lane).await?,
                    Considered::Waiting => {}
                    Considered::Passed => self.state.unwait(&key)?,
                }
            }
            // A change file gone from the log leaves its pages to the backfill.
            for key in keys.iter().filter(|key| !found.contains(*key)) {
                self.state.unwait(key)?;
            }
        }
        Ok(took)
    }

    /// Plans pages of `current/` while the budget and the rest lane's room last, starting only once the budget holds an input's worth; whether it took any.
    async fn backfill(&mut self) -> Result<bool> {
        // Waiting pages are re-planned first: both wait for an input's worth, and the re-plan runs before.
        if !self.settings.backfill || self.current.is_none() || !self.budget.holds(self.settings.pages_per_input) {
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
                    if let Considered::Taken(lane) = self.consider(page, Source::Backfill) {
                        self.write_when_full(lane).await?;
                    }
                }
                Ok(None) => {
                    let reading = self.reading.take().expect("a shard being read");
                    let shard = reading.shard;
                    // A shard that failed is read again from the last place kept.
                    reading.done.await?.with_context(|| format!("reading shard {shard} of {}", self.settings.current_prefix))?;
                    self.flush().await?;
                    self.counts.shards += 1;
                    if shard + 1 < self.settings.backfill_shards.end {
                        self.state.keep_backfill(shard + 1, None)?;
                    } else {
                        // Pages the log passed by, or that came in since, are planned on the next sweep.
                        self.state.keep_backfill(self.settings.backfill_shards.start, None)?;
                        self.next_sweep = Some(Instant::now() + self.settings.backfill_every);
                        println!("planner: the backfill read its last shard, {shard}; it sweeps again in {:?}", self.settings.backfill_every);
                        return Ok(took);
                    }
                }
            }
        }
    }

    /// Starts reading the backfill's next shard from where it was left; false while a finished sweep waits for the next.
    fn start_reading(&mut self) -> Result<bool> {
        if self.next_sweep.is_some_and(|next| Instant::now() < next) {
            return Ok(false);
        }
        let shards = self.settings.backfill_shards.clone();
        let (shard, after) = self.state.backfill()?.unwrap_or((shards.start, None));
        let (shard, after) = if shards.contains(&shard) { (shard, after) } else { (shards.start, None) };
        let reader = ShardReader {
            bucket: self.current.clone().expect("a current/ bucket"),
            prefix: self.settings.current_prefix.clone(),
            handle: Handle::current(),
            ranges_at_once: RANGES_AT_ONCE,
        };
        let (send, pages) = mpsc::channel(BACKFILL_AHEAD);
        let (set, langs, from) = (self.set.clone(), self.settings.langs.clone(), after.clone());
        let (state, page_first, withdrawn) = (self.state.clone(), self.settings.page_first, self.withdrawn.clone());
        let done = tokio::task::spawn_blocking(move || {
            let keep = |row: &PageRow| {
                let record = &row.record;
                recipe_of(record).is_some_and(|recipe| PLANNED_RECIPES.contains(&recipe))
                    && lang_wanted(&langs, &record.lang)
                    && set.as_ref().is_none_or(|set| set.admit(&record.url, &record.published, record.page_type.as_deref()).is_ok())
                    && match state.planned_version(RECIPE_V2, &row.key) {
                        Ok(Some(planned)) if planned.content_sha1 == record.content_sha1 => planned.page_only && !page_first,
                        _ => true,
                    }
                    // A version as new waits in the log, which is read for it instead.
                    && state.waiting(&row.key).ok().flatten().is_none_or(|held| held.fetched_at <= record.fetched_at)
            };
            reader.read(shard, from.as_deref(), withdrawn, keep, |page| send.blocking_send(page).is_ok())
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
        let settled = self.settle(&[Lane::Fresh, Lane::Rest]);
        self.state.settle_missing(&settled)?;
        Ok(read)
    }

    /// Considers every page of change file `seq`; the inner error says why the file cannot be read yet, the outer one fails the pass.
    async fn read(&mut self, seq: i64) -> Result<std::result::Result<(), Unread>> {
        match fetch(self.pages.clone(), seq).await? {
            Ok(rows) => self.take(rows, seq, true).await.map(Ok),
            Err(why) => Ok(Err(why)),
        }
    }

    /// Considers every page of one change file, in file order.
    async fn take(&mut self, rows: Vec<ChangeRow>, seq: i64, late: bool) -> Result<()> {
        for row in rows {
            match row {
                ChangeRow::Page(page) => {
                    let from = if late { Source::Late(page.change_seq) } else { Source::Log(page.change_seq) };
                    if let Considered::Taken(lane) = self.consider(*page, from) {
                        self.write_when_full(lane).await?;
                    }
                }
                ChangeRow::Removed(key) => {
                    for lane in Lane::ALL {
                        self.buffer_mut(lane).remove(&key);
                    }
                    self.state.forget_key(&key, seq)?;
                    self.counts.removed += 1;
                }
                ChangeRow::TaskWithdrawn(task_id) => {
                    for lane in Lane::ALL {
                        let buffer = self.buffer_mut(lane);
                        let taken: Vec<String> = buffer.pages.iter().filter(|page| page.task_id == task_id).map(|page| page.key.clone()).collect();
                        for key in taken {
                            buffer.remove(&key);
                        }
                    }
                    self.state.withdraw(&task_id, seq)?;
                    self.withdrawn.insert(task_id, seq);
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

    /// Takes back the pages of inputs the task API dropped; they are trimmed from its list only once forgotten, so a crash loses none.
    pub async fn forget_dropped(&mut self) -> Result<usize> {
        let dropped: Vec<String> = self.redis.lrange(EMBED_DROPPED, 0, (FILES_PER_PASS * 10) as isize - 1).await?;
        for raw in &dropped {
            let entry: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
            self.state.forget(&entry["pages"])?;
        }
        if !dropped.is_empty() {
            let () = self.redis.ltrim(EMBED_DROPPED, dropped.len() as isize, -1).await?;
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
            if let Some(warning) = self.stale_set_warning(Instant::now()) {
                eprintln!("{warning}");
            }
            if !busy {
                let _ = tokio::time::timeout(self.settings.poll, stop.changed()).await;
            }
        }
        self.flush().await?;
        Ok(())
    }

    /// Marks change file `seq` read and keeps the cursor just before the first change file a buffered page was read from, so a restart reads only those again.
    fn advance(&mut self, seq: i64) -> Result<()> {
        self.read_through = seq;
        let cursor = self.cursor_after(&[Lane::Fresh, Lane::Rest]);
        if cursor > self.kept_cursor {
            self.state.keep_cursor(cursor)?;
            self.kept_cursor = cursor;
        }
        Ok(())
    }

    /// Where a restart may start reading the log, given the pages still in `lanes`.
    fn cursor_after(&self, lanes: &[Lane]) -> i64 {
        lanes.iter().filter_map(|&lane| self.buffer(lane).log.keys().next().copied()).min().map_or(self.read_through, |first| first - 1)
    }

    /// The late files read whose pages are no longer in `lanes`, taken off the list of those still settling.
    fn settle(&mut self, lanes: &[Lane]) -> Vec<i64> {
        let held: HashSet<i64> = lanes.iter().flat_map(|&lane| self.buffer(lane).late.keys().copied()).collect();
        let (kept, settled): (Vec<i64>, Vec<i64>) = std::mem::take(&mut self.settled).into_iter().partition(|seq| held.contains(seq));
        self.settled = kept;
        settled
    }

    /// Takes the page into its lane's buffer if a planned recipe reads it, it is wanted, changed, and the budget and the API's room allow it; the lane it went in.
    pub fn consider(&mut self, page: PageRow, from: Source) -> Considered {
        self.counts.pages += 1;
        let record = &page.record;
        match self.state.removed(&page.key).ok().flatten() {
            Some(removed) if page.change_seq < removed => {
                self.counts.older += 1;
                return Considered::Passed;
            }
            Some(_) => {
                if let Err(error) = self.state.returned(&page.key) {
                    eprintln!("noting {} back: {error:#}", page.key);
                }
            }
            None => {}
        }
        if self.withdrawn.contains_key(&record.task_id) {
            self.counts.withdrawn += 1;
            return Considered::Passed;
        }
        // A newer row lets go of an older waiting version, whatever becomes of the row.
        let mut waiting = self.state.waiting(&page.key).ok().flatten().filter(|held| !self.withdrawn.contains_key(&held.task_id));
        if waiting.as_ref().is_some_and(|held| held.fetched_at < record.fetched_at) {
            if let Err(error) = self.state.unwait(&page.key) {
                eprintln!("letting {} stop waiting: {error:#}", page.key);
            }
            waiting = None;
        }
        let Some(recipe) = recipe_of(record).filter(|recipe| PLANNED_RECIPES.contains(recipe)) else {
            self.counts.other_recipe += 1;
            return Considered::Passed;
        };
        if !self.admitted(record) {
            self.counts.refused += 1;
            return Considered::Passed;
        }
        match self.set.as_ref().map(|set| set.admit(&record.url, &record.published, record.page_type.as_deref())) {
            Some(Err(refusal)) => {
                match refusal {
                    Refusal::Undated => self.counts.undated += 1,
                    Refusal::BeforeSince => self.counts.before_since += 1,
                    Refusal::Unlisted | Refusal::OffPaths => self.counts.outside_set += 1,
                }
                return Considered::Passed;
            }
            Some(Ok(Admitted::UndatedArticle)) => self.counts.undated_article_in += 1,
            Some(Ok(Admitted::ByRule)) | None => {}
        }
        let cap = match self.listed.as_ref().filter(|_| self.set.is_none()) {
            None => None,
            Some(listed) => match listed.cap(&record.domain) {
                Some(cap) => Some(cap),
                None => {
                    self.counts.unlisted += 1;
                    return Considered::Passed;
                }
            },
        };
        let buffered = Lane::ALL.into_iter().find_map(|lane| self.buffer(lane).at.get(&page.key).map(|&at| (lane, at)));
        let already = match buffered {
            Some((lane, at)) => {
                let held = &self.buffer(lane).pages[at];
                Some(Version {
                    content_sha1: held.content_sha1.clone(),
                    fetched_at: held.fetched_at.clone(),
                    page_only: held.page_only,
                    seq: Some(held.seq),
                    task_id: held.task_id.clone(),
                })
            }
            // A version from a task taken back was taken down, so the same text crawled again is planned again.
            None => self.state.planned_version(recipe, &page.key).ok().flatten().filter(|planned| !self.withdrawn.contains_key(&planned.task_id)),
        };
        if waiting.as_ref().is_some_and(|held| record.fetched_at < held.fetched_at) {
            self.counts.older += 1;
            return Considered::Passed;
        }
        let mut parts_only = false;
        if let Some(Version { content_sha1, fetched_at, page_only, .. }) = &already {
            if *content_sha1 == record.content_sha1 {
                if !page_only || self.settings.page_first || buffered.is_some() {
                    self.counts.unchanged += 1;
                    return Considered::Passed;
                }
                parts_only = true;
            }
            // Times are `iso` text of one width, so they order as text; a late file can hold an older version than the one planned.
            if record.fetched_at < *fetched_at {
                self.counts.older += 1;
                return Considered::Passed;
            }
        }
        if let Some((lane, at)) = buffered {
            let planned = planned_of(&page, recipe, self.settings.page_first && lane == Lane::Rest, false, from);
            self.buffer_mut(lane).replace(at, planned);
            return Considered::Taken(lane);
        }
        let new_in = match cap {
            Some(cap) if self.state.unplanned(&page.key).unwrap_or(false) => {
                let domain = domain_of_key(&page.key).to_string();
                let buffered: u64 = Lane::ALL.into_iter().map(|lane| self.buffer(lane).new.get(&domain).copied().unwrap_or(0)).sum();
                let held = self.state.site_pages(&domain).unwrap_or(0) + buffered;
                if held >= cap {
                    self.counts.over_cap += 1;
                    return Considered::Passed;
                }
                Some(domain)
            }
            _ => None,
        };
        // Only a page never planned before is fresh, within its lane's own budget, so pages re-dated by their site cannot flood it.
        let seen = !self.state.unplanned(&page.key).unwrap_or(false) || waiting.is_some();
        let fresh = self.is_fresh(record) && !seen && self.pending[Lane::Fresh as usize] < self.settings.max_pending;
        let site = self.set.as_ref().and_then(|set| set.site(&record.domain).map(|(host, _)| host.to_string()));
        let site = site.unwrap_or_else(|| domain_of_key(&page.key).to_string());
        let fresh = fresh && self.fresh_site_room(&site) && self.fresh_budget.take();
        if fresh {
            *self.fresh_sites.1.entry(site).or_default() += 1;
        }
        let lane = if fresh { Lane::Fresh } else { Lane::Rest };
        if self.pending[lane as usize] >= self.settings.max_pending {
            self.counts.no_room += 1;
            self.keep_waiting(&page);
            return Considered::Waiting;
        }
        if lane == Lane::Rest && !self.budget.take() {
            self.counts.over_budget += 1;
            self.keep_waiting(&page);
            return Considered::Waiting;
        }
        let planned = planned_of(&page, recipe, self.settings.page_first && lane == Lane::Rest, parts_only, from);
        if planned.texts.is_empty() {
            self.counts.unchanged += 1;
            return Considered::Passed;
        }
        self.counts.page_only += u64::from(planned.page_only);
        self.counts.parts += u64::from(parts_only);
        let buffer = self.buffer_mut(lane);
        if let Some(domain) = new_in {
            *buffer.new.entry(domain).or_default() += 1;
        }
        buffer.push(planned);
        self.counts.planned += 1;
        self.counts.fresh += u64::from(lane == Lane::Fresh);
        Considered::Taken(lane)
    }

    fn keep_waiting(&mut self, page: &PageRow) {
        let record = &page.record;
        let version = Version {
            content_sha1: record.content_sha1.clone(),
            fetched_at: record.fetched_at.clone(),
            page_only: false,
            seq: Some(page.change_seq),
            task_id: record.task_id.clone(),
        };
        match self.state.wait(&page.key, &version, self.settings.max_waiting) {
            Ok(true) => self.counts.waiting += 1,
            Ok(false) => self.counts.not_kept += 1,
            Err(error) => eprintln!("keeping {} waiting: {error:#}", page.key),
        }
    }

    /// Whether the page is dated within the fresh window and not past tomorrow, by the older of `published` and a date in its URL.
    fn is_fresh(&self, record: &Record) -> bool {
        let Some(window) = self.settings.fresh else { return false };
        let now = now_us() as f64 / 1e6;
        let date = match (iso_date(&record.published), url_date(&record.url)) {
            (Some(published), Some(path)) => Some(published.min(path)),
            (published, path) => published.or(path),
        };
        date.is_some_and(|date| date >= utc_day(now - window.as_secs_f64()) && date <= utc_day(now + 86_400.0))
    }

    /// Whether `site` has room in the fresh lane this hour.
    fn fresh_site_room(&mut self, site: &str) -> bool {
        let hour = now_us() / 3_600_000_000;
        if self.fresh_sites.0 != hour {
            self.fresh_sites = (hour, HashMap::new());
        }
        self.fresh_sites.1.get(site).copied().unwrap_or(0) < self.settings.fresh_per_site
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

    fn has_token(&mut self) -> bool {
        self.budget.has()
    }

    /// Writes every lane's buffer.
    pub async fn flush(&mut self) -> Result<()> {
        for lane in Lane::ALL {
            self.flush_lane(lane).await?;
        }
        Ok(())
    }

    /// Writes a lane's full buffer and hands it over at once, so the API's queue is fed while a pass goes on.
    async fn write_when_full(&mut self, lane: Lane) -> Result<()> {
        let buffer = self.buffer(lane);
        if buffer.pages.len() >= self.settings.pages_per_input || buffer.chars >= self.settings.chars_per_input {
            self.flush_lane(lane).await?;
            self.hand_over_unless_held().await?;
        }
        Ok(())
    }

    /// Hands over unless the API's last answer, under `ROOM_RETRY` ago, left inputs waiting or never came.
    async fn hand_over_unless_held(&mut self) -> Result<()> {
        if self.ask_room_at.is_none_or(|at| Instant::now() >= at) {
            self.hand_over().await?;
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
        // The cursor, late files and the backfill's place move only past what the other lane still holds.
        let other = if lane == Lane::Fresh { Lane::Rest } else { Lane::Fresh };
        let cursor = Some(self.cursor_after(&[other])).filter(|&cursor| cursor > self.kept_cursor);
        let settled = self.settle(&[other]);
        self.state.commit(&self.buffer(lane).pages, &input_key, &entry.to_string(), cursor, &settled)?;
        self.kept_cursor = cursor.unwrap_or(self.kept_cursor);
        if let Some(reading) = self.reading.as_ref().filter(|_| self.buffer(other).backfill == 0) {
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
                self.ask_room_at = Some(Instant::now() + ROOM_RETRY);
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
            push_input(&mut self.redis, &input_key, &entry).await?;
            self.state.handed_over(&input_key)?;
            self.pending[lane] -= 1;
            handed += 1;
        }
        self.counts.handed_over += handed as u64;
        self.counts.expired += expired as u64;
        self.ask_room_at = (self.pending.iter().sum::<usize>() > 0).then(|| Instant::now() + ROOM_RETRY);
        Ok(handed)
    }

    pub fn read_through(&self) -> i64 {
        self.read_through
    }
}

/// The page as the buffer holds it, its texts made only once it is taken: all of them, its page text alone, or its passages alone.
fn planned_of(page: &PageRow, recipe: &'static str, page_first: bool, parts_only: bool, source: Source) -> Planned {
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
        seq: page.change_seq,
        task_id: record.task_id.clone(),
        source,
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

/// Pushes an input for the task API once: a push tried again after a crash, before its pending entry was cleared, does nothing; whether it pushed.
pub async fn push_input(redis: &mut ConnectionManager, input_key: &str, entry: &str) -> Result<bool> {
    let pushed: i64 = PUSH.key(EMBED_INPUTS).key(format!("embed:pushed:{input_key}")).arg(entry).arg(PUSHED_TTL_S).invoke_async(redis).await?;
    Ok(pushed == 1)
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

/// Pages an hour, of which a quarter hour's worth can be spent at once.
struct Budget {
    per_hour: f64,
    tokens: f64,
    refilled: Instant,
}

impl Budget {
    fn new(per_hour: f64) -> Budget {
        Budget { per_hour, tokens: Budget::cap(per_hour), refilled: Instant::now() }
    }

    fn cap(per_hour: f64) -> f64 {
        (per_hour / 4.0).max(1.0)
    }

    fn has(&mut self) -> bool {
        self.holds(1)
    }

    /// Whether `pages` can be spent now, or the most the budget ever holds when that is fewer.
    fn holds(&mut self, pages: usize) -> bool {
        self.tokens = (self.tokens + self.refilled.elapsed().as_secs_f64() * self.per_hour / 3600.0).min(Budget::cap(self.per_hour));
        self.refilled = Instant::now();
        self.tokens >= (pages as f64).min(Budget::cap(self.per_hour))
    }

    fn take(&mut self) -> bool {
        if !self.has() {
            return false;
        }
        self.tokens -= 1.0;
        true
    }
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

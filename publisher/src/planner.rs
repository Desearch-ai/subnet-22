//! The planner: follows the change log and turns published pages that need a vector into embed inputs for the task API, within an hourly page budget and the room the API gives.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, Int32Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use desearch::feeds::CHANGES;
use desearch::r2::{self, Bucket, PARQUET};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use rocksdb::{DBCompressionType, IteratorMode, Options, WriteBatch, DB};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::watch;

use crate::current::{decode_changes, ChangeRow, PageRow};
use crate::records::Record;
use crate::service::{day, now_us};

/// The text recipe below; any change to what a text holds is a new embedding version, so every page is embedded again.
pub const RECIPE: &str = "page-v1";
pub const PAGE: &str = "page";
pub const PASSAGE: &str = "passage";
/// Enough for the model's 2,048 tokens; the model server cuts the rest.
const PAGE_CHARS: usize = 10_000;
/// An article longer than this also gets passages.
const PASSAGES_FROM_CHARS: usize = 2_000;
const PASSAGE_MIN_CHARS: usize = 600;
const PASSAGE_MAX_CHARS: usize = 1_500;
const MAX_PASSAGES: usize = 16;
/// The task API's list of inputs waiting for a round.
pub const EMBED_INPUTS: &str = "embed:inputs";
/// Inputs whose task the API dropped after its last attempt.
pub const EMBED_DROPPED: &str = "embed:dropped";
const CURSOR: &[u8] = b"cursor";
const ROOM_TIMEOUT: Duration = Duration::from_secs(20);
const FILES_PER_PASS: usize = 20;

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

/// The texts of one page: its page text, and the passages of an article long enough to need them, each led by the title and its section's heading.
pub fn texts(key: &str, record: &Record) -> Vec<Text> {
    let mut texts = vec![Text { text_id: format!("{key}#page"), kind: PAGE, index: 0, text: page_text(record) }];
    if record.page_type.as_deref() == Some("article") && record.text.chars().count() > PASSAGES_FROM_CHARS {
        let title = record.title.trim();
        for (i, passage) in passages(&record.text, &record.headings).into_iter().enumerate() {
            let lead = [Some(title).filter(|t| !t.is_empty()), passage.heading.as_deref()].into_iter().flatten().collect::<Vec<_>>().join(" › ");
            let text = if lead.is_empty() { passage.body } else { format!("{lead}\n{}", passage.body) };
            texts.push(Text { text_id: format!("{key}#p{}", i + 1), kind: PASSAGE, index: i as i32 + 1, text });
        }
    }
    texts
}

/// Title, description unless the text opens with it, then the text, one per line, cut to what the model reads.
pub fn page_text(record: &Record) -> String {
    let text = record.text.trim();
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

/// A passage of an article and the heading of the section it is in.
#[derive(Clone, Debug, PartialEq)]
pub struct Passage {
    pub heading: Option<String>,
    pub body: String,
}

/// Consecutive lines grouped into passages of about `PASSAGE_MIN_CHARS` to `PASSAGE_MAX_CHARS` characters; a line that is one of the page's headings starts a new section, and a long line is split at its sentences.
pub fn passages(text: &str, headings: &[Option<String>]) -> Vec<Passage> {
    let headings: HashSet<&str> = headings.iter().flatten().map(|h| h.trim()).filter(|h| !h.is_empty()).collect();
    let mut passages: Vec<Passage> = Vec::new();
    let mut heading: Option<String> = None;
    let (mut current, mut held) = (String::new(), 0);
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if headings.contains(line) {
            // A section's short tail stays with the next one rather than standing alone.
            if held >= PASSAGE_MIN_CHARS / 3 {
                passages.push(Passage { heading: heading.clone(), body: std::mem::take(&mut current) });
                held = 0;
                if passages.len() >= MAX_PASSAGES {
                    return passages;
                }
            }
            heading = Some(line.to_string());
            continue;
        }
        for piece in sentences_within(line, PASSAGE_MAX_CHARS) {
            let size = piece.chars().count();
            if held > 0 && held + 1 + size > PASSAGE_MAX_CHARS {
                passages.push(Passage { heading: heading.clone(), body: std::mem::take(&mut current) });
                held = 0;
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
            if passages.len() >= MAX_PASSAGES {
                return passages;
            }
        }
    }
    let room = passages.len() < MAX_PASSAGES;
    match passages.last_mut() {
        Some(last) if held > 0 && held < PASSAGE_MIN_CHARS / 2 && last.heading == heading => {
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
    /// Inputs written but not yet handed over; past this, new pages are left out.
    pub max_pending: usize,
    /// The temporary bucket keeps an input a day; one waiting longer than this is dropped, so no miner is sent a file already gone.
    pub pending_max_age: Duration,
    pub min_chars: usize,
    pub langs: Vec<String>,
    pub excluded: HashSet<String>,
    /// The change file to start after when nothing was planned before; the newest one when None.
    pub start_after: Option<i64>,
    pub poll: Duration,
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
            pending_max_age: Duration::from_secs_f64(number("PLANNER_PENDING_MAX_AGE_S", 12.0 * 3600.0)?),
            min_chars: number("PLANNER_MIN_CHARS", 300)?,
            langs: text("PLANNER_LANGS", "en").split(',').map(|lang| lang.trim().to_lowercase()).filter(|lang| !lang.is_empty()).collect(),
            excluded,
            start_after: std::env::var("PLANNER_START_AFTER").ok().filter(|v| !v.is_empty()).map(|v| v.parse()).transpose()?,
            poll: Duration::from_secs_f64(number("PLANNER_POLL_S", 10.0)?),
            hole_wait: Duration::from_secs_f64(number("PLANNER_HOLE_WAIT_S", 3600.0)?),
        })
    }
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

    /// The content hash of the version of `key` last planned.
    pub fn planned(&self, key: &str) -> Result<Option<String>> {
        Ok(self.db.get(planned_key(key))?.map(|raw| String::from_utf8_lossy(&raw).into_owned()))
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
        for page in pages.as_array().into_iter().flatten() {
            let (Some(key), Some(content_sha1)) = (page["page_key"].as_str(), page["content_sha1"].as_str()) else { continue };
            if self.planned(key)?.as_deref() == Some(content_sha1) {
                batch.delete(planned_key(key));
            }
        }
        Ok(batch)
    }

    fn keep_cursor(&self, seq: i64) -> Result<()> {
        Ok(self.db.put(CURSOR, seq.to_string())?)
    }

    /// One written input: its pages planned, the input pending, the cursor where the log is read up to, together.
    fn commit(&self, pages: &[Planned], input_key: &str, entry: &str, cursor: Option<i64>) -> Result<()> {
        let mut batch = WriteBatch::default();
        for page in pages {
            batch.put(planned_key(&page.key), &page.content_sha1);
        }
        batch.put(format!("pending/{input_key}"), entry);
        if let Some(seq) = cursor {
            batch.put(CURSOR, seq.to_string());
        }
        Ok(self.db.write(batch)?)
    }
}

fn planned_key(key: &str) -> String {
    format!("planned/{RECIPE}/{key}")
}

/// One page waiting in the input being filled.
#[derive(Clone, Debug)]
pub struct Planned {
    pub key: String,
    pub url: String,
    pub host: String,
    pub content_sha1: String,
    pub texts: Vec<Text>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Counts {
    pub files: u64,
    pub pages: u64,
    pub planned: u64,
    pub refused: u64,
    pub unchanged: u64,
    pub over_budget: u64,
    pub no_room: u64,
    pub inputs: u64,
    pub handed_over: u64,
    pub expired: u64,
    pub dropped: u64,
}

pub struct Planner {
    pub settings: Settings,
    pub state: State,
    pub pages: Bucket,
    pub temp: Bucket,
    pub redis: ConnectionManager,
    pub room: Room,
    pub counts: Counts,
    buffer: Vec<Planned>,
    buffered: HashMap<String, usize>,
    buffer_chars: usize,
    buffer_since: Option<Instant>,
    read_through: i64,
    tokens: f64,
    refilled: Instant,
    pending: usize,
    missing_since: HashMap<i64, Instant>,
}

impl Planner {
    pub async fn new(settings: Settings, pages: Bucket, temp: Bucket, redis: ConnectionManager, room: Room) -> Result<Planner> {
        std::fs::create_dir_all(&settings.dir)?;
        let state = State::open(&settings.dir)?;
        let read_through = match (state.cursor()?, settings.start_after) {
            (Some(seq), _) | (None, Some(seq)) => seq,
            (None, None) => latest(&pages).await?,
        };
        state.keep_cursor(read_through)?;
        let pending = state.pending()?.len();
        let tokens = budget_cap(settings.pages_per_hour);
        Ok(Planner {
            settings,
            state,
            pages,
            temp,
            redis,
            room,
            counts: Counts::default(),
            buffer: Vec::new(),
            buffered: HashMap::new(),
            buffer_chars: 0,
            buffer_since: None,
            read_through,
            tokens,
            refilled: Instant::now(),
            pending,
            missing_since: HashMap::new(),
        })
    }

    /// Reads what the log added since the last pass, writes the inputs that are due and hands over what the API has room for; whether it read anything.
    pub async fn pass(&mut self) -> Result<bool> {
        self.forget_dropped().await?;
        let latest = latest(&self.pages).await?;
        let mut read = 0;
        while self.read_through < latest && read < FILES_PER_PASS {
            let seq = self.read_through + 1;
            let Some(key) = change_key(&self.pages, seq).await? else {
                let since = *self.missing_since.entry(seq).or_insert_with(Instant::now);
                if since.elapsed() < self.settings.hole_wait {
                    break;
                }
                eprintln!("change file {seq} was never numbered; going on without it");
                self.advance(seq)?;
                continue;
            };
            match self.pages.get(&key).await {
                Ok(file) => {
                    let rows = tokio::task::spawn_blocking(move || decode_changes(file, seq)).await??;
                    for row in rows {
                        if let ChangeRow::Page(page) = row {
                            self.consider(*page);
                            if self.buffer.len() >= self.settings.pages_per_input || self.buffer_chars >= self.settings.chars_per_input {
                                self.flush().await?;
                            }
                        }
                    }
                }
                Err(r2::Error::Missing) => eprintln!("change file {seq} ({key}) is gone; going on without it"),
                Err(error) => return Err(anyhow!("reading change file {seq}: {error}")),
            }
            self.counts.files += 1;
            self.advance(seq)?;
            read += 1;
        }
        if self.buffer_since.is_some_and(|since| since.elapsed() >= self.settings.flush_after) {
            self.flush().await?;
        }
        self.hand_over().await?;
        Ok(read > 0)
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
        self.missing_since.remove(&seq);
        if self.buffer.is_empty() {
            self.state.keep_cursor(seq)?;
        }
        Ok(())
    }

    /// Takes the page into the buffer if it is wanted, changed, and the budget and the API's room allow it.
    pub fn consider(&mut self, page: PageRow) {
        self.counts.pages += 1;
        let record = &page.record;
        if !self.admitted(record) {
            self.counts.refused += 1;
            return;
        }
        let already = match self.buffered.get(&page.key) {
            Some(&at) => Some(self.buffer[at].content_sha1.clone()),
            None => self.state.planned(&page.key).ok().flatten(),
        };
        if already.as_deref() == Some(record.content_sha1.as_str()) {
            self.counts.unchanged += 1;
            return;
        }
        let planned = Planned {
            key: page.key.clone(),
            url: record.url.clone(),
            host: record.domain.clone(),
            content_sha1: record.content_sha1.clone(),
            texts: texts(&page.key, record),
        };
        if let Some(&at) = self.buffered.get(&page.key) {
            self.buffer_chars = self.buffer_chars - chars_of(&self.buffer[at]) + chars_of(&planned);
            self.buffer[at] = planned;
            return;
        }
        if self.pending >= self.settings.max_pending {
            self.counts.no_room += 1;
            return;
        }
        if !self.take_token() {
            self.counts.over_budget += 1;
            return;
        }
        self.buffered.insert(page.key, self.buffer.len());
        self.buffer_chars += chars_of(&planned);
        self.buffer.push(planned);
        self.buffer_since.get_or_insert_with(Instant::now);
        self.counts.planned += 1;
    }

    fn admitted(&self, record: &Record) -> bool {
        let lang = record.lang.to_lowercase();
        self.settings.langs.iter().any(|want| lang == *want || lang.starts_with(&format!("{want}-")) || lang.starts_with(&format!("{want}_")))
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
        let cap = budget_cap(self.settings.pages_per_hour);
        self.tokens = (self.tokens + self.refilled.elapsed().as_secs_f64() * self.settings.pages_per_hour / 3600.0).min(cap);
        self.refilled = Instant::now();
        if self.tokens < 1.0 {
            return false;
        }
        self.tokens -= 1.0;
        true
    }

    /// Writes the buffer as one input in the temp bucket and keeps it pending until the API has room.
    pub async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let rows: Vec<(&Planned, &Text)> = self.buffer.iter().flat_map(|page| page.texts.iter().map(move |text| (page, text))).collect();
        let file = encode_input(&rows)?;
        let input_key = format!("embed/inputs/dt={}/{}.parquet", day(now_us()), uuid::Uuid::new_v4().simple());
        let entry = json!({
            "input_key": input_key,
            "input_sha256": hex(&Sha256::digest(&file)),
            "written_at": now_us() as f64 / 1e6,
            "texts": rows.len(),
            "chars": rows.iter().map(|(_, text)| text.text.chars().count()).sum::<usize>(),
            "pages": self.buffer.iter().map(|p| json!({"page_key": p.key, "content_sha1": p.content_sha1, "url": p.url, "host": p.host})).collect::<Vec<_>>(),
        });
        // Kept until the input is stored, so a failed write is tried again with the same pages.
        self.temp.put(&input_key, Bytes::from(file), PARQUET, None).await.map_err(|error| anyhow!("writing {input_key}: {error}"))?;
        self.state.commit(&self.buffer, &input_key, &entry.to_string(), Some(self.read_through))?;
        self.buffer.clear();
        self.buffered.clear();
        self.buffer_chars = 0;
        self.buffer_since = None;
        self.pending += 1;
        self.counts.inputs += 1;
        Ok(())
    }

    /// Hands pending inputs to the task API, oldest first, as many as it has room for.
    pub async fn hand_over(&mut self) -> Result<usize> {
        let pending = self.state.pending()?;
        self.pending = pending.len();
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
        pending.sort_by(|a, b| a.2["written_at"].as_f64().unwrap_or(now).total_cmp(&b.2["written_at"].as_f64().unwrap_or(now)));
        let (mut handed, mut expired) = (0, 0);
        for (input_key, entry, parsed) in pending {
            if now - parsed["written_at"].as_f64().unwrap_or(now) > self.settings.pending_max_age.as_secs_f64() {
                eprintln!("{input_key} waited longer than {:?} for room; dropped", self.settings.pending_max_age);
                self.state.expire(&input_key, &parsed)?;
                expired += 1;
                continue;
            }
            if handed == room {
                break;
            }
            let _: i64 = self.redis.rpush(EMBED_INPUTS, &entry).await?;
            self.state.handed_over(&input_key)?;
            handed += 1;
        }
        self.pending -= handed + expired;
        self.counts.handed_over += handed as u64;
        self.counts.expired += expired as u64;
        Ok(handed)
    }

    pub fn read_through(&self) -> i64 {
        self.read_through
    }
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

//! A page's own text: its site's template lines out, whether enough of it is left for a vector, and the error and utility page rule, read as the indexer reads them (`tests/clean-golden.json`).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::SystemTime;

use anyhow::{bail, Context, Result};
use regex::Regex;
use sha2::{Digest, Sha256};

/// The table build's thresholds, kept equal to the indexer's.
pub const MIN_PAGES: usize = 20;
pub const STATIC_SHARE: f64 = 0.3;
pub const MIN_REPEATS: usize = 5;
/// Repeated lines are template only in a run of this many.
pub const RUN: usize = 3;
pub const LEDE_WORDS: usize = 12;
pub const LABEL_WORDS: usize = 4;
pub const LABELS: usize = 2;
pub const VALUE_WORDS: usize = 6;
pub const TITLE_CONTAINS_CHARS: usize = 20;
pub const PROSE_WORDS: usize = 10;
pub const MIN_PROSE: usize = 1;
pub const MIN_CHARS: usize = 100;
pub const RECORD_PAIRS: usize = 3;
/// A value under a label line has at most this many words.
pub const RECORD_VALUE_WORDS: usize = 8;
pub const NAV_TYPES: [&str; 3] = ["home", "landing", "listing"];
pub const ERROR_TEXT_CHARS: usize = 3000;
pub const ERROR_TITLE_CHARS: usize = 600;
/// An error phrase counts only in a line of at most this many words.
pub const ERROR_LINE_WORDS: usize = 15;
pub const STORE: &str = "templates-store.tpl";
pub const RECENT: &str = "templates-recent.tpl";

const FENCE: &str = "```";
const EDGES: [char; 7] = ['.', ':', ';', ',', '|', ' ', '\t'];
const MAGIC: &[u8; 8] = b"DSTPL01\n";
const HEAD: usize = 24;
const ENTRY: usize = 32;
const READ_CHUNK: usize = 1 << 20;

static MARKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:#{1,6}[ \t]+|[-*+•][ \t]+|>[ \t]*|\d{1,3}[.)][ \t]+)+").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\r\f\v\x{a0}\x{2000}-\x{200b}\x{202f}\x{205f}\x{3000}]+").unwrap());
static LABEL_VALUE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:[-*+•][ \t]+)?([^:|]{1,40}?):[ \t]+[^\s\x1c-\x1f]").unwrap());
static SEPARATOR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\|?[ \t:|-]+\|?$").unwrap());
static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#{1,6}[ \t]+").unwrap());
static ERROR_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)^(?:404(?: error)?(?: ?[-:|–]? ?(?:page )?not found)?|(?:page )?not found|error(?: \d{3})?",
        r"|an error (?:has )?occurred|access denied|forbidden|just a moment|attention required",
        r"|are you a (?:human|robot)\??|security check|site (?:is )?(?:under )?maintenance",
        r"|under construction|coming soon)$",
    ))
    .unwrap()
});
static ERROR_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)page (?:you (?:are|were) looking for|cannot be found|could not be found|not found|doesn.t exist)",
        r"|enable javascript and cookies|checking (?:if the site connection is secure|your browser)",
        r"|verify you are (?:a )?human|access to this page has been denied|the requested url was not found",
    ))
    .unwrap()
});
static UTILITY_SEGMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)^(?:privacy(?:[-_]?(?:policy|notice|statement|settings))?",
        r"|cookies?(?:[-_]?(?:policy|notice|settings|preferences))?",
        r"|terms(?:[-_](?:of[-_])?(?:use|service|services|and[-_]conditions|conditions))?|tos|terms-?conditions",
        r"|legal(?:[-_]notice)?|login|log-in|signin|sign-in|sign_in|wp-login\.php|logout",
        r"|register|registration|signup|sign-up|account|my-?account|cart|basket|checkout",
        r")(?:\.html?|\.php|\.aspx?)?$",
    ))
    .unwrap()
});
static LOCALE_SEGMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^[a-z]{2}(?:[-_][a-z]{2})?$").unwrap());
/// The parents a utility page may sit under, besides a locale.
const UTILITY_PARENTS: [&str; 18] =
    ["legal", "policies", "policy", "help", "about", "info", "company", "site", "support", "account", "accounts", "user", "users", "customer", "my", "pages", "p", "page"];
static UTILITY_QUERY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:^|&)(?:attachment_id|print|share)=").unwrap());
static UTILITY_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)^(?:privacy (?:policy|notice|statement)|cookies? (?:policy|notice|settings)",
        r"|terms of (?:use|service)|terms (?:and|&) conditions|legal notice",
        r"|log ?in|sign ?in|sign ?up|create (?:an )?account|my account|shopping cart)$",
    ))
    .unwrap()
});
static TITLE_SEP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\x1c-\x1f]+[|\-–—:·•»][\s\x1c-\x1f]+").unwrap());

/// Python's `str.isspace`.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Python's `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// The length of Python's `str.split()`.
fn py_words(s: &str) -> usize {
    s.split(py_space).filter(|word| !word.is_empty()).count()
}

/// A line as tables compare it: no markdown marks or edge punctuation, one space, lower case.
pub fn line_key(line: &str) -> String {
    let s = MARKS.replace(py_strip(line), "");
    SPACES.replace_all(&s, " ").trim_matches(EDGES).to_lowercase()
}

pub fn line_hash(key: &str) -> u64 {
    u64::from_le_bytes(Sha256::digest(key.as_bytes())[..8].try_into().unwrap())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Code,
    Table,
    Blank,
    Text,
}

/// Per line: code inside a fence, a table row, blank, else text.
pub fn kinds(lines: &[&str]) -> Vec<Kind> {
    let mut inside = false;
    lines
        .iter()
        .map(|line| {
            let s = py_strip(line);
            if s.starts_with(FENCE) {
                inside = !inside;
                Kind::Code
            } else if inside {
                Kind::Code
            } else if s.is_empty() {
                Kind::Blank
            } else if s.starts_with('|') {
                Kind::Table
            } else {
                Kind::Text
            }
        })
        .collect()
}

pub fn page_hashes(text: &str) -> HashSet<u64> {
    let lines: Vec<&str> = text.split('\n').collect();
    lines
        .iter()
        .zip(kinds(&lines))
        .filter(|(_, kind)| *kind == Kind::Text)
        .map(|(line, _)| line_key(line))
        .filter(|key| !key.is_empty())
        .map(|key| line_hash(&key))
        .collect()
}

/// A site's level for a line hash: 2 static, 1 repeated, 0 neither.
pub trait Levels {
    fn level(&self, hash: u64) -> u8;
}

/// One site's table: its sampled pages and its static and repeated line hashes, each sorted.
#[derive(Clone, Copy, Debug)]
pub struct SiteTable<'a> {
    pub pages: u32,
    pub statics: &'a [u64],
    pub repeated: &'a [u64],
}

impl Levels for SiteTable<'_> {
    fn level(&self, hash: u64) -> u8 {
        if self.statics.binary_search(&hash).is_ok() {
            2
        } else {
            u8::from(self.repeated.binary_search(&hash).is_ok())
        }
    }
}

/// An owned site table, its hashes sorted and unique.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pages: u32,
    statics: Vec<u64>,
    repeated: Vec<u64>,
}

impl Table {
    pub fn new(pages: u32, mut statics: Vec<u64>, mut repeated: Vec<u64>) -> Table {
        for hashes in [&mut statics, &mut repeated] {
            hashes.sort_unstable();
            hashes.dedup();
        }
        Table { pages, statics, repeated }
    }

    /// A table from hashes in hex, as the indexer's JSON tables write them.
    pub fn from_hex<S: AsRef<str>>(pages: u32, statics: &[S], repeated: &[S]) -> Result<Table> {
        let parse = |hexes: &[S]| {
            hexes.iter().map(|h| u64::from_str_radix(h.as_ref(), 16).with_context(|| format!("line hash {:?}", h.as_ref()))).collect::<Result<Vec<_>>>()
        };
        Ok(Table::new(pages, parse(statics)?, parse(repeated)?))
    }

    pub fn view(&self) -> SiteTable<'_> {
        SiteTable { pages: self.pages, statics: &self.statics, repeated: &self.repeated }
    }
}

impl Levels for Table {
    fn level(&self, hash: u64) -> u8 {
        self.view().level(hash)
    }
}

/// Tables of one site from several files: a line takes its highest level in any.
#[derive(Clone, Debug)]
pub struct Joined<'a>(pub Vec<SiteTable<'a>>);

impl Levels for Joined<'_> {
    fn level(&self, hash: u64) -> u8 {
        self.0.iter().map(|table| table.level(hash)).max().unwrap_or(0)
    }
}

/// Per line, whether it is template: static anywhere, repeated only in a run of RUN of them.
pub fn template_lines(table: Option<&dyn Levels>, text: &str, title: &str) -> Vec<bool> {
    let lines: Vec<&str> = text.split('\n').collect();
    let Some(table) = table else {
        return vec![false; lines.len()];
    };
    let ks = kinds(&lines);
    let keys: Vec<String> = lines.iter().zip(&ks).map(|(line, kind)| if *kind == Kind::Text { line_key(line) } else { String::new() }).collect();
    let found: Vec<u8> = keys.iter().map(|key| if key.is_empty() { 0 } else { table.level(line_hash(key)) }).collect();
    let t = line_key(title);
    let t_long = t.chars().count() > TITLE_CONTAINS_CHARS;
    let lede = keys.iter().zip(&found).find(|(key, level)| !key.is_empty() && py_words(key) >= LEDE_WORDS && **level != 2).map(|(key, _)| key.as_str());
    let mine = |k: &str| k == t || Some(k) == lede || (t_long && k.chars().count() > TITLE_CONTAINS_CHARS && (t.contains(k) || k.contains(t.as_str())));
    let level: Vec<u8> = keys.iter().zip(&found).map(|(key, &level)| if key.is_empty() || mine(key) { 0 } else { level }).collect();
    let mut drop = vec![false; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        if level[i] == 0 {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut members = Vec::new();
        while j < lines.len() && (level[j] != 0 || ks[j] == Kind::Blank) {
            if level[j] != 0 {
                members.push(j);
            }
            j += 1;
        }
        for &m in &members {
            drop[m] = level[m] == 2 || members.len() >= RUN;
        }
        i = j;
    }
    labels_kept(&ks, &keys, &drop)
}

/// Template labels, headings among them, stay before the page's own short value: up to LABELS lines.
pub fn labels_kept(ks: &[Kind], keys: &[String], drop: &[bool]) -> Vec<bool> {
    let mut out = drop.to_vec();
    let filled: Vec<usize> = (0..ks.len()).filter(|&i| ks[i] != Kind::Blank).collect();
    for (at, &i) in filled.iter().enumerate() {
        if drop[i] || ks[i] != Kind::Text || py_words(&keys[i]) > VALUE_WORDS {
            continue;
        }
        for &j in filled[at.saturating_sub(LABELS)..at].iter().rev() {
            if !drop[j] || py_words(&keys[j]) > LABEL_WORDS {
                break;
            }
            out[j] = false;
        }
    }
    out
}

/// The text without its template lines.
pub fn strip(table: Option<&dyn Levels>, text: &str, title: &str) -> String {
    let drop = template_lines(table, text, title);
    text.split('\n').zip(drop).filter(|(_, drop)| !drop).map(|(line, _)| line).collect::<Vec<_>>().join("\n")
}

pub fn prose_chars(text: &str) -> usize {
    text.split('\n').filter(|line| py_words(line) >= PROSE_WORDS).map(|line| line.chars().count()).sum()
}

/// Lines that pair a short label with a value: `Label: value`, or a table row of two or more cells.
pub fn record_pairs(text: &str) -> usize {
    let filled: Vec<&str> = text.split('\n').map(py_strip).filter(|line| !line.is_empty()).collect();
    let mut n = 0;
    for (i, s) in filled.iter().enumerate() {
        if s.starts_with('|') || LABEL_VALUE.is_match(s) {
            n += usize::from(record_pair(s));
            continue;
        }
        let label = HEADING.find(s).map_or(*s, |m| &s[m.end()..]);
        let words = py_words(label.trim_end_matches(':'));
        if (label.len() != s.len() || s.ends_with(':')) && (1..=LABEL_WORDS).contains(&words) {
            if let Some(value) = filled.get(i + 1) {
                n += usize::from(!HEADING.is_match(value) && py_words(value) <= RECORD_VALUE_WORDS);
            }
        }
    }
    n
}

fn record_pair(s: &str) -> bool {
    if s.starts_with('|') {
        let cells = s.trim_matches('|').split('|').filter(|cell| !py_strip(cell).is_empty()).count();
        return cells >= 2 && !SEPARATOR.is_match(s);
    }
    LABEL_VALUE.captures(s).is_some_and(|m| !s[..m.get(0).unwrap().end()].contains("//") && py_words(&m[1]) <= LABEL_WORDS)
}

/// Too little own text for a vector, unless a home, landing or listing page or a record of fields.
pub fn thin(text: &str, page_type: Option<&str>) -> bool {
    if NAV_TYPES.contains(&page_type.unwrap_or("")) {
        return false;
    }
    if prose_chars(text) >= MIN_PROSE || text.chars().nth(MIN_CHARS - 1).is_some() {
        return false;
    }
    record_pairs(text) < RECORD_PAIRS
}

/// Python's `re.I` reads `İ` and `ı` as `i`; the regex crate's case folding does not.
fn dotted_i(s: &str) -> Cow<'_, str> {
    if s.contains(['İ', 'ı']) {
        Cow::Owned(s.replace(['İ', 'ı'], "i"))
    } else {
        Cow::Borrowed(s)
    }
}

/// A URL's path and query as Python's `urlsplit` reads them.
fn path_query(url: &str) -> (String, String) {
    let url = url.trim_start_matches(|c: char| c <= ' ').replace(['\t', '\r', '\n'], "");
    let mut rest = url.as_str();
    if let Some(i) = rest.find(':') {
        if i > 0 && rest.as_bytes()[0].is_ascii_alphabetic() && rest[..i].bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')) {
            rest = &rest[i + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        rest = &after[after.find(['/', '?', '#']).unwrap_or(after.len())..];
    }
    let rest = rest.split_once('#').map_or(rest, |(rest, _)| rest);
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    (path.to_string(), query.to_string())
}

/// `error` for an error or challenge page, `utility` for a privacy, terms, login, account or cart page.
pub fn junk(url: &str, title: &str, text: &str) -> Option<&'static str> {
    let title = py_strip(title);
    let head = py_strip(TITLE_SEP.find(title).map_or(title, |m| &title[..m.start()])).trim_matches([' ', '.', '!']);
    let head = dotted_i(head);
    let chars = text.chars().take(ERROR_TEXT_CHARS).count();
    if chars < ERROR_TEXT_CHARS && text.split('\n').any(|line| py_words(line) <= ERROR_LINE_WORDS && ERROR_TEXT.is_match(&dotted_i(line))) {
        return Some("error");
    }
    if chars < ERROR_TITLE_CHARS && ERROR_TITLE.is_match(&head) {
        return Some("error");
    }
    let (path, query) = path_query(url);
    if utility_path(&path) || (!query.is_empty() && UTILITY_QUERY.is_match(&dotted_i(&query))) {
        return Some("utility");
    }
    if UTILITY_TITLE.is_match(&head) {
        return Some("utility");
    }
    None
}

/// A utility page's path: its segment alone, or under a locale or a site-section parent.
fn utility_path(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    if parts.is_empty() || parts.len() > 2 || !UTILITY_SEGMENT.is_match(&dotted_i(parts[parts.len() - 1])) {
        return false;
    }
    parts.len() == 1 || LOCALE_SEGMENT.is_match(&dotted_i(parts[0])) || UTILITY_PARENTS.contains(&parts[0].to_lowercase().as_str())
}

struct Entry {
    pages: u32,
    at: usize,
    statics: usize,
    repeated: usize,
}

/// One template table file, read whole; a site's table looked up on demand.
pub struct TableFile {
    index: HashMap<String, Entry>,
    hashes: Vec<u64>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl TableFile {
    pub fn read(path: &Path) -> Result<TableFile> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let size = file.metadata()?.len();
        let mut f = BufReader::with_capacity(READ_CHUNK, file);
        let mut head = [0u8; HEAD];
        f.read_exact(&mut head).with_context(|| format!("{} is not a template table file", path.display()))?;
        if &head[..8] != MAGIC {
            bail!("{} is not a template table file", path.display());
        }
        let (n, names_len) = (u64_at(&head, 8), u64_at(&head, 16));
        let body = n.checked_mul(ENTRY as u64).and_then(|e| e.checked_add(names_len)).and_then(|b| b.checked_add(HEAD as u64)).filter(|&b| b <= size);
        let Some(body) = body else {
            bail!("{} is shorter than its {n} hosts and {names_len} name bytes", path.display());
        };
        if (size - body) % 8 != 0 {
            bail!("{} has {} hash bytes, not whole hashes", path.display(), size - body);
        }
        let mut entries = vec![0u8; n as usize * ENTRY];
        f.read_exact(&mut entries)?;
        let mut names = vec![0u8; names_len as usize];
        f.read_exact(&mut names)?;
        let mut left = (size - body) as usize;
        let mut hashes = Vec::with_capacity(left / 8);
        let mut chunk = vec![0u8; READ_CHUNK.min(left)];
        while left > 0 {
            let take = chunk.len().min(left);
            f.read_exact(&mut chunk[..take])?;
            hashes.extend(chunk[..take].as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)));
            left -= take;
        }
        let mut index = HashMap::with_capacity(n as usize);
        for e in entries.as_chunks::<ENTRY>().0 {
            let (name_at, name_len) = (u64_at(e, 0) as usize, u32_at(e, 8) as usize);
            let entry = Entry { pages: u32_at(e, 12), at: u64_at(e, 16) as usize, statics: u32_at(e, 24) as usize, repeated: u32_at(e, 28) as usize };
            let Some(name) = names.get(name_at..name_at.saturating_add(name_len)) else {
                bail!("{}: a host name past the names", path.display());
            };
            let name = std::str::from_utf8(name).with_context(|| format!("{}: a host name not UTF-8", path.display()))?;
            if entry.at.saturating_add(entry.statics).saturating_add(entry.repeated) > hashes.len() {
                bail!("{}: {name}'s hashes past the file", path.display());
            }
            index.insert(name.to_string(), entry);
        }
        Ok(TableFile { index, hashes })
    }

    pub fn get(&self, host: &str) -> Option<SiteTable<'_>> {
        let e = self.index.get(host)?;
        let statics_end = e.at + e.statics;
        Some(SiteTable { pages: e.pages, statics: &self.hashes[e.at..statics_end], repeated: &self.hashes[statics_end..statics_end + e.repeated] })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

/// Writes a table file as the indexer does: hosts sorted, each with its static then repeated hashes sorted; written whole, then renamed in place.
pub fn write(path: &Path, tables: &BTreeMap<String, Table>) -> Result<()> {
    let (mut entries, mut names, mut hashes) = (Vec::new(), Vec::new(), Vec::new());
    for (host, t) in tables {
        entries.extend((names.len() as u64).to_le_bytes());
        entries.extend((host.len() as u32).to_le_bytes());
        entries.extend(t.pages.to_le_bytes());
        entries.extend((hashes.len() as u64 / 8).to_le_bytes());
        entries.extend((t.statics.len() as u32).to_le_bytes());
        entries.extend((t.repeated.len() as u32).to_le_bytes());
        names.extend(host.as_bytes());
        hashes.extend(t.statics.iter().chain(&t.repeated).flat_map(|h| h.to_le_bytes()));
    }
    let pad = names.len().next_multiple_of(8) - names.len();
    names.resize(names.len() + pad, 0);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".part");
    let tmp = PathBuf::from(tmp);
    let mut f = BufWriter::new(File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?);
    f.write_all(MAGIC)?;
    f.write_all(&(tables.len() as u64).to_le_bytes())?;
    f.write_all(&(names.len() as u64).to_le_bytes())?;
    for part in [&entries, &names, &hashes] {
        f.write_all(part)?;
    }
    f.into_inner().map_err(|e| e.into_error())?;
    fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
}

#[derive(Clone)]
struct Loaded {
    stamp: Stamp,
    file: Arc<TableFile>,
}

type Stamp = (Option<SystemTime>, u64);

/// The store and recent table files in a directory, read again when either changes; a file that fails to read leaves the one before it.
/// The store and recent table files of a directory; a clone shares the files read.
#[derive(Clone)]
pub struct Tables {
    dir: PathBuf,
    files: [Option<Loaded>; 2],
    /// A file that failed to read, not read again until it changes.
    failed: [Option<Stamp>; 2],
}

impl Tables {
    pub fn open(dir: impl Into<PathBuf>) -> Tables {
        let mut tables = Tables { dir: dir.into(), files: [None, None], failed: [None, None] };
        tables.reload();
        tables
    }

    /// Reads each file again whose modified time or size changed; whether any table changed.
    pub fn reload(&mut self) -> bool {
        let mut changed = false;
        for ((slot, failed), name) in self.files.iter_mut().zip(self.failed.iter_mut()).zip([STORE, RECENT]) {
            let path = self.dir.join(name);
            let meta = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(e) if e.kind() == ErrorKind::NotFound => {
                    changed |= slot.take().is_some();
                    continue;
                }
                Err(e) => {
                    eprintln!("warning: template table {} not read: {e}", path.display());
                    continue;
                }
            };
            let stamp = (meta.modified().ok(), meta.len());
            if slot.as_ref().is_some_and(|loaded| loaded.stamp == stamp) || *failed == Some(stamp) {
                continue;
            }
            match TableFile::read(&path) {
                Ok(file) => {
                    *slot = Some(Loaded { stamp, file: Arc::new(file) });
                    (*failed, changed) = (None, true);
                }
                Err(e) => {
                    *failed = Some(stamp);
                    eprintln!("warning: template table {} not read: {e:#}", path.display());
                }
            }
        }
        changed
    }

    pub fn get(&self, host: &str) -> Option<Joined<'_>> {
        let found: Vec<SiteTable<'_>> = self.files.iter().flatten().filter_map(|loaded| loaded.file.get(host)).collect();
        (!found.is_empty()).then_some(Joined(found))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("clean-{name}-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn keys_pages_and_junk_read_every_golden_case_as_the_indexer_does() {
        let golden: serde_json::Value = serde_json::from_str(include_str!("../tests/clean-golden.json")).unwrap();
        let c = &golden["constants"];
        let numbers = [
            "MIN_PAGES",
            "MIN_REPEATS",
            "RUN",
            "LEDE_WORDS",
            "LABEL_WORDS",
            "LABELS",
            "VALUE_WORDS",
            "TITLE_CONTAINS_CHARS",
            "PROSE_WORDS",
            "MIN_PROSE",
            "MIN_CHARS",
            "RECORD_PAIRS",
            "RECORD_VALUE_WORDS",
            "ERROR_TEXT_CHARS",
            "ERROR_TITLE_CHARS",
            "ERROR_LINE_WORDS",
        ]
        .map(|name| c[name].as_u64().unwrap() as usize);
        assert_eq!(
            numbers,
            [
                MIN_PAGES,
                MIN_REPEATS,
                RUN,
                LEDE_WORDS,
                LABEL_WORDS,
                LABELS,
                VALUE_WORDS,
                TITLE_CONTAINS_CHARS,
                PROSE_WORDS,
                MIN_PROSE,
                MIN_CHARS,
                RECORD_PAIRS,
                RECORD_VALUE_WORDS,
                ERROR_TEXT_CHARS,
                ERROR_TITLE_CHARS,
                ERROR_LINE_WORDS
            ]
        );
        assert_eq!(c["STATIC_SHARE"].as_f64().unwrap(), STATIC_SHARE);
        for k in golden["keys"].as_array().unwrap() {
            let (line, key, hash) = (k[0].as_str().unwrap(), k[1].as_str().unwrap(), k[2].as_str().unwrap());
            assert_eq!(line_key(line), key, "{line:?}");
            assert_eq!(format!("{:016x}", line_hash(key)), hash, "{line:?}");
        }
        let hexes = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|h| h.as_str().unwrap().to_string()).collect::<Vec<_>>();
        for case in golden["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let table = (!case["static"].is_null()).then(|| Table::from_hex(0, &hexes(&case["static"]), &hexes(&case["repeated"])).unwrap());
            let kept = strip(table.as_ref().map(|t| t as &dyn Levels), case["text"].as_str().unwrap(), case["title"].as_str().unwrap());
            assert_eq!(kept, case["kept"].as_str().unwrap(), "{name}");
            assert_eq!(prose_chars(&kept), case["prose"].as_u64().unwrap() as usize, "{name}");
            assert_eq!(record_pairs(&kept), case["pairs"].as_u64().unwrap() as usize, "{name}");
            assert_eq!(thin(&kept, case["page_type"].as_str()), case["thin"].as_bool().unwrap(), "{name}");
        }
        for case in golden["junk"].as_array().unwrap() {
            let (url, title, text) = (case["url"].as_str().unwrap(), case["title"].as_str().unwrap(), case["text"].as_str().unwrap());
            assert_eq!(junk(url, title, text), case["junk"].as_str(), "{url} {title:?}");
        }
    }

    #[test]
    fn whitespace_case_and_urls_follow_python() {
        assert_eq!(line_key("\x1c Most\u{1680}Read \u{85}"), "most\u{1680}read");
        assert_eq!(py_words("a\x1cb\u{1680}c\u{2028}d"), 4);
        assert_eq!(junk("https://a.com/x", "LOGİN", ""), Some("utility"));
        assert_eq!(junk("https://a.com/x", "ſign ın | Shop", ""), Some("utility"));
        assert_eq!(junk("https://a.com/x", "Sign in\x1f-\x1fFan Club", ""), Some("utility"));
        assert_eq!(junk(" \x01https://a.com/he\tlp/acc\nount/?x#print=1", "", ""), Some("utility"));
        assert_eq!(junk(" \x01https://a.com/se\tttings/acc\nount/?x#print=1", "", ""), None);
        assert_eq!(junk("https://a.com/p#?print=1", "", ""), None);
        assert_eq!(junk("a.com/cart", "", ""), None);
        assert_eq!(path_query("https://a.com"), (String::new(), String::new()));
        assert_eq!(junk("https://a.com/a", "", &format!("{} enable javascript and cookies", "é".repeat(2980))), None);
        assert_eq!(junk("https://a.com/a", "", &format!("{} enable javascript and cookies", "é".repeat(2960))), Some("error"));
    }

    #[test]
    fn a_table_file_is_laid_out_as_the_indexer_writes_it() {
        let dir = temp_dir("layout");
        let path = dir.join(STORE);
        let tables = BTreeMap::from([
            ("b.org".to_string(), Table::new(20, vec![], vec![7])),
            ("a.com".to_string(), Table::new(3, vec![5, 1], vec![2])),
            ("ü.de".to_string(), Table::new(1, vec![u64::MAX], vec![])),
        ]);
        write(&path, &tables).unwrap();
        let python = "445354504c30310a03000000000000001000000000000000000000000000000005000000030000000000000000000000020000000100000005000000000000000500000014000000030000000000000000000000010000000a00000000000000050000000100000004000000000000000100000000000000612e636f6d622e6f7267c3bc2e6465000100000000000000050000000000000002000000000000000700000000000000ffffffffffffffff";
        let hex: String = fs::read(&path).unwrap().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, python);
        let file = TableFile::read(&path).unwrap();
        assert_eq!(file.len(), 3);
        let a = file.get("a.com").unwrap();
        assert_eq!((a.pages, a.statics, a.repeated), (3, &[1, 5][..], &[2][..]));
        assert_eq!([1, 2, 5, 7].map(|h| a.level(h)), [2, 1, 2, 0]);
        assert_eq!(file.get("ü.de").unwrap().level(u64::MAX), 2);
        assert!(file.get("c.net").is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tables_join_store_and_recent_and_keep_a_file_that_fails_to_read() {
        let dir = temp_dir("tables");
        let store = BTreeMap::from([("a.com".to_string(), Table::new(30, vec![1], vec![2, 3]))]);
        let recent = BTreeMap::from([("a.com".to_string(), Table::new(25, vec![2], vec![4])), ("b.org".to_string(), Table::new(25, vec![9], vec![]))]);
        write(&dir.join(STORE), &store).unwrap();
        let mut tables = Tables::open(&dir);
        assert_eq!(tables.get("a.com").unwrap().level(2), 1);
        write(&dir.join(RECENT), &recent).unwrap();
        assert!(tables.reload());
        assert!(!tables.reload());
        let a = tables.get("a.com").unwrap();
        assert_eq!([1, 2, 3, 4, 5].map(|h| a.level(h)), [2, 2, 1, 1, 0]);
        assert_eq!(tables.get("b.org").unwrap().level(9), 2);
        assert!(tables.get("c.net").is_none());
        fs::write(dir.join(STORE), b"not a table file").unwrap();
        assert!(!tables.reload());
        assert_eq!(tables.get("a.com").unwrap().level(3), 1);
        fs::remove_file(dir.join(STORE)).unwrap();
        assert!(tables.reload());
        assert_eq!(tables.get("a.com").unwrap().level(3), 0);
        fs::remove_dir_all(dir).unwrap();
    }
}

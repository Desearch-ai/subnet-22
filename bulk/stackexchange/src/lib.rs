//! Stack Exchange data dumps (CC BY-SA): a page per question with its accepted and best answers, at the site's question address.

pub mod html;
pub mod slug;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{bail, Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use quick_xml::events::{BytesStart, Event};
use quick_xml::XmlVersion;

pub const SOURCE: &str = "stackexchange";
pub const PAGE_TYPE: &str = "article";
const CONTENT_TYPE: &str = "text/html";
const QUESTION: u8 = 1;
const ANSWER: u8 = 2;
const SKELETON_MAGIC: &[u8; 8] = b"SESKEL01";
pub const TAKEN: &str = "questions.txt";
const POSTS_XML: &str = "Posts.xml";
const USERS_XML: &str = "Users.xml";

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    #[command(flatten)]
    pub site: SiteArgs,
    #[command(flatten)]
    pub cut: CutArgs,
    /// Question or answer ids, one a line, whose questions are taken whatever the cut.
    #[arg(long)]
    pub include_ids: Option<PathBuf>,
    /// Only the questions `--include-ids` names.
    #[arg(long)]
    pub only_included: bool,
    /// Question ids to leave out, one a line, such as the `questions.txt` of a run before.
    #[arg(long)]
    pub skip_ids: Option<PathBuf>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct SiteArgs {
    /// The site's host, as in stackoverflow.com; by default the archive's name up to `.7z` or `-Posts.7z`.
    #[arg(long, default_value = "")]
    pub host: String,
    /// The site's name, as in Stack Overflow; by default the host.
    #[arg(long, default_value = "")]
    pub site_name: String,
    /// The first pass's scores and sizes, read when the file exists and written when it does not.
    #[arg(long)]
    pub skeleton: Option<PathBuf>,
    /// Users.xml, or the archive holding it; by default the posts' archive, or the `-Users.7z` beside a `-Posts.7z`.
    #[arg(long)]
    pub users: Option<PathBuf>,
    /// Posts.xml, or a site's .7z archive holding it.
    pub posts: PathBuf,
}

#[derive(clap::Args, Clone, Debug)]
pub struct CutArgs {
    /// Questions scored at least this.
    #[arg(long, default_value_t = 3)]
    pub min_score: i32,
    /// Questions scored below this only, for a run of one score band; no bound when not given.
    #[arg(long)]
    pub below_score: Option<i32>,
    /// Questions without an accepted answer or an answer scored 1 or more too.
    #[arg(long)]
    pub unanswered: bool,
    /// Answers a page holds at most: the accepted one first, then the best scored.
    #[arg(long, default_value_t = 3)]
    pub answers: usize,
    /// An answer other than the accepted one is kept from this score.
    #[arg(long, default_value_t = 1)]
    pub min_answer_score: i32,
}

#[derive(clap::Args, Clone, Debug)]
pub struct SizesArgs {
    #[command(flatten)]
    pub site: SiteArgs,
    /// Answers a page holds at most.
    #[arg(long, default_value_t = 3)]
    pub answers: usize,
    /// An answer other than the accepted one is kept from this score.
    #[arg(long, default_value_t = 1)]
    pub min_answer_score: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuestionStats {
    pub id: u32,
    pub score: i32,
    pub accepted: u32,
    pub body_len: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnswerStats {
    pub id: u32,
    pub parent: u32,
    pub score: i32,
    pub body_len: u32,
}

/// The first pass: every question's and answer's score and body size, and the newest post's time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Skeleton {
    pub questions: Vec<QuestionStats>,
    pub answers: Vec<AnswerStats>,
    pub newest: String,
}

/// The host a dump archive is named for: `stackoverflow.com-Posts.7z` and `math.stackexchange.com.7z`.
pub fn host_of(posts: &Path) -> String {
    let name = posts.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let name = name.strip_suffix(".7z").unwrap_or(&name);
    name.strip_suffix("-Posts").unwrap_or(name).to_string()
}

impl SiteArgs {
    pub fn host(&self) -> String {
        if self.host.is_empty() {
            host_of(&self.posts)
        } else {
            self.host.clone()
        }
    }

    pub fn users(&self) -> PathBuf {
        if let Some(users) = &self.users {
            return users.clone();
        }
        let name = self.posts.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        match name.strip_suffix("-Posts.7z") {
            Some(stem) => self.posts.with_file_name(format!("{stem}-Users.7z")),
            None if name.ends_with(".7z") => self.posts.clone(),
            None => self.posts.with_file_name("Users.xml"),
        }
    }

    pub fn site_name(&self) -> String {
        if self.site_name.is_empty() {
            self.host()
        } else {
            self.site_name.clone()
        }
    }
}

/// The posts' XML, read from the file or through `7z` from the archive.
struct Posts {
    child: Option<Child>,
    reader: quick_xml::Reader<Box<dyn BufRead>>,
    buf: Vec<u8>,
}

impl Posts {
    fn open(path: &Path, member: &str) -> Result<Self> {
        let (child, input): (Option<Child>, Box<dyn Read>) = if path.extension().is_some_and(|e| e == "7z") {
            let mut child =
                Command::new("7z").arg("e").arg("-so").arg(path).arg(member).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().context("starting 7z")?;
            let stdout = child.stdout.take().context("7z has no output")?;
            (Some(child), Box::new(stdout))
        } else {
            (None, Box::new(File::open(path).with_context(|| format!("opening {}", path.display()))?))
        };
        let input: Box<dyn BufRead> = Box::new(BufReader::with_capacity(8 << 20, input));
        Ok(Posts { child, reader: quick_xml::Reader::from_reader(input), buf: Vec::with_capacity(1 << 16) })
    }

    /// Calls `f` with each `<row>` until the file ends or `f` says to stop.
    fn each(&mut self, mut f: impl FnMut(&BytesStart) -> Result<bool>) -> Result<()> {
        loop {
            self.buf.clear();
            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) if e.local_name().as_ref() == b"row" => {
                    if !f(&e)? {
                        return self.stop();
                    }
                }
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(e) => bail!("XML error at byte {}: {e}", self.reader.buffer_position()),
            }
        }
        if let Some(mut child) = self.child.take() {
            let status = child.wait()?;
            if !status.success() {
                bail!("7z failed: {status}");
            }
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(())
    }
}

/// The id, post type, score, parent, accepted answer and raw body length of a row, without decoding its text.
#[derive(Default)]
struct RowStats {
    id: u32,
    kind: u8,
    score: i32,
    parent: u32,
    accepted: u32,
    body_len: u32,
    created: Vec<u8>,
}

fn row_stats(e: &BytesStart) -> Result<RowStats> {
    let mut row = RowStats::default();
    for a in e.attributes() {
        let a = a?;
        let value = a.value.as_ref();
        let number = || std::str::from_utf8(value).ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
        match a.key.as_ref() {
            b"Id" => row.id = number() as u32,
            b"PostTypeId" => row.kind = number() as u8,
            b"Score" => row.score = number() as i32,
            b"ParentId" => row.parent = number() as u32,
            b"AcceptedAnswerId" => row.accepted = number() as u32,
            b"Body" => row.body_len = value.len() as u32,
            b"CreationDate" => row.created = value.to_vec(),
            _ => {}
        }
    }
    Ok(row)
}

pub fn first_pass(path: &Path) -> Result<Skeleton> {
    let mut skeleton = Skeleton::default();
    let mut newest = Vec::new();
    Posts::open(path, POSTS_XML)?.each(|e| {
        let RowStats { id, kind, score, parent, accepted, body_len, created } = row_stats(e)?;
        match kind {
            QUESTION => skeleton.questions.push(QuestionStats { id, score, accepted, body_len }),
            ANSWER => skeleton.answers.push(AnswerStats { id, parent, score, body_len }),
            _ => {}
        }
        if created > newest {
            newest = created;
        }
        Ok(true)
    })?;
    skeleton.newest = String::from_utf8_lossy(&newest).to_string();
    Ok(skeleton)
}

impl Skeleton {
    pub fn write(&self, path: &Path) -> Result<()> {
        let partial = path.with_extension("partial");
        let mut out = BufWriter::new(File::create(&partial)?);
        out.write_all(SKELETON_MAGIC)?;
        let newest = self.newest.as_bytes();
        out.write_all(&(newest.len() as u32).to_le_bytes())?;
        out.write_all(newest)?;
        out.write_all(&(self.questions.len() as u64).to_le_bytes())?;
        for q in &self.questions {
            for n in [q.id, q.score as u32, q.accepted, q.body_len] {
                out.write_all(&n.to_le_bytes())?;
            }
        }
        out.write_all(&(self.answers.len() as u64).to_le_bytes())?;
        for a in &self.answers {
            for n in [a.id, a.parent, a.score as u32, a.body_len] {
                out.write_all(&n.to_le_bytes())?;
            }
        }
        out.into_inner()?.sync_all()?;
        std::fs::rename(&partial, path)?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != SKELETON_MAGIC {
            bail!("{} is not a skeleton file", path.display());
        }
        let mut word = [0u8; 4];
        let mut long = [0u8; 8];
        let mut u32_of = |input: &mut BufReader<File>| -> Result<u32> {
            input.read_exact(&mut word)?;
            Ok(u32::from_le_bytes(word))
        };
        let newest_len = u32_of(&mut input)? as usize;
        let mut newest = vec![0u8; newest_len];
        input.read_exact(&mut newest)?;
        input.read_exact(&mut long)?;
        let mut questions = Vec::with_capacity(u64::from_le_bytes(long) as usize);
        for _ in 0..u64::from_le_bytes(long) {
            questions.push(QuestionStats {
                id: u32_of(&mut input)?,
                score: u32_of(&mut input)? as i32,
                accepted: u32_of(&mut input)?,
                body_len: u32_of(&mut input)?,
            });
        }
        input.read_exact(&mut long)?;
        let mut answers = Vec::with_capacity(u64::from_le_bytes(long) as usize);
        for _ in 0..u64::from_le_bytes(long) {
            answers.push(AnswerStats {
                id: u32_of(&mut input)?,
                parent: u32_of(&mut input)?,
                score: u32_of(&mut input)? as i32,
                body_len: u32_of(&mut input)?,
            });
        }
        Ok(Skeleton { questions, answers, newest: String::from_utf8_lossy(&newest).to_string() })
    }

    /// The skeleton file when it exists, else the first pass, kept in the file when one is named.
    pub fn load(site: &SiteArgs) -> Result<Self> {
        match &site.skeleton {
            Some(path) if path.exists() => Self::read(path),
            Some(path) => {
                let skeleton = first_pass(&site.posts)?;
                skeleton.write(path)?;
                Ok(skeleton)
            }
            None => first_pass(&site.posts),
        }
    }

    /// Each question's answers, best first.
    pub fn answers_by_question(&self) -> HashMap<u32, Vec<AnswerStats>> {
        let mut by: HashMap<u32, Vec<AnswerStats>> = HashMap::new();
        for a in &self.answers {
            by.entry(a.parent).or_default().push(*a);
        }
        for list in by.values_mut() {
            list.sort_by_key(|a| (std::cmp::Reverse(a.score), a.id));
        }
        by
    }
}

/// The answers a question's page holds: the accepted one first, then the best scored from `min_score`, `most` in all.
pub fn chosen(question: &QuestionStats, answers: &[AnswerStats], most: usize, min_score: i32) -> Vec<AnswerStats> {
    let mut out: Vec<AnswerStats> = answers.iter().filter(|a| a.id == question.accepted).copied().collect();
    for a in answers {
        if out.len() >= most {
            break;
        }
        if a.id != question.accepted && a.score >= min_score {
            out.push(*a);
        }
    }
    out.truncate(most);
    out
}

/// Whether the question is in the cut, given the answers its page would hold.
pub fn in_cut(question: &QuestionStats, answers: &[AnswerStats], cut: &CutArgs) -> bool {
    question.score >= cut.min_score
        && cut.below_score.is_none_or(|below| question.score < below)
        && (cut.unanswered || answers.iter().any(|a| a.id == question.accepted || a.score >= 1))
}

fn ids_file(path: &Path) -> Result<HashSet<u32>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(text.lines().filter_map(|l| l.trim().parse().ok()).collect())
}

/// One post as the page needs it.
#[derive(Clone, Debug, Default)]
pub struct Post {
    pub id: u32,
    pub score: i32,
    pub title: String,
    pub body: String,
    pub tags: Vec<String>,
    pub created: String,
    pub active: String,
    pub licence: String,
    pub owner: u32,
    pub owner_name: String,
}

/// Every user's display name by id.
pub fn users(path: &Path) -> Result<HashMap<u32, String>> {
    let mut names = HashMap::new();
    Posts::open(path, USERS_XML)?.each(|e| {
        let (mut id, mut name) = (0u32, String::new());
        for a in e.attributes() {
            let a = a?;
            match a.key.as_ref() {
                b"Id" => id = std::str::from_utf8(a.value.as_ref()).ok().and_then(|v| v.parse().ok()).unwrap_or(0),
                b"DisplayName" => name = a.normalized_value(XmlVersion::Implicit1_0)?.into_owned(),
                _ => {}
            }
        }
        if id > 0 && !name.is_empty() {
            names.insert(id, name);
        }
        Ok(true)
    })?;
    Ok(names)
}

/// The author as the dump's licence asks: the name linked to the profile, the name alone for a removed user.
fn author_of(host: &str, post: &Post, users: &HashMap<u32, String>) -> Option<String> {
    match users.get(&post.owner) {
        Some(name) => {
            let slug = slug::slug(name);
            let profile = if slug.is_empty() { format!("https://{host}/users/{}", post.owner) } else { format!("https://{host}/users/{}/{slug}", post.owner) };
            Some(format!("{name} ({profile})"))
        }
        None => Some(post.owner_name.clone()).filter(|n| !n.is_empty()),
    }
}

fn post_of(e: &BytesStart) -> Result<Post> {
    let mut post = Post::default();
    for a in e.attributes() {
        let a = a?;
        let key = a.key.as_ref();
        if !matches!(
            key,
            b"Id" | b"Score" | b"Title" | b"Body" | b"Tags" | b"CreationDate" | b"LastActivityDate" | b"ContentLicense" | b"OwnerUserId" | b"OwnerDisplayName"
        ) {
            continue;
        }
        let value = a.normalized_value(XmlVersion::Implicit1_0)?.into_owned();
        match key {
            b"Id" => post.id = value.parse().unwrap_or(0),
            b"Score" => post.score = value.parse().unwrap_or(0),
            b"Title" => post.title = value,
            b"Body" => post.body = value,
            b"Tags" => post.tags = tags_of(&value),
            b"CreationDate" => post.created = iso_utc(&value),
            b"LastActivityDate" => post.active = iso_utc(&value),
            b"ContentLicense" => post.licence = value,
            b"OwnerUserId" => post.owner = value.parse().unwrap_or(0),
            b"OwnerDisplayName" => post.owner_name = value,
            _ => {}
        }
    }
    Ok(post)
}

/// `|python|list|` or `<python><list>` as its tag names.
pub fn tags_of(value: &str) -> Vec<String> {
    value.split(['|', '<', '>']).filter(|t| !t.is_empty()).map(String::from).collect()
}

/// A dump time, `2008-07-31T21:42:52.667`, as `2008-07-31T21:42:52Z`.
pub fn iso_utc(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    format!("{}Z", value.split('.').next().unwrap_or(value))
}

pub fn question_url(host: &str, id: u32, title: &str) -> String {
    let slug = slug::slug(title);
    if slug.is_empty() {
        format!("https://{host}/questions/{id}")
    } else {
        format!("https://{host}/questions/{id}/{slug}")
    }
}

/// The site the pages are of, with its users' names for the credit line.
pub struct Site {
    pub host: String,
    pub name: String,
    pub users: HashMap<u32, String>,
}

/// The question's page: its body and tags, then each answer under its own heading, then the credit line.
pub fn page_of(site: &Site, question: &Post, answers: &[(bool, Post)]) -> (String, Page) {
    let (host, site_name) = (site.host.as_str(), site.name.as_str());
    let url = question_url(host, question.id, &question.title);
    let mut lines = html::lines(&question.body);
    if !question.tags.is_empty() {
        lines.push(Line::new(format!("Tags: {}", question.tags.join(", "))));
    }
    for (accepted, answer) in answers {
        let body = html::lines(&answer.body);
        if body.is_empty() {
            continue;
        }
        lines.push(Line::new(if *accepted { "## Accepted answer" } else { "## Answer" }));
        lines.extend(body);
    }
    let mut licences: Vec<&str> =
        std::iter::once(question.licence.as_str()).chain(answers.iter().map(|(_, a)| a.licence.as_str())).filter(|l| !l.is_empty()).collect();
    licences.sort_unstable();
    licences.dedup();
    let licence = if licences.is_empty() { "CC BY-SA 4.0".to_string() } else { licences.join(", ") };
    let asker = author_of(host, question, &site.users);
    let mut answerers: Vec<String> = Vec::new();
    for (_, answer) in answers {
        if let Some(who) = author_of(host, answer, &site.users).filter(|who| !answerers.contains(who)) {
            answerers.push(who);
        }
    }
    let mut credit = format!("Source: {site_name}, {url}.");
    if let Some(asker) = &asker {
        credit.push_str(&format!(" Question by {asker}."));
    }
    if !answerers.is_empty() {
        credit.push_str(&format!(" {} by {}.", if answerers.len() == 1 { "Answer" } else { "Answers" }, answerers.join(", ")));
    }
    credit.push_str(&format!(" Licensed under {licence}."));
    lines.push(Line::new(credit));
    let meta = PaperMeta {
        lang: "en".into(),
        url: url.clone(),
        published: question.created.clone(),
        modified: question.active.clone(),
        site_name: site_name.to_string(),
        tags: question.tags.clone(),
        author: site.users.get(&question.owner).cloned().unwrap_or_else(|| question.owner_name.clone()),
    };
    (url, typed_page(PAGE_TYPE, &question.title, lines, meta))
}

struct Pending {
    question: Option<Post>,
    accepted: u32,
    left: usize,
    answers: Vec<Post>,
}

fn emit(batches: &mut Batches, site: &Site, pending: Pending, as_of_us: i64, taken: &mut Vec<u32>, counts: &mut Counts) -> Result<()> {
    let Some(question) = pending.question else {
        counts.add("question_missing", 1);
        return Ok(());
    };
    let mut answers: Vec<(bool, Post)> = pending.answers.into_iter().map(|a| (a.id == pending.accepted, a)).collect();
    answers.sort_by_key(|(accepted, a)| (!*accepted, std::cmp::Reverse(a.score), a.id));
    let (url, page) = page_of(site, &question, &answers);
    counts.add("answers", answers.len() as u64);
    counts.add("text_chars", page.text.chars().count() as u64);
    batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
    counts.add("pages", 1);
    taken.push(question.id);
    Ok(())
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let site = Site { host: args.site.host(), name: args.site.site_name(), users: users(&args.site.users())? };
    let skeleton = Skeleton::load(&args.site)?;
    let by_question = skeleton.answers_by_question();
    let parents: HashMap<u32, u32> = skeleton.answers.iter().map(|a| (a.id, a.parent)).collect();
    let included: HashSet<u32> = match &args.include_ids {
        Some(path) => ids_file(path)?.into_iter().map(|id| parents.get(&id).copied().unwrap_or(id)).collect(),
        None => HashSet::new(),
    };
    let skipped = match &args.skip_ids {
        Some(path) => ids_file(path)?,
        None => HashSet::new(),
    };
    let mut counts = Counts::default();
    let mut pending: HashMap<u32, Pending> = HashMap::new();
    let mut wanted_answers: HashMap<u32, u32> = HashMap::new();
    let none = Vec::new();
    for q in &skeleton.questions {
        if skipped.contains(&q.id) {
            counts.add("skipped", 1);
            continue;
        }
        let answers = by_question.get(&q.id).unwrap_or(&none);
        let picked = chosen(q, answers, args.cut.answers, args.cut.min_answer_score);
        let take = if args.only_included { included.contains(&q.id) } else { included.contains(&q.id) || in_cut(q, &picked, &args.cut) };
        if !take {
            continue;
        }
        if included.contains(&q.id) {
            counts.add("included", 1);
        }
        for a in &picked {
            wanted_answers.insert(a.id, q.id);
        }
        pending.insert(q.id, Pending { question: None, accepted: q.accepted, left: picked.len(), answers: Vec::new() });
    }
    counts.add("questions_in_dump", skeleton.questions.len() as u64);
    counts.add("planned", pending.len() as u64);
    drop(by_question);
    drop(parents);
    let limit = args.run.limit;
    let mut taken = Vec::with_capacity(pending.len());
    Posts::open(&args.site.posts, POSTS_XML)?.each(|e| {
        let (id, kind) = row_stats_head(e)?;
        let owner = match kind {
            QUESTION if pending.contains_key(&id) => id,
            ANSWER => match wanted_answers.get(&id) {
                Some(&q) => q,
                None => return Ok(true),
            },
            _ => return Ok(true),
        };
        let post = post_of(e)?;
        let Some(entry) = pending.get_mut(&owner) else { return Ok(true) };
        if kind == QUESTION {
            entry.question = Some(post);
        } else {
            entry.answers.push(post);
            entry.left = entry.left.saturating_sub(1);
        }
        if entry.left == 0 && entry.question.is_some() {
            let done = pending.remove(&owner).expect("pending question");
            emit(batches, &site, done, as_of_us, &mut taken, &mut counts)?;
        }
        Ok(limit == 0 || counts.get("pages") < limit)
    })?;
    let mut rest: Vec<(u32, Pending)> = pending.into_iter().filter(|(_, p)| p.question.is_some()).collect();
    rest.sort_by_key(|(id, _)| *id);
    for (_, p) in rest {
        if limit > 0 && counts.get("pages") >= limit {
            break;
        }
        counts.add("answers_missing", 1);
        emit(batches, &site, p, as_of_us, &mut taken, &mut counts)?;
    }
    let text: String = taken.iter().map(|id| format!("{id}\n")).collect();
    bulk_core::rows::write_synced(&args.run.out.join(TAKEN), text.as_bytes())?;
    Ok(counts)
}

fn row_stats_head(e: &BytesStart) -> Result<(u32, u8)> {
    let (mut id, mut kind) = (0u32, 0u8);
    for a in e.attributes() {
        let a = a?;
        let parse = || std::str::from_utf8(a.value.as_ref()).ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        match a.key.as_ref() {
            b"Id" => id = parse(),
            b"PostTypeId" => kind = parse() as u8,
            _ => {}
        }
        if id > 0 && kind > 0 {
            break;
        }
    }
    Ok((id, kind))
}

/// The page count and text size of each cut, as JSON lines; body sizes are of the dump's escaped HTML.
pub fn sizes(args: &SizesArgs) -> Result<Vec<serde_json::Value>> {
    let skeleton = Skeleton::load(&args.site)?;
    let by_question = skeleton.answers_by_question();
    let none = Vec::new();
    let mut out = Vec::new();
    for unanswered in [false, true] {
        for min_score in [0, 1, 2, 3, 5, 10, 25] {
            let cut = CutArgs { min_score, below_score: None, unanswered, answers: args.answers, min_answer_score: args.min_answer_score };
            let (mut pages, mut answers, mut bytes) = (0u64, 0u64, 0u64);
            for q in &skeleton.questions {
                let picked = chosen(q, by_question.get(&q.id).unwrap_or(&none), cut.answers, cut.min_answer_score);
                if in_cut(q, &picked, &cut) {
                    pages += 1;
                    answers += picked.len() as u64;
                    bytes += u64::from(q.body_len) + picked.iter().map(|a| u64::from(a.body_len)).sum::<u64>();
                }
            }
            out.push(serde_json::json!({"host": args.site.host(), "min_score": min_score, "unanswered": unanswered, "pages": pages, "answers": answers, "body_bytes": bytes}));
        }
    }
    out.push(serde_json::json!({"host": args.site.host(), "questions": skeleton.questions.len(), "answers_in_dump": skeleton.answers.len(), "newest": skeleton.newest}));
    Ok(out)
}

pub fn sizes_main(args: SizesArgs) -> Result<()> {
    for line in sizes(&args)? {
        println!("{line}");
    }
    Ok(())
}

pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const POSTS: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" AcceptedAnswerId="3" CreationDate="2008-07-31T21:42:52.667" Score="12" OwnerUserId="7" Body="&lt;p&gt;How do I &lt;code&gt;sort&lt;/code&gt; a dict &amp;amp; keep it?&lt;/p&gt;&#xA;" Title="How do I sort a dict by value?" Tags="|python|dictionary|" LastActivityDate="2024-01-01T00:00:00.000" ContentLicense="CC BY-SA 4.0" />
  <row Id="2" PostTypeId="1" CreationDate="2008-08-01T00:00:00.000" Score="1" Body="&lt;p&gt;low&lt;/p&gt;" Title="Low score question" Tags="|x|" ContentLicense="CC BY-SA 2.5" />
  <row Id="3" PostTypeId="2" ParentId="1" CreationDate="2008-08-01T01:00:00.000" Score="5" Body="&lt;pre&gt;&lt;code&gt;sorted(d.items(), key=lambda kv: kv[1])&#xA;&lt;/code&gt;&lt;/pre&gt;" ContentLicense="CC BY-SA 3.0" />
  <row Id="4" PostTypeId="2" ParentId="1" CreationDate="2008-08-02T01:00:00.000" Score="40" OwnerDisplayName="gone user" Body="&lt;p&gt;Use &lt;code&gt;operator.itemgetter&lt;/code&gt;.&lt;/p&gt;" ContentLicense="CC BY-SA 4.0" />
  <row Id="5" PostTypeId="2" ParentId="1" CreationDate="2008-08-03T01:00:00.000" Score="0" Body="&lt;p&gt;zero&lt;/p&gt;" ContentLicense="CC BY-SA 4.0" />
  <row Id="6" PostTypeId="2" ParentId="2" CreationDate="2009-08-03T01:00:00.000" Score="2" Body="&lt;p&gt;an answer&lt;/p&gt;" ContentLicense="CC BY-SA 4.0" />
</posts>
"#;

    const USERS: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<users>
  <row Id="7" Reputation="10" DisplayName="Jane Doe" />
</users>
"#;

    fn site(dir: &Path) -> SiteArgs {
        let posts = dir.join("Posts.xml");
        std::fs::write(&posts, POSTS).unwrap();
        std::fs::write(dir.join("Users.xml"), USERS).unwrap();
        SiteArgs { host: "stackoverflow.com".into(), site_name: "Stack Overflow".into(), skeleton: Some(dir.join("skeleton.bin")), users: None, posts }
    }

    fn args(dir: &Path, out: &str) -> Args {
        Args {
            run: RunArgs { out: dir.join(out), as_of: "2024-04-01T00:00:00Z".into(), run: "20240401-stackoverflow".into(), limit: 0 },
            site: site(dir),
            cut: CutArgs { min_score: 3, below_score: None, unanswered: false, answers: 3, min_answer_score: 1 },
            include_ids: None,
            only_included: false,
            skip_ids: None,
        }
    }

    #[test]
    fn host_from_archive_names() {
        assert_eq!(host_of(Path::new("/d/stackoverflow.com-Posts.7z")), "stackoverflow.com");
        assert_eq!(host_of(Path::new("math.stackexchange.com.7z")), "math.stackexchange.com");
    }

    #[test]
    fn skeleton_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let skeleton = Skeleton::load(&site(dir.path())).unwrap();
        assert_eq!(skeleton.questions.len(), 2);
        assert_eq!(skeleton.answers.len(), 4);
        assert_eq!(skeleton.newest, "2009-08-03T01:00:00.000");
        assert_eq!(Skeleton::read(&dir.path().join("skeleton.bin")).unwrap(), skeleton);
    }

    #[test]
    fn question_page_holds_accepted_then_best_answers() {
        let dir = tempfile::tempdir().unwrap();
        let args = args(dir.path(), "run");
        let (mut batches, as_of) = bulk_core::run::start(&args.run, SOURCE).unwrap();
        let counts = run(&args, &mut batches, as_of).unwrap();
        assert_eq!(counts.get("pages"), 1);
        assert_eq!(counts.get("answers"), 2);
        assert_eq!(std::fs::read_to_string(dir.path().join("run").join(TAKEN)).unwrap(), "1\n");
        let site = args.site.clone();
        let skeleton = Skeleton::load(&site).unwrap();
        let q = skeleton.questions[0];
        let picked = chosen(&q, &skeleton.answers_by_question()[&1], 3, 1);
        assert_eq!(picked.iter().map(|a| a.id).collect::<Vec<_>>(), vec![3, 4]);
    }

    #[test]
    fn page_text_and_credit() {
        let question = Post {
            id: 1,
            score: 12,
            title: "How do I sort a dict by value?".into(),
            body: "<p>How do I <code>sort</code> a dict &amp; keep it?</p>".into(),
            tags: vec!["python".into(), "dictionary".into()],
            created: "2008-07-31T21:42:52Z".into(),
            licence: "CC BY-SA 4.0".into(),
            owner: 7,
            ..Post::default()
        };
        let accepted = Post {
            id: 3,
            body: "<pre><code>sorted(d.items())\n</code></pre>".into(),
            licence: "CC BY-SA 3.0".into(),
            owner_name: "gone user".into(),
            ..Post::default()
        };
        let site = Site { host: "stackoverflow.com".into(), name: "Stack Overflow".into(), users: HashMap::from([(7, "Jane Doe".to_string())]) };
        let (url, page) = page_of(&site, &question, &[(true, accepted)]);
        assert_eq!(page.author, "Jane Doe");
        assert_eq!(url, "https://stackoverflow.com/questions/1/how-do-i-sort-a-dict-by-value");
        assert_eq!(page.page_type, PAGE_TYPE);
        assert_eq!(page.published, "2008-07-31T21:42:52Z");
        assert_eq!(
            page.text,
            "How do I sort a dict by value?\nHow do I `sort` a dict & keep it?\nTags: python, dictionary\n## Accepted answer\n```\nsorted(d.items())\n```\nSource: Stack Overflow, https://stackoverflow.com/questions/1/how-do-i-sort-a-dict-by-value. Question by Jane Doe (https://stackoverflow.com/users/7/jane-doe). Answer by gone user. Licensed under CC BY-SA 3.0, CC BY-SA 4.0."
        );
        assert_eq!(page.headings, vec!["Accepted answer"]);
    }

    #[test]
    fn score_band() {
        let q = QuestionStats { id: 1, score: 5, accepted: 2, body_len: 0 };
        let a = [AnswerStats { id: 2, parent: 1, score: 0, body_len: 0 }];
        let band = |min_score, below_score| in_cut(&q, &a, &CutArgs { min_score, below_score, unanswered: false, answers: 3, min_answer_score: 1 });
        assert!(band(3, None) && band(5, Some(10)) && !band(0, Some(5)) && !band(6, None));
    }

    #[test]
    fn included_and_skipped_ids() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("cited.txt"), "6\n").unwrap();
        let mut only = args(dir.path(), "only");
        only.include_ids = Some(dir.path().join("cited.txt"));
        only.only_included = true;
        let (mut batches, as_of) = bulk_core::run::start(&only.run, SOURCE).unwrap();
        let counts = run(&only, &mut batches, as_of).unwrap();
        assert_eq!((counts.get("pages"), counts.get("included")), (1, 1));
        assert_eq!(std::fs::read_to_string(dir.path().join("only").join(TAKEN)).unwrap(), "2\n");
        let mut rest = args(dir.path(), "rest");
        rest.skip_ids = Some(dir.path().join("only").join(TAKEN));
        let (mut batches, as_of) = bulk_core::run::start(&rest.run, SOURCE).unwrap();
        let counts = run(&rest, &mut batches, as_of).unwrap();
        assert_eq!((counts.get("pages"), counts.get("skipped")), (1, 1));
    }

    #[test]
    fn sizes_per_cut() {
        let dir = tempfile::tempdir().unwrap();
        let lines = sizes(&SizesArgs { site: site(dir.path()), answers: 3, min_answer_score: 1 }).unwrap();
        let at =
            |score: i64, unanswered: bool| lines.iter().find(|l| l["min_score"] == score && l["unanswered"] == unanswered).unwrap()["pages"].as_u64().unwrap();
        assert_eq!((at(0, false), at(3, false), at(25, true)), (2, 1, 0));
    }
}

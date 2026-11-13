//! Project Gutenberg: a page per English book public domain in the USA, its licence and every Project Gutenberg mention taken out, the text cut to its opening.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use regex::Regex;

pub const SOURCE: &str = "gutenberg";
pub const BOOK: &str = "article";
const CONTENT_TYPE: &str = "text/plain";
/// The opening a page keeps, about 32 passages of the planner's size, so a book never floods the passage lane.
pub const MAX_CHARS: usize = 48_000;
const PUBLIC_DOMAIN: &str = "Public domain in the USA.";
static ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"cache/epub/(\d+)/pg(\d+)\.txt$").expect("a valid pattern"));
static RIGHTS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<dcterms:rights>([^<]*)</dcterms:rights>").expect("a valid pattern"));
static START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK").expect("a valid pattern"));
static END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK").expect("a valid pattern"));
static NAMED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)project\s+gutenberg|gutenberg\.org|gutenberg[-\s]*(tm|™)|pglaf").expect("a valid pattern"));
static ILLUSTRATION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\[illustration[^\]]*\]?").expect("a valid pattern"));
static COPYRIGHTED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)this is a copyrighted project gutenberg e-?book").expect("a valid pattern"));

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The catalog, pg_catalog.csv.
    #[arg(long)]
    pub catalog: PathBuf,
    /// The unpacked rdf-files.tar.bz2, its cache/epub/<id>/pg<id>.rdf files.
    #[arg(long)]
    pub rdf: PathBuf,
    /// Characters of each book's text a page keeps.
    #[arg(long, default_value_t = MAX_CHARS)]
    pub max_chars: usize,
    /// txt-files.tar.zip, every book's plain text.
    pub texts: PathBuf,
}

/// One catalog row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Book {
    pub id: String,
    pub title: String,
    pub authors: String,
    pub subjects: String,
    pub issued: String,
}

/// The catalog's English texts, by ebook number.
pub fn english_texts(catalog: &str) -> Result<HashMap<String, Book>> {
    let mut reader = csv::Reader::from_reader(catalog.as_bytes());
    let mut found = HashMap::new();
    for row in reader.deserialize::<HashMap<String, String>>() {
        let row = row?;
        let get = |k: &str| row.get(k).map(|v| v.trim().to_string()).unwrap_or_default();
        if get("Type") != "Text" || get("Language") != "en" {
            continue;
        }
        let id = get("Text#");
        let title = get("Title").lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(": ");
        found.insert(id.clone(), Book { id, title, authors: get("Authors"), subjects: get("Subjects"), issued: get("Issued") });
    }
    Ok(found)
}

/// The book's rights statement from its RDF record.
pub fn rights(rdf: &str) -> String {
    RIGHTS.captures(rdf).map(|c| c[1].trim().to_string()).unwrap_or_default()
}

/// The book's own text: between the start and end markers, without the licence and any line naming Project Gutenberg; None when it says it is copyrighted.
pub fn body(text: &str) -> Option<Vec<String>> {
    if COPYRIGHTED.is_match(text) {
        return None;
    }
    let lines: Vec<&str> = text.trim_start_matches('\u{feff}').lines().collect();
    let start = lines.iter().position(|l| START.is_match(l.trim())).map(|i| i + 1).unwrap_or(0);
    let end = lines.iter().skip(start).position(|l| END.is_match(l.trim())).map(|i| i + start).unwrap_or(lines.len());
    let mut paragraphs = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in &lines[start..end] {
        let line = line.trim();
        if line.is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        paragraphs.push(current.join(" "));
    }
    Some(
        paragraphs
            .into_iter()
            .map(|p| ILLUSTRATION.replace_all(&p, " ").split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|p| !p.is_empty() && !NAMED.is_match(p))
            .collect(),
    )
}

/// The paragraphs up to `most` characters, cut at a paragraph's end.
pub fn opening(paragraphs: Vec<String>, most: usize) -> (Vec<String>, bool) {
    let mut kept = Vec::new();
    let mut total = 0;
    for p in paragraphs {
        let n = p.chars().count();
        if total + n > most && !kept.is_empty() {
            return (kept, true);
        }
        total += n + 1;
        kept.push(p);
    }
    (kept, false)
}

pub fn page_of(book: &Book, paragraphs: Vec<String>, cut: bool) -> (String, Page) {
    let url = format!("https://www.gutenberg.org/ebooks/{}", book.id);
    let mut lines = vec![Line::new(format!("Source: Project Gutenberg ebook #{}, {url} (public domain in the USA).", book.id))];
    for (label, value) in [("Author", &book.authors), ("Subjects", &book.subjects), ("Released", &book.issued)] {
        if !value.is_empty() {
            lines.push(Line::new(format!("{label}: {value}")));
        }
    }
    if cut {
        lines.push(Line::new("The opening of the book; the whole text is at the source."));
    }
    lines.extend(paragraphs.into_iter().map(Line::new));
    // The licence allows naming Project Gutenberg only as an acknowledgement with a link, so the site name stays empty.
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), published: book.issued.clone(), author: book.authors.clone(), ..PaperMeta::default() };
    (url, typed_page(BOOK, &book.title, lines, meta))
}

fn rdf_rights(folder: &Path, id: &str) -> String {
    std::fs::read_to_string(folder.join("cache/epub").join(id).join(format!("pg{id}.rdf"))).map(|r| rights(&r)).unwrap_or_default()
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let books = english_texts(&std::fs::read_to_string(&args.catalog).with_context(|| format!("reading {}", args.catalog.display()))?)?;
    let mut counts = Counts::default();
    let file = std::fs::File::open(&args.texts).with_context(|| format!("opening {}", args.texts.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    let inner = archive.file_names().find(|n| n.ends_with(".tar")).map(String::from).context("no tar in the texts zip")?;
    let mut tar = tar::Archive::new(archive.by_name(&inner)?);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(c) = ID.captures(&path) else { continue };
        if c[1] != c[2] {
            continue;
        }
        let id = c[1].to_string();
        let Some(book) = books.get(&id) else { continue };
        if [&book.title, &book.authors, &book.subjects].iter().any(|field| NAMED.is_match(field)) {
            counts.add("project_gutenberg_work", 1);
            continue;
        }
        if rdf_rights(&args.rdf, &id) != PUBLIC_DOMAIN {
            counts.add("not_public_domain", 1);
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        let Some(paragraphs) = body(&String::from_utf8_lossy(&bytes)) else {
            counts.add("copyrighted_banner", 1);
            continue;
        };
        let (kept, cut) = opening(paragraphs, args.max_chars);
        if kept.is_empty() {
            counts.add("empty", 1);
            continue;
        }
        let (url, page) = page_of(book, kept, cut);
        batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
        counts.add("pages", 1);
        if cut {
            counts.add("cut", 1);
        }
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            break;
        }
    }
    Ok(counts)
}

pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "\u{feff}The Project Gutenberg eBook of Pride and Prejudice\n\nThis ebook is for the use of anyone anywhere.\n\n*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n\nProduced by a volunteer for Project Gutenberg.\n\n[Illustration: 1894]\n\nIt is a truth universally\nacknowledged, that a single man.\n\nChapter I\n\nHowever little known.\n\n*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n\nUpdated editions will replace the previous one.\n";

    #[test]
    fn licence_and_mentions_out_paragraphs_joined() {
        let paragraphs = body(TEXT).unwrap();
        assert_eq!(paragraphs, ["It is a truth universally acknowledged, that a single man.", "Chapter I", "However little known."]);
        assert!(body("*** This is a COPYRIGHTED Project Gutenberg eBook, Details Below ***").is_none());
    }

    #[test]
    fn opening_stops_at_a_paragraph() {
        let (kept, cut) = opening(vec!["a".repeat(10), "b".repeat(10), "c".repeat(10)], 25);
        assert_eq!((kept.len(), cut), (2, true));
        let (kept, cut) = opening(vec!["a".repeat(30)], 25);
        assert_eq!((kept.len(), cut), (1, false));
    }

    #[test]
    fn catalog_english_texts_and_rights() {
        let csv = "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n1342,Text,1998-06-01,\"Pride and Prejudice\nA novel\",en,\"Austen, Jane, 1775-1817\",England -- Fiction,PR,\n2,Sound,1972-12-01,Bill,en,,,,\n3,Text,1999-01-01,Un livre,fr,,,,\n";
        let books = english_texts(csv).unwrap();
        assert_eq!(books.len(), 1);
        assert_eq!(books["1342"].title, "Pride and Prejudice: A novel");
        assert_eq!(rights("<dcterms:rights>Public domain in the USA.</dcterms:rights>"), PUBLIC_DOMAIN);
    }

    #[test]
    fn page_credits_the_ebook_and_says_it_is_cut() {
        let book = Book { id: "1342".into(), title: "Pride and Prejudice".into(), authors: "Austen, Jane, 1775-1817".into(), issued: "1998-06-01".into(), ..Book::default() };
        let (url, page) = page_of(&book, vec!["It is a truth universally acknowledged.".into()], true);
        assert_eq!(url, "https://www.gutenberg.org/ebooks/1342");
        let lines: Vec<&str> = page.text.lines().collect();
        assert_eq!(lines[1], "Source: Project Gutenberg ebook #1342, https://www.gutenberg.org/ebooks/1342 (public domain in the USA).");
        assert!(lines.contains(&"The opening of the book; the whole text is at the source."));
        assert_eq!((page.author.as_str(), page.published.as_str(), page.site_name.as_str()), ("Austen, Jane, 1775-1817", "1998-06-01", ""));
    }
}

//! The planner: the texts a page gives, and the change log turned into embed inputs within the budget and the task API's room.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use bytes::Bytes;
use desearch::feeds::CHANGES;
use desearch::r2::{self, Bucket, Credentials, JSON, PARQUET};
use desearch::stub;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use publisher::changes::{self, Body, Change, Kind};
use publisher::current::{self, FileRef, Manifest, PageRow};
use publisher::planner::{
    page_text, page_text_v2, passages, passages_v2, texts, Listed, Passage, Planner, Room, Settings, EMBED_DROPPED, EMBED_INPUTS, PAGE, PASSAGE, RECIPE_V2,
};
use publisher::records::{iso, page_key, record_version, Record};
use publisher::set::Set;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const REDIS: &str = "redis://127.0.0.1:6379/12";

fn record(url: &str, text: &str, page_type: &str, lang: &str) -> Record {
    Record {
        url: url.into(),
        domain: "news.example".into(),
        title: "A title".into(),
        lang: lang.into(),
        text: text.into(),
        fetched_at: "2026-10-14T10:00:00+00:00".into(),
        content_sha1: format!("{:x}", Sha256::digest(text.as_bytes())),
        page_type: Some(page_type.into()),
        task_id: "t1".into(),
        miner: "5Miner".into(),
        ..Record::default()
    }
}

fn v2(record: Record) -> Record {
    Record { extraction: Some("v2".into()), ..record }
}

fn change(record: Record) -> Change {
    Change {
        key: page_key(&record.url).unwrap(),
        kind: Kind::Page,
        previous_content_sha1: String::new(),
        published_at: "2026-10-14T10:00:00+00:00".into(),
        body: Body::Page { version: record_version(&record), record: Box::new(record) },
    }
}

fn article(paragraphs: usize) -> String {
    (0..paragraphs).map(|i| format!("Paragraph {i}: {}", "word ".repeat(60))).collect::<Vec<_>>().join("\n")
}

#[test]
fn every_page_gets_its_page_text_and_only_long_articles_get_passages() {
    let mut landing = record("https://news.example/about", &article(40), "landing", "en");
    landing.description = Some("What we do".into());
    let page = texts("pages/news.example/a", &landing);
    assert_eq!(page.len(), 1, "a landing page is one text however long");
    assert_eq!((page[0].text_id.as_str(), page[0].kind, page[0].index), ("pages/news.example/a#page", PAGE, 0));
    assert!(page[0].text.starts_with("A title\nWhat we do\nParagraph 0:"));
    assert_eq!(page[0].text.chars().count(), 10_000, "cut to what the model reads");

    let short = texts("k", &record("https://news.example/s", &article(3), "article", "en"));
    assert_eq!(short.len(), 1, "a short article is whole in its page text");

    let long = texts("k", &record("https://news.example/l", &article(40), "article", "en"));
    let parts: Vec<_> = long.iter().skip(1).collect();
    assert_eq!(parts.len(), 16, "at most 16 passages");
    assert!(parts.iter().enumerate().all(|(i, p)| p.kind == PASSAGE && p.index == i as i32 + 1 && p.text_id == format!("k#p{}", i + 1)));
    assert!(parts.iter().all(|p| p.text.starts_with("A title\nParagraph ")), "each passage starts with the title");

    let mut sectioned = record("https://news.example/h", &format!("{}\nHow it works\n{}", article(4), article(4)), "article", "en");
    sectioned.headings = vec![Some("How it works".into())];
    let led: Vec<String> = texts("k", &sectioned).into_iter().skip(1).map(|t| t.text.lines().next().unwrap().to_string()).collect();
    assert_eq!(led.first().map(String::as_str), Some("A title"));
    assert_eq!(led.last().map(String::as_str), Some("A title › How it works"), "a section's passages say where they are");
}

#[test]
fn passages_group_paragraphs_and_split_long_lines_at_sentences() {
    let grouped = passages(&["a".repeat(400), "b".repeat(400), "c".repeat(100)].join("\n"), &[]);
    let bodies: Vec<String> = grouped.iter().map(|p| p.body.clone()).collect();
    assert_eq!(bodies, [format!("{}\n{}\n{}", "a".repeat(400), "b".repeat(400), "c".repeat(100))], "short paragraphs join; a small tail joins the last");

    let sentence = format!("{}. ", "x".repeat(299));
    let long_line = sentence.repeat(12);
    let pieces: Vec<String> = passages(&long_line, &[]).into_iter().map(|p| p.body).collect();
    assert!(pieces.iter().all(|p| p.chars().count() <= 1_500 + 300), "{:?}", pieces.iter().map(|p| p.chars().count()).collect::<Vec<_>>());
    assert!(pieces.iter().all(|p| p.ends_with('.')), "split after a sentence");
    assert_eq!(pieces.concat().replace(['.', ' ', '\n'], "").len(), 12 * 299, "nothing lost");

    let sectioned = ["Intro line one.", &"x".repeat(700), "Pricing", &"p".repeat(300), &"q".repeat(400), "FAQ", "Short answer."].join("\n");
    let found = passages(&sectioned, &[Some("Pricing".into()), Some("FAQ".into()), None]);
    let shape: Vec<(Option<&str>, usize)> = found.iter().map(|p| (p.heading.as_deref(), p.body.chars().count())).collect();
    assert_eq!(shape, [(None, 716), (Some("Pricing"), 701), (Some("FAQ"), 13)], "a heading starts a section, and names the passages under it");
    assert!(!found.iter().any(|p| p.body.contains("Pricing")), "the heading leads the passage rather than sitting in it");

    let mut capped: Vec<String> = (0..15).map(|i| format!("{i}").repeat(600)).collect();
    capped.extend(["s".repeat(300), "Later".into()]);
    capped.extend((0..5).map(|_| "z".repeat(700)));
    assert_eq!(passages(&capped.join("\n"), &[Some("Later".into())]).len(), 16, "no more than 16, a heading closing the 16th included");
    let mut overflowing: Vec<String> = (0..15).map(|i| format!("{i}").repeat(600)).collect();
    overflowing.extend(["s".repeat(500), "o".repeat(1_200)]);
    assert_eq!(passages(&overflowing.join("\n"), &[]).len(), 16, "no more than 16 when a long line both closes one passage and fills the next");

    let mut description = record("https://news.example/d", "Opening line. More.", "other", "en");
    description.description = Some("Opening line.".into());
    assert_eq!(page_text(&description), "A title\nOpening line. More.", "a description the text opens with is not repeated");
}

#[test]
fn a_v2_page_text_holds_the_title_once_then_the_description_and_the_markdown() {
    let mut page = v2(record("https://news.example/a", "A title\n## Heading\nBody line.", "article", "en"));
    page.description = Some("What it is".into());
    assert_eq!(page_text_v2(&page), "A title\nWhat it is\n## Heading\nBody line.");
    assert_eq!(texts("k", &page)[0].text, page_text_v2(&page));
    page.text = "# A title\nBody line.".into();
    assert_eq!(page_text_v2(&page), "A title\nWhat it is\nBody line.", "the title as an h1 is the title line too");
    for first in ["# A TITLE", "a title", "#A Title", "##  a tItLe  "] {
        page.text = format!("{first}\nBody line.");
        assert_eq!(page_text_v2(&page), "A title\nWhat it is\nBody line.", "{first:?} is the title in another case");
    }
    page.text = "Another opening\nBody line.".into();
    assert_eq!(page_text_v2(&page), "A title\nWhat it is\nAnother opening\nBody line.");
    page.text = "A title\nWhat it is, and more.".into();
    assert_eq!(page_text_v2(&page), "A title\nWhat it is, and more.", "a description the text opens with is not repeated");
    page.text = format!("A title\n{}", "y".repeat(20_000));
    assert_eq!(page_text_v2(&page).chars().count(), 10_000, "cut to what the model reads");
    assert!(texts("k", &Record { extraction: Some("v9".into()), ..page }).is_empty(), "no recipe reads an unknown extraction");
}

#[test]
fn v2_passages_start_at_markdown_headings_with_v1_sizes_and_up_to_32() {
    let text = ["x".repeat(700), "## Pricing".into(), "p".repeat(300), "q".repeat(400), "### FAQ".into(), "Short answer.".into()].join("\n");
    let not_headings = ["#hashtag line", "####### seven"].join("\n");
    let found = passages_v2(&format!("{text}\n{not_headings}"));
    let shape: Vec<(Option<&str>, usize)> = found.iter().map(|p| (p.heading.as_deref(), p.body.chars().count())).collect();
    assert_eq!(
        shape,
        [(None, 700), (Some("Pricing"), 701), (Some("FAQ"), 13 + 1 + not_headings.chars().count())],
        "a heading line starts a section and names it"
    );
    assert!(found[2].body.contains("#hashtag line"), "a hashtag is not a heading");
    assert!(passages(&text, &[]).iter().all(|p| p.heading.is_none()), "v1 passages split only at the headings list");

    let sections: Vec<String> = (0..40).map(|i| format!("## Part {i}\n{}", "w".repeat(700))).collect();
    let capped = passages_v2(&sections.join("\n"));
    assert_eq!(capped.len(), 32, "no more than 32");
    assert_eq!((capped[31].heading.as_deref(), capped[31].body.len()), (Some("Part 31"), 700));
    let plain_sections: Vec<String> = (0..40).map(|i| format!("Part {i}\n{}", "w".repeat(700))).collect();
    let headings: Vec<Option<String>> = (0..40).map(|i| Some(format!("Part {i}"))).collect();
    assert_eq!(passages(&plain_sections.join("\n"), &headings).len(), 16, "v1 keeps its 16");
    let sentence = format!("{}. ", "x".repeat(299));
    let plain = [sentence.repeat(12), "a".repeat(400), "b".repeat(100)].join("\n");
    assert_eq!(passages_v2(&plain), passages(&plain, &[]), "prose without headings splits as v1 does, at most 1,500 characters");
}

#[test]
fn a_v2_table_row_is_never_split_and_a_passage_continuing_a_table_repeats_its_header() {
    let header = "| Name | Value |\n|---|---|";
    let rows: Vec<String> = (0..38).map(|i| format!("| row {i:02} | {} |", "v".repeat(90))).collect();
    let found = passages_v2(&format!("Intro line.\n{header}\n{}", rows.join("\n")));
    let bodies: Vec<&str> = found.iter().map(|p| p.body.as_str()).collect();
    assert_eq!(bodies.len(), 7);
    assert!(bodies[0].starts_with(&format!("Intro line.\n{header}\n| row 00 ")));
    assert!(bodies[1..].iter().all(|b| b.starts_with(&format!("{header}\n| row "))), "each continuation opens with the header row and its separator");
    assert!(bodies.iter().all(|b| b.matches("| Name | Value |").count() == 1), "one header a passage");
    let placed: Vec<&str> = bodies.iter().flat_map(|b| b.lines()).filter(|l| l.starts_with("| row ")).collect();
    assert_eq!(placed, rows.iter().map(String::as_str).collect::<Vec<_>>(), "every row once, whole and in order");
    assert_eq!(bodies[6], format!("{header}\n{}\n{}", rows[36], rows[37]), "a short tail of a table stands alone under its header");

    let unruled: Vec<String> = (0..30).map(|i| format!("| {} | winner {i} |", 1990 + i)).chain(["after".repeat(150)]).collect();
    let found = passages_v2(&format!("| Year | Winner |\n{}", unruled.join("\n")));
    assert!(found.len() > 1 && found[0].body.starts_with("| Year | Winner |\n| 1990 "));
    assert!(found[1..].iter().all(|p| !p.body.contains("| Year | Winner |")), "without a separator the first row is data, never repeated");

    let wide = format!("| {} |", "cell. ".repeat(400));
    let found = passages_v2(&format!("| A |\n|---|\n{wide}\n| short |"));
    assert!(found.iter().any(|p| p.body.lines().any(|l| l == wide)), "a row longer than a passage stays whole");
}

#[test]
fn a_v2_list_item_keeps_its_sub_items_and_a_long_one_is_cut_between_lines_under_its_first_line() {
    let sub = ["  - sub a", "    - deeper", "  - sub b"].join("\n");
    let first = format!("- Item one {}", "word ".repeat(120).trim_end());
    let long = format!("12. {}", "A sentence here. ".repeat(110).trim_end());
    let text = ["Lead-in line:".to_string(), first.clone(), sub.clone(), "- Item two".into(), long.clone(), "Closing prose.".into()].join("\n");
    let found = passages_v2(&text);
    let item = format!("{first}\n{sub}");
    assert!(found.iter().any(|p| p.body.contains(&item)), "an item and its indented sub-items stay together: {found:#?}");
    assert!(found.iter().any(|p| p.body.lines().any(|l| l == long)), "a single line longer than a passage stays whole");
    assert!(passages(&text, &[]).iter().all(|p| !p.body.lines().any(|l| l == long)), "v1 splits the same line at its sentences");

    let entries: Vec<String> = (0..500).map(|i| format!("{}- entry {i:03} {}", if i % 5 == 0 { "  " } else { "    " }, "z".repeat(30))).collect();
    let nested = format!("- Index of entries\n{}", entries.join("\n"));
    let found = passages_v2(&nested);
    assert!(found.len() > 10);
    assert!(found.iter().all(|p| p.body.chars().count() <= 1_500), "{:?}", found.iter().map(|p| p.body.chars().count()).max());
    assert!(found.iter().all(|p| p.body.starts_with("- Index of entries\n")), "each continuation opens with the parent's first line");
    let placed: Vec<&str> = found.iter().flat_map(|p| p.body.lines().skip(1)).collect();
    assert_eq!(placed, entries.iter().map(String::as_str).collect::<Vec<_>>(), "every sub-item line once, whole and in order");
}

#[test]
fn a_v2_code_block_holds_no_headings_and_a_long_one_is_cut_between_lines_in_its_own_fences() {
    let code: Vec<String> = (0..30).map(|i| format!("# comment {i}\nvalue_{i} = compute({})", "x".repeat(40))).collect();
    let block = format!("```python\n{}\n```", code.join("\n"));
    let found = passages_v2(&format!("Install it first.\n{block}\n## Next\nAfter the code."));
    assert!(block.chars().count() > 1_500);
    let pieces: Vec<&Passage> = found.iter().filter(|p| p.body.contains("value_")).collect();
    assert!(pieces.len() > 1 && pieces.iter().all(|p| p.body.chars().count() <= 1_500));
    for piece in &pieces {
        let at = piece.body.find("```python\n").expect("each piece opens the fence with its info string");
        assert!(piece.body.ends_with("\n```") && piece.body[at..].matches("```").count() == 2, "{:?}", piece.body);
    }
    let lines: Vec<&str> = pieces.iter().flat_map(|p| p.body.lines()).filter(|l| l.starts_with("# comment") || l.starts_with("value_")).collect();
    assert_eq!(lines, code.iter().flat_map(|c| c.lines()).collect::<Vec<_>>(), "every code line once, whole and in order");
    assert_eq!(found.iter().filter_map(|p| p.heading.as_deref()).collect::<Vec<_>>(), ["Next"], "a # line inside the block is code");

    let short = format!("```\n# a comment\n{}\n```", "s".repeat(1_000));
    assert!(passages_v2(&format!("Run:\n{short}")).iter().any(|p| p.body.contains(&short)), "a block that fits stays whole");
    let wide = "w".repeat(2_000);
    assert!(passages_v2(&format!("```\n{wide}\n```")).iter().any(|p| p.body == format!("```\n{wide}\n```")), "a line longer than a passage stays whole");
    let unclosed = passages_v2(&format!("```\n{}\n## Real heading\nbody line", "y".repeat(700)));
    assert_eq!(unclosed.last().unwrap().heading.as_deref(), Some("Real heading"), "a fence never closed is a plain line");
}

#[test]
fn every_long_v2_page_gets_passages_led_by_the_title_and_its_heading() {
    let long = format!("A title\n{}\n## How it works\n{}", article(4), article(4));
    let docs = texts("k", &v2(record("https://news.example/docs", &long, "docs", "en")));
    let led: Vec<String> = docs.iter().skip(1).map(|t| t.text.lines().take(2).collect::<Vec<_>>().join("\n")).collect();
    assert_eq!(led.len(), 4);
    assert!(led[0].starts_with("A title\nParagraph 0: "), "the title line is not repeated in the first passage: {:?}", led[0]);
    assert!(led[3].starts_with("A title › How it works\nParagraph 2: "), "{:?}", led[3]);
    assert!(docs.iter().skip(1).all(|t| t.kind == PASSAGE && t.text.chars().count() <= 1_500 + 40));
    for page_type in ["article", "landing", "home", "product", "listing", "other"] {
        assert_eq!(texts("k", &v2(record("https://news.example/l", &long, page_type, "en"))).len(), 5, "{page_type}");
    }
    let untyped = Record { page_type: None, ..v2(record("https://news.example/u", &long, "other", "en")) };
    assert_eq!(texts("k", &untyped).len(), 5, "a page without a type too");
    assert_eq!(texts("k", &record("https://news.example/d", &long, "docs", "en")).len(), 1, "v1 keeps passages for articles only");
    let short = format!("A title\n{}", article(5));
    assert_eq!(texts("k", &v2(record("https://news.example/s", &short, "article", "en"))).len(), 1, "2,000 characters or fewer is one text");
    let sections: Vec<String> = (0..40).map(|i| format!("## Part {i}\n{}", "w".repeat(700))).collect();
    assert_eq!(texts("k", &v2(record("https://news.example/c", &sections.join("\n"), "landing", "en"))).len(), 33, "the page text and 32 passages");
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("publisher-planner-{name}-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bucket(addr: std::net::SocketAddr, name: &str) -> Bucket {
    let credentials = Credentials { access_key: "test".into(), secret_key: "test".into(), region: "auto".into() };
    Bucket::new(r2::client().unwrap(), &format!("http://{addr}"), name, "", credentials, Arc::new(AtomicU64::new(0))).unwrap()
}

/// Writes one change file and numbers it, as the publisher does.
async fn log(pages: &Bucket, seq: u64, records: Vec<Record>) {
    let file = changes::encode(&records.into_iter().map(change).collect::<Vec<_>>()).unwrap();
    let key = format!("changes/dt=2026-10-14/{seq}.parquet");
    pages.put(&key, Bytes::from(file), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(seq), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    pages.put(&CHANGES.latest_key(), Bytes::from(json!({"seq": seq}).to_string()), JSON, None).await.unwrap();
}

/// Writes and indexes change file `seq` after later ones, leaving the newest number where it is.
async fn number_late(pages: &Bucket, seq: u64, records: Vec<Record>) {
    let file = changes::encode(&records.into_iter().map(change).collect::<Vec<_>>()).unwrap();
    let key = format!("changes/dt=2026-10-14/{seq}.parquet");
    pages.put(&key, Bytes::from(file), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(seq), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
}

fn settings(dir: PathBuf, pages_per_hour: f64, start_after: i64) -> Settings {
    Settings {
        dir,
        pages_per_hour,
        pages_per_input: 3,
        chars_per_input: 3_000_000,
        flush_after: Duration::from_secs(3600),
        max_pending: 10,
        fresh: Some(Duration::from_secs(48 * 3600)),
        fresh_per_hour: 3600.0,
        pending_max_age: Duration::from_secs(3600),
        min_chars: 300,
        langs: vec!["en".into()],
        excluded: HashSet::from(["spam.example".to_string()]),
        list_file: None,
        set_file: None,
        start_after: Some(start_after),
        reads: 4,
        page_first: false,
        backfill: false,
        backfill_shards: 0..1024,
        current_prefix: "current/".into(),
        poll: Duration::from_millis(10),
        hole_wait: Duration::from_secs(3600),
    }
}

static REDIS_TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The test database, emptied, held by one test at a time.
async fn redis_db() -> Option<(tokio::sync::MutexGuard<'static, ()>, ConnectionManager)> {
    let turn = REDIS_TURN.lock().await;
    let client = redis::Client::open(REDIS).ok()?;
    let mut redis = tokio::time::timeout(Duration::from_secs(2), client.get_connection_manager()).await.ok()?.ok()?;
    let () = redis::cmd("FLUSHDB").query_async(&mut redis).await.ok()?;
    Some((turn, redis))
}

fn input_recipes(file: Bytes) -> HashSet<String> {
    let mut recipes = HashSet::new();
    for batch in ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap().map(Result::unwrap) {
        recipes.extend(batch.column_by_name("recipe").unwrap().as_string::<i32>().iter().map(|r| r.unwrap().to_string()));
    }
    recipes
}

fn input_rows(file: Bytes) -> Vec<(String, String, i32)> {
    let mut rows = Vec::new();
    for batch in ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap().map(Result::unwrap) {
        let (ids, kinds) = (batch.column_by_name("text_id").unwrap().as_string::<i32>(), batch.column_by_name("kind").unwrap().as_string::<i32>());
        let index = batch.column_by_name("index").unwrap().as_primitive::<Int32Type>();
        rows.extend((0..batch.num_rows()).map(|i| (ids.value(i).to_string(), kinds.value(i).to_string(), index.value(i))));
    }
    rows
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn published_pages_become_inputs_once_per_version_within_the_budget_and_the_room() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("flow");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize, text: &str, kind: &str, lang: &str| v2(record(&format!("https://news.example/story/{n}"), text, kind, lang));
    let mut spam = page(9, &article(2), "article", "en");
    spam.domain = "www.spam.example".into();
    log(&pages, 1, vec![page(0, &article(2), "article", "en")]).await;
    log(
        &pages,
        2,
        vec![
            page(1, &article(40), "article", "en"),
            page(2, &article(2), "landing", "en"),
            page(3, "too short", "article", "en"),
            page(4, &article(2), "article", "de"),
            spam,
        ],
    )
    .await;

    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.read_through(), 2, "starts after the change file it was told to");
    assert_eq!((planner.counts.planned, planner.counts.refused, planner.counts.inputs), (2, 3, 0), "short, German and excluded pages are left out");
    planner.flush().await.unwrap();
    assert_eq!(planner.counts.inputs, 1);
    let waiting: i64 = redis.llen(EMBED_INPUTS).await.unwrap();
    assert_eq!(waiting, 0, "no room: the input waits");

    planner.room = Room::Fixed(5);
    assert_eq!(planner.hand_over().await.unwrap(), 1);
    let entry: Value = serde_json::from_str(&redis.lpop::<_, String>(EMBED_INPUTS, None).await.unwrap()).unwrap();
    let file = temp.get(entry["input_key"].as_str().unwrap()).await.unwrap();
    assert_eq!(entry["input_sha256"], format!("{:x}", Sha256::digest(&file)));
    assert_eq!(input_recipes(file.clone()), HashSet::from([RECIPE_V2.to_string()]));
    let rows = input_rows(file);
    assert_eq!(entry["texts"], rows.len());
    assert_eq!(rows.iter().filter(|(_, kind, _)| kind == "page").count(), 2);
    assert_eq!(rows.iter().filter(|(_, kind, _)| kind == "passage").count(), 20, "the long article's 20 passages, under page-v2's cap of 32");
    let urls: HashSet<&str> = entry["pages"].as_array().unwrap().iter().map(|p| p["url"].as_str().unwrap()).collect();
    assert_eq!(urls, HashSet::from(["https://news.example/story/1", "https://news.example/story/2"]));

    log(&pages, 3, vec![page(1, &article(40), "article", "en"), page(2, &format!("{} changed", article(2)), "landing", "en")]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.unchanged, planner.counts.planned), (1, 3), "the same text again is not planned; a changed one is");

    drop(planner);
    let mut restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    assert_eq!(restarted.read_through(), 2, "the buffered page's change file is read again after a restart");
    restarted.pass().await.unwrap();
    assert_eq!((restarted.counts.planned, restarted.counts.unchanged), (1, 1));

    let mut budget = Planner::new(settings(dir.join("budget"), 4.0, 1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    budget.pass().await.unwrap();
    assert_eq!((budget.counts.planned, budget.counts.over_budget), (1, 2), "4 pages an hour allow one at once; the rest is left for later");

    let mut small = settings(dir.join("small"), 3600.0, 1);
    small.chars_per_input = 1;
    let mut split = Planner::new(small, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    split.pass().await.unwrap();
    assert_eq!((split.counts.planned, split.counts.inputs), (3, 3), "an input is written as soon as its texts reach the character cap");

    let mut one_page = settings(dir.join("order"), 3600.0, 1);
    one_page.pages_per_input = 1;
    let mut ordered = Planner::new(one_page, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    ordered.pass().await.unwrap();
    assert_eq!(ordered.counts.inputs, 3);
    let () = redis.del(EMBED_INPUTS).await.unwrap();
    ordered.room = Room::Fixed(1);
    assert_eq!(ordered.hand_over().await.unwrap(), 1);
    let first: Value = serde_json::from_str(&redis.lpop::<_, String>(EMBED_INPUTS, None).await.unwrap()).unwrap();
    assert_eq!(first["pages"][0]["url"], "https://news.example/story/1", "the input written first is handed over first");

    let mut stale = settings(dir.join("stale"), 3600.0, 1);
    stale.pending_max_age = Duration::ZERO;
    let mut waited = Planner::new(stale, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    waited.pass().await.unwrap();
    waited.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    waited.room = Room::Fixed(5);
    let before: i64 = redis.llen(EMBED_INPUTS).await.unwrap();
    assert_eq!(waited.hand_over().await.unwrap(), 0);
    assert_eq!((waited.counts.expired, redis.llen::<_, i64>(EMBED_INPUTS).await.unwrap()), (1, before), "an input that waited too long is dropped, not sent");
    log(&pages, 4, vec![page(1, &article(40), "article", "en")]).await;
    waited.pass().await.unwrap();
    assert_eq!((waited.counts.planned, waited.counts.unchanged), (3, 1), "a dropped input's pages are planned again when the log shows them next");

    waited.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&waited.state.pending().unwrap()[0].1).unwrap();
    let _: i64 = redis.rpush(EMBED_DROPPED, json!({"input_key": entry["input_key"], "pages": entry["pages"]}).to_string()).await.unwrap();
    log(&pages, 5, vec![page(1, &article(40), "article", "en")]).await;
    waited.pass().await.unwrap();
    assert_eq!((waited.counts.dropped, waited.counts.planned), (1, 4), "pages of an input the task API dropped are planned again");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_v2_pages_are_planned_and_a_v2_fetch_of_a_v1_page_is_planned() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("recipes");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let url = |n: usize| format!("https://news.example/story/{n}");
    let unknown = Record { extraction: Some("v9".into()), ..record(&url(3), &article(2), "article", "en") };
    log(&pages, 1, vec![record(&url(1), &article(2), "article", "en"), v2(record(&url(2), &format!("# {}", article(2)), "article", "en")), unknown]).await;
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.other_recipe), (1, 2), "the v2 page is planned; v1 and an unknown extraction are not");
    planner.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&planner.state.pending().unwrap()[0].1).unwrap();
    let file = temp.get(entry["input_key"].as_str().unwrap()).await.unwrap();
    assert_eq!(input_recipes(file), HashSet::from([RECIPE_V2.to_string()]));

    log(&pages, 2, vec![v2(record(&url(1), &format!("# {}", article(2)), "article", "en")), record(&url(2), &article(3), "article", "en")]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.other_recipe), (2, 3), "a v2 fetch of a v1 page is planned; a v1 fetch of a v2 page is not");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_file_numbered_late_holds_up_nothing_and_never_replaces_a_newer_version() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("late");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize, text: String, fetched_at: &str| {
        let mut record = v2(record(&format!("https://news.example/story/{n}"), &text, "article", "en"));
        record.fetched_at = fetched_at.into();
        record
    };
    log(&pages, 1, vec![page(1, article(2), "2026-10-14T10:00:00+00:00")]).await;
    log(&pages, 3, vec![page(2, article(3), "2026-10-14T12:00:00+00:00")]).await;
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.read_through(), 3, "the number not indexed yet does not stop the reading");
    assert_eq!((planner.counts.files, planner.counts.planned), (2, 2));

    let late = vec![page(3, article(4), "2026-10-14T10:30:00+00:00"), page(2, format!("{} before the edit", article(3)), "2026-10-14T11:00:00+00:00")];
    number_late(&pages, 2, late).await;
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.late, planner.counts.planned, planner.counts.older),
        (1, 3, 1),
        "the late file's new page is planned, its older version of story 2 is not"
    );

    log(&pages, 5, vec![page(4, article(5), "2026-10-14T13:00:00+00:00")]).await;
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!(planner.state.missing().unwrap().iter().map(|(seq, _)| *seq).collect::<Vec<_>>(), [4], "kept across restarts");
    drop(planner);

    let mut impatient = settings(dir.join("state"), 3600.0, 0);
    impatient.hole_wait = Duration::ZERO;
    let mut restarted = Planner::new(impatient, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    restarted.pass().await.unwrap();
    assert!(restarted.state.missing().unwrap().is_empty(), "a number never indexed is given up after the wait");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_numbered_change_file_that_cannot_be_read_is_tried_until_it_can_never_given_up() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("unreadable");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &article(n + 1), "article", "en"));
    log(&pages, 1, vec![page(1)]).await;
    let key = "changes/dt=2026-10-14/2.parquet";
    pages.put(key, Bytes::from_static(b"not a parquet file"), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(2), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    log(&pages, 3, vec![page(3)]).await;
    let mut impatient = settings(dir.join("state"), 3600.0, 0);
    impatient.hole_wait = Duration::ZERO;
    let mut planner = Planner::new(impatient, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.read_through(), planner.counts.planned), (3, 2), "the unreadable file holds up nothing");
    assert_eq!(planner.state.missing().unwrap().iter().map(|(seq, _)| *seq).collect::<Vec<_>>(), [2], "and is kept past the wait");

    number_late(&pages, 2, vec![page(2)]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.late, planner.counts.planned), (1, 3), "read once it can be");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_page_the_publisher_sends_back_is_planned_again_alone() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("sent-back");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &article(2), "article", "en"));
    log(&pages, 1, vec![page(0), page(1)]).await;
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!((planner.counts.planned, planner.hand_over().await.unwrap()), (2, 1));

    let malformed = page(0);
    let one = json!({"page_key": page_key(&malformed.url).unwrap(), "content_sha1": malformed.content_sha1, "url": malformed.url});
    let _: i64 = redis.rpush(EMBED_DROPPED, json!({"input_key": "embed/inputs/t1.parquet", "pages": [one]}).to_string()).await.unwrap();
    log(&pages, 2, vec![page(0), page(1)]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.dropped, planner.counts.planned, planner.counts.unchanged), (1, 3, 1), "only the page sent back is planned again");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_listed_site_takes_a_third_of_its_crawl_cap_for_itself_and_the_hosts_under_it() {
    let list = json!([
        {"host": "www.docs.example", "type": "docs", "send_cap": 600_000},
        {"host": "news.example", "type": "news", "backlog": 3_000_000},
        {"host": "open.example", "type": "media"}
    ]);
    let listed = Listed::parse(list.to_string().as_bytes()).unwrap();
    assert_eq!(listed.len(), 3);
    assert_eq!(listed.cap("docs.example"), Some(200_000));
    assert_eq!(listed.cap("WWW.Docs.Example"), Some(200_000));
    assert_eq!(listed.cap("api.docs.example"), Some(200_000), "a host under a listed site takes its cap");
    assert_eq!(listed.cap("news.example"), Some(u64::MAX), "a news set rolls, so the index keeps every article");
    assert_eq!(listed.cap("open.example"), Some(u64::MAX), "no crawl cap, no index cap");
    assert_eq!((listed.cap("example"), listed.cap("other.example")), (None, None));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_listed_sites_are_planned_each_up_to_its_index_cap_and_a_page_given_back_frees_its_place() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("listed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let on = |url: String, text: String, domain: &str| {
        let mut page = v2(record(&url, &text, "article", "en"));
        page.domain = domain.into();
        page
    };
    let page = |n: usize| on(format!("https://docs.example/guide/{n}"), article(2), "docs.example");
    let story = |n: usize| on(format!("https://news.example/story/{n}"), article(2), "news.example");
    let list = dir.join("domains.json");
    let news = json!({"host": "news.example", "type": "news", "backlog": 6});
    std::fs::write(&list, json!([{"host": "www.docs.example", "type": "docs", "send_cap": 6}, news]).to_string()).unwrap();
    log(&pages, 1, vec![page(0), page(1), page(2), on("https://other.example/a".into(), article(2), "other.example"), story(0), story(1), story(2)]).await;

    let mut listed = settings(dir.join("state"), 3600.0, 0);
    listed.pages_per_input = 10;
    listed.list_file = Some(list.clone());
    let mut planner = Planner::new(listed, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.planned, planner.counts.over_cap, planner.counts.unlisted),
        (5, 1, 1),
        "a crawl cap of 6 keeps 2 pages, a news set keeps every article, and other.example is not listed"
    );
    planner.flush().await.unwrap();
    assert_eq!((planner.state.site_pages("docs.example").unwrap(), planner.state.site_pages("news.example").unwrap()), (2, 3));

    log(&pages, 2, vec![on("https://docs.example/guide/0".into(), format!("{} changed", article(2)), "docs.example")]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.over_cap), (6, 1), "a new version of a page the index holds is planned at the cap");
    planner.flush().await.unwrap();
    assert_eq!(planner.state.site_pages("docs.example").unwrap(), 2);

    let mut entries = planner.state.pending().unwrap().into_iter().map(|(_, entry)| serde_json::from_str::<Value>(&entry).unwrap()).collect::<Vec<_>>();
    entries.sort_by(|a, b| a["written_at"].as_f64().unwrap().total_cmp(&b["written_at"].as_f64().unwrap()));
    let first = &entries[0];
    let _: i64 = redis.rpush(EMBED_DROPPED, json!({"input_key": first["input_key"], "pages": first["pages"]}).to_string()).await.unwrap();
    log(&pages, 3, vec![page(2)]).await;
    planner.pass().await.unwrap();
    assert_eq!(planner.state.site_pages("docs.example").unwrap(), 1, "guide 1 is given back; guide 0 keeps its newer version");
    assert_eq!((planner.counts.planned, planner.counts.over_cap), (7, 1), "the place guide 1 left takes guide 2");
    planner.flush().await.unwrap();
    assert_eq!(planner.state.site_pages("docs.example").unwrap(), 2);

    std::fs::write(&list, json!([{"host": "docs.example", "type": "docs", "send_cap": 300}, news]).to_string()).unwrap();
    std::fs::File::options().write(true).open(&list).unwrap().set_modified(std::time::SystemTime::now() + Duration::from_secs(60)).unwrap();
    log(&pages, 4, vec![page(3)]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.over_cap), (8, 1), "a raised cap in a changed list is read without a restart");
    planner.flush().await.unwrap();

    let kept = planner.state.site_pages("docs.example").unwrap();
    assert_eq!(planner.state.sites_from_planned().unwrap().get("docs.example").copied(), Some(kept), "the kept count equals one built from the planned pages");
    drop(planner);
    let restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert_eq!(restarted.state.site_pages("docs.example").unwrap(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_set_takes_every_golden_case_as_the_other_readers_do() {
    let golden: Value = serde_json::from_str(include_str!("v1-set-golden.json")).unwrap();
    let set = Set::parse(golden.to_string().as_bytes()).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let (url, published) = (case["url"].as_str().unwrap(), case["published"].as_str().unwrap_or(""));
        assert_eq!(set.admit(url, published).is_ok(), case["in"].as_bool().unwrap(), "{url} published {published:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_a_set_only_pages_its_rules_take_are_planned_and_the_crawl_list_is_not_read() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("set");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |url: &str, published: &str| {
        let mut page = v2(record(url, &format!("{url} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    let set = dir.join("v1-set.json");
    std::fs::write(
        &set,
        json!({"sites": [
            {"host": "news.example", "rule": "news", "since": "2020-01-01"},
            {"host": "docs.example", "rule": "whole", "paths": ["/guide/"]},
            {"host": "ref.example", "rule": "whole"},
            {"host": "old.ref.example", "rule": "out"}
        ]})
        .to_string(),
    )
    .unwrap();
    let list = dir.join("domains.json");
    std::fs::write(&list, json!([{"host": "other.example", "type": "docs", "send_cap": 3}]).to_string()).unwrap();
    log(
        &pages,
        1,
        vec![
            page("https://news.example/story/a", "2024-02-01T08:00:00Z"),
            page("https://news.example/2021/03/04/b", ""),
            page("https://news.example/story/c", ""),
            page("https://news.example/2019/12/31/d", ""),
            page("https://docs.example/guide/start", ""),
            page("https://docs.example/blog/post", ""),
            page("https://www.api.ref.example/x", ""),
            page("https://old.ref.example/y", ""),
            page("https://other.example/z", ""),
        ],
    )
    .await;

    let mut with_set = settings(dir.join("state"), 3600.0, 0);
    with_set.pages_per_input = 10;
    (with_set.set_file, with_set.list_file) = (Some(set), Some(list));
    let mut planner = Planner::new(with_set, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    let counts = &planner.counts;
    assert_eq!((counts.planned, counts.undated, counts.before_since, counts.outside_set, counts.unlisted), (4, 1, 1, 3, 0));
    planner.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&planner.state.pending().unwrap()[0].1).unwrap();
    let urls: HashSet<&str> = entry["pages"].as_array().unwrap().iter().map(|p| p["url"].as_str().unwrap()).collect();
    assert_eq!(
        urls,
        HashSet::from([
            "https://news.example/story/a",
            "https://news.example/2021/03/04/b",
            "https://docs.example/guide/start",
            "https://www.api.ref.example/x"
        ])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn change_files_read_at_once_are_planned_in_log_order() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("order");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let version = |n: usize| {
        let mut page = v2(record("https://news.example/story/1", &format!("version {n}\n{}", article(2 + n % 3)), "article", "en"));
        page.fetched_at = format!("2026-10-14T10:{n:02}:00+00:00");
        page
    };
    for n in 1..=30 {
        log(&pages, n as u64, vec![version(n)]).await;
    }
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.read_through(), planner.counts.files, planner.counts.older), (30, 30, 0), "no version is seen after a newer one");
    planner.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&planner.state.pending().unwrap()[0].1).unwrap();
    assert_eq!(entry["pages"][0]["content_sha1"], version(30).content_sha1, "the last file's version is the one planned");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pages_dated_in_the_last_two_days_go_first_outside_the_budget_and_a_restart_reads_a_buffered_lane_again() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("fresh");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let page = |name: &str, published: &str| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    log(&pages, 1, vec![page("a", "2020-05-01"), page("b", "2021-05-01"), page("f1", &today), page("f2", &today)]).await;

    let mut one = settings(dir.join("state"), 4.0, 0);
    one.pages_per_input = 1;
    let mut planner = Planner::new(one, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(2)).await.unwrap();
    planner.pass().await.unwrap();
    let counts = &planner.counts;
    assert_eq!((counts.planned, counts.fresh, counts.over_budget, counts.inputs), (3, 2, 1, 3), "the budget takes one old page; fresh pages spend none");
    let handed: Vec<Value> = redis.lrange::<_, Vec<String>>(EMBED_INPUTS, 0, -1).await.unwrap().iter().map(|raw| serde_json::from_str(raw).unwrap()).collect();
    let urls: Vec<&str> = handed.iter().map(|entry| entry["pages"][0]["url"].as_str().unwrap()).collect();
    assert_eq!(urls, ["https://news.example/story/f1", "https://news.example/story/f2"], "fresh inputs go first though the old one was written before them");
    assert!(handed.iter().all(|entry| entry["lane"] == "fresh"));
    assert_eq!(planner.state.pending().unwrap().len(), 1);
    drop(planner);

    log(&pages, 2, vec![page("c", "2022-05-01"), page("f3", &today), page("d", "2022-06-01")]).await;
    let mut two = settings(dir.join("state"), 3600.0, 1);
    two.pages_per_input = 2;
    let mut planner = Planner::new(two, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.inputs, planner.counts.fresh), (1, 1), "c and d fill an input while f3 waits in the fresh lane");
    drop(planner);
    let mut restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert_eq!(restarted.read_through(), 1, "the cursor stayed before the file whose fresh page was never written");
    restarted.pass().await.unwrap();
    assert_eq!((restarted.counts.planned, restarted.counts.fresh, restarted.counts.unchanged), (1, 1, 2));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Writes shard `shard` of `current/` as `files`, the base first, each sorted by key.
async fn shard(current: &Bucket, shard: usize, files: Vec<Vec<PageRow>>) {
    let mut refs = Vec::new();
    for (at, mut rows) in files.into_iter().enumerate() {
        rows.sort_by(|a, b| a.key.cmp(&b.key));
        let body = current::encode(&rows).unwrap();
        let key = format!("current/shard={shard:04}/{}-{at:012}.parquet", if at == 0 { "base" } else { "delta" });
        refs.push(FileRef { key: key.clone(), rows: rows.len() as u64, bytes: body.len() as u64, through_seq: at as i64, level: 0 });
        current.put(&key, Bytes::from(body), PARQUET, None).await.unwrap();
    }
    let manifest = Manifest { shard, through_seq: 10, merged_through: 0, base: Some(refs.remove(0)), deltas: refs, retired: Vec::new() };
    current.put(&format!("current/shard={shard:04}/manifest.json"), Bytes::from(serde_json::to_vec(&manifest).unwrap()), JSON, None).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_plans_each_set_page_s_standing_v2_version_as_the_budget_allows_and_goes_on_where_it_stopped() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let row = |url: &str, text: &str, published: &str, fetched_at: &str, change_seq: i64| {
        let mut record = v2(record(url, &format!("{url} {text} {}", article(2)), "article", "en"));
        (record.published, record.fetched_at) = (published.into(), fetched_at.into());
        PageRow { key: page_key(url).unwrap(), removed: false, change_seq, version: record_version(&record), published_at: fetched_at.into(), record }
    };
    let at = |hour: u32| format!("2026-10-20T{hour:02}:00:00+00:00");
    let old = row("https://news.example/story/b", "first", "2024-01-02", &at(1), 1);
    let newer = row("https://news.example/story/b", "second", "2024-01-02", &at(2), 5);
    let mut v1 = row("https://news.example/story/d", "", "2024-01-04", &at(1), 1);
    v1.record.extraction = None;
    let mut german = row("https://news.example/story/e", "", "2024-01-05", &at(1), 1);
    german.record.lang = "de".into();
    let mut base = vec![
        row("https://news.example/story/a", "", "2024-01-01", &at(1), 1),
        old,
        row("https://news.example/story/c", "", "2024-01-03", &at(1), 1),
        v1,
        german,
        row("https://news.example/story/f", "", "", &at(1), 1),
        row("https://docs.example/guide/g", "", "", &at(1), 1),
    ];
    base.extend((0..2_500).map(|n| row(&format!("https://other.example/{n}"), "", "", &at(1), 1)));
    let delta = vec![
        newer.clone(),
        PageRow::removal(page_key("https://news.example/story/c").unwrap(), 6),
        row("https://news.example/2025/02/01/h", "", "", &at(3), 6),
    ];
    shard(&current, 7, vec![base, delta]).await;
    log(&pages, 1, vec![v2(record("https://other.example/log", &article(2), "article", "en"))]).await;
    let set = dir.join("v1-set.json");
    std::fs::write(
        &set,
        json!({"sites": [{"host": "news.example", "rule": "news", "since": "2020-01-01"}, {"host": "docs.example", "rule": "whole"}]}).to_string(),
    )
    .unwrap();
    let backfilling = |pages_per_hour: f64| {
        let mut backfilling = settings(dir.join("state"), pages_per_hour, 0);
        (backfilling.backfill, backfilling.backfill_shards, backfilling.set_file, backfilling.pages_per_input) = (true, 7..8, Some(set.clone()), 1);
        backfilling
    };

    let mut planner = Planner::new(backfilling(4.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.backfill, planner.counts.planned), (1, 1), "the budget's one page is taken, and the reading waits for more");
    drop(planner);

    let mut restarted = Planner::new(backfilling(3600.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    restarted.current = Some(current.clone());
    for _ in 0..50 {
        restarted.pass().await.unwrap();
        if restarted.state.backfill().unwrap() == Some((8, None)) {
            break;
        }
    }
    assert_eq!(restarted.state.backfill().unwrap(), Some((8, None)), "the shard is read through");
    assert_eq!((restarted.counts.planned, restarted.counts.unchanged), (3, 0), "it goes on after the page planned before the restart");
    let planned: HashMap<String, String> = restarted
        .state
        .pending()
        .unwrap()
        .iter()
        .map(|(_, entry)| serde_json::from_str::<Value>(entry).unwrap())
        .map(|entry| (entry["pages"][0]["url"].as_str().unwrap().to_string(), entry["pages"][0]["content_sha1"].as_str().unwrap().to_string()))
        .collect();
    let urls: HashSet<&str> = planned.keys().map(String::as_str).collect();
    assert_eq!(
        urls,
        HashSet::from(["https://news.example/story/a", "https://news.example/story/b", "https://docs.example/guide/g", "https://news.example/2025/02/01/h"]),
        "removed, v1, German, undated and unlisted pages are left out"
    );
    assert_eq!(planned["https://news.example/story/b"], newer.record.content_sha1, "the later fetch in the delta stands");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_page_vectors_first_a_page_gets_its_page_text_now_and_its_passages_once_that_is_off() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("page-first");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let long = v2(record("https://news.example/story/long", &article(40), "article", "en"));
    let short = v2(record("https://news.example/story/short", &article(2), "article", "en"));
    let mut fresh = v2(record("https://news.example/story/fresh", &format!("fresh {}", article(40)), "article", "en"));
    fresh.published = today;
    log(&pages, 1, vec![long.clone(), short.clone(), fresh.clone()]).await;
    let rows_of = |entry: String| {
        let temp = temp.clone();
        async move {
            let entry: Value = serde_json::from_str(&entry).unwrap();
            input_rows(temp.get(entry["input_key"].as_str().unwrap()).await.unwrap())
        }
    };

    let mut first = settings(dir.join("state"), 3600.0, 0);
    (first.page_first, first.pages_per_input) = (true, 10);
    let mut planner = Planner::new(first, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.page_only, planner.counts.fresh), (3, 1, 1), "a short page has nothing more to wait for");
    let mut texts_by_page: HashMap<String, Vec<i32>> = HashMap::new();
    for (_, entry) in planner.state.pending().unwrap() {
        for (text_id, _, index) in rows_of(entry).await {
            texts_by_page.entry(text_id.split('#').next().unwrap().to_string()).or_default().push(index);
        }
    }
    let key = |record: &Record| page_key(&record.url).unwrap();
    assert_eq!(texts_by_page[&key(&long)], [0], "the long page's page text alone");
    assert!(texts_by_page[&key(&fresh)].len() > 1, "a fresh page gets every text");
    assert!(planner.state.planned_version(RECIPE_V2, &key(&long)).unwrap().unwrap().2);
    log(&pages, 2, vec![long.clone()]).await;
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.unchanged, 1, "its passages wait while page vectors go first");
    drop(planner);

    let mut then = settings(dir.join("state"), 3600.0, 0);
    then.pages_per_input = 10;
    let mut planner = Planner::new(then, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    let before: HashSet<String> = planner.state.pending().unwrap().into_iter().map(|(key, _)| key).collect();
    log(&pages, 3, vec![long.clone(), short.clone()]).await;
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!((planner.counts.parts, planner.counts.unchanged), (1, 1), "the long page is planned again for its passages; the short one is done");
    let (_, entry) = planner.state.pending().unwrap().into_iter().find(|(key, _)| !before.contains(key)).unwrap();
    let indexes: Vec<i32> = rows_of(entry).await.into_iter().map(|(_, _, index)| index).collect();
    assert_eq!(indexes, (1..=indexes.len() as i32).collect::<Vec<_>>(), "its passages alone, without the page text again");
    assert!(!planner.state.planned_version(RECIPE_V2, &key(&long)).unwrap().unwrap().2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_page_removed_and_back_with_the_same_text_is_planned_again() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("removed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let story = v2(record("https://news.example/story/1", &article(2), "article", "en"));
    log(&pages, 1, vec![story.clone()]).await;
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!(planner.state.site_pages("news.example").unwrap(), 1);

    let removal = Change { kind: Kind::Removed, body: Body::Removed { url: story.url.clone(), domain: story.domain.clone() }, ..change(story.clone()) };
    let key = format!("changes/dt=2026-10-14/{}.parquet", 2);
    pages.put(&key, Bytes::from(changes::encode(&[removal]).unwrap()), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(2), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    log(&pages, 3, vec![story.clone()]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.removed, planner.counts.planned, planner.counts.unchanged), (1, 2, 0));
    planner.flush().await.unwrap();
    assert_eq!(planner.state.site_pages("news.example").unwrap(), 1, "the site's count fell with the removal and rose with the return");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_page_its_site_re_dates_goes_with_the_rest_and_the_fresh_lane_keeps_to_its_own_budget() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("redated");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let page = |name: &str, text: &str, published: &str| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {text} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    log(&pages, 1, vec![page("old", "", "2020-05-01")]).await;
    log(&pages, 2, vec![page("old", "edited", &today), page("n1", "", &today), page("n2", "", &today)]).await;
    let mut one_fresh = settings(dir.join("state"), 3600.0, 0);
    one_fresh.fresh_per_hour = 4.0;
    let mut planner = Planner::new(one_fresh, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.fresh), (3, 1), "the re-dated page and the second new one go with the rest");
    let _ = std::fs::remove_dir_all(&dir);
}

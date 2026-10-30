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
    page_text, page_text_v2, passages, passages_v2, push_input, texts, Listed, Passage, Planner, Room, Settings, EMBED_DROPPED, EMBED_INPUTS, PAGE, PASSAGE,
    RECIPE_V2,
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
        max_waiting: 1_000,
        fresh: Some(Duration::from_secs(48 * 3600)),
        fresh_per_hour: 3600.0,
        budget_at_start: Duration::from_secs(3600),
        fresh_per_site: 2_000,
        fresh_flush_after: Duration::from_secs(3600),
        pending_max_age: Duration::from_secs(3600),
        min_chars: 300,
        langs: vec!["en".into()],
        excluded: HashSet::from(["spam.example".to_string()]),
        list_file: None,
        set_file: None,
        start_after: Some(start_after),
        reads: 4,
        writes: 8,
        page_first: false,
        backfill: false,
        backfill_shards: 0..1024,
        backfill_every: Duration::from_secs(3600),
        current_prefix: "current/".into(),
        poll: Duration::from_millis(10),
        hole_wait: Duration::from_secs(3600),
    }
}

static REDIS_TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The test database, emptied, held by one test at a time.
async fn redis_db() -> Option<(tokio::sync::MutexGuard<'static, ()>, ConnectionManager)> {
    let turn = REDIS_TURN.lock().await;
    let connected = async {
        let client = redis::Client::open(REDIS).ok()?;
        let mut redis = tokio::time::timeout(Duration::from_secs(2), client.get_connection_manager()).await.ok()?.ok()?;
        let () = redis::cmd("FLUSHDB").query_async(&mut redis).await.ok()?;
        Some(redis)
    };
    match connected.await {
        Some(redis) => Some((turn, redis)),
        None if std::env::var("PUBLISHER_TEST_REDIS").is_ok_and(|need| need == "1") => panic!("PUBLISHER_TEST_REDIS=1 and no Redis on {REDIS}"),
        None => None,
    }
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
    assert_eq!((waited.counts.planned, waited.counts.unchanged), (4, 2), "a dropped input's pages are planned again from their change files, once");

    waited.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&waited.state.pending().unwrap()[0].1).unwrap();
    let _: i64 = redis.rpush(EMBED_DROPPED, json!({"input_key": entry["input_key"], "pages": entry["pages"]}).to_string()).await.unwrap();
    log(&pages, 5, vec![page(1, &article(40), "article", "en")]).await;
    waited.pass().await.unwrap();
    assert_eq!((waited.counts.dropped, waited.counts.planned), (1, 6), "pages of an input the task API dropped are planned again from their change files");
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
    assert_eq!(
        (planner.counts.dropped, planner.counts.planned, planner.counts.unchanged),
        (1, 3, 2),
        "only the page sent back is planned again, once though both the log and its change file show it"
    );
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
    assert_eq!(
        (planner.counts.planned, planner.counts.over_cap),
        (10, 2),
        "the place guide 1 left takes guide 2; the stories given back are planned again from their change file"
    );
    planner.flush().await.unwrap();
    assert_eq!(planner.state.site_pages("docs.example").unwrap(), 2);

    std::fs::write(&list, json!([{"host": "docs.example", "type": "docs", "send_cap": 300}, news]).to_string()).unwrap();
    std::fs::File::options().write(true).open(&list).unwrap().set_modified(std::time::SystemTime::now() + Duration::from_secs(60)).unwrap();
    log(&pages, 4, vec![page(3)]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.over_cap), (11, 2), "a raised cap in a changed list is read without a restart");
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
    for raw in [include_str!("v1-set-golden.json"), include_str!("v1-set-golden-undated.json")] {
        let golden: Value = serde_json::from_str(raw).unwrap();
        let set = Set::parse(golden.to_string().as_bytes()).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let (url, published, page_type) = (case["url"].as_str().unwrap(), case["published"].as_str().unwrap_or(""), case["page_type"].as_str());
            assert_eq!(set.admit(url, published, page_type).is_ok(), case["in"].as_bool().unwrap(), "{url} published {published:?} as {page_type:?}");
        }
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
async fn change_files_fetched_ahead_are_planned_in_log_order_whichever_arrives_first() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("ahead-order");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let urls: Vec<String> = (1..=6).map(|n| format!("https://news.example/story/{n}")).collect();
    for (seq, url) in (1..).zip(&urls) {
        log(&pages, seq, vec![v2(record(url, &format!("{url} {}", article(2)), "article", "en"))]).await;
    }
    stub.state.slow("GET", "desearch-pages/changes/dt=2026-10-14/1.parquet", 1, Duration::from_millis(1500));
    stub.state.slow("GET", "desearch-pages/changes/dt=2026-10-14/2.parquet", 1, Duration::from_millis(800));
    let mut one_a_page = settings(dir.join("state"), 3600.0, 0);
    one_a_page.pages_per_input = 1;
    let mut planner = Planner::new(one_a_page, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.read_through(), planner.counts.files, planner.state.cursor().unwrap()), (6, 6, Some(6)));
    assert_eq!(saved_urls(&planner), urls, "files 3 to 6 arrive before 1 and 2 and are still cut after them");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn among_change_files_fetched_ahead_one_unreadable_and_one_not_numbered_yet_hold_up_nothing_and_are_read_late() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("ahead-gaps");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &article(n + 1), "article", "en"));
    log(&pages, 1, vec![page(1)]).await;
    let key = "changes/dt=2026-10-14/2.parquet";
    pages.put(key, Bytes::from_static(b"not a parquet file"), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(2), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    log(&pages, 4, vec![page(4)]).await;
    log(&pages, 5, vec![page(5)]).await;
    // The gaps and the files after them are fetched before the first file arrives.
    stub.state.slow("GET", "desearch-pages/changes/dt=2026-10-14/1.parquet", 1, Duration::from_millis(1500));
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.read_through(), planner.counts.files, planner.counts.planned), (5, 3, 3));
    assert_eq!(planner.state.missing().unwrap().iter().map(|(seq, _)| *seq).collect::<Vec<_>>(), [2, 3]);

    number_late(&pages, 2, vec![page(2)]).await;
    number_late(&pages, 3, vec![page(3)]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.late, planner.counts.planned), (2, 5), "both read once they can be");
    planner.flush().await.unwrap();
    assert!(planner.state.missing().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_planner_counts_its_time_waiting_on_change_files_input_writes_and_state_writes() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("spent");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    log(&pages, 1, vec![v2(record("https://news.example/story/1", &article(2), "article", "en"))]).await;
    stub.state.slow("GET", "desearch-pages/changes/dt=", 1, Duration::from_secs(1));
    stub.state.slow("PUT", "subnet-22/embed/inputs/", 1, Duration::from_secs(1));
    let mut one_a_page = settings(dir.join("state"), 3600.0, 0);
    one_a_page.pages_per_input = 1;
    let mut planner = Planner::new(one_a_page, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    let spent = planner.spent.clone();
    assert!(spent.log_reads >= Duration::from_secs(1) && spent.log >= spent.log_reads, "{spent:?}");
    assert!(spent.input_writes >= Duration::from_millis(800), "{spent:?}");
    assert!(planner.state.writing() > Duration::ZERO);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pages_dated_in_the_last_two_days_get_their_own_inputs_outside_the_budget_and_a_restart_reads_a_buffered_lane_again() {
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
    let mut first = vec![page("a", "2020-05-01"), page("b", "2021-05-01"), page("f1", &today), page("f2", &today)];
    first.extend((1..=6).map(|n| page(&format!("g{n}"), &today)));
    log(&pages, 1, first).await;

    let mut one = settings(dir.join("state"), 4.0, 0);
    one.pages_per_input = 1;
    let mut planner = Planner::new(one, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(2)).await.unwrap();
    planner.pass().await.unwrap();
    let counts = &planner.counts;
    assert_eq!((counts.planned, counts.fresh, counts.over_budget, counts.inputs), (9, 8, 1, 9), "the budget takes one old page; fresh pages spend none");
    let handed: Vec<Value> = redis.lrange::<_, Vec<String>>(EMBED_INPUTS, 0, -1).await.unwrap().iter().map(|raw| serde_json::from_str(raw).unwrap()).collect();
    let urls: Vec<&str> = handed.iter().map(|entry| entry["pages"][0]["url"].as_str().unwrap()).collect();
    assert_eq!(
        urls,
        ["https://news.example/story/a", "https://news.example/story/f1", "https://news.example/story/f2"],
        "a goes as soon as it is saved, when the ninth input is cut; the fresh ones saved at the pass end go as the room of 2 allows"
    );
    assert_eq!(handed.iter().map(|entry| entry["lane"].as_str()).collect::<Vec<_>>(), [None, Some("fresh"), Some("fresh")]);
    assert_eq!(planner.state.pending().unwrap().len(), 6);
    drop(planner);

    log(&pages, 2, vec![page("c", "2022-05-01"), page("f3", &today), page("d", "2022-06-01")]).await;
    let mut two = settings(dir.join("state"), 3600.0, 1);
    two.pages_per_input = 2;
    let mut planner = Planner::new(two, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.inputs, planner.counts.fresh), (1, 1), "c and d fill an input while f3 waits in the fresh lane");
    drop(planner);
    let mut serving = settings(dir.join("state"), 3600.0, 0);
    serving.budget_at_start = Duration::ZERO;
    let mut restarted = Planner::new(serving, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert_eq!(restarted.read_through(), 1, "the cursor stayed before the file whose fresh page was never written");
    restarted.pass().await.unwrap();
    assert_eq!(
        (restarted.counts.planned, restarted.counts.fresh, restarted.counts.unchanged),
        (1, 1, 2),
        "f3 again, in the fresh lane, though the planner starts with no other budget"
    );
    let b = page_key("https://news.example/story/b").unwrap();
    assert!(restarted.state.waiting(&b).unwrap().is_some(), "b still waits though the planner stopped with it buffered");
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

    let mut again = backfilling(3600.0);
    again.backfill_every = Duration::ZERO;
    let mut restarted = Planner::new(again, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    restarted.current = Some(current.clone());
    for _ in 0..50 {
        restarted.pass().await.unwrap();
        if restarted.counts.shards == 1 {
            break;
        }
    }
    assert_eq!(
        (restarted.counts.shards, restarted.state.backfill().unwrap()),
        (1, Some((7, None))),
        "the shard is read through, and the next sweep starts over"
    );
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
    let read = restarted.counts.backfill;
    for _ in 0..50 {
        restarted.pass().await.unwrap();
        if restarted.counts.shards == 2 {
            break;
        }
    }
    assert_eq!((restarted.counts.shards, restarted.counts.backfill), (2, read), "a second sweep sends nothing already planned at its text");
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
    assert!(planner.state.planned_version(RECIPE_V2, &key(&long)).unwrap().unwrap().page_only);
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
    assert!(!planner.state.planned_version(RECIPE_V2, &key(&long)).unwrap().unwrap().page_only);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pages_left_for_budget_or_forgotten_are_planned_again_from_their_change_file() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |name: &str| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    let key = |name: &str| page_key(&page(name).url).unwrap();
    log(&pages, 1, vec![page("a"), page("b")]).await;
    log(&pages, 2, vec![page("c")]).await;
    let per_page = |pages_per_hour: f64| {
        let mut per_page = settings(dir.join("state"), pages_per_hour, 0);
        per_page.pages_per_input = 1;
        per_page
    };
    let mut planner = Planner::new(per_page(4.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 2), "the budget's one page is planned; b and c wait");
    assert_eq!(planner.state.waiting(&key("b")).unwrap().unwrap().seq, Some(1));
    drop(planner);
    pages.delete("changes/dt=2026-10-14/2.parquet").await.unwrap();

    let mut planner = Planner::new(per_page(3600.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.planned), (1, 1), "b is read again from change file 1 though the log moved on");
    assert!(planner.state.waiting(&key("b")).unwrap().is_none());
    assert!(planner.state.waiting(&key("c")).unwrap().is_none(), "c's change file is gone, so c is left to the backfill");

    let (_, entry) = planner.state.pending().unwrap().into_iter().find(|(_, entry)| entry.contains("story/a")).unwrap();
    let entry: Value = serde_json::from_str(&entry).unwrap();
    let _: i64 = redis.rpush(EMBED_DROPPED, json!({"input_key": entry["input_key"], "pages": entry["pages"]}).to_string()).await.unwrap();
    planner.pass().await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.dropped, planner.counts.replanned, planner.counts.planned),
        (1, 2, 2),
        "a page given back is planned again without waiting for the log"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_does_not_plan_a_version_older_than_one_the_log_showed() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("newer");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let version = |text: &str, fetched_at: &str| {
        let mut record = v2(record("https://news.example/story/p", &format!("{text} {}", article(2)), "article", "en"));
        record.fetched_at = fetched_at.into();
        record
    };
    let (older, newer) = (version("older", "2026-10-20T01:00:00+00:00"), version("newer", "2026-10-20T05:00:00+00:00"));
    let row = PageRow {
        key: page_key(&older.url).unwrap(),
        removed: false,
        change_seq: 1,
        version: record_version(&older),
        published_at: String::new(),
        record: older,
    };
    shard(&current, 3, vec![vec![row]]).await;
    log(&pages, 1, vec![v2(record("https://news.example/story/q", &article(2), "article", "en")), newer.clone()]).await;
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.waiting, 1);
    drop(planner);

    let mut backfilling = settings(dir.join("state"), 3600.0, 0);
    (backfilling.backfill, backfilling.backfill_shards, backfilling.pages_per_input) = (true, 3..4, 1);
    let mut planner = Planner::new(backfilling, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..50 {
        planner.pass().await.unwrap();
        if planner.counts.shards == 1 {
            break;
        }
    }
    let planned = planner.state.planned_version(publisher::planner::RECIPE_V2, &page_key(&newer.url).unwrap()).unwrap().unwrap();
    assert_eq!(planned.content_sha1, newer.content_sha1, "the log's newer version is planned, current/'s older one is not");
    assert_eq!(planner.counts.planned, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fresh_input_is_written_within_its_own_short_wait() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("fresh-flush");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let mut fresh = v2(record("https://news.example/story/new", &article(2), "article", "en"));
    fresh.published = iso(publisher::service::now_us());
    log(&pages, 1, vec![fresh, v2(record("https://news.example/story/old", &article(3), "article", "en"))]).await;
    let mut quick = settings(dir.join("state"), 3600.0, 0);
    (quick.pages_per_input, quick.fresh_flush_after) = (10, Duration::ZERO);
    let mut planner = Planner::new(quick, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    let pending = planner.state.pending().unwrap();
    assert_eq!(pending.len(), 1, "the other lane keeps filling");
    assert!(pending[0].1.contains("\"lane\":\"fresh\""));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_input_pushed_again_after_a_crash_reaches_the_task_api_once() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let entry = json!({"input_key": "embed/inputs/a.parquet", "pages": []}).to_string();
    assert!(push_input(&mut redis, "embed/inputs/a.parquet", &entry).await.unwrap());
    assert!(!push_input(&mut redis, "embed/inputs/a.parquet", &entry).await.unwrap());
    assert_eq!(redis.llen::<_, i64>(EMBED_INPUTS).await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_both_lanes_busy_a_restart_reads_again_only_the_files_still_buffered() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("cursor");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let page = |name: String, published: &str| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    for n in 1..=12u64 {
        log(&pages, n, vec![page(format!("old-{n}-a"), "2021-01-01"), page(format!("old-{n}-b"), "2021-01-01"), page(format!("new-{n}"), &today)]).await;
    }
    let mut threes = settings(dir.join("state"), 3600.0, 0);
    threes.pages_per_input = 3;
    let mut planner = Planner::new(threes, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.read_through(), planner.counts.planned, planner.counts.inputs), (12, 36, 12));
    drop(planner);

    let mut restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(restarted.read_through() >= 10, "the cursor kept up though one lane always held pages: {}", restarted.read_through());
    restarted.pass().await.unwrap();
    assert_eq!(
        restarted.counts.planned + restarted.counts.unchanged,
        3 * restarted.counts.files,
        "each page of a file read again is planned again or unchanged"
    );
    assert!(restarted.counts.files <= 2, "only the files of pages never written are read again: {}", restarted.counts.files);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plans `a` and leaves `b` waiting, from change file 1, in a fresh state under `dir`.
async fn one_planned_one_waiting(dir: &std::path::Path, pages: &Bucket, temp: &Bucket, redis: &ConnectionManager, b: Record) {
    let a = v2(record("https://news.example/story/a", &format!("a {}", article(2)), "article", "en"));
    log(pages, 1, vec![a, b]).await;
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_waiting_page_whose_change_file_the_log_expired_index_and_all_stops_waiting() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("expired");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    one_planned_one_waiting(&dir, &pages, &temp, &redis, b.clone()).await;
    pages.delete("changes/dt=2026-10-14/1.parquet").await.unwrap();
    pages.delete(&CHANGES.seq_key(1)).await.unwrap();

    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert!(planner.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_none(), "left to the backfill");
    assert!(planner.state.replan_group(0).unwrap().is_none(), "nothing is read for it again");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_waiting_page_the_re_plan_buffered_still_waits_after_a_crash() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-crash");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    one_planned_one_waiting(&dir, &pages, &temp, &redis, b.clone()).await;
    let roomy = |pages_per_input: usize| {
        let mut roomy = settings(dir.join("state"), 3600.0, 0);
        roomy.pages_per_input = pages_per_input;
        roomy
    };

    let mut planner = Planner::new(roomy(10), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.inputs), (1, 0), "b is buffered, not written");
    drop(planner);
    let mut restarted = Planner::new(roomy(1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(restarted.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_some(), "the crash lost nothing");
    restarted.pass().await.unwrap();
    assert_eq!((restarted.counts.replanned, restarted.counts.inputs), (1, 1));
    assert!(restarted.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_none(), "written, so no longer waiting");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_waiting_page_the_re_plan_buffered_still_waits_after_a_crash_when_a_later_sweep_met_it_in_the_buffer() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-crash-twice");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    one_planned_one_waiting(&dir, &pages, &temp, &redis, b.clone()).await;
    let roomy = |pages_per_input: usize| {
        let mut roomy = settings(dir.join("state"), 3600.0, 0);
        roomy.pages_per_input = pages_per_input;
        roomy
    };

    let mut planner = Planner::new(roomy(10), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.unchanged, planner.counts.inputs), (2, 1, 0), "the second sweep met b in its buffer");
    drop(planner);
    let mut restarted = Planner::new(roomy(1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(restarted.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_some(), "the crash lost nothing");
    restarted.pass().await.unwrap();
    assert_eq!((restarted.counts.replanned, restarted.counts.inputs), (1, 1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_re_plan_fetches_only_the_change_files_its_budget_will_plan() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-budget-reads");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    for seq in 1..=8u64 {
        log(&pages, seq, vec![v2(record(&format!("https://news.example/story/{seq}"), &format!("{seq} {}", article(2)), "article", "en"))]).await;
    }
    let tight = |pages_per_hour: f64| {
        let mut tight = settings(dir.join("state"), pages_per_hour, 0);
        tight.pages_per_input = 1;
        tight
    };
    let mut planner = Planner::new(tight(4.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 7));
    drop(planner);

    // A budget of two pages, with reads 4: two change files are worth fetching, not four.
    let mut planner = Planner::new(tight(60.0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    let before = stub.state.count("GET");
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.replanned, 2);
    assert_eq!(stub.state.count("GET") - before, 1 + 2 * 2, "the newest number, then an index and a file for each of the two");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_plans_current_s_copy_of_a_waiting_version_whose_change_file_cannot_be_read() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("same-version");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    one_planned_one_waiting(&dir, &pages, &temp, &redis, b.clone()).await;
    pages.put("changes/dt=2026-10-14/1.parquet", Bytes::from_static(b"not a parquet file"), PARQUET, None).await.unwrap();
    let row =
        PageRow { key: page_key(&b.url).unwrap(), removed: false, change_seq: 1, version: record_version(&b), published_at: String::new(), record: b.clone() };
    shard(&current, 5, vec![vec![row]]).await;

    let mut backfilling = settings(dir.join("state"), 3600.0, 0);
    (backfilling.backfill, backfilling.backfill_shards, backfilling.pages_per_input) = (true, 5..6, 1);
    let mut planner = Planner::new(backfilling, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..50 {
        planner.pass().await.unwrap();
        if planner.counts.shards == 1 {
            break;
        }
    }
    assert_eq!((planner.counts.backfill, planner.counts.planned), (1, 1), "the same version, from current/");
    assert!(planner.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_newer_row_lets_go_of_an_older_waiting_version_and_an_older_row_never_passes_one() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("newer-waiting");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let version = |name: &str, text: &str, hour: u32| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {text} {}", article(2)), "article", "en"));
        page.fetched_at = format!("2026-10-20T{hour:02}:00:00+00:00");
        page
    };
    let key = |name: &str| page_key(&format!("https://news.example/story/{name}")).unwrap();
    log(&pages, 1, vec![version("p", "a", 1)]).await;
    log(&pages, 2, vec![version("p", "b", 2), version("q", "y", 5)]).await;
    log(&pages, 3, vec![version("p", "a", 3)]).await;
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting, planner.counts.unchanged), (1, 2, 1));
    assert!(planner.state.waiting(&key("p")).unwrap().is_none(), "p's text went back to the planned one, so b no longer waits");
    assert!(planner.state.waiting(&key("q")).unwrap().is_some());
    drop(planner);

    log(&pages, 5, vec![version("r", "", 1)]).await;
    number_late(&pages, 4, vec![version("q", "x", 4)]).await;
    let mut roomy = settings(dir.join("state"), 3600.0, 0);
    roomy.pages_per_input = 1;
    let mut planner = Planner::new(roomy, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.older, 1, "q's late older row is passed over");
    let planned = planner.state.planned_version(publisher::planner::RECIPE_V2, &key("q")).unwrap().unwrap();
    assert_eq!(planned.content_sha1, version("q", "y", 5).content_sha1, "the waiting newer version is the one planned");
    assert_eq!(planner.state.planned_version(publisher::planner::RECIPE_V2, &key("p")).unwrap().unwrap().content_sha1, version("p", "a", 1).content_sha1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_newer_row_refused_by_its_text_or_its_site_lets_go_of_an_older_waiting_version() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("refused-newer");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let version = |url: &str, text: &str, hour: u32| {
        let mut page = v2(record(url, text, "article", "en"));
        (page.domain, page.fetched_at) = (url.split('/').nth(2).unwrap().into(), format!("2026-10-20T{hour:02}:00:00+00:00"));
        page
    };
    let (p, q) = ("https://news.example/story/p", "https://q.example/story/q");
    log(&pages, 1, vec![version("https://news.example/story/a", &article(2), 1), version(p, &article(3), 1), version(q, &article(4), 1)]).await;
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 2));
    drop(planner);

    log(&pages, 2, vec![version(p, "too short now", 2), version(q, &article(5), 2)]).await;
    let mut excluding = settings(dir.join("state"), 3600.0, 0);
    excluding.excluded.insert("q.example".into());
    let mut planner = Planner::new(excluding, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.refused, planner.counts.replanned, planner.counts.planned),
        (2, 0, 0),
        "p refused by its length and q by its excluded site, and neither older version is planned again"
    );
    for url in [p, q] {
        assert!(planner.state.waiting(&page_key(url).unwrap()).unwrap().is_none(), "{url} no longer waits");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Writes change file `seq` holding one removal of `page`, numbered now.
async fn log_removal(pages: &Bucket, seq: u64, page: &Record) {
    let removal = Change { kind: Kind::Removed, body: Body::Removed { url: page.url.clone(), domain: page.domain.clone() }, ..change(page.clone()) };
    let key = format!("changes/dt=2026-10-14/{seq}.parquet");
    pages.put(&key, Bytes::from(changes::encode(&[removal]).unwrap()), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(seq), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    pages.put(&CHANGES.latest_key(), Bytes::from(json!({"seq": seq}).to_string()), JSON, None).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_removal_takes_a_buffered_page_out_and_a_late_row_from_before_it_does_not_bring_it_back() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("removed-buffered");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let story = |text: &str| v2(record("https://news.example/story/p", &format!("{text} {}", article(2)), "article", "en"));
    let other = v2(record("https://news.example/story/o", &format!("o {}", article(2)), "article", "en"));
    log(&pages, 1, vec![story("first"), other.clone()]).await;
    log_removal(&pages, 3, &story("first")).await;
    let mut ten = settings(dir.join("state"), 3600.0, 0);
    ten.pages_per_input = 10;
    let mut planner = Planner::new(ten, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    number_late(&pages, 2, vec![story("late")]).await;
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    let entry: Value = serde_json::from_str(&planner.state.pending().unwrap()[0].1).unwrap();
    let urls: Vec<&str> = entry["pages"].as_array().unwrap().iter().map(|p| p["url"].as_str().unwrap()).collect();
    assert_eq!(urls, [other.url.as_str()], "the removed page left the buffer, and its late row did not come back");
    assert_eq!((planner.counts.removed, planner.counts.older), (1, 1));
    let key = page_key(&story("first").url).unwrap();
    assert!(planner.state.planned(publisher::planner::RECIPE_V2, &key).unwrap().is_none());

    log(&pages, 4, vec![story("back")]).await;
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.planned, 3, "a row after the removal brings the page back");
    assert!(planner.state.removed(&key).unwrap().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Writes change file `seq` holding one take-back of `task_id`, numbered now.
async fn log_take_back(pages: &Bucket, seq: u64, task_id: &str) {
    let take_back = Change {
        key: String::new(),
        kind: Kind::TaskWithdrawn,
        previous_content_sha1: String::new(),
        published_at: "2026-10-14T10:00:00+00:00".into(),
        body: Body::TaskWithdrawn { task_id: task_id.into() },
    };
    let key = format!("changes/dt=2026-10-14/{seq}.parquet");
    pages.put(&key, Bytes::from(changes::encode(&[take_back]).unwrap()), PARQUET, None).await.unwrap();
    pages.put(&CHANGES.seq_key(seq), Bytes::from(json!({"key": key, "rows": 1}).to_string()), JSON, None).await.unwrap();
    pages.put(&CHANGES.latest_key(), Bytes::from(json!({"seq": seq}).to_string()), JSON, None).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_page_from_a_task_taken_back_is_embedded_again_when_crawled_again_with_the_same_text() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("take-back");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let by = |name: &str, task_id: &str, hour: u32| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
        (page.task_id, page.fetched_at) = (task_id.into(), format!("2026-10-20T{hour:02}:00:00+00:00"));
        page
    };
    log(&pages, 1, vec![by("p", "t1", 1)]).await;
    let mut one = settings(dir.join("state"), 3600.0, 0);
    one.pages_per_input = 1;
    let mut planner = Planner::new(one, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.planned, 1);
    drop(planner);

    log_take_back(&pages, 2, "t1").await;
    log(&pages, 3, vec![by("p", "t1", 2), by("p", "t2", 3), by("q", "t3", 3)]).await;
    log_take_back(&pages, 4, "t3").await;
    let mut ten = settings(dir.join("state"), 3600.0, 0);
    ten.pages_per_input = 10;
    let mut planner = Planner::new(ten, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!((planner.counts.withdrawn, planner.counts.unchanged), (1, 0), "t1's later row is dropped; t2's same text is not unchanged");
    let entries: Vec<Value> = planner.state.pending().unwrap().iter().map(|(_, entry)| serde_json::from_str(entry).unwrap()).collect();
    let urls: Vec<&str> = entries.iter().flat_map(|entry| entry["pages"].as_array().unwrap().iter().map(|p| p["url"].as_str().unwrap())).collect();
    assert_eq!(urls.iter().filter(|url| url.ends_with("/q")).count(), 0, "q's buffered page left with its task");
    assert_eq!(urls.iter().filter(|url| url.ends_with("/p")).count(), 2, "p again, crawled by t2");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_picks_the_version_standing_after_a_take_back_as_the_compactor_does() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-take-back");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let row = |text: &str, task_id: &str, hour: u32, change_seq: i64| {
        let mut record = v2(record("https://news.example/story/p", &format!("{text} {}", article(2)), "article", "en"));
        (record.task_id, record.fetched_at) = (task_id.into(), format!("2026-10-20T{hour:02}:00:00+00:00"));
        PageRow { key: page_key(&record.url).unwrap(), removed: false, change_seq, version: record_version(&record), published_at: String::new(), record }
    };
    let (taken, honest) = (row("forged", "t9", 5, 1), row("honest", "t8", 1, 3));
    shard(&current, 6, vec![vec![taken], vec![honest.clone()]]).await;
    current.put("current/withdrawn.json", Bytes::from(json!({"tasks": {"t9": {"seq": 2, "last": 2}}}).to_string()), JSON, None).await.unwrap();
    log(&pages, 1, vec![]).await;

    let mut backfilling = settings(dir.join("state"), 3600.0, 0);
    (backfilling.backfill, backfilling.backfill_shards, backfilling.pages_per_input) = (true, 6..7, 1);
    let mut planner = Planner::new(backfilling, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..50 {
        planner.pass().await.unwrap();
        if planner.counts.shards == 1 {
            break;
        }
    }
    let planned = planner.state.planned_version(publisher::planner::RECIPE_V2, &honest.key).unwrap().unwrap();
    assert_eq!(planned.content_sha1, honest.record.content_sha1, "the taken-back newer fetch does not stand");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_site_fills_at_most_its_share_of_the_fresh_lane_and_an_old_url_date_is_not_fresh() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("fresh-site");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let on = |site: &str, path: &str| {
        let mut page = v2(record(&format!("https://{site}/{path}"), &format!("{site} {path} {}", article(2)), "article", "en"));
        (page.domain, page.published) = (site.into(), today.clone());
        page
    };
    log(&pages, 1, vec![on("a.example", "1"), on("a.example", "2"), on("a.example", "3"), on("b.example", "1"), on("b.example", "2019/03/04/x")]).await;
    let mut two = settings(dir.join("state"), 3600.0, 0);
    two.fresh_per_site = 2;
    let mut planner = Planner::new(two, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.fresh), (5, 3), "two of a.example's, one of b.example's; the page with a 2019 URL goes with the rest");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_inputs_leave_the_task_api_s_list_only_once_their_pages_are_forgotten() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("dropped-order");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let story = v2(record("https://news.example/story/d", &article(2), "article", "en"));
    log(&pages, 1, vec![story.clone()]).await;
    let mut one = settings(dir.join("state"), 3600.0, 0);
    one.pages_per_input = 1;
    let mut planner = Planner::new(one, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    let entry: Value = serde_json::from_str(&planner.state.pending().unwrap()[0].1).unwrap();
    let dropped = json!({"input_key": entry["input_key"], "pages": entry["pages"]}).to_string();
    let _: i64 = redis.rpush(EMBED_DROPPED, &dropped).await.unwrap();

    assert_eq!(planner.forget_dropped().await.unwrap(), 1);
    let key = page_key(&story.url).unwrap();
    assert!(planner.state.planned(publisher::planner::RECIPE_V2, &key).unwrap().is_none());
    assert!(planner.state.waiting(&key).unwrap().is_some());
    assert_eq!(redis.llen::<_, i64>(EMBED_DROPPED).await.unwrap(), 0);
    let _: i64 = redis.rpush(EMBED_DROPPED, &dropped).await.unwrap();
    assert_eq!(planner.forget_dropped().await.unwrap(), 1, "one read again, as after a crash before the trim, changes nothing");
    assert!(planner.state.waiting(&key).unwrap().is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_more_pages_wait_than_the_cap_and_the_count_survives_a_restart() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("waiting-cap");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &format!("{n} {}", article(2)), "article", "en"));
    log(&pages, 1, (0..5).map(page).collect()).await;
    let mut capped = settings(dir.join("state"), 4.0, 0);
    (capped.max_waiting, capped.pages_per_input) = (2, 1);
    let mut planner = Planner::new(capped, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting, planner.counts.not_kept), (1, 2, 2));
    assert_eq!(planner.state.waiting_count(), 2);
    drop(planner);

    let mut roomy = settings(dir.join("state"), 3600.0, 0);
    roomy.pages_per_input = 1;
    let mut planner = Planner::new(roomy, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert_eq!(planner.state.waiting_count(), 2, "kept on disk");
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.state.waiting_count()), (2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_undated_articles_on_an_undated_news_article_is_planned_and_counted_apart() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("undated-article");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |path: &str, page_type: &str| v2(record(&format!("https://news.example/{path}"), &format!("{path} {}", article(2)), page_type, "en"));
    log(&pages, 1, vec![page("story/a", "article"), page("topic/b", "listing"), page("2019/04/05/c", "article")]).await;
    let set = dir.join("v1-set.json");
    let write_set = |undated: &str| {
        std::fs::write(&set, json!({"undated": undated, "sites": [{"host": "news.example", "rule": "news", "since": "2020-01-01"}]}).to_string()).unwrap();
    };
    for (undated, state, want) in [("out", "off", (0, 2, 0, 1)), ("article", "on", (1, 1, 1, 1))] {
        write_set(undated);
        let mut with_set = settings(dir.join(state), 3600.0, 0);
        with_set.set_file = Some(set.clone());
        let mut planner = Planner::new(with_set, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
        planner.pass().await.unwrap();
        let counts = &planner.counts;
        assert_eq!((counts.planned, counts.undated, counts.undated_article_in, counts.before_since), want, "undated {undated}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_input_goes_to_the_task_api_as_it_is_written_and_inputs_left_from_before_a_restart_go_first() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("hand-over");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &format!("{n} {}", article(2)), "article", "en"));
    log(&pages, 1, (0..3).map(page).collect()).await;
    let mut one_pending = settings(dir.join("state"), 3600.0, 0);
    (one_pending.pages_per_input, one_pending.max_pending) = (1, 1);
    let mut planner = Planner::new(one_pending, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(10)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.no_room, planner.counts.handed_over), (3, 0, 3), "one input pending at most holds up no page");
    assert_eq!(redis.llen::<_, i64>(EMBED_INPUTS).await.unwrap(), 3);

    let () = redis.del(EMBED_INPUTS).await.unwrap();
    log(&pages, 2, vec![page(3)]).await;
    planner.room = Room::Fixed(0);
    planner.pass().await.unwrap();
    assert_eq!(planner.state.pending().unwrap().len(), 1);
    drop(planner);
    let unreadable = bucket(stub.addr, "no-log");
    let mut restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), unreadable, temp.clone(), redis.clone(), Room::Fixed(10)).await.unwrap();
    assert!(restarted.pass().await.is_err(), "the log cannot be read");
    assert_eq!(redis.llen::<_, i64>(EMBED_INPUTS).await.unwrap(), 1, "the input written before the restart went out first");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waiting_pages_are_read_again_only_once_the_budget_holds_an_input_s_worth() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-budget");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize| v2(record(&format!("https://news.example/story/{n}"), &format!("{n} {}", article(2)), "article", "en"));
    log(&pages, 1, (0..5).map(page).collect()).await;
    let mut five = settings(dir.join("state"), 150.0, 0);
    five.max_pending = 1;
    let mut planner = Planner::new(five, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.no_room), (3, 2), "five pages of budget; the first input fills the room");

    planner.room = Room::Fixed(5);
    planner.hand_over().await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.handed_over, planner.counts.replanned), (1, 0), "two pages of budget left, less than an input of three");
    assert_eq!(planner.state.waiting_count(), 2);
    drop(planner);

    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.planned, planner.state.waiting_count()), (2, 2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_version_planned_before_change_files_were_kept_is_left_to_the_backfill_not_kept_waiting() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("unnumbered");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |name: &str| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    let (old, stuck, kept, later) = (page("old"), page("stuck"), page("kept"), page("later"));
    let [old_key, stuck_key, kept_key, later_key] = [&old, &stuck, &kept, &later].map(|page| page_key(&page.url).unwrap());
    let raw_state = |write: &dyn Fn(&rocksdb::DB)| {
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        options.set_compression_type(rocksdb::DBCompressionType::Zstd);
        write(&rocksdb::DB::open(&options, dir.join("state")).unwrap());
    };
    raw_state(&|db| {
        db.put(format!("planned/{RECIPE_V2}/{old_key}"), &old.content_sha1).unwrap();
        db.put(format!("waiting/{stuck_key}"), format!("{}\t\t\t\t", stuck.content_sha1)).unwrap();
        db.put(format!("waiting/{kept_key}"), format!("{}\t{}\t\t9\tt1", kept.content_sha1, kept.fetched_at)).unwrap();
        db.put(format!("replan/{:020}/{kept_key}", 9), "").unwrap();
        db.put("waiting-count", "2").unwrap();
    });
    log(&pages, 1, vec![page("other")]).await;
    let planner = Planner::new(settings(dir.join("state"), 3600.0, 1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    assert_eq!(planner.state.waiting_count(), 1, "a page waiting with no change file is let go at start");
    assert!(planner.state.waiting(&stuck_key).unwrap().is_none());
    assert_eq!(planner.state.waiting(&kept_key).unwrap().and_then(|version| version.seq), Some(9), "one waiting with its change file stays");
    drop(planner);

    raw_state(&|db| {
        db.put(format!("waiting/{later_key}"), format!("{}\t\t\t\t", later.content_sha1)).unwrap();
        db.put("waiting-count", "2").unwrap();
    });
    let mut planner = Planner::new(settings(dir.join("state"), 3600.0, 1), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(5)).await.unwrap();
    assert!(planner.state.waiting(&later_key).unwrap().is_some(), "the start looks for them only once");

    let dropped = json!({"input_key": "embed/inputs/old.parquet", "pages": [{"page_key": old_key, "content_sha1": old.content_sha1}]});
    let _: i64 = redis.rpush(EMBED_DROPPED, dropped.to_string()).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.dropped, 1);
    assert!(planner.state.waiting(&old_key).unwrap().is_none(), "nothing would plan it from a change file");
    assert!(planner.state.planned(RECIPE_V2, &old_key).unwrap().is_none(), "forgotten, so the backfill or the log plans it again");
    log(&pages, 2, vec![old.clone()]).await;
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!(planner.state.planned(RECIPE_V2, &old_key).unwrap(), Some(old.content_sha1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_set_s_age_is_from_when_it_was_built_or_else_when_its_file_was_written() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("set-age");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    log(&pages, 1, vec![v2(record("https://news.example/story/a", &article(2), "article", "en"))]).await;
    let set = dir.join("v1-set.json");
    let sites = json!([{"host": "news.example", "rule": "news"}]);
    let three_days_ago = iso(publisher::service::now_us() - 3 * 86_400_000_000).replace("+00:00", "Z");
    let four_days_ago = iso(publisher::service::now_us() - 4 * 86_400_000_000).replace("+00:00", "Z");
    for (file, state, days) in [
        (json!({"built": three_days_ago, "sites": sites}), "built", 3.0),
        (json!({"sites": sites}), "written", 0.0),
        (json!({"built": "yesterday", "sites": sites}), "unparsable", 0.0),
    ] {
        std::fs::write(&set, file.to_string()).unwrap();
        let mut with_set = settings(dir.join(state), 3600.0, 0);
        with_set.set_file = Some(set.clone());
        let mut planner = Planner::new(with_set, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
        let age = planner.set_age().unwrap().as_secs_f64() / 86_400.0;
        assert!((age - days).abs() < 0.01, "{state}: {age} days");
        let (now, hour) = (std::time::Instant::now(), Duration::from_secs(3600));
        if state != "built" {
            assert_eq!(planner.stale_set_warning(now + hour), None, "a set built within two days is not warned about");
            continue;
        }
        assert_eq!(planner.stale_set_warning(now), None, "warned when it was read, so not again within the hour");
        std::fs::write(&set, file.to_string()).unwrap();
        assert!(planner.reload_set().unwrap());
        let warning = planner.stale_set_warning(now + hour).expect("the same build read again keeps the hourly timer");
        assert!(warning.contains("built 3.0 days ago"), "{warning}");
        assert_eq!(planner.stale_set_warning(now + hour + hour / 2), None);
        assert!(planner.stale_set_warning(now + hour * 2).is_some(), "and every hour after");
        std::fs::write(&set, json!({"built": four_days_ago, "sites": sites}).to_string()).unwrap();
        assert!(planner.reload_set().unwrap());
        let warning = planner.stale_set_warning(now + hour * 2 + hour / 2).expect("a new build time restarts the timer when it is read");
        assert!(warning.contains("built 4.0 days ago"), "{warning}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_shares_the_budget_with_waiting_pages_but_never_takes_what_the_re_plan_stopped_short_of() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-budget");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let page = |name: &str| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    log(&pages, 1, Vec::from(["a", "b", "w1", "w2"].map(page))).await;
    let mut held = settings(dir.join("state"), 3600.0, 0);
    (held.pages_per_input, held.max_pending) = (2, 1);
    let mut planner = Planner::new(held, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.inputs, planner.state.waiting_count()), (1, 2), "w1 and w2 wait for room");
    drop(planner);

    let row = |n: usize| {
        let record = v2(record(&format!("https://news.example/archive/{n}"), &format!("{n} {}", article(2)), "article", "en"));
        PageRow {
            key: page_key(&record.url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    shard(&current, 7, vec![(0..1_200).map(row).collect()]).await;
    let file_key = "changes/dt=2026-10-14/1.parquet";
    let file = pages.get(file_key).await.unwrap();
    pages.put(file_key, Bytes::from_static(b"unreadable for now"), PARQUET, None).await.unwrap();
    let mut backfilling = settings(dir.join("state"), 3600.0, 0);
    (backfilling.pages_per_input, backfilling.backfill, backfilling.backfill_shards) = (2, true, 7..8);
    let mut planner = Planner::new(backfilling, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(100_000)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..100 {
        planner.pass().await.unwrap();
        if planner.counts.backfill >= 115 {
            break;
        }
    }
    planner.pass().await.unwrap();
    assert!(
        planner.counts.backfill >= 115,
        "with the re-plan at its sweep's end, as the waiting pages' change file cannot be read, the backfill takes the budget"
    );

    pages.put(file_key, file, PARQUET, None).await.unwrap();
    let backfilled = planner.counts.backfill;
    let started = std::time::Instant::now();
    while planner.state.waiting_count() > 0 && started.elapsed() < Duration::from_secs(6) {
        planner.pass().await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        (planner.counts.replanned, planner.state.waiting_count(), planner.counts.backfill),
        (2, 0, backfilled),
        "the backfill never takes budget the re-plan stopped short of, so the re-plan takes it once it holds an input's worth"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backfill_pass_reads_at_most_max_pending_inputs_worth_so_the_log_is_read_between() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-bound");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let page = |name: &str| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    let row = |n: usize| {
        let record = v2(record(&format!("https://news.example/archive/{n}"), &format!("{n} {}", article(2)), "article", "en"));
        PageRow {
            key: page_key(&record.url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    shard(&current, 7, vec![(0..300).map(row).collect()]).await;
    log(&pages, 1, vec![page("first")]).await;
    let mut bounded = settings(dir.join("state"), 3_600_000.0, 0);
    (bounded.backfill, bounded.backfill_shards, bounded.max_pending) = (true, 7..8, 2);
    let mut planner = Planner::new(bounded, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(1_000)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..20 {
        planner.pass().await.unwrap();
        if planner.counts.backfill > 0 {
            break;
        }
    }
    assert_eq!((planner.counts.planned, planner.counts.backfill), (7, 6), "the log's page, then two inputs of three from current/");

    log(&pages, 2, vec![page("second")]).await;
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.backfill), (14, 12), "the next pass reads the log before going on");
    for _ in 0..100 {
        planner.pass().await.unwrap();
        if planner.counts.shards == 1 {
            break;
        }
    }
    assert_eq!((planner.counts.planned, planner.counts.backfill), (302, 300));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backfill_pass_also_ends_once_it_has_run_the_fresh_lane_s_flush_wait() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-slice");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let row = |n: usize| {
        let record = v2(record(&format!("https://news.example/archive/{n}"), &format!("{n} {}", article(2)), "article", "en"));
        PageRow {
            key: page_key(&record.url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    shard(&current, 7, vec![(0..20).map(row).collect()]).await;
    log(&pages, 1, vec![v2(record("https://news.example/story/first", &article(2), "article", "en"))]).await;
    let mut sliced = settings(dir.join("state"), 3_600_000.0, 0);
    (sliced.backfill, sliced.backfill_shards, sliced.fresh_flush_after) = (true, 7..8, Duration::ZERO);
    let mut planner = Planner::new(sliced, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(1_000)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..20 {
        planner.pass().await.unwrap();
        if planner.counts.backfill > 0 {
            break;
        }
    }
    assert_eq!(planner.counts.backfill, 1, "with no wait left, a pass reads one page, though ten inputs' worth would fit");
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.backfill, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A task API that answers the embed room `room` holds, or fails while it is below 0, takes 2.5 s for the next `slow` answers, and counts in `asked` how often it is asked.
async fn room_api(room: Arc<std::sync::atomic::AtomicI64>, slow: Arc<AtomicU64>, asked: Arc<AtomicU64>) -> String {
    use std::sync::atomic::Ordering::SeqCst;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let (mut request, mut chunk) = (Vec::new(), [0; 1024]);
            while !request.ends_with(b"\r\n\r\n") {
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&chunk[..read]),
                }
            }
            asked.fetch_add(1, SeqCst);
            if slow.fetch_update(SeqCst, SeqCst, |left| left.checked_sub(1)).is_ok() {
                tokio::time::sleep(Duration::from_millis(2_500)).await;
            }
            let room = room.load(SeqCst);
            let (status, body) = if room < 0 { ("503 Service Unavailable", String::new()) } else { ("200 OK", json!({"room_tasks": room}).to_string()) };
            let answer = format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
            let _ = socket.write_all(answer.as_bytes()).await;
        }
    });
    url
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn while_the_task_api_is_full_slow_or_down_inputs_are_held_without_asking_it_for_each_and_fresh_ones_go_first() {
    use std::sync::atomic::Ordering::SeqCst;
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("room-held");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let page = |name: &str, published: &str| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    let old = |name: &str, n: usize| (1..=n).map(|i| page(&format!("{name}{i}"), "2020-05-01")).collect::<Vec<_>>();
    log(&pages, 1, [vec![page("r0", "2020-05-01"), page("f1", &today)], old("r", 8)].concat()).await;
    let (room, slow, asked) = (Arc::new(std::sync::atomic::AtomicI64::new(0)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let api = room_api(room.clone(), slow.clone(), asked.clone()).await;
    let one = || {
        let mut one = settings(dir.join("state"), 3600.0, 0);
        (one.pages_per_input, one.max_pending) = (1, 50);
        one
    };
    let mut planner = Planner::new(one(), pages.clone(), temp.clone(), redis.clone(), Room::api(&api).unwrap()).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.inputs, asked.load(SeqCst)),
        (10, 2),
        "asked when the first input was saved, as the ninth was cut, and at the pass end, not for the one saved between"
    );

    room.store(1, SeqCst);
    planner.pass().await.unwrap();
    let handed: Vec<Value> = redis.lrange::<_, Vec<String>>(EMBED_INPUTS, 0, -1).await.unwrap().iter().map(|raw| serde_json::from_str(raw).unwrap()).collect();
    let urls: Vec<&str> = handed.iter().map(|entry| entry["pages"][0]["url"].as_str().unwrap()).collect();
    assert_eq!(urls, ["https://news.example/story/f1"], "the fresh input goes ahead of the one cut before it");
    assert_eq!(asked.load(SeqCst), 3, "the next pass asks only at its end");
    drop(planner);

    room.store(-1, SeqCst);
    log(&pages, 2, old("s", 10)).await;
    let mut planner = Planner::new(one(), pages.clone(), temp.clone(), redis.clone(), Room::api(&api).unwrap()).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.inputs, asked.load(SeqCst)), (10, 5), "a failed answer holds inputs too: asked at the start and the end");
    assert_eq!(planner.state.pending().unwrap().len(), 19);
    drop(planner);

    room.store(100, SeqCst);
    slow.store(1, SeqCst);
    log(&pages, 3, old("t", 10)).await;
    let mut planner = Planner::new(one(), pages.clone(), temp.clone(), redis.clone(), Room::api(&api).unwrap()).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.handed_over, asked.load(SeqCst)),
        (29, 7),
        "a slow answer with room holds inputs too: the 19 held go at the start, the 10 cut after at the end"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// One-page inputs written three at once, a change file read at a time so each input is cut after the last one's write started.
fn three_at_once(dir: PathBuf) -> Settings {
    let mut three = settings(dir, 3600.0, 0);
    (three.pages_per_input, three.reads, three.writes) = (1, 1, 3);
    three
}

/// The URLs of the saved inputs' pages, oldest input first.
fn saved_urls(planner: &Planner) -> Vec<String> {
    let mut saved: Vec<Value> = planner.state.pending().unwrap().iter().map(|(_, entry)| serde_json::from_str(entry).unwrap()).collect();
    saved.sort_by(|a, b| a["written_at"].as_f64().unwrap().total_cmp(&b["written_at"].as_f64().unwrap()));
    saved.iter().flat_map(|entry| entry["pages"].as_array().unwrap().iter().map(|page| page["url"].as_str().unwrap().to_string())).collect()
}

async fn handed_urls(redis: &mut ConnectionManager) -> Vec<String> {
    let handed: Vec<String> = redis.lrange(EMBED_INPUTS, 0, -1).await.unwrap();
    handed.iter().map(|raw| serde_json::from_str::<Value>(raw).unwrap()["pages"][0]["url"].as_str().unwrap().to_string()).collect()
}

/// Logs a page of its own in each of change files 1, 2 and 3; their URLs.
async fn three_files(pages: &Bucket) -> Vec<String> {
    let urls: Vec<String> = (1..=3).map(|n| format!("https://news.example/story/{n}")).collect();
    for (seq, url) in (1..).zip(&urls) {
        log(pages, seq, vec![v2(record(url, &format!("{url} {}", article(2)), "article", "en"))]).await;
    }
    urls
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_write_holds_back_the_inputs_cut_after_it_and_is_written_again_with_the_same_pages() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("write-failed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let urls = three_files(&pages).await;
    // The first write is answered late and by then fails every try; the other two are written meanwhile.
    stub.state.slow("PUT", "subnet-22/embed/inputs/", 1, Duration::from_secs(1));
    let failing = stub.state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        failing.fail("PUT", "subnet-22/embed/inputs/", 5);
    });
    let mut planner = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    let failed = planner.pass().await.unwrap_err();
    assert!(format!("{failed:#}").contains("writing embed/inputs/"), "{failed:#}");
    let saved = saved_urls(&planner);
    assert!(saved.len() < 3 && saved[..] == urls[..saved.len()], "only inputs cut before the failed one are saved: {saved:?}");
    assert_eq!(planner.state.cursor().unwrap(), Some(saved.len() as i64), "a restart would read the log again from the failed input's page");

    planner.pass().await.unwrap();
    assert_eq!(saved_urls(&planner), urls, "the next pass writes it again and saves all three in the order cut");
    assert_eq!((planner.counts.planned, planner.counts.inputs, planner.state.cursor().unwrap()), (3, 3, Some(3)));
    assert_eq!(stub.state.keys("subnet-22", "embed/inputs/").len(), 3, "the same input written again, not one cut anew");
    planner.room = Room::Fixed(10);
    planner.hand_over().await.unwrap();
    assert_eq!(handed_urls(&mut redis).await, urls);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_crash_while_inputs_are_written_loses_no_page_and_a_restart_plans_again_only_those_never_saved() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("write-crash");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let urls = three_files(&pages).await;
    stub.state.slow("PUT", "subnet-22/embed/inputs/", 1, Duration::from_secs(3));
    let mut planner = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(1), planner.pass()).await.is_err(), "stopped while the first write is answered late");
    let saved = saved_urls(&planner);
    assert_eq!(stub.state.keys("subnet-22", "embed/inputs/").len(), 2, "the other two are written");
    assert!(saved.len() < 3 && saved[..] == urls[..saved.len()], "none is saved behind the unwritten one: {saved:?}");
    assert_eq!(planner.state.cursor().unwrap(), Some(saved.len() as i64));
    drop(planner);

    let mut restarted = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    restarted.pass().await.unwrap();
    assert_eq!(saved_urls(&restarted), urls, "each page saved once, in log order");
    assert_eq!(restarted.counts.planned, 3 - saved.len() as u64, "only the pages of inputs never saved are planned again");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inputs_written_at_once_go_out_in_the_order_cut_whichever_is_written_first_and_a_page_seen_again_meanwhile_is_not_planned_twice() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("write-order");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let today = iso(publisher::service::now_us());
    let page = |name: &str, published: &str| {
        let mut page = v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
        page.published = published.into();
        page
    };
    log(&pages, 1, vec![page("r1", "2020-05-01")]).await;
    log(&pages, 2, vec![page("r2", "2020-06-01"), page("r1", "2020-05-01")]).await;
    log(&pages, 3, vec![page("f", &today)]).await;
    log(&pages, 4, vec![page("r3", "2020-07-01")]).await;
    stub.state.slow("PUT", "subnet-22/embed/inputs/", 1, Duration::from_millis(500));
    let mut planner = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(10)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.unchanged), (4, 1), "r1 seen again while its input is written is planned once");
    let urls: Vec<String> = ["r1", "f", "r2", "r3"].iter().map(|name| format!("https://news.example/story/{name}")).collect();
    assert_eq!(
        handed_urls(&mut redis).await,
        urls,
        "r1 goes when r3 is cut and the API is asked, though written last; f goes ahead of r2 and r3, saved with them at the pass end"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_removal_of_a_page_whose_input_is_being_written_forgets_it_once_the_input_is_saved() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("write-removed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let a = v2(record("https://news.example/story/a", &format!("a {}", article(2)), "article", "en"));
    log(&pages, 1, vec![a.clone()]).await;
    log_removal(&pages, 2, &a).await;
    log(&pages, 3, vec![a.clone()]).await;
    let mut planner = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.planned, planner.counts.removed, planner.counts.inputs),
        (2, 1, 2),
        "a came back with the same text after the removal, so it is planned again"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_input_whose_save_fails_stays_first_in_line_so_the_log_is_not_read_past_its_pages() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("save-failed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let a = v2(record("https://a.example/story", &format!("a {}", article(2)), "article", "en"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    log(&pages, 1, vec![a]).await;
    // An unreadable page count for a's site fails the save of any input holding a new page of it.
    let site_count = |count: &str| rocksdb::DB::open_default(dir.join("state")).unwrap().put("sites/a.example", count).unwrap();
    site_count("x");
    let mut planner = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(planner.pass().await.is_err());
    log(&pages, 2, vec![b]).await;
    assert!(planner.pass().await.is_err(), "the next pass tries the same input again before reading on");
    assert_eq!(
        (planner.state.cursor().unwrap(), planner.state.pending().unwrap().len(), planner.counts.inputs),
        (Some(0), 0, 0),
        "nothing saved, and the log kept before a"
    );
    drop(planner);

    site_count("0");
    let mut restarted = Planner::new(three_at_once(dir.join("state")), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    restarted.pass().await.unwrap();
    assert_eq!(saved_urls(&restarted), ["https://a.example/story", "https://news.example/story/b"]);
    assert_eq!(restarted.counts.inputs, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waiting_pages_are_read_again_from_several_change_files_at_once() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-reads");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    for seq in 1..=8u64 {
        log(&pages, seq, vec![v2(record(&format!("https://news.example/story/{seq}"), &format!("{seq} {}", article(2)), "article", "en"))]).await;
    }
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 7));
    drop(planner);

    stub.state.slow("GET", "desearch-pages/changes/dt=", 7, Duration::from_secs(1));
    let mut ample = settings(dir.join("state"), 3600.0, 0);
    (ample.pages_per_input, ample.reads) = (1, 4);
    let mut planner = Planner::new(ample, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    let started = std::time::Instant::now();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.planned), (7, 7));
    assert!(started.elapsed() < Duration::from_secs(4), "seven files a second late each, four at once: {:?}", started.elapsed());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restart_plans_and_hands_over_at_the_hourly_pace_from_the_start() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("restart-pace");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: u64| v2(record(&format!("https://news.example/story/{n}"), &format!("{n} {}", article(2)), "article", "en"));
    for seq in 1..=6 {
        log(&pages, seq, vec![page(seq)]).await;
    }
    let mut full = settings(dir.join("state"), 3600.0, 0);
    full.pages_per_input = 1;
    let mut planner = Planner::new(full, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.inputs, 6, "written while the API had no room");
    drop(planner);

    for seq in 7..=9 {
        log(&pages, seq, vec![page(seq)]).await;
    }
    // As in service: nothing banked, and both budgets together a page a second.
    let mut serving = settings(dir.join("state"), 3600.0, 0);
    (serving.pages_per_input, serving.fresh_per_hour, serving.budget_at_start) = (1, 0.0, Duration::ZERO);
    let mut planner = Planner::new(serving, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(100)).await.unwrap();
    let started = std::time::Instant::now();
    planner.pass().await.unwrap();
    let paced = (0.2 + started.elapsed().as_secs_f64()) as u64;
    assert!(planner.counts.planned <= paced && planner.counts.over_budget + planner.counts.planned == 3, "the log's new pages wait for budget");
    let mut handed: u64 = 0;
    while started.elapsed() < Duration::from_secs(3) {
        planner.hand_over().await.unwrap();
        handed = redis.llen(EMBED_INPUTS).await.unwrap();
        assert!(handed as f64 <= 0.2 + started.elapsed().as_secs_f64(), "{handed} inputs of a page by {:?}", started.elapsed());
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!((2..=3).contains(&handed), "a page a second: {handed}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_waiting_page_the_re_plan_buffered_still_waits_after_a_crash_when_its_change_file_expired_meanwhile() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-crash-expired");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let b = v2(record("https://news.example/story/b", &format!("b {}", article(2)), "article", "en"));
    one_planned_one_waiting(&dir, &pages, &temp, &redis, b.clone()).await;
    let mut roomy = settings(dir.join("state"), 3600.0, 0);
    roomy.pages_per_input = 10;
    let mut planner = Planner::new(roomy, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.inputs), (1, 0), "b is buffered, not written");
    pages.delete("changes/dt=2026-10-14/1.parquet").await.unwrap();
    pages.delete(&CHANGES.seq_key(1)).await.unwrap();
    planner.pass().await.unwrap();
    drop(planner);
    let restarted = Planner::new(settings(dir.join("state"), 3600.0, 0), pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    assert!(restarted.state.waiting(&page_key(&b.url).unwrap()).unwrap().is_some(), "the crash lost nothing");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_re_plan_fetches_only_the_change_files_its_room_will_plan_counting_inputs_its_characters_keep_small() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("replan-room-reads");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    for seq in 1..=8u64 {
        log(&pages, seq, vec![v2(record(&format!("https://news.example/story/{seq}"), &format!("{seq} {}", article(2)), "article", "en"))]).await;
    }
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.inputs, planner.counts.waiting), (1, 7));
    drop(planner);

    // Room for two more inputs of one page each, as the characters cap cuts every page alone.
    let mut small = settings(dir.join("state"), 3600.0, 0);
    (small.pages_per_input, small.chars_per_input, small.max_pending) = (3, 1, 3);
    let mut planner = Planner::new(small, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    let before = stub.state.count("GET");
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.replanned, planner.counts.inputs), (2, 2));
    assert_eq!(stub.state.count("GET") - before, 1 + 2 * 2, "the newest number, then an index and a file for each of the two");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Six one-page inputs saved while the API had no room, the state under `dir`.
async fn six_held(dir: &std::path::Path, pages: &Bucket, temp: &Bucket, redis: &ConnectionManager) {
    for seq in 1..=6 {
        log(pages, seq, vec![v2(record(&format!("https://news.example/story/{seq}"), &format!("{seq} {}", article(2)), "article", "en"))]).await;
    }
    let mut full = settings(dir.join("state"), 3600.0, 0);
    full.pages_per_input = 1;
    let mut planner = Planner::new(full, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.inputs, 6);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inputs_held_back_only_by_the_pace_are_offered_again_at_once_not_after_the_room_wait() {
    use std::sync::atomic::Ordering::SeqCst;
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("pace-asks");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    six_held(&dir, &pages, &temp, &redis).await;
    let (room, slow, asked) = (Arc::new(std::sync::atomic::AtomicI64::new(100)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let api = room_api(room, slow, asked.clone()).await;
    let mut serving = settings(dir.join("state"), 36.0, 0);
    (serving.fresh, serving.budget_at_start) = (None, Duration::ZERO);
    let mut planner = Planner::new(serving, pages.clone(), temp.clone(), redis.clone(), Room::api(&api).unwrap()).await.unwrap();
    for _ in 0..4 {
        planner.pass().await.unwrap();
    }
    assert_eq!((planner.counts.handed_over, asked.load(SeqCst)), (0, 8), "asked at each pass's start and end, none paced yet");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn from_a_zero_start_inputs_left_from_before_go_first_in_their_order_whatever_their_size() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("pace-order");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |name: &str| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    log(&pages, 1, vec![page("a1"), page("a2"), page("a3"), page("b")]).await;
    let mut full = settings(dir.join("state"), 3600.0, 0);
    full.pages_per_input = 3;
    let mut planner = Planner::new(full, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    planner.flush().await.unwrap();
    assert_eq!(planner.state.pending().unwrap().len(), 2, "a of three pages, then b of one");
    drop(planner);

    log(&pages, 2, vec![page("c")]).await;
    // Two pages a second, to plan and to hand over.
    let mut serving = settings(dir.join("state"), 7200.0, 0);
    (serving.pages_per_input, serving.fresh, serving.budget_at_start) = (1, None, Duration::ZERO);
    let mut planner = Planner::new(serving, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(100)).await.unwrap();
    let started = std::time::Instant::now();
    let mut handed = Vec::new();
    while handed.len() < 3 && started.elapsed() < Duration::from_secs(10) {
        planner.pass().await.unwrap();
        handed = handed_urls(&mut redis).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let first = |name: &str| format!("https://news.example/story/{name}");
    assert_eq!(handed, [first("a1"), first("b"), first("c")], "a goes first though b and c fit the pace sooner");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_no_budget_and_the_fresh_lane_off_inputs_left_from_before_still_go() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("pace-off");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    six_held(&dir, &pages, &temp, &redis).await;
    let mut stopped = settings(dir.join("state"), 0.0, 0);
    (stopped.fresh, stopped.budget_at_start) = (None, Duration::ZERO);
    let mut planner = Planner::new(stopped, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(100)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(planner.counts.handed_over, 6);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_log_s_pages_go_before_waiting_ones_as_the_re_plan_leaves_what_the_log_last_asked_for() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("log-first");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: u64| v2(record(&format!("https://news.example/story/{n}"), &format!("{n} {}", article(2)), "article", "en"));
    for seq in 1..=9 {
        log(&pages, seq, vec![page(seq)]).await;
    }
    let mut tight = settings(dir.join("state"), 4.0, 0);
    tight.pages_per_input = 1;
    let mut planner = Planner::new(tight, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.waiting), (1, 8));
    drop(planner);

    // Ten pages banked; at 300 an hour the test refills nothing that counts.
    for seq in 10..=13 {
        log(&pages, seq, vec![page(seq)]).await;
    }
    let mut ten = settings(dir.join("state"), 300.0, 0);
    (ten.pages_per_input, ten.max_pending) = (1, 50);
    let mut planner = Planner::new(ten, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.pass().await.unwrap();
    assert_eq!(
        (planner.counts.planned, planner.counts.replanned, planner.counts.over_budget),
        (6, 2, 0),
        "the log's four, then two waiting pages, leaving four for the log's next pass"
    );
    for seq in 14..=17 {
        log(&pages, seq, vec![page(seq)]).await;
    }
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.over_budget), (10, 0), "the log's next four find their budget");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_plans_no_page_of_a_site_the_set_marks_out_even_once_its_shard_is_being_read() {
    let Some((_turn, mut redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-out");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let row = |url: &str| {
        let mut record = v2(record(url, &format!("{url} {}", article(2)), "article", "en"));
        (record.domain, record.published) = (url.split('/').nth(2).unwrap().into(), "2024-01-01".into());
        PageRow {
            key: page_key(url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    let urls = [
        "https://news.example/a",
        "https://news.example/b",
        "https://junk.example/x",
        "https://spam.news.example/y",
        "https://later.example/c",
        "https://later.example/d",
    ];
    shard(&current, 7, vec![urls.iter().map(|url| row(url)).collect()]).await;
    log(&pages, 1, vec![v2(record("https://other.example/log", &article(2), "article", "en"))]).await;
    let set = dir.join("v1-set.json");
    let write_set = |later: &str| {
        let sites = json!([
            {"host": "news.example", "rule": "news", "since": "2020-01-01"},
            {"host": "spam.news.example", "rule": "out"},
            {"host": "junk.example", "rule": "out"},
            {"host": "later.example", "rule": later},
        ]);
        std::fs::write(&set, json!({"sites": sites}).to_string()).unwrap();
    };
    write_set("whole");
    // A page a pass, each handed over at the pass's end.
    let mut one = settings(dir.join("state"), 3600.0, 0);
    (one.backfill, one.backfill_shards, one.set_file, one.pages_per_input, one.max_pending) = (true, 7..8, Some(set.clone()), 1, 1);
    let mut planner = Planner::new(one, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(100)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..20 {
        planner.pass().await.unwrap();
        if planner.counts.backfill == 1 {
            break;
        }
    }
    assert_eq!(planner.counts.planned, 1, "a later.example page goes first, as its keys sort first");
    tokio::time::sleep(Duration::from_millis(20)).await;
    write_set("out");
    for _ in 0..50 {
        planner.pass().await.unwrap();
        if planner.counts.shards == 1 {
            break;
        }
    }
    let planned: HashSet<String> = handed_urls(&mut redis).await.into_iter().collect();
    let later = if planned.contains(urls[4]) { urls[4] } else { urls[5] };
    assert_eq!(planned, HashSet::from([urls[0].to_string(), urls[1].to_string(), later.to_string()]));
    assert_eq!(planner.counts.backfill, 4, "out sites' pages are never handed to the planner, and the other later.example page is refused");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_shard_the_backfill_cannot_read_holds_up_neither_the_pass_nor_the_log_and_is_tried_again_later() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("backfill-failed");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let row = |url: &str| {
        let record = v2(record(url, &format!("{url} {}", article(2)), "article", "en"));
        PageRow {
            key: page_key(url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    shard(&current, 7, vec![vec![row("https://news.example/a")]]).await;
    current.put("current/shard=0007/base-000000000000.parquet", Bytes::from_static(b"not a parquet file"), PARQUET, None).await.unwrap();
    log(&pages, 1, vec![v2(record("https://news.example/log", &article(2), "article", "en"))]).await;
    let mut backfilling = settings(dir.join("state"), 3600.0, 0);
    (backfilling.backfill, backfilling.backfill_shards, backfilling.pages_per_input) = (true, 7..8, 1);
    let mut planner = Planner::new(backfilling, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    for _ in 0..3 {
        planner.pass().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert_eq!((planner.counts.planned, planner.counts.shards), (1, 0), "the log's page is planned");
    let before = stub.state.count("GET");
    planner.pass().await.unwrap();
    assert_eq!(stub.state.count("GET") - before, 1, "only the log's newest number is read while the shard waits to be tried again");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_backfill_leaves_what_the_log_last_asked_for_too() {
    let Some((_turn, redis)) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("log-first-backfill");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp, current) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"), bucket(stub.addr, "desearch-current"));
    let page = |name: String| v2(record(&format!("https://news.example/story/{name}"), &format!("{name} {}", article(2)), "article", "en"));
    let row = |n: u64| {
        let record = page(format!("archive-{n}"));
        PageRow {
            key: page_key(&record.url).unwrap(),
            removed: false,
            change_seq: 1,
            version: record_version(&record),
            published_at: record.fetched_at.clone(),
            record,
        }
    };
    shard(&current, 7, vec![(0..10).map(row).collect()]).await;
    for seq in 1..=4 {
        log(&pages, seq, vec![page(format!("{seq}"))]).await;
    }
    // Ten pages banked; at 300 an hour the test refills nothing that counts.
    let mut ten = settings(dir.join("state"), 300.0, 0);
    (ten.backfill, ten.backfill_shards, ten.pages_per_input, ten.max_pending) = (true, 7..8, 1, 50);
    let mut planner = Planner::new(ten, pages.clone(), temp.clone(), redis.clone(), Room::Fixed(0)).await.unwrap();
    planner.current = Some(current.clone());
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.backfill), (6, 2), "the log's four, then two of current/, leaving four for the log");
    for seq in 5..=8 {
        log(&pages, seq, vec![page(format!("{seq}"))]).await;
    }
    planner.pass().await.unwrap();
    assert_eq!((planner.counts.planned, planner.counts.over_budget), (10, 0), "the log's next four find their budget");
    let _ = std::fs::remove_dir_all(&dir);
}

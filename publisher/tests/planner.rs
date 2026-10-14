//! The planner: the texts a page gives, and the change log turned into embed inputs within the budget and the task API's room.

use std::collections::HashSet;
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
use publisher::planner::{page_text, passages, texts, Planner, Room, Settings, EMBED_INPUTS, PAGE, PASSAGE};
use publisher::records::{page_key, record_version, Record};
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const REDIS: &str = "redis://127.0.0.1:6379/13";

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
}

#[test]
fn passages_group_paragraphs_and_split_long_lines_at_sentences() {
    let grouped = passages(&["a".repeat(400), "b".repeat(400), "c".repeat(100)].join("\n"));
    assert_eq!(grouped, [format!("{}\n{}\n{}", "a".repeat(400), "b".repeat(400), "c".repeat(100))], "short paragraphs join; a small tail joins the last");

    let sentence = format!("{}. ", "x".repeat(299));
    let long_line = sentence.repeat(12);
    let pieces = passages(&long_line);
    assert!(pieces.iter().all(|p| p.chars().count() <= 1_500 + 300), "{:?}", pieces.iter().map(|p| p.chars().count()).collect::<Vec<_>>());
    assert!(pieces.iter().all(|p| p.ends_with('.')), "split after a sentence");
    assert_eq!(pieces.concat().replace(['.', ' ', '\n'], "").len(), 12 * 299, "nothing lost");

    let mut description = record("https://news.example/d", "Opening line. More.", "other", "en");
    description.description = Some("Opening line.".into());
    assert_eq!(page_text(&description), "A title\nOpening line. More.", "a description the text opens with is not repeated");
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

fn settings(dir: PathBuf, pages_per_hour: f64, start_after: i64) -> Settings {
    Settings {
        dir,
        pages_per_hour,
        pages_per_input: 3,
        flush_after: Duration::from_secs(3600),
        max_pending: 10,
        pending_max_age: Duration::from_secs(3600),
        min_chars: 300,
        langs: vec!["en".into()],
        excluded: HashSet::from(["spam.example".to_string()]),
        start_after: Some(start_after),
        poll: Duration::from_millis(10),
        hole_wait: Duration::from_secs(3600),
    }
}

async fn redis_db() -> Option<ConnectionManager> {
    let client = redis::Client::open(REDIS).ok()?;
    let mut redis = tokio::time::timeout(Duration::from_secs(2), client.get_connection_manager()).await.ok()?.ok()?;
    let () = redis::cmd("FLUSHDB").query_async(&mut redis).await.ok()?;
    Some(redis)
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
    let Some(mut redis) = redis_db().await else {
        eprintln!("skipped: no Redis on {REDIS}");
        return;
    };
    let dir = scratch("flow");
    let stub = stub::start(dir.join("r2"), "127.0.0.1:0").await.unwrap();
    let (pages, temp) = (bucket(stub.addr, "desearch-pages"), bucket(stub.addr, "subnet-22"));
    let page = |n: usize, text: &str, kind: &str, lang: &str| record(&format!("https://news.example/story/{n}"), text, kind, lang);
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
    let rows = input_rows(file);
    assert_eq!(entry["texts"], rows.len());
    assert_eq!(rows.iter().filter(|(_, kind, _)| kind == "page").count(), 2);
    assert_eq!(rows.iter().filter(|(_, kind, _)| kind == "passage").count(), 16, "the long article's passages");
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
}

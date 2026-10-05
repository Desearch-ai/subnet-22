//! Ready lists, the outcome feed and the dispatcher, on throwaway stores and a fake task API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use desearch_bot::allowed::{self, Allowed};
use desearch_bot::buckets::{bucket_of, BucketStore, Buckets, Changes, Resources};
use desearch_bot::dispatch::{allot, batch_id, interleave, settle, shares, slots, split, Dispatcher, HourlyCap, Pace, Progress, Want};
use desearch_bot::hotkey::{self, Hotkey};
use desearch_bot::outcomes::{self, Next, OutcomeFeed, OutcomeRow};
use desearch_bot::ready::{Order, Outcome, PageOutcome, Pick, Reason, SentPage, BACKOFF, CHANGED, FRESH, LANES, NO_OUTCOME, REQUEUED};
use desearch_bot::records;
use desearch_bot::states::State;
use desearch_bot::taskapi::{QueuedUrl, TaskApi};
use desearch_bot::timetable::UNRANKED;
use desearch_bot::urls::{self, Record, Url, HTTPS};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const RECRAWL: u32 = 7 * 86_400;

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let path = std::env::temp_dir().join(format!("desearch-bot-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        Dir(path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn resources() -> Resources {
    Resources::new(8 << 20, 8 << 20, 1)
}

fn open(dir: &Dir) -> BucketStore {
    BucketStore::open(&dir.0, &resources()).unwrap()
}

fn url(path: &str) -> Url {
    urls::parse(&format!("https://example.com/{path}"), "example.com").unwrap()
}

fn page(path: &str, lastmod: u32) -> (Url, u32, bool) {
    (url(path), lastmod, false)
}

/// Everything due, new pages first, as a test wants it.
fn take(store: &BucketStore, want: usize) -> Vec<Pick> {
    let mut quotas = [0; LANES];
    quotas[FRESH] = want;
    store.take_ready("example.com", quotas).unwrap()
}

fn rest(path: &str) -> Vec<u8> {
    format!("example.com/{path}").into_bytes()
}

fn record(store: &BucketStore, path: &str) -> Record {
    store.url(&url(path)).unwrap().unwrap()
}

/// The ready list of example.com, as paths, in the order they go.
fn ready(store: &BucketStore) -> Vec<String> {
    store.ready_list("example.com", 1000).unwrap().iter().map(|e| String::from_utf8(e.rest[12..].to_vec()).unwrap()).collect()
}

fn orders(store: &BucketStore, domain: &str) -> Vec<Order> {
    store.ready_list(domain, 1000).unwrap().iter().map(|e| e.order).collect()
}

#[test]
fn new_pages_join_their_ready_list_newest_first_without_junk() {
    let dir = Dir::new("ready-order");
    let store = open(&dir);
    assert!(store.take_noticed().is_empty(), "growth is counted once a dispatcher looks");
    let listing = store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20), page("tag/x", 30), page("files/a.pdf", 5)], 100).unwrap();
    assert_eq!((listing.new, listing.ready), (4, 2));
    let listing = store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20), page("c", 5)], 200).unwrap();
    assert_eq!((listing.new, listing.ready), (1, 1));
    assert_eq!(ready(&store), ["c", "b", "a"], "newest found first, then newest lastmod");
    assert_eq!(store.ready_domains().unwrap(), [("example.com".to_string(), 3)]);
    assert_eq!(store.take_noticed(), HashMap::from([("example.com".to_string(), 3)]));
    assert!(store.take_noticed().is_empty());

    let listing = store.record_listing("example.com", 7, vec![page("a", 11), page("b", 20), page("c", 5)], 300).unwrap();
    assert_eq!((listing.moved, listing.ready), (1, 0), "a page never sent is on its list already");
    assert_eq!(store.ready_count("example.com").unwrap(), 3);
}

#[test]
fn a_sent_page_returns_once_its_lastmod_moves_past_what_it_was_sent_with() {
    let dir = Dir::new("ready-changed");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20)], 100).unwrap();
    let picks = take(&store, 10);
    assert_eq!(picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/b", "https://example.com/a"]);
    assert!(picks.iter().all(|p| p.reason == Reason::Fresh));
    store.mark_sent(&picks, 150).unwrap();
    assert_eq!(store.ready_count("example.com").unwrap(), 0);
    assert!(store.ready_domains().unwrap().is_empty());
    assert_eq!(record(&store, "a").pushed_at, 150);
    assert_eq!(store.sent("example.com", &rest("a")).unwrap().unwrap().lastmod, 10);

    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 25)], 200).unwrap();
    assert_eq!(orders(&store, "example.com"), [Order::Changed { lastmod: 25 }]);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 30)], 300).unwrap();
    let picks = take(&store, 10);
    assert_eq!(picks.len(), 1, "a page changed twice before it went goes once");
    assert_eq!((picks[0].reason, picks[0].lastmod), (Reason::Changed, 30));
    store.mark_sent(&picks, 400).unwrap();
    assert_eq!(store.ready_count("example.com").unwrap(), 0);

    let listing = store.record_listing("example.com", 7, vec![page("a", 9), page("b", 30)], 500).unwrap();
    assert_eq!((listing.moved, listing.ready), (1, 0), "a date older than the one sent is no change");
}

#[test]
fn a_sitemap_that_restamps_its_pages_queues_only_the_ten_newest() {
    let dir = Dir::new("ready-restamp");
    let store = open(&dir);
    let listed = |lastmod: &dyn Fn(u32) -> u32| (0..200).map(|i| page(&format!("p{i}"), lastmod(i))).collect::<Vec<_>>();
    store.record_listing("example.com", 7, listed(&|i| 1000 + i), 100).unwrap();
    let picks = take(&store, 1000);
    assert_eq!(picks.len(), 200);
    store.mark_sent(&picks, 150).unwrap();

    let listing = store.record_listing("example.com", 7, listed(&|i| 5000 + i), 200).unwrap();
    assert_eq!((listing.moved, listing.ready), (200, 10));
    let lastmods: Vec<u32> = orders(&store, "example.com").iter().map(|o| if let Order::Changed { lastmod } = o { *lastmod } else { 0 }).collect();
    assert_eq!(lastmods, (5190..5200).rev().collect::<Vec<_>>());

    let listing = store.record_listing("example.com", 7, listed(&|i| if i < 3 { 9000 + i } else { 5000 + i }), 300).unwrap();
    assert_eq!((listing.moved, listing.ready), (3, 3), "a few real changes all go");
}

#[test]
fn outcomes_stamp_crawls_and_bring_failures_back_after_a_backoff() {
    let dir = Dir::new("ready-outcomes");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("undated", 0)], 100).unwrap();
    let picks = take(&store, 10);
    store.mark_sent(&picks, 1000).unwrap();
    let row = |path: &str, outcome: Outcome, at: u32| PageOutcome { domain: "example.com".into(), rest: rest(path), outcome, at };

    let applied = store.apply_outcomes(&[row("a", Outcome::Published, 2000), row("undated", Outcome::Unchanged, 2000)], RECRAWL).unwrap();
    assert_eq!(applied.crawled, 2);
    assert_eq!(record(&store, "a").crawled_at, 2000);
    assert_eq!(store.waiting().unwrap(), [(2000 + RECRAWL, "example.com".to_string(), rest("undated"))], "only a page without a lastmod is re-crawled on a schedule");

    let mut at = 3000;
    for (failures, wait) in BACKOFF.iter().enumerate() {
        assert_eq!(store.apply_outcomes(&[row("a", Outcome::Failed, at)], RECRAWL).unwrap().retried, 1);
        assert_eq!(store.requeue_due(at + wait - 1, 100).unwrap(), 0);
        assert_eq!(store.requeue_due(at + wait, 100).unwrap(), 1);
        let picks = take(&store, 10);
        assert_eq!(picks.len(), 1);
        assert_eq!((picks[0].reason, picks[0].retries), (Reason::Requeued, failures as u8 + 1));
        at += wait + 10;
        store.mark_sent(&picks, at).unwrap();
        at += 100;
    }
    let applied = store.apply_outcomes(&[row("a", Outcome::Dropped, at)], RECRAWL).unwrap();
    assert_eq!((applied.retried, applied.gave_up), (0, 1), "after three retries a page waits for a new lastmod");
    assert_eq!(store.waiting().unwrap().len(), 1);
    assert_eq!(store.apply_outcomes(&[row("a", Outcome::Failed, 500)], RECRAWL).unwrap().ignored, 1, "an outcome of an earlier send");
    assert_eq!(store.apply_outcomes(&[row("gone", Outcome::Failed, at)], RECRAWL).unwrap().ignored, 1);

    store.record_listing("example.com", 7, vec![page("a", 20), page("undated", 0)], at).unwrap();
    let picks = take(&store, 10);
    assert_eq!((picks.len(), picks[0].reason), (1, Reason::Changed));
    store.mark_sent(&picks, at + 1).unwrap();
    assert_eq!(store.sent("example.com", &rest("a")).unwrap().unwrap().retries, 0, "a new lastmod starts the retries over");

    assert_eq!(store.requeue_due(2000 + RECRAWL, 100).unwrap(), 1);
    let picks = take(&store, 10);
    assert_eq!((picks.len(), picks[0].url.as_str(), picks[0].reason), (1, "https://example.com/undated", Reason::Requeued));
}

#[test]
fn the_backfill_queues_old_pages_once_and_resumes_where_it_stopped() {
    let dir = Dir::new("ready-backfill");
    {
        let db = rocksdb::DB::open_default(&dir.0).unwrap();
        let put = |key: String, lastmod: u32| {
            let record = Record { sitemap_id: 1, lastmod, first_seen: 50, last_seen: 50, flags: HTTPS, ..Record::default() };
            db.put([b"U", key.as_bytes()].concat(), record.pack()).unwrap();
        };
        for i in 0..5 {
            put(format!("example.com\0example.com/old-{i}"), 10 + i);
        }
        put("example.com\0example.com/tag/old".into(), 10);
        put("example.com\0example.com/sent-same".into(), 30);
        put("example.com\0example.com/sent-moved".into(), 40);
        for i in 0..100 {
            put(format!("big.org\0big.org/p{i:03}"), 100 + i);
        }
    }
    let store = open(&dir);
    let sent_page = |domain: &str, rest: Vec<u8>, lastmod: u32| SentPage { domain: domain.into(), rest, lastmod, at: 60 };
    let mut sent = vec![sent_page("example.com", rest("sent-same"), 30), sent_page("example.com", rest("sent-moved"), 35)];
    sent.extend((0..60).map(|i| sent_page("big.org", format!("big.org/p{i:03}").into_bytes(), 0)));
    sent.push(sent_page("example.com", rest("never-listed"), 1));
    assert_eq!(store.import_sent(&sent).unwrap(), 62);
    assert_eq!(record(&store, "sent-same").pushed_at, 60);

    let mut state = store.backfill_state(1000).unwrap();
    store.backfill_step(&mut state, 3).unwrap();
    assert!(!state.done);
    drop(state);
    let mut state = store.backfill_state(5000).unwrap();
    assert_eq!(state.started, 1000, "a resumed walk keeps its start");
    while !state.done {
        store.backfill_step(&mut state, 7).unwrap();
    }
    let mut example = ready(&store);
    example.sort();
    assert_eq!(example, ["old-0", "old-1", "old-2", "old-3", "old-4", "sent-moved"]);
    assert_eq!(store.ready_count("example.com").unwrap(), 6);
    let big = orders(&store, "big.org");
    assert_eq!(big.len(), 50, "40 never sent, and only the 10 newest of 60 re-dated pages");
    assert_eq!(big[0], Order::Changed { lastmod: 159 });
    assert_eq!(big[10], Order::Fresh { first_seen: 50, lastmod: 199 });
    assert_eq!(store.ready_count("big.org").unwrap(), 50);

    let mut again = store.backfill_state(9000).unwrap();
    assert!(again.done);
    store.backfill_step(&mut again, 100).unwrap();
    assert_eq!(store.ready_count("example.com").unwrap(), 6);
}

#[test]
fn turns_go_best_rank_first_with_more_slots_and_resume_where_they_stopped() {
    assert_eq!([1, 9, 10, 99, 100, 9_999, 10_000, 999_999, 1_000_000, UNRANKED].map(slots), [7, 7, 6, 6, 5, 4, 3, 2, 1, 1]);
    let want = |rank: i64, ready: u64, allowance: u64| Want { rank, ready, allowance };
    let wants = [want(1, 100, 100), want(500, 100, 100), want(2_000_000, 100, 100)];
    assert_eq!(allot(&wants, 13, 0), (vec![7, 5, 1], 0));
    let (given, next) = allot(&wants, 10, 0);
    assert_eq!((given, next), (vec![7, 3, 0], 2));
    assert_eq!(allot(&wants, 3, next).0, [2, 0, 1], "the next pass starts where this one stopped");
    assert_eq!(allot(&[want(1, 100, 3), want(500, 100, 100)], 20, 0).0, [3, 17], "the hourly allowance caps a domain");
    assert_eq!(allot(&[want(1, 4, 100)], 50, 0).0, [4]);
    assert_eq!(allot(&[], 50, 0).0, Vec::<u64>::new());
}

#[test]
fn the_hourly_cap_forgets_sends_older_than_an_hour() {
    let mut cap = HourlyCap::new(2000);
    cap.record("a.com", 1000, 1500);
    assert_eq!(cap.allowance("a.com", 1000), 500);
    cap.record("a.com", 2000, 500);
    assert_eq!(cap.allowance("a.com", 4599), 0);
    assert_eq!(cap.allowance("a.com", 4600), 1500);
    assert_eq!(cap.allowance("b.com", 4600), 2000);
}

#[test]
fn a_batch_id_depends_only_on_the_batch() {
    let urls = |paths: &[&str]| paths.iter().map(|p| QueuedUrl { host: "example.com".into(), url: format!("https://example.com/{p}") }).collect::<Vec<_>>();
    assert_eq!(batch_id(&urls(&["a", "b"])), batch_id(&urls(&["a", "b"])));
    assert_ne!(batch_id(&urls(&["a", "b"])), batch_id(&urls(&["a", "c"])));
    assert_ne!(batch_id(&urls(&["a", "b"])), batch_id(&urls(&["b", "a"])));
    assert_eq!(batch_id(&urls(&["a"])).len(), 64);
    assert_eq!(interleave(vec![vec![1, 2, 3], vec![10], vec![20, 21]]), [1, 10, 20, 2, 21, 3]);
}

#[test]
fn outcome_files_parse_and_land_on_their_pages() {
    let data = std::fs::read(format!("{}/tests/data/outcomes.parquet", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let rows = outcomes::read_parquet(data.into()).unwrap();
    assert_eq!(rows.len(), 4, "an outcome this bot does not know is skipped");
    assert_eq!(rows[0], OutcomeRow { url: "https://example.com/news/a".into(), outcome: Outcome::Published, at: 1_759_400_000_500_000 });
    assert_eq!(rows.iter().map(|r| r.outcome).collect::<Vec<_>>(), [Outcome::Published, Outcome::Unchanged, Outcome::Failed, Outcome::Dropped]);

    let dir = Dir::new("outcome-files");
    let buckets = Buckets::open(&dir.0, &[bucket_of("example.com")], &resources()).unwrap();
    let store = buckets.store("example.com");
    let listed = ["https://example.com/news/a", "https://www.example.com/news/b", "http://blog.example.com/c", "https://example.com/d"];
    store.record_listing("example.com", 7, listed.iter().map(|u| (urls::parse(u, "example.com").unwrap(), 10, false)).collect(), 100).unwrap();
    let picks = take(store, 10);
    let mut sent: Vec<&str> = picks.iter().map(|p| p.url.as_str()).collect();
    sent.sort();
    assert_eq!(sent, ["http://blog.example.com/c", "https://example.com/d", "https://example.com/news/a", "https://www.example.com/news/b"]);
    store.mark_sent(&picks, 1_759_399_000).unwrap();
    for url in listed {
        assert_eq!(outcomes::locate(&buckets, url).map(|found| found.0).as_deref(), Some("example.com"), "{url}");
    }
    assert!(outcomes::locate(&buckets, "https://example.com/never-listed").is_none());

    let applied = outcomes::apply(&buckets, &rows, RECRAWL).unwrap();
    assert_eq!((applied.crawled, applied.retried, applied.ignored), (2, 2, 0));
    assert_eq!(store.url(&urls::parse(listed[0], "example.com").unwrap()).unwrap().unwrap().crawled_at, 1_759_400_000);
    assert_eq!(store.waiting().unwrap().len(), 2);

    assert_eq!(outcomes::saved_cursor(&buckets).unwrap(), None);
    outcomes::save_cursor(&buckets, 8).unwrap();
    assert_eq!(outcomes::saved_cursor(&buckets).unwrap(), Some(8));
}

#[derive(Default)]
struct FakeApi {
    room: i64,
    fail_next: usize,
    rooms: usize,
    enqueued: Vec<(HashMap<String, String>, Vec<u8>)>,
    files: HashMap<String, Vec<u8>>,
}

/// One request per connection: the room, or an enqueue that fails while `fail_next` lasts.
async fn serve_api(listener: TcpListener, api: Arc<Mutex<FakeApi>>) {
    while let Ok((mut socket, _)) = listener.accept().await {
        let api = api.clone();
        tokio::spawn(async move {
            let mut buffer = Vec::new();
            let end = loop {
                if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end;
                }
                let mut chunk = [0u8; 8192];
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                }
            };
            let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
            let mut lines = head.split("\r\n");
            let request_line = lines.next().unwrap_or_default().to_string();
            let mut headers: HashMap<String, String> =
                lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string())).collect();
            let length: usize = headers.get("content-length").and_then(|l| l.parse().ok()).unwrap_or(0);
            let mut body = buffer[end + 4..].to_vec();
            while body.len() < length {
                let mut chunk = [0u8; 65536];
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => body.extend_from_slice(&chunk[..n]),
                }
            }
            let (method, path) = request_line.split_once(' ').map(|(m, rest)| (m, rest.split(' ').next().unwrap_or(""))).unwrap_or_default();
            headers.insert(":method".into(), method.into());
            headers.insert(":path".into(), path.into());
            let (status, reply) = {
                let mut api = api.lock().unwrap();
                match path {
                    "/v1/room" => {
                        api.rooms += 1;
                        (200, format!(r#"{{"room_tasks":{},"queue":0,"unrevealed":0,"refusing":false}}"#, api.room))
                    }
                    "/v1/admin/enqueue" => {
                        api.enqueued.push((headers, body));
                        if api.fail_next > 0 {
                            api.fail_next -= 1;
                            (500, r#"{"detail":"try again"}"#.to_string())
                        } else {
                            (200, r#"{"round_id":"r1","batches":1}"#.to_string())
                        }
                    }
                    _ => match api.files.get(path) {
                        Some(file) => (200, String::from_utf8_lossy(file).into_owned()),
                        None => (404, r#"{"detail":"no such route"}"#.to_string()),
                    },
                }
            };
            let response = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len());
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dispatcher_fills_the_room_by_rank_within_the_hourly_cap_and_resends_a_failed_batch_unchanged() {
    let dir = Dir::new("dispatcher");
    let mut owned = vec![bucket_of("example.com"), bucket_of("example.org")];
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &resources()).unwrap());
    for (host, rank) in [("example.com", 5), ("example.org", 50_000)] {
        let store = buckets.store(host);
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(rank), Some("com"), &[], State::Active, None, None));
        store.write(changes).unwrap();
        let pages = (0..30).map(|i| (urls::parse(&format!("https://{host}/p{i:02}"), host).unwrap(), 100 + i, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, fail_next: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let key = Hotkey::from_uri("//test").unwrap();
    let dispatcher = |progress: Arc<Progress>| Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 20, shares(50, 10), progress);

    let now = 1_759_400_000;
    let progress = Arc::new(Progress::default());
    let mut first = dispatcher(progress.clone()).await.unwrap();
    assert_eq!(progress.ready.load(std::sync::atomic::Ordering::Relaxed), 60);
    assert_eq!(first.pass(now).await.unwrap(), 0, "the API failed the batch");
    drop(first);

    let mut second = dispatcher(progress.clone()).await.unwrap();
    assert_eq!(second.pass(now + 15).await.unwrap(), 40, "the batch kept across a restart went again");
    {
        let api = fake.lock().unwrap();
        assert_eq!(api.enqueued.len(), 2);
        assert_eq!(api.enqueued[0].1, api.enqueued[1].1, "sent again byte for byte, under the same batch_id");
        let body: Value = serde_json::from_slice(&api.enqueued[1].1).unwrap();
        let sent: Vec<(&str, &str)> =
            body["urls"].as_array().unwrap().iter().map(|u| (u["host"].as_str().unwrap(), u["url"].as_str().unwrap())).collect();
        assert_eq!(sent.len(), 40);
        assert_eq!(sent.iter().filter(|(host, _)| *host == "example.com").count(), 20, "20 an hour per domain");
        assert_eq!(&sent[..3], [("example.com", "https://example.com/p29"), ("example.org", "https://example.org/p29"), ("example.com", "https://example.com/p28")]);
        assert_eq!(body["batch_id"].as_str().unwrap().len(), 64);
        for (headers, body) in &api.enqueued {
            assert_eq!(headers["x-hotkey"], key.ss58());
            assert_eq!(headers["x-nonce"].len(), 32);
            let payload = hotkey::signing_payload("POST", "/v1/admin/enqueue", body, &headers["x-timestamp"], &headers["x-nonce"]);
            let signature: Vec<u8> = (0..128).step_by(2).map(|i| u8::from_str_radix(&headers["x-signature"][i..i + 2], 16).unwrap()).collect();
            assert!(hotkey::verify(&key.public(), &payload, &signature));
        }
    }
    let com = buckets.store("example.com");
    assert_eq!(com.ready_count("example.com").unwrap(), 10);
    assert_eq!(com.url(&urls::parse("https://example.com/p29", "example.com").unwrap()).unwrap().unwrap().pushed_at, now);

    assert_eq!(second.pass(now + 30).await.unwrap(), 0, "both domains used their hour");
    assert_eq!(fake.lock().unwrap().enqueued.len(), 2);
    assert_eq!(second.pass(now + 3600).await.unwrap(), 20, "an hour later the rest goes");
    assert_eq!(buckets.store("example.org").ready_count("example.org").unwrap(), 0);
    assert_eq!(progress.dispatched.load(std::sync::atomic::Ordering::Relaxed), 60);
    assert_eq!(progress.ready.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn each_lane_takes_its_quota_and_an_empty_lane_leaves_its_part_to_the_others() {
    let dir = Dir::new("ready-lanes");
    let store = open(&dir);
    store.record_listing("example.com", 7, (0..20).map(|i| page(&format!("old{i:02}"), 10)).collect(), 100).unwrap();
    let sent = take(&store, 20);
    store.mark_sent(&sent, 150).unwrap();
    store.record_listing("example.com", 7, (0..20).map(|i| page(&format!("old{i:02}"), 50)).chain((0..20).map(|i| page(&format!("new{i:02}"), 60))).collect(), 200).unwrap();

    let mut quotas = [0; LANES];
    quotas[FRESH] = 5;
    quotas[CHANGED] = 4;
    quotas[REQUEUED] = 1;
    let picks = store.take_ready("example.com", quotas).unwrap();
    let count = |reason| picks.iter().filter(|p| p.reason == reason).count();
    assert_eq!((count(Reason::Fresh), count(Reason::Changed)), (6, 4), "both lanes go at once; the empty retry lane's page goes to new pages");

    store.mark_sent(&picks, 250).unwrap();
    let mut quotas = [0; LANES];
    quotas[CHANGED] = 40;
    let rest = store.take_ready("example.com", quotas).unwrap();
    assert_eq!(rest.len(), 30, "what refreshes cannot fill, new pages do");
}

#[test]
fn small_wants_still_split_by_the_shares_over_many_passes() {
    let shares = shares(50, 10);
    let mut credit = [0.0; LANES];
    let mut sent = [0; LANES];
    for _ in 0..100 {
        let quotas = split(&mut credit, &shares, 1);
        settle(&mut credit, quotas, quotas);
        for lane in 0..LANES {
            sent[lane] += quotas[lane];
        }
    }
    assert_eq!((sent[FRESH], sent[CHANGED], sent[REQUEUED]), (50, 40, 10));
}

#[test]
fn a_lane_that_came_up_short_is_not_owed_a_burst_later() {
    let shares = shares(50, 10);
    let mut credit = [0.0; LANES];
    for _ in 0..50 {
        let quotas = split(&mut credit, &shares, 10);
        let mut taken = quotas;
        taken[CHANGED] += taken[REQUEUED];
        taken[REQUEUED] = 0;
        settle(&mut credit, quotas, taken);
    }
    assert!(credit.iter().all(|owed| (-1.0..=1.0).contains(owed)));
    assert!(split(&mut credit, &shares, 10)[REQUEUED] <= 2, "retries come back at their share and one page of credit, not all they missed");
}

#[test]
fn a_sent_page_with_no_outcome_in_a_day_comes_back_as_a_retry() {
    let dir = Dir::new("ready-overdue");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20)], 100).unwrap();
    let picks = take(&store, 10);
    store.mark_sent(&picks, 1000).unwrap();
    let answered = PageOutcome { domain: "example.com".into(), rest: rest("b"), outcome: Outcome::Published, at: 2000 };
    store.apply_outcomes(&[answered], RECRAWL).unwrap();

    assert_eq!(store.expire_unanswered(1000 + NO_OUTCOME - 1, 100).unwrap(), 0);
    assert_eq!(store.expire_unanswered(1000 + NO_OUTCOME, 100).unwrap(), 1, "b's outcome came, a's never did");
    assert_eq!(store.requeue_due(1000 + NO_OUTCOME, 100).unwrap(), 1);
    let again = take(&store, 10);
    assert_eq!(again.iter().map(|p| (p.url.as_str(), p.reason, p.retries)).collect::<Vec<_>>(), [("https://example.com/a", Reason::Requeued, 1)]);

    store.mark_sent(&again, 1000 + NO_OUTCOME + 10).unwrap();
    store.record_listing("example.com", 7, vec![page("a", 30), page("b", 20)], 1000 + NO_OUTCOME + 20).unwrap();
    let changed = take(&store, 10);
    store.mark_sent(&changed, 1000 + NO_OUTCOME + 30).unwrap();
    assert_eq!(store.expire_unanswered(1000 + 2 * NO_OUTCOME + 10, 100).unwrap(), 0, "a deadline left from an earlier send of a resent page");
    assert_eq!(store.expire_unanswered(1000 + 2 * NO_OUTCOME + 30, 100).unwrap(), 1);
}

#[tokio::test]
async fn the_outcome_feed_tells_a_number_not_written_yet_from_one_that_is_gone() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi::default()));
    tokio::spawn(serve_api(listener, fake.clone()));
    let feed = OutcomeFeed::new(&base).unwrap();

    assert!(feed.latest().await.unwrap().is_none());
    assert!(matches!(feed.fetch(5).await.unwrap(), Next::Wait));
    fake.lock().unwrap().files.insert("/outcomes/latest.json".into(), br#"{"seq": 7}"#.to_vec());
    assert_eq!(feed.latest().await.unwrap(), Some(7), "where a bot reading the feed for the first time starts");
    assert!(matches!(feed.fetch(5).await.unwrap(), Next::Missing), "later numbers exist, so 5 is gone or late");
    assert!(matches!(feed.fetch(8).await.unwrap(), Next::Wait), "8 is simply not written yet");
}


#[test]
fn the_domain_list_takes_names_or_hosts_and_is_read_again_when_its_file_changes() {
    let names = allowed::parse(br#"["a.com", {"host": "b.org", "rank": 2}, 7]"#).unwrap();
    assert_eq!(names, ["a.com".to_string(), "b.org".to_string()].into_iter().collect());
    assert!(allowed::parse(b"{}").is_err());
    assert!(Allowed::all().allows("anything.net"));

    let dir = Dir::new("allowed");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["a.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    assert!(list.allows("a.com") && !list.allows("b.org"));
    assert!(!list.reload().unwrap(), "an unchanged file is not read again");
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, r#"["b.org"]"#).unwrap();
    assert!(list.reload().unwrap());
    assert!(list.allows("b.org") && !list.allows("a.com"));
}

#[test]
fn with_a_domain_list_only_listed_domains_get_ready_pages_and_the_backfill_jumps_the_rest() {
    let dir = Dir::new("ready-allowed");
    {
        let db = rocksdb::DB::open_default(&dir.0).unwrap();
        for domain in ["aaa.com", "example.com", "example.com.au", "zzz.org"] {
            for i in 0..50 {
                let record = Record { sitemap_id: 1, lastmod: 10, first_seen: 50, last_seen: 50, flags: HTTPS, ..Record::default() };
                db.put([b"U", format!("{domain}\0{domain}/p{i:02}").as_bytes()].concat(), record.pack()).unwrap();
            }
        }
    }
    let store = open(&dir);
    store.allow(Allowed::of(["example.com"]));

    let mut state = store.backfill_state(1000).unwrap();
    let mut steps = 0;
    while !state.done {
        store.backfill_step(&mut state, 20).unwrap();
        steps += 1;
    }
    assert_eq!(store.ready_count("example.com").unwrap(), 50);
    for other in ["aaa.com", "example.com.au", "zzz.org"] {
        assert_eq!(store.ready_count(other).unwrap(), 0, "{other} is not listed");
    }
    assert!(steps <= 7, "unlisted domains are jumped, not read: {steps} steps");

    let pages = (0..5).map(|i| (urls::parse(&format!("https://zzz.org/new{i}"), "zzz.org").unwrap(), 200, false)).collect();
    store.record_listing("zzz.org", 2, pages, 300).unwrap();
    assert_eq!(store.ready_count("zzz.org").unwrap(), 0, "a sitemap read of an unlisted domain queues nothing");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dispatcher_sends_only_listed_domains_even_with_pages_queued_before_the_list() {
    let dir = Dir::new("dispatcher-allowed");
    let mut owned = vec![bucket_of("example.com"), bucket_of("example.org")];
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &resources()).unwrap());
    for host in ["example.com", "example.org"] {
        let store = buckets.store(host);
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(5), Some("com"), &[], State::Active, None, None));
        store.write(changes).unwrap();
        let pages = (0..10).map(|i| (urls::parse(&format!("https://{host}/p{i}"), host).unwrap(), 100 + i, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
    }
    buckets.allow(&Allowed::of(["example.org"]));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 100, shares(50, 10), Arc::new(Progress::default())).await.unwrap();

    assert_eq!(dispatcher.pass(1_759_400_000).await.unwrap(), 10);
    let body: Value = serde_json::from_slice(&fake.lock().unwrap().enqueued[0].1).unwrap();
    assert!(body["urls"].as_array().unwrap().iter().all(|u| u["host"] == "example.org"));
    assert_eq!(buckets.store("example.com").ready_count("example.com").unwrap(), 10, "kept, not sent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn several_batches_go_out_together_and_only_a_failed_one_is_sent_again() {
    let dir = Dir::new("dispatcher-waves");
    let host = "example.net";
    let buckets = Arc::new(Buckets::open(&dir.0, &[bucket_of(host)], &resources()).unwrap());
    let store = buckets.store(host);
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(5), Some("net"), &[], State::Active, None, None));
    store.write(changes).unwrap();
    let pages = (0..25_000).map(|i| (urls::parse(&format!("https://{host}/p{i:05}"), host).unwrap(), 100 + i, false)).collect();
    store.record_listing(host, 1, pages, 100).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 30, fail_next: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let dispatcher = |progress: Arc<Progress>| Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 1_000_000, shares(50, 10), progress);

    let now = 1_759_400_000;
    let progress = Arc::new(Progress::default());
    let first = dispatcher(progress.clone()).await.unwrap().pass(now).await.unwrap();
    let failed = fake.lock().unwrap().enqueued[0].1.clone();
    assert_eq!(fake.lock().unwrap().enqueued.len(), 3, "three batches in one wave");
    assert!(first == 15_000 || first == 20_000, "all but the failed batch went: {first}");

    let second = dispatcher(progress.clone()).await.unwrap().pass(now + 15).await.unwrap();
    let api = fake.lock().unwrap();
    assert_eq!(api.enqueued[3].1, failed, "after a restart the failed batch went again unchanged, first");
    assert_eq!(first + second, 25_000);
    assert_eq!(buckets.store(host).ready_count(host).unwrap(), 0);
}

#[test]
fn the_pace_grows_steadily_and_saves_up_only_five_minutes() {
    let mut pace = Pace::new(3_600_000, 1000);
    assert_eq!(pace.allowance(1000), 300_000, "five minutes' worth to start");
    pace.spend(300_000);
    assert_eq!(pace.allowance(1060), 60_000, "a minute later, a minute's worth");
    assert_eq!(pace.allowance(1000 + 7200), 300_000, "an idle hour saves no more than five minutes");
    assert_eq!(Pace::new(0, 1000).allowance(2000), u64::MAX, "no pace set sends whatever there is room for");
}

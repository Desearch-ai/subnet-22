//! Ready lists, the outcome feed and the dispatcher, on throwaway stores and a fake task API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use desearch_bot::allowed::{self, Allowed, Terms};
use desearch_bot::buckets::{bucket_of, BucketStore, Buckets, Changes, Resources};
use desearch_bot::dispatch::{
    allot, batch_id, interleave, recent_share, settle, shares, slots, split, yield_share, Dispatcher, HourlyCap, Pace, Progress, StockWanted, Want, FILL_FLOOR,
    PROBE_PER_HOUR, YIELD_JUDGED, YIELD_RECENT_JUDGED,
};
use desearch_bot::hotkey::{self, Hotkey};
use desearch_bot::outcomes::{self, HourlySpread, Next, OutcomeFeed, OutcomeReader, OutcomeRow, ResentPages};
use desearch_bot::ready::{Order, Outcome, Pick, Reason, Requeued, Scheduled, SentPage, Why, Yield, CHANGED, FRESH, LANES, REQUEUED, YIELD_HALF_LIFE_S};
use desearch_bot::recrawl::{Recrawl, DAY};
use desearch_bot::records;
use desearch_bot::states::State;
use desearch_bot::taskapi::{QueuedUrl, TaskApi};
use desearch_bot::timetable::UNRANKED;
use desearch_bot::urls::{self, Record, Url, HTTPS};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const RECRAWL: u32 = 7 * 86_400;
/// 2020-01-01, with a page from before it and one after.
const FLOOR: u32 = 1_577_836_800;
const EARLY: u32 = 1_559_347_200;
const LATE: u32 = 1_609_459_200;
/// 2026-10-27, a `resend_before`.
const CUTOFF: u32 = 1_793_059_200;
const GONE: u32 = 30 * 86_400;

fn rules() -> Recrawl {
    Recrawl::new(RECRAWL, GONE)
}

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

/// Stores holding these domains' records, as the outcome reader finds domains.
fn known(dir: &Dir, hosts: &[&str]) -> Buckets {
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Buckets::open(&dir.0, &owned, &resources()).unwrap();
    for host in hosts {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(10), Some("com"), &[], State::Active, None, None));
        buckets.store(host).write(changes).unwrap();
    }
    buckets
}

fn outcome(url: &str, outcome: Outcome, at: u32) -> OutcomeRow {
    let host = url.split('/').nth(2).unwrap().trim_start_matches("www.").to_string();
    OutcomeRow { url: url.into(), host, outcome, at: i64::from(at) * 1_000_000 }
}

fn requeue(store: &BucketStore, now: u32) -> Requeued {
    store.requeue_due(now, 100, &rules()).unwrap()
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
fn pages_robots_txt_disallows_leave_the_ready_list_unsent() {
    let dir = Dir::new("ready-robots");
    let store = open(&dir);
    let blog = urls::parse("https://blog.example.com/members/x", "example.com").unwrap();
    let pages = vec![page("news/a", 10), page("members/b", 20), page("members/join", 30), (blog, 40, false)];
    store.record_listing("example.com", 7, pages, 100).unwrap();
    let mut record = records::new_domain(Some(1), Some("com"), &[], State::Active, None, None);
    record.insert("rules".into(), serde_json::json!({"allow": ["/members/join"], "disallow": ["/members/"]}));
    let mut changes = Changes::default();
    changes.domain("example.com", &record);
    store.write(changes).unwrap();

    let mut sent: Vec<String> = take(&store, 10).into_iter().map(|p| p.url).collect();
    sent.sort();
    assert_eq!(sent, ["https://blog.example.com/members/x", "https://example.com/members/join", "https://example.com/news/a"]);
    assert_eq!(store.ready_count("example.com").unwrap(), 3, "a disallowed page leaves its list; the picks wait to be marked sent");

    store.record_listing("example.com", 7, vec![page("news/c", 50)], 200).unwrap();
    record.remove("rules");
    record.insert("allows".into(), false.into());
    let mut changes = Changes::default();
    changes.domain("example.com", &record);
    store.write(changes).unwrap();
    let sent: Vec<String> = take(&store, 10).into_iter().map(|p| p.url).collect();
    assert_eq!(sent, ["https://blog.example.com/members/x"], "a domain blocked before its rules were kept sends only other hosts' pages");
}

#[test]
fn a_sent_page_returns_once_its_lastmod_moves_past_what_it_was_sent_with() {
    let dir = Dir::new("ready-changed");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20)], 100).unwrap();
    let picks = take(&store, 10);
    assert_eq!(picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/b", "https://example.com/a"]);
    assert!(picks.iter().all(|p| p.reason == Reason::Fresh));
    store.mark_sent(&picks, 150, &rules()).unwrap();
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
    store.mark_sent(&picks, 400, &rules()).unwrap();
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
    store.mark_sent(&picks, 150, &rules()).unwrap();

    let listing = store.record_listing("example.com", 7, listed(&|i| 5000 + i), 200).unwrap();
    assert_eq!((listing.moved, listing.ready), (200, 10));
    let lastmods: Vec<u32> = orders(&store, "example.com").iter().map(|o| if let Order::Changed { lastmod } = o { *lastmod } else { 0 }).collect();
    assert_eq!(lastmods, (5190..5200).rev().collect::<Vec<_>>());

    let listing = store.record_listing("example.com", 7, listed(&|i| if i < 3 { 9000 + i } else { 5000 + i }), 300).unwrap();
    assert_eq!((listing.moved, listing.ready), (3, 3), "a few real changes all go");
}

#[test]
fn a_page_without_a_lastmod_goes_again_on_its_domains_interval_counted_from_its_send() {
    let dir = Dir::new("ready-recrawl");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("dated", 10), page("undated", 0)], 100).unwrap();
    let picks = take(&store, 10);
    store.mark_sent(&picks, 1000, &rules()).unwrap();
    assert_eq!(store.waiting().unwrap(), [(1000 + RECRAWL, "example.com".to_string(), rest("undated"))], "a page with a lastmod waits for it to move");
    assert_eq!(store.next_send("example.com", &rest("undated")).unwrap(), Some(Scheduled { why: Why::Recrawl, due: 1000 + RECRAWL, at: 1000 }));

    let undated: Vec<Pick> = picks.into_iter().filter(|p| p.url.ends_with("undated")).collect();
    store.mark_sent(&undated, 1500, &rules()).unwrap();
    assert_eq!(requeue(&store, 1000 + RECRAWL), Requeued::default(), "a send replaces the schedule an earlier send set");
    assert_eq!(requeue(&store, 1500 + RECRAWL - 1), Requeued::default());
    assert_eq!(requeue(&store, 1500 + RECRAWL), Requeued { moved: 1, gone: 0 });
    assert_eq!(store.next_send("example.com", &rest("undated")).unwrap(), None, "a schedule that came due is spent");
    let again = take(&store, 10);
    assert_eq!(again.iter().map(|p| (p.url.as_str(), p.reason)).collect::<Vec<_>>(), [("https://example.com/undated", Reason::Requeued)]);

    let daily = rules().with_overrides("example.com 1\n").unwrap();
    store.mark_sent(&again, 2000 + RECRAWL, &daily).unwrap();
    assert_eq!(store.waiting().unwrap(), [(2000 + RECRAWL + DAY, "example.com".to_string(), rest("undated"))], "a domain listed for daily re-crawls");
}

#[test]
fn recrawl_overrides_cover_subdomains_and_refuse_lines_they_cannot_read() {
    let recrawl = Recrawl::new(7 * DAY, 30 * DAY).with_overrides("# news, daily\nbbc.co.uk 1\nwww.Wikipedia.org 0.5  # twice a day\n\n").unwrap();
    assert_eq!(recrawl.overridden(), 2);
    assert_eq!(recrawl.interval("bbc.co.uk"), DAY);
    assert_eq!(recrawl.interval("news.bbc.co.uk"), DAY);
    assert_eq!(recrawl.interval("en.wikipedia.org"), DAY / 2);
    assert_eq!(recrawl.interval("notbbc.co.uk"), 7 * DAY, "a parent counts only at a dot");
    assert_eq!(recrawl.interval("example.com"), 7 * DAY);
    for bad in ["bbc.co.uk", "bbc.co.uk soon", "bbc.co.uk 0", "bbc.co.uk 1 2"] {
        assert!(Recrawl::default().with_overrides(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_failed_page_waits_its_interval_and_after_two_failures_in_a_row_the_long_wait() {
    let dir = Dir::new("ready-failed");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("a", 10)], 100).unwrap();
    store.mark_sent(&take(store, 10), 1000, &rules()).unwrap();
    let mut reader = OutcomeReader::new(rules(), 1000);

    reader.apply(&buckets, &[outcome("https://example.com/a", Outcome::Failed, 1100)], 1100).unwrap();
    assert_eq!(store.next_send("example.com", &rest("a")).unwrap(), Some(Scheduled { why: Why::Failed, due: 1100 + RECRAWL, at: 1100 }));
    assert_eq!(requeue(store, 1100 + RECRAWL - 1), Requeued::default(), "a failure waits for the normal re-crawl");
    assert_eq!(requeue(store, 1100 + RECRAWL), Requeued { moved: 1, gone: 0 });
    let retry = take(store, 10);
    assert_eq!((retry[0].reason, retry[0].failures), (Reason::Requeued, 1));

    let sent = 1200 + RECRAWL;
    store.mark_sent(&retry, sent, &rules()).unwrap();
    reader.apply(&buckets, &[outcome("https://example.com/a", Outcome::Failed, sent + 100)], sent + 100).unwrap();
    assert_eq!(requeue(store, sent + 100 + RECRAWL), Requeued { moved: 0, gone: 1 }, "the second failure in a row");
    assert_eq!(store.next_send("example.com", &rest("a")).unwrap(), Some(Scheduled { why: Why::Gone, due: sent + 100 + GONE, at: sent }));
    assert!(take(store, 10).is_empty());
    assert_eq!(requeue(store, sent + 100 + GONE - 1), Requeued::default());
    assert_eq!(requeue(store, sent + 100 + GONE), Requeued { moved: 1, gone: 0 }, "after the long wait it goes once more");
    let last = take(store, 10);
    assert_eq!((last[0].reason, last[0].failures), (Reason::Requeued, 2));
    let weekly = Recrawl::new(RECRAWL, RECRAWL);
    store.mark_sent(&last, sent + 200 + GONE, &weekly).unwrap();
    let failed_at = sent + 300 + GONE;
    OutcomeReader::new(weekly.clone(), 1000).apply(&buckets, &[outcome("https://example.com/a", Outcome::Failed, failed_at)], failed_at).unwrap();
    assert_eq!(store.requeue_due(failed_at + RECRAWL, 100, &weekly).unwrap(), Requeued { moved: 0, gone: 1 });
    assert_eq!(store.waiting().unwrap(), [(failed_at + RECRAWL, "example.com".to_string(), rest("a"))], "a long wait as long as the re-crawl lands on the entry it replaces");
    assert_eq!(store.requeue_due(failed_at + RECRAWL, 100, &weekly).unwrap(), Requeued { moved: 1, gone: 0 });
}

#[test]
fn a_page_set_to_the_long_wait_goes_earlier_when_its_lastmod_moves() {
    let dir = Dir::new("ready-gone-changed");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("a", 10)], 100).unwrap();
    let mut reader = OutcomeReader::new(rules(), 1000);
    let mut at = 1000;
    for _ in 0..2 {
        store.mark_sent(&take(store, 10), at, &rules()).unwrap();
        reader.apply(&buckets, &[outcome("https://example.com/a", Outcome::Failed, at + 100)], at + 100).unwrap();
        requeue(store, at + 100 + RECRAWL);
        at += 200 + RECRAWL;
    }
    assert_eq!(store.next_send("example.com", &rest("a")).unwrap().map(|n| n.why), Some(Why::Gone));

    store.record_listing("example.com", 7, vec![page("a", 20)], at).unwrap();
    let changed = take(store, 10);
    assert_eq!((changed.len(), changed[0].reason), (1, Reason::Changed));
    store.mark_sent(&changed, at + 10, &rules()).unwrap();
    assert_eq!(store.sent("example.com", &rest("a")).unwrap().unwrap().failures, 0, "a new lastmod starts the failures over");
    assert_eq!(requeue(store, at + GONE), Requeued::default(), "the long wait of the earlier send is spent");
    assert_eq!(store.next_send("example.com", &rest("a")).unwrap(), None);
}

#[test]
fn a_failure_replaces_the_recrawl_of_a_page_without_a_lastmod_and_a_clean_recrawl_clears_its_failures() {
    let dir = Dir::new("ready-undated-failed");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("undated", 0)], 100).unwrap();
    store.mark_sent(&take(store, 10), 1000, &rules()).unwrap();
    let mut reader = OutcomeReader::new(rules(), 1000);
    reader.apply(&buckets, &[outcome("https://example.com/undated", Outcome::Failed, 1100)], 1100).unwrap();
    assert_eq!(requeue(store, 1000 + RECRAWL), Requeued::default(), "the re-crawl the send set is replaced");
    assert_eq!(requeue(store, 1100 + RECRAWL), Requeued { moved: 1, gone: 0 });
    let retry = take(store, 10);
    assert_eq!(retry[0].failures, 1);

    let sent = 1200 + RECRAWL;
    store.mark_sent(&retry, sent, &rules()).unwrap();
    reader.apply(&buckets, &[outcome("https://example.com/undated", Outcome::Published, sent + 100)], sent + 100).unwrap();
    assert_eq!(requeue(store, sent + RECRAWL), Requeued { moved: 1, gone: 0 }, "a published page without a lastmod is re-crawled on its interval");
    assert_eq!(store.sent("example.com", &rest("undated")).unwrap().unwrap().failures, 0);
}

#[test]
fn an_outcome_of_an_earlier_send_brings_nothing_back_once_the_page_went_again() {
    let dir = Dir::new("ready-stale-outcome");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("a", 10)], 100).unwrap();
    store.mark_sent(&take(store, 10), 1000, &rules()).unwrap();
    let mut reader = OutcomeReader::new(rules(), 1000);
    reader.apply(&buckets, &[outcome("https://example.com/a", Outcome::Failed, 1100)], 1100).unwrap();
    store.record_listing("example.com", 7, vec![page("a", 20)], 1500).unwrap();
    store.mark_sent(&take(store, 10), 2000, &rules()).unwrap();
    assert_eq!(requeue(store, 1100 + RECRAWL), Requeued::default());
    assert!(take(store, 10).is_empty());
    assert_eq!(store.next_send("example.com", &rest("a")).unwrap(), None);
}

#[test]
fn dropped_pages_go_again_soon_and_a_burst_is_spread_over_the_following_hours() {
    let hour = 3600 * 100;
    let mut spread = HourlySpread::new(2);
    assert_eq!([hour + 5, hour + 6, hour + 7, hour + 8, hour + 9].map(|now| spread.due(now)), [hour + 5, hour + 6, hour + 3600, hour + 3600, hour + 7200]);
    assert_eq!(spread.due(hour + 3 * 3600 + 1), hour + 3 * 3600 + 1, "an hour with room takes pages at once again");
    assert_eq!(HourlySpread::new(0).due(hour), u32::MAX, "no budget leaves them to their re-crawl");

    let dir = Dir::new("ready-dropped");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 11), page("c", 12)], 100).unwrap();
    store.mark_sent(&take(store, 10), hour, &rules()).unwrap();
    let mut reader = OutcomeReader::new(rules(), 2);
    let rows = ["a", "b", "c"].map(|path| outcome(&format!("https://example.com/{path}"), Outcome::Dropped, hour + 50));
    let counted = reader.apply(&buckets, &rows, hour + 60).unwrap();
    assert_eq!((counted.dropped, counted.dropped_now), (3, 2));
    assert_eq!(store.domain_yield("example.com").unwrap(), None, "a dropped task says nothing about the domain");
    assert_eq!(requeue(store, hour + 60), Requeued { moved: 2, gone: 0 });
    assert_eq!(requeue(store, hour + 3599), Requeued::default());
    assert_eq!(requeue(store, hour + 3600), Requeued { moved: 1, gone: 0 });
    let again = take(store, 10);
    assert_eq!(again.len(), 3);
    assert!(again.iter().all(|p| p.reason == Reason::Requeued && p.failures == 0), "a dropped page is no failure");
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
    let want = |rank: i64, ready: u64, allowance: u64| Want { rank, ready, allowance, share: 1.0, weight: 1 };
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
fn a_domain_whose_pages_do_not_come_back_gets_fewer_slots_and_then_only_probes() {
    let counted = |ok: f64, failed: f64| Some(Yield { ok, failed, at: 1000 });
    assert_eq!(yield_share(None, 1000), 1.0, "nothing judged yet");
    assert_eq!(yield_share(counted(0.0, YIELD_JUDGED - 1.0), 1000), 1.0, "too few to judge");
    assert_eq!(yield_share(counted(5.0, 995.0), 1000), 0.0, "half a percent: probes only");
    assert_eq!(yield_share(counted(100.0, 900.0), 1000), 0.5, "a tenth comes back: half the slots");
    assert_eq!(yield_share(counted(300.0, 700.0), 1000), 1.0);
    let three_days = 1000 + (3.0 * YIELD_HALF_LIFE_S) as u32;
    assert_eq!(yield_share(counted(5.0, 995.0), three_days), 1.0, "old outcomes fade until the domain is judged afresh");

    let want = |share: f64, allowance: u64| Want { rank: 1, ready: 100, allowance, share, weight: 1 };
    assert_eq!(allot(&[want(1.0, 100), want(0.5, 100)], 22, 0).0, [14, 8], "seven slots a turn against half of seven, rounded up");
    assert_eq!(allot(&[want(0.0, PROBE_PER_HOUR), want(1.0, 100)], 100, 0).0, [13, 87], "a probing domain takes one a turn");
    assert_eq!(allot(&[want(0.0, PROBE_PER_HOUR), want(1.0, 100)], 500, 0).0, [PROBE_PER_HOUR, 100], "and never more than its probes");
}

#[test]
fn outcomes_tally_each_domains_yield_once_per_file_under_the_domain_its_host_was_sent_under() {
    let dir = Dir::new("ready-yield");
    let buckets = known(&dir, &["example.com", "example.org", "shop.example.org"]);
    buckets.allow(&Allowed::of(["example.com", "example.org"]));
    let mut reader = OutcomeReader::new(rules(), 1000);
    let rows = [
        outcome("https://example.com/a", Outcome::Published, 2000),
        outcome("https://www.example.com/b", Outcome::Failed, 2001),
        outcome("http://blog.example.com/c", Outcome::Unchanged, 2002),
        outcome("https://example.com/d", Outcome::Dropped, 2003),
        outcome("https://shop.example.org/e", Outcome::Failed, 2004),
        outcome("https://unknown.net/f", Outcome::Published, 2005),
    ];
    let counted = reader.apply(&buckets, &rows, 2010).unwrap();
    assert_eq!((counted.published, counted.failed, counted.dropped, counted.unknown), (3, 2, 1, 1));
    assert_eq!(buckets.store("example.com").domain_yield("example.com").unwrap(), Some(Yield { ok: 2.0, failed: 1.0, at: 2003 }));
    assert_eq!(reader.domain_of(&buckets, "blog.example.com").as_deref(), Some("example.com"), "a host that is no domain of its own was listed under its parent");
    assert_eq!(reader.domain_of(&buckets, "shop.example.org").as_deref(), Some("example.org"), "a domain the list does not queue sent nothing");
    assert_eq!(buckets.store("example.org").domain_yield("example.org").unwrap(), Some(Yield { ok: 0.0, failed: 1.0, at: 2004 }));

    reader.apply(&buckets, &[outcome("https://example.com/a", Outcome::Published, 2003 + YIELD_HALF_LIFE_S as u32)], 0).unwrap();
    let tallied = buckets.store("example.com").domain_yield("example.com").unwrap().unwrap();
    assert_eq!((tallied.ok, tallied.failed), (2.0, 0.5), "earlier outcomes count half a day later");
}

#[test]
fn the_hourly_cap_spreads_a_domains_hour_and_forgets_sends_older_than_an_hour() {
    let mut cap = HourlyCap::new(36_000);
    assert_eq!(cap.allowance("a.com", 1000), 1200, "two minutes' worth goes at once, not the hour");
    cap.record("a.com", 1000, 1200);
    assert_eq!(cap.allowance("a.com", 1119), 0);
    assert_eq!(cap.allowance("a.com", 1120), 1200, "then the next two minutes' worth");
    cap.record("a.com", 1120, 34_000);
    assert_eq!(cap.allowance("a.com", 1240), 800, "never more than the hour leaves");
    assert_eq!(cap.allowance("a.com", 4600), 1200, "sends older than an hour are forgotten");
    assert_eq!(cap.allowance("b.com", 1000), 1200);
    assert_eq!(HourlyCap::new(20).allowance("c.com", 1000), 20, "a small hour goes whole");
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
fn outcome_files_parse_with_the_host_each_url_was_queued_under() {
    let data = std::fs::read(format!("{}/tests/data/outcomes.parquet", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let rows = outcomes::read_parquet(data.into()).unwrap();
    assert_eq!(rows.len(), 4, "an outcome this bot does not know is skipped");
    assert_eq!(rows[0], OutcomeRow { url: "https://example.com/news/a".into(), host: "example.com".into(), outcome: Outcome::Published, at: 1_759_400_000_500_000 });
    assert_eq!(rows.iter().map(|r| r.outcome).collect::<Vec<_>>(), [Outcome::Published, Outcome::Unchanged, Outcome::Failed, Outcome::Dropped]);
    assert_eq!(rows.iter().map(|r| r.host.as_str()).collect::<Vec<_>>(), ["example.com", "example.com", "blog.example.com", "example.com"]);

    let dir = Dir::new("outcome-files");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    let listed = ["https://example.com/news/a", "https://www.example.com/news/b", "http://blog.example.com/c", "https://example.com/d"];
    store.record_listing("example.com", 7, listed.iter().map(|u| (urls::parse(u, "example.com").unwrap(), 10, false)).collect(), 100).unwrap();
    store.mark_sent(&take(store, 10), 1_759_399_000, &rules()).unwrap();
    let counted = OutcomeReader::new(rules(), 1000).apply(&buckets, &rows, 1_759_400_100).unwrap();
    assert_eq!((counted.published, counted.failed, counted.dropped, counted.dropped_now, counted.unknown), (2, 1, 1, 1, 0));
    assert_eq!(store.next_send("example.com", b"blog.example.com/c").unwrap().map(|n| (n.why, n.due)), Some((Why::Failed, 1_759_400_002 + RECRAWL)));
    assert_eq!(store.next_send("example.com", b"example.com/d").unwrap().map(|n| (n.why, n.due)), Some((Why::Dropped, 1_759_400_100)));
    assert_eq!(store.url(&urls::parse(listed[0], "example.com").unwrap()).unwrap().unwrap().crawled_at, 0, "a published page is not looked up");
}

#[derive(Default)]
struct FakeApi {
    room: i64,
    fail_next: usize,
    /// Batches with more URLs are refused as too large, and any batch holding `refused` as invalid.
    max_urls: usize,
    refused: Option<String>,
    rooms: usize,
    enqueued: Vec<(HashMap<String, String>, Vec<u8>)>,
    files: HashMap<String, Vec<u8>>,
    /// What a file not there is answered with: 404, or 403 as a store with a private bucket answers.
    forbid_missing: bool,
    /// Paths of the files asked for, in order.
    asked: Vec<String>,
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
                        (200, format!(r#"{{"room_tasks":{},"queue":0,"unrevealed":0,"refusing":false}}"#, api.room).into_bytes())
                    }
                    "/v1/admin/enqueue" => {
                        let sent: Value = serde_json::from_slice(&body).unwrap_or_default();
                        let urls: Vec<&str> = sent["urls"].as_array().into_iter().flatten().filter_map(|u| u["url"].as_str()).collect();
                        if api.max_urls > 0 && urls.len() > api.max_urls {
                            (413, br#"{"detail":"too large"}"#.to_vec())
                        } else if api.refused.as_deref().is_some_and(|bad| urls.contains(&bad)) {
                            (422, br#"{"detail":"not a URL"}"#.to_vec())
                        } else {
                            api.enqueued.push((headers, body));
                            if api.fail_next > 0 {
                                api.fail_next -= 1;
                                (500, br#"{"detail":"try again"}"#.to_vec())
                            } else {
                                (200, br#"{"round_id":"r1","batches":1}"#.to_vec())
                            }
                        }
                    }
                    _ => {
                        api.asked.push(path.to_string());
                        match api.files.get(path) {
                            Some(file) => (200, file.clone()),
                            None if api.forbid_missing => (403, b"<Error><Code>AccessDenied</Code></Error>".to_vec()),
                            None => (404, br#"{"detail":"no such route"}"#.to_vec()),
                        }
                    }
                }
            };
            let head = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reply.len());
            let _ = socket.write_all(&[head.into_bytes(), reply].concat()).await;
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
    let dispatcher = |progress: Arc<Progress>| Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 20, shares(50, 10), rules(), progress);

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
    store.mark_sent(&sent, 150, &rules()).unwrap();
    store.record_listing("example.com", 7, (0..20).map(|i| page(&format!("old{i:02}"), 50)).chain((0..20).map(|i| page(&format!("new{i:02}"), 60))).collect(), 200).unwrap();

    let mut quotas = [0; LANES];
    quotas[FRESH] = 5;
    quotas[CHANGED] = 4;
    quotas[REQUEUED] = 1;
    let picks = store.take_ready("example.com", quotas).unwrap();
    let count = |reason| picks.iter().filter(|p| p.reason == reason).count();
    assert_eq!((count(Reason::Fresh), count(Reason::Changed)), (6, 4), "both lanes go at once; the empty retry lane's page goes to new pages");

    store.mark_sent(&picks, 250, &rules()).unwrap();
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
fn a_sent_page_with_no_outcome_is_not_sent_again_a_day_later() {
    let dir = Dir::new("ready-no-outcome");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 20)], 100).unwrap();
    store.mark_sent(&take(&store, 10), 1000, &rules()).unwrap();
    assert!(store.waiting().unwrap().is_empty());
    assert_eq!(requeue(&store, 1000 + 2 * DAY), Requeued::default());
    assert!(take(&store, 10).is_empty(), "a page with a lastmod goes again when it moves, not when an outcome is late");
    assert!(!store.forget_expecting().unwrap(), "no deadline is kept for a send");
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

    fake.lock().unwrap().forbid_missing = true;
    assert!(matches!(feed.fetch(5).await.unwrap(), Next::Missing), "a 403 for a key the store does not have is a missing number too");
    assert!(matches!(feed.fetch(8).await.unwrap(), Next::Wait));
    fake.lock().unwrap().files.insert("/outcomes/seq/000000000006.json".into(), br#"{"key": "outcomes/dt=2026-10-22/a.parquet", "rows": 3}"#.to_vec());
    assert!(matches!(feed.fetch(6).await.unwrap(), Next::Gone), "an index whose file expired is skipped, not waited on");
}


#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_outcome_feed_is_read_from_its_newest_file_and_skips_ahead_when_an_hour_behind() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi::default()));
    tokio::spawn(serve_api(listener, fake.clone()));
    // Its rows are from 2025, so every look at it is far more than an hour behind.
    let file = std::fs::read(format!("{}/tests/data/outcomes.parquet", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let publish = |api: &mut FakeApi, seqs: std::ops::RangeInclusive<u64>| {
        for seq in seqs.clone() {
            api.files.insert(format!("/outcomes/seq/{seq:012}.json"), format!(r#"{{"key": "outcomes/f{seq}.parquet", "rows": 4}}"#).into_bytes());
            api.files.insert(format!("/outcomes/f{seq}.parquet"), file.clone());
        }
        api.files.insert("/outcomes/latest.json".into(), format!(r#"{{"seq": {}}}"#, seqs.end()).into_bytes());
    };
    publish(&mut fake.lock().unwrap(), 1..=5);

    let dir = Dir::new("outcome-follow");
    let buckets = Arc::new(known(&dir, &["example.com"]));
    let progress = Arc::new(Progress::default());
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let feed = OutcomeFeed::new(&base).unwrap();
    let following = tokio::spawn(desearch_bot::dispatch::follow_outcomes(buckets.clone(), feed, OutcomeReader::new(rules(), 1000), progress.clone(), stopped));
    let seq = || progress.outcome_seq.load(std::sync::atomic::Ordering::Relaxed);
    let wait_for = |want: u64| async move {
        for _ in 0..200 {
            if seq() == want {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("the feed reached {} and not {want}", seq());
    };
    wait_for(6).await;
    publish(&mut fake.lock().unwrap(), 6..=40);
    wait_for(41).await;
    stop.send(true).unwrap();
    following.await.unwrap();

    let read: Vec<u64> = fake.lock().unwrap().asked.iter().filter_map(|p| p.strip_prefix("/outcomes/f")?.strip_suffix(".parquet")?.parse().ok()).collect();
    assert_eq!(read, [5, 6, 7, 8, 9, 10, 11, 12, 13, 40], "the newest file first, one run of eight, then a jump to the newest");
    assert_eq!(progress.skipped.load(std::sync::atomic::Ordering::Relaxed), 26);
    assert_eq!(progress.outcomes.load(std::sync::atomic::Ordering::Relaxed), 40);
    assert_eq!(buckets.store("example.com").domain_yield("example.com").unwrap().map(|y| y.judged()), Some(30.0), "three judged rows a file, ten files");
}

/// A list file the test rewrites, read again on demand.
fn relist(path: &std::path::Path, list: &Allowed, json: &str) {
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(path, json).unwrap();
    assert!(list.reload().unwrap());
}

#[test]
fn a_domain_joining_the_list_queues_its_stored_pages_never_sent_newest_first_up_to_its_cap() {
    let dir = Dir::new("join");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    for (i, name) in ["a", "b", "c", "d", "e", "tag/x"].iter().enumerate() {
        store.record_listing("example.com", 7, vec![page(name, 10)], 100 + 10 * i as u32).unwrap();
    }
    store.import_sent(&[SentPage { domain: "example.com".into(), rest: rest("a"), lastmod: 10, at: 150 }]).unwrap();
    assert!(ready(&store).is_empty(), "pages of a domain not on the list wait in the store");

    relist(&path, &list, r#"[{"host": "example.com", "weight": 4, "send_cap": 3}]"#);
    let terms = list.get("example.com").unwrap();
    let joined = store.join_domain("example.com", terms, 1000, &mut |_| true).unwrap();
    assert_eq!((joined.scanned, joined.sent, joined.queued), (6, 1, 2), "the cap of 3 leaves room for 2 beside the page already sent");
    assert_eq!(ready(&store), ["e", "d"], "the newest found go, listing pages never");
    assert_eq!(store.ready_count("example.com").unwrap(), 2);
    assert_eq!((store.sent_count("example.com").unwrap(), store.joined("example.com").unwrap()), (Some(1), Some(Some(3))));

    store.record_listing("example.com", 7, vec![page("f", 10)], 2000).unwrap();
    assert_eq!(ready(&store), ["f", "e", "d"], "a page new to a domain still under its cap joins its list");
    let picks = take(&store, 10);
    assert_eq!(picks.len(), 2, "but only as many go as the cap leaves");
    store.mark_sent(&picks, 2100, &rules()).unwrap();
    assert_eq!(store.sent_count("example.com").unwrap(), Some(3));
    assert_eq!(store.take_capped(), ["example.com".to_string()]);
    assert_eq!(store.clear_fresh("example.com", &mut |_| true).unwrap(), (1, true));
    assert!(ready(&store).is_empty());
    assert_eq!(store.ready_count("example.com").unwrap(), 0, "the count follows the entries cleared");
    store.record_listing("example.com", 7, vec![page("g", 10)], 2200).unwrap();
    assert!(ready(&store).is_empty(), "a domain at its cap queues no new pages");
    store.record_listing("example.com", 7, vec![page("e", 20)], 2300).unwrap();
    assert_eq!(orders(&store, "example.com"), [Order::Changed { lastmod: 20 }], "while a sent page that changed still goes");
    let spilled = take(&store, 10);
    assert_eq!(spilled.iter().map(|p| p.reason).collect::<Vec<_>>(), [Reason::Changed], "the new-page lane gives nothing at the cap, and its share goes to the changed page");

    relist(&path, &list, r#"[{"host": "example.com", "weight": 4}]"#);
    let again = store.join_domain("example.com", list.get("example.com").unwrap(), 3000, &mut |_| true).unwrap();
    assert_eq!((again.sent, again.queued, again.cleared), (3, 4, 0), "without a cap every page never sent goes, b, c, f and g");
    assert_eq!(store.joined("example.com").unwrap(), Some(None));
}

#[test]
fn the_list_walker_walks_each_listed_domain_once_and_again_when_its_cap_changes() {
    let dir = Dir::new("join-round");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"[{"host": "example.com", "send_cap": 10}, {"host": "example.org", "weight": 3}]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com", "example.org"]);
    buckets.allow(&list);
    let progress = Progress::default();
    let stop = std::sync::atomic::AtomicBool::new(false);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 2);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 0, "a domain walked under its cap is not walked again");
    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 20}, {"host": "example.org", "weight": 3}, {"host": "example.net"}]"#);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1, "a new cap walks it again; a domain no store here holds is passed over");
    assert_eq!(buckets.terms("example.org"), Terms { weight: 3, ..Terms::default() });
}

#[test]
fn a_domain_already_sending_is_only_counted_and_cleared_when_its_cap_falls_below_its_sends() {
    let dir = Dir::new("join-capped");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"[{"host": "example.com", "send_cap": 10}]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, (0..8).map(|i| page(&format!("p{i}"), 10)).collect(), 100).unwrap();
    store.mark_sent(&take(store, 3), 200, &rules()).unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &Progress::default(), &stop).unwrap(), 1);
    assert_eq!((store.ready_count("example.com").unwrap(), store.joined("example.com").unwrap()), (5, Some(Some(10))));

    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 2}]"#);
    let progress = Progress::default();
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert_eq!(store.sent_count("example.com").unwrap(), Some(3), "its sent pages are counted from their records");
    assert_eq!(store.ready_count("example.com").unwrap(), 0, "past its cap, its waiting new pages are cleared");
    assert_eq!(progress.capped_cleared.load(std::sync::atomic::Ordering::Relaxed), 5);
}

#[test]
fn a_domain_without_a_cap_queues_at_most_its_backlog_newest_first_and_clears_nothing() {
    let dir = Dir::new("join-backlog");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    for (i, name) in ["a", "b", "c", "d"].iter().enumerate() {
        store.record_listing("example.com", 7, vec![page(name, 10)], 100 + 10 * i as u32).unwrap();
    }

    relist(&path, &list, r#"[{"host": "example.com", "type": "news", "backlog": 2, "hourly": 5000}]"#);
    let joined = store.join_domain("example.com", list.get("example.com").unwrap(), 1000, &mut |_| true).unwrap();
    assert_eq!((joined.queued, joined.cleared), (2, 0), "the two newest go, and nothing is cleared without a cap");
    assert_eq!(ready(&store), ["d", "c"]);
    assert_eq!(store.joined("example.com").unwrap(), Some(None));
    store.record_listing("example.com", 7, vec![page("e", 10)], 2000).unwrap();
    assert_eq!(ready(&store), ["e", "d", "c"], "pages found later join as usual, with no bound");
}

#[test]
fn a_walk_asked_to_stop_returns_without_marking_the_domain_walked() {
    let dir = Dir::new("join-stop");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 10)], 100).unwrap();

    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 5}]"#);
    let joined = store.join_domain("example.com", list.get("example.com").unwrap(), 1000, &mut |_| false).unwrap();
    assert!(joined.stopped);
    assert_eq!(joined.queued, 0);
    assert_eq!(store.joined("example.com").unwrap(), None, "the next round walks it again");
}

#[test]
fn a_clear_asked_to_stop_leaves_the_domain_for_the_next_round_to_clear() {
    let dir = Dir::new("join-clear-stop");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["example.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    store.record_listing("example.com", 7, (0..50_002).map(|i| page(&format!("p{i}"), 10)).collect(), 100).unwrap();
    store.mark_sent(&take(&store, 1), 200, &rules()).unwrap();

    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 1}]"#);
    let terms = list.get("example.com").unwrap();
    let mut paced = 0;
    let joined = store.cap_domain("example.com", terms, &mut |_| {
        paced += 1;
        paced < 2
    }).unwrap();
    assert!(joined.stopped);
    assert_eq!((joined.cleared, store.ready_count("example.com").unwrap()), (50_000, 1), "one chunk went before the stop");
    assert_eq!(store.joined("example.com").unwrap(), None, "the next round takes it again");

    let again = store.cap_domain("example.com", terms, &mut |_| true).unwrap();
    assert_eq!((again.stopped, again.cleared, store.ready_count("example.com").unwrap()), (false, 1, 0));
    assert_eq!(store.joined("example.com").unwrap(), Some(Some(1)));
}

#[test]
fn a_refresh_stock_holds_the_pages_sent_longest_ago_and_hands_them_out_oldest_first() {
    let dir = Dir::new("refresh-stock");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 10), page("c", 10), page("d", 10)], 100).unwrap();
    let picks = take(&store, 10);
    for (i, path) in ["c", "a", "d", "b"].iter().enumerate() {
        let pick: Vec<Pick> = picks.iter().filter(|p| p.url.ends_with(&format!("/{path}"))).cloned().collect();
        store.mark_sent(&pick, 1000 + i as u32, &rules()).unwrap();
    }
    store.record_listing("example.com", 7, vec![page("d", 20)], 1500).unwrap();
    assert_eq!(store.stock_refresh("example.com", 2, None, usize::MAX, &mut |_| true).unwrap().map(|s| s.count), Some(2), "d changed, so it goes as a change");
    let urls = |stock: &desearch_bot::ready::Stock| stock.picks.iter().map(|p| p.url.clone()).collect::<Vec<_>>();

    let young = store.take_stock("example.com", 10, 1500, 1000).unwrap();
    assert!(young.picks.is_empty());
    assert_eq!(young.young_until, Some(2000), "c went at 1000; nothing goes until a floor past it");
    let first = store.take_stock("example.com", 1, 2001, 1000).unwrap();
    assert_eq!(urls(&first), ["https://example.com/c"], "sent longest ago first");
    assert!(first.picks.iter().all(|p| p.reason == Reason::Refresh));
    assert!(!first.empty);

    store.mark_sent(&take(&store, 10), 2002, &rules()).unwrap();
    let rest = store.take_stock("example.com", 10, 4000, 1000).unwrap();
    assert_eq!(urls(&rest), ["https://example.com/a"], "the stock kept two, and c left it when taken");
    assert!(rest.empty, "the stock ran out");
    assert!(store.stock_refresh("example.com", 10, None, usize::MAX, &mut |_| false).unwrap().is_none(), "a choice asked to stop writes nothing");
}

#[test]
fn a_refresh_stock_passes_over_pages_dated_before_the_floor() {
    let dir = Dir::new("refresh-stock-floor");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("old", FLOOR - 86_400 * 30), page("new", FLOOR + 10), page("undated", 0)], 100).unwrap();
    store.mark_sent(&take(store, 10), 1000, &rules()).unwrap();
    buckets.allow(&Allowed::with_terms([("example.com", Terms { since: Some(FLOOR), ..Terms::default() })]));
    store.stock_refresh("example.com", 10, None, usize::MAX, &mut |_| true).unwrap();
    let stock = store.take_stock("example.com", 10, 3000, 1000).unwrap();
    let mut urls: Vec<String> = stock.picks.iter().map(|p| p.url.clone()).collect();
    urls.sort();
    assert_eq!(urls, ["https://example.com/new", "https://example.com/undated"], "a page dated before the floor isn't sent again");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spare_room_goes_to_the_oldest_sent_pages_of_domains_whose_pages_come_back() {
    let dir = Dir::new("dispatcher-fill");
    let buckets = Arc::new(known(&dir, &["example.com", "example.org"]));
    buckets.allow(&Allowed::with_terms([("example.com", Terms { hourly: Some(3600), ..Terms::default() }), ("example.org", Terms::default())]));
    let later = 1000 + FILL_FLOOR;
    for (host, ok, failed) in [("example.com", 1000.0, 10.0), ("example.org", 5.0, 1000.0)] {
        let store = buckets.store(host);
        let pages = (0..150).map(|i| (urls::parse(&format!("https://{host}/p{i:03}"), host).unwrap(), 100, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
        let mut quotas = [0; LANES];
        quotas[FRESH] = 1000;
        store.mark_sent(&store.take_ready(host, quotas).unwrap(), 1000, &rules()).unwrap();
        store.note_outcomes(&[], &[(host.to_string(), Yield { ok, failed, at: later })]).unwrap();
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let progress = Arc::new(Progress::default());
    let wanted = Arc::new(StockWanted::default());
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap().refreshing(wanted.clone());

    assert_eq!(dispatcher.pass(later).await.unwrap(), 0, "no stock chosen yet");
    assert_eq!(wanted.waiting(), 1, "example.com waits for its stock; example.org's pages rarely come back");
    assert_eq!(buckets.store("example.com").stock_refresh("example.com", 1000, None, usize::MAX, &mut |_| true).unwrap().map(|s| s.count), Some(150));
    assert_eq!(dispatcher.pass(later + 1).await.unwrap(), 0, "a domain found out of stock waits for it");
    let later = later + 600;
    assert_eq!(dispatcher.pass(later).await.unwrap(), 120, "two minutes of example.com's 3,600 an hour");
    let body: Value = serde_json::from_slice(&fake.lock().unwrap().enqueued[0].1).unwrap();
    assert!(body["urls"].as_array().unwrap().iter().all(|u| u["host"] == "example.com"));
    assert_eq!(progress.sent_fill.load(std::sync::atomic::Ordering::Relaxed), 120);
    assert_eq!(progress.fill_domains.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(dispatcher.pass(later + 120).await.unwrap(), 30, "the rest after the next two minutes");
}

#[test]
fn a_domain_that_starts_refusing_loses_its_slots_long_before_its_day_long_yield_notices() {
    let dir = Dir::new("ready-recent");
    std::fs::create_dir_all(&dir.0).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.note_outcomes(&[], &[("example.com".into(), Yield { ok: 10_000.0, failed: 0.0, at: 1000 })]).unwrap();
    let later = 1000 + 6 * 3600;
    store.note_outcomes(&[], &[("example.com".into(), Yield { ok: 20.0, failed: 20_000.0, at: later })]).unwrap();
    assert_eq!(yield_share(store.domain_yield("example.com").unwrap(), later), 1.0, "the day-long count still says almost a third come back");
    assert_eq!(recent_share(store.domain_yield_recent("example.com").unwrap(), later), 0.0, "the last hours say under 2%: probes only");
    assert_eq!(recent_share(None, later), 1.0);
    assert_eq!(recent_share(Some(Yield { ok: 0.0, failed: YIELD_RECENT_JUDGED - 1.0, at: later }), later), 1.0, "too few recent outcomes to judge");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_listed_hourly_limit_replaces_the_global_one_for_its_domain() {
    let dir = Dir::new("dispatcher-hourly");
    let mut owned = vec![bucket_of("example.com"), bucket_of("example.org")];
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &resources()).unwrap());
    buckets.allow(&Allowed::with_terms([("example.com", Terms { hourly: Some(5), ..Terms::default() }), ("example.org", Terms::default())]));
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
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 20, shares(50, 10), rules(), Arc::new(Progress::default())).await.unwrap();
    assert_eq!(dispatcher.pass(1_759_400_000).await.unwrap(), 25, "5 an hour for the domain the list limits, 20 for the other");
    assert_eq!(buckets.store("example.com").ready_count("example.com").unwrap(), 25);
}

#[test]
fn a_domains_weight_multiplies_the_slots_its_rank_earns() {
    let want = |weight: u8| Want { rank: 1, ready: 1000, allowance: 1000, share: 1.0, weight };
    assert_eq!(allot(&[want(4), want(1)], 70, 0).0, [56, 14], "news at weight 4 takes four turns' worth to the rest's one");
    let slowed = Want { share: 0.5, ..want(2) };
    assert_eq!(allot(&[slowed, want(1)], 14, 0).0, [7, 7], "the yield share applies on top of the weight");
}

#[test]
fn the_domain_list_takes_names_or_hosts_and_is_read_again_when_its_file_changes() {
    let names = allowed::parse(br#"["a.com", {"host": "b.org", "rank": 2}, 7, {"host": "c.net", "type": "news", "weight": 4, "send_cap": 1500}, {"host": "d.io", "weight": 9}]"#).unwrap();
    let mut hosts: Vec<&str> = names.keys().map(String::as_str).collect();
    hosts.sort_unstable();
    assert_eq!(hosts, ["a.com", "b.org", "c.net", "d.io"]);
    assert_eq!(names["a.com"], Terms::default());
    assert_eq!(names["c.net"], Terms { weight: 4, send_cap: Some(1500), news: true, ..Terms::default() }, "the crawl list's type, weight and cap");
    let news = allowed::parse(br#"[{"host": "e.com", "type": "news", "weight": 4, "hourly": 20000, "backlog": 3000000}]"#).unwrap();
    assert_eq!(news["e.com"], Terms { weight: 4, send_cap: None, hourly: Some(20_000), backlog: Some(3_000_000), since: None, news: true, resend_before: None }, "an hourly limit and a backlog bound");
    let floored = allowed::parse(br#"[{"host": "f.com", "type": "news", "since": "2020-01-01"}, {"host": "g.com", "since": "soon"}]"#).unwrap();
    assert_eq!((floored["f.com"].since, floored["g.com"].since), (Some(FLOOR), None), "a date floor as an ISO date; one that doesn't read is none");
    assert_eq!(names["d.io"].weight, 4, "weights stay within 1 to 4");
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
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 100, shares(50, 10), rules(), Arc::new(Progress::default())).await.unwrap();

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
    let dispatcher = |progress: Arc<Progress>| Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 1_000_000, shares(50, 10), rules(), progress);

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
fn the_pace_starts_empty_grows_steadily_and_saves_up_only_five_minutes() {
    let mut pace = Pace::new(3_600_000, 1000);
    assert_eq!(pace.allowance(1000), 0, "nothing saved at the start, so a restart sends no burst");
    assert_eq!(pace.allowance(1060), 60_000, "a minute later, a minute's worth");
    pace.spend(60_000);
    assert_eq!(pace.allowance(1000 + 7200), 300_000, "an idle hour saves no more than five minutes");
    assert_eq!(Pace::new(0, 1000).allowance(2000), u64::MAX, "no pace set sends whatever there is room for");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_batch_refused_as_too_large_or_invalid_goes_in_halves_and_only_the_page_at_fault_is_set_aside() {
    let dir = Dir::new("dispatcher-refused");
    let host = "example.net";
    let buckets = Arc::new(Buckets::open(&dir.0, &[bucket_of(host)], &resources()).unwrap());
    let store = buckets.store(host);
    let mut changes = Changes::default();
    changes.domain(host, &records::new_domain(Some(5), Some("net"), &[], State::Active, None, None));
    store.write(changes).unwrap();
    let pages = (0..40).map(|i| (urls::parse(&format!("https://{host}/p{i:02}"), host).unwrap(), 100 + i, false)).collect();
    store.record_listing(host, 1, pages, 100).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let bad = format!("https://{host}/p07");
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, max_urls: 16, refused: Some(bad.clone()), ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let progress = Arc::new(Progress::default());
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap();

    let now = 1_759_400_000;
    assert_eq!(dispatcher.pass(now).await.unwrap(), 39, "everything but the bad page went in the same pass");
    let taken: Vec<String> = fake
        .lock()
        .unwrap()
        .enqueued
        .iter()
        .flat_map(|(_, body)| {
            serde_json::from_slice::<Value>(body).unwrap()["urls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|u| u["url"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!((taken.len(), taken.contains(&bad)), (39, false));
    assert_eq!(progress.set_aside.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(buckets.store(host).ready_count(host).unwrap(), 0, "the bad page left the ready list too");
    assert_eq!(dispatcher.pass(now + 15).await.unwrap(), 0);
}

#[test]
fn dropping_a_domain_deletes_its_pages_and_leaves_its_neighbours_whole() {
    let dir = Dir::new("ready-drop");
    let owned = [bucket_of("junk.com")];
    let buckets = Buckets::open(&dir.0, &owned, &resources()).unwrap();
    let store = buckets.store("junk.com");
    for host in ["junk.com", "junk.com.au"] {
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(5), Some("com"), &[], State::Active, None, None));
        store.write(changes).unwrap();
        let pages = (0..20).map(|i| (urls::parse(&format!("https://{host}/p{i}"), host).unwrap(), 100 + i, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
    }
    let picks: Vec<Pick> = store.take_ready("junk.com", [0, 0, 5]).unwrap();
    store.mark_sent(&picks, 1000, &rules()).unwrap();
    let page = |host: &str, i: u32| store.url(&urls::parse(&format!("https://{host}/p{i}"), host).unwrap()).unwrap();

    let counted = store.drop_domain("junk.com", "junk", false).unwrap();
    assert_eq!((counted.known, counted.ready), (true, 15));
    assert!(page("junk.com", 0).is_some(), "counting deletes nothing");

    store.drop_domain("junk.com", "hacked", true).unwrap();
    assert!((0..20).all(|i| page("junk.com", i).is_none()));
    assert_eq!(store.ready_count("junk.com").unwrap(), 0);
    assert_eq!(store.ready_domains().unwrap(), [("junk.com.au".to_string(), 20)], "a domain that only starts with its name is untouched");
    assert!((0..20).all(|i| page("junk.com.au", i).is_some()));
    let record = store.domain("junk.com").unwrap().unwrap();
    assert_eq!((record["state"].as_str(), record["reason"].as_str(), record["due"].is_null()), (Some("excluded"), Some("hacked"), true));
    assert!(!store.drop_domain("unknown.com", "junk", true).unwrap().known);
}

/// Run with --ignored --nocapture: how long passes take when tens of thousands of domains have pages waiting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn passes_stay_short_with_tens_of_thousands_of_domains_waiting() {
    let domains: usize = std::env::var("DOMAINS").ok().and_then(|n| n.parse().ok()).unwrap_or(20_000);
    let dir = Dir::new("many-domains");
    let hosts: Vec<String> = (0..domains).map(|i| format!("site{i}.example")).collect();
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &Resources::new(256 << 20, 4 << 30, owned.len())).unwrap());
    let started = std::time::Instant::now();
    for (i, host) in hosts.iter().enumerate() {
        let store = buckets.store(host);
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1000 + i as i64), Some("example"), &[], State::Active, None, None));
        store.write(changes).unwrap();
        let pages = (0..30).map(|p| (urls::parse(&format!("https://{host}/page{p}"), host).unwrap(), 100 + p, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
    }
    println!("{domains} domains x 30 pages written in {:.1}s", started.elapsed().as_secs_f64());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 400, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let progress = Arc::new(Progress::default());
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap();
    let now = 1_759_400_000;
    for pass in 0..5 {
        let started = std::time::Instant::now();
        let sent = dispatcher.pass(now + 20 * pass).await.unwrap();
        let ms: Vec<u64> = progress.pass_ms.iter().map(|m| m.load(std::sync::atomic::Ordering::Relaxed)).collect();
        println!("pass {pass}: {sent} URLs in {:.2}s (absorb {} ms, pick {} ms, send {} ms)", started.elapsed().as_secs_f64(), ms[0], ms[1], ms[2]);
    }
}


#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a scale check, run by hand: DOMAINS=90000 cargo test --release --test dispatch fill_passes_stay_short -- --ignored --nocapture"]
async fn fill_passes_stay_short_with_tens_of_thousands_of_domains_to_refresh() {
    let domains: usize = std::env::var("DOMAINS").ok().and_then(|n| n.parse().ok()).unwrap_or(20_000);
    let dir = Dir::new("many-fillers");
    let hosts: Vec<String> = (0..domains).map(|i| format!("site{i}.example")).collect();
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &Resources::new(256 << 20, 4 << 30, owned.len())).unwrap());
    buckets.allow(&Allowed::with_terms(hosts.iter().map(|h| (h.clone(), Terms { hourly: Some(5000), ..Terms::default() }))));
    let later = 1000 + FILL_FLOOR;
    let started = std::time::Instant::now();
    for (i, host) in hosts.iter().enumerate() {
        let store = buckets.store(host);
        let mut changes = Changes::default();
        changes.domain(host, &records::new_domain(Some(1000 + i as i64), Some("example"), &[], State::Active, None, None));
        store.write(changes).unwrap();
        let pages = (0..10).map(|p| (urls::parse(&format!("https://{host}/page{p}"), host).unwrap(), 100, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
        let mut quotas = [0; LANES];
        quotas[FRESH] = 100;
        store.mark_sent(&store.take_ready(host, quotas).unwrap(), 1000, &rules()).unwrap();
        store.note_outcomes(&[], &[(host.clone(), Yield { ok: 1000.0, failed: 10.0, at: later })]).unwrap();
        store.stock_refresh(host, 1000, None, usize::MAX, &mut |_| true).unwrap();
    }
    println!("{domains} domains x 10 sent and stocked pages written in {:.1}s", started.elapsed().as_secs_f64());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 400, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let progress = Arc::new(Progress::default());
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let wanted = Arc::new(StockWanted::default());
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap().refreshing(wanted);
    for pass in 0..8 {
        let started = std::time::Instant::now();
        let sent = dispatcher.pass(later + 15 * pass).await.unwrap();
        let ms: Vec<u64> = progress.pass_ms.iter().map(|m| m.load(std::sync::atomic::Ordering::Relaxed)).collect();
        println!(
            "pass {pass}: {sent} URLs in {:.2}s (absorb {} ms, pick {} ms, send {} ms), {} fill domains",
            started.elapsed().as_secs_f64(),
            ms[0],
            ms[1],
            ms[2],
            progress.fill_domains.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
}

#[test]
#[ignore = "a rate check, run by hand: SENT=300000 cargo test --release --test dispatch stock_choice_rate -- --ignored --nocapture"]
fn stock_choice_rate_on_a_big_domain() {
    let sent: usize = std::env::var("SENT").ok().and_then(|n| n.parse().ok()).unwrap_or(100_000);
    let dir = Dir::new("stock-rate");
    let store = BucketStore::open(&dir.0.join("store"), &Resources::new(256 << 20, 1 << 30, 1)).unwrap();
    let started = std::time::Instant::now();
    for chunk in (0..sent).collect::<Vec<_>>().chunks(20_000) {
        store.record_listing("example.com", 7, chunk.iter().map(|i| page(&format!("article/{i:08}"), 10)).collect(), 100).unwrap();
        let mut quotas = [0; LANES];
        quotas[FRESH] = chunk.len();
        store.mark_sent(&store.take_ready("example.com", quotas).unwrap(), 1000 + chunk[0] as u32, &rules()).unwrap();
    }
    println!("{sent} sent pages written in {:.1}s", started.elapsed().as_secs_f64());
    for most in [usize::MAX, 200_000, 20_000] {
        let started = std::time::Instant::now();
        let stocked = store.stock_refresh("example.com", 1250, None, most, &mut |_| true).unwrap().unwrap();
        let took = started.elapsed().as_secs_f64();
        println!("most {most}: {} stocked in {took:.2}s, {:.0} sent records a second", stocked.count, sent.min(most) as f64 / took);
    }
}

#[test]
fn a_date_floor_leaves_older_pages_unsent_wherever_they_would_queue() {
    let dir = Dir::new("join-floor");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    let stored = [page("old", EARLY), page("new", LATE), page("2018/03/dated", 0), page("2021-02-04-dated", 0), page("undated", 0), page("floor-day-east", FLOOR - 4 * 3_600)];
    store.record_listing("example.com", 7, stored.to_vec(), 100).unwrap();

    relist(&path, &list, r#"[{"host": "example.com", "type": "news", "backlog": 100, "since": "2020-01-01"}]"#);
    let joined = store.join_domain("example.com", list.get("example.com").unwrap(), 1000, &mut |_| true).unwrap();
    assert_eq!((joined.queued, joined.before_floor), (4, 2), "a lastmod before the floor, or a month in the path before it, stays");
    let mut queued = ready(&store);
    queued.sort();
    assert_eq!(queued, ["2021-02-04-dated", "floor-day-east", "new", "undated"], "a lastmod hours before the floor in UTC may be the floor's day where it was written");
    assert_eq!(store.joined_since("example.com").unwrap(), Some(FLOOR));

    store.record_listing("example.com", 7, vec![page("found-old", EARLY), page("found-new", LATE)], 2000).unwrap();
    assert!(ready(&store).contains(&"found-new".to_string()) && !ready(&store).contains(&"found-old".to_string()), "a sitemap read queues no page dated before the floor");
}

#[test]
fn a_higher_floor_drops_older_pages_as_they_would_go_and_a_lower_floor_walks_the_domain_again() {
    let dir = Dir::new("join-floor-round");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"[{"host": "example.com", "type": "news", "backlog": 100}]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, vec![page("old", EARLY), page("new", LATE), page("undated", 0)], 100).unwrap();
    let progress = Progress::default();
    let stop = std::sync::atomic::AtomicBool::new(false);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert_eq!(store.ready_count("example.com").unwrap(), 3, "with no floor every page goes");

    relist(&path, &list, r#"[{"host": "example.com", "type": "news", "backlog": 100, "since": "2020-01-01"}]"#);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert_eq!(store.ready_count("example.com").unwrap(), 3, "a higher floor is only counted: the domain isn't walked or its lane cleared");
    let mut taken: Vec<String> = take(store, 10).iter().map(|p| p.url.rsplit('/').next().unwrap().to_string()).collect();
    taken.sort();
    assert_eq!(taken, ["new", "undated"], "the page dated before the floor drops as it would go");
    assert!(!ready(store).contains(&"old".to_string()));
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 0, "the same floor walks it no more");

    relist(&path, &list, r#"[{"host": "example.com", "type": "news", "backlog": 100}]"#);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert!(ready(store).contains(&"old".to_string()), "a floor taken off walks the domain again and brings the older page back");
    assert_eq!(store.joined_since("example.com").unwrap(), None);
}

#[test]
fn the_list_reads_resend_before_as_a_date() {
    let parsed = allowed::parse(br#"[{"host": "example.com", "resend_before": "2026-10-27"}, {"host": "example.org"}]"#).unwrap();
    assert_eq!(parsed["example.com"].resend_before, Some(CUTOFF));
    assert_eq!(parsed["example.org"].resend_before, None);
}

#[test]
fn a_resend_walk_lists_pages_sent_before_the_cutoff_newest_first_and_not_twice() {
    let dir = Dir::new("resend");
    let store = open(&dir);
    let names = ["a", "b", "c", "old", "failing", "later", "future"];
    let lastmods = [LATE, LATE + 300, LATE + 200, EARLY, LATE, LATE, u32::MAX - 10];
    store.record_listing("example.com", 7, names.iter().zip(lastmods).map(|(name, lastmod)| page(name, lastmod)).collect(), 100).unwrap();
    let picks = take(&store, 10);
    let by = |name: &str| picks.iter().filter(|p| p.url.ends_with(&format!("/{name}"))).cloned().collect::<Vec<Pick>>();
    store.mark_sent(&[by("a"), by("b"), by("c"), by("old"), by("future")].concat(), 1000, &rules()).unwrap();
    let mut failing = by("failing");
    failing[0].reason = Reason::Requeued;
    failing[0].failures = 1;
    store.mark_sent(&failing, 1000, &rules()).unwrap();
    store.mark_sent(&by("later"), 3000, &rules()).unwrap();
    store.record_listing("example.com", 7, vec![page("never", LATE)], 1500).unwrap();

    let now = 1_800_000_000;
    let terms = Terms { since: Some(FLOOR), resend_before: Some(2000), ..Terms::default() };
    let resent = store.resend_domain("example.com", terms, now, &mut |_| true, &mut |_| true).unwrap();
    assert_eq!((resent.queued, resent.before_floor), (4, 1), "a, b, c and future; the page dated before the floor stays off, the failing one waits for its retry, and the one sent since the cutoff and the one never sent aren't touched");
    let orders: Vec<Order> = store.resend_list("example.com", 10).unwrap().iter().map(|e| e.order).collect();
    assert_eq!(orders, [Order::Resend { lastmod: now }, Order::Resend { lastmod: LATE + 300 }, Order::Resend { lastmod: LATE + 200 }, Order::Resend { lastmod: LATE }], "newest lastmod first, a future one held to now");
    assert_eq!(store.ready_count("example.com").unwrap(), 1, "they stay off the ready list, which holds only the page never sent");
    assert_eq!(store.resend_count("example.com").unwrap(), 4);
    assert_eq!(store.resent_before("example.com").unwrap(), Some(2000));

    let (again, left) = store.take_resend("example.com", 10, &Default::default()).unwrap();
    assert_eq!((again.len(), left), (4, 4));
    assert!(again.iter().all(|p| p.reason == Reason::Resend));
    store.mark_sent(&again, 2500, &rules()).unwrap();
    assert_eq!(store.resend_count("example.com").unwrap(), 0);
    assert_eq!(store.resend_domain("example.com", terms, now, &mut |_| true, &mut |_| true).unwrap().queued, 0, "a page that went again isn't sent again by the same cutoff");
}

#[test]
fn a_resend_walk_stopped_midway_finishes_on_the_next_round_without_listing_a_page_twice() {
    let dir = Dir::new("resend-stop");
    let store = open(&dir);
    store.record_listing("example.com", 7, (0..45_000).map(|i| page(&format!("p{i:05}"), LATE)).collect(), 100).unwrap();
    store.mark_sent(&take(&store, 45_000), 1000, &rules()).unwrap();
    let terms = Terms { resend_before: Some(2000), ..Terms::default() };
    let mut chunks = 0;
    let first = store.resend_domain("example.com", terms, 5000, &mut |_| true, &mut |_| {
        chunks += 1;
        chunks < 3
    }).unwrap();
    assert!(first.stopped);
    assert_eq!(first.queued, 6_000, "three chunks went before the stop");
    assert_eq!(store.resent_before("example.com").unwrap(), None, "a stopped walk leaves no mark, so the next round walks the domain again");
    let second = store.resend_domain("example.com", terms, 5000, &mut |_| true, &mut |_| true).unwrap();
    assert!(!second.stopped);
    assert_eq!(second.queued, 39_000, "the rest, the pages already listed passed over");
    assert_eq!(store.resend_count("example.com").unwrap(), 45_000, "each page listed once");
    assert_eq!(store.resent_before("example.com").unwrap(), Some(2000));
}

#[test]
fn the_resend_walk_walks_a_domain_once_per_resend_before_at_its_own_pace() {
    let dir = Dir::new("resend-round");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    let listed = r#"[{"host": "example.com", "resend_before": "2026-10-27"}]"#;
    std::fs::write(&path, listed).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, (0..3001).map(|i| page(&format!("p{i:04}"), LATE)).collect(), 100).unwrap();
    let picks = take(store, 4000);
    store.mark_sent(&picks[..3000], CUTOFF - 86_400, &rules()).unwrap();
    store.mark_sent(&picks[3000..], CUTOFF + 3600, &rules()).unwrap();
    let progress = Progress::default();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let resent = || progress.join_resent.load(std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();
    assert_eq!(desearch_bot::dispatch::resend_round(&buckets, &list, 1_000_000, 4_000, &progress, &stop).unwrap(), 1);
    assert!(started.elapsed() >= std::time::Duration::from_millis(600), "3,000 pages at 4,000 a second take at least 0.75 s");
    assert_eq!(resent(), 3000, "the pages sent before the cutoff go again");
    relist(&path, &list, listed);
    assert_eq!(desearch_bot::dispatch::resend_round(&buckets, &list, 1_000_000, 1_000_000, &progress, &stop).unwrap(), 0, "a list built again with the same cutoff walks nothing");
    relist(&path, &list, r#"[{"host": "example.com", "resend_before": "2026-10-28"}]"#);
    assert_eq!(desearch_bot::dispatch::resend_round(&buckets, &list, 1_000_000, 1_000_000, &progress, &stop).unwrap(), 1, "a moved cutoff walks it once more");
    assert_eq!(resent(), 3001, "and takes the page sent before the new cutoff");
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1, "the list walker walks the domain for its cap and floor, not for resends");
}

#[test]
fn unresend_takes_every_waiting_resend_back_and_frees_its_pages() {
    let dir = Dir::new("unresend");
    let store = open(&dir);
    store.record_listing("example.com", 7, (0..5).map(|i| page(&format!("p{i}"), LATE)).collect(), 100).unwrap();
    store.mark_sent(&take(&store, 10), 1000, &rules()).unwrap();
    let terms = Terms { resend_before: Some(2000), ..Terms::default() };
    assert_eq!(store.resend_domain("example.com", terms, 5000, &mut |_| true, &mut |_| true).unwrap().queued, 5);
    assert!(store.sent("example.com", &rest("p0")).unwrap().unwrap().requeued);
    assert_eq!(store.unresend().unwrap(), 5);
    assert!(store.resend_list("example.com", 10).unwrap().is_empty());
    assert_eq!((store.resend_count("example.com").unwrap(), store.resent_before("example.com").unwrap()), (0, None));
    assert!(!store.sent("example.com", &rest("p0")).unwrap().unwrap().requeued, "the page can be refreshed again");
    assert_eq!(store.stock_refresh("example.com", 10, None, usize::MAX, &mut |_| true).unwrap().map(|s| s.count), Some(5));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pages_going_once_more_take_only_the_room_new_pages_leave() {
    let dir = Dir::new("dispatcher-resend");
    let hosts: Vec<String> = (0..4).map(|i| format!("old{i}.com")).chain(["new.com".to_string()]).collect();
    let names: Vec<&str> = hosts.iter().map(String::as_str).collect();
    let buckets = Arc::new(known(&dir, &names));
    buckets.allow(&Allowed::with_terms(hosts.iter().map(|host| (host.clone(), Terms { resend_before: Some(5000), ..Terms::default() }))));
    for host in &hosts[..4] {
        let store = buckets.store(host);
        store.record_listing(host, 1, (0..500).map(|i| (urls::parse(&format!("https://{host}/p{i:03}"), host).unwrap(), 100, false)).collect(), 100).unwrap();
        let mut quotas = [0; LANES];
        quotas[FRESH] = 1000;
        store.mark_sent(&store.take_ready(host, quotas).unwrap(), 1000, &rules()).unwrap();
        store.resend_domain(host, buckets.terms(host), 2000, &mut |_| true, &mut |_| true).unwrap();
    }
    let now = 1_800_000_000;
    buckets.store("new.com").record_listing("new.com", 1, (0..1500).map(|i| (urls::parse(&format!("https://new.com/n{i:04}"), "new.com").unwrap(), 100, false)).collect(), now).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    let progress = Arc::new(Progress::default());
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap();
    let sent = |host: &str| fake.lock().unwrap().enqueued.iter().map(|(_, body)| serde_json::from_slice::<Value>(body).unwrap()["urls"].as_array().unwrap().iter().filter(|u| u["host"] == host).count()).sum::<usize>();
    let resent = || progress.sent_resend.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(dispatcher.pass(now).await.unwrap(), 1000);
    assert_eq!((sent("new.com"), resent()), (1000, 0), "new pages take the whole room");
    assert_eq!(dispatcher.pass(now + 5).await.unwrap(), 1000);
    assert_eq!(sent("new.com"), 1500, "new pages keep the room they had, though 4 domains have 2,000 pages to send again");
    assert_eq!(resent(), 500, "resends take only what new pages leave");
    assert_eq!(dispatcher.pass(now + 10).await.unwrap(), 1000, "then all of it");
    assert_eq!(resent(), 1500);
}

#[test]
fn outcomes_of_pages_sent_once_more_stay_out_of_the_yield() {
    let dir = Dir::new("resent-outcomes");
    let buckets = known(&dir, &["example.com"]);
    let rows = vec![
        outcome("https://example.com/old", Outcome::Failed, 5000),
        outcome("https://example.com/new", Outcome::Published, 5000),
        outcome("https://example.com/gone", Outcome::Failed, 5000),
        outcome("https://example.com/lost", Outcome::Dropped, 5000),
    ];
    let resent = Arc::new(ResentPages::default());
    resent.insert("example.com", b"example.com/old");
    resent.insert("example.com", b"example.com/lost");
    let mut reader = OutcomeReader::new(rules(), 1000).with_resent(resent.clone());
    reader.apply(&buckets, &rows, 5000).unwrap();
    let store = buckets.store("example.com");
    let counted = store.domain_yield("example.com").unwrap().unwrap();
    assert_eq!((counted.ok, counted.failed), (1.0, 1.0), "the resend's failure stays out; the other failure counts");
    assert!(store.next_send("example.com", &rest("old")).is_ok_and(|next| next.is_none()), "a failed resend gets no retry of its own, whose outcome would count; the fill takes the page in its turn");
    assert!(store.next_send("example.com", &rest("lost")).unwrap().is_some(), "a resend whose task was lost goes again");
    assert!(store.next_send("example.com", &rest("gone")).unwrap().is_some());

    let path = dir.0.join("resent-pages.bin");
    resent.save(&path).unwrap();
    let kept = ResentPages::load(&path);
    assert!(kept.contains("example.com", b"example.com/old") && !kept.contains("example.com", b"example.com/new"), "the set outlives a restart");
    assert!(!ResentPages::load(&dir.0.join("none.bin")).active());
}

#[test]
fn a_big_cap_that_holds_every_page_streams_them_all() {
    let dir = Dir::new("join-big-cap");
    let store = open(&dir);
    store.record_listing("example.com", 7, (0..30).map(|i| page(&format!("p{i:02}"), LATE)).collect(), 100).unwrap();
    store.clear_fresh_before("example.com", u32::MAX, &mut |_| true).unwrap();
    let joined = store.join_domain("example.com", Terms { send_cap: Some(2_000_000), ..Terms::default() }, 5000, &mut |_| true).unwrap();
    assert_eq!((joined.queued, joined.sent, store.ready_count("example.com").unwrap()), (30, 0, 30), "counted first, all 30 fit, and they stream without a heap");
}

#[test]
fn a_resend_take_passes_over_a_page_the_lists_took_this_pass() {
    let dir = Dir::new("resend-picked");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("a", LATE), page("b", LATE + 1)], 100).unwrap();
    store.mark_sent(&take(&store, 10), 1000, &rules()).unwrap();
    store.resend_domain("example.com", Terms { resend_before: Some(2000), ..Terms::default() }, 5000, &mut |_| true, &mut |_| true).unwrap();
    let picked: std::collections::HashSet<Vec<u8>> = [rest("b")].into();
    let (picks, left) = store.take_resend("example.com", 10, &picked).unwrap();
    assert_eq!(picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/a"], "b went on a list this pass, so it isn't taken twice");
    assert_eq!(left, 2, "and it stays waiting");
}

#[test]
fn a_resend_walk_held_to_a_slow_pace_still_stops_when_told() {
    let dir = Dir::new("resend-stop-pace");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"[{"host": "example.com", "resend_before": "2026-10-27"}]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, (0..3000).map(|i| page(&format!("p{i:04}"), LATE)).collect(), 100).unwrap();
    store.mark_sent(&take(store, 4000), CUTOFF - 86_400, &rules()).unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopping = stop.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        stopping.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let started = std::time::Instant::now();
    desearch_bot::dispatch::resend_round(&buckets, &list, 1_000_000, 1, &Progress::default(), &stop).unwrap();
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "a page a second would sleep 2,000 s after the first chunk; the walk wakes when told to stop");
    assert_eq!(store.resent_before("example.com").unwrap(), None, "and leaves the domain to be walked again");
}

#[test]
fn unresend_reads_on_past_what_it_took() {
    let dir = Dir::new("unresend-big");
    let store = open(&dir);
    store.record_listing("example.com", 7, (0..25_000).map(|i| page(&format!("p{i:05}"), LATE)).collect(), 100).unwrap();
    store.mark_sent(&take(&store, 25_000), 1000, &rules()).unwrap();
    assert_eq!(store.resend_domain("example.com", Terms { resend_before: Some(2000), ..Terms::default() }, 5000, &mut |_| true, &mut |_| true).unwrap().queued, 25_000);
    assert_eq!(store.unresend().unwrap(), 25_000, "three chunks of 10,000, each read on from the last");
    assert_eq!(store.resend_count("example.com").unwrap(), 0);
    assert!(store.resend_list("example.com", 10).unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_site_with_only_resends_keeps_its_crawl_delay() {
    let dir = Dir::new("resend-delay");
    let buckets = Arc::new(known(&dir, &["slow.com"]));
    let store = buckets.store("slow.com");
    let mut record = records::new_domain(Some(5), Some("com"), &[], State::Active, None, None);
    record.insert("delay".into(), 60.0.into());
    let mut changes = Changes::default();
    changes.domain("slow.com", &record);
    store.write(changes).unwrap();
    store.record_listing("slow.com", 1, (0..30).map(|i| (urls::parse(&format!("https://slow.com/p{i:02}"), "slow.com").unwrap(), 100, false)).collect(), 100).unwrap();
    let mut quotas = [0; LANES];
    quotas[FRESH] = 100;
    store.mark_sent(&store.take_ready("slow.com", quotas).unwrap(), 1000, &rules()).unwrap();
    store.resend_domain("slow.com", Terms { resend_before: Some(5000), ..Terms::default() }, 2000, &mut |_| true, &mut |_| true).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let mut dispatcher = Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 1000, shares(50, 10), rules(), Arc::new(Progress::default())).await.unwrap();
    let now = 1_800_000_000;
    assert_eq!(dispatcher.pass(now).await.unwrap(), 2, "a minute between requests is 60 an hour, two minutes' worth at once, for resends too");
    assert_eq!(dispatcher.pass(now + 60).await.unwrap(), 0);
    assert_eq!(dispatcher.pass(now + 121).await.unwrap(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_site_with_resends_waiting_gets_no_fill_on_top_of_them() {
    let dir = Dir::new("resend-fill");
    let buckets = Arc::new(known(&dir, &["example.com"]));
    buckets.allow(&Allowed::with_terms([("example.com", Terms { hourly: Some(3600), ..Terms::default() })]));
    let store = buckets.store("example.com");
    store.record_listing("example.com", 1, (0..150).map(|i| (urls::parse(&format!("https://example.com/p{i:03}"), "example.com").unwrap(), 100, false)).collect(), 100).unwrap();
    let mut quotas = [0; LANES];
    quotas[FRESH] = 1000;
    let picks = store.take_ready("example.com", quotas).unwrap();
    store.mark_sent(&picks[..100], 1000, &rules()).unwrap();
    store.mark_sent(&picks[100..], 3000, &rules()).unwrap();
    store.resend_domain("example.com", Terms { resend_before: Some(2000), ..Terms::default() }, 2500, &mut |_| true, &mut |_| true).unwrap();
    let later = 3000 + FILL_FLOOR;
    store.note_outcomes(&[], &[("example.com".to_string(), Yield { ok: 1000.0, failed: 10.0, at: later })]).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let progress = Arc::new(Progress::default());
    let wanted = Arc::new(StockWanted::default());
    let api = TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap();
    assert_eq!(store.stock_refresh("example.com", 1000, None, usize::MAX, &mut |_| true).unwrap().map(|s| s.count), Some(50), "the 50 sent after the cutoff can be refreshed");
    let mut dispatcher = Dispatcher::load(buckets.clone(), api, 1_000_000, shares(50, 10), rules(), progress.clone()).await.unwrap().refreshing(wanted.clone());
    let count = |counter: &std::sync::atomic::AtomicU64| counter.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(dispatcher.pass(later).await.unwrap(), 120, "two minutes of its 3,600 an hour, once");
    assert_eq!((count(&progress.sent_resend), count(&progress.sent_fill)), (100, 20), "the resends first; the fill takes only what they left, once none wait");
}

#[test]
fn only_a_listed_subdomain_whose_record_works_takes_its_pages_from_the_parent() {
    let dir = Dir::new("apart");
    let buckets = known(&dir, &["example.com", "news.example.com", "blog.other.org"]);
    let list = Allowed::of(["example.com", "news.example.com", "blog.other.org"]);
    buckets.allow(&list);
    buckets.note_apart(&list);
    assert!(list.apart("news.example.com"));
    assert!(!list.apart("blog.other.org"), "no listed parent");
    assert!(!list.apart("example.com"));
    let mut changes = Changes::default();
    changes.domain("news.example.com", &records::new_domain(Some(10), Some("com"), &[], State::Down, None, None));
    buckets.store("news.example.com").write(changes).unwrap();
    buckets.note_apart(&list);
    assert!(!list.apart("news.example.com"), "a subdomain that is down leaves its pages under its parent");
    let mut changes = Changes::default();
    changes.domain("news.example.com", &records::new_domain(Some(10), Some("com"), &[], State::Blocked, Some("robots"), None));
    buckets.store("news.example.com").write(changes).unwrap();
    buckets.note_apart(&list);
    assert!(list.apart("news.example.com"), "a subdomain whose robots.txt refuses us keeps its pages from going under the parent too");
    for (reason, apart) in [("not_english", true), ("homepage_error", false)] {
        let mut changes = Changes::default();
        changes.domain("news.example.com", &records::new_domain(Some(10), Some("com"), &[], State::Ineligible, Some(reason), None));
        buckets.store("news.example.com").write(changes).unwrap();
        buckets.note_apart(&list);
        assert_eq!(list.apart("news.example.com"), apart, "ineligible for {reason}");
    }
}

#[test]
fn a_refresh_stock_on_a_floored_site_keeps_its_pages_inside_the_floor_coming() {
    let dir = Dir::new("stock-floor-rounds");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    let mut pages: Vec<_> = (0..1500).map(|i| page(&format!("old{i:04}"), EARLY)).collect();
    pages.extend((0..500).map(|i| page(&format!("new{i:04}"), LATE)));
    store.record_listing("example.com", 7, pages, 100).unwrap();
    let (old, new): (Vec<Pick>, Vec<Pick>) = take(store, 2000).into_iter().partition(|p| p.url.contains("/old"));
    store.mark_sent(&old, 1000, &rules()).unwrap();
    store.mark_sent(&new, 2000, &rules()).unwrap();
    buckets.allow(&Allowed::with_terms([("example.com", Terms { since: Some(FLOOR), ..Terms::default() })]));
    for round in 0..3 {
        store.stock_refresh("example.com", 100, None, usize::MAX, &mut |_| true).unwrap();
        let stock = store.take_stock("example.com", 100, 100_000, 1000).unwrap();
        assert_eq!(stock.picks.len(), 100, "round {round}: the pages sent longest ago are dated before the floor and never go, so they don't hold the stock");
        assert!(stock.picks.iter().all(|p| p.url.contains("/new")));
        store.mark_sent(&stock.picks, 50_000 + round, &rules()).unwrap();
    }
}

#[test]
fn a_floor_that_only_rises_is_recorded_without_walking_the_domain() {
    let dir = Dir::new("floor-rise-idle");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"[{"host": "example.com", "type": "news"}]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    store.record_listing("example.com", 7, (0..30).map(|i| page(&format!("p{i}"), LATE)).collect(), 100).unwrap();
    store.mark_sent(&take(store, 100), 200, &rules()).unwrap();
    let progress = Progress::default();
    let stop = std::sync::atomic::AtomicBool::new(false);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    relist(&path, &list, r#"[{"host": "example.com", "type": "news", "since": "2020-01-01"}]"#);
    let started = std::time::Instant::now();
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1, &progress, &stop).unwrap(), 1);
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "an idle domain whose floor rose isn't walked: at a record a second, a walk of its 30 pages would take 30 s");
    assert_eq!(store.joined_since("example.com").unwrap(), Some(FLOOR));
    relist(&path, &list, r#"[{"host": "example.com", "type": "news"}]"#);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1, "a floor taken off walks it");
    assert_eq!(store.joined_since("example.com").unwrap(), None);
}

#[test]
fn a_backlog_cuts_by_the_pages_stored_not_the_urls_its_sitemaps_last_listed() {
    let dir = Dir::new("join-stored-bound");
    let buckets = known(&dir, &["example.com"]);
    let store = buckets.store("example.com");
    let mut record = records::new_domain(Some(10), Some("com"), &[], State::Active, None, None);
    record.insert("urls".into(), 5.into());
    let mut changes = Changes::default();
    changes.domain("example.com", &record);
    store.write(changes).unwrap();
    store.record_listing("example.com", 7, (0..50).map(|i| page(&format!("p{i:02}"), LATE + i)).collect(), 100).unwrap();
    store.clear_fresh_before("example.com", u32::MAX, &mut |_| true).unwrap();
    let joined = store.join_domain("example.com", Terms { backlog: Some(10), ..Terms::default() }, 5000, &mut |_| true).unwrap();
    assert_eq!(joined.queued, 10, "a news feed listing 5 URLs over 50 stored pages still takes only its 10 newest");
}

#[test]
fn a_retry_dated_before_the_floor_drops_as_it_would_go_and_is_counted() {
    let dir = Dir::new("retry-floor");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("2019/05/01/story", 0)], 100).unwrap();
    store.mark_sent(&take(&store, 10), 1000, &rules()).unwrap();
    assert_eq!(requeue(&store, 1000 + RECRAWL).moved, 1, "an undated page goes again on its re-crawl interval");
    store.allow(Allowed::with_terms([("example.com", Terms { since: Some(FLOOR), ..Terms::default() })]));
    let mut quotas = [0; LANES];
    quotas[REQUEUED] = 10;
    assert!(store.take_ready("example.com", quotas).unwrap().is_empty(), "its URL dates it before the floor, so the re-crawl stops");
    assert_eq!(store.floor_dropped.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(store.ready_count("example.com").unwrap(), 0);
    assert!(!store.sent("example.com", &rest("2019/05/01/story")).unwrap().unwrap().requeued, "its going-again mark goes too, so a lower floor can take it later");
}

#[test]
fn a_take_passes_over_a_bounded_run_of_pages_before_the_floor_and_leaves_the_rest_for_later() {
    let dir = Dir::new("stale-run");
    let store = open(&dir);
    store.record_listing("example.com", 7, vec![page("inside", LATE)], 100).unwrap();
    store.record_listing("example.com", 8, (0..25_000).map(|i| page(&format!("old{i:05}"), EARLY)).collect(), 200).unwrap();
    store.allow(Allowed::with_terms([("example.com", Terms { since: Some(FLOOR), ..Terms::default() })]));
    assert!(take(&store, 10).is_empty(), "the newest-found run is all before the floor, and one take passes over only so many");
    let dropped = || store.floor_dropped.load(std::sync::atomic::Ordering::Relaxed);
    assert!((20_000..25_000).contains(&dropped()), "{} dropped in the first take", dropped());
    let picks = take(&store, 10);
    assert_eq!(picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/inside"], "the next take finishes the run and reaches it");
    assert_eq!(dropped(), 25_000);
}

#[test]
fn a_domain_already_sending_whose_cap_rises_queues_the_stored_pages_its_old_cap_left_out() {
    let dir = Dir::new("join-cap-rise");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let buckets = known(&dir, &["example.com"]);
    buckets.allow(&list);
    let store = buckets.store("example.com");
    for (i, name) in ["a", "b", "c", "d", "e", "f"].iter().enumerate() {
        store.record_listing("example.com", 7, vec![page(name, 10)], 100 + 10 * i as u32).unwrap();
    }
    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 3}]"#);
    let progress = Progress::default();
    let stop = std::sync::atomic::AtomicBool::new(false);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert_eq!(ready(store), ["f", "e", "d"]);
    store.mark_sent(&take(store, 1), 2000, &rules()).unwrap();
    assert_eq!(store.ready_count("example.com").unwrap(), 2, "still sending when its cap rises");

    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 10}]"#);
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 1);
    assert_eq!(ready(store), ["e", "d", "c", "b", "a"], "every page never sent is queued, the ones the old cap left out too");
    assert_eq!(store.ready_count("example.com").unwrap(), 5);
    assert_eq!(store.joined("example.com").unwrap(), Some(Some(10)));
    assert_eq!(desearch_bot::dispatch::join_round(&buckets, &list, 1_000_000, &progress, &stop).unwrap(), 0);
}

#[test]
fn a_page_on_a_subdomain_listed_on_its_own_goes_only_under_that_subdomain() {
    let dir = Dir::new("listed-apart");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["example.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    let on_sub = |name: &str| (urls::parse(&format!("https://news.example.com/{name}"), "example.com").unwrap(), 10, false);
    let queued = |store: &BucketStore| store.ready_list("example.com", 100).unwrap().iter().map(|e| String::from_utf8(e.rest.clone()).unwrap()).collect::<Vec<_>>();
    store.record_listing("example.com", 7, vec![page("a", 10), on_sub("s1")], 100).unwrap();
    assert_eq!(queued(&store).len(), 2, "an unlisted subdomain's pages go under the domain whose sitemap listed them");

    relist(&path, &list, r#"["example.com", "news.example.com"]"#);
    store.record_listing("example.com", 7, vec![on_sub("s2")], 200).unwrap();
    assert!(queued(&store).contains(&"news.example.com/s2".to_string()), "a listed subdomain whose own record doesn't work leaves its pages under the parent");
    list.set_apart(["news.example.com".to_string()].into());
    store.record_listing("example.com", 7, vec![on_sub("s3")], 300).unwrap();
    assert!(!queued(&store).contains(&"news.example.com/s3".to_string()), "once its record works, a sitemap read leaves its pages to it");
    let picks = take(&store, 10);
    assert_eq!(picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/a"], "and those already waiting drop as they would go");
    assert_eq!(queued(&store), ["example.com/a"], "the subdomain's page left the list; the page taken waits to be marked sent");
}

#[test]
fn a_crawl_delay_sets_the_pages_an_hour_a_site_can_take() {
    let record = |delay: Value| {
        let mut record = records::new_domain(None, None, &[], State::Active, None, None);
        record.insert("delay".into(), delay);
        record
    };
    assert_eq!(desearch_bot::dispatch::delay_hourly(&record(Value::from(15.0))), Some(240));
    assert_eq!(desearch_bot::dispatch::delay_hourly(&record(Value::from(1))), Some(3600));
    assert_eq!(desearch_bot::dispatch::delay_hourly(&record(Value::from(600))), Some(60), "at most a minute apart, as the bot reads them");
    assert_eq!(desearch_bot::dispatch::delay_hourly(&record(Value::from(0))), None);
    assert_eq!(desearch_bot::dispatch::delay_hourly(&record(Value::Null)), None);
    let mut cap = HourlyCap::new(36_000);
    assert_eq!(cap.allowance_within("arxiv.org", 1000, 240, 1), 8, "two minutes of 240 an hour, with no floor of a hundred");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_site_asking_for_a_delay_between_requests_gets_no_more_than_the_delay_allows() {
    let dir = Dir::new("crawl-delay");
    let mut owned = vec![bucket_of("example.com"), bucket_of("example.org")];
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&dir.0, &owned, &resources()).unwrap());
    for (host, delay) in [("example.com", Some(60.0)), ("example.org", None)] {
        let store = buckets.store(host);
        let mut record = records::new_domain(Some(5), Some("com"), &[], State::Active, None, None);
        if let Some(delay) = delay {
            record.insert("delay".into(), delay.into());
        }
        let mut changes = Changes::default();
        changes.domain(host, &record);
        store.write(changes).unwrap();
        let pages = (0..30).map(|i| (urls::parse(&format!("https://{host}/p{i:02}"), host).unwrap(), 100 + i, false)).collect();
        store.record_listing(host, 1, pages, 100).unwrap();
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(FakeApi { room: 1, ..FakeApi::default() }));
    tokio::spawn(serve_api(listener, fake.clone()));
    let mut dispatcher = Dispatcher::load(buckets.clone(), TaskApi::new(&base, Hotkey::from_uri("//test").unwrap()).unwrap(), 1000, shares(50, 10), rules(), Arc::new(Progress::default())).await.unwrap();
    let sent_by = |fake: &Arc<Mutex<FakeApi>>, host: &str| {
        fake.lock().unwrap().enqueued.iter().map(|(_, body)| serde_json::from_slice::<Value>(body).unwrap()["urls"].as_array().unwrap().iter().filter(|u| u["host"] == host).count()).sum::<usize>()
    };
    let now = 1_759_400_000;
    dispatcher.pass(now).await.unwrap();
    assert_eq!((sent_by(&fake, "example.com"), sent_by(&fake, "example.org")), (2, 30), "a minute between requests is 60 an hour, two minutes' worth at once");
    dispatcher.pass(now + 60).await.unwrap();
    assert_eq!(sent_by(&fake, "example.com"), 2);
    dispatcher.pass(now + 121).await.unwrap();
    assert_eq!(sent_by(&fake, "example.com"), 4);
}

#[test]
fn an_inbox_file_lists_its_hosts_pages_like_a_sitemap_and_is_read_again_only_when_it_changes() {
    let dir = Dir::new("inbox");
    std::fs::create_dir_all(dir.0.join("inbox")).unwrap();
    let buckets = known(&dir, &["example.com", "example.net"]);
    buckets.allow(&Allowed::with_terms([("example.com", Terms { since: Some(FLOOR), ..Terms::default() }), ("example.net", Terms::default())]));
    let inbox = dir.0.join("inbox");
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut lines = b"https://example.com/a\t2021-03-04\nhttps://example.com/old\t2019-05-06\nhttps://example.com/undated\nhttps://elsewhere.org/x\nnot a url\n".to_vec();
    lines.extend_from_slice(b"https://example.com/caf\xe9\n");
    std::fs::write(inbox.join("www.example.com.txt"), lines).unwrap();
    std::fs::write(inbox.join("unknown.net.txt"), "https://unknown.net/a\n").unwrap();
    std::fs::create_dir_all(inbox.join("example.net.txt")).unwrap();
    let taken = desearch_bot::inbox::take_in(&buckets, &inbox, 5000, &stop).unwrap();
    assert_eq!(taken.failed, 1, "a file that can't be read is skipped, and the rest of the inbox goes on");
    assert_eq!((taken.files, taken.urls, taken.queued), (1, 4, 3), "the host's own URLs, a bad byte read as a replacement, the one before its floor stored but not queued");
    let store = buckets.store("example.com");
    let mut queued = ready(store);
    queued.sort();
    assert_eq!(queued.len(), 3);
    assert!(queued.contains(&"a".to_string()) && queued.contains(&"undated".to_string()));
    assert_eq!(record(store, "a").lastmod, 1_614_816_000);
    assert_eq!(desearch_bot::inbox::take_in(&buckets, &inbox, 6000, &stop).unwrap().files, 0, "an unchanged file is not read again, after a restart too: its time is kept in the store");

    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(inbox.join("www.example.com.txt"), "https://example.com/a\t2021-03-04\nhttps://example.com/b\t2022-01-01\n").unwrap();
    let again = desearch_bot::inbox::take_in(&buckets, &inbox, 7000, &stop).unwrap();
    assert_eq!((again.files, again.queued), (1, 1), "only the new page joins the list");
}

#[test]
fn a_capped_walk_keeps_the_pages_found_while_it_ran() {
    let dir = Dir::new("join-found-during");
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join("domains.json");
    std::fs::write(&path, r#"["other.com"]"#).unwrap();
    let list = Allowed::from_file(&path).unwrap();
    let store = BucketStore::open(&dir.0.join("store"), &resources()).unwrap();
    store.allow(list.clone());
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 10)], 100).unwrap();
    relist(&path, &list, r#"[{"host": "example.com", "send_cap": 10}]"#);
    store.record_listing("example.com", 7, vec![page("found-during", 10)], 2000).unwrap();
    assert_eq!(ready(&store), ["found-during"], "found after the walk began, it went on the list when found");
    store.join_domain("example.com", list.get("example.com").unwrap(), 1500, &mut |_| true).unwrap();
    let mut queued = ready(&store);
    queued.sort();
    assert_eq!(queued, ["a", "b", "found-during"], "the walk's clear leaves what was found since it began");
}

//! Due queues on throwaway stores: building a domain's set, taking what is due, and moving pages after sends and outcomes.

use std::path::PathBuf;

use desearch_bot::allowed::{Allowed, SiteType, Terms};
use desearch_bot::buckets::{BucketStore, Changes, Resources};
use desearch_bot::due::{front_page, interval_for, Built};
use desearch_bot::ready::{Order, Pick, PageSchedule, Reason, Scheduled, Why, FRESH, LANES};
use desearch_bot::recrawl::Recrawl;
use desearch_bot::records;
use desearch_bot::states::State;
use desearch_bot::urls::{self, Url};

const DAY: u32 = 86_400;
const HOUR: u32 = 3_600;

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let path = std::env::temp_dir().join(format!("desearch-bot-due-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        Dir(path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn url(path: &str) -> Url {
    urls::parse(&format!("https://example.com/{path}"), "example.com").unwrap()
}

fn page(path: &str, lastmod: u32) -> (Url, u32, bool) {
    (url(path), lastmod, false)
}

fn rules() -> Recrawl {
    Recrawl::new(7 * DAY, 30 * DAY)
}

/// A store holding example.com, listed with these terms.
fn listed_store(dir: &Dir, terms: Terms) -> BucketStore {
    let store = BucketStore::open(&dir.0, &Resources::new(8 << 20, 8 << 20, 1)).unwrap();
    store.allow(Allowed::with_terms([("example.com", terms)]));
    let mut changes = Changes::default();
    changes.domain("example.com", &records::new_domain(Some(10), Some("com"), &[], State::Active, None, None));
    store.write(changes).unwrap();
    store
}

/// Send these pages through the ready lists, as the bot did before its queues.
fn send(store: &BucketStore, paths: &[&str], at: u32) {
    let mut quotas = [0; LANES];
    quotas[FRESH] = 10_000;
    let picks: Vec<Pick> = store.take_ready("example.com", quotas).unwrap().into_iter().filter(|p| paths.iter().any(|path| p.url.ends_with(&format!("/{path}")))).collect();
    assert_eq!(picks.len(), paths.len());
    store.mark_sent(&picks, at, &rules()).unwrap();
}

fn queue(store: &BucketStore, now: u32) -> Vec<(String, u32, Reason)> {
    let taken = store.take_due("example.com", 1000, now, None).unwrap();
    taken.picks.iter().map(|p| (p.url.trim_start_matches("https://example.com/").to_string(), due_of(p), p.reason)).collect()
}

fn due_of(pick: &Pick) -> u32 {
    match pick.entry.order {
        Order::Due { due, .. } => due,
        _ => panic!("not from a queue"),
    }
}

#[test]
fn intervals_follow_the_site_type_and_for_news_the_article_age() {
    let at = 100 * DAY;
    assert_eq!(interval_for(SiteType::News, at - HOUR, at), 12 * HOUR, "a fresh article");
    assert_eq!(interval_for(SiteType::News, at - 3 * DAY, at), DAY);
    assert_eq!(interval_for(SiteType::News, at - 10 * DAY, at), 7 * DAY);
    assert_eq!(interval_for(SiteType::News, at - 40 * DAY, at), 30 * DAY);
    assert_eq!(interval_for(SiteType::Reference, 0, at), 7 * DAY);
    assert_eq!(interval_for(SiteType::Community, 0, at), 14 * DAY);
    assert_eq!(interval_for(SiteType::Company, 0, at), 30 * DAY);
    let mut overridden = Recrawl::new(7 * DAY, 30 * DAY).with_overrides("bbc.co.uk 2").unwrap();
    assert_eq!(desearch_bot::due::page_interval(&overridden, SiteType::News, "news.bbc.co.uk", at, at), 2 * DAY, "a domain's own line wins");
    overridden = Recrawl::default();
    assert_eq!(desearch_bot::due::page_interval(&overridden, SiteType::News, "bbc.co.uk", at, at), 12 * HOUR);
}

#[test]
fn front_pages_are_the_root_and_short_section_paths() {
    for rest in ["example.com/", "example.com", "example.com/world", "example.com/sport/", "example.com/us-news", "example.com/?ref=x"] {
        assert!(front_page(rest.as_bytes()), "{rest}");
    }
    for rest in ["example.com/2026/10/26/story", "example.com/world/europe-floods-2026", "example.com/a-long-article-slug", "example.com/story123", "example.com/page.html"] {
        assert!(!front_page(rest.as_bytes()), "{rest}");
    }
}

#[test]
fn a_built_queue_holds_every_sent_page_not_gone_and_the_newest_unsent_up_to_the_cap() {
    let dir = Dir::new("build");
    let store = listed_store(&dir, Terms { send_cap: Some(5), site: SiteType::Reference, ..Terms::default() });
    let paths = ["a", "b", "c", "d", "e", "f", "g", "tag/x"];
    for (i, path) in paths.iter().enumerate() {
        store.record_listing("example.com", 7, vec![page(path, 10)], 100 + 10 * i as u32).unwrap();
    }
    send(&store, &["a", "b", "c"], 1000);
    // b failed twice in a row: gone. c's lastmod moved since it went.
    store.note_outcomes(&[PageSchedule { domain: "example.com".into(), rest: b"example.com/b".to_vec(), next: Scheduled { why: Why::Failed, due: 1100, at: 1000 } }], &[]).unwrap();
    store.requeue_due(1200, 100, &rules()).unwrap();
    let mut quotas = [0; LANES];
    quotas[1] = 10;
    let retry: Vec<Pick> = store.take_ready("example.com", quotas).unwrap().into_iter().filter(|p| p.url.ends_with("/b")).collect();
    store.mark_sent(&retry, 1300, &rules()).unwrap();
    store.note_outcomes(&[PageSchedule { domain: "example.com".into(), rest: b"example.com/b".to_vec(), next: Scheduled { why: Why::Failed, due: 1400, at: 1300 } }], &[]).unwrap();
    store.requeue_due(1500, 100, &rules()).unwrap();
    store.record_listing("example.com", 7, vec![page("c", 20)], 1600).unwrap();

    let built = store.build_queue("example.com", store.terms("example.com"), &rules(), 2000, &mut |_| true).unwrap();
    assert_eq!(built, Built { scanned: 8, members: 2, admitted: 3, stopped: false }, "a and c are kept, b is gone, and the three newest unsent fill the cap of 5");
    assert_eq!(store.set_size("example.com").unwrap(), Some(5));
    let due = queue(&store, 2000);
    assert_eq!(due.iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["e", "f", "g", "c"], "the three newest unsent, first crawls first in their second; then c, changed");
    assert!(due[..3].iter().all(|(_, at, why)| *at == 2000 && *why == Reason::Fresh));
    assert_eq!(due[3].2, Reason::Changed);
    let later = queue(&store, 1000 + 7 * DAY);
    assert_eq!(later.last().map(|(path, due, why)| (path.as_str(), *due, *why)), Some(("a", 1000 + 7 * DAY, Reason::Refresh)), "a falls due a reference interval after it went");
    assert_eq!(store.due_head("example.com").unwrap(), Some(2000));
}

#[test]
fn a_take_goes_by_due_time_drops_stale_keys_and_pulls_forward_only_past_the_floor() {
    let dir = Dir::new("take");
    let store = listed_store(&dir, Terms { site: SiteType::Company, ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 10)], 100).unwrap();
    send(&store, &["a"], 1000);
    store.build_queue("example.com", store.terms("example.com"), &rules(), 2000, &mut |_| true).unwrap();

    let now = store.take_due("example.com", 10, 2000, None).unwrap();
    assert_eq!(now.picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/b"], "only b is due; a waits 30 days");
    assert_eq!(now.head, Some(1000 + 30 * DAY));
    store.mark_sent(&now.picks, 2000, &rules()).unwrap();
    assert_eq!(store.due_head("example.com").unwrap(), Some(1000 + 30 * DAY), "b went at 2000 and falls due after a's");

    let early = store.take_due("example.com", 10, 2000 + DAY - 1, Some(DAY)).unwrap();
    assert_eq!(early.picks.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(), ["https://example.com/a"], "a went over a day ago, so it may go early");
    assert_eq!(early.picks[0].reason, Reason::Pulled);
    assert_eq!(early.floor_until, Some(2000 + DAY), "b went within the day and holds the rest back");
}

#[test]
fn a_send_from_the_queue_sets_the_next_due_and_a_change_in_flight_goes_at_once() {
    let dir = Dir::new("sent");
    let store = listed_store(&dir, Terms { site: SiteType::News, backlog: Some(100), ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("2026/a-story", 10), page("2026/b-story", 10)], 100).unwrap();
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    let picks = store.take_due("example.com", 10, 1000, None).unwrap().picks;
    assert_eq!(picks.len(), 2);
    store.record_listing("example.com", 7, vec![page("2026/b-story", 30)], 1500).unwrap();
    store.mark_sent(&picks, 2000, &rules()).unwrap();
    let due = queue(&store, 2000 + 12 * HOUR);
    assert_eq!(due, [("2026/b-story".to_string(), 2000, Reason::Changed), ("2026/a-story".to_string(), 2000 + 12 * HOUR, Reason::Refresh)], "b changed in flight; a is a fresh article");
}

#[test]
fn sitemap_pages_join_while_the_cap_leaves_room_and_news_rolls() {
    let dir = Dir::new("listing");
    let store = listed_store(&dir, Terms { send_cap: Some(2), site: SiteType::Reference, ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("a", 10)], 100).unwrap();
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    store.record_listing("example.com", 7, vec![page("b", 10), page("c", 10)], 1100).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(2), "b joins, c waits for a slot");
    assert_eq!(queue(&store, 1100).iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["a", "b"]);

    let dir = Dir::new("rolling");
    let news = listed_store(&dir, Terms { backlog: Some(2), site: SiteType::News, ..Terms::default() });
    news.record_listing("example.com", 7, vec![page("2026/old-story", 10), page("2026/older-story", 10)], 100).unwrap();
    news.build_queue("example.com", news.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    let picks = news.take_due("example.com", 10, 1000, None).unwrap().picks;
    news.mark_sent(&picks, 1000, &rules()).unwrap();
    news.record_listing("example.com", 7, vec![page("2026/new-story", 10), page("world", 10)], 5 * DAY).unwrap();
    assert_eq!(news.set_size("example.com").unwrap(), Some(2), "the set stays at its backlog");
    let due = queue(&news, 400 * DAY);
    assert_eq!(due.len(), 2, "one article left as the new one joined; the section page never joins");
    assert_eq!(due.iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["2026/old-story", "2026/new-story"]);
    assert_eq!(due[1], ("2026/new-story".to_string(), 5 * DAY, Reason::Fresh));
}

#[test]
fn a_failure_falls_due_a_day_later_the_second_leaves_the_set_and_a_dropped_page_goes_as_the_reader_says() {
    let dir = Dir::new("outcomes");
    let store = listed_store(&dir, Terms { site: SiteType::Reference, ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("a", 10), page("b", 10)], 100).unwrap();
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    let picks = store.take_due("example.com", 10, 1000, None).unwrap().picks;
    store.mark_sent(&picks, 1000, &rules()).unwrap();
    let outcome = |path: &str, why: Why, due: u32, at: u32| PageSchedule { domain: "example.com".into(), rest: format!("example.com/{path}").into_bytes(), next: Scheduled { why, due, at } };

    store.note_outcomes(&[outcome("a", Why::Failed, 0, 1000), outcome("b", Why::Dropped, 1200, 1000)], &[]).unwrap();
    let due = queue(&store, 1000 + DAY);
    assert_eq!(due, [("b".to_string(), 1200, Reason::Requeued), ("a".to_string(), 1000 + DAY, Reason::Requeued)]);
    let retry = store.take_due("example.com", 10, 1000 + DAY, None).unwrap().picks;
    store.mark_sent(&retry, 1000 + DAY, &rules()).unwrap();
    assert_eq!(retry.iter().find(|p| p.url.ends_with("/a")).map(|p| p.failures), Some(1), "a retry keeps its count");

    store.note_outcomes(&[outcome("a", Why::Failed, 0, 1000 + DAY)], &[]).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(1), "a failed twice in a row and left the set");
    store.note_outcomes(&[outcome("b", Why::Failed, 0, 900)], &[]).unwrap();
    assert_eq!(queue(&store, 400 * DAY).iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["b"], "an outcome of an earlier send of b changes nothing");
}

#[test]
fn a_refresh_overdue_when_the_queue_is_built_falls_due_within_its_next_interval_behind_first_crawls() {
    let dir = Dir::new("overdue");
    let store = listed_store(&dir, Terms { site: SiteType::Company, ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("old", 10)], 100).unwrap();
    send(&store, &["old"], 1000);
    store.record_listing("example.com", 7, vec![page("new", 10)], 2000).unwrap();
    let now = 1000 + 40 * DAY;
    store.build_queue("example.com", store.terms("example.com"), &rules(), now, &mut |_| true).unwrap();
    let due = queue(&store, now + 30 * DAY);
    assert_eq!(due[0], ("new".to_string(), now, Reason::Fresh), "the first crawl goes now");
    assert_eq!(due[1].0, "old");
    assert!((now..now + 30 * DAY).contains(&due[1].1), "the overdue refresh goes within the next 30 days, not all at once");
}

#[test]
fn a_cap_cut_below_the_set_drops_the_pages_seen_longest_ago() {
    let dir = Dir::new("shrink");
    let store = listed_store(&dir, Terms { send_cap: Some(10), site: SiteType::Reference, ..Terms::default() });
    for (i, path) in ["a", "b", "c", "d", "e", "f"].iter().enumerate() {
        store.record_listing("example.com", 7, vec![page(path, 10)], 100 + 10 * i as u32).unwrap();
    }
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(6));
    let cut = Terms { send_cap: Some(3), site: SiteType::Reference, ..Terms::default() };
    let shrunk = store.shrink_set("example.com", cut, &mut |_| true).unwrap();
    assert_eq!(shrunk.scanned, 6);
    assert_eq!(store.set_size("example.com").unwrap(), Some(3));
    let mut left: Vec<String> = queue(&store, 1000).into_iter().map(|(path, _, _)| path).collect();
    left.sort();
    assert_eq!(left, ["d", "e", "f"], "the three seen last stay");
    assert!(store.next_send("example.com", b"example.com/a").unwrap().is_none(), "a dropped page keeps no schedule");
    assert!(store.url(&url("a")).unwrap().is_some(), "its stored record stays");
}

#[test]
fn a_page_the_index_does_not_take_leaves_its_set_after_its_crawl_and_stays_out_of_a_rebuild() {
    assert_eq!(desearch_bot::ready::Outcome::parse("unindexable"), Some(desearch_bot::ready::Outcome::Unindexable));
    let dir = Dir::new("leave");
    let store = listed_store(&dir, Terms { site: SiteType::News, backlog: Some(100), ..Terms::default() });
    store.record_listing("example.com", 7, vec![page("2026/english-story", 10), page("2026/hindi-story", 10)], 100).unwrap();
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    let picks = store.take_due("example.com", 10, 1000, None).unwrap().picks;
    store.mark_sent(&picks, 1000, &rules()).unwrap();
    let leave = PageSchedule { domain: "example.com".into(), rest: b"example.com/2026/hindi-story".to_vec(), next: Scheduled { why: Why::Leave, due: 1600, at: 1000 } };
    store.note_outcomes(&[leave], &[]).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(1));
    assert_eq!(queue(&store, 100 * DAY).iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["2026/english-story"]);
    store.build_queue("example.com", store.terms("example.com"), &rules(), 2000, &mut |_| true).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(1), "a rebuild leaves it out too");
    store.record_listing("example.com", 7, vec![page("2026/hindi-story", 20)], 3000).unwrap();
    assert_eq!(store.set_size("example.com").unwrap(), Some(1), "and a newer lastmod does not bring it back");
}

#[test]
fn a_sitemap_read_over_two_domains_sends_each_page_to_its_own_domain() {
    let dir = Dir::new("two-domains");
    let store = BucketStore::open(&dir.0, &Resources::new(8 << 20, 8 << 20, 1)).unwrap();
    let terms = Terms { site: SiteType::Reference, ..Terms::default() };
    store.allow(Allowed::with_terms([("example.com", terms), ("example.net", terms)]));
    let other = |path: &str| (urls::parse(&format!("https://example.net/{path}"), "example.net").unwrap(), 10, false);
    store.record_listing("example.com", 7, vec![page("a", 10)], 100).unwrap();
    store.build_queue("example.com", store.terms("example.com"), &rules(), 1000, &mut |_| true).unwrap();
    store.record_listing("example.com", 7, vec![page("b", 10), other("x")], 1100).unwrap();
    assert_eq!(queue(&store, 1100).iter().map(|(path, _, _)| path.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(store.ready_count("example.net").unwrap(), 1, "example.net has no queue, so its page waits on its ready list");
    assert!(store.take_due("example.net", 10, 1100, None).unwrap().picks.is_empty());
}

//! On copies of real bucket stores: old send deadlines are deleted and never resend, old waiting entries still come back, and new schedules land when they should.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use desearch_bot::buckets::{Buckets, Resources};
use desearch_bot::outcomes::{OutcomeReader, OutcomeRow};
use desearch_bot::ready::{Outcome, Pick, Reason, Requeued, Why, REQUEUED};
use desearch_bot::recrawl::Recrawl;
use rocksdb::{IteratorMode, Options, ReadOptions, DB};

const DOMAINS: usize = 20;
const PER_LANE: usize = 100;

/// Keys of one type, and how many of them are due by `now` when the type is time-keyed.
fn count(dir: &Path, kind: u8, now: u32) -> (u64, u64) {
    let db = DB::open_for_read_only(&Options::default(), dir, false).unwrap();
    let mut options = ReadOptions::default();
    options.set_iterate_upper_bound([kind + 1]);
    let (mut all, mut due) = (0, 0);
    for item in db.iterator_opt(IteratorMode::From(&[kind], rocksdb::Direction::Forward), options) {
        let (key, _) = item.unwrap();
        all += 1;
        due += u64::from(key.get(1..5).is_some_and(|b| u32::from_be_bytes(b.try_into().unwrap()) <= now));
    }
    (all, due)
}

/// Ready entries by lane: changed, retries, new.
fn lanes(dir: &Path) -> [u64; 3] {
    let db = DB::open_for_read_only(&Options::default(), dir, false).unwrap();
    let mut options = ReadOptions::default();
    options.set_iterate_upper_bound([b'R' + 1]);
    let mut out = [0; 3];
    for item in db.iterator_opt(IteratorMode::From(b"R", rocksdb::Direction::Forward), options) {
        let (key, _) = item.unwrap();
        if let Some(lane) = key.iter().position(|&b| b == 0).and_then(|at| key.get(at + 1)) {
            out[usize::from(*lane).min(2)] += 1;
        }
    }
    out
}

fn counts(root: &Path, owned: &[usize], now: u32, when: &str) {
    for &bucket in owned {
        let dir = root.join(format!("{bucket:03}"));
        let ((x, x_due), (w, w_due), (n, _)) = (count(&dir, b'X', now), count(&dir, b'W', now), count(&dir, b'N', now));
        let [changed, retries, new] = lanes(&dir);
        println!("store {bucket:03} {when}: {x} send deadlines (X, {x_due} due), {w} waiting entries (W, {w_due} due), {n} schedules (N); ready {changed} changed, {retries} retries, {new} new");
    }
}

fn rows(picks: &[&Pick], outcome: Outcome, at: u32) -> Vec<OutcomeRow> {
    let host = |url: &str| url.split('/').nth(2).unwrap_or("").trim_start_matches("www.").to_string();
    picks.iter().map(|p| OutcomeRow { url: p.url.clone(), host: host(&p.url), outcome, at: i64::from(at) * 1_000_000 }).collect()
}

fn by_domain(picks: &[Pick]) -> BTreeMap<String, Vec<Pick>> {
    let mut groups: BTreeMap<String, Vec<Pick>> = BTreeMap::new();
    for pick in picks {
        groups.entry(pick.entry.domain.clone()).or_default().push(pick.clone());
    }
    groups
}

/// Every schedule due by `now`, in passes of the dispatcher's size.
fn drain(buckets: &Buckets, now: u32, recrawl: &Recrawl) -> (Requeued, f64) {
    let started = Instant::now();
    let mut total = Requeued::default();
    for store in buckets.stores() {
        loop {
            let done = store.requeue_due(now, 5_000, recrawl).unwrap();
            total.moved += done.moved;
            total.gone += done.gone;
            if done == Requeued::default() {
                break;
            }
        }
    }
    (total, started.elapsed().as_secs_f64())
}

/// Pages back on their ready lists as retries, with the failures counted so far.
fn requeued(buckets: &Buckets, pages: &[&Pick], retries: u8) -> usize {
    let sent = |p: &&&Pick| buckets.store(&p.entry.domain).sent(&p.entry.domain, &p.entry.rest).unwrap();
    pages.iter().filter(|p| sent(p).is_some_and(|s| s.requeued && s.retries == retries)).count()
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("directory holding copied stores named by bucket"));
    let mut owned: Vec<usize> = std::fs::read_dir(&root).unwrap().filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok()).collect();
    owned.sort_unstable();
    let now = desearch_bot::dispatch::now();
    counts(&root, &owned, now, "before");

    let resources = Resources::new(1 << 30, 256 << 20, owned.len());
    let buckets = Buckets::open(&root, &owned, &resources).unwrap();
    let started = Instant::now();
    let forgotten = buckets.stores().filter(|s| s.forget_expecting().unwrap()).count();
    println!("forget_expecting: {forgotten} stores had deadlines, {:.2} s", started.elapsed().as_secs_f64());

    let mut domains: Vec<(String, u64)> = buckets.stores().flat_map(|s| s.ready_domains().unwrap()).collect();
    domains.sort_by_key(|(_, ready)| std::cmp::Reverse(*ready));
    domains.truncate(DOMAINS);
    let daily = domains[0].0.clone();
    let recrawl = Recrawl::default().with_overrides(&format!("{daily} 1")).unwrap();
    let mut picks: Vec<Pick> = Vec::new();
    for (domain, _) in &domains {
        picks.extend(buckets.store(domain).take_ready(domain, [PER_LANE; 3]).unwrap());
    }
    for (domain, group) in by_domain(&picks) {
        buckets.store(&domain).mark_sent(&group, now, &recrawl).unwrap();
    }
    let undated = picks.iter().filter(|p| p.lastmod == 0).count();
    let wrong = picks
        .iter()
        .filter(|p| {
            let next = buckets.store(&p.entry.domain).next_send(&p.entry.domain, &p.entry.rest).unwrap().map(|n| (n.why, n.due));
            next != (p.lastmod == 0).then(|| (Why::Recrawl, now + recrawl.interval(&p.entry.domain)))
        })
        .count();
    println!("sent {} pages of {} domains at {now}: {undated} without a lastmod got a re-crawl, {} with one got none; {wrong} not as expected", picks.len(), domains.len(), picks.len() - undated);
    println!("  stores with send deadlines after the sends: {}", buckets.stores().filter(|s| s.forget_expecting().unwrap()).count());

    let failed: Vec<&Pick> = picks.iter().step_by(3).collect();
    let dropped: Vec<&Pick> = picks.iter().skip(1).step_by(3).step_by(3).collect();
    let taken: HashSet<&str> = failed.iter().chain(&dropped).map(|p| p.url.as_str()).collect();
    let published: Vec<&Pick> = picks.iter().filter(|p| !taken.contains(p.url.as_str())).collect();
    let mut reader = OutcomeReader::new(recrawl.clone(), 200_000);
    let at = now + 60;
    let file = [rows(&failed, Outcome::Failed, at), rows(&dropped, Outcome::Dropped, at), rows(&published, Outcome::Published, at)].concat();
    let started = Instant::now();
    let counted = reader.apply(&buckets, &file, at).unwrap();
    println!("outcomes {counted:?} in {:.3} s", started.elapsed().as_secs_f64());
    let off = |pages: &[&Pick], why: Why, due: &dyn Fn(&Pick) -> u32| {
        pages.iter().filter(|p| buckets.store(&p.entry.domain).next_send(&p.entry.domain, &p.entry.rest).unwrap().map(|n| (n.why, n.due)) != Some((why, due(p)))).count()
    };
    let failed_off = off(&failed, Why::Failed, &|p| at + recrawl.interval(&p.entry.domain));
    println!("  {} failed: {failed_off} not due at outcome + interval; {} dropped: {} not due now", failed.len(), dropped.len(), off(&dropped, Why::Dropped, &|_| at));

    let (first, took) = drain(&buckets, at, &recrawl);
    println!("requeue at the outcome time: {first:?} in {took:.1} s, old waiting entries due by now included");
    println!("  {} of {} dropped pages back on their lists; {} failed pages back early", requeued(&buckets, &dropped, 0), dropped.len(), requeued(&buckets, &failed, 1));

    let (daily_run, _) = drain(&buckets, now + 86_400, &recrawl);
    let daily_undated = published.iter().filter(|p| p.lastmod == 0 && p.entry.domain == daily).count();
    println!("a day later ({daily} re-crawls daily, {daily_undated} of its published pages have no lastmod): {daily_run:?}");

    let week = at + recrawl.every;
    let (weekly, _) = drain(&buckets, week, &recrawl);
    println!("a week after the outcomes: {weekly:?}; {} of {} failed pages back as retries", requeued(&buckets, &failed, 1), failed.len());

    let wanted: HashSet<(&str, &[u8])> = failed.iter().map(|p| (p.entry.domain.as_str(), p.entry.rest.as_slice())).collect();
    let mut quotas = [0; 3];
    quotas[REQUEUED] = 50_000;
    let mut retries = Vec::new();
    for (domain, _) in &domains {
        let taken = buckets.store(domain).take_ready(domain, quotas).unwrap();
        retries.extend(taken.into_iter().filter(|p| p.reason == Reason::Requeued && wanted.contains(&(p.entry.domain.as_str(), p.entry.rest.as_slice()))));
    }
    let resend = week + 100;
    for (domain, group) in by_domain(&retries) {
        buckets.store(&domain).mark_sent(&group, resend, &recrawl).unwrap();
    }
    let again: Vec<&Pick> = retries.iter().collect();
    reader.apply(&buckets, &rows(&again, Outcome::Failed, resend + 60), resend + 60).unwrap();
    let longest = again.iter().map(|p| recrawl.interval(&p.entry.domain)).max().unwrap_or(0);
    let (second, _) = drain(&buckets, resend + 60 + longest, &recrawl);
    let waiting = off(&again, Why::Gone, &|_| resend + 60 + recrawl.gone);
    println!("{} retries sent at {resend} failed again: {second:?}; {} not waiting to {}", again.len(), waiting, resend + 60 + recrawl.gone);
    let (gone_back, _) = drain(&buckets, resend + 60 + recrawl.gone, &recrawl);
    println!("after the long wait: {gone_back:?}; {} of them back as retries", requeued(&buckets, &again, 2));
    drop(buckets);
    counts(&root, &owned, now, "after");
}

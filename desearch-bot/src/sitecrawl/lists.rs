//! A URL list fetched: hosts a few at once, one request at a time each, same-host redirects followed.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use indexmap::{IndexMap, IndexSet};

use super::hosts::Host;
use super::links::{host_of, normalize};
use super::robots::Rules;
use super::scope;
use super::sites::{cookie_round_trip, fetch_page, try_join_all, Ctx, MAX_HOPS};
use super::store;
use super::{Counts, Crash};

/// The list's URLs by host in the order first named, after normalizing and the skip and set checks.
pub fn group_urls<S: AsRef<str>>(
    urls: impl IntoIterator<Item = S>,
    skip: &HashSet<String>,
    admit: impl Fn(&str) -> bool,
) -> (Counts, IndexMap<String, Vec<String>>) {
    let mut counts = Counts::default();
    let mut by_host: IndexMap<String, Vec<String>> = IndexMap::new();
    for raw in urls {
        match normalize(raw.as_ref(), "") {
            None => counts.add("bad_url", 1),
            Some(url) if skip.contains(&url) => counts.add("skipped_held", 1),
            Some(url) if !admit(&url) => counts.add("out_of_set", 1),
            Some(url) => by_host.entry(host_of(&url)).or_default().push(url),
        }
    }
    (counts, by_host)
}

struct HostRun {
    ctx: Arc<Ctx>,
    name: String,
    rules: Arc<Rules>,
    host: Host,
    source: String,
    counts: Arc<Mutex<Counts>>,
    left: Mutex<VecDeque<String>>,
    done: Mutex<HashSet<String>>,
    tripped: Mutex<HashSet<String>>,
}

impl HostRun {
    fn count(&self, key: &str) {
        self.counts.lock().unwrap().add(key, 1);
    }

    async fn worker(&self) -> Result<(), Crash> {
        loop {
            let next = {
                let mut left = self.left.lock().unwrap();
                if left.is_empty() || self.host.stopped().is_some() {
                    return Ok(());
                }
                left.pop_front()
            };
            if let Some(url) = next {
                self.one_url(url).await?;
            }
        }
    }

    async fn one_url(&self, mut url: String) -> Result<(), Crash> {
        for _ in 0..=MAX_HOPS {
            let repeat = self.done.lock().unwrap().contains(&url);
            if repeat || !self.rules.allowed(&url) {
                self.count(if repeat { "repeat" } else { "robots_disallowed" });
                break;
            }
            self.done.lock().unwrap().insert(url.clone());
            let f = fetch_page(&self.ctx.http, &self.host, &url).await;
            if f.outcome != "redirect" {
                self.count(&f.outcome);
                let lang = f.doc.as_ref().map(scope::declared_lang).unwrap_or_default();
                if !lang.is_empty() && !scope::wanted_lang(&lang) {
                    self.count("other_language");
                } else if f.outcome == "page" && self.ctx.admit(&url) {
                    let line = store::record_line(&f.response, &f.html, &self.source);
                    self.ctx.writer.lock().unwrap().add(&line).map_err(|_| Crash("OSError"))?;
                }
                break;
            }
            if let Some(location) = f.location.as_deref().filter(|l| host_of(l) == self.name) {
                self.count("redirect");
                url = location.to_string();
                continue;
            }
            let mut back = None;
            if let Some(location) = f.location.as_deref() {
                if !self.tripped.lock().unwrap().contains(&url) {
                    back = cookie_round_trip(&self.ctx, &url, location).await?;
                }
            }
            let Some(back) = back.filter(|b| !b.is_empty()) else {
                self.count("redirect_offsite");
                break;
            };
            self.tripped.lock().unwrap().insert(url.clone());
            self.count("cookie_round_trip");
            if back == url {
                self.done.lock().unwrap().remove(&url);
            }
            url = back;
        }
        Ok(())
    }
}

async fn one_host(ctx: Arc<Ctx>, name: String, todo: Vec<String>, source: String, counts: Arc<Mutex<Counts>>) -> Result<(), Crash> {
    let rules = ctx.robots.rules(&name).await?;
    if let Some(why) = &rules.refused {
        counts.lock().unwrap().add(why, todo.len() as i64);
        return Ok(());
    }
    let host = Host::new(&name, rules.delay, None, Some(ctx.min_delay));
    let left: IndexSet<String> = todo.into_iter().collect();
    let run = HostRun {
        ctx,
        name,
        rules,
        host,
        source,
        counts: counts.clone(),
        left: Mutex::new(left.into_iter().collect()),
        done: Mutex::new(HashSet::new()),
        tripped: Mutex::new(HashSet::new()),
    };
    try_join_all((0..run.host.slots).map(|_| run.worker()).collect()).await?;
    if let Some(why) = run.host.stopped() {
        counts.lock().unwrap().add(&why, 1);
    }
    Ok(())
}

/// Every URL of the list fetched into the run's writer, `hosts_at_once` hosts at a time in the order they come; counts say what became of each.
pub async fn crawl_urls<S: AsRef<str>>(ctx: &Arc<Ctx>, urls: impl IntoIterator<Item = S>, hosts_at_once: usize, source: &str) -> anyhow::Result<Counts> {
    if hosts_at_once == 0 {
        anyhow::bail!("hosts_at_once must be at least 1");
    }
    let (first, by_host) = group_urls(urls, &ctx.skip, |url| ctx.admit(url));
    let counts = Arc::new(Mutex::new(first));
    let gate = Arc::new(tokio::sync::Semaphore::new(hosts_at_once));
    let mut tasks = tokio::task::JoinSet::new();
    for (name, todo) in by_host {
        let turn = gate.clone().acquire_owned().await?;
        while let Some(done) = tasks.try_join_next() {
            done?;
        }
        let (ctx, counts, source) = (ctx.clone(), counts.clone(), source.to_string());
        tasks.spawn(async move {
            let _turn = turn;
            if let Err(crash) = one_host(ctx, name, todo, source, counts.clone()).await {
                counts.lock().unwrap().add(&format!("crashed_{}", crash.0), 1);
            }
        });
    }
    while let Some(done) = tasks.join_next().await {
        done?;
    }
    let counts = counts.lock().unwrap().clone();
    Ok(counts)
}

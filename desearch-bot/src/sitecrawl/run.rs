//! A run of site crawls as the Python crawler's `run_sites` drives one: hosts from a hosts file, a few at once, one record folder.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use indexmap::{IndexMap, IndexSet};

use super::fetch::{Connect, Http};
use super::hosts::MIN_DELAY;
use super::robots::{Robots, RETRY_PAUSE};
use super::sites::{self, Ctx, Site};
use super::store::{counts_json, Writer};
use super::Counts;
use crate::pystr;
use crate::setrule::SetRule;

pub struct Options {
    pub hosts: Vec<String>,
    pub hosts_file: Option<PathBuf>,
    pub out: PathBuf,
    pub seed: Vec<String>,
    pub prefix: Vec<String>,
    pub max_pages: i64,
    pub max_depth: i64,
    pub no_sitemaps: bool,
    pub hosts_at_once: usize,
    pub skip: Option<PathBuf>,
    pub set: Option<PathBuf>,
    pub found: Option<PathBuf>,
    pub seeds_file: Option<PathBuf>,
    pub connect: Connect,
    pub min_delay: f64,
    pub retry_pause: Duration,
}

impl Options {
    pub fn new(out: PathBuf) -> Self {
        Options {
            hosts: Vec::new(),
            hosts_file: None,
            out,
            seed: Vec::new(),
            prefix: Vec::new(),
            max_pages: 1_000,
            max_depth: 6,
            no_sitemaps: false,
            hosts_at_once: 8,
            skip: None,
            set: None,
            found: None,
            seeds_file: None,
            connect: Connect::default(),
            min_delay: MIN_DELAY,
            retry_pause: RETRY_PAUSE,
        }
    }
}

/// A file's non-empty lines in order, stripped, duplicates dropped; none for no file.
fn read_lines(path: Option<&PathBuf>) -> Result<Vec<String>> {
    let Some(path) = path else { return Ok(Vec::new()) };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let lines: IndexSet<String> = pystr::splitlines(&text).into_iter().map(|l| pystr::strip(l).to_string()).filter(|l| !l.is_empty()).collect();
    Ok(lines.into_iter().collect())
}

/// Python's `int(text)` for decimal text.
fn py_int(text: &str) -> Result<i64> {
    let t = pystr::strip(text);
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    let b = digits.as_bytes();
    let ok = !b.is_empty()
        && b[0].is_ascii_digit()
        && b[b.len() - 1].is_ascii_digit()
        && b.iter().all(|c| c.is_ascii_digit() || *c == b'_')
        && !digits.contains("__");
    let value: i64 = if ok {
        digits.replace('_', "").parse().map_err(|_| anyhow!("invalid literal for int(): {text:?}"))?
    } else {
        bail!("invalid literal for int(): {text:?}")
    };
    Ok(if t.starts_with('-') { -value } else { value })
}

fn host_line(host: &str, counts: &Counts) -> String {
    let rest = counts_json(&counts.0);
    let name = serde_json::to_string(host).expect("a string serializes");
    if counts.0.is_empty() {
        format!("{{\"host\": {name}}}")
    } else {
        format!("{{\"host\": {name}, {}", &rest[1..])
    }
}

/// Every host's crawl into one record folder; `progress` gets each host's counts line as it finishes.
pub async fn run_sites(o: &Options, progress: Arc<dyn Fn(&str) + Send + Sync>) -> Result<Counts> {
    if o.hosts_at_once == 0 {
        bail!("--hosts-at-once must be at least 1");
    }
    let writer = Arc::new(Mutex::new(Writer::new(&o.out)?));
    let skip = Arc::new(match &o.skip {
        Some(path) => crate::pagebuild::read_urls(path)?,
        None => HashSet::new(),
    });
    let set = o.set.as_deref().map(SetRule::load).transpose()?.map(Arc::new);
    let file_lines = read_lines(o.hosts_file.as_ref())?;
    let mut terms: HashMap<String, Vec<String>> = HashMap::new();
    for line in &file_lines {
        let mut cols = line.split('\t').map(str::to_string);
        let name = cols.next().unwrap_or_default();
        terms.insert(name, cols.collect());
    }
    let mut caps = HashMap::new();
    let mut slots = HashMap::new();
    let mut near = HashMap::new();
    for (host, t) in &terms {
        if let Some(cap) = t.first().filter(|c| !c.is_empty()) {
            caps.insert(host.clone(), py_int(cap)?);
        }
        if let Some(n) = t.get(1).filter(|c| !c.is_empty()) {
            slots.insert(host.clone(), usize::try_from(py_int(n)?).map_err(|_| anyhow!("{host}: slots below zero"))?);
        }
        if let Some(paths) = t.get(2).filter(|c| !c.is_empty()) {
            near.insert(host.clone(), pystr::split_ws(paths).map(str::to_string).collect::<Vec<_>>());
        }
    }
    let hosts: IndexSet<String> = o.hosts.iter().cloned().chain(file_lines.iter().map(|l| l.split('\t').next().unwrap_or("").to_string())).collect();
    let mut seeds: HashMap<String, Vec<String>> = HashMap::new();
    for url in read_lines(o.seeds_file.as_ref())? {
        let host = crate::urls::split(&url).ok().and_then(|p| p.hostname()).unwrap_or_default();
        seeds.entry(host).or_default().push(url);
    }
    let http = Arc::new(Http::new(&o.connect)?);
    let mut robots = Robots::new(http.clone(), "https");
    robots.retry_pause = o.retry_pause;
    let ctx = Arc::new(Ctx { http, robots: Arc::new(robots), writer: writer.clone(), skip, set, min_delay: o.min_delay });
    let gate = Arc::new(tokio::sync::Semaphore::new(o.hosts_at_once));
    let total = Arc::new(Mutex::new(Counts::default()));
    let found = Arc::new(Mutex::new(Vec::new()));
    let mut tasks = tokio::task::JoinSet::new();
    let finished = |done: Result<Result<()>, tokio::task::JoinError>| done.map_err(anyhow::Error::from).and_then(|result| result);
    for name in hosts {
        // Hosts take their turns in list order, as asyncio's semaphore hands them out.
        let turn = gate.clone().acquire_owned().await.expect("the gate stays open");
        while let Some(done) = tasks.try_join_next() {
            finished(done)?;
        }
        let (ctx, total, found, progress) = (ctx.clone(), total.clone(), found.clone(), progress.clone());
        let base = Site {
            prefixes: o.prefix.clone(),
            max_pages: caps.get(&name).copied().unwrap_or(o.max_pages),
            slots: slots.get(&name).copied(),
            near: near.get(&name).cloned().unwrap_or_default(),
            max_depth: o.max_depth,
            sitemaps: !o.no_sitemaps,
            ..Site::new(&name)
        };
        let (extra, seeds) = (o.seed.clone(), seeds.clone());
        tasks.spawn(async move {
            let _turn = turn;
            let host = sites::home_host(&ctx, &name).await.map_err(|crash| anyhow!("{name}: {} before its crawl began", crash.0))?;
            let paths = ctx.set.as_ref().map(|set| set.seed_paths(&host)).unwrap_or_default();
            let mut site = Site { host: host.clone(), ..base };
            site.seeds =
                extra.into_iter().chain(paths.iter().map(|p| format!("https://{host}{p}"))).chain(seeds.get(&host).cloned().unwrap_or_default()).collect();
            let counts = match sites::crawl(&ctx, &site, Some(&*found)).await {
                Ok(counts) => counts,
                Err(crash) => Counts(IndexMap::from([(format!("crashed_{}", crash.0), 1)])),
            };
            progress(&host_line(&host, &counts));
            total.lock().unwrap().update(&counts);
            Ok(())
        });
    }
    while let Some(done) = tasks.join_next().await {
        finished(done)?;
    }
    writer.lock().unwrap().close()?;
    if let Some(path) = &o.found {
        let found = found.lock().unwrap();
        std::fs::write(path, found.iter().map(|u| format!("{u}\n")).collect::<String>())?;
    }
    let total = total.lock().unwrap().clone();
    Ok(total)
}

/// The line `python -m crawler site` prints at the end.
pub fn total_line(total: &Counts) -> String {
    format!("{{\"total\": {}}}", counts_json(&total.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_and_lines_as_python_reads_them() {
        assert_eq!(py_int(" 12 ").unwrap(), 12);
        assert_eq!(py_int("1_000").unwrap(), 1000);
        assert_eq!(py_int("-3").unwrap(), -3);
        assert!(py_int("1.5").is_err() && py_int("x").is_err() && py_int("1__0").is_err() && py_int("_1").is_err());
        let mut counts = Counts::default();
        assert_eq!(host_line("a.test", &counts), "{\"host\": \"a.test\"}");
        counts.add("page", 2);
        counts.add("left_in_queue", 0);
        assert_eq!(host_line("a.test", &counts), "{\"host\": \"a.test\", \"page\": 2, \"left_in_queue\": 0}");
        assert_eq!(total_line(&counts), "{\"total\": {\"page\": 2, \"left_in_queue\": 0}}");
    }
}

//! `bulk bittensor-fetch`: chain, registration dates, sites and GitHub, for the subnet builds.

use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::worker::{PythonArgs, Worker, ANCHORS};
use bulk_core::{py, pyjson, Counts};
use desearch::chain::Chain;
use serde_json::{json, Map, Value};

pub mod chain;
pub mod github;
pub mod http;
pub mod links;
pub mod robots;
pub mod sites;
pub mod yarl;

pub use chain::ChainError;

pub const USER_AGENT: &str = "DesearchBulk/1.0 (+https://desearch.ai)";
pub const CHAIN: &str = "subnets-chain.json";
pub const SITES: &str = "sites";
pub const GITHUB: &str = "github";
pub const REMOVED: &str = "removed.json";
pub const FETCH: &str = "fetch.json";

#[derive(clap::Args, Clone, Debug)]
pub struct FetchArgs {
    /// A new folder for this fetch.
    #[arg(long)]
    pub out: PathBuf,
    /// An earlier fetch whose pages are named removed when gone; by default the newest one beside --out.
    #[arg(long)]
    pub previous: Option<PathBuf>,
    /// Read the chain at this block instead of the best one.
    #[arg(long)]
    pub block: Option<i64>,
    #[command(flatten)]
    pub python: PythonArgs,
    #[arg(long, default_value = chain::NETWORK, hide = true)]
    pub chain: String,
    /// A `chain_snapshot` saved as JSON, read instead of the chain.
    #[arg(long, hide = true)]
    pub chain_snapshot: Option<PathBuf>,
    /// Stop after the chain and its registration dates.
    #[arg(long, hide = true)]
    pub only_chain: bool,
    #[arg(long, hide = true)]
    pub no_delay: bool,
    #[arg(long, hide = true)]
    pub test_resolve_all: Option<SocketAddr>,
    #[arg(long, hide = true)]
    pub test_ca_cert: Option<PathBuf>,
}

/// Whether the crawl waits between requests; tests skip the waits.
#[derive(Clone, Copy, Debug)]
pub struct Pace {
    pub waits: bool,
}

impl Pace {
    pub async fn sleep(self, seconds: f64) {
        if self.waits && seconds > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
        }
    }
}

/// The `a` elements' hrefs as lxml parses a home page, from one `python -m desearch.extraction.worker` started when first needed.
pub struct Anchors {
    spawn: Box<dyn Fn() -> Result<Worker> + Send + Sync>,
    worker: tokio::sync::Mutex<Option<Worker>>,
}

impl Anchors {
    pub fn new(spawn: Box<dyn Fn() -> Result<Worker> + Send + Sync>) -> Anchors {
        Anchors { spawn, worker: tokio::sync::Mutex::new(None) }
    }

    /// The hrefs in document order, or the `repr` of the ValueError or TypeError lxml raised; other errors end the fetch.
    pub async fn hrefs(&self, body: &[u8]) -> Result<std::result::Result<Vec<Option<String>>, String>> {
        let mut guard = self.worker.lock().await;
        let answer = tokio::task::block_in_place(|| -> Result<Value> {
            if guard.is_none() {
                *guard = Some((self.spawn)()?);
            }
            guard.as_mut().expect("a worker").ask_json(&[ANCHORS, body])
        })?;
        if let Some(raised) = answer.get("raised") {
            let repr = raised.get("repr").and_then(Value::as_str).unwrap_or("").to_string();
            if raised.get("caught").and_then(Value::as_bool) == Some(true) {
                return Ok(Err(repr));
            }
            bail!("lxml failed on a home page: {repr}");
        }
        let hrefs = answer.get("hrefs").and_then(Value::as_array).ok_or_else(|| anyhow!("an anchors answer without hrefs"))?;
        Ok(Ok(hrefs.iter().map(|h| h.as_str().map(str::to_string)).collect()))
    }
}

/// What a fetch reads and where it writes.
pub struct Options {
    pub out: PathBuf,
    pub previous: Option<PathBuf>,
    pub block: Option<i64>,
    pub chain: String,
    pub chain_snapshot: Option<PathBuf>,
    pub only_chain: bool,
    pub pace: Pace,
    pub route: http::TestRoute,
    pub token: String,
}

/// `Path.resolve()`: absolute, with symlinks resolved as far as the path exists.
pub fn resolve(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let mut existing = absolute.clone();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    while std::fs::symlink_metadata(&existing).is_err() {
        let Some(name) = existing.file_name().map(|n| n.to_os_string()) else { break };
        rest.push(name);
        existing.pop();
    }
    let mut out = std::fs::canonicalize(&existing).unwrap_or(existing);
    for name in rest.into_iter().rev() {
        match Path::new(&name).components().next() {
            Some(Component::ParentDir) => {
                out.pop();
            }
            Some(Component::CurDir) => {}
            _ => out.push(name),
        }
    }
    Ok(out)
}

/// `previous_fetch`: the newest earlier fetch beside this one, as the refresh runner lays out `work/<source>/<run>/src`.
pub fn previous_fetch(out: &Path) -> Result<Option<PathBuf>> {
    let resolved = resolve(out)?;
    let (Some(name), Some(grand)) = (resolved.file_name(), resolved.parent().and_then(Path::parent)) else { return Ok(None) };
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let Ok(runs) = std::fs::read_dir(grand) else { return Ok(None) };
    for run in runs.flatten() {
        let candidate = run.path().join(name);
        let marker = candidate.join(FETCH);
        let Ok(meta) = std::fs::metadata(&marker) else { continue };
        if !meta.is_file() || candidate == resolved {
            continue;
        }
        let modified = meta.modified()?;
        if best.as_ref().is_none_or(|(at, _)| modified > *at) {
            best = Some((modified, candidate));
        }
    }
    Ok(best.map(|(_, path)| path))
}

fn read_json(path: &Path, default: Value) -> Result<Value> {
    if path.exists() {
        py::load_json(path)
    } else {
        Ok(default)
    }
}

/// `fetch`: the whole Bittensor fetch into `out`, naming what an earlier fetch had and is now gone.
pub async fn fetch(options: &Options, anchors: Arc<Anchors>) -> Result<Counts> {
    let out = &options.out;
    if out.join(CHAIN).exists() {
        bail!("FileExistsError: {} already holds a fetch; give each fetch its own folder", bulk_core::run::path_str(out));
    }
    let previous = match &options.previous {
        Some(path) => Some(path.clone()),
        None => previous_fetch(out)?,
    };
    std::fs::create_dir_all(out)?;
    let http = Arc::new(http::Http::new(USER_AGENT, &options.route)?);
    let client = Chain::with_client(&options.chain, http.client(), false)?;
    let mut counts = Counts::default();
    let mut found: Map<String, Value> = match &options.chain_snapshot {
        Some(path) => match py::load_json(path)? {
            Value::Object(map) => map,
            _ => bail!("{} is not a chain snapshot", path.display()),
        },
        None => chain::snapshot(&client, options.block).await?,
    };
    found.remove("endpoint");
    let subnets: Vec<Value> = found.get("subnets").and_then(Value::as_array).cloned().unwrap_or_default();
    counts.add("subnets", subnets.len() as u64);
    counts.add("identities", subnets.iter().filter(|s| py::truthy(s.get("identity"))).count() as u64);
    let before = match &previous {
        Some(previous) => read_json(&previous.join(CHAIN), json!({}))?.get("subnets").cloned().unwrap_or(json!([])),
        None => json!([]),
    };
    let now: std::collections::HashSet<String> = subnets.iter().filter_map(|s| s.get("netuid").map(Value::to_string)).collect();
    let mut gone: Vec<Value> = Vec::new();
    for subnet in py::list(&before, "the previous subnets")? {
        let netuid = py::item(subnet, "netuid")?;
        if !now.contains(&netuid.to_string()) && !gone.contains(netuid) {
            gone.push(netuid.clone());
        }
    }
    gone.sort_by(|a, b| a.as_f64().partial_cmp(&b.as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    counts.add("subnets_removed", gone.len() as u64);
    found.insert("removed_netuids".into(), Value::Array(gone));
    chain::registration_dates(&client, &mut found, &mut counts, options.pace).await?;
    if !options.only_chain {
        let subnets: Vec<Value> = found.get("subnets").and_then(Value::as_array).cloned().unwrap_or_default();
        let sites = match &previous {
            Some(previous) => read_json(&previous.join(SITES).join("sites.json"), json!([]))?,
            None => json!([]),
        };
        let repos = match &previous {
            Some(previous) => read_json(&previous.join(GITHUB).join("github.json"), json!([]))?,
            None => json!([]),
        };
        sites::fetch_sites(http.clone(), &subnets, &out.join(SITES), &mut counts, py::list(&sites, "sites.json")?, anchors, options.pace).await?;
        github::fetch_github(&http, &subnets, &out.join(GITHUB), &options.token, &mut counts, py::list(&repos, "github.json")?, options.pace).await?;
    }
    std::fs::write(out.join(CHAIN), pyjson::dumps(&Value::Object(found), Some(1)))?;
    let mut summary: Map<String, Value> = counts.iter().map(|(k, n)| (k.to_string(), json!(n))).collect();
    summary.insert("previous".into(), previous.map_or(Value::Null, |p| json!(bulk_core::run::path_str(&p))));
    std::fs::write(out.join(FETCH), pyjson::dumps(&Value::Object(summary), Some(1)))?;
    Ok(counts)
}

/// `bulk bittensor-fetch`: the fetch, then its counts on stdout; a short chain read exits 2.
pub fn main(args: FetchArgs) -> Result<()> {
    let python_path = args.python.python_path();
    let python = args.python.python.clone();
    let anchors = Arc::new(Anchors::new(Box::new(move || Worker::spawn(&python, &python_path, &[]))));
    let options = Options {
        out: args.out.clone(),
        previous: args.previous.clone(),
        block: args.block,
        chain: args.chain.clone(),
        chain_snapshot: args.chain_snapshot.clone(),
        only_chain: args.only_chain,
        pace: Pace { waits: !args.no_delay },
        route: http::TestRoute { resolve_all: args.test_resolve_all, ca_cert: args.test_ca_cert.clone() },
        token: std::env::var("GITHUB_TOKEN").unwrap_or_default(),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().context("starting the runtime")?;
    match runtime.block_on(fetch(&options, anchors)) {
        Ok(counts) => {
            let mut line: Map<String, Value> = Map::new();
            line.insert("out".into(), json!(bulk_core::run::path_str(&args.out)));
            for (key, n) in counts.iter() {
                line.insert(key.into(), json!(n));
            }
            println!("{}", pyjson::dumps(&Value::Object(line), None));
            Ok(())
        }
        Err(error) => match error.downcast_ref::<ChainError>() {
            Some(short) => {
                eprintln!("chain read failed: {short}");
                std::process::exit(2);
            }
            None => Err(error),
        },
    }
}

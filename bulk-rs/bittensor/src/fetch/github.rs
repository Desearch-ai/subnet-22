//! `GitHub` and `fetch_github`: each listed repository's metadata and README through the REST API, waiting out rate limits.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::{py, pyjson, Counts};
use regex::Regex;
use serde_json::{json, Map, Value};

use super::http::{text_of, Http};
use super::sites::GONE;
use super::Pace;

pub const GITHUB_API: &str = "https://api.github.com";
/// Unauthenticated GitHub allows 60 calls an hour; waits for a reset stop after this long.
pub const MAX_RATE_WAIT: f64 = 5.0 * 3600.0;
pub const REPO_FIELDS: [&str; 14] = [
    "full_name",
    "html_url",
    "description",
    "homepage",
    "topics",
    "stargazers_count",
    "forks_count",
    "language",
    "license",
    "pushed_at",
    "created_at",
    "archived",
    "fork",
    "default_branch",
];

static REPO: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)").expect("pattern compiles"));
static DOT_GIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\.git\n?\z").expect("pattern compiles"));

/// Python's `float(text)` of a header.
fn py_float(text: &str) -> Result<f64> {
    let value = desearch::text::strip(text);
    value.replace('_', "").parse::<f64>().map_err(|_| anyhow!("ValueError: could not convert string to float: {}", py::repr_str(value)))
}

pub struct GitHub<'a> {
    http: &'a Http,
    token: String,
    pub waited: f64,
    pace: Pace,
}

impl GitHub<'_> {
    pub fn new<'a>(http: &'a Http, token: &str, pace: Pace) -> GitHub<'a> {
        GitHub { http, token: token.to_string(), waited: 0.0, pace }
    }

    /// `GitHub.get`: the status and body text, waiting out a rate limit while the total wait allows.
    pub async fn get(&mut self, path: &str, accept: &str) -> Result<(u16, String)> {
        loop {
            let auth = format!("Bearer {}", self.token);
            let mut headers = vec![("X-GitHub-Api-Version", "2022-11-28")];
            if !self.token.is_empty() {
                headers.push(("Authorization", auth.as_str()));
            }
            headers.push(("Accept", accept));
            let got = self.http.get(&format!("{GITHUB_API}/{path}"), &headers).await.map_err(|e| anyhow!("{} on {path}", e.repr))?;
            let body = text_of(&got)?;
            let header = |name: &str| got.headers.get(name).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned());
            let limited = (got.status == 403 || got.status == 429) && header("x-ratelimit-remaining").as_deref() == Some("0");
            let reset = match header("x-ratelimit-reset").filter(|v| !v.is_empty()) {
                Some(text) => py_float(&text)?,
                None => 0.0,
            };
            if !limited {
                return Ok((got.status, body));
            }
            let wait = (reset - desearch::time::now() + 5.0).max(1.0);
            if self.waited + wait > MAX_RATE_WAIT {
                return Ok((got.status, body));
            }
            self.waited += wait;
            self.pace.sleep(wait).await;
        }
    }
}

/// `fetch_github`: github.json, the README files and removed.json in `folder`, with the GitHub counts.
pub async fn fetch_github(http: &Http, subnets: &[Value], folder: &Path, token: &str, counts: &mut Counts, previous: &[Value], pace: Pace) -> Result<()> {
    std::fs::create_dir_all(folder)?;
    let mut api = GitHub::new(http, token, pace);
    let mut before: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    for record in previous {
        if let Some(repo) = py::when_truthy(py::get(record, "repo")?) {
            before.insert(py::item(record, "netuid")?.to_string(), py::item(repo, "html_url")?.clone());
        }
    }
    let mut found: Vec<Map<String, Value>> = Vec::new();
    let mut removed: Vec<Value> = Vec::new();
    for subnet in subnets {
        let identity = py::dict_or_empty(py::get(subnet, "identity")?, "identity")?;
        let listed = match py::when_truthy(identity.get("github_repo")) {
            Some(v) => desearch::text::strip(py::str_of(v, "github_repo")?).to_string(),
            None => String::new(),
        };
        if listed.is_empty() {
            continue;
        }
        let netuid = py::item(subnet, "netuid")?.clone();
        let mut record = Map::new();
        record.insert("netuid".into(), netuid.clone());
        record.insert("listed".into(), json!(listed));
        let Some(caps) = REPO.captures(&listed).filter(|c| &c[1] != "orgs") else {
            record.insert("error".into(), json!("not a repo url"));
            found.push(record);
            continue;
        };
        let owner = caps[1].to_string();
        let name = DOT_GIT.replace(&caps[2], "").into_owned();
        let (status, body) = api.get(&format!("repos/{owner}/{name}"), "application/vnd.github+json").await?;
        if status != 200 {
            record.insert("error".into(), json!(format!("repo: HTTP {status}")));
            if status == 404 || status == 410 {
                let mut urls: Vec<Value> = vec![json!(format!("https://github.com/{owner}/{name}"))];
                if let Some(earlier) = before.get(&netuid.to_string()) {
                    urls.push(earlier.clone());
                }
                let mut kept: Vec<Value> = Vec::new();
                for url in urls {
                    if py::truthy(Some(&url)) && !kept.contains(&url) {
                        kept.push(url);
                    }
                }
                removed.extend(kept.into_iter().map(|url| json!({"url": url, "why": GONE})));
            }
            found.push(record);
            continue;
        }
        let meta: Value = serde_json::from_str(&body).context("a repository's JSON")?;
        let Value::Object(fields) = &meta else { bail!("AttributeError: the repository JSON is not an object") };
        let repo: Map<String, Value> = REPO_FIELDS.iter().map(|k| (k.to_string(), fields.get(*k).cloned().unwrap_or(Value::Null))).collect();
        record.insert("repo".into(), Value::Object(repo));
        let full_name = py::format(fields.get("full_name").ok_or_else(|| anyhow!("KeyError: 'full_name'"))?)?;
        let (status, body) = api.get(&format!("repos/{full_name}/readme"), "application/vnd.github.html").await?;
        if status != 200 {
            record.insert("readme_error".into(), json!(format!("HTTP {status}")));
            found.push(record);
            continue;
        }
        let file = format!("{}.html", full_name.replace('/', "__"));
        std::fs::write(folder.join(&file), body.as_bytes())?;
        record.insert("readme_file".into(), json!(file));
        found.push(record);
    }
    let records: Vec<Value> = found.iter().cloned().map(Value::Object).collect();
    std::fs::write(folder.join("github.json"), pyjson::dumps(&Value::Array(records), Some(1)))?;
    std::fs::write(folder.join(super::REMOVED), pyjson::dumps(&Value::Array(removed.clone()), Some(1)))?;
    counts.set("github_removed_gone", removed.len() as u64);
    counts.set("github_listed", found.len() as u64);
    counts.set("github_repos", found.iter().filter(|r| py::truthy(r.get("repo"))).count() as u64);
    counts.set("github_readmes", found.iter().filter(|r| py::truthy(r.get("readme_file"))).count() as u64);
    counts.set("github_rate_wait_s", api.waited.round_ties_even() as u64);
    Ok(())
}

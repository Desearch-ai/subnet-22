//! One refresh run: fetch, build, diff each build, publish through the outbox, hide what is gone, report.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::diff::{self, Counts, Refused};
use crate::sources::{Source, PYTHON, REPO};

pub const OUTBOX: &str = "outbox";
pub const RECEIPTS: &str = "receipts";
pub const READY: &str = ".ready";
pub const STATUS: &str = "status.jsonl";
pub const HIDE_SSH: &str = "refresh-hide";
pub const WEBHOOK_ENV: &str = "REFRESH_DISCORD_WEBHOOK";
const USER_AGENT: &str = "DesearchRefresh/1.0 (+https://desearch.ai)";
const PUBLISH_TIMEOUT: Duration = Duration::from_secs(4 * 3600);

pub fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(path),
    }
}

/// UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn iso(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let (h, m, s) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

/// One argument filled in; a path with a wildcard becomes the files it matches.
pub fn expand(part: &str, values: &HashMap<&str, String>) -> Vec<String> {
    let mut filled = part.to_string();
    for (key, value) in values {
        filled = filled.replace(&format!("{{{key}}}"), value);
    }
    if filled.starts_with('/') && filled.contains(['*', '?', '[']) {
        let mut found: Vec<String> =
            glob::glob(&filled).map(|paths| paths.filter_map(|p| p.ok()).map(|p| p.display().to_string()).collect()).unwrap_or_default();
        found.sort();
        if !found.is_empty() {
            return found;
        }
    }
    vec![filled]
}

pub fn run_command(argv: &[&str], values: &HashMap<&str, String>, log: &Path, source: &Source) -> Result<()> {
    let mut values = values.clone();
    values.insert("python", expand_home(if source.python.is_empty() { PYTHON } else { source.python }).display().to_string());
    let command: Vec<String> = argv.iter().flat_map(|part| expand(part, &values)).collect();
    let cwd = expand_home(if source.cwd.is_empty() { REPO } else { source.cwd });
    let mut out = OpenOptions::new().create(true).append(true).open(log)?;
    writeln!(out, "$ {}", command.join(" "))?;
    let limit = source.memory_bytes;
    let mut cmd = Command::new(&command[0]);
    cmd.args(&command[1..]).current_dir(&cwd).stdout(out.try_clone()?).stderr(out);
    if limit > 0 {
        unsafe {
            cmd.pre_exec(move || {
                let cap = libc::rlimit { rlim_cur: limit as libc::rlim_t, rlim_max: limit as libc::rlim_t };
                if libc::setrlimit(libc::RLIMIT_AS, &cap) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let mut child = cmd.spawn().with_context(|| format!("starting {}", command[0]))?;
    let deadline = Instant::now() + Duration::from_secs(source.timeout_s);
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!("{} exited with {status}", command.iter().take(5).cloned().collect::<Vec<_>>().join(" "));
            }
            return Ok(());
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            bail!("{} ran past {} h", command.iter().take(5).cloned().collect::<Vec<_>>().join(" "), source.timeout_s / 3600);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Puts a run folder in the outbox the planner box pulls from and waits for its receipt; returns the publish line.
pub fn publish(folder: &Path, root: &Path, poll: Duration) -> Result<String> {
    let (outbox, receipts) = (root.join(OUTBOX), root.join(RECEIPTS));
    std::fs::create_dir_all(&outbox)?;
    std::fs::create_dir_all(&receipts)?;
    let name = folder.file_name().context("a run folder without a name")?.to_string_lossy().to_string();
    std::fs::rename(folder, outbox.join(&name))?;
    std::fs::write(outbox.join(format!("{name}{READY}")), "")?;
    let receipt = receipts.join(format!("{name}.txt"));
    let deadline = Instant::now() + PUBLISH_TIMEOUT;
    while !receipt.exists() {
        if Instant::now() > deadline {
            bail!("{name}: no receipt from the planner box in {} h", PUBLISH_TIMEOUT.as_secs() / 3600);
        }
        std::thread::sleep(poll);
    }
    let text = std::fs::read_to_string(&receipt)?;
    let lines: Vec<&str> = text.lines().collect();
    if !lines.contains(&"rc=0") {
        bail!("publishing {name} failed: {}", lines.iter().rev().take(5).rev().cloned().collect::<Vec<_>>().join(" | "));
    }
    Ok(lines.iter().rev().find(|l| l.contains("bulk files")).or(lines.last()).unwrap_or(&"").to_string())
}

/// Sends the source's whole list of removed URLs to index-1, which rewrites the source's section to exactly them.
pub fn hide(source: &str, urls: &[String]) -> Result<()> {
    let mut child = Command::new("ssh")
        .args(["-o", "BatchMode=yes", HIDE_SSH, source])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().context("ssh stdin")?.write_all(urls.iter().map(|u| format!("{u}\n")).collect::<String>().as_bytes())?;
    let done = child.wait_with_output()?;
    if !done.status.success() {
        bail!("hiding {source}'s removed pages failed: {}", String::from_utf8_lossy(&done.stderr).trim());
    }
    Ok(())
}

/// The as-of time of the source's last run that published, or "".
pub fn last_ok(root: &Path, source: &str) -> String {
    let Ok(file) = std::fs::File::open(root.join(STATUS)) else { return String::new() };
    let mut last = String::new();
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(entry) = serde_json::from_str::<Value>(&line) else { continue };
        if entry["source"] == source && entry["ok"] == true && entry["dry_run"] != true {
            let as_of = entry["as_of"].as_str().unwrap_or_default();
            if as_of > last.as_str() {
                last = as_of.to_string();
            }
        }
    }
    last
}

pub struct Hooks<'a> {
    pub publish: &'a dyn Fn(&Path, &Path) -> Result<String>,
    pub hide: &'a dyn Fn(&str, &[String]) -> Result<()>,
}

pub fn refresh(source: &Source, root: &Path, dry_run: bool, now: u64, since: &str, reuse: Option<&Path>, hooks: &Hooks) -> Result<Value> {
    let as_of = iso(now);
    let run = format!("{}{}{}-{}{}", &as_of[0..4], &as_of[5..7], &as_of[8..10], &as_of[11..13], &as_of[14..16]);
    let since = if since.is_empty() { last_ok(root, source.name).chars().take(10).collect() } else { since.to_string() };
    let needs_since = source.fetch.iter().copied().chain(source.builds.iter().map(|b| b.argv)).flatten().any(|p| p.contains("{since}"));
    if since.is_empty() && needs_since {
        return Err(Refused(format!("{} has no run that published yet; give its first run --since", source.name)).into());
    }
    let work = match reuse {
        Some(dir) => dir.to_path_buf(),
        None => root.join("work").join(source.name).join(&run),
    };
    if reuse.is_none() {
        if work.exists() {
            bail!("{} already exists", work.display());
        }
        std::fs::create_dir_all(&work)?;
    }
    let log = work.join("log.txt");
    let mut values: HashMap<&str, String> = HashMap::new();
    values.insert("work", work.display().to_string());
    values.insert("as_of", as_of.clone());
    values.insert("run", run.clone());
    values.insert("since", since.clone());
    values.insert("until", as_of[..10].to_string());
    if reuse.is_none() {
        for argv in source.fetch {
            run_command(argv, &values, &log, source)?;
        }
    }
    let mut con = diff::open_state(&root.join("state").join(format!("{}.sqlite", source.name)))?;
    let mut diffs: Vec<(usize, Counts, PathBuf)> = Vec::new();
    std::fs::create_dir_all(work.join("build"))?;
    for (i, build) in source.builds.iter().enumerate() {
        let folder = work.join("build").join(build.name);
        let mut with_out = values.clone();
        with_out.insert("out", folder.display().to_string());
        if reuse.is_none() {
            run_command(build.argv, &with_out, &log, source)?;
        } else if !folder.join(diff::MANIFEST).exists() {
            bail!("{} holds no finished {} build to reuse", work.display(), build.name);
        }
        let out = work.join("publish").join(format!("refresh-{}-{run}", build.name));
        let counts = diff::diff(&con, build, &folder, &out, &format!("{}-{run}", build.name), &as_of)?;
        diffs.push((i, counts, out));
    }
    let mut builds = serde_json::Map::new();
    for (i, mut counts, out) in diffs {
        let build = &source.builds[i];
        if !dry_run {
            if counts.new + counts.changed > 0 {
                counts.published = (hooks.publish)(&out, root)?;
            }
            diff::keep(&mut con, build, &counts, &run, &as_of)?;
        }
        builds.insert(build.name.to_string(), serde_json::to_value(&counts)?);
    }
    let hidden = diff::hidden_urls(&con)?;
    if !dry_run {
        (hooks.hide)(source.name, &hidden)?;
    }
    Ok(json!({
        "source": source.name, "run": run, "as_of": as_of, "since": since, "dry_run": dry_run,
        "builds": builds, "hidden": hidden.len(),
    }))
}

/// The Discord line for a run: what each build published, or why the run stopped.
pub fn summary(report: &Value) -> String {
    let head = format!("refresh {} {}", report["source"].as_str().unwrap_or_default(), report["run"].as_str().unwrap_or_default());
    let head = head.trim_end();
    let seconds = report["seconds"].as_u64().unwrap_or_default();
    if report["ok"] != true {
        let kind = if report["refused"] == true { "refused" } else { "failed" };
        let error: String = report["error"].as_str().unwrap_or_default().chars().take(600).collect();
        return format!("⚠️ {head} {kind} after {seconds} s: {error}");
    }
    let mut parts = Vec::new();
    if let Some(builds) = report["builds"].as_object() {
        for (name, c) in builds {
            let sent = c["published"].as_str().filter(|s| !s.is_empty()).map(|s| format!(", {s}")).unwrap_or_default();
            let flag = if c["large"] == true { " ⚠️ over 30% changed" } else { "" };
            parts.push(format!("{name} {} new, {} changed, {} removed of {}{sent}{flag}", c["new"], c["changed"], c["removed"], c["seen"]));
        }
    }
    let dry = if report["dry_run"] == true { " (dry run)" } else { "" };
    format!("{head}{dry}: {}; {} hidden; {seconds} s", parts.join("; "), report["hidden"].as_u64().unwrap_or_default())
}

pub fn notify(report: &Value) {
    let Ok(hook) = std::env::var(WEBHOOK_ENV) else { return };
    if hook.is_empty() {
        return;
    }
    let content: String = summary(report).chars().take(1900).collect();
    let sent = ureq::post(&hook).set("User-Agent", USER_AGENT).timeout(Duration::from_secs(20)).send_json(json!({ "content": content }));
    if let Err(e) = sent {
        eprintln!("Discord post failed: {e}");
    }
}

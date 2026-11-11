//! `refresh`: keeps bulk sources current, and rewrites a source's hidden-URL section on index-1.

mod diff;
mod hidden;
mod run;
mod sources;

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;

const HOME_ENV: &str = "REFRESH_HOME";
const DEFAULT_HOME: &str = "~/refresh";
const RECEIPT_POLL: Duration = Duration::from_secs(30);

#[derive(Parser)]
#[command(name = "refresh", about = "Keeps bulk sources current")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Refresh one source now.
    Run {
        source: String,
        /// Fetch, build and diff, but publish and hide nothing.
        #[arg(long)]
        dry_run: bool,
        /// First day to fetch, YYYY-MM-DD; the last published run's day by default.
        #[arg(long, default_value = "")]
        since: String,
    },
    /// Record runs already published, before a source's first refresh.
    Seed { source: String, build: String, folders: Vec<PathBuf> },
    /// Diff one build folder against the source's state into OUT, recording nothing.
    Diff { source: String, build: String, folder: PathBuf, out: PathBuf },
    /// On index-1: replace a source's section of the hidden-URLs file with the URLs on stdin.
    HideSection {
        source: Option<String>,
        #[arg(long, default_value = hidden::HIDDEN)]
        file: PathBuf,
    },
}

fn home() -> PathBuf {
    run::expand_home(&std::env::var(HOME_ENV).unwrap_or_else(|_| DEFAULT_HOME.to_string()))
}

fn source(name: &str) -> Result<&'static sources::Source> {
    sources::find(name).with_context(|| {
        format!("no source {name}; there are {}", sources::SOURCES.iter().map(|s| s.name).collect::<Vec<_>>().join(", "))
    })
}

fn build<'a>(source: &'a sources::Source, name: &str) -> Result<&'a sources::Build> {
    source.builds.iter().find(|b| b.name == name).with_context(|| {
        format!("{} has no build {name}; it has {}", source.name, source.builds.iter().map(|b| b.name).collect::<Vec<_>>().join(", "))
    })
}

fn lock(root: &std::path::Path, name: &str) -> Result<std::fs::File> {
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(root.join(format!("{name}.lock")))?;
    use std::os::fd::AsRawFd;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        bail!("locking {name}: {}", std::io::Error::last_os_error());
    }
    Ok(file)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = home();
    match cli.command {
        Cmd::HideSection { source, file } => {
            let name = std::env::var("SSH_ORIGINAL_COMMAND").ok().or(source).unwrap_or_default();
            let name = name.trim();
            if !hidden::valid_name(name) {
                eprintln!("not a source name: {name:?}");
                std::process::exit(2);
            }
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            println!("{}", hidden::hide_section(name, &file, &input)?);
        }
        Cmd::Seed { source: name, build: build_name, folders } => {
            let source = source(&name)?;
            let build = build(source, &build_name)?;
            let mut con = diff::open_state(&root.join("state").join(format!("{}.sqlite", source.name)))?;
            println!("{}/{}: {} pages recorded", source.name, build.name, diff::seed(&mut con, build, &folders)?);
        }
        Cmd::Diff { source: name, build: build_name, folder, out } => {
            let source = source(&name)?;
            let build = build(source, &build_name)?;
            let con = diff::open_state(&root.join("state").join(format!("{}.sqlite", source.name)))?;
            let counts = diff::diff(&con, build, &folder, &out, "check", &run::iso(run::now_secs()))?;
            println!("{}", serde_json::to_string(&counts)?);
        }
        Cmd::Run { source: name, dry_run, since } => {
            let source = source(&name)?;
            std::fs::create_dir_all(&root)?;
            let started = Instant::now();
            let _lock = lock(&root, source.name)?;
            let hooks = run::Hooks { publish: &|folder, root| run::publish(folder, root, RECEIPT_POLL), hide: &run::hide };
            let mut report = match run::refresh(source, &root, dry_run, run::now_secs(), &since, &hooks) {
                Ok(mut report) => {
                    report["ok"] = json!(true);
                    report
                }
                Err(e) => json!({
                    "source": source.name, "ok": false, "dry_run": dry_run,
                    "refused": e.downcast_ref::<diff::Refused>().is_some(),
                    "error": format!("{e:#}").chars().take(1000).collect::<String>(),
                }),
            };
            report["seconds"] = json!(started.elapsed().as_secs());
            report["finished"] = json!(run::iso(run::now_secs()));
            let mut status = OpenOptions::new().create(true).append(true).open(root.join(run::STATUS))?;
            writeln!(status, "{}", serde_json::to_string(&report)?)?;
            run::notify(&report);
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report["ok"] != true {
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

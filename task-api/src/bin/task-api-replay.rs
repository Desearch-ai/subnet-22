//! Replays the task API's log from a folder laid out as the uploads bucket keeps it (`seq/<n>.json`, `blobs/<sha256>`), going on from
//! the state of the last run, and prints every output the API's rules do not give as one JSON line, then a summary line.
//!
//! `task-api-replay --dir <folder> --state <file> --signer <ss58> [--signer <ss58>…] [--join] [--anchors <file>]`, a signer for
//! each holder of the log, and the anchors a validator's own node shows (`[{"seq", "head", "block"}, …]`, those whose head the
//! log holds), so a round committed again although anchored in time shows even when the log leaves that anchor out. Exits 2 when
//! the log's links or signatures break. `task-api-replay --version` prints the version of the rules it replays with.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use serde_json::json;
use task_api::replay::{replay_dir, Replay};

struct Args {
    dir: PathBuf,
    state: Option<PathBuf>,
    signers: Vec<String>,
    join: bool,
    anchors: Option<PathBuf>,
}

fn args() -> Result<Args> {
    let (mut dir, mut state, mut signers, mut join, mut anchors) = (None, None, Vec::new(), false, None);
    let mut given = std::env::args().skip(1);
    while let Some(name) = given.next() {
        match name.as_str() {
            "--dir" => dir = given.next().map(PathBuf::from),
            "--state" => state = given.next().map(PathBuf::from),
            // Once per holder of the log, as the custody chain lists them.
            "--signer" => signers.extend(given.next().into_iter().flat_map(|s| s.split(',').map(String::from).collect::<Vec<_>>())),
            "--join" => join = true,
            "--anchors" => anchors = given.next().map(PathBuf::from),
            other => bail!("unknown argument {other:?}"),
        }
    }
    if signers.is_empty() {
        bail!("--signer is required");
    }
    Ok(Args { dir: dir.context("--dir is required")?, state, signers, join, anchors })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("task-api-replay: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode> {
    if std::env::args().nth(1).as_deref() == Some("--version") {
        println!("task-api-replay {}", env!("CARGO_PKG_VERSION"));
        return Ok(ExitCode::SUCCESS);
    }
    let args = args()?;
    let mut replay = Replay::open(args.state.as_deref())?;
    if let Some(path) = &args.anchors {
        let listed: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)?;
        for anchor in listed {
            let (Some(seq), Some(block)) = (anchor["seq"].as_i64(), anchor["block"].as_i64()) else { bail!("an anchor without seq and block: {anchor}") };
            replay.note_anchor(seq, block)?;
        }
    }
    let outcome = replay_dir(&mut replay, &args.dir, &args.signers, args.join)?;
    for divergence in &replay.divergences {
        println!("{}", divergence.to_json());
    }
    let (last, head) = replay.position()?;
    let fraud = replay.fraud().count();
    match outcome {
        Ok(files) => {
            println!(
                "{}",
                json!({"replayed_files": files, "last": last, "head": head, "fraud": fraud, "started": replay.started()?, "since": replay.since()?})
            );
            Ok(ExitCode::SUCCESS)
        }
        Err(broken) => {
            println!("{}", json!({"severity": "fraud", "seq": last + 1, "kind": "chain", "what": broken.to_string(), "api": null, "replay": null}));
            Ok(ExitCode::from(2))
        }
    }
}

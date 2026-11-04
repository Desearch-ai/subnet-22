//! Replays the task API's log from a folder laid out as the uploads bucket keeps it (`seq/<n>.json`, `blobs/<sha256>`), going on from
//! the state of the last run, and prints every output the API's rules do not give as one JSON line, then a summary line.
//!
//! `task-api-replay --dir <folder> --state <file> --signer <ss58> [--join]`; exits 2 when the log's links or signatures break.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use serde_json::json;
use task_api::replay::{replay_dir, Replay};

struct Args {
    dir: PathBuf,
    state: Option<PathBuf>,
    signer: String,
    join: bool,
}

fn args() -> Result<Args> {
    let (mut dir, mut state, mut signer, mut join) = (None, None, None, false);
    let mut given = std::env::args().skip(1);
    while let Some(name) = given.next() {
        match name.as_str() {
            "--dir" => dir = given.next().map(PathBuf::from),
            "--state" => state = given.next().map(PathBuf::from),
            "--signer" => signer = given.next(),
            "--join" => join = true,
            other => bail!("unknown argument {other:?}"),
        }
    }
    Ok(Args { dir: dir.context("--dir is required")?, state, signer: signer.context("--signer is required")?, join })
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
    let args = args()?;
    let mut replay = Replay::open(args.state.as_deref())?;
    let outcome = replay_dir(&mut replay, &args.dir, &args.signer, args.join)?;
    for divergence in &replay.divergences {
        println!("{}", divergence.to_json());
    }
    let (last, head) = replay.position()?;
    let fraud = replay.fraud().count();
    match outcome {
        Ok(files) => {
            println!("{}", json!({"replayed_files": files, "last": last, "head": head, "fraud": fraud, "started": replay.started()?}));
            Ok(ExitCode::SUCCESS)
        }
        Err(broken) => {
            println!("{}", json!({"severity": "fraud", "seq": last + 1, "kind": "chain", "what": broken.to_string(), "api": null, "replay": null}));
            Ok(ExitCode::from(2))
        }
    }
}

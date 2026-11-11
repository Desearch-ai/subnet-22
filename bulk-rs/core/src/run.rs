//! The flags every source command takes, and the start and summary line every run shares.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::counts::Counts;
use crate::pyjson;
use crate::rows::{parse_as_of, utc_format, Batches, Summary};

#[derive(clap::Args, Clone, Debug)]
pub struct RunArgs {
    /// A new folder for this run's batch files.
    #[arg(long)]
    pub out: PathBuf,
    /// When the source's data was taken, as an ISO time with a zone; the pages' fetch time.
    #[arg(long = "as-of")]
    pub as_of: String,
    /// The run's name in its task id, bulk-<source>-<run>; the as-of date by default.
    #[arg(long, default_value = "")]
    pub run: String,
    /// Stop after this many pages, for trials.
    #[arg(long, default_value_t = 0)]
    pub limit: u64,
}

/// The run's batches and its as-of time in microseconds, as `python -m bulk` opens them for `source`.
pub fn start(args: &RunArgs, source: &str) -> Result<(Batches, i64)> {
    let as_of_us = parse_as_of(&args.as_of)?;
    let run = if args.run.is_empty() { utc_format(as_of_us, "%Y%m%d") } else { args.run.clone() };
    Ok((Batches::new(&args.out, source, &run, as_of_us)?, as_of_us))
}

/// Closes the run and prints its summary line exactly as `python -m bulk` prints it.
pub fn finish(args: &RunArgs, batches: Batches, counts: &Counts) -> Result<()> {
    let summary = batches.close(counts)?;
    println!("{}", summary_line(&args.out, &summary, counts));
    Ok(())
}

/// `json.dumps({"out", "files", "rows", "removed", **counts})`, `out` as `str(Path(out))` spells it.
pub fn summary_line(out: &Path, summary: &Summary, counts: &Counts) -> String {
    let mut line = Map::new();
    line.insert("out".into(), json!(path_str(out)));
    line.insert("files".into(), json!(summary.files));
    line.insert("rows".into(), json!(summary.rows));
    line.insert("removed".into(), json!(summary.removed));
    for (key, n) in counts.iter() {
        line.insert(key.into(), json!(n));
    }
    pyjson::dumps(&Value::Object(line), None)
}

/// `str(pathlib.Path(path))` on POSIX: empty and `.` parts dropped, a trailing slash dropped, `//` kept only as a two-slash root.
pub fn path_str(path: &Path) -> String {
    let text = path.to_string_lossy();
    let root = if text.starts_with("//") && !text.starts_with("///") {
        "//"
    } else if text.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = text.split('/').filter(|part| !part.is_empty() && *part != ".").collect();
    if root.is_empty() && parts.is_empty() {
        return ".".into();
    }
    format!("{root}{}", parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_as_pathlib() {
        for (given, want) in
            [("out", "out"), ("./out/", "out"), ("a//b/./c", "a/b/c"), ("", "."), ("/", "/"), ("//x", "//x"), ("///x/", "/x"), ("../x", "../x")]
        {
            assert_eq!(path_str(Path::new(given)), want, "{given}");
        }
    }
}

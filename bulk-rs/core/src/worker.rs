//! Python worker processes that answer one length-framed request at a time, for the work that stays in shared Python.

use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::rows::Page;

/// The shared extraction's worker: `python -m bulk.page_worker`.
pub const PAGE_WORKER: &str = "bulk.page_worker";
/// The largest answer a worker may send; a longer length means the framing broke.
const MAX_ANSWER: usize = 1 << 30;

#[derive(clap::Args, Clone, Debug)]
pub struct PythonArgs {
    /// The Python that runs the workers.
    #[arg(long, default_value = "python3")]
    pub python: String,
    /// The repo root, put on the workers' PYTHONPATH; found above the binary, else the current directory.
    #[arg(long = "python-path")]
    pub python_path: Option<PathBuf>,
}

impl PythonArgs {
    pub fn python_path(&self, module: &str) -> PathBuf {
        self.python_path.clone().unwrap_or_else(|| default_python_path(module))
    }
}

/// The nearest folder above the binary holding the module's file, else the current directory.
pub fn default_python_path(module: &str) -> PathBuf {
    let file = format!("{}.py", module.replace('.', "/"));
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().find(|dir| dir.join(&file).is_file()).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub struct Worker {
    module: String,
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    pub fn spawn(python: &str, python_path: &Path, module: &str, args: &[OsString]) -> Result<Self> {
        let mut path = OsString::from(python_path);
        if let Some(existing) = std::env::var_os("PYTHONPATH").filter(|p| !p.is_empty()) {
            path.push(":");
            path.push(existing);
        }
        let mut child = Command::new(python)
            .args(["-m", module])
            .args(args)
            .env("PYTHONPATH", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("starting {python} -m {module}"))?;
        let stdin = child.stdin.take().map(BufWriter::new);
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no worker stdout"))?);
        Ok(Worker { module: module.to_string(), child, stdin, stdout })
    }

    /// One answer; on any failure the worker is killed, since its framing can no longer be trusted.
    pub fn ask(&mut self, parts: &[&[u8]]) -> Result<Vec<u8>> {
        let answer = self.exchange(parts);
        if answer.is_err() {
            let _ = self.child.kill();
        }
        answer.with_context(|| format!("{} worker stopped; check --python and --python-path", self.module))
    }

    fn exchange(&mut self, parts: &[&[u8]]) -> Result<Vec<u8>> {
        let size = u32::try_from(parts.iter().map(|p| p.len()).sum::<usize>()).context("a request over 4 GiB")?;
        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("worker closed"))?;
        stdin.write_all(&size.to_be_bytes())?;
        for part in parts {
            stdin.write_all(part)?;
        }
        stdin.flush()?;
        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header)?;
        let size = u32::from_be_bytes(header) as usize;
        if size > MAX_ANSWER {
            bail!("a {size}-byte answer");
        }
        let mut data = vec![0u8; size];
        self.stdout.read_exact(&mut data)?;
        Ok(data)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// `desearch.extraction.Page` from `dataclasses.asdict(page)`.
pub fn page_from_json(value: &Value) -> Result<Page> {
    let text = |key: &str| -> Result<String> {
        match value.get(key) {
            Some(Value::String(s)) => Ok(s.clone()),
            other => bail!("page field {key} is {other:?}"),
        }
    };
    let list = |key: &str| -> Result<Vec<String>> {
        match value.get(key) {
            Some(Value::Array(items)) => {
                items.iter().map(|item| item.as_str().map(str::to_string).ok_or_else(|| anyhow!("page field {key} holds {item}"))).collect()
            }
            other => bail!("page field {key} is {other:?}"),
        }
    };
    Ok(Page {
        page_type: text("page_type")?,
        text_mode: text("text_mode")?,
        title: text("title")?,
        description: text("description")?,
        lang: text("lang")?,
        canonical: text("canonical")?,
        published: text("published")?,
        modified: text("modified")?,
        author: text("author")?,
        site_name: text("site_name")?,
        image: text("image")?,
        section: text("section")?,
        tags: list("tags")?,
        breadcrumbs: list("breadcrumbs")?,
        json_ld_types: list("json_ld_types")?,
        headings: list("headings")?,
        entity: text("entity")?,
        ld: text("ld")?,
        paywalled: match value.get("paywalled") {
            Some(Value::Bool(b)) => Some(*b),
            Some(Value::Null) => None,
            other => bail!("page field paywalled is {other:?}"),
        },
        robots: list("robots")?,
        text: text("text")?,
    })
}

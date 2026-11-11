//! The `python -m bulk.pmc_render` workers that render article abstracts and bodies with the shared HTML renderer.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use bulk_core::Line;
use serde::Deserialize;

pub const MODULE: &str = "bulk.pmc_render";

#[derive(clap::Args, Clone, Debug)]
pub struct RenderArgs {
    /// The Python that runs the render workers (`python -m bulk.pmc_render`).
    #[arg(long, default_value = "python3")]
    pub python: String,
    /// The repo root, put on the workers' PYTHONPATH; found above the binary, else the current directory.
    #[arg(long = "python-path")]
    pub python_path: Option<PathBuf>,
    /// Render worker processes; the CPU count by default.
    #[arg(long = "render-workers")]
    pub render_workers: Option<usize>,
}

impl RenderArgs {
    pub fn python_path(&self) -> PathBuf {
        self.python_path.clone().unwrap_or_else(default_python_path)
    }

    pub fn workers(&self) -> usize {
        self.render_workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())).max(1)
    }
}

/// The nearest folder above the binary holding `bulk/pmc_render.py`, else the current directory.
pub fn default_python_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().find(|dir| dir.join("bulk/pmc_render.py").is_file()).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The parts `article_page` renders: each abstract it keeps, in order, and the body when there is one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    pub abstracts: Vec<Vec<Line>>,
    pub body: Option<Vec<Line>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rendered {
    /// lxml could not parse the article.
    BadXml,
    /// An XMLSyntaxError while rendering, which `article_page` meets only after its front-matter checks.
    RenderFailed,
    Parts(Parts),
}

type WireLine = (String, bool, bool, bool);

#[derive(Deserialize)]
struct Answer {
    error: Option<String>,
    detail: Option<String>,
    #[serde(default)]
    abstracts: Vec<Vec<WireLine>>,
    body: Option<Vec<WireLine>>,
}

fn lines(wire: Vec<WireLine>) -> Vec<Line> {
    wire.into_iter().map(|(text, linked, verbatim, row)| Line { text, linked, verbatim, row }).collect()
}

struct Worker {
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    fn spawn(python: &str, python_path: &Path) -> Result<Self> {
        let mut path = std::ffi::OsString::from(python_path);
        if let Some(existing) = std::env::var_os("PYTHONPATH").filter(|p| !p.is_empty()) {
            path.push(":");
            path.push(existing);
        }
        let mut child = Command::new(python)
            .args(["-m", MODULE])
            .env("PYTHONPATH", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("starting {python} -m {MODULE}"))?;
        let stdin = child.stdin.take().map(BufWriter::new);
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no worker stdout"))?);
        Ok(Worker { child, stdin, stdout })
    }

    fn ask(&mut self, xml: &[u8]) -> Result<Answer> {
        let size = u32::try_from(xml.len()).context("an article over 4 GiB")?;
        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("worker closed"))?;
        stdin.write_all(&size.to_be_bytes())?;
        stdin.write_all(xml)?;
        stdin.flush()?;
        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header)?;
        let mut data = vec![0u8; u32::from_be_bytes(header) as usize];
        self.stdout.read_exact(&mut data)?;
        Ok(serde_json::from_slice(&data)?)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// A pool of render workers shared by the render threads.
pub struct Renderer {
    idle: Sender<Worker>,
    waiting: Mutex<Receiver<Worker>>,
    broken: AtomicBool,
    workers: usize,
}

impl Renderer {
    pub fn start(args: &RenderArgs) -> Result<Self> {
        Self::spawn(&args.python, &args.python_path(), args.workers())
    }

    pub fn spawn(python: &str, python_path: &Path, workers: usize) -> Result<Self> {
        let (idle, waiting) = channel();
        for _ in 0..workers.max(1) {
            idle.send(Worker::spawn(python, python_path)?).expect("the pool holds its receiver");
        }
        Ok(Renderer { idle, waiting: Mutex::new(waiting), broken: AtomicBool::new(false), workers: workers.max(1) })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    fn take(&self) -> Result<Worker> {
        loop {
            if self.broken.load(Ordering::Relaxed) {
                bail!("a render worker failed");
            }
            let waiting = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
            match waiting.recv_timeout(Duration::from_millis(200)) {
                Ok(worker) => return Ok(worker),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => bail!("no render workers"),
            }
        }
    }

    /// One article's rendered parts.
    pub fn render(&self, xml: &[u8]) -> Result<Rendered> {
        let mut worker = self.take()?;
        let answer = match worker.ask(xml) {
            Ok(answer) => answer,
            Err(error) => {
                self.broken.store(true, Ordering::Relaxed);
                return Err(error.context(format!("{MODULE} worker stopped; check --python and --python-path")));
            }
        };
        let _ = self.idle.send(worker);
        match answer.error.as_deref() {
            None => Ok(Rendered::Parts(Parts { abstracts: answer.abstracts.into_iter().map(lines).collect(), body: answer.body.map(lines) })),
            Some("xml") => Ok(Rendered::BadXml),
            Some("render") => Ok(Rendered::RenderFailed),
            Some(_) => bail!("{MODULE} failed:\n{}", answer.detail.unwrap_or_default()),
        }
    }
}

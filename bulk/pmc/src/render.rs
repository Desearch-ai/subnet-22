//! PMC renders in the shared worker's `p` requests: article abstracts and bodies with the shared HTML renderer.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{bail, Result};
use bulk_core::worker::{PythonArgs, Worker, MODULE, PMC};
use bulk_core::Line;
use serde::Deserialize;

#[derive(clap::Args, Clone, Debug)]
pub struct RenderArgs {
    #[command(flatten)]
    pub python: PythonArgs,
    /// Render worker processes; the CPU count by default.
    #[arg(long = "render-workers")]
    pub render_workers: Option<usize>,
}

impl RenderArgs {
    pub fn workers(&self) -> usize {
        self.render_workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())).max(1)
    }
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

/// A pool of render workers shared by the render threads.
pub struct Renderer {
    idle: Sender<Worker>,
    waiting: Mutex<Receiver<Worker>>,
    broken: AtomicBool,
    workers: usize,
}

impl Renderer {
    pub fn start(args: &RenderArgs) -> Result<Self> {
        Self::spawn(&args.python.python, &args.python.python_path(), args.workers())
    }

    pub fn spawn(python: &str, python_path: &Path, workers: usize) -> Result<Self> {
        let (idle, waiting) = channel();
        for _ in 0..workers.max(1) {
            idle.send(Worker::spawn(python, python_path, &[])?).expect("the pool holds its receiver");
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
        let answer = match worker.ask(&[PMC, xml]).and_then(|data| Ok(serde_json::from_slice::<Answer>(&data)?)) {
            Ok(answer) => answer,
            Err(error) => {
                self.broken.store(true, Ordering::Relaxed);
                return Err(error);
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

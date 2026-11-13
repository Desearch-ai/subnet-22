//! The shared extraction's Python worker for the builders: the `--python` flags, and items spread over several workers.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{anyhow, bail, Result};
pub use desearch::worker::{default_python_path, Worker, ANCHORS, BLOCKED, BODY, EXTRACT, MODULE, MODULE_FILE, PMC, WIKIPEDIA};
use serde_json::Value;

use crate::rows::Page;

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
    pub fn python_path(&self) -> PathBuf {
        self.python_path.clone().unwrap_or_else(default_python_path)
    }
}

/// Each item's result in item order, from up to `workers` workers that each take the next item when free.
pub fn map_in_order<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    spawn: &(dyn Fn() -> Result<Worker> + Sync),
    each: &(dyn Fn(&mut Worker, &T) -> Result<R> + Sync),
) -> Result<Vec<R>> {
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let threads = workers.max(1).min(items.len());
    let done: Vec<Result<Vec<(usize, R)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| -> Result<Vec<(usize, R)>> {
                    let mut worker = spawn().inspect_err(|_| failed.store(true, Ordering::Relaxed))?;
                    let mut done = Vec::new();
                    while !failed.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        match each(&mut worker, item) {
                            Ok(result) => done.push((i, result)),
                            Err(error) => {
                                failed.store(true, Ordering::Relaxed);
                                return Err(error);
                            }
                        }
                    }
                    Ok(done)
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap_or_else(|_| Err(anyhow!("a worker thread panicked")))).collect()
    });
    let mut results: Vec<Option<R>> = std::iter::repeat_with(|| None).take(items.len()).collect();
    for found in done {
        for (i, result) in found? {
            results[i] = Some(result);
        }
    }
    results.into_iter().map(|r| r.ok_or_else(|| anyhow!("an item without a result"))).collect()
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

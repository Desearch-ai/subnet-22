//! Uploads and change files in local folders, for tests, benchmarks and replays.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::reading::{Counted, LocalFile, RangeRead, Traffic};
use crate::worker::{ChangeFeed, Fault, Job, Uploads};

/// Each job's upload as `<task_id>.parquet` in one folder.
pub struct LocalUploads {
    pub dir: PathBuf,
    pub traffic: Arc<Traffic>,
}

impl LocalUploads {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        LocalUploads { dir: dir.into(), traffic: Arc::default() }
    }
}

impl Uploads for LocalUploads {
    fn open(&self, job: &Job) -> Result<Box<dyn RangeRead>, Fault> {
        let path = self.dir.join(format!("{}.parquet", job.task_id));
        match LocalFile::open(&path) {
            Ok(file) => Ok(Box::new(Counted { inner: file, traffic: self.traffic.clone() })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(Fault::Gone("expired")),
            Err(error) => Err(Fault::Retry(format!("{}: {error}", path.display()))),
        }
    }
}

/// Change files numbered from `first`, written as `<seq>.parquet`.
pub struct LocalFeed {
    pub dir: PathBuf,
    next: AtomicU64,
}

impl LocalFeed {
    pub fn new(dir: impl Into<PathBuf>, first: u64) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(LocalFeed { dir, next: AtomicU64::new(first) })
    }
}

impl ChangeFeed for LocalFeed {
    fn append(&self, file: Vec<u8>, _rows: usize) -> Result<u64> {
        let seq = self.next.fetch_add(1, Ordering::SeqCst);
        let path = self.dir.join(format!("{seq:012}.parquet"));
        std::fs::write(&path, file).with_context(|| format!("writing {}", path.display()))?;
        Ok(seq)
    }
}

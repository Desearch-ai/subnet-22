//! SQLite on threads of its own: one writer that runs every change in order, and one read-only reader for the log pages.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{anyhow, Result};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, Row};
use serde_json::{Map, Value};
use tokio::sync::oneshot;

use crate::{budgets, checks, embeddings, roundlog, roundstore, validations};

pub const FILE: &str = "task_api.db";
/// Log reads queued at once before more are turned away.
const MAX_WAITING: usize = 32;

type Job = Box<dyn FnOnce(&mut Connection) + Send>;

fn connect(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    // Receipts are promises to miners, so every commit is synced in full.
    conn.pragma_update(None, "synchronous", "FULL")?;
    Ok(conn)
}

fn spawn(name: &str, mut conn: Connection) -> Result<mpsc::Sender<Job>> {
    let (jobs, inbox) = mpsc::channel::<Job>();
    std::thread::Builder::new().name(name.into()).spawn(move || {
        for job in inbox {
            job(&mut conn);
        }
    })?;
    Ok(jobs)
}

#[derive(Clone)]
pub struct Db {
    jobs: mpsc::Sender<Job>,
}

impl Db {
    pub fn open(path: &Path) -> Result<Db> {
        let conn = connect(path)?;
        budgets::create(&conn)?;
        roundlog::create(&conn)?;
        roundstore::create(&conn)?;
        validations::create(&conn)?;
        checks::create(&conn)?;
        embeddings::create(&conn)?;
        Ok(Db { jobs: spawn("sqlite", conn)? })
    }

    /// Runs `work` in one transaction on the writer thread: all of it lands, or none of it.
    pub async fn run<T: Send + 'static>(&self, work: impl FnOnce(&Connection) -> Result<T> + Send + 'static) -> Result<T> {
        let (done, result) = oneshot::channel();
        let job: Job = Box::new(move |conn| {
            let outcome = conn.transaction().map_err(anyhow::Error::from).and_then(|tx| {
                let value = work(&tx)?;
                tx.commit()?;
                Ok(value)
            });
            let _ = done.send(outcome);
        });
        self.jobs.send(job).map_err(|_| anyhow!("the SQLite writer stopped"))?;
        result.await.map_err(|_| anyhow!("the SQLite writer stopped"))?
    }
}

/// The log reader is busy; the caller is asked to come back.
#[derive(Debug)]
pub struct Busy;

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the log reader is busy")
    }
}

impl std::error::Error for Busy {}

/// Log reads have a connection and a thread of their own, so they never hold up a claim.
pub struct Reader {
    jobs: mpsc::Sender<Job>,
    waiting: Arc<AtomicUsize>,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Reader> {
        let conn = connect(path)?;
        conn.pragma_update(None, "query_only", "ON")?;
        Ok(Reader { jobs: spawn("logs", conn)?, waiting: Arc::default() })
    }

    pub async fn read<T: Send + 'static>(&self, work: impl FnOnce(&Connection) -> Result<T> + Send + 'static) -> Result<T> {
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= MAX_WAITING {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(Busy.into());
        }
        let waiting = self.waiting.clone();
        let (done, result) = oneshot::channel();
        let job: Job = Box::new(move |conn| {
            let _ = done.send(work(conn));
            waiting.fetch_sub(1, Ordering::SeqCst);
        });
        if self.jobs.send(job).is_err() {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(anyhow!("the SQLite reader stopped"));
        }
        result.await.map_err(|_| anyhow!("the SQLite reader stopped"))?
    }
}

/// A column as Python's sqlite3 hands it over: int, float, str or None.
pub fn json_of(value: ValueRef) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(n) => n.into(),
        ValueRef::Real(x) => x.into(),
        ValueRef::Text(text) => String::from_utf8_lossy(text).into_owned().into(),
        ValueRef::Blob(blob) => String::from_utf8_lossy(blob).into_owned().into(),
    }
}

/// The row's first `names.len()` columns as an object, as `dict(zip(names, row))`.
pub fn object(row: &Row, names: &[&str], from: usize) -> rusqlite::Result<Map<String, Value>> {
    names.iter().enumerate().map(|(i, name)| Ok((name.to_string(), json_of(row.get_ref(from + i)?)))).collect()
}

/// A REAL column Python may have written as an integer.
pub fn real(row: &Row, at: usize) -> rusqlite::Result<f64> {
    Ok(match row.get_ref(at)? {
        ValueRef::Integer(n) => n as f64,
        ValueRef::Real(x) => x,
        _ => 0.0,
    })
}

pub fn optional_real(row: &Row, at: usize) -> rusqlite::Result<Option<f64>> {
    Ok(match row.get_ref(at)? {
        ValueRef::Null => None,
        _ => Some(real(row, at)?),
    })
}

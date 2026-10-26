//! SQLite on threads of its own: one writer that runs every change in order, and read-only readers for everything else.

use std::future::Future;
use std::panic::Location;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, Row, TransactionBehavior};
use serde_json::{Map, Value};
use tokio::sync::oneshot;

use crate::{budgets, checks, hours, registry, roundlog, roundstore, validations};

pub const FILE: &str = "task_api.db";

type Job = Box<dyn FnOnce(&mut Connection) + Send>;

const BUSY: Duration = Duration::from_secs(5);
const STARTUP_BUSY: Duration = Duration::from_secs(60);
/// A database call that waited and ran this long in all is logged with the code that made it.
const SLOW: Duration = Duration::from_secs(1);

fn note_slow(kind: &str, pool: &str, caller: &Location, waited: Duration, ran: Duration) {
    if waited + ran >= SLOW {
        eprintln!("sqlite {kind} on {pool} from {caller} waited {:.1}s, ran {:.1}s", waited.as_secs_f64(), ran.as_secs_f64());
    }
}

fn connect(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(BUSY)?;
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
        // During a rolling deploy the running copy holds the write lock often; starting waits its turn instead of failing.
        conn.busy_timeout(STARTUP_BUSY)?;
        budgets::create(&conn)?;
        roundlog::create(&conn)?;
        roundstore::create(&conn)?;
        validations::create(&conn)?;
        checks::create(&conn)?;
        registry::create(&conn)?;
        hours::create(&conn)?;
        conn.busy_timeout(BUSY)?;
        Ok(Db { jobs: spawn("sqlite", conn)? })
    }

    /// Runs `work` in one transaction on the writer thread: all of it lands, or none of it.
    #[track_caller]
    pub fn run<T: Send + 'static>(&self, work: impl FnOnce(&Connection) -> Result<T> + Send + 'static) -> impl Future<Output = Result<T>> + '_ {
        let (caller, queued) = (Location::caller(), Instant::now());
        async move {
            let (done, result) = oneshot::channel();
            let job: Job = Box::new(move |conn| {
                let started = Instant::now();
                // Immediate, so a second copy of the API writing the same file waits its turn instead of failing.
                let outcome = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(anyhow::Error::from).and_then(|tx| {
                    let value = work(&tx)?;
                    tx.commit()?;
                    Ok(value)
                });
                note_slow("write", "writer", caller, started - queued, started.elapsed());
                let _ = done.send(outcome);
            });
            self.jobs.send(job).map_err(|_| anyhow!("the SQLite writer stopped"))?;
            result.await.map_err(|_| anyhow!("the SQLite writer stopped"))?
        }
    }
}

/// Every read connection of a pool is busy; the caller is asked to come back.
#[derive(Debug)]
pub struct Busy;

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the database readers are busy")
    }
}

impl std::error::Error for Busy {}

/// Read-only connections on threads of their own, so reads never wait behind the writer or hold it up.
pub struct Reader {
    name: String,
    jobs: mpsc::Sender<Job>,
    waiting: Arc<AtomicUsize>,
    max_waiting: usize,
}

impl Reader {
    /// `threads` connections share the reads; past `max_waiting` queued reads, more are turned away.
    pub fn open(path: &Path, name: &str, threads: usize, max_waiting: usize) -> Result<Reader> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let inbox = Arc::new(Mutex::new(inbox));
        for n in 0..threads {
            let mut conn = connect(path)?;
            conn.pragma_update(None, "query_only", "ON")?;
            let inbox = inbox.clone();
            std::thread::Builder::new().name(format!("{name}-{n}")).spawn(move || loop {
                let job = inbox.lock().expect("reader inbox").recv();
                match job {
                    Ok(job) => job(&mut conn),
                    Err(_) => break,
                }
            })?;
        }
        Ok(Reader { name: name.into(), jobs, waiting: Arc::default(), max_waiting })
    }

    #[track_caller]
    pub fn read<T: Send + 'static>(&self, work: impl FnOnce(&Connection) -> Result<T> + Send + 'static) -> impl Future<Output = Result<T>> + '_ {
        let (caller, queued) = (Location::caller(), Instant::now());
        async move {
            if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.max_waiting {
                self.waiting.fetch_sub(1, Ordering::SeqCst);
                return Err(Busy.into());
            }
            let (waiting, pool) = (self.waiting.clone(), self.name.clone());
            let (done, result) = oneshot::channel();
            let job: Job = Box::new(move |conn| {
                let started = Instant::now();
                let _ = done.send(work(conn));
                note_slow("read", &pool, caller, started - queued, started.elapsed());
                waiting.fetch_sub(1, Ordering::SeqCst);
            });
            if self.jobs.send(job).is_err() {
                self.waiting.fetch_sub(1, Ordering::SeqCst);
                return Err(anyhow!("the SQLite reader stopped"));
            }
            result.await.map_err(|_| anyhow!("the SQLite reader stopped"))?
        }
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

/// The share of the volume holding `path` in use, from 0 to 1.
#[cfg(unix)]
pub fn disk_used(path: &Path) -> Option<f64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: statvfs only writes into the struct it is given, and the path is NUL-terminated.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let (total, free) = (stat.f_blocks as f64 * stat.f_frsize as f64, stat.f_bavail as f64 * stat.f_frsize as f64);
    (total > 0.0).then(|| 1.0 - free / total)
}

#[cfg(not(unix))]
pub fn disk_used(_path: &Path) -> Option<f64> {
    None
}

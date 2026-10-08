//! Who may call: the subnet's hotkeys and which of them vote, refreshed on a timer so no request waits on the chain.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use desearch::hotkey::Hotkey;
use rusqlite::{params, Connection};
use tokio::sync::watch;

use crate::chain::{Chain, Neuron};
use crate::db::Db;

const REFRESH: Duration = Duration::from_secs(600);
const RETRY: Duration = Duration::from_secs(60);
/// A request waits this long for a first metagraph, then is asked to come back.
const FIRST_LOAD_WAIT: Duration = Duration::from_secs(5);
const MIN_TOTAL_STAKE: f64 = 10_000.0;
const MIN_ALPHA_STAKE: f64 = 20.0;
const RAO: f64 = 1e9;

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub hotkey: String,
    pub uid: Option<i64>,
    pub is_validator: bool,
    pub coldkey: Option<String>,
}

/// The metagraph has not loaded since this copy started and none was kept from before.
#[derive(Debug)]
pub struct NotLoaded;

pub fn create(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS registry (
            hotkey       TEXT PRIMARY KEY,
            uid          INTEGER,
            is_validator INTEGER NOT NULL,
            coldkey      TEXT
        )",
    )?;
    Ok(())
}

/// Keeps the metagraph as last loaded, so a restart serves at once whatever the chain does.
pub fn save(conn: &Connection, entries: &[Entry]) -> Result<()> {
    conn.execute("DELETE FROM registry", [])?;
    let mut insert = conn.prepare("INSERT INTO registry (hotkey, uid, is_validator, coldkey) VALUES (?, ?, ?, ?)")?;
    for entry in entries {
        insert.execute(params![entry.hotkey, entry.uid, entry.is_validator, entry.coldkey])?;
    }
    Ok(())
}

pub fn saved(conn: &Connection) -> Result<Vec<Entry>> {
    let mut statement = conn.prepare("SELECT hotkey, uid, is_validator, coldkey FROM registry")?;
    let rows = statement.query_map([], |row| Ok(Entry { hotkey: row.get(0)?, uid: row.get(1)?, is_validator: row.get(2)?, coldkey: row.get(3)? }))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// A permit alone is not enough: votes decide results, so a vote costs stake.
pub fn is_validator(neuron: &Neuron) -> bool {
    neuron.validator_permit && neuron.total_stake_rao as f64 / RAO >= MIN_TOTAL_STAKE && neuron.alpha_stake_rao as f64 / RAO >= MIN_ALPHA_STAKE
}

/// The ss58 addresses of comma-separated secret URIs.
pub fn addresses(uris: &str) -> Result<HashSet<String>> {
    uris.split(',').map(str::trim).filter(|uri| !uri.is_empty()).map(|uri| Ok(Hotkey::from_uri(uri)?.ss58())).collect()
}

pub enum Registry {
    /// Every hotkey is let in; the listed ones vote.
    Local {
        validators: HashSet<String>,
    },
    Chain {
        entries: RwLock<Arc<HashMap<String, Entry>>>,
        loaded: watch::Sender<bool>,
    },
}

impl Registry {
    pub fn local(validators: HashSet<String>) -> Registry {
        Registry::Local { validators }
    }

    pub fn chain() -> Registry {
        Registry::Chain { entries: RwLock::default(), loaded: watch::channel(false).0 }
    }

    /// Reloads the metagraph every ten minutes, keeping the last one when a reload fails.
    pub fn refresh(self: &Arc<Self>, chain: Arc<Chain>, netuid: u16, db: Db) {
        let Registry::Chain { .. } = self.as_ref() else { return };
        let registry = self.clone();
        tokio::spawn(async move {
            loop {
                let wait = match chain.metagraph(netuid).await {
                    Ok(neurons) => {
                        let entries = registry.load(neurons);
                        if let Err(error) = db.run(move |conn| save(conn, &entries)).await {
                            eprintln!("could not keep the metagraph: {error:#}");
                        }
                        REFRESH
                    }
                    Err(error) => {
                        eprintln!("metagraph refresh failed; keeping {} entries: {error:#}", registry.size());
                        RETRY
                    }
                };
                tokio::time::sleep(wait).await;
            }
        });
    }

    /// Loads a fresh metagraph and returns its entries.
    pub fn load(&self, neurons: Vec<Neuron>) -> Vec<Entry> {
        let fresh: Vec<Entry> = neurons
            .iter()
            .map(|n| Entry { hotkey: n.hotkey.clone(), uid: Some(n.uid), is_validator: is_validator(n), coldkey: Some(n.coldkey.clone()) })
            .collect();
        self.restore(fresh.clone());
        fresh
    }

    /// Serves these entries until the next load.
    pub fn restore(&self, kept: Vec<Entry>) {
        let Registry::Chain { entries, loaded } = self else { return };
        *entries.write().expect("registry lock") = Arc::new(kept.into_iter().map(|entry| (entry.hotkey.clone(), entry)).collect());
        loaded.send_replace(true);
    }

    fn size(&self) -> usize {
        match self {
            Registry::Local { validators } => validators.len(),
            Registry::Chain { entries, .. } => entries.read().expect("registry lock").len(),
        }
    }

    /// The caller's entry; before any metagraph is known, waits a few seconds for the first.
    pub async fn lookup(&self, hotkey: &str) -> Result<Option<Entry>, NotLoaded> {
        match self {
            Registry::Local { validators } => Ok(Some(Entry { hotkey: hotkey.into(), uid: None, is_validator: validators.contains(hotkey), coldkey: None })),
            Registry::Chain { entries, loaded } => {
                let mut ready = loaded.subscribe();
                if !matches!(tokio::time::timeout(FIRST_LOAD_WAIT, ready.wait_for(|loaded| *loaded)).await, Ok(Ok(_))) {
                    return Err(NotLoaded);
                }
                Ok(entries.read().expect("registry lock").get(hotkey).cloned())
            }
        }
    }

    /// The hotkey's metagraph entry as last loaded, without waiting on the chain.
    pub fn registered(&self, hotkey: &str) -> Option<Entry> {
        match self {
            Registry::Local { .. } => None,
            Registry::Chain { entries, .. } => entries.read().expect("registry lock").get(hotkey).cloned(),
        }
    }
}

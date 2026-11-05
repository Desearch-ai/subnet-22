//! Who may call: the subnet's hotkeys and which of them vote, refreshed on a timer so no request waits on the chain.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use desearch::hotkey::Hotkey;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::chain::{Chain, Neuron};
use crate::db::Db;

const REFRESH: Duration = Duration::from_secs(600);
const RETRY: Duration = Duration::from_secs(60);
/// A request waits this long for a first metagraph, then is asked to come back.
const FIRST_LOAD_WAIT: Duration = Duration::from_secs(5);
const RAO: f64 = 1e9;

/// What a permit needs staked behind it to vote, in TAO: in all, and in the subnet's alpha. The default is mainnet's; a testnet
/// sets lower.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinStake {
    pub total: f64,
    pub alpha: f64,
}

impl Default for MinStake {
    fn default() -> Self {
        MinStake { total: 10_000.0, alpha: 20.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub hotkey: String,
    pub uid: Option<i64>,
    pub is_validator: bool,
    pub coldkey: Option<String>,
    /// Its share of the subnet's incentive on chain; None where no metagraph gives one.
    pub incentive: Option<f64>,
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
    let has_incentive: i64 = conn.query_row("SELECT COUNT(*) FROM pragma_table_info('registry') WHERE name = 'incentive'", [], |row| row.get(0))?;
    if has_incentive == 0 {
        conn.execute_batch("ALTER TABLE registry ADD COLUMN incentive REAL")?;
    }
    Ok(())
}

/// Keeps the metagraph as last loaded, so a restart serves at once whatever the chain does.
pub fn save(conn: &Connection, entries: &[Entry]) -> Result<()> {
    conn.execute("DELETE FROM registry", [])?;
    let mut insert = conn.prepare("INSERT INTO registry (hotkey, uid, is_validator, coldkey, incentive) VALUES (?, ?, ?, ?, ?)")?;
    for entry in entries {
        insert.execute(params![entry.hotkey, entry.uid, entry.is_validator, entry.coldkey, entry.incentive])?;
    }
    Ok(())
}

/// Each hotkey's coldkey as the registry table holds it, the hotkey itself for one off the metagraph. The table is written in the
/// transaction that logs the metagraph, so what a verdict reads here in its own transaction is what the log names before it.
pub fn coldkeys_of<'a>(conn: &Connection, hotkeys: impl IntoIterator<Item = &'a str>) -> Result<HashMap<String, String>> {
    let mut statement = conn.prepare("SELECT coldkey FROM registry WHERE hotkey = ?")?;
    let mut found = HashMap::new();
    for hotkey in hotkeys {
        let coldkey: Option<Option<String>> = statement.query_row([hotkey], |row| row.get(0)).optional()?;
        found.insert(hotkey.to_string(), coldkey.flatten().filter(|c| !c.is_empty()).unwrap_or_else(|| hotkey.to_string()));
    }
    Ok(found)
}

pub fn saved(conn: &Connection) -> Result<Vec<Entry>> {
    let mut statement = conn.prepare("SELECT hotkey, uid, is_validator, coldkey, incentive FROM registry")?;
    let rows = statement
        .query_map([], |row| Ok(Entry { hotkey: row.get(0)?, uid: row.get(1)?, is_validator: row.get(2)?, coldkey: row.get(3)?, incentive: row.get(4)? }))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Writes the metagraph into the log when who is on the subnet, who may vote or the incentive changed since it was last written; the
/// neurons with their stakes and incentive go in a blob, so a replay sees who could vote and what capped each budget, and every
/// validator checks it against its own node at the block it was read at.
pub fn log_metagraph(conn: &Connection, netuid: u16, block: Option<i64>, neurons: &[Neuron], entries: &[Entry]) -> Result<()> {
    let mut members: Vec<Value> = entries.iter().map(|e| json!([e.uid, e.hotkey, e.coldkey, e.is_validator, e.incentive])).collect();
    members.sort_by_key(|member| member.to_string());
    let members_hash = crate::proofs::sha256(&crate::py::canonical(&Value::Array(members)));
    let last: Option<String> =
        conn.query_row("SELECT entry FROM journal WHERE kind = 'metagraph' ORDER BY seq DESC LIMIT 1", [], |row| row.get(0)).optional()?;
    if last.and_then(|entry| serde_json::from_str::<Value>(&entry).ok()).is_some_and(|entry| entry["output"]["members"] == members_hash.as_str()) {
        return Ok(());
    }
    let listed: Vec<Value> = neurons
        .iter()
        .map(|n| json!([n.uid, n.hotkey, n.coldkey, n.validator_permit, n.total_stake_rao.to_string(), n.alpha_stake_rao.to_string(), n.incentive]))
        .collect();
    let blob = crate::journal::Blob::of(crate::py::canonical(&Value::Array(listed)));
    let validators: Vec<&str> = entries.iter().filter(|e| e.is_validator).map(|e| e.hotkey.as_str()).collect();
    let input = json!({"netuid": netuid, "block": block, "neurons": blob.reference()});
    let output = json!({"members": members_hash, "validators": validators});
    crate::journal::append(conn, crate::journal::Record::new("metagraph", desearch::time::now(), input, output).with_blobs(vec![blob]))?;
    Ok(())
}

/// A permit alone is not enough: votes decide results, so a vote costs stake.
pub fn is_validator(neuron: &Neuron, min: MinStake) -> bool {
    neuron.validator_permit && neuron.total_stake_rao as f64 / RAO >= min.total && neuron.alpha_stake_rao as f64 / RAO >= min.alpha
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
        min_stake: MinStake,
    },
}

impl Registry {
    pub fn local(validators: HashSet<String>) -> Registry {
        Registry::Local { validators }
    }

    pub fn chain(min_stake: MinStake) -> Registry {
        Registry::Chain { entries: RwLock::default(), loaded: watch::channel(false).0, min_stake }
    }

    /// Reloads the metagraph every ten minutes, keeping the last one when a reload fails.
    pub fn refresh(self: &Arc<Self>, chain: Arc<Chain>, netuid: u16, db: Db) {
        let Registry::Chain { .. } = self.as_ref() else { return };
        let registry = self.clone();
        tokio::spawn(async move {
            loop {
                let wait = match chain.metagraph(netuid).await {
                    Ok((block, neurons)) => {
                        let entries = registry.load(neurons.clone());
                        let saved = db.run(move |conn| {
                            save(conn, &entries)?;
                            log_metagraph(conn, netuid, Some(block), &neurons, &entries)
                        });
                        if let Err(error) = saved.await {
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
        let min_stake = match self {
            Registry::Chain { min_stake, .. } => *min_stake,
            Registry::Local { .. } => MinStake::default(),
        };
        let fresh: Vec<Entry> = neurons
            .iter()
            .map(|n| Entry {
                hotkey: n.hotkey.clone(),
                uid: Some(n.uid),
                is_validator: is_validator(n, min_stake),
                coldkey: Some(n.coldkey.clone()),
                incentive: n.incentive,
            })
            .collect();
        self.restore(fresh.clone());
        fresh
    }

    /// Serves these entries until the next load.
    pub fn restore(&self, kept: Vec<Entry>) {
        let Registry::Chain { entries, loaded, .. } = self else { return };
        *entries.write().expect("registry lock") = Arc::new(kept.into_iter().map(|entry| (entry.hotkey.clone(), entry)).collect());
        loaded.send_replace(true);
    }

    fn size(&self) -> usize {
        match self {
            Registry::Local { validators } => validators.len(),
            Registry::Chain { entries, .. } => entries.read().expect("registry lock").len(),
        }
    }

    /// Whether log pages list this hotkey: one on the subnet now, or any while no metagraph is known.
    pub fn lists(&self, hotkey: &str) -> bool {
        !self.ready() || matches!(self, Registry::Local { .. }) || self.registered(hotkey).is_some()
    }

    /// Whether a metagraph is known, kept from before or loaded since starting.
    pub fn ready(&self) -> bool {
        match self {
            Registry::Local { .. } => true,
            Registry::Chain { loaded, .. } => *loaded.borrow(),
        }
    }

    /// The caller's entry; before any metagraph is known, waits a few seconds for the first.
    pub async fn lookup(&self, hotkey: &str) -> Result<Option<Entry>, NotLoaded> {
        match self {
            Registry::Local { validators } => {
                Ok(Some(Entry { hotkey: hotkey.into(), uid: None, is_validator: validators.contains(hotkey), coldkey: None, incentive: None }))
            }
            Registry::Chain { entries, loaded, .. } => {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn neuron(uid: i64, hotkey: &str, coldkey: &str) -> Neuron {
        Neuron {
            uid,
            hotkey: hotkey.into(),
            coldkey: coldkey.into(),
            validator_permit: true,
            total_stake_rao: 20_000_000_000_000,
            alpha_stake_rao: 30_000_000_000,
            incentive: None,
        }
    }

    #[test]
    fn a_verdict_reads_coldkeys_as_the_logged_metagraph_names_them() {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        crate::journal::create(&conn).unwrap();
        let registry = Registry::chain(MinStake::default());
        let neurons = vec![neuron(0, "5V1", "5Owner"), neuron(1, "5V2", "5Owner"), neuron(2, "5V3", "")];
        let entries = registry.load(neurons.clone());
        // The table and the log entry are written in one transaction, which is what a verdict's own transaction reads.
        save(&conn, &entries).unwrap();
        log_metagraph(&conn, 22, Some(100), &neurons, &entries).unwrap();
        let found = coldkeys_of(&conn, ["5V1", "5V2", "5V3", "5Gone"]).unwrap();
        assert_eq!(found["5V1"], "5Owner");
        assert_eq!(found["5V2"], "5Owner", "two hotkeys of one owner count once");
        assert_eq!((found["5V3"].as_str(), found["5Gone"].as_str()), ("5V3", "5Gone"), "no coldkey, or off the metagraph: the hotkey");
        let logged: String = conn.query_row("SELECT entry FROM journal WHERE kind = 'metagraph'", [], |row| row.get(0)).unwrap();
        assert!(logged.contains("\"block\":100"));
    }

    #[test]
    fn a_testnet_lowers_the_stake_a_vote_needs() {
        let staked = |total_tao: u128, alpha_tao: u128| Neuron {
            total_stake_rao: total_tao * 1_000_000_000,
            alpha_stake_rao: alpha_tao * 1_000_000_000,
            ..neuron(0, "5V", "5O")
        };
        let testnet = MinStake { total: 100.0, alpha: 1.0 };
        assert!(!is_validator(&staked(500, 5), MinStake::default()), "short of mainnet's 10,000 TAO");
        assert!(is_validator(&staked(500, 5), testnet));
        assert!(!is_validator(&staked(99, 5), testnet) && !is_validator(&staked(500, 0), testnet), "each minimum holds on its own");
        assert!(!is_validator(&Neuron { validator_permit: false, ..staked(500, 5) }, testnet), "still a permit first");
        let entries = Registry::chain(testnet).load(vec![staked(500, 5)]);
        assert!(entries[0].is_validator, "the registry votes by the minimums it was made with");
    }
}

//! Who may call: the subnet's hotkeys and which of them vote, refreshed on a timer so no request waits on the chain.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use desearch::hotkey::Hotkey;
use tokio::sync::watch;

use crate::chain::{Chain, Neuron};

const REFRESH: Duration = Duration::from_secs(600);
const RETRY: Duration = Duration::from_secs(60);
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
    pub fn refresh(self: &Arc<Self>, chain: Arc<Chain>, netuid: u16) {
        let Registry::Chain { .. } = self.as_ref() else { return };
        let registry = self.clone();
        tokio::spawn(async move {
            loop {
                let wait = match chain.metagraph(netuid).await {
                    Ok(neurons) => {
                        registry.load(neurons);
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

    pub fn load(&self, neurons: Vec<Neuron>) {
        let Registry::Chain { entries, loaded } = self else { return };
        let loaded_entries: HashMap<String, Entry> = neurons
            .iter()
            .map(|n| (n.hotkey.clone(), Entry { hotkey: n.hotkey.clone(), uid: Some(n.uid), is_validator: is_validator(n), coldkey: Some(n.coldkey.clone()) }))
            .collect();
        *entries.write().expect("registry lock") = Arc::new(loaded_entries);
        loaded.send_replace(true);
    }

    fn size(&self) -> usize {
        match self {
            Registry::Local { validators } => validators.len(),
            Registry::Chain { entries, .. } => entries.read().expect("registry lock").len(),
        }
    }

    /// The caller's entry, once the metagraph has loaded the first time.
    pub async fn lookup(&self, hotkey: &str) -> Option<Entry> {
        match self {
            Registry::Local { validators } => Some(Entry { hotkey: hotkey.into(), uid: None, is_validator: validators.contains(hotkey), coldkey: None }),
            Registry::Chain { entries, loaded } => {
                let mut ready = loaded.subscribe();
                let _ = ready.wait_for(|loaded| *loaded).await;
                entries.read().expect("registry lock").get(hotkey).cloned()
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

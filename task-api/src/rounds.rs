//! Rounds of tasks: URLs packed into batches, committed to by a manifest hash, served in an order a later block's hash decides.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::proofs::{self, sha256};
use crate::py::canonical;

pub const CLAIM_TTL_S: i64 = 180;
/// Past the expiry miners are told, so an upload that ends a few seconds late still counts.
pub const UPLOAD_GRACE_S: f64 = 10.0;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Url {
    pub host: String,
    pub url: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    pub batch_id: String,
    pub urls: Vec<Url>,
    /// What a task of this kind carries beyond its URLs, e.g. an embed batch's input file.
    pub extra: Map<String, Value>,
    /// The count and hash a closed round's URLs had, kept once the URLs themselves are dropped.
    pub sealed: Option<Value>,
}

impl Batch {
    pub fn new(batch_id: String, urls: Vec<Url>, extra: Map<String, Value>) -> Self {
        Batch { batch_id, urls, extra, sealed: None }
    }

    pub fn urls_hash(&self) -> String {
        let urls: Vec<Value> = self.urls.iter().map(|u| Value::from(u.url.as_str())).collect();
        sha256(&canonical(&Value::Array(urls)))
    }

    pub fn seal(&self) -> Value {
        self.sealed.clone().unwrap_or_else(|| json!({"url_count": self.urls.len(), "urls_hash": self.urls_hash()}))
    }

    pub fn manifest_entry(&self) -> Value {
        let mut entry = Map::new();
        entry.insert("batch_id".into(), self.batch_id.clone().into());
        if let Value::Object(sealed) = self.seal() {
            entry.extend(sealed);
        }
        if let Some(input) = self.extra.get("input_sha256") {
            entry.insert("input_sha256".into(), input.clone());
        }
        Value::Object(entry)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Round {
    pub round_id: String,
    pub batches: Vec<Batch>,
    pub manifest_hash: String,
    pub seed_block: i64,
    pub opened_at: f64,
    pub kind: String,
    pub seed: Option<String>,
    pub order: Vec<String>,
    pub closed_at: Option<f64>,
}

impl Round {
    pub fn revealed(&self) -> bool {
        self.seed.is_some()
    }

    pub fn batch(&self, batch_id: &str) -> Option<&Batch> {
        self.batches.iter().find(|b| b.batch_id == batch_id)
    }

    pub fn manifest(&self) -> Vec<Value> {
        self.batches.iter().map(Batch::manifest_entry).collect()
    }

    pub fn public_view(&self) -> Map<String, Value> {
        let mut view = Map::new();
        view.insert("round_id".into(), self.round_id.clone().into());
        view.insert("kind".into(), self.kind.clone().into());
        view.insert("algorithm".into(), proofs::ALGORITHM.into());
        view.insert("manifest_hash".into(), self.manifest_hash.clone().into());
        view.insert("seed_block".into(), self.seed_block.into());
        view.insert("opened_at".into(), self.opened_at.into());
        view.insert("seed".into(), self.seed.clone().into());
        view.insert("closed_at".into(), self.closed_at.into());
        if self.closed_at.is_some() {
            let mut manifest = self.manifest();
            manifest.sort_by(|a, b| a["batch_id"].as_str().cmp(&b["batch_id"].as_str()));
            view.insert("manifest".into(), manifest.into());
            view.insert("serve_order".into(), self.order.clone().into());
        }
        view
    }
}

/// Round-robin hosts so a refusing site costs a row, not a task.
pub fn spread(urls: Vec<Url>) -> Vec<Url> {
    let mut queues: Vec<std::collections::VecDeque<Url>> = Vec::new();
    let mut by_host: HashMap<String, usize> = HashMap::new();
    for url in urls {
        let at = *by_host.entry(url.host.clone()).or_insert_with(|| {
            queues.push(Default::default());
            queues.len() - 1
        });
        queues[at].push_back(url);
    }
    let mut mixed = Vec::with_capacity(queues.iter().map(|q| q.len()).sum());
    while !queues.is_empty() {
        queues.retain(|queue| !queue.is_empty());
        for queue in &mut queues {
            mixed.extend(queue.pop_front());
        }
    }
    mixed
}

pub fn pack(urls: Vec<Url>, task_urls: usize) -> Vec<Batch> {
    spread(urls).chunks(task_urls.max(1)).map(|chunk| Batch::new(new_id(), chunk.to_vec(), Map::new())).collect()
}

pub fn open_batches(batches: Vec<Batch>, seed_block: i64, kind: &str, now: f64) -> Round {
    let entries: Vec<Value> = batches.iter().map(Batch::manifest_entry).collect();
    Round {
        round_id: new_id(),
        manifest_hash: proofs::manifest_hash(&entries, seed_block),
        batches,
        seed_block,
        opened_at: now,
        kind: kind.into(),
        seed: None,
        order: Vec::new(),
        closed_at: None,
    }
}

pub fn reveal(round: &mut Round, seed: &str) -> Vec<String> {
    round.seed = Some(seed.into());
    let ids: Vec<String> = round.batches.iter().map(|b| b.batch_id.clone()).collect();
    round.order = proofs::serve_order(seed, &ids);
    round.order.clone()
}

/// The first 16 hex digits of a random UUID, as Python's `uuid.uuid4().hex[:16]`.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(host: &str, n: usize) -> Url {
        Url { host: host.into(), url: format!("https://{host}/{n}") }
    }

    #[test]
    fn hosts_take_turns() {
        let urls = vec![url("a", 1), url("a", 2), url("a", 3), url("b", 1), url("c", 1), url("c", 2)];
        let mixed: Vec<String> = spread(urls).into_iter().map(|u| u.url).collect();
        assert_eq!(mixed, ["https://a/1", "https://b/1", "https://c/1", "https://a/2", "https://c/2", "https://a/3"]);
        assert_eq!(pack(vec![url("a", 1), url("a", 2), url("b", 1)], 2).iter().map(|b| b.urls.len()).collect::<Vec<_>>(), [2, 1]);
    }

    #[test]
    fn a_sealed_batch_keeps_its_manifest() {
        let batch = Batch::new("b1".into(), vec![url("a", 1), url("a", 2)], Map::new());
        let sealed = Batch { urls: Vec::new(), sealed: Some(batch.seal()), ..batch.clone() };
        assert_eq!(sealed.manifest_entry(), batch.manifest_entry());
        assert_eq!(batch.urls_hash(), "0b42b29524602911540cc312be701c278d6591ab08034be133c6720f49a83543");
    }

    #[test]
    fn a_batch_mixes_hosts_and_keeps_every_url() {
        let urls: Vec<Url> =
            (0..5).flat_map(|h| (0..10).map(move |n| Url { host: format!("site{h}.example"), url: format!("https://site{h}.example/{n}") })).collect();
        let batches = pack(urls.clone(), 10);
        assert_eq!(batches.len(), 5);
        assert!(batches.iter().all(|b| b.urls.iter().map(|u| &u.host).collect::<std::collections::HashSet<_>>().len() == 5));
        let many: Vec<Url> = (0..2530).map(|n| Url { host: format!("s{}.example", n % 40), url: format!("https://s{}.example/{n}", n % 40) }).collect();
        assert_eq!(pack(many, 1000).iter().map(|b| b.urls.len()).collect::<Vec<_>>(), [1000, 1000, 530]);
        let mut lopsided: Vec<Url> = (0..25).map(|n| Url { host: "big.example".into(), url: format!("https://big.example/{n}") }).collect();
        lopsided.push(Url { host: "small.example".into(), url: "https://small.example/1".into() });
        let batches = pack(lopsided, 10);
        assert_eq!(batches.iter().map(|b| b.urls.len()).sum::<usize>(), 26);
        assert_eq!(batches[0].urls[1].host, "small.example");
    }
}

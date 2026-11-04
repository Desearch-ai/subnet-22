//! Rounds of tasks: URLs packed into batches, committed to by a manifest hash, served in an order a later block's hash decides.
//!
//! A crawl round commits to its URL set alone (`desearch-round-3`): the batches are cut only once the seed block's hash is
//! known, every host's URLs dealt across them in turn, so no one can choose which URLs land together and every batch has the
//! round's mix. Embed rounds commit to their batches, one per planner input (`desearch-serve-order-2`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::proofs::{self, sha256};
use crate::py::canonical;

pub const CLAIM_TTL_S: i64 = 180;
/// An embed task is a GPU batch, so its claim lasts longer than a crawl task's.
pub const EMBED_CLAIM_TTL_S: i64 = 300;
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

    /// The batch as SQLite keeps it: its count and hash, without the URLs.
    pub fn sealed(&self) -> Batch {
        Batch { urls: Vec::new(), sealed: Some(self.seal()), ..self.clone() }
    }

    /// The batch as SQLite keeps it once Redis holds its task: sealed, and without an embed input's page list.
    pub fn queued(&self) -> Batch {
        let mut kept = self.sealed();
        kept.extra.remove("pages");
        kept
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
    /// A crawl round's URL set as committed before its seed existed: `url_count`, `urls_hash`, `task_urls` and `tasks`.
    pub commitment: Option<Value>,
}

impl Round {
    pub fn revealed(&self) -> bool {
        self.seed.is_some()
    }

    pub fn batch(&self, batch_id: &str) -> Option<&Batch> {
        self.batches.iter().find(|b| b.batch_id == batch_id)
    }

    pub fn sealed(&self) -> Round {
        Round { batches: self.batches.iter().map(Batch::sealed).collect(), ..self.clone() }
    }

    pub fn has_urls(&self) -> bool {
        self.batches.iter().any(|batch| !batch.urls.is_empty())
    }

    /// Each batch's URLs, as staged in Redis between commit and reveal.
    pub fn url_lists(&self) -> Value {
        Value::Object(
            self.batches.iter().map(|batch| (batch.batch_id.clone(), batch.urls.iter().map(|u| json!([u.host, u.url])).collect::<Vec<_>>().into())).collect(),
        )
    }

    /// Puts staged URL lists back on their batches; false when some batch has none.
    pub fn restore_urls(&mut self, lists: &Value) -> bool {
        let mut whole = true;
        for batch in &mut self.batches {
            match lists.get(&batch.batch_id).and_then(Value::as_array) {
                Some(urls) => {
                    batch.urls = urls
                        .iter()
                        .map(|pair| Url { host: pair[0].as_str().unwrap_or_default().into(), url: pair[1].as_str().unwrap_or_default().into() })
                        .collect()
                }
                None => whole = false,
            }
        }
        whole
    }

    pub fn manifest(&self) -> Vec<Value> {
        self.batches.iter().map(Batch::manifest_entry).collect()
    }

    /// Tasks the round has or will have once its seed cuts it.
    pub fn tasks(&self) -> usize {
        match &self.commitment {
            Some(commitment) => commitment["tasks"].as_u64().unwrap_or(0) as usize,
            None => self.batches.len(),
        }
    }

    pub fn public_view(&self) -> Map<String, Value> {
        let mut view = Map::new();
        view.insert("round_id".into(), self.round_id.clone().into());
        view.insert("kind".into(), self.kind.clone().into());
        let algorithm = if self.commitment.is_some() { proofs::SET_ALGORITHM } else { proofs::ALGORITHM };
        view.insert("algorithm".into(), algorithm.into());
        if let Some(commitment) = &self.commitment {
            for name in ["url_count", "urls_hash", "task_urls"] {
                view.insert(name.into(), commitment[name].clone());
            }
        }
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
        commitment: None,
    }
}

/// The hash of a URL set: its `[host, url]` pairs sorted by URL, as canonical JSON.
pub fn set_hash(urls: &[Url]) -> String {
    let mut pairs: Vec<(&str, &str)> = urls.iter().map(|u| (u.url.as_str(), u.host.as_str())).collect();
    pairs.sort_unstable();
    let listed: Vec<Value> = pairs.into_iter().map(|(url, host)| json!([host, url])).collect();
    sha256(&canonical(&Value::Array(listed)))
}

/// The tasks `url_count` URLs make, none over `task_urls`.
pub fn tasks_for(url_count: usize, task_urls: usize) -> usize {
    url_count.div_ceil(task_urls.max(1))
}

/// A crawl round committed to its URL set before its seed block exists; its batches are cut by `cut` once the seed is known.
pub fn open_set(urls: &[Url], seed_block: i64, task_urls: usize, now: f64) -> Round {
    let round_id = new_id();
    let (url_count, urls_hash) = (urls.len(), set_hash(urls));
    let commitment = json!({"url_count": url_count, "urls_hash": urls_hash, "task_urls": task_urls, "tasks": tasks_for(url_count, task_urls)});
    Round {
        manifest_hash: proofs::set_manifest_hash(&round_id, seed_block, task_urls, url_count, &urls_hash),
        round_id,
        batches: Vec::new(),
        seed_block,
        opened_at: now,
        kind: "crawl".into(),
        seed: None,
        order: Vec::new(),
        closed_at: None,
        commitment: Some(commitment),
    }
}

/// The batches a committed URL set makes under a seed, the same for anyone who has the set and the seed:
/// hosts in the order of `sha256("{seed}:host:{host}")`, each host's URLs in the order of `sha256("{seed}:url:{url}")`,
/// the whole run dealt across `tasks_for` batches in turn, each batch's hosts then taking turns (`spread`), and batch `i`
/// named by the first 16 hex digits of `sha256("{seed}:{round_id}:{i}")`.
pub fn cut(urls: &[Url], seed: &str, round_id: &str, task_urls: usize) -> Vec<Batch> {
    let count = tasks_for(urls.len(), task_urls);
    if count == 0 {
        return Vec::new();
    }
    let key = |kind: &str, value: &str| sha256(format!("{seed}:{kind}:{value}").as_bytes());
    let mut by_host: HashMap<&str, Vec<&Url>> = HashMap::new();
    for url in urls {
        by_host.entry(url.host.as_str()).or_default().push(url);
    }
    let mut hosts: Vec<(String, &str)> = by_host.keys().map(|host| (key("host", host), *host)).collect();
    hosts.sort_unstable();
    let mut dealt: Vec<Vec<Url>> = vec![Vec::new(); count];
    let mut at = 0;
    for (_, host) in hosts {
        let mut own: Vec<(String, &Url)> = by_host[host].iter().map(|url| (key("url", &url.url), *url)).collect();
        own.sort_unstable_by(|a, b| (&a.0, &a.1.url).cmp(&(&b.0, &b.1.url)));
        for (_, url) in own {
            dealt[at % count].push(url.clone());
            at += 1;
        }
    }
    dealt
        .into_iter()
        .enumerate()
        .map(|(i, urls)| Batch::new(sha256(format!("{seed}:{round_id}:{i}").as_bytes())[..16].to_string(), spread(urls), Map::new()))
        .collect()
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

    fn golden_set() -> Vec<Url> {
        let mut urls: Vec<Url> = (0..5).map(|n| url("a.example", n)).collect();
        urls.extend((0..3).map(|n| url("b.example", n)));
        urls.push(url("c.example", 0));
        urls
    }

    #[test]
    fn the_cut_matches_tools_verify_round() {
        let urls = golden_set();
        assert_eq!(set_hash(&urls), "a2315215c5ad3ef308edb903a7a7bbee6b987b971d557b36bcb21e764b064332");
        assert_eq!(proofs::set_manifest_hash("r1", 77, 4, 9, &set_hash(&urls)), "3ebc415543cf7d37d74b7a1043ebd7db706b685a2717047212aa974cd2eca0a6");
        let cut: Vec<(String, Vec<String>)> =
            cut(&urls, "seed", "r1", 4).into_iter().map(|b| (b.batch_id, b.urls.into_iter().map(|u| u.url).collect())).collect();
        assert_eq!(
            cut,
            [
                ("34af69571cea3b51".to_string(), vec!["https://c.example/0".to_string(), "https://a.example/0".into(), "https://b.example/1".into()]),
                ("5388cbf670e889e5".to_string(), vec!["https://a.example/3".to_string(), "https://b.example/0".into(), "https://a.example/1".into()]),
                ("7006229857aa481c".to_string(), vec!["https://a.example/2".to_string(), "https://b.example/2".into(), "https://a.example/4".into()]),
            ]
        );
    }

    #[test]
    fn the_seed_alone_decides_which_urls_go_together_and_every_batch_has_the_rounds_mix() {
        let urls: Vec<Url> = (0..2530).map(|n| Url { host: format!("s{}.example", n % 40), url: format!("https://s{}.example/{n}", n % 40) }).collect();
        let batches = cut(&urls, "seed", "r", 1000);
        assert_eq!(batches.iter().map(|b| b.urls.len()).collect::<Vec<_>>(), [844, 843, 843], "as even as the URLs allow, none over 1000");
        let mut all: Vec<&str> = batches.iter().flat_map(|b| b.urls.iter().map(|u| u.url.as_str())).collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), urls.len(), "every URL once");
        for host in 0..40 {
            let counts: Vec<usize> = batches.iter().map(|b| b.urls.iter().filter(|u| u.host == format!("s{host}.example")).count()).collect();
            assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1, "host {host} spread {counts:?}");
        }
        let mut shuffled = urls.clone();
        shuffled.reverse();
        assert_eq!(cut(&shuffled, "seed", "r", 1000), batches, "the order the bot sent them in plays no part");
        assert_ne!(cut(&urls, "other", "r", 1000)[0].urls, batches[0].urls);
        assert_eq!(set_hash(&shuffled), set_hash(&urls));
    }

    #[test]
    fn a_round_commits_to_its_set_before_any_batch_exists() {
        let round = open_set(&golden_set(), 77, 4, 1_800_000_000.0);
        assert!(round.batches.is_empty());
        assert_eq!(round.tasks(), 3);
        assert_eq!(round.manifest_hash, proofs::set_manifest_hash(&round.round_id, 77, 4, 9, &set_hash(&golden_set())));
        assert_eq!(round.public_view()["algorithm"], proofs::SET_ALGORITHM);
        assert_eq!(tasks_for(0, 1000), 0);
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

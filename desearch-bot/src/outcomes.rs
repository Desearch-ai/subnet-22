//! What became of the URLs the bot queued, read in order from the task API's public outcome feed.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use serde_json::{json, Value};

use crate::buckets::{bucket_of, Buckets};
use crate::dispatch::on_readers;
use crate::ready::{Applied, Outcome, PageOutcome};
use crate::urls::{self, Normaliser, Url};

const CURSOR: &str = "outcomes_seq";
const TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeRow {
    pub url: String,
    pub outcome: Outcome,
    /// Epoch microseconds.
    pub at: i64,
}

/// The rows of one outcome file; rows with an outcome this bot does not know are skipped.
pub fn read_parquet(data: Bytes) -> Result<Vec<OutcomeRow>> {
    let reader = SerializedFileReader::new(data).context("not a Parquet file")?;
    let mut rows = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let (mut url, mut outcome, mut at) = (None, None, None);
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("url", Field::Str(value)) => url = Some(value.clone()),
                ("outcome", Field::Str(value)) => outcome = Outcome::parse(value),
                ("at", Field::TimestampMicros(us)) => at = Some(*us),
                ("at", Field::TimestampMillis(ms)) => at = Some(ms * 1000),
                ("at", Field::Long(value)) => at = Some(if *value > 100_000_000_000_000_000 { value / 1000 } else { *value }),
                _ => {}
            }
        }
        if let (Some(url), Some(outcome), Some(at)) = (url, outcome, at) {
            rows.push(OutcomeRow { url, outcome, at });
        }
    }
    Ok(rows)
}

/// The domain a queued URL was sent under and its store key: its host, then each parent domain, whichever this process holds the page for.
pub fn locate(buckets: &Buckets, url: &str) -> Option<(String, Url)> {
    let parts = urls::split(url).ok()?;
    let host = urls::ascii(&parts.hostname()?).ok()?;
    let bare = host.strip_prefix("www.").unwrap_or(&host);
    let mut candidates = vec![host.as_str()];
    let mut parent = bare;
    loop {
        if candidates.last() != Some(&parent) {
            candidates.push(parent);
        }
        match parent.split_once('.') {
            Some((_, rest)) if rest.contains('.') => parent = rest,
            _ => break,
        }
    }
    for domain in candidates {
        if !buckets.owns(domain) {
            continue;
        }
        let Some(key) = Normaliser::new(domain).ok().and_then(|n| n.parse(url)) else {
            continue;
        };
        if buckets.store(domain).url(&key).ok().flatten().is_some() {
            return Some((domain.to_string(), key));
        }
    }
    None
}

/// Keep a file's outcomes in the stores that hold their pages.
pub fn apply(buckets: &Buckets, rows: &[OutcomeRow], recrawl_after: u32) -> Result<Applied> {
    let mut by_bucket: HashMap<usize, Vec<PageOutcome>> = HashMap::new();
    let mut total = Applied::default();
    for row in rows {
        let Some((domain, key)) = locate(buckets, &row.url) else {
            total.ignored += 1;
            continue;
        };
        let rest = key.key[domain.len() + 1..].to_vec();
        let at = row.at.div_euclid(1_000_000).clamp(0, u32::MAX.into()) as u32;
        by_bucket.entry(bucket_of(&domain)).or_default().push(PageOutcome { domain, rest, outcome: row.outcome, at });
    }
    let groups: Vec<Vec<PageOutcome>> = by_bucket.into_values().collect();
    for applied in on_readers(groups, |rows| buckets.store(&rows[0].domain).apply_outcomes(&rows, recrawl_after))? {
        total.crawled += applied.crawled;
        total.retried += applied.retried;
        total.gave_up += applied.gave_up;
        total.ignored += applied.ignored;
    }
    Ok(total)
}

/// The next sequence number to read: the lowest any store has kept, so no store misses a file; None before the first read.
pub fn saved_cursor(buckets: &Buckets) -> Result<Option<u64>> {
    let mut lowest = None;
    for store in buckets.stores() {
        if let Some(next) = store.meta(CURSOR)?.as_ref().and_then(Value::as_u64) {
            lowest = Some(lowest.map_or(next, |l: u64| l.min(next)));
        }
    }
    Ok(lowest)
}

pub fn save_cursor(buckets: &Buckets, next: u64) -> Result<()> {
    for store in buckets.stores() {
        store.set_meta(CURSOR, &json!(next))?;
    }
    Ok(())
}

/// One numbered outcome file, nothing yet, or a number that is gone while later ones exist.
pub enum Next {
    Wait,
    Missing,
    File { rows: Vec<OutcomeRow>, published_at: Option<i64> },
}

pub struct OutcomeFeed {
    base: String,
    http: reqwest::Client,
}

impl OutcomeFeed {
    /// The newest number the task API has written, from `outcomes/latest.json`.
    pub async fn latest(&self) -> Result<Option<u64>> {
        let found = self.http.get(format!("{}/outcomes/latest.json", self.base)).send().await?;
        if found.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !found.status().is_success() {
            bail!("outcomes/latest.json: HTTP {}", found.status());
        }
        let latest: Value = serde_json::from_slice(&found.bytes().await?).context("outcomes/latest.json that is not JSON")?;
        Ok(latest.get("seq").and_then(Value::as_u64))
    }
}

impl OutcomeFeed {
    pub fn new(base: &str) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(TIMEOUT).build()?;
        Ok(OutcomeFeed { base: base.trim_end_matches('/').to_string(), http })
    }

    /// File number `seq`; a number not published yet means wait, since files are read strictly in order.
    pub async fn fetch(&self, seq: u64) -> Result<Next> {
        let index = self.http.get(format!("{}/outcomes/seq/{seq:012}.json", self.base)).send().await?;
        if index.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(if self.latest().await?.is_some_and(|latest| latest > seq) { Next::Missing } else { Next::Wait });
        }
        if !index.status().is_success() {
            bail!("outcome index {seq}: HTTP {}", index.status());
        }
        let published_at = index
            .headers()
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| chrono::DateTime::parse_from_rfc2822(v).ok())
            .map(|t| t.timestamp());
        let index: Value = serde_json::from_slice(&index.bytes().await?).context("an outcome index that is not JSON")?;
        let key = index.get("key").and_then(Value::as_str).context("an outcome index without a key")?;
        let file = self.http.get(format!("{}/{}", self.base, key.trim_start_matches('/'))).send().await?;
        if !file.status().is_success() {
            bail!("outcome file {key}: HTTP {}", file.status());
        }
        let rows = read_parquet(file.bytes().await?)?;
        let published_at = published_at.or_else(|| rows.iter().map(|r| r.at.div_euclid(1_000_000)).max());
        Ok(Next::File { rows, published_at })
    }
}

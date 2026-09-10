//! A domain's and a sitemap's crawl state, as the bucket store keeps them.

use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::buckets::{sitemap_id, Json};
use crate::crawl::DomainWrite;
use crate::schedule::{Trust, DEFAULT_INTERVAL, SECOND};
use crate::states::State;
use crate::visit::{Known, KnownSitemap, SitemapUpdate};

/// A stored whole number, tolerating a float written for it.
pub fn int(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    value.as_i64().or_else(|| value.as_f64().map(|f| f as i64))
}

/// A stored Unix timestamp as epoch microseconds.
pub fn moment(value: Option<&Value>) -> Option<i64> {
    int(value).map(|seconds| seconds * SECOND)
}

fn seconds(micros: Option<i64>) -> Value {
    micros.map_or(Value::Null, |us| Value::from(us.div_euclid(SECOND)))
}

fn text(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_string)
}

/// A domain the process has not visited yet.
pub fn new_domain(
    rank: Option<i64>,
    group: Option<&str>,
    categories: &[String],
    state: State,
    reason: Option<&str>,
    due: Option<i64>,
) -> Json {
    let mut categories = categories.to_vec();
    categories.sort();
    let record = json!({
        "state": state.as_str(),
        "reason": reason,
        "failures": 0,
        "due": seconds(due),
        "rank": rank,
        "group": group,
        "categories": categories,
        "urls": 0,
    });
    record.as_object().cloned().unwrap_or_default()
}

/// What a visit needs to know about a domain before it starts.
pub fn known(host: &str, domain: &Json, sitemaps: &IndexMap<String, Json>) -> Option<Known> {
    Some(Known {
        host: host.to_string(),
        state: State::parse(domain.get("state")?.as_str()?)?,
        failures: int(domain.get("failures")).unwrap_or(0),
        last_ok_at: moment(domain.get("ok")),
        robots_checked_at: moment(domain.get("robots")),
        robots_allows: domain.get("allows").and_then(Value::as_bool),
        crawl_delay: domain.get("delay").and_then(Value::as_f64),
        language: text(domain.get("lang")),
        categories: domain
            .get("categories")
            .and_then(Value::as_array)
            .map(|all| all.iter().filter_map(|c| c.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        sitemaps: sitemaps.iter().filter_map(|(url, record)| Some((url.clone(), known_sitemap(url, record)?))).collect(),
        canonical_host: text(domain.get("canonical")),
    })
}

fn known_sitemap(url: &str, record: &Json) -> Option<KnownSitemap> {
    Some(KnownSitemap {
        id: int(record.get("id"))?,
        url: url.to_string(),
        kind: text(record.get("kind")),
        depth: int(record.get("depth")).unwrap_or(0),
        parent_id: int(record.get("parent")),
        etag: text(record.get("etag")),
        last_modified: text(record.get("modified")),
        content_hash: text(record.get("hash")),
        interval: record.get("interval").and_then(Value::as_f64).map_or(DEFAULT_INTERVAL, |s| (s * 1e6).round() as i64),
        next_check_at: moment(record.get("next")),
        trust: record.get("trust").and_then(Value::as_str).and_then(Trust::parse).unwrap_or(Trust::Unknown),
        index_lastmod: text(record.get("index_lastmod")),
        url_count: int(record.get("urls")).unwrap_or(0),
    })
}

/// The domain record after a visit: robots fields only when robots was read.
pub fn written_domain(previous: &Json, write: &DomainWrite, url_count: i64) -> Json {
    let visit = &write.visit;
    let mut record = previous.clone();
    record.insert("state".into(), write.state.as_str().into());
    record.insert("reason".into(), write.reason.clone().into());
    record.insert("failures".into(), write.failures.into());
    record.insert("due".into(), seconds(write.next_due_at));
    record.insert("ok".into(), seconds(write.last_ok_at));
    record.insert("checked".into(), seconds(Some(write.checked_at)));
    record.insert("canonical".into(), write.canonical_host.clone().into());
    record.insert("urls".into(), url_count.into());
    if visit.robots_read {
        record.insert("robots".into(), seconds(Some(write.checked_at)));
        record.insert("robots_status".into(), visit.robots_status.into());
        record.insert("allows".into(), visit.robots_allows.into());
        record.insert("delay".into(), visit.crawl_delay.into());
    }
    if let Some(language) = &visit.language {
        record.insert("lang".into(), language.clone().into());
    }
    if let Some(declared) = &visit.declared_lang {
        record.insert("declared".into(), declared.clone().into());
    }
    if let Some(chars) = visit.home_chars {
        record.insert("chars".into(), chars.into());
    }
    record
}

/// The sitemap record after a read; its counts change only when its content did.
pub fn written_sitemap(previous: Option<&Json>, update: &SitemapUpdate, at: i64) -> Json {
    let mut record = previous.cloned().unwrap_or_default();
    record.insert("id".into(), update.id.into());
    record.insert("depth".into(), update.depth.into());
    record.insert("parent".into(), update.parent_id.into());
    record.insert("status".into(), update.status.into());
    record.insert("error".into(), update.error.clone().into());
    record.insert("etag".into(), update.etag.clone().into());
    record.insert("modified".into(), update.last_modified.clone().into());
    record.insert("hash".into(), update.content_hash.clone().into());
    record.insert("trust".into(), update.trust.as_str().into());
    record.insert("index_lastmod".into(), update.index_lastmod.clone().into());
    record.insert("interval".into(), (update.interval / SECOND).into());
    record.insert("next".into(), seconds(update.next_check_at));
    record.insert("fetched".into(), seconds(Some(at)));
    if let Some(kind) = &update.kind {
        record.insert("kind".into(), kind.clone().into());
    }
    if update.changed {
        record.insert("urls".into(), update.url_count.into());
        record.insert("children".into(), update.child_count.into());
        record.insert("changed".into(), seconds(Some(at)));
    }
    record.entry("urls").or_insert(0.into());
    record
}

/// A sitemap found in an index that a later visit will read.
pub fn unread_sitemap(url: &str, depth: i64, parent: Option<i64>) -> Json {
    let record = json!({
        "id": sitemap_id(url),
        "depth": depth,
        "parent": parent,
        "status": "ok",
        "trust": Trust::Unknown.as_str(),
        "interval": DEFAULT_INTERVAL / SECOND,
        "next": null,
        "urls": 0,
    });
    record.as_object().cloned().unwrap_or_default()
}

/// URLs a domain lists across the sitemaps that last answered.
pub fn url_count<'a>(sitemaps: impl Iterator<Item = &'a Json>) -> i64 {
    sitemaps.filter(|s| s.get("status").and_then(Value::as_str) == Some("ok")).map(|s| int(s.get("urls")).unwrap_or(0)).sum()
}

//! The outcome rows of one written batch, for the bot that queued its URLs.

use desearch::canonical::domain_of;
use desearch::outcomes::{Outcome, DROPPED, FAILED, PUBLISHED, UNCHANGED};

use crate::changes::Change;
use crate::index::Withdrawn;
use crate::worker::{Failed, Page};

/// Published pages, pages seen again unchanged, URLs that failed, then pages taken back so the bot sends them again.
pub fn rows(changes: &[Change], unchanged: &[Page], failed: &[Failed], removed: &[Withdrawn]) -> Vec<Outcome> {
    let mut rows = Vec::with_capacity(changes.len() + unchanged.len() + failed.len() + removed.len());
    for change in changes {
        if let Some(record) = change.record() {
            rows.push((record.assigned_url.clone(), PUBLISHED, record.task_id.clone()));
        }
    }
    rows.extend(unchanged.iter().map(|page| (page.record.assigned_url.clone(), UNCHANGED, page.record.task_id.clone())));
    rows.extend(failed.iter().map(|f| (f.url.clone(), FAILED, f.task_id.clone())));
    rows.extend(removed.iter().map(|page| (page.url.clone(), DROPPED, String::new())));
    rows.into_iter().map(|(url, outcome, task_id)| Outcome { host: domain_of(&url).unwrap_or_default(), url, outcome, task_id }).collect()
}

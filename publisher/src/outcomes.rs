//! The outcome rows of one written batch, and of withdrawn pages taken out of the index, for the bot that queued their URLs.

use desearch::canonical::domain_of;
use desearch::outcomes::{Outcome, DROPPED, FAILED, PUBLISHED, UNCHANGED};

use crate::changes::Change;
use crate::index::Withdrawn;
use crate::worker::{Failed, Page};

/// Published pages, pages seen again unchanged, then URLs that failed.
pub fn rows(changes: &[Change], unchanged: &[Page], failed: &[Failed]) -> Vec<Outcome> {
    let mut rows = Vec::with_capacity(changes.len() + unchanged.len() + failed.len());
    for change in changes {
        if let Some(record) = change.record() {
            rows.push((record.assigned_url.clone(), PUBLISHED, record.task_id.clone()));
        }
    }
    rows.extend(unchanged.iter().map(|page| (page.record.assigned_url.clone(), UNCHANGED, page.record.task_id.clone())));
    rows.extend(failed.iter().map(|f| (f.url.clone(), FAILED, f.task_id.clone())));
    rows.into_iter().map(|(url, outcome, task_id)| outcome_of(url, outcome, task_id)).collect()
}

/// Pages taken back, so the bot sends them again.
pub fn dropped(removed: &[Withdrawn]) -> Vec<Outcome> {
    removed.iter().map(|page| outcome_of(page.url.clone(), DROPPED, String::new())).collect()
}

fn outcome_of(url: String, outcome: &'static str, task_id: String) -> Outcome {
    Outcome { host: domain_of(&url).unwrap_or_default(), url, outcome, task_id }
}

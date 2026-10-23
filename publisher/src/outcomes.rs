//! The outcome rows of one written batch, for the bot that queued their URLs.

use desearch::canonical::domain_of;
use desearch::outcomes::{Outcome, DROPPED, FAILED, PUBLISHED};

use crate::changes::Change;
use crate::worker::Failed;

/// Published pages, then URLs that failed.
pub fn rows(changes: &[Change], failed: &[Failed], dropped: &[Failed]) -> Vec<Outcome> {
    let mut rows = Vec::with_capacity(changes.len() + failed.len() + dropped.len());
    for change in changes {
        if let Some(record) = change.record() {
            rows.push((record.assigned_url.clone(), PUBLISHED, record.task_id.clone()));
        }
    }
    rows.extend(failed.iter().map(|f| (f.url.clone(), FAILED, f.task_id.clone())));
    rows.extend(dropped.iter().map(|f| (f.url.clone(), DROPPED, f.task_id.clone())));
    rows.into_iter().map(|(url, outcome, task_id)| outcome_of(url, outcome, task_id)).collect()
}

fn outcome_of(url: String, outcome: &'static str, task_id: String) -> Outcome {
    Outcome { host: domain_of(&url).unwrap_or_default(), url, outcome, task_id }
}

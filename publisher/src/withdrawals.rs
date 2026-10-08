//! Withdrawn tasks' pages taken out of the version index off the publish path, a chunk at a time.

use std::collections::HashSet;

use anyhow::Result;

use crate::index::{VersionIndex, Withdrawn, WITHDRAWN_KEEP_S};

/// Pages taken out of the index at a time, each chunk reported in one outcome file.
pub const CHUNK: usize = 5_000;

/// Drains every withdrawn task not in `drained` yet, passing each chunk's pages to `removed`, then forgets tasks drained and withdrawn over seven days before `now`; returns the pages taken out.
pub fn drain(
    index: &VersionIndex,
    drained: &mut HashSet<String>,
    now: f64,
    chunk: usize,
    stopping: &dyn Fn() -> bool,
    removed: &mut dyn FnMut(&[Withdrawn]) -> Result<()>,
) -> Result<usize> {
    let mut taken = 0;
    let mut expired = Vec::new();
    for (task_id, at) in index.withdrawn()? {
        let mut after: Option<String> = None;
        while !drained.contains(&task_id) {
            if stopping() {
                return Ok(taken);
            }
            let step = index.drain(&task_id, after.as_deref(), chunk, |pages| {
                if !pages.is_empty() {
                    removed(pages)?;
                    taken += pages.len();
                }
                Ok(())
            })?;
            match step {
                Some(last) => after = Some(last),
                None => {
                    drained.insert(task_id.clone());
                }
            }
        }
        if at < now - WITHDRAWN_KEEP_S {
            expired.push(task_id);
        }
    }
    index.forget_withdrawn(&expired)?;
    for task_id in &expired {
        drained.remove(task_id);
    }
    Ok(taken)
}

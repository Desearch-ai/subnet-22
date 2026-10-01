//! When each domain the process owns is next due, kept in memory. Times are Unix seconds.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use crate::states::State;

pub const UNRANKED: i64 = (1 << 31) - 1;

type Slot = Reverse<(i64, i64, Arc<str>)>;

/// Domains in the order they fall due: known sites first, then discovery by rank.
#[derive(Default)]
pub struct Timetable {
    refresh: BinaryHeap<Slot>,
    discovery: BinaryHeap<Slot>,
    due: HashMap<Arc<str>, i64>,
}

impl Timetable {
    pub fn len(&self) -> usize {
        self.due.len()
    }

    pub fn is_empty(&self) -> bool {
        self.due.is_empty()
    }

    /// Schedule a domain's next visit, replacing any earlier time; None takes it off.
    pub fn set(&mut self, host: &str, state: State, due: Option<i64>, rank: Option<i64>) {
        let Some(when) = due else {
            self.due.remove(host);
            return;
        };
        let host: Arc<str> = self.due.get_key_value(host).map_or_else(|| Arc::from(host), |(key, _)| key.clone());
        self.due.insert(host.clone(), when);
        let heap = if state.refreshes() { &mut self.refresh } else { &mut self.discovery };
        heap.push(Reverse((when, rank.unwrap_or(UNRANKED), host)));
    }

    /// Up to limit domains whose time has come; each leaves the timetable until set again.
    pub fn take(&mut self, limit: usize, now: i64) -> Vec<Arc<str>> {
        let mut taken = Vec::new();
        for heap in [&mut self.refresh, &mut self.discovery] {
            while taken.len() < limit {
                match heap.peek() {
                    Some(Reverse((when, _, _))) if *when <= now => {}
                    _ => break,
                }
                let Reverse((when, _, host)) = heap.pop().unwrap();
                if self.due.get(&host) == Some(&when) {
                    self.due.remove(&host);
                    taken.push(host);
                }
            }
        }
        taken
    }

    /// The earliest scheduled time.
    pub fn next_at(&self) -> Option<i64> {
        [&self.refresh, &self.discovery].into_iter().filter_map(|heap| heap.peek().map(|Reverse((when, _, _))| *when)).min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sites_come_before_discovery_and_rescheduling_replaces() {
        let mut table = Timetable::default();
        table.set("new.com", State::New, Some(10), Some(5));
        table.set("top.com", State::New, Some(10), Some(1));
        table.set("known.com", State::Active, Some(20), None);
        table.set("later.com", State::Active, Some(100), None);
        table.set("gone.com", State::New, Some(5), None);
        table.set("gone.com", State::Excluded, None, None);
        assert_eq!(table.take(10, 50).iter().map(|h| h.to_string()).collect::<Vec<_>>(), ["known.com", "top.com", "new.com"]);
        table.set("later.com", State::Active, Some(200), None);
        assert!(table.take(10, 150).is_empty());
        assert_eq!(table.next_at(), Some(200));
        assert_eq!(table.take(10, 250).len(), 1);
        assert!(table.is_empty());
    }
}

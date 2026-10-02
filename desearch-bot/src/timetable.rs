//! When each domain the process owns is next due, and which due domain goes first: the best ranked. Times are Unix seconds.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use crate::states::State;

pub const UNRANKED: i64 = (1 << 31) - 1;

/// The three kinds of work that share the visit slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lane {
    /// Known sites whose robots.txt or sitemaps are due again.
    Refresh,
    /// Known sites with sitemaps found but not read yet, continued a visit at a time.
    Backlog,
    /// Domains never checked, and ones without sitemaps checked again now and then.
    Discovery,
}

impl Lane {
    pub const ALL: [Lane; 3] = [Lane::Refresh, Lane::Backlog, Lane::Discovery];

    pub fn of(state: State, backlog: bool) -> Lane {
        match (state.refreshes(), backlog) {
            (false, _) => Lane::Discovery,
            (true, true) => Lane::Backlog,
            (true, false) => Lane::Refresh,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Lane::Refresh => "refresh",
            Lane::Backlog => "backlog",
            Lane::Discovery => "discovery",
        }
    }
}

type Waiting = Reverse<(i64, Lane, i64, Arc<str>)>;
type Ready = Reverse<(i64, i64, Arc<str>)>;

/// Domains wait by time; once due they queue in their lane by rank.
#[derive(Default)]
pub struct Timetable {
    waiting: BinaryHeap<Waiting>,
    ready: [BinaryHeap<Ready>; 3],
    due: HashMap<Arc<str>, (i64, Lane)>,
}

impl Timetable {
    pub fn len(&self) -> usize {
        self.due.len()
    }

    pub fn is_empty(&self) -> bool {
        self.due.is_empty()
    }

    /// Schedule a domain's next visit, replacing any earlier time; None takes it off.
    pub fn set(&mut self, host: &str, lane: Lane, due: Option<i64>, rank: Option<i64>) {
        let Some(when) = due else {
            self.due.remove(host);
            return;
        };
        let host: Arc<str> = self.due.get_key_value(host).map_or_else(|| Arc::from(host), |(key, _)| key.clone());
        self.due.insert(host.clone(), (when, lane));
        self.waiting.push(Reverse((when, lane, rank.unwrap_or(UNRANKED), host)));
    }

    /// Up to limit due domains of one lane, best ranked first; each leaves the timetable until set again.
    pub fn take(&mut self, lane: Lane, limit: usize, now: i64) -> Vec<Arc<str>> {
        self.promote(now);
        let mut taken = Vec::new();
        let ready = &mut self.ready[lane.index()];
        while taken.len() < limit {
            let Some(Reverse((_, when, host))) = ready.pop() else {
                break;
            };
            if self.due.get(&host) == Some(&(when, lane)) {
                self.due.remove(&host);
                taken.push(host);
            }
        }
        taken
    }

    /// Due domains queued in a lane, counting ones rescheduled since, which take() drops.
    pub fn queued(&self, lane: Lane) -> usize {
        self.ready[lane.index()].len()
    }

    /// The earliest scheduled time.
    pub fn next_at(&self) -> Option<i64> {
        let ready = self.ready.iter().filter_map(|heap| heap.peek().map(|Reverse((_, when, _))| *when));
        self.waiting.peek().map(|Reverse((when, ..))| *when).into_iter().chain(ready).min()
    }

    fn promote(&mut self, now: i64) {
        while let Some(Reverse((when, ..))) = self.waiting.peek() {
            if *when > now {
                break;
            }
            let Reverse((when, lane, rank, host)) = self.waiting.pop().unwrap();
            if self.due.get(&host) == Some(&(when, lane)) {
                self.ready[lane.index()].push(Reverse((rank, when, host)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(hosts: Vec<Arc<str>>) -> Vec<String> {
        hosts.iter().map(|h| h.to_string()).collect()
    }

    #[test]
    fn each_lane_serves_its_best_ranked_due_domains() {
        let mut table = Timetable::default();
        table.set("late.com", Lane::Refresh, Some(10), Some(900));
        table.set("top.com", Lane::Refresh, Some(40), Some(3));
        table.set("unranked.com", Lane::Refresh, Some(5), None);
        table.set("new.com", Lane::Discovery, Some(10), Some(50));
        table.set("later.com", Lane::Refresh, Some(100), Some(1));
        assert_eq!(names(table.take(Lane::Refresh, 10, 50)), ["top.com", "late.com", "unranked.com"]);
        assert_eq!(names(table.take(Lane::Discovery, 10, 50)), ["new.com"]);
        assert!(table.take(Lane::Backlog, 10, 50).is_empty());
        assert_eq!(table.next_at(), Some(100));
        assert_eq!(names(table.take(Lane::Refresh, 1, 150)), ["later.com"]);
        assert!(table.is_empty());
    }

    #[test]
    fn rescheduling_replaces_even_a_domain_already_queued() {
        let mut table = Timetable::default();
        table.set("site.com", Lane::Refresh, Some(10), Some(5));
        table.set("other.com", Lane::Refresh, Some(10), Some(6));
        assert_eq!(names(table.take(Lane::Refresh, 1, 20)), ["site.com"]);
        assert_eq!(table.queued(Lane::Refresh), 1);
        table.set("other.com", Lane::Backlog, Some(30), Some(6));
        table.set("gone.com", Lane::Discovery, Some(5), None);
        table.set("gone.com", Lane::Discovery, None, None);
        assert!(table.take(Lane::Refresh, 10, 40).is_empty());
        assert!(table.take(Lane::Discovery, 10, 40).is_empty());
        assert_eq!(names(table.take(Lane::Backlog, 10, 40)), ["other.com"]);
        assert!(table.is_empty());
    }

    #[test]
    fn lanes_follow_state_and_backlog() {
        assert_eq!(Lane::of(State::Active, false), Lane::Refresh);
        assert_eq!(Lane::of(State::Active, true), Lane::Backlog);
        assert_eq!(Lane::of(State::Failing, false), Lane::Refresh);
        assert_eq!(Lane::of(State::New, true), Lane::Discovery);
        assert_eq!(Lane::of(State::NoSitemap, false), Lane::Discovery);
    }
}

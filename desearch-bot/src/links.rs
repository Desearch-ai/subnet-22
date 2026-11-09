//! Link work waiting its turn: a bounded queue a freed runner takes the earliest due site from, at a capped pace, so a waiting site costs nothing.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

/// Sites whose link work waits, each under the time it is due, in Unix seconds.
pub struct LinkQueue {
    heap: BinaryHeap<Reverse<(i64, Arc<str>)>>,
    due: HashMap<Arc<str>, i64>,
    most: usize,
    /// Sites turned away while the queue was full; their next visit offers them again.
    pub dropped: u64,
}

impl LinkQueue {
    pub fn new(most: usize) -> Self {
        LinkQueue { heap: BinaryHeap::new(), due: HashMap::new(), most, dropped: 0 }
    }

    /// Queues a site, or moves it to a new time; a full queue takes no new site.
    pub fn push(&mut self, host: &str, at: i64) {
        let key = match self.due.get_key_value(host) {
            Some((_, held)) if *held == at => return,
            Some((key, _)) => key.clone(),
            None if self.due.len() >= self.most => {
                self.dropped += 1;
                return;
            }
            None => Arc::from(host),
        };
        self.due.insert(key.clone(), at);
        self.heap.push(Reverse((at, key)));
        // Moved sites leave their old places behind; the heap is rebuilt before those outgrow the sites.
        if self.heap.len() > 2 * self.due.len() + 1024 {
            self.heap = self.due.iter().map(|(host, at)| Reverse((*at, host.clone()))).collect();
        }
    }

    pub fn remove(&mut self, host: &str) {
        self.due.remove(host);
    }

    /// The site due earliest, if it is due by `now`.
    pub fn pop_due(&mut self, now: i64) -> Option<Arc<str>> {
        while let Some(Reverse((at, _))) = self.heap.peek() {
            if *at > now {
                return None;
            }
            let Reverse((at, host)) = self.heap.pop().expect("peeked");
            if self.due.get(&host) == Some(&at) {
                self.due.remove(&host);
                return Some(host);
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.due.len()
    }

    pub fn is_empty(&self) -> bool {
        self.due.is_empty()
    }

    /// Bytes the queue holds, roughly: each site's name twice over and its places.
    pub fn bytes(&self) -> usize {
        let names: usize = self.due.keys().map(|host| host.len() + 16).sum();
        names + self.due.capacity() * 40 + self.heap.capacity() * 24
    }
}

/// Link runs started an hour at most, at a steady pace that saves up a minute of it; 0 sets no limit.
pub struct LinkPace {
    per_hour: f64,
    tokens: f64,
    at: Instant,
}

impl LinkPace {
    pub fn new(per_hour: u64) -> Self {
        let per_hour = per_hour as f64;
        LinkPace { per_hour, tokens: Self::burst(per_hour), at: Instant::now() }
    }

    fn burst(per_hour: f64) -> f64 {
        (per_hour / 60.0).max(1.0)
    }

    /// Whether a run may start now; one that may takes its share.
    pub fn take(&mut self) -> bool {
        self.take_at(Instant::now())
    }

    fn take_at(&mut self, now: Instant) -> bool {
        if self.per_hour == 0.0 {
            return true;
        }
        let elapsed = now.saturating_duration_since(self.at).as_secs_f64();
        self.at = now;
        self.tokens = (self.tokens + elapsed * self.per_hour / 3600.0).min(Self::burst(self.per_hour));
        if self.tokens < 1.0 {
            return false;
        }
        self.tokens -= 1.0;
        true
    }

    /// Gives back the share of a run that only waited its turn.
    pub fn refund(&mut self) {
        if self.per_hour > 0.0 {
            self.tokens = (self.tokens + 1.0).min(Self::burst(self.per_hour));
        }
    }

    pub fn ready(&self) -> bool {
        self.per_hour == 0.0 || self.tokens + self.at.elapsed().as_secs_f64() * self.per_hour / 3600.0 >= 1.0
    }
}

/// Seconds after a site first qualifies before its first run: the same spot in `spread` for the same site, so turning link work on spreads first runs over that long.
pub fn first_wave(host: &str, spread: i64) -> i64 {
    if spread <= 0 {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    host.hash(&mut hasher);
    (hasher.finish() % spread as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_queue_hands_out_the_earliest_due_site_and_takes_none_past_its_bound() {
        let mut queue = LinkQueue::new(3);
        queue.push("b.test", 20);
        queue.push("a.test", 10);
        queue.push("c.test", 30);
        queue.push("d.test", 5);
        assert_eq!((queue.len(), queue.dropped), (3, 1), "full: the fourth waits for its next visit");
        queue.push("c.test", 1);
        assert_eq!(queue.pop_due(9).as_deref(), Some("c.test"), "moved earlier");
        assert_eq!(queue.pop_due(9), None, "nothing else due yet");
        assert_eq!(queue.pop_due(25).as_deref(), Some("a.test"));
        assert_eq!(queue.pop_due(25).as_deref(), Some("b.test"));
        assert!(queue.pop_due(i64::MAX).is_none() && queue.is_empty(), "c's old place is gone with it");
    }

    #[test]
    fn runs_start_no_faster_than_the_hourly_pace() {
        let start = Instant::now();
        let mut pace = LinkPace { per_hour: 120.0, tokens: 2.0, at: start };
        assert!(pace.take_at(start) && pace.take_at(start), "a minute saved up");
        assert!(!pace.take_at(start));
        assert!(!pace.take_at(start + Duration::from_secs(20)));
        assert!(pace.take_at(start + Duration::from_secs(31)), "one every 30 s at 120 an hour");
        assert!(LinkPace::new(0).take(), "no limit");
    }

    #[test]
    fn first_runs_spread_over_the_window_and_stay_put_for_a_site() {
        let spread = 3 * 86_400;
        let spots: Vec<i64> = (0..1000).map(|i| first_wave(&format!("site{i}.test"), spread)).collect();
        assert!(spots.iter().all(|&at| (0..spread).contains(&at)));
        let per_day = |day: i64| spots.iter().filter(|&&at| at / 86_400 == day).count();
        assert!((250..420).contains(&per_day(0)) && (250..420).contains(&per_day(2)), "about a third a day");
        assert_eq!(first_wave("site7.test", spread), spots[7]);
        assert_eq!(first_wave("site7.test", 0), 0);
    }
}

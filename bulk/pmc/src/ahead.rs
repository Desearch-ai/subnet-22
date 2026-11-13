//! A map over a thread pool, in input order, with at most `ahead` calls submitted and not yet taken.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;

pub struct BoundedMap<I: Iterator, R> {
    items: I,
    pool: rayon::ThreadPool,
    call: Arc<dyn Fn(I::Item) -> R + Send + Sync>,
    window: VecDeque<Receiver<R>>,
    ahead: usize,
}

pub fn bounded_map<I, R, F>(items: I, threads: usize, ahead: usize, call: F) -> BoundedMap<I, R>
where
    I: Iterator,
    I::Item: Send + 'static,
    R: Send + 'static,
    F: Fn(I::Item) -> R + Send + Sync + 'static,
{
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().expect("a thread pool");
    BoundedMap { items, pool, call: Arc::new(call), window: VecDeque::new(), ahead: ahead.max(1) }
}

impl<I, R> Iterator for BoundedMap<I, R>
where
    I: Iterator,
    I::Item: Send + 'static,
    R: Send + 'static,
{
    type Item = R;

    fn next(&mut self) -> Option<R> {
        while self.window.len() < self.ahead {
            let Some(item) = self.items.next() else { break };
            let (sender, receiver) = sync_channel(1);
            let call = self.call.clone();
            self.pool.spawn(move || {
                if let Ok(result) = catch_unwind(AssertUnwindSafe(|| call(item))) {
                    let _ = sender.send(result);
                }
            });
            self.window.push_back(receiver);
        }
        let receiver = self.window.pop_front()?;
        Some(receiver.recv().expect("a bounded_map call panicked"))
    }
}

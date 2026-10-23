//! Sitemaps that list new URLs on every read, as a hacked site's injected sitemaps do: a suspect read is held back and read again minutes later to decide.

use crate::listed::fingerprint;
use crate::schedule::{DAY, SECOND};

/// Fewer URLs than this say too little either way.
pub const MIN_URLS: usize = 500;
/// A re-read is suspect when this share of its URLs is new to the domain.
pub const NEW_SHARE: f64 = 0.9;
/// A second read minutes later that shares less than this with the suspect one is generated.
pub const SAME_SHARE: f64 = 0.1;
pub const CONFIRM_AFTER: i64 = 300 * SECOND;
/// A suspect read not confirmed within this long is forgotten, and the sitemap judged afresh.
pub const SUSPECT_FOR: i64 = 6 * 3600 * SECOND;
pub const GENERATED_FOR: i64 = 7 * DAY;
/// URLs kept to compare two reads: the ones with the smallest hashes, the same ones whenever the lists are the same.
pub const SKETCH: usize = 128;

/// What the store knows about a sitemap before this read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Standing {
    /// It listed URLs before, so a read of it is a re-read.
    pub listed_before: bool,
    pub suspect_at: Option<i64>,
    pub sketch: Vec<u64>,
    pub generated_until: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Recorded as usual; a suspect mark it had is cleared.
    Clear,
    /// Held back until a second read decides.
    Suspect { sketch: Vec<u64> },
    /// Held back until the time given.
    Generated { until: i64 },
}

/// The URL keys' hashes, smallest first, at most `SKETCH`.
pub fn sketch<'a>(keys: impl Iterator<Item = &'a [u8]>) -> Vec<u64> {
    let mut hashes: Vec<u64> = keys.map(|key| fingerprint(key, 0)).collect();
    hashes.sort_unstable();
    hashes.dedup();
    hashes.truncate(SKETCH);
    hashes
}

/// How much two lists share, estimated from their sketches: of the smallest hashes of both together, the part both hold.
pub fn same_share(a: &[u64], b: &[u64]) -> f64 {
    let mut union: Vec<u64> = a.iter().chain(b).copied().collect();
    union.sort_unstable();
    union.dedup();
    union.truncate(SKETCH.min(a.len()).min(b.len()).max(1));
    let both = union.iter().filter(|h| a.binary_search(h).is_ok() && b.binary_search(h).is_ok()).count();
    both as f64 / union.len() as f64
}

/// Decides a read from what the sitemap was before, its URLs' sketch, and the share of them the domain has never seen.
pub fn judge(standing: &Standing, listed: usize, sketch: Vec<u64>, new_share: impl FnOnce() -> f64, now: i64) -> Verdict {
    if let Some(until) = standing.generated_until.filter(|&until| until > now) {
        return Verdict::Generated { until };
    }
    let pending = standing.suspect_at.filter(|&at| now - at < SUSPECT_FOR && !standing.sketch.is_empty());
    if pending.is_some() {
        return if listed >= MIN_URLS && same_share(&standing.sketch, &sketch) < SAME_SHARE {
            Verdict::Generated { until: now + GENERATED_FOR }
        } else {
            Verdict::Clear
        };
    }
    if standing.listed_before && listed >= MIN_URLS && new_share() >= NEW_SHARE {
        return Verdict::Suspect { sketch };
    }
    Verdict::Clear
}

pub fn encode(sketch: &[u64]) -> String {
    sketch.iter().map(|h| format!("{h:016x}")).collect()
}

pub fn decode(text: &str) -> Vec<u64> {
    (0..text.len() / 16).filter_map(|i| text.get(i * 16..i * 16 + 16)).filter_map(|h| u64::from_str_radix(h, 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(prefix: &str, count: usize) -> Vec<Vec<u8>> {
        (0..count).map(|i| format!("example.com\0example.com/{prefix}/{i}").into_bytes()).collect()
    }

    fn sketched(keys: &[Vec<u8>]) -> Vec<u64> {
        sketch(keys.iter().map(Vec::as_slice))
    }

    #[test]
    fn two_reads_of_the_same_list_share_everything_and_two_random_lists_nothing() {
        let (a, b) = (keys("a", 3000), keys("b", 3000));
        assert_eq!(same_share(&sketched(&a), &sketched(&a)), 1.0);
        assert_eq!(same_share(&sketched(&a), &sketched(&b)), 0.0);
        let rolled: Vec<Vec<u8>> = a[300..].iter().cloned().chain(keys("c", 300)).collect();
        let share = same_share(&sketched(&a), &sketched(&rolled));
        assert!((0.7..=0.95).contains(&share), "a list that moved a tenth on still shares most of itself: {share}");
        assert_eq!(decode(&encode(&sketched(&a))), sketched(&a));
    }

    #[test]
    fn a_re_read_of_new_urls_is_suspect_and_a_second_read_decides() {
        let now = 1_000 * SECOND;
        let before = Standing { listed_before: true, ..Standing::default() };
        let first = sketched(&keys("first", 3000));
        assert_eq!(judge(&Standing::default(), 3000, first.clone(), || 1.0, now), Verdict::Clear, "a first read is never judged");
        assert_eq!(judge(&before, 400, first.clone(), || 1.0, now), Verdict::Clear, "too few URLs");
        assert_eq!(judge(&before, 3000, first.clone(), || 0.5, now), Verdict::Clear, "half of them known");
        let Verdict::Suspect { sketch } = judge(&before, 3000, first.clone(), || 0.95, now) else { panic!("suspect") };

        let suspect = Standing { listed_before: true, suspect_at: Some(now), sketch, generated_until: None };
        let later = now + CONFIRM_AFTER;
        assert_eq!(judge(&suspect, 3000, first.clone(), || 1.0, later), Verdict::Clear, "the same URLs again: a real list, recorded now");
        let other = sketched(&keys("other", 3000));
        assert_eq!(judge(&suspect, 3000, other.clone(), || 1.0, later), Verdict::Generated { until: later + GENERATED_FOR });
        assert_eq!(judge(&suspect, 3000, other.clone(), || 1.0, now + SUSPECT_FOR), Verdict::Suspect { sketch: other }, "too late to confirm: judged afresh");

        let generated = Standing { generated_until: Some(later + GENERATED_FOR), ..before };
        assert_eq!(judge(&generated, 3000, first.clone(), || 0.0, later + DAY), Verdict::Generated { until: later + GENERATED_FOR });
        assert_eq!(judge(&generated, 3000, first, || 0.0, later + GENERATED_FOR), Verdict::Clear, "a week on, judged afresh");
    }
}

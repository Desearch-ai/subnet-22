//! How often to re-read a sitemap, and whether to believe its dates. Times are epoch microseconds.

use crate::text;

pub const SECOND: i64 = 1_000_000;
pub const MINUTE: i64 = 60 * SECOND;
pub const HOUR: i64 = 60 * MINUTE;
pub const DAY: i64 = 24 * HOUR;

pub const MIN_INTERVAL: i64 = 10 * MINUTE;
pub const MAX_INTERVAL: i64 = 7 * DAY;
pub const DEFAULT_INTERVAL: i64 = DAY;
pub const NEWS_INTERVAL: i64 = HOUR;

const MIN_DATED: usize = 20;
const AGREEMENT: f64 = 0.9;
const STAMP_WINDOW: i64 = HOUR;
const EARLIEST: i64 = 788_918_400 * SECOND;
const FUTURE_SLACK: i64 = DAY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trust {
    Unknown,
    Suspect,
    Trusted,
    Untrusted,
}

impl Trust {
    pub fn as_str(self) -> &'static str {
        match self {
            Trust::Unknown => "unknown",
            Trust::Suspect => "suspect",
            Trust::Trusted => "trusted",
            Trust::Untrusted => "untrusted",
        }
    }

    pub fn parse(value: &str) -> Option<Trust> {
        match value {
            "unknown" => Some(Trust::Unknown),
            "suspect" => Some(Trust::Suspect),
            "trusted" => Some(Trust::Trusted),
            "untrusted" => Some(Trust::Untrusted),
            _ => None,
        }
    }
}

/// Where a new sitemap's schedule starts, before its behaviour is known.
pub fn first_interval(changefreq: Option<&str>, news: bool) -> i64 {
    let declared = match text::strip(changefreq.unwrap_or("")).to_lowercase().as_str() {
        "always" => MIN_INTERVAL,
        "hourly" => HOUR,
        "daily" => DAY,
        "weekly" | "monthly" | "yearly" | "never" => MAX_INTERVAL,
        _ => DEFAULT_INTERVAL,
    };
    if news {
        declared.min(NEWS_INTERVAL)
    } else {
        declared
    }
}

/// Look sooner at a file that changed, and later at one that did not.
pub fn next_interval(current: i64, changed: bool) -> i64 {
    let scaled = if changed { round_half_even(current, 2) } else { round_half_even(current * 3, 2) };
    scaled.clamp(MIN_INTERVAL, MAX_INTERVAL)
}

pub fn plausible(date: Option<i64>, fetched_at: i64) -> Option<i64> {
    date.filter(|&d| d >= EARLIEST && d <= fetched_at + FUTURE_SLACK)
}

/// Whether a sitemap's lastmod values carry information, judged from one read.
pub fn assess_dates(dates: &[Option<i64>], fetched_at: i64, previous: Trust) -> Trust {
    let mut dated: Vec<i64> = dates.iter().filter_map(|&d| plausible(d, fetched_at)).collect();
    if dated.len() < MIN_DATED {
        return previous;
    }
    let needed = AGREEMENT * dated.len() as f64;
    let stamped = dated.iter().filter(|&&d| (fetched_at - d).abs() <= STAMP_WINDOW).count();
    dated.sort_unstable();
    let most_common = dated.chunk_by(|a, b| a == b).map(<[i64]>::len).max().unwrap_or(0);
    if most_common as f64 >= needed {
        return Trust::Untrusted;
    }
    if stamped as f64 >= needed {
        return if previous == Trust::Suspect { Trust::Untrusted } else { Trust::Suspect };
    }
    Trust::Trusted
}

pub fn relies_on_dates(trust: Trust) -> bool {
    trust == Trust::Trusted
}

/// Python's `timedelta * float` rounding: the nearest microsecond, halves to even.
pub fn round_half_even(numerator: i64, denominator: i64) -> i64 {
    let quotient = numerator.div_euclid(denominator);
    let twice = 2 * numerator.rem_euclid(denominator);
    if twice > denominator || (twice == denominator && quotient % 2 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervals_move_within_bounds() {
        assert_eq!(first_interval(Some(" Weekly "), false), MAX_INTERVAL);
        assert_eq!(first_interval(Some("daily"), true), NEWS_INTERVAL);
        assert_eq!(next_interval(DAY, true), 12 * HOUR);
        assert_eq!(next_interval(MAX_INTERVAL, false), MAX_INTERVAL);
        assert_eq!(next_interval(MIN_INTERVAL, true), MIN_INTERVAL);
    }

    #[test]
    fn identical_or_stamped_dates_lose_trust() {
        let now = 1_700_000_000 * SECOND;
        let same = vec![Some(now - DAY); 30];
        assert_eq!(assess_dates(&same, now, Trust::Unknown), Trust::Untrusted);
        let stamped: Vec<_> = (0..30).map(|i| Some(now - i * SECOND)).collect();
        assert_eq!(assess_dates(&stamped, now, Trust::Unknown), Trust::Suspect);
        assert_eq!(assess_dates(&stamped, now, Trust::Suspect), Trust::Untrusted);
        let spread: Vec<_> = (0..30).map(|i| Some(now - i * DAY)).collect();
        assert_eq!(assess_dates(&spread, now, Trust::Unknown), Trust::Trusted);
    }
}

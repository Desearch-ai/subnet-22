//! Calendar dates of Unix times, without a date library.

use std::time::{SystemTime, UNIX_EPOCH};

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// Seconds since the epoch, as Python's `time.time()`.
pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Year, month, day, hour, minute and second of a Unix second in UTC.
pub fn utc(seconds: i64) -> [i64; 6] {
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let rem = seconds.rem_euclid(86_400);
    [year, month, day, rem / 3600, rem % 3600 / 60, rem % 60]
}

/// `2026-10-08`, the day partition in object keys.
pub fn utc_day(seconds: f64) -> String {
    let [year, month, day, ..] = utc(seconds.floor() as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

/// The Unix second of an HTTP date such as `Wed, 21 Oct 2015 07:28:00 GMT`.
pub fn parse_http_date(text: &str) -> Option<i64> {
    let mut parts = text.split_whitespace().skip(1);
    let day: i64 = parts.next()?.parse().ok()?;
    let month = parts.next()?;
    let month = MONTHS.iter().position(|m| *m == month)? as i64 + 1;
    let year: i64 = parts.next()?.parse().ok()?;
    let mut clock = parts.next()?.split(':').map(|n| n.parse::<i64>().ok());
    let (hour, minute, second) = (clock.next()??, clock.next()??, clock.next()??);
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second)
}

pub fn http_date(seconds: i64) -> String {
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    let [year, month, day, hour, minute, second] = utc(seconds);
    let weekday = DAYS[seconds.div_euclid(86_400).rem_euclid(7) as usize];
    format!("{weekday}, {day:02} {} {year:04} {hour:02}:{minute:02}:{second:02} GMT", MONTHS[month as usize - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip() {
        assert_eq!(utc_day(1_791_324_600.5), "2026-10-06");
        assert_eq!(http_date(1_445_412_480), "Wed, 21 Oct 2015 07:28:00 GMT");
        assert_eq!(parse_http_date("Wed, 21 Oct 2015 07:28:00 GMT"), Some(1_445_412_480));
        assert_eq!(parse_http_date("not a date"), None);
    }
}

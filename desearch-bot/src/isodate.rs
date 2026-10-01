//! Python's `datetime.fromisoformat`, ported from CPython's C parser so every lastmod reads the same.

use chrono::{NaiveDate, Weekday};

use crate::text;

/// A time as UTC epoch microseconds; text without an offset counts as UTC.
pub fn fromisoformat(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let separator = find_separator(b)?;
    let (mut year, mut month, mut day) = parse_date(b, separator)?;
    let mut time = Time::default();
    let mut offset_us = 0i64;
    if b.len() > separator {
        let lead = at(b, separator);
        let width = if lead & 0x80 == 0 {
            1
        } else {
            match lead & 0xf0 {
                0xe0 => 3,
                0xf0 => 4,
                _ => 2,
            }
        };
        (time, offset_us) = parse_time(b, separator + width)?;
    }
    if time.hour == 24 && month <= 12 {
        let days = days_in_month(year, month);
        if day <= days {
            if time.minute != 0 || time.second != 0 || time.micro != 0 {
                return None;
            }
            time.hour = 0;
            day += 1;
            if day > days {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }
    }
    if !(1..=9999).contains(&year) || time.hour > 23 || time.minute > 59 || time.second > 59 {
        return None;
    }
    let local = NaiveDate::from_ymd_opt(year, month, day)?
        .and_hms_micro_opt(time.hour, time.minute, time.second, time.micro)?
        .and_utc()
        .timestamp_micros();
    Some(local - offset_us)
}

/// A lastmod: the whole value, else its first 19 or 10 characters.
pub fn parse_lastmod(value: Option<&str>) -> Option<i64> {
    let value = value.filter(|v| !v.is_empty())?;
    let text = text::strip(value).replace('Z', "+00:00");
    for candidate in [text.as_str(), text::head(&text, 19), text::head(&text, 10)] {
        if let Some(parsed) = fromisoformat(candidate) {
            return Some(parsed);
        }
    }
    None
}

#[derive(Default)]
struct Time {
    hour: u32,
    minute: u32,
    second: u32,
    micro: u32,
}

fn at(b: &[u8], i: usize) -> u8 {
    b.get(i).copied().unwrap_or(0)
}

fn digits(b: &[u8], p: &mut usize, count: usize) -> Option<u32> {
    let mut value = 0u32;
    for _ in 0..count {
        let c = at(b, *p);
        *p += 1;
        if !c.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u32::from(c - b'0');
    }
    Some(value)
}

fn find_separator(b: &[u8]) -> Option<usize> {
    let len = b.len();
    if len == 7 {
        return Some(7);
    }
    if at(b, 4) == b'-' {
        if at(b, 5) != b'W' {
            return Some(10);
        }
        if len < 8 {
            return None;
        }
        if len > 8 && at(b, 8) == b'-' {
            if len == 9 {
                return None;
            }
            if len > 10 && at(b, 10).is_ascii_digit() {
                return Some(8);
            }
            return Some(10);
        }
        return Some(8);
    }
    if at(b, 4) == b'W' {
        let mut idx = 7;
        while idx < len && at(b, idx).is_ascii_digit() {
            idx += 1;
        }
        if idx < 9 {
            return Some(idx);
        }
        return Some(if idx % 2 == 0 { 7 } else { 8 });
    }
    Some(8)
}

fn parse_date(b: &[u8], len: usize) -> Option<(i32, u32, u32)> {
    let mut p = 0;
    let year = digits(b, &mut p, 4)?;
    let uses_separator = at(b, p) == b'-';
    if uses_separator {
        p += 1;
    }
    let dash = |p: &mut usize| {
        if !uses_separator {
            return true;
        }
        let c = at(b, *p);
        *p += 1;
        c == b'-'
    };
    if at(b, p) == b'W' {
        p += 1;
        let week = digits(b, &mut p, 2)?;
        let weekday = if p < len {
            if !dash(&mut p) {
                return None;
            }
            digits(b, &mut p, 1)?
        } else {
            1
        };
        return iso_to_ymd(year as i32, week, weekday);
    }
    let month = digits(b, &mut p, 2)?;
    if !dash(&mut p) {
        return None;
    }
    let day = digits(b, &mut p, 2)?;
    Some((year as i32, month, day))
}

fn iso_to_ymd(year: i32, week: u32, weekday: u32) -> Option<(i32, u32, u32)> {
    use chrono::Datelike;
    if !(1..=9999).contains(&year) || !(1..=7).contains(&weekday) {
        return None;
    }
    let weekday = Weekday::try_from(weekday as u8 - 1).ok()?;
    let date = NaiveDate::from_isoywd_opt(year, week, weekday)?;
    Some((date.year(), date.month(), date.day()))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

/// The time and offset after the date; the offset is in microseconds east of UTC.
fn parse_time(b: &[u8], start: usize) -> Option<(Time, i64)> {
    let end = b.len();
    let mut tz = start;
    loop {
        if matches!(at(b, tz), b'Z' | b'+' | b'-') {
            break;
        }
        tz += 1;
        if tz >= end {
            break;
        }
    }
    let (time, trailing) = hh_mm_ss_ff(b, start, tz)?;
    if tz >= end {
        return if trailing { None } else { Some((time, 0)) };
    }
    if at(b, tz) == b'Z' {
        return if at(b, tz + 1) != 0 { None } else { Some((time, 0)) };
    }
    let sign = if at(b, tz) == b'-' { -1 } else { 1 };
    let (offset, trailing) = hh_mm_ss_ff(b, tz + 1, end)?;
    if trailing {
        return None;
    }
    let seconds = i64::from(offset.hour * 3600 + offset.minute * 60 + offset.second);
    if seconds == 0 {
        return Some((time, 0));
    }
    let micros = sign * (seconds * 1_000_000 + i64::from(offset.micro));
    if micros.abs() >= 86_400_000_000 {
        return None;
    }
    Some((time, micros))
}

/// CPython's `parse_hh_mm_ss_ff`; the flag says text follows where the string should have ended.
fn hh_mm_ss_ff(b: &[u8], start: usize, end: usize) -> Option<(Time, bool)> {
    let mut values = [0u32; 3];
    let mut p = start;
    let mut has_separator = true;
    let mut fraction = false;
    for i in 0..3 {
        values[i] = digits(b, &mut p, 2)?;
        let c = at(b, p);
        p += 1;
        if i == 0 {
            has_separator = c == b':';
        }
        if p >= end {
            let time = Time { hour: values[0], minute: values[1], second: values[2], micro: 0 };
            return Some((time, c != 0));
        } else if has_separator && c == b':' {
            if i == 2 {
                return None;
            }
        } else if c == b'.' || c == b',' {
            if i < 2 {
                return None;
            }
            fraction = true;
            break;
        } else if !has_separator {
            p -= 1;
        } else {
            return None;
        }
    }
    let mut micro = 0;
    if fraction || p < end {
        let count = (end - p).min(6);
        micro = digits(b, &mut p, count)?;
        micro *= 10u32.pow(6 - count as u32);
        while at(b, p).is_ascii_digit() {
            p += 1;
        }
    }
    let time = Time { hour: values[0], minute: values[1], second: values[2], micro };
    Some((time, at(b, p) != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso(s: &str) -> Option<String> {
        fromisoformat(s).map(|us| chrono::DateTime::from_timestamp_micros(us).unwrap().format("%Y-%m-%dT%H:%M:%S%.6f").to_string())
    }

    #[test]
    fn reads_what_python_reads() {
        assert_eq!(iso("2024-01-05").unwrap(), "2024-01-05T00:00:00.000000");
        assert_eq!(iso("2024-W01-2").unwrap(), "2024-01-02T00:00:00.000000");
        assert_eq!(iso("2024-01-05T10:20:30.1").unwrap(), "2024-01-05T10:20:30.100000");
        assert_eq!(iso("2024-01-05T10:20:30+05:30").unwrap(), "2024-01-05T04:50:30.000000");
        assert_eq!(iso("2024-01-05T10:20:30 +05:30").unwrap(), "2024-01-05T04:50:30.000000");
        assert_eq!(iso("2024-01-05T24:00:00").unwrap(), "2024-01-06T00:00:00.000000");
        assert_eq!(iso("2024-01-05T102030").unwrap(), "2024-01-05T10:20:30.000000");
        for bad in ["2024-1-5", "2024-01", "2024-01-05T10:20:30.", "2024-01-05T10:20:30z", "2024-01-05T10:20:30+24:00", "2024-01-05 "] {
            assert_eq!(iso(bad), None, "{bad}");
        }
    }
}

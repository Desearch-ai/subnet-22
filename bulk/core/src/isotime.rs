//! CPython 3.13's C `datetime.fromisoformat`, byte for byte over the UTF-8 text, as microseconds and a UTC offset.

use desearch::time::days_from_civil;

/// A parsed time: microseconds since the epoch of its wall clock, and its offset in microseconds when it has one.
pub struct IsoTime {
    pub wall_us: i64,
    pub offset_us: Option<i64>,
}

struct Text<'a>(&'a [u8]);

impl Text<'_> {
    fn at(&self, i: usize) -> u8 {
        self.0.get(i).copied().unwrap_or(0)
    }

    fn digits(&self, at: usize, n: usize) -> Option<(usize, i64)> {
        let mut value = 0;
        for i in at..at + n {
            let c = self.at(i);
            if !c.is_ascii_digit() {
                return None;
            }
            value = value * 10 + i64::from(c - b'0');
        }
        Some((at + n, value))
    }
}

fn separator(t: &Text, len: usize) -> usize {
    if len == 7 {
        return 7;
    }
    if t.at(4) == b'-' {
        if t.at(5) != b'W' {
            return 10;
        }
        if len > 8 && t.at(8) == b'-' {
            if len == 9 {
                return usize::MAX;
            }
            return if len > 10 && t.at(10).is_ascii_digit() { 8 } else { 10 };
        }
        return 8;
    }
    if t.at(4) == b'W' {
        let mut idx = 7;
        while idx < len && t.at(idx).is_ascii_digit() {
            idx += 1;
        }
        if idx < 9 {
            return idx;
        }
        return if idx % 2 == 0 { 7 } else { 8 };
    }
    8
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn iso_to_ymd(year: i64, week: i64, day: i64) -> Option<(i64, i64, i64)> {
    if !(1..=9999).contains(&year) {
        return None;
    }
    let jan1 = days_from_civil(year, 1, 1);
    // 1970-01-01 was a Thursday; Monday is 0.
    let jan1_weekday = (jan1 + 3).rem_euclid(7);
    if !(1..53).contains(&week) {
        let long_year = jan1_weekday == 3 || (jan1_weekday == 2 && is_leap(year));
        if week != 53 || !long_year {
            return None;
        }
    }
    if !(1..8).contains(&day) {
        return None;
    }
    let week1_monday = if jan1_weekday > 3 { jan1 - jan1_weekday + 7 } else { jan1 - jan1_weekday };
    Some(desearch::time::civil_from_days(week1_monday + (week - 1) * 7 + day - 1))
}

fn date(t: &Text, len: usize) -> Option<(i64, i64, i64)> {
    let (mut p, year) = t.digits(0, 4)?;
    let dashed = t.at(p) == b'-';
    p += usize::from(dashed);
    if t.at(p) == b'W' {
        let (q, week) = t.digits(p + 1, 2)?;
        p = q;
        let mut day = 1;
        if p < len {
            if dashed {
                if t.at(p) != b'-' {
                    return None;
                }
                p += 1;
            }
            day = t.digits(p, 1)?.1;
        }
        return iso_to_ymd(year, week, day);
    }
    let (mut p, month) = t.digits(p, 2)?;
    if dashed {
        if t.at(p) != b'-' {
            return None;
        }
        p += 1;
    }
    let (_, day) = t.digits(p, 2)?;
    Some((year, month, day))
}

/// `parse_hh_mm_ss_ff`: hour, minute, second, microsecond, and whether text is left before the NUL terminator.
fn clock(t: &Text, start: usize, end: usize) -> Option<([i64; 4], bool)> {
    let mut parts = [0i64; 4];
    let mut p = start;
    let mut dashed = true;
    for i in 0..3 {
        let (q, value) = t.digits(p, 2)?;
        parts[i] = value;
        let c = t.at(q);
        p = q + 1;
        if i == 0 {
            dashed = c == b':';
        }
        if p >= end {
            return Some((parts, c != 0));
        } else if dashed && c == b':' {
            continue;
        } else if c == b'.' || c == b',' {
            break;
        } else if !dashed {
            p -= 1;
        } else {
            return None;
        }
    }
    let to_parse = (end - p).min(6);
    let (mut q, mut micros) = t.digits(p, to_parse)?;
    if to_parse < 6 {
        micros *= 10i64.pow(6 - to_parse as u32);
    }
    parts[3] = micros;
    while t.at(q).is_ascii_digit() {
        q += 1;
    }
    Some((parts, t.at(q) != 0))
}

fn time(t: &Text, start: usize, len: usize) -> Option<([i64; 4], Option<i64>)> {
    let mut tz = start;
    loop {
        if matches!(t.at(tz), b'Z' | b'+' | b'-') {
            break;
        }
        tz += 1;
        if tz >= len {
            break;
        }
    }
    let (parts, rest) = clock(t, start, tz)?;
    if tz == len {
        return if rest { None } else { Some((parts, None)) };
    }
    if t.at(tz) == b'Z' {
        return if t.at(tz + 1) != 0 { None } else { Some((parts, Some(0))) };
    }
    let sign = if t.at(tz) == b'-' { -1 } else { 1 };
    let (offset, rest) = clock(t, tz + 1, len)?;
    if rest {
        return None;
    }
    let seconds = sign * (offset[0] * 3600 + offset[1] * 60 + offset[2]);
    // CPython takes a whole-second offset of 0 as UTC and drops its fraction.
    let micros = if seconds == 0 { 0 } else { seconds * 1_000_000 + sign * offset[3] };
    Some((parts, Some(micros)))
}

/// The first non-ASCII character at a place a separator can be (7, 8 or 10) as `T`, as CPython sanitizes it.
fn ascii_separator(text: &str) -> std::borrow::Cow<'_, str> {
    for at in [7, 8, 10] {
        match text.char_indices().nth(at) {
            Some((i, c)) if !c.is_ascii() => return format!("{}T{}", &text[..i], &text[i + c.len_utf8()..]).into(),
            Some(_) => {}
            None => break,
        }
    }
    text.into()
}

/// `datetime.fromisoformat(text)`, or None where Python raises.
pub fn fromisoformat(text: &str) -> Option<IsoTime> {
    if text.chars().count() < 7 {
        return None;
    }
    let text = ascii_separator(text);
    let t = Text(text.as_bytes());
    let len = t.0.len();
    let sep = separator(&t, len);
    if sep == usize::MAX {
        return None;
    }
    let (year, month, day) = date(&t, sep)?;
    let (parts, offset) = if len > sep { time(&t, sep + 1, len)? } else { ([0; 4], None) };
    let [hour, minute, second, micros] = parts;
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) || !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    if offset.is_some_and(|o| o.abs() >= 86_400 * 1_000_000) {
        return None;
    }
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    Some(IsoTime { wall_us: seconds * 1_000_000 + micros, offset_us: offset })
}

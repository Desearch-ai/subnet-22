//! The record published for each page, its key and its version, built exactly as `publisher.records` builds them.

use anyhow::{bail, Result};
use sha1::{Digest, Sha1};

use desearch::canonical::{self, canonicalize, domain_of, hex, url_sha1};
use desearch::time::{civil_from_days, days_from_civil};

pub const PREFIX: &str = "pages";
pub const SOURCE: &str = "subnet22";
pub const ROW_COLUMNS: [&str; 16] = [
    "url",
    "final_url",
    "status",
    "error",
    "fetched_at",
    "page_type",
    "title",
    "description",
    "lang",
    "canonical",
    "published",
    "author",
    "json_ld_types",
    "headings",
    "text",
    "text_sha256",
];
pub const SECOND: i64 = 1_000_000;
const CLOCK_SKEW: i64 = 300 * SECOND;
const DEFAULT_CLAIM_S: f64 = 900.0;
/// Python's `datetime.min` and `datetime.max`, in microseconds since the epoch.
pub const MIN_US: i64 = -62_135_596_800 * SECOND;
pub const MAX_US: i64 = 253_402_300_800 * SECOND - 1;
const NAMESPACE_URL: [u8; 16] = [0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8];

/// One row of an upload: the columns the publisher reads, None where the file holds a null.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub url: Option<String>,
    pub final_url: Option<String>,
    pub status: Option<i32>,
    pub error: Option<String>,
    /// Microseconds since the epoch; None where the column is null or not a timestamp.
    pub fetched_at: Option<i64>,
    pub page_type: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub lang: Option<String>,
    pub canonical: Option<String>,
    pub published: Option<String>,
    pub author: Option<String>,
    pub json_ld_types: Option<Vec<Option<String>>>,
    pub headings: Option<Vec<Option<String>>>,
    pub text: Option<String>,
    pub text_sha256: Option<String>,
}

/// A page as published; Python's record also carries `html`, `lastmod` and `etag`, always empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub url: String,
    pub domain: String,
    pub title: String,
    pub published: String,
    pub author: String,
    pub lang: String,
    pub text: String,
    pub fetched_at: String,
    pub content_sha1: String,
    pub captured_at: String,
    pub doc_id: String,
    pub assigned_url: String,
    pub final_url: Option<String>,
    pub canonical: Option<String>,
    pub status: Option<i32>,
    pub page_type: Option<String>,
    pub description: Option<String>,
    pub json_ld_types: Vec<Option<String>>,
    pub headings: Vec<Option<String>>,
    pub text_sha256: Option<String>,
    pub task_id: String,
    pub miner: String,
    pub validator: Option<String>,
    pub validators: Vec<String>,
}

/// The earliest and latest fetch time a task's rows may claim, in microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub earliest: i64,
    pub latest: i64,
}

/// A row's fetch time must fall between claim and completion; None is a job value Python reads as false.
pub fn publish_window(completed_at: Option<f64>, claim_ttl: Option<f64>, now: i64) -> Result<Window> {
    let latest = match completed_at {
        Some(completed) => from_timestamp(completed)?,
        None => now,
    };
    let claim = timedelta(claim_ttl.unwrap_or(DEFAULT_CLAIM_S))?;
    Ok(Window { earliest: in_range(latest - claim - CLOCK_SKEW)?, latest })
}

pub struct Context<'a> {
    pub task_id: &'a str,
    pub miner: &'a str,
    pub window: Window,
    pub captured_at: i64,
    pub validator: Option<&'a str>,
    pub validators: &'a [String],
}

pub fn build_record(row: &Row, context: &Context) -> Result<Record> {
    let assigned = row.url.clone().unwrap_or_default();
    let url = canonicalize(&assigned)?;
    let text = row.text.clone().unwrap_or_default();
    let Window { earliest, latest } = context.window;
    // Clamped so a miner's clock cannot pick the winning version.
    let fetched = row.fetched_at.unwrap_or(latest).max(earliest).min(latest);
    let validators = if !context.validators.is_empty() {
        context.validators.to_vec()
    } else {
        context.validator.filter(|v| !v.is_empty()).map(|v| vec![v.to_string()]).unwrap_or_default()
    };
    Ok(Record {
        domain: domain_of(&url)?,
        title: row.title.clone().unwrap_or_default(),
        published: row.published.clone().unwrap_or_default(),
        author: row.author.clone().unwrap_or_default(),
        lang: row.lang.clone().unwrap_or_default(),
        fetched_at: iso(fetched),
        content_sha1: canonical::sha1_hex(text.as_bytes()),
        captured_at: iso(context.captured_at),
        doc_id: uuid5_url(&url),
        assigned_url: assigned,
        final_url: row.final_url.clone(),
        canonical: row.canonical.clone(),
        status: row.status,
        page_type: row.page_type.clone(),
        description: row.description.clone(),
        json_ld_types: row.json_ld_types.clone().unwrap_or_default(),
        headings: row.headings.clone().unwrap_or_default(),
        text_sha256: row.text_sha256.clone(),
        task_id: context.task_id.to_string(),
        miner: context.miner.to_string(),
        validator: context.validator.map(str::to_string),
        validators,
        url,
        text,
    })
}

pub fn page_key(url: &str) -> Result<String> {
    Ok(format!("{PREFIX}/{}/{}", domain_of(url)?, url_sha1(url)?))
}

pub fn record_key(record: &Record) -> Result<String> {
    page_key(&record.url)
}

/// SHA-1 of Python's `json.dumps(stable, sort_keys=True, ensure_ascii=False)` over the content fields only.
pub fn record_version(record: &Record) -> String {
    let mut json = JsonSha1::default();
    json.raw("{\"author\": ");
    json.string(&record.author);
    json.raw(", \"canonical\": ");
    json.optional(record.canonical.as_deref());
    json.raw(", \"description\": ");
    json.optional(record.description.as_deref());
    json.raw(", \"headings\": ");
    json.list(&record.headings);
    json.raw(", \"json_ld_types\": ");
    json.list(&record.json_ld_types);
    json.raw(", \"lang\": ");
    json.string(&record.lang);
    json.raw(", \"page_type\": ");
    json.optional(record.page_type.as_deref());
    json.raw(", \"published\": ");
    json.string(&record.published);
    json.raw(", \"text\": ");
    json.string(&record.text);
    json.raw(", \"title\": ");
    json.string(&record.title);
    json.raw(", \"url\": ");
    json.string(&record.url);
    json.raw("}");
    hex(&json.hasher.finalize())
}

#[derive(Default)]
struct JsonSha1 {
    hasher: Sha1,
}

impl JsonSha1 {
    fn raw(&mut self, text: &str) {
        self.hasher.update(text.as_bytes());
    }

    fn string(&mut self, value: &str) {
        self.hasher.update(b"\"");
        let bytes = value.as_bytes();
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b >= 0x20 && b != b'"' && b != b'\\' {
                continue;
            }
            self.hasher.update(&bytes[start..i]);
            start = i + 1;
            let escaped: &[u8] = match b {
                b'"' => b"\\\"",
                b'\\' => b"\\\\",
                b'\n' => b"\\n",
                b'\r' => b"\\r",
                b'\t' => b"\\t",
                0x08 => b"\\b",
                0x0c => b"\\f",
                _ => {
                    const DIGITS: &[u8; 16] = b"0123456789abcdef";
                    self.hasher.update([b'\\', b'u', b'0', b'0', DIGITS[(b >> 4) as usize], DIGITS[(b & 15) as usize]]);
                    continue;
                }
            };
            self.hasher.update(escaped);
        }
        self.hasher.update(&bytes[start..]);
        self.hasher.update(b"\"");
    }

    fn optional(&mut self, value: Option<&str>) {
        match value {
            Some(value) => self.string(value),
            None => self.raw("null"),
        }
    }

    fn list(&mut self, values: &[Option<String>]) {
        self.raw("[");
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                self.raw(", ");
            }
            self.optional(value.as_deref());
        }
        self.raw("]");
    }
}

/// Python's `str(uuid.uuid5(uuid.NAMESPACE_URL, url))`.
pub fn uuid5_url(url: &str) -> String {
    let digest = Sha1::new().chain_update(NAMESPACE_URL).chain_update(url.as_bytes()).finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&digest[..16]);
    b[6] = (b[6] & 0x0f) | 0x50;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = hex(&b);
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

/// Python's `_iso`: whole seconds in UTC, as `isoformat(timespec="seconds")` writes them.
pub fn iso(us: i64) -> String {
    let seconds = us.div_euclid(SECOND);
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let rem = seconds.rem_euclid(86_400);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// The Unix seconds of a time `iso` wrote, or None for any other text.
pub fn parse_iso(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    if b.len() != 25 || &b[19..] != b"+00:00" || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let number =
        |range: std::ops::Range<usize>| -> Option<i64> { b[range].iter().try_fold(0i64, |n, &d| d.is_ascii_digit().then(|| n * 10 + i64::from(d - b'0'))) };
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    (iso(seconds * SECOND) == text).then_some(seconds)
}

fn in_range(us: i64) -> Result<i64> {
    if !(MIN_US..=MAX_US).contains(&us) {
        bail!("a time outside the range Python's datetime holds");
    }
    Ok(us)
}

/// Python's `datetime.fromtimestamp(seconds, timezone.utc)`, rounding to the microsecond half to even.
pub fn from_timestamp(seconds: f64) -> Result<i64> {
    if !seconds.is_finite() || seconds.abs() > 1e12 {
        bail!("timestamp {seconds} is out of range");
    }
    let whole = seconds.trunc();
    let mut micros = ((seconds - whole) * 1e6).round_ties_even();
    let mut whole = whole;
    if micros >= 1e6 {
        micros -= 1e6;
        whole += 1.0;
    } else if micros < 0.0 {
        micros += 1e6;
        whole -= 1.0;
    }
    in_range(whole as i64 * SECOND + micros as i64)
}

/// Python's `timedelta(seconds=seconds)` in microseconds.
pub fn timedelta(seconds: f64) -> Result<i64> {
    if !seconds.is_finite() || seconds.abs() > 1e12 {
        bail!("{seconds} seconds is out of range");
    }
    let whole = seconds.trunc();
    let micros = (seconds - whole) * 1e6;
    let leftover = micros - micros.trunc();
    let mut total = whole as i64 * SECOND + micros.trunc() as i64;
    if leftover != 0.0 {
        let mut rounded = leftover.round();
        if (rounded - leftover).abs() == 0.5 {
            let odd = (total & 1) as f64;
            rounded = 2.0 * ((leftover + odd) * 0.5).round() - odd;
        }
        total += rounded as i64;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_round_trips() {
        for us in [0, MIN_US, MAX_US, 1_791_310_017_842_173, -1] {
            let text = iso(us);
            assert_eq!(parse_iso(&text), Some(us.div_euclid(SECOND)), "{text}");
        }
        assert_eq!(iso(MIN_US), "0001-01-01T00:00:00+00:00");
        assert_eq!(iso(MAX_US), "9999-12-31T23:59:59+00:00");
        assert_eq!(parse_iso("2026-10-06T22:10:00Z"), None);
    }

    #[test]
    fn timestamps_round_like_python() {
        assert_eq!(from_timestamp(1791310017.8421726).unwrap(), 1_791_310_017_842_173);
        assert_eq!(from_timestamp(-1.5).unwrap(), -1_500_000);
        assert_eq!(timedelta(180.0).unwrap(), 180 * SECOND);
    }
}

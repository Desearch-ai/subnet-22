//! Uploads read in ranges with `publisher.reading`'s guards: only published columns, refused when malformed or too large.

mod common;

use std::io;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, TimestampMillisecondArray, TimestampMicrosecondArray, TimestampNanosecondArray};
use arrow_schema::{DataType, TimeUnit};
use bytes::Bytes;
use common::{page, parquet, Upload};
use publisher::reading::{read_rows, Counted, RangeRead, Traffic};
use publisher::records::SECOND;

struct Memory(Bytes);

impl RangeRead for Memory {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    fn read(&self, range: Range<u64>) -> io::Result<Bytes> {
        Ok(self.0.slice(range.start as usize..range.end as usize))
    }
}

fn read(file: Vec<u8>, assigned: usize) -> Option<Vec<publisher::records::Row>> {
    read_rows(&Memory(file.into()), assigned).unwrap()
}

const AT: i64 = 1_791_288_000 * SECOND;

fn uploads() -> Vec<Upload> {
    let mut sparse = page("https://ex.com/2", "", AT);
    sparse.row.title = None;
    sparse.row.json_ld_types = None;
    sparse.row.headings = Some(vec![Some("H".into()), None]);
    sparse.row.error = Some("timeout".into());
    sparse.row.status = None;
    vec![page("https://ex.com/1", "caf\u{e9} \u{1f600}", AT + 1), sparse, page("https://ex.com/3", "three", AT + 2)]
}

#[test]
fn every_row_reads_back_with_its_nulls() {
    let rows = read(parquet(&uploads(), None, None), 3).unwrap();
    let expected: Vec<_> = uploads().into_iter().map(|u| u.row).collect();
    assert_eq!(rows, expected);
}

#[test]
fn fetch_times_read_with_or_without_a_timezone() {
    let micros = vec![AT, AT + 1_500, AT + 2];
    let millis: Vec<i64> = micros.iter().map(|us| us / 1000).collect();
    let cases: [(DataType, ArrayRef, Vec<i64>); 3] = [
        (DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), Arc::new(TimestampMicrosecondArray::from(micros.clone()).with_timezone("UTC")), micros.clone()),
        (DataType::Timestamp(TimeUnit::Microsecond, None), Arc::new(TimestampMicrosecondArray::from(micros.clone())), micros.clone()),
        (
            DataType::Timestamp(TimeUnit::Millisecond, Some("+02:00".into())),
            Arc::new(TimestampMillisecondArray::from(millis.clone()).with_timezone("+02:00")),
            millis.iter().map(|ms| ms * 1000).collect(),
        ),
    ];
    for (kind, times, expected) in cases {
        let rows = read(parquet(&uploads(), Some((kind.clone(), times)), None), 3).unwrap();
        assert_eq!(rows.iter().map(|r| r.fetched_at).collect::<Vec<_>>(), expected.into_iter().map(Some).collect::<Vec<_>>(), "{kind}");
    }
    let numbers: ArrayRef = Arc::new(Int64Array::from(micros));
    let rows = read(parquet(&uploads(), Some((DataType::Int64, numbers)), None), 3).unwrap();
    assert!(rows.iter().all(|r| r.fetched_at.is_none()), "a column of numbers is no time, as Python's _utc treats it");
    let nanos: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![AT * 1000 + 1, AT * 1000, AT * 1000]).with_timezone("UTC"));
    let kind = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));
    assert_eq!(read(parquet(&uploads(), Some((kind, nanos)), None), 3), None, "Python cannot read nanoseconds as a datetime");
}

#[test]
fn malformed_or_oversized_uploads_are_refused() {
    let file = parquet(&uploads(), None, None);
    assert_eq!(read(file[..11].to_vec(), 3), None);
    let mut bad_head = file.clone();
    bad_head[0] = b'X';
    assert_eq!(read(bad_head, 3), None);
    let mut bad_tail = file.clone();
    let end = bad_tail.len();
    bad_tail[end - 1] = b'X';
    assert_eq!(read(bad_tail, 3), None);
    assert_eq!(read(parquet(&uploads(), None, Some("text_sha256")), 3), None, "a missing column");
    assert!(read(parquet(&uploads(), None, Some("html")), 3).is_some(), "the HTML is never read");
    assert_eq!(read(file.clone(), 1), None, "more than twice the rows assigned");
    assert!(read(file, 2).is_some());
    let huge = vec![page("https://ex.com/big", &"a".repeat(7_000_000), AT)];
    assert_eq!(read(parquet(&huge, None, None), 1), None, "more text than the rows assigned may hold");
    assert!(read(parquet(&huge, None, None), 2).is_some());
}

#[test]
fn the_html_is_never_fetched() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut noise = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u8
    };
    let mut pages = uploads();
    for upload in &mut pages {
        upload.html = (0..1_000_000).map(|_| noise()).collect();
    }
    let file = parquet(&pages, None, None);
    let traffic = Arc::new(Traffic::default());
    let source = Counted { inner: Memory(file.clone().into()), traffic: traffic.clone() };
    assert_eq!(read_rows(&source, 3).unwrap().unwrap().len(), 3);
    let fetched = traffic.bytes.load(Ordering::Relaxed);
    assert!(fetched < 100_000 && file.len() > 3_000_000, "fetched {fetched} of {} bytes", file.len());
}

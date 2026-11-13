//! Parquet objects read over HTTP by byte ranges: the footer first, then only the column chunks a reader needs.

use std::io::Cursor;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type};
use arrow_array::{Array, ArrayRef};
use bytes::Bytes;
use parquet::errors::ParquetError;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader, RowGroupMetaData};
use parquet::file::reader::{ChunkReader, Length};
use parquet::schema::types::SchemaDescriptor;
use reqwest::header::{CONTENT_RANGE, RANGE};
use reqwest::StatusCode;

const TRIES: u32 = 6;
const TAIL: u64 = 64 << 10;

/// Range reads, retrying timeouts, dropped connections and 5xx answers 6 times with 2**n s waits.
pub struct RangeClient(reqwest::blocking::Client);

impl RangeClient {
    pub fn new(user_agent: &str) -> Result<Self> {
        Ok(RangeClient(reqwest::blocking::Client::builder().user_agent(user_agent).timeout(crate::http::TIMEOUT).build()?))
    }

    /// The bytes of `range` (as the Range header writes it) and the object's whole length.
    pub fn get(&self, url: &str, range: &str) -> Result<(Bytes, u64)> {
        for attempt in 0..TRIES {
            let last = attempt == TRIES - 1;
            match self.0.get(url).header(RANGE, format!("bytes={range}")).send() {
                Ok(response) if response.status() == StatusCode::PARTIAL_CONTENT || response.status() == StatusCode::OK => {
                    let whole = response.status() == StatusCode::OK;
                    let total = response
                        .headers()
                        .get(CONTENT_RANGE)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.rsplit('/').next())
                        .and_then(|v| v.parse::<u64>().ok());
                    match response.bytes() {
                        Ok(body) => {
                            let total = if whole { body.len() as u64 } else { total.ok_or_else(|| anyhow!("{url}: no Content-Range length"))? };
                            return Ok((body, total));
                        }
                        Err(error) if last => return Err(error.into()),
                        Err(_) => {}
                    }
                }
                Ok(response) if matches!(response.status().as_u16(), 500 | 502 | 503 | 504) && !last => {}
                Ok(response) => bail!("{url}: HTTP {}", response.status()),
                Err(error) if last || !(error.is_timeout() || error.is_connect() || error.is_request() || error.is_body()) => return Err(error.into()),
                Err(_) => {}
            }
            sleep(Duration::from_secs(1 << attempt));
        }
        unreachable!("the last try returns")
    }
}

/// The fetched byte ranges of one Parquet object, as a ChunkReader.
pub struct Ranges {
    len: u64,
    chunks: Vec<(u64, Bytes)>,
}

impl Ranges {
    fn slice(&self, start: u64, length: Option<usize>) -> parquet::errors::Result<Bytes> {
        for (at, chunk) in &self.chunks {
            let end = at + chunk.len() as u64;
            if start >= *at && (start < end || (start == end && length == Some(0))) {
                let from = (start - at) as usize;
                let to = length.map_or(chunk.len(), |l| from + l);
                if to <= chunk.len() {
                    return Ok(chunk.slice(from..to));
                }
            }
        }
        Err(ParquetError::General(format!("bytes {start}+{length:?} were not fetched")))
    }
}

impl Length for Ranges {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for Ranges {
    type T = Cursor<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        Ok(Cursor::new(self.slice(start, None)?))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        self.slice(start, Some(length))
    }
}

/// The object's footer metadata and its length.
pub fn metadata(client: &RangeClient, url: &str) -> Result<(ParquetMetaData, u64)> {
    let (tail, len) = client.get(url, &format!("-{TAIL}"))?;
    if tail.len() < 8 || &tail[tail.len() - 4..] != b"PAR1" {
        bail!("{url} is not a Parquet file");
    }
    let size = u32::from_le_bytes(tail[tail.len() - 8..tail.len() - 4].try_into()?) as usize;
    let footer = if size + 8 <= tail.len() {
        tail.slice(tail.len() - 8 - size..tail.len() - 8)
    } else {
        let start = len.checked_sub(8 + size as u64).ok_or_else(|| anyhow!("{url}: a footer longer than the file"))?;
        client.get(url, &format!("{start}-{}", len - 9))?.0
    };
    Ok((ParquetMetaDataReader::decode_metadata(&footer)?, len))
}

/// A string column of any of arrow's string layouts.
pub enum Strings<'a> {
    Small(&'a arrow_array::StringArray),
    Large(&'a arrow_array::LargeStringArray),
    View(&'a arrow_array::StringViewArray),
}

impl<'a> Strings<'a> {
    pub fn of(array: &'a ArrayRef) -> Result<Self> {
        if let Some(a) = array.as_string_opt::<i32>() {
            return Ok(Strings::Small(a));
        }
        if let Some(a) = array.as_string_opt::<i64>() {
            return Ok(Strings::Large(a));
        }
        array.as_string_view_opt().map(Strings::View).ok_or_else(|| anyhow!("a column holds {} rather than strings", array.data_type()))
    }

    pub fn get(&self, i: usize) -> Option<&'a str> {
        match self {
            Strings::Small(a) => a.is_valid(i).then(|| a.value(i)),
            Strings::Large(a) => a.is_valid(i).then(|| a.value(i)),
            Strings::View(a) => a.is_valid(i).then(|| a.value(i)),
        }
    }
}

/// An integer column's value, None when null.
pub fn int_at(array: &ArrayRef, i: usize) -> Result<Option<i64>> {
    if array.is_null(i) {
        return Ok(None);
    }
    macro_rules! value {
        ($($t:ty),*) => {
            $(if let Some(a) = array.as_primitive_opt::<$t>() {
                return Ok(Some(a.value(i) as i64));
            })*
        };
    }
    value!(Int64Type, Int32Type, Int16Type, Int8Type, UInt64Type, UInt32Type, UInt16Type, UInt8Type);
    bail!("a column holds {} rather than integers", array.data_type())
}

/// The leaf columns under each of `paths` (dotted from the top, without the `list`, `element` and `key_value` levels
/// Parquet puts in lists and maps), in schema order.
pub fn leaves(schema: &SchemaDescriptor, paths: &[&str]) -> Vec<usize> {
    (0..schema.num_columns())
        .filter(|&i| {
            let column = schema.column(i);
            let parts: Vec<&str> = column.path().parts().iter().map(String::as_str).filter(|p| !matches!(*p, "list" | "element" | "key_value")).collect();
            paths.iter().any(|path| {
                let want: Vec<&str> = path.split('.').collect();
                parts.len() >= want.len() && parts.iter().zip(&want).all(|(a, b)| a == b)
            })
        })
        .collect()
}

/// The given leaves' column chunks of one row group, fetched one range each.
pub fn column_chunks(client: &RangeClient, url: &str, len: u64, row_group: &RowGroupMetaData, leaves: &[usize]) -> Result<Ranges> {
    let mut chunks = Vec::new();
    for &leaf in leaves {
        let (start, length) = row_group.column(leaf).byte_range();
        if length > 0 {
            chunks.push((start, client.get(url, &format!("{start}-{}", start + length - 1))?.0));
        }
    }
    Ok(Ranges { len, chunks })
}

//! The few object-storage steps the task API takes beyond plain reads and writes.

use anyhow::{anyhow, bail};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use bytes::Bytes;
use desearch::r2::{Bucket, Error, Head, Pin, JSON};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use serde_json::Value;

use crate::rules::Counted;

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// URL bytes a recount decodes at most: the file's word on its size is not taken.
const MAX_COUNTED_BYTES: usize = 64 << 20;

pub async fn put_json(bucket: &Bucket, key: &str, value: &Value, cache_control: Option<&str>) -> Result<(), Error> {
    bucket.put(key, Bytes::from(value.to_string()), JSON, cache_control).await
}

/// Only the magic bytes at both ends of the version `found` names are read; nothing of the miner's is parsed here.
pub async fn is_parquet(bucket: &Bucket, key: &str, found: &Head) -> Result<bool, Error> {
    let magic = PARQUET_MAGIC.len() as u64;
    if found.size < 2 * magic {
        return Ok(false);
    }
    let pin = Pin::of(found);
    let (head, tail) = tokio::try_join!(bucket.get_range_at(key, 0..magic, pin), bucket.get_range_at(key, found.size - magic..found.size, pin))?;
    Ok(head.as_ref() == PARQUET_MAGIC && tail.as_ref() == PARQUET_MAGIC)
}

/// A crawl upload's row counts from its `url` and `error` columns alone, as the validators' scoring counts its rows.
pub fn row_counts(job: &Value, bytes: Bytes) -> anyhow::Result<Counted> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let mask = ProjectionMask::columns(builder.parquet_schema(), ["url", "error"]);
    let reader = builder.with_projection(mask).with_batch_size(4096).build()?;
    let (mut rows, mut decoded): (Vec<(String, bool)>, usize) = (Vec::new(), 0);
    for batch in reader {
        let batch = batch?;
        let (Some(urls), Some(errors)) = (batch.column_by_name("url"), batch.column_by_name("error")) else { bail!("no url or error column") };
        for at in 0..batch.num_rows() {
            let url = string_at(urls, at)?;
            decoded += url.len();
            if decoded > MAX_COUNTED_BYTES {
                bail!("more than {MAX_COUNTED_BYTES} bytes of URLs");
            }
            rows.push((url, !errors.is_null(at)));
        }
    }
    Ok(Counted::of_rows(job, rows.iter().map(|(url, error)| (url.as_str(), *error))))
}

fn string_at(column: &ArrayRef, at: usize) -> anyhow::Result<String> {
    if column.is_null(at) {
        return Ok(String::new());
    }
    if let Some(strings) = column.as_string_opt::<i32>() {
        return Ok(strings.value(at).to_string());
    }
    column.as_string_opt::<i64>().map(|strings| strings.value(at).to_string()).ok_or_else(|| anyhow!("a url column that is not text"))
}

pub async fn delete_quietly(bucket: &Bucket, key: &str) {
    if let Err(error) = bucket.delete(key).await {
        eprintln!("could not delete {key} from {}: {error}", bucket.bucket);
    }
}

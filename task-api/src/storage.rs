//! The few object-storage steps the task API takes beyond plain reads and writes.

use bytes::Bytes;
use desearch::r2::{Bucket, Error, JSON};
use serde_json::Value;

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

pub async fn put_json(bucket: &Bucket, key: &str, value: &Value, cache_control: Option<&str>) -> Result<(), Error> {
    bucket.put(key, Bytes::from(value.to_string()), JSON, cache_control).await
}

/// Only the magic bytes at both ends are read; nothing of the miner's is parsed here.
pub async fn is_parquet(bucket: &Bucket, key: &str, size: u64) -> Result<bool, Error> {
    if size < 2 * PARQUET_MAGIC.len() as u64 {
        return Ok(false);
    }
    let head = bucket.get_range(key, 0..PARQUET_MAGIC.len() as u64, None).await?;
    let tail = bucket.get_tail(key, PARQUET_MAGIC.len() as u64).await?;
    Ok(head.as_ref() == PARQUET_MAGIC && tail.as_ref() == PARQUET_MAGIC)
}

pub async fn delete_quietly(bucket: &Bucket, key: &str) {
    if let Err(error) = bucket.delete(key).await {
        eprintln!("could not delete {key} from {}: {error}", bucket.bucket);
    }
}

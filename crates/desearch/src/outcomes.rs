//! What became of every URL, in the rows `app.outcomes` writes for the bot that queued it.

use std::sync::{Arc, LazyLock};

use anyhow::Result;
use arrow_array::builder::{StringBuilder, TimestampMicrosecondBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

pub const PUBLISHED: &str = "published";
pub const UNCHANGED: &str = "unchanged";
pub const FAILED: &str = "failed";
pub const DROPPED: &str = "dropped";

pub static SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("url", DataType::Utf8, true),
        Field::new("host", DataType::Utf8, true),
        Field::new("outcome", DataType::Utf8, true),
        Field::new("task_id", DataType::Utf8, true),
        Field::new("at", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), true),
    ]))
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub url: String,
    pub host: String,
    pub outcome: &'static str,
    pub task_id: String,
}

/// One outcome file, every row stamped `at` (microseconds since the epoch).
pub fn encode(rows: &[Outcome], at: i64) -> Result<Vec<u8>> {
    let text = |value: fn(&Outcome) -> &str| -> ArrayRef {
        let mut builder = StringBuilder::new();
        for row in rows {
            builder.append_value(value(row));
        }
        Arc::new(builder.finish())
    };
    let mut times = TimestampMicrosecondBuilder::with_capacity(rows.len()).with_timezone("UTC");
    for _ in rows {
        times.append_value(at);
    }
    let columns = vec![text(|r| &r.url), text(|r| &r.host), text(|r| r.outcome), text(|r| &r.task_id), Arc::new(times.finish()) as ArrayRef];
    let batch = RecordBatch::try_new(SCHEMA.clone(), columns)?;
    let properties = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(1)?)).build();
    let mut file = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut file, SCHEMA.clone(), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(file)
}

/// Where a numbered feed file's index lives, as `app.feeds.Feed.seq_key` names it.
pub fn seq_key(feed: &str, seq: u64) -> String {
    format!("{feed}/seq/{seq:012}.json")
}

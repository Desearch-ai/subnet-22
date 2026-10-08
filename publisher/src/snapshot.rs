//! A copy of the version index on request, as Parquet beside the pages it describes, streamed from a checkpoint into a multipart upload.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result};
use arrow_array::builder::{Int64Builder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use tokio::runtime::Handle;

use crate::index::{Current, VersionIndex};
use desearch::r2::{Bucket, PARQUET};

/// R2 takes parts of at least 5 MiB; bigger parts mean fewer requests for a large index.
const PART_BYTES: usize = 64 << 20;
const BATCH_ROWS: usize = 65_536;
const ROW_GROUP_ROWS: usize = 524_288;
const CHECKPOINT_CACHE: usize = 64 << 20;

pub static SNAPSHOT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let text = |name: &str| Field::new(name, DataType::Utf8, true);
    Arc::new(Schema::new(vec![
        text("key"),
        text("url"),
        text("version"),
        text("fetched_at"),
        text("task_id"),
        text("content_sha1"),
        Field::new("change_seq", DataType::Int64, true),
        Field::new("change_row", DataType::Int64, true),
    ]))
});

pub fn snapshot_key(day: &str) -> String {
    format!("index/snapshots/{day}.parquet")
}

/// Uploads the day's snapshot, replacing an earlier one of the same day; returns the pages written.
pub fn upload(index: &VersionIndex, index_dir: &Path, pages: &Bucket, day: &str, handle: &Handle) -> Result<u64> {
    let key = snapshot_key(day);
    let checkpoint = checkpoint_dir(index_dir);
    let _ = std::fs::remove_dir_all(&checkpoint);
    index.checkpoint(&checkpoint)?;
    let written = (|| {
        let copy = VersionIndex::open_read_only(&checkpoint, CHECKPOINT_CACHE)?;
        let upload = Multipart::new(pages.clone(), key.clone(), handle.clone());
        let (upload, rows) = write_parquet(&copy, upload)?;
        upload.finish()?;
        Ok(rows)
    })();
    let _ = std::fs::remove_dir_all(&checkpoint);
    written
}

fn checkpoint_dir(index_dir: &Path) -> PathBuf {
    let name = index_dir.file_name().map_or("index".into(), |n| n.to_string_lossy().into_owned());
    index_dir.with_file_name(format!("{name}.snapshot"))
}

/// Every page of the index as Parquet into `out`, a batch at a time.
pub fn write_parquet<W: Write + Send>(index: &VersionIndex, out: W) -> Result<(W, u64)> {
    let properties =
        WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).set_max_row_group_row_count(Some(ROW_GROUP_ROWS)).build();
    let mut writer = ArrowWriter::try_new(out, SNAPSHOT_SCHEMA.clone(), Some(properties))?;
    let mut pending: Vec<(String, Current)> = Vec::with_capacity(BATCH_ROWS);
    let mut rows = 0u64;
    index.scan(|key, current| {
        pending.push((key, current));
        if pending.len() == BATCH_ROWS {
            rows += pending.len() as u64;
            writer.write(&batch(&pending)?)?;
            pending.clear();
        }
        Ok(())
    })?;
    if !pending.is_empty() {
        rows += pending.len() as u64;
        writer.write(&batch(&pending)?)?;
    }
    Ok((writer.into_inner()?, rows))
}

fn batch(pages: &[(String, Current)]) -> Result<RecordBatch> {
    let text = |value: fn(&(String, Current)) -> &str| -> ArrayRef {
        let mut builder = StringBuilder::with_capacity(pages.len(), pages.len() * 48);
        pages.iter().for_each(|page| builder.append_value(value(page)));
        Arc::new(builder.finish())
    };
    let number = |value: fn(&Current) -> Option<i64>| -> ArrayRef {
        let mut builder = Int64Builder::with_capacity(pages.len());
        pages.iter().for_each(|(_, current)| builder.append_option(value(current)));
        Arc::new(builder.finish())
    };
    Ok(RecordBatch::try_new(
        SNAPSHOT_SCHEMA.clone(),
        vec![
            text(|(key, _)| key),
            text(|(_, c)| &c.url),
            text(|(_, c)| &c.version),
            text(|(_, c)| &c.fetched_at),
            text(|(_, c)| &c.task_id),
            text(|(_, c)| &c.content_sha1),
            number(|c| c.change_seq),
            number(|c| c.change_row),
        ],
    )?)
}

/// An object written through `Write`: each full part goes up as soon as it is written, so memory holds one part.
pub struct Multipart {
    bucket: Bucket,
    key: String,
    handle: Handle,
    upload_id: Option<String>,
    parts: Vec<(u32, String)>,
    buffer: Vec<u8>,
    part_bytes: usize,
    pub bytes: u64,
}

impl Multipart {
    pub fn new(bucket: Bucket, key: String, handle: Handle) -> Self {
        Multipart { bucket, key, handle, upload_id: None, parts: Vec::new(), buffer: Vec::new(), part_bytes: PART_BYTES, bytes: 0 }
    }

    pub fn with_part_bytes(mut self, part_bytes: usize) -> Self {
        self.part_bytes = part_bytes;
        self
    }

    fn send_part(&mut self, body: Vec<u8>) -> Result<()> {
        let upload_id = match &self.upload_id {
            Some(id) => id.clone(),
            None => {
                let id = self.handle.block_on(self.bucket.start_multipart(&self.key, PARQUET))?;
                self.upload_id = Some(id.clone());
                id
            }
        };
        let number = self.parts.len() as u32 + 1;
        let etag = self.handle.block_on(self.bucket.upload_part(&self.key, &upload_id, number, Bytes::from(body)))?;
        self.parts.push((number, etag));
        Ok(())
    }

    /// The last part and the completion; an object smaller than one part goes up in a single PUT.
    pub fn finish(mut self) -> Result<u64> {
        let finished = (|| -> Result<()> {
            if self.upload_id.is_none() {
                let body = Bytes::from(std::mem::take(&mut self.buffer));
                return Ok(self.handle.block_on(self.bucket.put(&self.key, body, PARQUET, None))?);
            }
            if !self.buffer.is_empty() {
                let last = std::mem::take(&mut self.buffer);
                self.send_part(last)?;
            }
            let upload_id = self.upload_id.clone().unwrap_or_default();
            Ok(self.handle.block_on(self.bucket.complete_multipart(&self.key, &upload_id, &self.parts))?)
        })();
        if finished.is_err() {
            self.abort();
        }
        finished.map(|()| self.bytes).with_context(|| format!("uploading {}", self.key))
    }

    fn abort(&mut self) {
        if let Some(upload_id) = self.upload_id.take() {
            let _ = self.handle.block_on(self.bucket.abort_multipart(&self.key, &upload_id));
        }
    }
}

impl Write for Multipart {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(data);
        self.bytes += data.len() as u64;
        while self.buffer.len() >= self.part_bytes {
            let rest = self.buffer.split_off(self.part_bytes);
            let part = std::mem::replace(&mut self.buffer, rest);
            if let Err(error) = self.send_part(part) {
                self.abort();
                return Err(io::Error::other(format!("{error:#}")));
            }
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

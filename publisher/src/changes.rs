//! Change files: every new, changed or removed page of a batch with its full record, in `CHANGE_SCHEMA`.

use std::sync::{Arc, LazyLock};

use anyhow::Result;
use arrow_array::builder::{Int32Builder, LargeStringBuilder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::records::{Record, SOURCE};

/// Small enough that one page's record is a single ranged read away.
pub const CHANGE_ROW_GROUP: usize = 1000;
/// pyarrow's default zstd level.
const ZSTD_LEVEL: i32 = 1;

pub static CHANGE_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    let text = |name: &str| Field::new(name, DataType::Utf8, true);
    let list = |name: &str| Field::new(name, DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true);
    Arc::new(Schema::new(vec![
        text("key"),
        text("kind"),
        text("previous_content_sha1"),
        text("published_at"),
        text("url"),
        text("domain"),
        text("doc_id"),
        text("title"),
        text("published"),
        text("author"),
        text("lang"),
        Field::new("text", DataType::LargeUtf8, true),
        text("fetched_at"),
        text("content_sha1"),
        text("source"),
        text("captured_at"),
        text("assigned_url"),
        text("final_url"),
        text("canonical"),
        Field::new("status", DataType::Int32, true),
        text("page_type"),
        text("description"),
        list("json_ld_types"),
        list("headings"),
        text("text_sha256"),
        text("task_id"),
        text("miner"),
        text("validator"),
        list("validators"),
    ]))
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    New,
    Changed,
    Removed,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::New => "new",
            Kind::Changed => "changed",
            Kind::Removed => "removed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Body {
    Page { record: Box<Record>, version: String },
    /// A page taken back: readers drop it from what they hold.
    Removed { url: String, domain: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    pub key: String,
    pub kind: Kind,
    pub previous_content_sha1: String,
    pub published_at: String,
    pub body: Body,
}

impl Change {
    pub fn record(&self) -> Option<&Record> {
        match &self.body {
            Body::Page { record, .. } => Some(record),
            Body::Removed { .. } => None,
        }
    }

    pub fn url(&self) -> &str {
        match &self.body {
            Body::Page { record, .. } => &record.url,
            Body::Removed { url, .. } => url,
        }
    }
}

/// The change file for these rows, compressed as pyarrow's `write_table(compression="zstd")` does.
pub fn encode(changes: &[Change]) -> Result<Vec<u8>> {
    let batch = RecordBatch::try_new(CHANGE_SCHEMA.clone(), columns(changes))?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL)?))
        .set_max_row_group_row_count(Some(CHANGE_ROW_GROUP))
        .set_coerce_types(true)
        .build();
    let mut file = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut file, CHANGE_SCHEMA.clone(), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(file)
}

fn columns(changes: &[Change]) -> Vec<ArrayRef> {
    let text = |value: fn(&Change) -> Option<&str>| -> ArrayRef {
        let mut builder = StringBuilder::with_capacity(changes.len(), changes.len() * 32);
        for change in changes {
            builder.append_option(value(change));
        }
        Arc::new(builder.finish())
    };
    let field = |value: fn(&Record) -> Option<&str>| -> ArrayRef {
        let mut builder = StringBuilder::with_capacity(changes.len(), changes.len() * 32);
        for change in changes {
            builder.append_option(change.record().and_then(value));
        }
        Arc::new(builder.finish())
    };
    let list = |value: fn(&Record) -> Vec<Option<&str>>| -> ArrayRef {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for change in changes {
            if let Some(record) = change.record() {
                for item in value(record) {
                    builder.values().append_option(item);
                }
            }
            builder.append(true);
        }
        Arc::new(builder.finish())
    };
    let mut body = LargeStringBuilder::with_capacity(changes.len(), changes.iter().map(|c| c.record().map_or(0, |r| r.text.len())).sum());
    let mut status = Int32Builder::with_capacity(changes.len());
    for change in changes {
        body.append_option(change.record().map(|r| r.text.as_str()));
        status.append_option(change.record().and_then(|r| r.status));
    }
    vec![
        text(|c| Some(&c.key)),
        text(|c| Some(c.kind.as_str())),
        text(|c| Some(&c.previous_content_sha1)),
        text(|c| Some(&c.published_at)),
        text(|c| Some(c.url())),
        text(|c| match &c.body {
            Body::Page { record, .. } => Some(&record.domain),
            Body::Removed { domain, .. } => Some(domain),
        }),
        field(|r| Some(&r.doc_id)),
        field(|r| Some(&r.title)),
        field(|r| Some(&r.published)),
        field(|r| Some(&r.author)),
        field(|r| Some(&r.lang)),
        Arc::new(body.finish()),
        field(|r| Some(&r.fetched_at)),
        field(|r| Some(&r.content_sha1)),
        field(|_| Some(SOURCE)),
        field(|r| Some(&r.captured_at)),
        text(|c| match &c.body {
            Body::Page { record, .. } => Some(&record.assigned_url),
            Body::Removed { url, .. } => Some(url),
        }),
        field(|r| r.final_url.as_deref()),
        field(|r| r.canonical.as_deref()),
        Arc::new(status.finish()),
        field(|r| r.page_type.as_deref()),
        field(|r| r.description.as_deref()),
        list(|r| r.json_ld_types.iter().map(Option::as_deref).collect()),
        list(|r| r.headings.iter().map(Option::as_deref).collect()),
        field(|r| r.text_sha256.as_deref()),
        field(|r| Some(&r.task_id)),
        field(|r| Some(&r.miner)),
        field(|r| r.validator.as_deref()),
        list(|r| r.validators.iter().map(|v| Some(v.as_str())).collect()),
    ]
}

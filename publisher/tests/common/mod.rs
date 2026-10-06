//! Uploads written the way miners write them, for the tests to read back.

use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Int32Builder, LargeStringBuilder, ListBuilder, StringBuilder, TimestampMicrosecondBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use publisher::records::Row;

/// One crawled page: the columns the publisher reads, plus the HTML it never reads.
#[derive(Clone, Default)]
pub struct Upload {
    pub row: Row,
    pub html: Vec<u8>,
}

pub fn page(url: &str, text: &str, fetched_at: i64) -> Upload {
    Upload {
        row: Row {
            url: Some(url.into()),
            final_url: Some(url.into()),
            status: Some(200),
            fetched_at: Some(fetched_at),
            page_type: Some("article".into()),
            title: Some(format!("About {url}")),
            description: Some(String::new()),
            lang: Some("en".into()),
            canonical: Some(url.into()),
            published: Some(String::new()),
            author: Some(String::new()),
            json_ld_types: Some(vec![Some("NewsArticle".into())]),
            headings: Some(vec![]),
            text: Some(text.into()),
            text_sha256: Some("0".repeat(64)),
            ..Row::default()
        },
        html: b"<html></html>".to_vec(),
    }
}

/// An upload in the miners' schema; `fetched_at` overrides that column, `drop` leaves one out.
pub fn parquet(rows: &[Upload], fetched_at: Option<(DataType, ArrayRef)>, drop: Option<&str>) -> Vec<u8> {
    let text = |value: &dyn Fn(&Row) -> Option<String>| -> ArrayRef {
        let mut b = StringBuilder::new();
        rows.iter().for_each(|u| b.append_option(value(&u.row)));
        Arc::new(b.finish())
    };
    let list = |value: &dyn Fn(&Row) -> Option<Vec<Option<String>>>| -> ArrayRef {
        let mut b = ListBuilder::new(StringBuilder::new());
        for u in rows {
            match value(&u.row) {
                Some(items) => {
                    items.iter().for_each(|i| b.values().append_option(i.as_deref()));
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        Arc::new(b.finish())
    };
    let int = |value: &dyn Fn(&Upload) -> Option<i32>| -> ArrayRef {
        let mut b = Int32Builder::new();
        rows.iter().for_each(|u| b.append_option(value(u)));
        Arc::new(b.finish())
    };
    let (time_type, times) = fetched_at.unwrap_or_else(|| {
        let mut b = TimestampMicrosecondBuilder::new().with_timezone("UTC");
        rows.iter().for_each(|u| b.append_option(u.row.fetched_at));
        (DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), Arc::new(b.finish()) as ArrayRef)
    });
    let mut html = BinaryBuilder::new();
    rows.iter().for_each(|u| html.append_value(&u.html));
    let mut body = LargeStringBuilder::new();
    rows.iter().for_each(|u| body.append_option(u.row.text.as_deref()));
    let html: ArrayRef = Arc::new(arrow_cast::cast(&html.finish(), &DataType::LargeBinary).unwrap());
    let utf8 = DataType::Utf8;
    let strings = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let columns: Vec<(&str, DataType, ArrayRef)> = vec![
        ("url", utf8.clone(), text(&|r| r.url.clone())),
        ("final_url", utf8.clone(), text(&|r| r.final_url.clone())),
        ("status", DataType::Int32, int(&|u| u.row.status)),
        ("error", utf8.clone(), text(&|r| r.error.clone())),
        ("fetched_at", time_type, times),
        ("elapsed_ms", DataType::Int32, int(&|_| Some(120))),
        ("content_type", utf8.clone(), text(&|_| Some("text/html".into()))),
        ("html_bytes", DataType::Int32, int(&|u| Some(u.html.len() as i32))),
        ("html", DataType::LargeBinary, html),
        ("html_sha256", utf8.clone(), text(&|_| Some("1".repeat(64)))),
        ("page_type", utf8.clone(), text(&|r| r.page_type.clone())),
        ("title", utf8.clone(), text(&|r| r.title.clone())),
        ("description", utf8.clone(), text(&|r| r.description.clone())),
        ("lang", utf8.clone(), text(&|r| r.lang.clone())),
        ("canonical", utf8.clone(), text(&|r| r.canonical.clone())),
        ("published", utf8.clone(), text(&|r| r.published.clone())),
        ("author", utf8.clone(), text(&|r| r.author.clone())),
        ("json_ld_types", strings.clone(), list(&|r| r.json_ld_types.clone())),
        ("headings", strings, list(&|r| r.headings.clone())),
        ("text", DataType::LargeUtf8, Arc::new(body.finish())),
        ("text_sha256", utf8, text(&|r| r.text_sha256.clone())),
    ];
    let kept: Vec<_> = columns.into_iter().filter(|(name, _, _)| Some(*name) != drop).collect();
    let schema = Arc::new(Schema::new(kept.iter().map(|(name, kind, _)| Field::new(*name, kind.clone(), true)).collect::<Vec<_>>()));
    let batch = RecordBatch::try_new(schema.clone(), kept.into_iter().map(|(_, _, array)| array).collect()).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(1).unwrap()))
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut file = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut file, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    file
}

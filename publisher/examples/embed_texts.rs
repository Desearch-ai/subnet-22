//! The planner's embed texts for page records, JSON lines in and out: `embed_texts < records.jsonl > texts.jsonl`.

use std::io::{BufRead, BufWriter, Write};

use desearch::canonical::{canonicalize, sha1_hex};
use publisher::planner::{recipe_of, texts};
use publisher::records::{page_key, Record};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Page {
    url: String,
    title: String,
    text: String,
    domain: String,
    lang: String,
    page_type: String,
    headings: Vec<String>,
    #[serde(default)]
    description: Option<String>,
    /// Absent for a v1 text.
    #[serde(default)]
    extraction: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let mut out = BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let page: Page = serde_json::from_str(&line?)?;
        let record = Record {
            url: canonicalize(&page.url)?,
            domain: page.domain,
            title: page.title,
            lang: page.lang,
            content_sha1: sha1_hex(page.text.as_bytes()),
            text: page.text,
            page_type: Some(page.page_type),
            headings: page.headings.into_iter().map(Some).collect(),
            description: page.description,
            extraction: page.extraction,
            ..Default::default()
        };
        let key = page_key(&record.url)?;
        for text in texts(&key, &record) {
            let row = json!({
                "text_id": text.text_id,
                "page_key": key,
                "url": record.url,
                "content_sha1": record.content_sha1,
                "kind": text.kind,
                "index": text.index,
                "text": text.text,
                "recipe": recipe_of(&record),
            });
            serde_json::to_writer(&mut out, &row)?;
            out.write_all(b"\n")?;
        }
    }
    Ok(out.flush()?)
}

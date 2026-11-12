//! `pagebuild::build` end to end with a stub extraction worker: record order, the checks around extraction, the run folder.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use arrow_array::{Array, LargeStringArray, StringArray, TimestampMicrosecondArray};
use chrono::DateTime;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};

use desearch_bot::pagebuild::{build, Options};

const STUB: &str = r#"
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r["op"] == "doc":
        out = {"page": None, "which": "pdf_overtime"}
    else:
        page = dict(page_type="other", text_mode="full", title=r["url"].rsplit("/", 1)[-1], description="", lang="en",
                    canonical="", published="", modified="", author="", site_name="", image="", section="", tags=[],
                    breadcrumbs=[], json_ld_types=[], headings=[], entity="", ld="", paywalled=None, robots=[],
                    text=r["html"])
        out = {"page": page, "blocked": "BLOCKED" in r.get("orig", r["html"])}
    print(json.dumps(out), flush=True)
"#;

fn write_records(path: &Path, records: &[Value]) {
    let mut enc = zstd::stream::write::Encoder::new(File::create(path).unwrap(), 3).unwrap();
    for r in records {
        writeln!(enc, "{r}").unwrap();
    }
    enc.finish().unwrap();
}

fn column<'a>(batch: &'a arrow_array::RecordBatch, name: &str) -> &'a dyn Array {
    batch.column_by_name(name).unwrap().as_ref()
}

#[test]
fn a_build_keeps_record_order_and_python_counts() {
    if Command::new("python3").arg("-c").arg("1").status().is_err() {
        eprintln!("no python3; skipped");
        return;
    }
    let dir = tempfile_dir();
    let records_dir = dir.join("records");
    std::fs::create_dir_all(&records_dir).unwrap();
    let english = "This is the page that we were reading, and it has all of the words you want to see in it. ".repeat(6);
    let body = "word ".repeat(100);
    let tenk = format!("ITEM 1. BUSINESS\n{body}\nITEM 1A. RISK FACTORS\n{body}\nITEM 2. PROPERTIES\n{body}");
    let at = "2026-11-10T23:46:42Z";
    let rec = |url: &str, html: &str| json!({"url": url, "status": 200, "fetched_at": at, "content_type": "text/html", "source": "site", "html": html});
    write_records(
        &records_dir.join("pages-000001.jsonl.zst"),
        &[
            json!({"url": "https://a.example/gone", "status": 404, "fetched_at": at, "html": "x"}),
            rec("https://a.example/one", &english),
            rec("https://a.example/one", &english),
            rec("https://a.example/held", &english),
            rec("https://a.example/short", "too short"),
            rec("https://a.example/blocked", "BLOCKED"),
            rec("https://a.example/wall", "Anubis could not load its JavaScript."),
        ],
    );
    write_records(
        &records_dir.join("pages-000002.jsonl.zst"),
        &[
            json!({"url": "https://www.sec.gov/Archives/edgar/data/1/2/k.htm", "status": 200, "fetched_at": at, "source": "edgar",
                   "html": format!("{english}\n{tenk}"), "meta": {"form": "10-K", "company": "ACME", "filed": "2026-11-01"}}),
            json!({"url": "https://a.example/doc.pdf", "status": 200, "fetched_at": at, "source": "list", "html": "", "body_b64": "JVBERi0="}),
            rec("https://a.example/long", &"long english words and the text of it ".repeat(6000)),
        ],
    );
    let stub = dir.join("stub.py");
    std::fs::write(&stub, STUB).unwrap();
    let mut skip = HashSet::new();
    skip.insert("https://a.example/held".to_string());
    skip.insert("https://www.sec.gov/Archives/edgar/data/1/2/k.htm?item=2".to_string());
    let out = dir.join("run");
    let options = Options {
        records: vec![records_dir],
        out: out.clone(),
        run: "t".into(),
        as_of: DateTime::parse_from_rfc3339("2026-11-11T00:00:00Z").unwrap(),
        set: None,
        skip,
        drop_mirrors: true,
        python: "python3".into(),
        worker: Some(stub),
        subnet22: dir.clone(),
        workers: 3,
        worker_memory_mb: 0,
        fetched_later_s: 1,
    };
    let counts = build(&options).unwrap();
    let got: Vec<(&str, u64)> = counts.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(
        got,
        vec![("not_a_page", 1), ("page_site", 2), ("repeat", 1), ("held", 1), ("ineligible", 1), ("blocked", 1), ("wall", 1), ("page_edgar_item", 2), ("document_pdf_overtime", 1)]
    );
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["rows"], 4);
    let file = File::open(out.join("crawl-000001.parquet")).unwrap();
    let batch = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap().next().unwrap().unwrap();
    let urls = column(&batch, "url").as_any().downcast_ref::<StringArray>().unwrap();
    let urls: Vec<&str> = (0..urls.len()).map(|i| urls.value(i)).collect();
    assert_eq!(
        urls,
        vec![
            "https://a.example/one",
            "https://www.sec.gov/Archives/edgar/data/1/2/k.htm?item=1",
            "https://www.sec.gov/Archives/edgar/data/1/2/k.htm?item=1A",
            "https://a.example/long"
        ]
    );
    let titles = column(&batch, "title").as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(titles.value(1), "ACME 10-K 2026-11-01, Item 1");
    let texts = column(&batch, "text").as_any().downcast_ref::<LargeStringArray>().unwrap();
    assert_eq!(texts.value(3).chars().count(), 200_000);
    let fetched = column(&batch, "fetched_at").as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
    assert_eq!(fetched.value(0), DateTime::parse_from_rfc3339("2026-11-10T23:46:43Z").unwrap().timestamp_micros());
    std::fs::remove_dir_all(&dir).unwrap();
}

fn tempfile_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("crawler-build-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

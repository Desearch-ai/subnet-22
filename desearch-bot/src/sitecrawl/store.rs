//! Fetched pages on disk as the Python crawler's `store.Writer` writes them: zstd JSON-lines files of records, one folder per run.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use indexmap::IndexMap;

use super::fetch::Response;

pub const RECORDS_PER_FILE: usize = 2_000;
const LEVEL: i32 = 6;

pub struct Writer {
    out: PathBuf,
    per_file: usize,
    files: usize,
    in_file: usize,
    pub records: usize,
    stream: Option<zstd::Encoder<'static, File>>,
}

fn json_string(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// `json.dumps(counts)`: Python's separators, keys in the order they came.
pub fn counts_json(counts: &IndexMap<String, i64>) -> String {
    let fields: Vec<String> = counts.iter().map(|(k, v)| format!("{}: {v}", json_string(k))).collect();
    format!("{{{}}}", fields.join(", "))
}

/// One record line as `json.dumps(record(r, html, source), ensure_ascii=False)` writes it.
pub fn record_line(r: &Response, html: &str, source: &str) -> String {
    format!(
        "{{\"url\": {}, \"status\": {}, \"fetched_at\": {}, \"content_type\": {}, \"source\": {}, \"html\": {}, \"meta\": {{}}}}",
        json_string(&r.url),
        r.status,
        json_string(&r.fetched_at.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        json_string(&r.content_type()),
        json_string(source),
        json_string(html),
    )
}

impl Writer {
    pub fn new(out: &Path) -> Result<Self> {
        Self::with_per_file(out, RECORDS_PER_FILE)
    }

    pub fn with_per_file(out: &Path, per_file: usize) -> Result<Self> {
        std::fs::create_dir_all(out)?;
        let taken = std::fs::read_dir(out)?.filter_map(|e| e.ok()).any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("pages-") && name.ends_with(".jsonl.zst")
        });
        if taken {
            bail!("{} already holds fetched pages; give each run its own folder", out.display());
        }
        Ok(Writer { out: out.to_path_buf(), per_file, files: 0, in_file: 0, records: 0, stream: None })
    }

    pub fn add(&mut self, line: &str) -> Result<()> {
        if self.stream.is_none() {
            self.files += 1;
            let file = File::create(self.out.join(format!("pages-{:06}.jsonl.zst", self.files)))?;
            self.stream = Some(zstd::Encoder::new(file, LEVEL)?);
        }
        let stream = self.stream.as_mut().expect("opened above");
        stream.write_all(line.as_bytes())?;
        stream.write_all(b"\n")?;
        self.in_file += 1;
        self.records += 1;
        if self.in_file >= self.per_file {
            self.close()?;
        }
        Ok(())
    }

    pub fn close(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            stream.finish()?.flush()?;
        }
        self.in_file = 0;
        Ok(())
    }
}

/// The records of record files, or of every record file in the folders given.
pub fn read(paths: &[PathBuf]) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for file in crate::pagebuild::record_files(paths)? {
        let text = String::from_utf8(zstd::decode_all(File::open(&file)?)?)?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn round_trip_and_one_run_a_folder() {
        let dir = std::env::temp_dir().join(format!("sitecrawl-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = Writer::with_per_file(&dir, 2).unwrap();
        for i in 0..5 {
            let r = Response {
                url: format!("https://ex.test/{i}"),
                status: 200,
                headers: HashMap::from([("content-type".into(), "Text/HTML; charset=utf-8".into())]),
                body: vec![],
                fetched_at: chrono::Utc::now(),
                error: None,
            };
            w.add(&record_line(&r, "é \"q\"\n\u{1}", "site")).unwrap();
        }
        w.close().unwrap();
        let files = std::fs::read_dir(&dir).unwrap().count();
        let records = read(std::slice::from_ref(&dir)).unwrap();
        assert_eq!(files, 3);
        assert_eq!(records.iter().map(|r| r["url"].as_str().unwrap().chars().last().unwrap()).collect::<String>(), "01234");
        assert_eq!(
            (records[0]["html"].as_str(), records[0]["content_type"].as_str(), records[0]["meta"].as_object().map(|m| m.len())),
            (Some("é \"q\"\n\u{1}"), Some("text/html"), Some(0))
        );
        assert!(Writer::new(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn lines_as_python_writes_them() {
        let r = Response {
            url: "https://a.test/x".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            fetched_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
            error: None,
        };
        assert_eq!(
            record_line(&r, "ü\u{7f}\u{2028}\t", "site"),
            "{\"url\": \"https://a.test/x\", \"status\": 200, \"fetched_at\": \"1970-01-01T00:00:00Z\", \"content_type\": \"\", \"source\": \"site\", \"html\": \"ü\u{7f}\u{2028}\\t\", \"meta\": {}}"
        );
        assert_eq!(counts_json(&IndexMap::from([("page".to_string(), 2), ("left_in_queue".to_string(), 0)])), "{\"page\": 2, \"left_in_queue\": 0}");
    }
}

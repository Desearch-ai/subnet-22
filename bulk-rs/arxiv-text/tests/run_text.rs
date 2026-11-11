use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use bulk_arxiv::Paper;
use bulk_arxiv_text::{full_text_page, oai_pages, run_text};
use bulk_core::{Batches, Counts};
use flate2::write::GzEncoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

mod common;

use common::common_pile_text;

const AS_OF_US: i64 = 1_793_491_200_000_000;
const BY: &str = "<license>http://creativecommons.org/licenses/by/4.0/</license>";

fn oai_record(arxiv_id: &str, licence: &str) -> String {
    format!(
        r#"<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{arxiv_id}</id><created>2024-01-02</created>
<authors><author><keyname>Tian</keyname><forenames>Zehua</forenames></author></authors>
<title>Probing cosmic strings</title><categories>gr-qc</categories>{licence}<abstract>Short.</abstract></arXiv></metadata></record>"#
    )
}

fn paper(arxiv_id: &str) -> Paper {
    Paper {
        arxiv_id: arxiv_id.to_string(),
        title: "Probing cosmic strings".to_string(),
        authors: vec!["Zehua Tian".to_string()],
        abstract_text: "Short.".to_string(),
        categories: "gr-qc".to_string(),
        created: "2024-01-02".to_string(),
        licence: "http://creativecommons.org/licenses/by/4.0/".to_string(),
        ..Paper::default()
    }
}

fn isotopes() -> String {
    format!("# Isotope effects\n\nV.G. Plekhanov\nCollege, Tallinn\n\nAbstract. {}", "Isotopes shift the exciton. ".repeat(80))
}

fn write_pile(path: &Path, lines: &[String]) {
    let mut out = GzEncoder::new(std::fs::File::create(path).unwrap(), flate2::Compression::default());
    for line in lines {
        out.write_all(line.as_bytes()).unwrap();
    }
    out.finish().unwrap();
}

fn pile_line(arxiv_id: &str, text: &str) -> String {
    format!("{}\n", serde_json::json!({"id": arxiv_id, "text": text}))
}

fn the_test_pile(dir: &Path) -> PathBuf {
    let text = common_pile_text();
    let pile = dir.join("00000_arxiv-papers.jsonl.gz");
    write_pile(
        &pile,
        &[
            pile_line("2401.00001", &text),
            pile_line("2208.05351", &isotopes()),
            pile_line("2402.00002", &text),
            pile_line("2403.00003", &text),
            pile_line("2404.00004", "# Probing cosmic strings\n\nToo short."),
            pile_line("2405.00005", &text),
        ],
    );
    pile
}

fn the_test_papers() -> HashMap<String, Paper> {
    ["2401.00001", "2208.05351", "2404.00004"].into_iter().map(|id| (id.to_string(), paper(id))).collect()
}

fn counts_of(counts: &Counts) -> Vec<(String, u64)> {
    counts.iter().map(|(k, n)| (k.to_string(), n)).collect()
}

fn rows_of(out: &Path) -> Vec<HashMap<String, String>> {
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    let mut rows = Vec::new();
    for name in manifest["files"].as_array().unwrap() {
        let file = std::fs::File::open(out.join(name.as_str().unwrap())).unwrap();
        for batch in ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap() {
            let batch = batch.unwrap();
            for i in 0..batch.num_rows() {
                let mut row = HashMap::new();
                for column in ["url", "content_type", "page_type", "published"] {
                    row.insert(column.to_string(), batch.column_by_name(column).unwrap().as_string::<i32>().value(i).to_string());
                }
                row.insert("text".to_string(), batch.column_by_name("text").unwrap().as_string::<i64>().value(i).to_string());
                rows.push(row);
            }
        }
    }
    rows
}

#[test]
fn a_full_text_page_frames_the_body_under_the_paper_heading_on_its_html_view() {
    let body = bulk_arxiv_text::body_lines(&common_pile_text());
    let (url, page) = full_text_page(&paper("2401.00001"), body);
    assert_eq!(url, "https://arxiv.org/html/2401.00001");
    let light = "They bend light. ".repeat(120);
    assert_eq!(
        page.text.lines().collect::<Vec<_>>(),
        vec![
            "Probing cosmic strings",
            "Zehua Tian",
            "arXiv:2401.00001 · gr-qc · Submitted 2024-01-02",
            "## Abstract",
            "Quantum metrology studies the ultimate precision limit of physical quantities.",
            "## 1 Introduction",
            &format!("Cosmic strings are line-like defects. {}", light.trim()),
            "Lemma 1",
            "The deficit angle is small. Fig. 3. shows $\\Delta_{12}$.",
            "* first point",
            "* second point",
        ]
    );
    assert_eq!((page.page_type.as_str(), page.published.as_str()), ("paper", "2024-01-02"));
    assert_eq!(full_text_page(&paper("2208.05351"), vec![]).0, "https://ar5iv.labs.arxiv.org/html/2208.05351");
}

#[test]
fn full_text_is_taken_for_openly_licensed_papers_onto_their_html_view() {
    let dir = tempfile::tempdir().unwrap();
    let oai = dir.path().join("oai");
    std::fs::create_dir(&oai).unwrap();
    let records: String = [
        oai_record("2401.00001", BY),
        oai_record("2208.05351", "<license>http://creativecommons.org/publicdomain/zero/1.0/</license>"),
        oai_record("2402.00002", "<license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>"),
        oai_record("2403.00003", ""),
        oai_record("2404.00004", BY),
    ]
    .concat();
    std::fs::write(
        oai.join("oai-000001.xml"),
        format!(r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{records}</ListRecords></OAI-PMH>"#),
    )
    .unwrap();
    let pile = the_test_pile(dir.path());
    let papers = bulk_arxiv::open_papers(&oai_pages(&oai)).unwrap();
    let mut ids: Vec<_> = papers.keys().cloned().collect();
    ids.sort();
    assert_eq!(ids, ["2208.05351", "2401.00001", "2404.00004"]);
    let mut batches = Batches::new(&dir.path().join("run"), "arxiv-text", "20261110", AS_OF_US).unwrap();
    let counts = run_text(&[pile], &papers, &mut batches, AS_OF_US, 0).unwrap();
    assert_eq!(counts_of(&counts), [("pages".to_string(), 2), ("not_open".to_string(), 3), ("short".to_string(), 1)]);
    batches.close(&counts).unwrap();
    let rows = rows_of(&dir.path().join("run"));
    let urls: Vec<&str> = rows.iter().map(|r| r["url"].as_str()).collect();
    assert_eq!(urls, ["https://arxiv.org/html/2401.00001", "https://ar5iv.labs.arxiv.org/html/2208.05351"]);
    assert_eq!((rows[0]["page_type"].as_str(), rows[0]["content_type"].as_str(), rows[0]["published"].as_str()), ("paper", "text/html", "2024-01-02"));
    assert!(rows[1]["text"].lines().nth(3).unwrap().starts_with("Abstract. Isotopes shift the exciton."));
}

#[test]
fn the_limit_stops_at_the_page_that_reaches_it() {
    let dir = tempfile::tempdir().unwrap();
    let pile = the_test_pile(dir.path());
    let papers = the_test_papers();
    for (limit, want) in [(1, vec![("pages", 1)]), (2, vec![("pages", 2)]), (3, vec![("pages", 2), ("not_open", 3), ("short", 1)])] {
        let mut batches = Batches::new(&dir.path().join(format!("run-{limit}")), "arxiv-text", "x", AS_OF_US).unwrap();
        let counts = run_text(std::slice::from_ref(&pile), &papers, &mut batches, AS_OF_US, limit).unwrap();
        assert_eq!(counts_of(&counts), want.iter().map(|&(k, n)| (k.to_string(), n)).collect::<Vec<_>>());
        assert_eq!(batches.rows(), counts.get("pages"));
    }
}

#[test]
fn line_ends_and_lone_surrogates_read_as_python_reads_them() {
    let dir = tempfile::tempdir().unwrap();
    let pile = dir.path().join("crlf.jsonl.gz");
    let text = isotopes().replace('\n', "\\n");
    write_pile(
        &pile,
        &[
            format!("{{\"id\": \"2208.05351\", \"text\": \"{text}\\ud800\"}}\r\n"),
            "{\"id\": \"2405.00005\", \"text\": \"x\"}\r".to_string(),
            "{\"id\": 7, \"text\": null}\n".to_string(),
            "{\"id\": \"2404.00004\", \"text\": \"# T\\n\\nToo short.\", \"id\": \"2401.00001\"}".to_string(),
        ],
    );
    let mut batches = Batches::new(&dir.path().join("run"), "arxiv-text", "x", AS_OF_US).unwrap();
    let counts = run_text(&[pile], &the_test_papers(), &mut batches, AS_OF_US, 0).unwrap();
    assert_eq!(counts_of(&counts), [("pages".to_string(), 1), ("not_open".to_string(), 2), ("short".to_string(), 1)]);
}

#[test]
fn a_bad_line_fails_the_run_only_when_reached() {
    let dir = tempfile::tempdir().unwrap();
    let pile = dir.path().join("bad.jsonl.gz");
    write_pile(&pile, &[pile_line("2208.05351", &isotopes()), "{\"id\": \"2401.00001\"}\n".to_string(), "not json\n".to_string()]);
    let papers = the_test_papers();
    let mut batches = Batches::new(&dir.path().join("run"), "arxiv-text", "x", AS_OF_US).unwrap();
    assert_eq!(run_text(std::slice::from_ref(&pile), &papers, &mut batches, AS_OF_US, 1).unwrap().get("pages"), 1);
    let mut batches = Batches::new(&dir.path().join("run-2"), "arxiv-text", "x", AS_OF_US).unwrap();
    assert!(run_text(&[pile], &papers, &mut batches, AS_OF_US, 0).is_err());
}

#[test]
fn an_empty_file_holds_no_lines() {
    let dir = tempfile::tempdir().unwrap();
    let pile = dir.path().join("empty.jsonl.gz");
    std::fs::write(&pile, b"").unwrap();
    let mut batches = Batches::new(&dir.path().join("run"), "arxiv-text", "x", AS_OF_US).unwrap();
    assert!(run_text(&[pile], &the_test_papers(), &mut batches, AS_OF_US, 0).unwrap().is_empty());
}

#[test]
fn oai_pages_are_sorted_like_python_paths_and_include_hidden_names() {
    let dir = tempfile::tempdir().unwrap();
    for name in [".h.xml", "b.xml", "a.XML", ".xml", "Z.xml", "c.xml.gz"] {
        std::fs::write(dir.path().join(name), "").unwrap();
    }
    std::fs::create_dir(dir.path().join("sub.xml")).unwrap();
    let names: Vec<String> = oai_pages(dir.path()).iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, [".h.xml", ".xml", "Z.xml", "b.xml", "sub.xml"]);
    assert!(oai_pages(&dir.path().join("missing")).is_empty());
}

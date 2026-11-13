//! bulk-pubmed against golden records of hand-made PubMed XML, golden runs of small files and a golden run of a real slice.

use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use bulk_core::rows::parse_as_of;
use bulk_core::{Batches, Counts, Page};
use bulk_pubmed::{abstract_lines, article_page, author_name, full_date, run, text_of, url_of, Article, Element, Records, RECORDS};
use serde_json::{json, Value};

const AS_OF: &str = "2026-11-01T00:00:00Z";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn page_json(page: &Page) -> Value {
    json!({
        "page_type": page.page_type, "text_mode": page.text_mode, "title": page.title, "description": page.description,
        "lang": page.lang, "canonical": page.canonical, "published": page.published, "modified": page.modified,
        "author": page.author, "site_name": page.site_name, "image": page.image, "section": page.section, "tags": page.tags,
        "breadcrumbs": page.breadcrumbs, "json_ld_types": page.json_ld_types, "headings": page.headings, "entity": page.entity,
        "ld": page.ld, "paywalled": page.paywalled, "robots": page.robots, "text": page.text,
    })
}

/// What the generator records for one record, built the same way from the Rust functions.
fn record_json(record: &Element) -> Vec<Value> {
    if record.tag == "DeleteCitation" {
        return record
            .iterfind("PMID")
            .into_iter()
            .map(|p| match p.text() {
                Some(t) => json!({"kind": "deleted", "url": url_of(desearch::text::strip(t))}),
                None => json!({"kind": "deleted", "error": "AttributeError"}),
            })
            .collect();
    }
    let mut entry = json!({"kind": "article", "pmid": record.findtext("MedlineCitation/PMID")});
    if let Some(body) = record.find("MedlineCitation").and_then(|c| c.find("Article")) {
        let journal = body.find("Journal");
        entry["title_text"] = json!(text_of(body.find("ArticleTitle")));
        entry["dates"] = json!([full_date(body.find("ArticleDate")), full_date(journal.and_then(|j| j.find("JournalIssue/PubDate")))]);
        entry["authors"] = json!(body.iterfind("AuthorList/Author").into_iter().map(author_name).collect::<Vec<_>>());
        entry["abstract"] = match body.find("Abstract") {
            Some(a) => json!(abstract_lines(a).iter().map(|l| json!([l.text, l.linked, l.verbatim, l.row])).collect::<Vec<_>>()),
            None => Value::Null,
        };
    }
    entry["result"] = match article_page(record) {
        Ok(Article::Page { url, page }) => json!({"url": url, "page": page_json(&page)}),
        Ok(Article::Skipped(why)) => json!({"skipped": why}),
        Err(_) => json!({"error": "AttributeError"}),
    };
    vec![entry]
}

/// Every record's expectations, and the parse error that ended the file, if any.
fn records_json(xml: &str) -> (Vec<Value>, Option<anyhow::Error>) {
    let mut found = Records::new(Cursor::new(xml.as_bytes().to_vec()), RECORDS);
    let mut out = Vec::new();
    loop {
        match found.next_record() {
            Ok(Some(record)) => out.extend(record_json(&record)),
            Ok(None) => return (out, None),
            Err(e) => return (out, Some(e)),
        }
    }
}

fn assert_records(name: &str, got: &[Value], want: &[Value]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{name} record {i}:\ngot      {g}\nexpected {w}");
    }
    assert_eq!(got.len(), want.len(), "{name}: record count");
}

#[test]
fn records_match_the_golden_records() {
    let fixture = load(&fixtures().join("records.json"));
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (got, error) = records_json(case["xml"].as_str().unwrap());
        assert!(error.is_none(), "{name}: {error:?}");
        assert_records(name, &got, case["records"].as_array().unwrap());
    }
}

#[test]
fn records_fail_where_the_golden_records_fail() {
    let fixture = load(&fixtures().join("records.json"));
    for case in fixture["errors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (got, error) = records_json(case["xml"].as_str().unwrap());
        match case["parse_error"].as_str() {
            Some(_) => assert!(error.is_some(), "{name}: expected a parse error"),
            None => {
                assert!(error.is_none(), "{name}: {error:?}");
                assert_records(name, &got, case["records"].as_array().unwrap());
            }
        }
    }
}

fn pairs(counts: &Counts) -> Vec<(String, u64)> {
    counts.iter().map(|(k, n)| (k.to_string(), n)).collect()
}

#[test]
fn runs_match_the_golden_run_folders() {
    let as_of_us = parse_as_of(AS_OF).unwrap();
    for case in load(&fixtures().join("runs.json")).as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let folder = fixtures().join("runs").join(name);
        let inputs: Vec<PathBuf> = case["inputs"].as_array().unwrap().iter().map(|i| folder.join("input").join(i.as_str().unwrap())).collect();
        let skip: HashSet<String> = case["skip"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("rust");
        let rows_per_file = case["rows_per_file"].as_u64().unwrap() as usize;
        let mut batches = Batches::with_rows_per_file(&out, "pubmed", "20261101", as_of_us, rows_per_file).unwrap();
        let counts = run(&inputs, &mut batches, as_of_us, &skip, case["limit"].as_u64().unwrap());
        if case["error"].is_string() {
            assert!(counts.is_err(), "{name}: expected an error");
            continue;
        }
        let counts = counts.unwrap();
        let want: Vec<(String, u64)> = serde_json::from_value(case["counts"].clone()).unwrap();
        assert_eq!(pairs(&counts), want, "{name}: counts");
        batches.close(&counts).unwrap();
        let found = bulk_core::runs::diff(&folder.join("expected"), &out, 20).unwrap();
        assert!(found.same(), "{name}: {found:#?}");
    }
}

#[test]
fn a_failing_record_stops_the_run_after_the_ones_before_it() {
    let as_of_us = parse_as_of(AS_OF).unwrap();
    let input = fixtures().join("runs/error_after_pages/input/pubmed26n0001.xml");
    let tmp = tempfile::tempdir().unwrap();
    let mut batches = Batches::new(&tmp.path().join("rust"), "pubmed", "20261101", as_of_us).unwrap();
    assert!(run(&[input], &mut batches, as_of_us, &HashSet::new(), 0).is_err());
    assert_eq!(batches.rows(), 1);
    assert_eq!(batches.removed().to_vec(), vec!["https://pubmed.ncbi.nlm.nih.gov/9/".to_string()]);
}

#[test]
fn a_missing_file_fails_only_when_the_run_reaches_it() {
    let as_of_us = parse_as_of(AS_OF).unwrap();
    let first = fixtures().join("runs/misc_gz/input/pubmed26n0001.xml.gz");
    let missing = fixtures().join("runs/no-such-file.xml.gz");
    let tmp = tempfile::tempdir().unwrap();
    let mut batches = Batches::new(&tmp.path().join("a"), "pubmed", "20261101", as_of_us).unwrap();
    assert!(run(&[first.clone(), missing.clone()], &mut batches, as_of_us, &HashSet::new(), 0).is_err());
    assert_eq!(batches.rows(), 15);
    let mut batches = Batches::new(&tmp.path().join("b"), "pubmed", "20261101", as_of_us).unwrap();
    let counts = run(&[first, missing], &mut batches, as_of_us, &HashSet::new(), 15).unwrap();
    assert_eq!(counts.get("pages"), 15);
}

#[test]
fn a_real_pubmed_slice_matches_the_golden_run() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/pubmed");
    let expected = root.join("expected");
    let manifest = load(&expected.join("manifest.json"));
    let as_of_us = parse_as_of(manifest["as_of"].as_str().unwrap()).unwrap();
    let skip = bulk_pubmed::read_skip(&root.join("input/skip-pmids.txt")).unwrap();
    let inputs = [root.join("input/pubmed26n1333.xml.gz"), root.join("input/pubmed26n1724.xml.gz")];
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("rust");
    let mut batches = Batches::new(&out, "pubmed", manifest["run"].as_str().unwrap(), as_of_us).unwrap();
    let counts = run(&inputs, &mut batches, as_of_us, &skip, 0).unwrap();
    batches.close(&counts).unwrap();
    let found = bulk_core::runs::diff(&expected, &out, 20).unwrap();
    assert!(found.same(), "{found:#?}");
}

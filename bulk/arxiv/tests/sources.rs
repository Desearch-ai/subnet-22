use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use bulk_arxiv::{open_papers, run};
use bulk_core::rows::{page_schema, parse_as_of, MANIFEST};
use bulk_core::{Batches, Counts};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const ARXIV: &str = r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:1303.7276</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>1303.7276</id><created>2013-03-29</created><updated>2026-10-26</updated>
<authors><author><keyname>Li</keyname><forenames>Chen</forenames></author></authors>
<title>The effect of limb
 kinematics</title><categories>physics.bio-ph cs.RO</categories><abstract>  Legged robots on sand.  </abstract></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:hep-th/9901001</identifier></header></record>
</ListRecords></OAI-PMH>"#;

fn as_of() -> i64 {
    parse_as_of("2026-11-01T00:00:00Z").unwrap()
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn counts(pairs: &[(&str, u64)]) -> Counts {
    let mut counts = Counts::default();
    for (key, n) in pairs {
        counts.add(key, *n);
    }
    counts
}

/// Each row's string columns by name, from every file the manifest lists.
fn rows_of(out: &Path) -> (Vec<HashMap<String, String>>, serde_json::Value) {
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join(MANIFEST)).unwrap()).unwrap();
    let mut rows = Vec::new();
    for name in manifest["files"].as_array().unwrap() {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(out.join(name.as_str().unwrap())).unwrap()).unwrap().build().unwrap();
        for batch in reader {
            let batch: RecordBatch = batch.unwrap();
            for i in 0..batch.num_rows() {
                let mut row = HashMap::new();
                for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
                    let any = column.as_any();
                    let value = match (any.downcast_ref::<StringArray>(), any.downcast_ref::<LargeStringArray>()) {
                        (Some(c), _) if !c.is_null(i) => c.value(i).to_string(),
                        (_, Some(c)) if !c.is_null(i) => c.value(i).to_string(),
                        _ => continue,
                    };
                    row.insert(field.name().clone(), value);
                }
                rows.push(row);
            }
        }
    }
    (rows, manifest)
}

#[test]
fn arxiv_records_become_abstract_pages_and_deleted_records_removals() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("oai-000001.xml");
    std::fs::write(&source, ARXIV).unwrap();
    let mut batches = Batches::new(&tmp.path().join("run"), "arxiv", "20261101", as_of()).unwrap();
    let counted = run(&[source], &mut batches, as_of(), 0).unwrap();
    assert_eq!((counted.get("pages"), counted.get("deleted")), (1, 1));
    assert_eq!(batches.removed(), ["https://arxiv.org/abs/hep-th/9901001"]);
    batches.close(&counted).unwrap();
    let (rows, _) = rows_of(&tmp.path().join("run"));
    let row = &rows[0];
    assert_eq!(row["url"], "https://arxiv.org/abs/1303.7276");
    assert_eq!(
        row["text"].lines().collect::<Vec<_>>(),
        [
            "The effect of limb kinematics",
            "Chen Li",
            "arXiv:1303.7276 · physics.bio-ph cs.RO · Submitted 2013-03-29, updated 2026-10-26",
            "## Abstract",
            "Legged robots on sand.",
        ]
    );
    assert_eq!((row["published"].as_str(), row["modified"].as_str()), ("2013-03-29", "2026-10-26"));
    assert_eq!((row["content_type"].as_str(), row["page_type"].as_str(), row["site_name"].as_str()), ("application/xml", "paper", "arXiv"));
}

#[test]
fn batch_files_hold_the_upload_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("oai.xml");
    std::fs::write(&source, ARXIV).unwrap();
    let mut batches = Batches::with_rows_per_file(&tmp.path().join("run"), "arxiv", "x", as_of(), 1).unwrap();
    run(&[source.clone(), source], &mut batches, as_of(), 0).unwrap();
    batches.close(&Counts::default()).unwrap();
    let (_, manifest) = rows_of(&tmp.path().join("run"));
    assert_eq!(manifest["files"], serde_json::json!(["arxiv-000001.parquet", "arxiv-000002.parquet"]));
    let file = File::open(tmp.path().join("run/arxiv-000001.parquet")).unwrap();
    let schema = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().schema().as_ref().clone();
    assert_eq!(schema.with_metadata(Default::default()), page_schema().as_ref().clone().with_metadata(Default::default()));
}

#[test]
fn counts_keep_the_order_they_were_first_counted_and_the_limit_stops_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let mut batches = Batches::new(&tmp.path().join("run"), "arxiv", "x", as_of()).unwrap();
    let counted = run(&[data("mixed/input/oai-000001.xml")], &mut batches, as_of(), 0).unwrap();
    assert_eq!(counted, counts(&[("pages", 2), ("deleted", 2), ("empty", 1)]));
    let mut batches = Batches::new(&tmp.path().join("limited"), "arxiv", "x", as_of()).unwrap();
    let counted = run(&[data("mixed/input/oai-000001.xml"), data("mixed/input/snapshot.json")], &mut batches, as_of(), 3).unwrap();
    assert_eq!(counted, counts(&[("pages", 3), ("deleted", 2), ("empty", 1)]), "the third page is the snapshot's first");
    assert_eq!(batches.rows(), 3);
    let mut batches = Batches::new(&tmp.path().join("one"), "arxiv", "x", as_of()).unwrap();
    let counted = run(&[data("mixed/input/oai-000001.xml")], &mut batches, as_of(), 1).unwrap();
    assert_eq!(counted, counts(&[("pages", 1)]));
    assert!(batches.removed().is_empty(), "records after the limit are not read");
}

#[test]
fn errors_after_the_limit_are_not_reached_and_errors_before_it_fail_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let bad = tmp.path().join("bad.xml");
    std::fs::write(&bad, "<OAI-PMH").unwrap();
    let good = data("mixed/input/oai-000001.xml");
    let mut batches = Batches::new(&tmp.path().join("a"), "arxiv", "x", as_of()).unwrap();
    assert_eq!(run(&[good.clone(), bad.clone()], &mut batches, as_of(), 1).unwrap().get("pages"), 1);
    let mut batches = Batches::new(&tmp.path().join("b"), "arxiv", "x", as_of()).unwrap();
    assert!(run(&[good, bad], &mut batches, as_of(), 0).is_err());
    let snapshot = tmp.path().join("snapshot.json");
    std::fs::write(&snapshot, "{\"id\": \"1\", \"title\": \"T\", \"abstract\": \"An abstract here.\"}\n\n").unwrap();
    let mut batches = Batches::new(&tmp.path().join("c"), "arxiv", "x", as_of()).unwrap();
    assert!(run(std::slice::from_ref(&snapshot), &mut batches, as_of(), 0).is_err(), "json.loads fails on a blank line");
    let mut batches = Batches::new(&tmp.path().join("d"), "arxiv", "x", as_of()).unwrap();
    assert_eq!(run(&[snapshot], &mut batches, as_of(), 1).unwrap().get("pages"), 1);
}

#[test]
fn snapshot_lines_become_pages_with_lone_surrogates_dropped_where_python_drops_them() {
    let tmp = tempfile::tempdir().unwrap();
    let mut batches = Batches::new(&tmp.path().join("run"), "arxiv", "x", as_of()).unwrap();
    let counted = run(&[data("mixed/input/snapshot.json")], &mut batches, as_of(), 0).unwrap();
    assert_eq!(counted, counts(&[("pages", 3), ("empty", 1)]));
    batches.close(&counted).unwrap();
    let (rows, _) = rows_of(&tmp.path().join("run"));
    let by_url: HashMap<&str, &HashMap<String, String>> = rows.iter().map(|r| (r["url"].as_str(), r)).collect();
    let first = by_url["https://arxiv.org/abs/0704.0001"];
    assert_eq!(
        first["text"].split('\n').collect::<Vec<_>>(),
        [
            "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies",
            "C. Balázs, E. L. Berger, P. M. Nadolsky Jr, Solo",
            "arXiv:0704.0001 · hep-ph · Submitted 2007-04-02, updated 2007-07-24",
            "## Abstract",
            "A fully differential calculation in perturbative quantum chromodynamics is presented.",
            "Comments: 37 pages, 15 figures; published version",
            "Journal reference: Phys.Rev.D76:013009,2007",
            "DOI: 10.1103/PhysRevD.76.013009",
        ]
    );
    let second = by_url["https://arxiv.org/abs/2401.00002"];
    assert_eq!(second["title"], "Same day \u{1d49c}");
    assert_eq!(second["text"].split('\n').next_back(), Some("Text with a lone  surrogate and  here"));
    assert_eq!(by_url["https://arxiv.org/abs/2401.00004"]["published"], "");
}

#[test]
fn a_python_run_over_the_same_pages_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("run");
    let mut batches = Batches::new(&out, "arxiv", "20261101", as_of()).unwrap();
    let counted = run(&[data("mixed/input/oai-000001.xml"), data("mixed/input/snapshot.json")], &mut batches, as_of(), 0).unwrap();
    batches.close(&counted).unwrap();
    let diff = bulk_core::runs::diff(&out, &data("mixed/python"), 20).unwrap();
    assert!(diff.same(), "{diff:#?}");
}

#[test]
fn only_openly_licensed_papers_are_kept_for_full_text() {
    let record = |id: &str, licence: &str| {
        format!(
            r#"<record><header><identifier>oai:arXiv.org:{id}</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{id}</id><created>2024-01-02</created>
<authors><author><keyname>Tian</keyname><forenames>Zehua</forenames></author></authors>
<title>Probing cosmic strings</title><categories>gr-qc</categories>{licence}<abstract>Short.</abstract></arXiv></metadata></record>"#
        )
    };
    let by = "<license>http://creativecommons.org/licenses/by/4.0/</license>";
    let tmp = tempfile::tempdir().unwrap();
    let page = format!(
        r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{}{}{}{}{}</ListRecords></OAI-PMH>"#,
        record("2401.00001", by),
        record("2208.05351", "<license>http://creativecommons.org/publicdomain/zero/1.0/</license>"),
        record("2402.00002", "<license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>"),
        record("2403.00003", ""),
        record("2404.00004", by),
    );
    std::fs::write(tmp.path().join("oai-000001.xml"), page).unwrap();
    let later = format!(
        r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{}{}</ListRecords></OAI-PMH>"#,
        record("2401.00001", "<license>http://creativecommons.org/licenses/by-nc/4.0/</license>").replace("Probing", "Closed"),
        record("2404.00004", by).replace("Probing", "Newer"),
    );
    std::fs::write(tmp.path().join("oai-000002.xml"), later).unwrap();
    let papers = open_papers(&[tmp.path().join("oai-000001.xml"), tmp.path().join("oai-000002.xml")]).unwrap();
    let mut ids: Vec<&str> = papers.keys().map(String::as_str).collect();
    ids.sort();
    assert_eq!(ids, ["2208.05351", "2401.00001", "2404.00004"]);
    assert_eq!(papers["2401.00001"].title, "Probing cosmic strings", "a later closed record does not replace an open one");
    assert_eq!(papers["2404.00004"].title, "Newer cosmic strings", "a later open record does");
    assert_eq!(papers["2401.00001"].authors, ["Zehua Tian"]);
}

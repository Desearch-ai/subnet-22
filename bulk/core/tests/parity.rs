//! bulk-core's text cleaning, paper pages, as-of parsing and batch files against golden cases and a golden run.

use std::fs::File;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::Array;
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use bulk_core::rows::{parse_as_of, sha256_hex, utc_format, PaperMeta};
use bulk_core::text::{clean, drop_lone_surrogates, finish, heading_level, text_headings};
use bulk_core::{authors_line, paper_page, upload_row, Batches, Counts, Line, Page};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap()).unwrap()
}

fn s(v: &Value) -> String {
    v.as_str().unwrap_or_else(|| panic!("not a string: {v}")).to_string()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(s).collect()
}

fn line(v: &Value) -> Line {
    Line { text: s(&v[0]), linked: v[1].as_bool().unwrap(), verbatim: v[2].as_bool().unwrap(), row: v[3].as_bool().unwrap() }
}

fn lines(v: &Value) -> Vec<Line> {
    v.as_array().unwrap().iter().map(line).collect()
}

fn page(v: &Value) -> Page {
    Page {
        page_type: s(&v["page_type"]),
        text_mode: s(&v["text_mode"]),
        title: s(&v["title"]),
        description: s(&v["description"]),
        lang: s(&v["lang"]),
        canonical: s(&v["canonical"]),
        published: s(&v["published"]),
        modified: s(&v["modified"]),
        author: s(&v["author"]),
        site_name: s(&v["site_name"]),
        image: s(&v["image"]),
        section: s(&v["section"]),
        tags: strings(&v["tags"]),
        breadcrumbs: strings(&v["breadcrumbs"]),
        json_ld_types: strings(&v["json_ld_types"]),
        headings: strings(&v["headings"]),
        entity: s(&v["entity"]),
        ld: s(&v["ld"]),
        paywalled: v["paywalled"].as_bool(),
        robots: strings(&v["robots"]),
        text: v["text"].as_str().unwrap_or_default().to_string(),
    }
}

fn counts(v: &Value) -> Counts {
    let mut counts = Counts::default();
    for pair in v.as_array().unwrap() {
        counts.add(pair[0].as_str().unwrap(), pair[1].as_u64().unwrap());
    }
    counts
}

#[test]
fn clean_matches_the_golden_cases() {
    let cases = fixture("clean.json");
    let mut failures = Vec::new();
    for (n, case) in cases.as_array().unwrap().iter().enumerate() {
        let input = lines(&case["lines"]);
        for (cards, key) in [(false, "plain"), (true, "cards")] {
            let want: Vec<Line> = case[key].as_array().unwrap().iter().map(|i| input[i.as_u64().unwrap() as usize].clone()).collect();
            let got = clean(input.clone(), cards);
            if got != want {
                failures.push(format!(
                    "case {n} {key}: got {:?}\nwant {:?}",
                    got.iter().map(|l| &l.text).collect::<Vec<_>>(),
                    want.iter().map(|l| &l.text).collect::<Vec<_>>()
                ));
            }
        }
    }
    assert!(cases.as_array().unwrap().len() >= 400);
    assert!(failures.is_empty(), "{} of {} differ:\n{}", failures.len(), cases.as_array().unwrap().len() * 2, failures[..failures.len().min(5)].join("\n"));
}

#[test]
fn single_lines_kept_as_the_golden_cases_keep_them() {
    let mut failures = Vec::new();
    for case in fixture("singles.json").as_array().unwrap() {
        let text = s(&case["text"]);
        for (cards, key) in [(false, "plain"), (true, "cards")] {
            let kept = !clean(vec![Line::new(text.clone())], cards).is_empty();
            if kept != case[key].as_bool().unwrap() {
                failures.push(format!("{key} {text:?}: kept {kept}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn heading_levels_match_the_golden_cases() {
    for case in fixture("heading_level.json").as_array().unwrap() {
        assert_eq!(heading_level(&Line::new(s(&case["text"]))) as u64, case["level"].as_u64().unwrap(), "{}", case["text"]);
    }
}

#[test]
fn finish_matches_the_golden_cases() {
    let cases = fixture("finish.json");
    for case in cases.as_array().unwrap() {
        let input: Vec<Line> = strings(&case["lines"]).into_iter().map(Line::new).collect();
        let cap = case["cap"].as_u64().unwrap() as usize;
        assert_eq!(finish(&s(&case["title"]), &input, cap), s(&case["text"]), "{case}");
    }
}

#[test]
fn finish_cuts_by_code_points_at_the_text_cap() {
    let wide = "é".repeat(199_990);
    let text = finish("Big", &[Line::new(wide.clone()), Line::new("tail line")], bulk_core::text::MAX_TEXT_CHARS);
    assert_eq!(text, format!("Big\n{wide}"));
    let unbroken = "é".repeat(200_010);
    assert_eq!(finish("", &[Line::new(unbroken.clone())], 200_000).chars().count(), 200_000);
}

#[test]
fn text_headings_match_the_golden_cases() {
    for case in fixture("headings.json").as_array().unwrap() {
        assert_eq!(text_headings(&s(&case["text"])), strings(&case["headings"]), "{}", case["text"]);
    }
}

#[test]
fn paper_page_matches_the_golden_cases() {
    for case in fixture("paper_page.json").as_array().unwrap() {
        let meta = &case["meta"];
        let meta = PaperMeta {
            lang: s(&meta["lang"]),
            url: s(&meta["url"]),
            published: s(&meta["published"]),
            modified: s(&meta["modified"]),
            author: s(&meta["author"]),
            site_name: s(&meta["site_name"]),
            tags: strings(&meta["tags"]),
        };
        let input = match case.get("repeat") {
            Some(repeat) => {
                let mut input = vec![Line::new(s(&repeat[0])); repeat[1].as_u64().unwrap() as usize];
                input.push(Line::new(s(&repeat[2])));
                input
            }
            None => lines(&case["lines"]),
        };
        let mut got = paper_page(&s(&case["title"]), input, meta);
        if case.get("text_sha256").is_some() {
            assert_eq!(got.text.chars().count() as u64, case["text_chars"].as_u64().unwrap());
            assert_eq!(sha256_hex(&got.text), s(&case["text_sha256"]));
            got.text.clear();
        }
        assert_eq!(got, page(&case["page"]), "{}", case["title"]);
    }
}

#[test]
fn authors_line_matches_the_golden_cases() {
    for case in fixture("authors.json").as_array().unwrap() {
        assert_eq!(authors_line(&strings(&case["names"]), case["most"].as_u64().unwrap() as usize), s(&case["line"]));
    }
}

#[test]
fn parse_as_of_matches_python() {
    let mut failures = Vec::new();
    for case in fixture("as_of.json").as_array().unwrap() {
        let text = s(&case["text"]);
        match (parse_as_of(&text), case["us"].as_i64()) {
            (Ok(us), Some(want)) => {
                if us != want || utc_format(us, "%Y%m%d") != s(&case["run"]) || utc_format(us, "%Y-%m-%dT%H:%M:%SZ") != s(&case["as_of"]) {
                    failures.push(format!("{text:?}: {us} != {want}"));
                }
            }
            (Err(_), None) => {}
            (got, _) => failures.push(format!("{text:?}: got {got:?}, Python {}", case.get("error").or(case.get("us")).unwrap())),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn lone_surrogates_dropped_as_python_leaves_them() {
    for case in fixture("surrogates.json").as_array().unwrap() {
        let text = s(&case["json"]);
        let parsed: Value = serde_json::from_str(&drop_lone_surrogates(&text)).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(parsed, case["value"], "{text}");
    }
}

fn arrow_schema(path: &Path) -> SchemaRef {
    ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap().schema().clone()
}

/// pyarrow's spelling of a type, as `str(field.type)` prints it.
fn pyarrow_type(data_type: &DataType) -> String {
    match data_type {
        DataType::Utf8 => "string".into(),
        DataType::LargeUtf8 => "large_string".into(),
        DataType::LargeBinary => "large_binary".into(),
        DataType::Int32 => "int32".into(),
        DataType::Boolean => "bool".into(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => format!("timestamp[us, tz={tz}]"),
        DataType::List(item) => {
            assert!(item.is_nullable(), "pyarrow's list items are nullable");
            format!("list<item: {}>", pyarrow_type(item.data_type()))
        }
        other => format!("{other:?}"),
    }
}

/// The golden run written again from batches.json.
fn rust_run(out: &Path) -> (Value, bulk_core::rows::Summary, Counts) {
    let spec = fixture("batches.json");
    let as_of_us = parse_as_of(&s(&spec["as_of"])).unwrap();
    assert_eq!(as_of_us, spec["as_of_us"].as_i64().unwrap());
    let rows_per_file = spec["rows_per_file"].as_u64().unwrap() as usize;
    let mut batches = Batches::with_rows_per_file(out, &s(&spec["source"]), &s(&spec["run"]), as_of_us, rows_per_file).unwrap();
    for row in spec["rows"].as_array().unwrap() {
        batches.add(upload_row(&s(&row["url"]), page(&row["page"]), as_of_us, &s(&row["content_type"]))).unwrap();
    }
    for url in strings(&spec["removed"]) {
        batches.remove(&url);
    }
    let counts = counts(&spec["counts"]);
    let summary = batches.close(&counts).unwrap();
    (spec, summary, counts)
}

#[test]
fn batches_write_the_golden_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = std::env::var_os("BULK_CORE_KEEP_RUN").map(PathBuf::from).unwrap_or_else(|| dir.path().join("run"));
    let (spec, summary, counts) = rust_run(&out);
    assert_eq!(std::fs::read_to_string(out.join("manifest.json")).unwrap(), s(&spec["manifest"]));
    assert_eq!(std::fs::read_to_string(out.join("removed.txt")).unwrap(), s(&spec["removed_txt"]));
    assert_eq!(bulk_core::run::summary_line(Path::new(&s(&spec["out"])), &summary, &counts), s(&spec["summary"]));
    let empty = bulk_core::rows::Summary { out: PathBuf::from("x"), files: 0, rows: 0, removed: 0 };
    assert_eq!(bulk_core::run::summary_line(Path::new("x"), &empty, &self::counts(&spec["override_counts"])), s(&spec["override_summary"]));

    let found = bulk_core::runs::diff(&fixtures().join("expected-run"), &out, 20).unwrap();
    assert!(found.same(), "{found:?}");
    assert_eq!((found.rows_a, found.rows_b), (7, 7));
    assert!(Batches::new(&out, "pubmed", "again", 0).is_err(), "a folder with a manifest holds a run");
}

#[test]
fn written_files_read_back_with_page_schema() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let (spec, _, _) = rust_run(&out);
    let expected_schema = arrow_schema(&fixtures().join("expected-run/pubmed-000001.parquet"));
    let mut rows = Vec::new();
    for name in ["pubmed-000001.parquet", "pubmed-000002.parquet", "pubmed-000003.parquet"] {
        let path = out.join(name);
        let schema = arrow_schema(&path);
        assert_eq!(schema, expected_schema, "{name} reads back with the golden file's arrow schema");
        let want = spec["schema"].as_array().unwrap();
        assert_eq!(schema.fields().len(), want.len());
        for (field, want) in schema.fields().iter().zip(want) {
            assert_eq!(field.name(), &s(&want["name"]));
            assert_eq!(pyarrow_type(field.data_type()), s(&want["type"]), "{}", field.name());
            assert_eq!(field.is_nullable(), want["nullable"].as_bool().unwrap(), "{}", field.name());
        }
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap()).unwrap();
        let groups: Vec<i64> = reader.metadata().row_groups().iter().map(|g| g.num_rows()).collect();
        assert!(groups.iter().all(|&n| n <= 3), "{groups:?}");
        for batch in reader.build().unwrap() {
            rows.push(batch.unwrap());
        }
    }
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 7);
    let first = &rows[0];
    let text = |name: &str| first.column_by_name(name).unwrap().as_string::<i32>().value(0).to_string();
    assert_eq!(text("url"), "https://example.org/0");
    assert_eq!(text("final_url"), "https://example.org/0");
    assert_eq!(text("extraction"), "v2");
    assert_eq!(text("html_sha256"), "");
    let body = first.column_by_name("text").unwrap().as_string::<i64>().value(0).to_string();
    assert_eq!(text("text_sha256"), sha256_hex(&body));
    assert!(first.column_by_name("error").unwrap().is_null(0));
    assert!(first.column_by_name("html").unwrap().is_null(0));
    assert!(first.column_by_name("paywalled").unwrap().is_null(0));
}

#[test]
fn diff_reports_changed_rows_and_manifests() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let spec = fixture("batches.json");
    let as_of_us = spec["as_of_us"].as_i64().unwrap();
    let mut batches = Batches::with_rows_per_file(&out, "pubmed", "test-run", as_of_us, 3).unwrap();
    for (i, row) in spec["rows"].as_array().unwrap().iter().enumerate() {
        let mut page = page(&row["page"]);
        if i == 4 {
            page.title.push_str(" changed");
        }
        batches.add(upload_row(&s(&row["url"]), page, as_of_us, &s(&row["content_type"]))).unwrap();
    }
    let mut counts = counts(&spec["counts"]);
    counts.add("extra", 1);
    batches.close(&counts).unwrap();
    let found = bulk_core::runs::diff(&fixtures().join("expected-run"), &out, 5).unwrap();
    assert!(!found.same());
    assert_eq!(found.rows_differing, 1);
    assert!(found.examples[0].starts_with("pubmed-000002.parquet row 1 title"), "{:?}", found.examples);
    assert!(found.manifest_differences.iter().any(|d| d.starts_with("counts:")), "{:?}", found.manifest_differences);
    assert!(found.manifest_differences.iter().any(|d| d.starts_with("removed")), "{:?}", found.manifest_differences);
}

#[test]
fn row_groups_and_files_split_at_their_sizes() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let mut batches = Batches::with_rows_per_file(&out, "arxiv", "r", 0, 1200).unwrap();
    for i in 0..1300 {
        batches.add(upload_row(&format!("https://x/{i}"), Page::default(), 0, "text/html")).unwrap();
    }
    assert_eq!(batches.files(), ["arxiv-000001.parquet", "arxiv-000002.parquet"]);
    batches.close(&Counts::default()).unwrap();
    let groups = |name: &str| -> Vec<i64> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(out.join(name)).unwrap()).unwrap();
        reader.metadata().row_groups().iter().map(|g| g.num_rows()).collect()
    };
    assert_eq!(groups("arxiv-000001.parquet"), [500, 500, 200]);
    assert_eq!(groups("arxiv-000002.parquet"), [100]);
}

#[test]
fn an_empty_run_has_a_manifest_and_no_files() {
    let dir = tempfile::tempdir().unwrap();
    let args = bulk_core::RunArgs { out: dir.path().join("empty"), as_of: "2026-11-01T23:30:00-01:00".into(), run: String::new(), limit: 0 };
    let (batches, as_of_us) = bulk_core::run::start(&args, "pmc").unwrap();
    assert_eq!(as_of_us, parse_as_of("2026-11-02T00:30:00Z").unwrap());
    let summary = batches.close(&Counts::default()).unwrap();
    let manifest = std::fs::read_to_string(args.out.join("manifest.json")).unwrap();
    let want = "{\n \"source\": \"pmc\",\n \"run\": \"20261102\",\n \"as_of\": \"2026-11-02T00:30:00Z\",\n \"files\": [],\n \"removed\": null,\n \"rows\": 0,\n \"counts\": {}\n}";
    assert_eq!(manifest, want);
    assert!(!args.out.join("removed.txt").exists());
    assert_eq!((summary.files, summary.rows, summary.removed), (0, 0, 0));
}

#[test]
fn diff_by_url_ignores_order_and_the_named_columns() {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, order: &[usize], fetched: i64, changed: Option<usize>| {
        let out = dir.path().join(name);
        let mut batches = Batches::with_rows_per_file(&out, "edgar", name, 0, 2).unwrap();
        for &i in order {
            let mut page = Page { title: format!("t{i}"), text: format!("text {i}"), ..Page::default() };
            if changed == Some(i) {
                page.text.push_str(" changed");
            }
            batches.add(upload_row(&format!("https://x/{i}"), page, fetched, "text/html")).unwrap();
        }
        batches.close(&Counts::default()).unwrap();
        out
    };
    let a = write("a", &[0, 1, 2, 3], 10, None);
    let b = write("b", &[3, 1, 0, 2], 99, None);
    let ignore = vec!["fetched_at".to_string()];
    assert!(bulk_core::runs::diff_by_url(&a, &b, &ignore, 5).unwrap().same());
    assert!(!bulk_core::runs::diff_by_url(&a, &b, &[], 5).unwrap().same());
    let c = write("c", &[2, 0, 1], 10, Some(1));
    let found = bulk_core::runs::diff_by_url(&a, &c, &ignore, 5).unwrap();
    assert_eq!(found.rows_differing, 2);
    assert!(found.examples.iter().any(|e| e.starts_with("https://x/1: text (+1 columns)")), "{:?}", found.examples);
    assert!(found.examples.iter().any(|e| e == "https://x/3: only in a"), "{:?}", found.examples);
    let counted = |name: &str, keys: &[(&str, u64)]| {
        let out = dir.path().join(name);
        let mut batches = Batches::with_rows_per_file(&out, "edgar", name, 0, 2).unwrap();
        batches.add(upload_row("https://x/0", Page::default(), 0, "text/html")).unwrap();
        let mut counts = Counts::default();
        for (key, n) in keys {
            counts.add(key, *n);
        }
        batches.close(&counts).unwrap();
        out
    };
    let x = counted("x", &[("page_edgar_item", 2), ("page_edgar", 1)]);
    let y = counted("y", &[("page_edgar", 1), ("page_edgar_item", 2)]);
    let z = counted("z", &[("page_edgar", 1), ("page_edgar_item", 3)]);
    assert!(bulk_core::runs::diff_by_url(&x, &y, &ignore, 5).unwrap().same());
    assert!(!bulk_core::runs::diff_by_url(&x, &z, &ignore, 5).unwrap().same());
}

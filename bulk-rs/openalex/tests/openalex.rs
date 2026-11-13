#[path = "../../pmc/tests/support/mod.rs"]
#[allow(dead_code)]
mod support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bulk_core::ranged::RangeClient;
use bulk_core::RunArgs;
use bulk_openalex::snapshot::{civil, file_works};
use bulk_openalex::work::{clean_abstract, flags};
use bulk_openalex::{outcome, Args, Held, Outcome, Work};
use serde_json::Value;
use support::{serve, Reply};

const AS_OF: i64 = 1_760_400_000_000_000;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/openalex/input")
}

fn json_works() -> Vec<Work> {
    std::fs::read_to_string(fixtures().join("works.jsonl"))
        .unwrap()
        .lines()
        .map(|line| Work::from_json(&serde_json::from_str(line).unwrap()).unwrap())
        .collect()
}

fn in_cut(works: Vec<Work>) -> Vec<Work> {
    works.into_iter().filter(|w| w.language == "en" && w.cited >= 20).collect()
}

fn snapshot_works() -> (Vec<Work>, usize) {
    let file = Arc::new(std::fs::read(fixtures().join("part_0000.parquet")).unwrap());
    let served = file.clone();
    let server = serve(move |request| if request.path == "/part_0000.parquet" { Reply::ranged(&served, request.range.as_deref()) } else { Reply::status(404) });
    let client = RangeClient::new("test").unwrap();
    let works = file_works(&client, &format!("{}/part_0000.parquet", server.base), 20).unwrap();
    (works, server.requests().len())
}

#[test]
fn the_snapshot_reads_the_works_the_api_gives() {
    let (works, requests) = snapshot_works();
    assert_eq!(works, in_cut(json_works()), "every field the checks and pages read");
    assert_eq!(works.len(), 21, "the French work and the one with 19 citations are outside the cut");
    assert!(requests > 1 && requests <= 1 + 3 * 31, "the footer, then the page's column chunks of each row group");
}

fn outcomes() -> Vec<(String, Result<String, &'static str>)> {
    let held = Held::read(&fixtures().join("held.tsv")).unwrap();
    in_cut(json_works())
        .iter()
        .map(|w| {
            let result = match outcome(w, &held, AS_OF) {
                Outcome::Page { row, .. } => Ok(row.page.text.clone()),
                Outcome::Out(why) => Err(why),
            };
            (w.id.rsplit('W').next().unwrap().to_string(), result)
        })
        .collect()
}

#[test]
fn works_stay_out_for_their_reasons() {
    let out: HashMap<String, &str> = outcomes().into_iter().filter_map(|(n, r)| r.err().map(|why| (n, why))).collect();
    let expected = [
        ("4", "no_doi"),
        ("5", "arxiv"),
        ("6", "pmc"),
        ("7", "pmc"),
        ("8", "pmc"),
        ("9", "retracted"),
        ("10", "paratext"),
        ("11", "short"),
        ("12", "garbled"),
        ("17", "empty"),
        ("18", "no_title"),
        ("21", "not_latin"),
        ("22", "page_chrome"),
        ("23", "abstract_dropped"),
    ];
    assert_eq!(out, expected.into_iter().map(|(n, w)| (n.to_string(), w)).collect::<HashMap<_, _>>());
}

#[test]
fn a_page_holds_the_title_authors_venue_doi_citations_and_abstract() {
    let pages: HashMap<String, String> = outcomes().into_iter().filter_map(|(n, r)| r.ok().map(|t| (n, t))).collect();
    assert_eq!(
        pages["1"],
        "Thin film conductivity study 1\n\
         Ada Byron (University of London; Royal Institution), Charles Babbage\n\
         Journal of Applied Films, 2019, volume 12, issue 3, pages 45–52. doi: 10.1234/fixture.1\n\
         Cited by 25 works in OpenAlex as of 2025-10-14.\n\
         We measured the thermal conductivity of thin films grown on sapphire at temperatures from 10 to 300 K. The films show a sharp drop below 40 K that matches a phonon boundary scattering model. These results bound the grain size and suggest a route to better thermoelectric devices."
    );
    assert_eq!(pages["13"].lines().last(), pages["1"].lines().last(), "the label and the copyright line are cut");
    assert_eq!(pages["14"].lines().nth(1), Some("Grace Hopper (Yale University)"), "an author without a name is left out, an institution named once");
    assert!(pages["14"].ends_with(pages["1"].lines().last().unwrap()));
    assert_eq!(pages["15"].lines().nth(2), Some("2019, page 7. doi: 10.1234/fixture.15"));
    assert_eq!(pages["19"].lines().nth(1), Some("2019, volume 12, issue 3, pages 45–52. doi: 10.1234/fixture.19"));
    assert!(pages["20"].lines().last().unwrap().starts_with("Summary statistics of We measured"), "a label is cut only before a capital");
}

#[test]
fn a_run_keeps_the_first_work_of_each_doi() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let args = Args {
        run: RunArgs { out: out.clone(), as_of: "2025-10-14T00:00:00Z".into(), run: "fixture".into(), limit: 0 },
        works: Some(fixtures().join("works.jsonl")),
        manifest: String::new(),
        first_file: 0,
        files: 0,
        held: Some(fixtures().join("held.tsv")),
        min_cited: 20,
        chunk_pages: 0,
        workers: 1,
        flags_out: None,
    };
    bulk_openalex::main(args).unwrap();
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    let counts = &manifest["counts"];
    assert_eq!((counts["pages"].as_u64(), counts["duplicate_doi"].as_u64(), counts["outside_cut"].as_u64()), (Some(6), Some(1), Some(2)));
}

#[test]
fn abstracts_lose_their_label_and_copyright_line() {
    assert_eq!(clean_abstract("AbstractMinangkabau nagari serve as the lowest level."), "Minangkabau nagari serve as the lowest level.");
    assert_eq!(clean_abstract("Abstract and Summary When a male smooth newt courts."), "When a male smooth newt courts.");
    let body = "x".repeat(400);
    assert_eq!(clean_abstract(&format!("{body} (© Wiley‐VCH Verlag GmbH, 2003)")), body);
    assert_eq!(clean_abstract(&format!("{body} Copyright © 2004 Society of Chemical Industry")), body);
    let late = format!("{body} {}", "y".repeat(250));
    assert_eq!(clean_abstract(&format!("{late} ©1999")), late, "a copyright mark under the cap still goes");
    assert_eq!(clean_abstract("ABSTRACTS of the 12th Meeting of the Society."), "ABSTRACTS of the 12th Meeting of the Society.");
    assert_eq!(clean_abstract("ABSTRACT: The meeting was held."), "The meeting was held.");
    let early = format!("© the authors. {body}");
    assert_eq!(clean_abstract(&early), early, "a mark near the start stays");
}

#[test]
fn a_full_chunk_takes_the_counts_of_the_files_after_it_until_the_next_page() {
    let file = Arc::new(std::fs::read(fixtures().join("part_0000.parquet")).unwrap());
    let served = file.clone();
    let base = Arc::new(std::sync::Mutex::new(String::new()));
    let listed = base.clone();
    let server = serve(move |request| match request.path.as_str() {
        "/manifest.json" => {
            let base = listed.lock().unwrap().clone();
            Reply::ok(format!(r#"{{"files": [{{"url": "{base}/a.parquet"}}, {{"url": "{base}/b.parquet"}}, {{"url": "{base}/c.parquet"}}]}}"#))
        }
        "/a.parquet" | "/b.parquet" | "/c.parquet" => Reply::ranged(&served, request.range.as_deref()),
        _ => Reply::status(404),
    });
    *base.lock().unwrap() = server.base.clone();
    let manifest = format!("{}/manifest.json", server.base);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let run = |first_file: usize| Args {
        run: RunArgs { out: out.clone(), as_of: "2025-10-14T00:00:00Z".into(), run: "fixture".into(), limit: 0 },
        works: None,
        manifest: manifest.clone(),
        first_file,
        files: 0,
        held: Some(fixtures().join("held.tsv")),
        min_cited: 20,
        chunk_pages: 6,
        workers: 2,
        flags_out: None,
    };
    bulk_openalex::main(run(0)).unwrap();
    let counts = |name: &str| -> Value {
        serde_json::from_str::<Value>(&std::fs::read_to_string(dir.path().join(name).join("manifest.json")).unwrap()).unwrap()["counts"].clone()
    };
    let first = counts("run-01");
    assert_eq!((first["pages"].as_u64(), first["duplicate_doi"].as_u64(), first["no_doi"].as_u64()), (Some(6), Some(15), Some(3)));
    assert!(!dir.path().join("run-02").exists(), "the later files hold no new page, so no second folder");
    std::fs::remove_dir_all(dir.path().join("run-01")).unwrap();
    bulk_openalex::main(run(7)).unwrap();
    assert_eq!(counts("run-01")["pages"].as_u64(), None, "a first file past the manifest's end reads nothing");
}

#[test]
fn a_doi_is_percent_encoded_where_a_url_path_cannot_hold_it() {
    let work = Work { doi: "https://doi.org/10.1002/(sici)1097-4636(199601)30:1<1::aid-jbm1>3.0.co;2-#".into(), ..Work::default() };
    assert_eq!(work.url(), "https://doi.org/10.1002/(sici)1097-4636(199601)30:1%3C1::aid-jbm1%3E3.0.co;2-%23");
    let odd = Work { doi: "https://doi.org/10.1/a b?c%d é".into(), ..Work::default() };
    assert_eq!(odd.url(), "https://doi.org/10.1/a%20b%3Fc%25d%20%C3%A9");
}

#[test]
fn a_broken_inverted_index_leaves_its_work_out_instead_of_stopping_the_run() {
    let held = Held::default();
    for index in [serde_json::json!("{not json"), serde_json::json!([1, 2]), serde_json::json!({"word": ["x"]})] {
        let work = Work::from_json(&serde_json::json!({"doi": "https://doi.org/10.1/x", "title": "T", "abstract_inverted_index": index})).unwrap();
        assert!(work.bad_index);
        assert!(matches!(outcome(&work, &held, AS_OF), Outcome::Out("bad_index")));
    }
}

#[test]
fn days_since_the_epoch_print_as_dates() {
    assert_eq!(civil(0), "1970-01-01");
    assert_eq!(civil(18_023), "2019-05-07");
    assert_eq!(civil(-1), "1969-12-31");
    assert_eq!(civil(20_510), "2026-02-26");
}

/// The pilot's 2,002 works against its quality script's flags, when OPENALEX_PILOT names the folder holding
/// `sample.jsonl` and `quality.jsonl`.
#[test]
fn flags_match_the_pilot_script() {
    let Some(folder) = std::env::var_os("OPENALEX_PILOT").map(PathBuf::from) else {
        eprintln!("skipped: set OPENALEX_PILOT to the pilot's folder");
        return;
    };
    let expected: HashMap<String, Value> = std::fs::read_to_string(folder.join("quality.jsonl"))
        .unwrap()
        .lines()
        .map(|line| {
            let v: Value = serde_json::from_str(line).unwrap();
            (v["id"].as_str().unwrap().to_string(), v)
        })
        .collect();
    let keys = [
        "doi",
        "title",
        "year",
        "type",
        "cited",
        "class",
        "markup",
        "chars",
        "gaps",
        "dups",
        "arxiv",
        "pmcid",
        "pmid",
        "landing",
        "source",
        "oa_license",
        "abstract",
    ];
    let mut rows = 0;
    for line in std::fs::read_to_string(folder.join("sample.jsonl")).unwrap().lines() {
        let got = flags(&Work::from_json(&serde_json::from_str(line).unwrap()).unwrap());
        let want = &expected[got["id"].as_str().unwrap()];
        for key in keys {
            assert_eq!(got[key], want[key], "{} {key}", got["id"]);
        }
        rows += 1;
    }
    assert_eq!(rows, 2002);
}

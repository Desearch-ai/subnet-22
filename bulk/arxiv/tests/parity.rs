use std::path::{Path, PathBuf};

use bulk_core::rows::{parse_as_of, MANIFEST};
use bulk_core::Batches;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/arxiv")
}

#[test]
fn the_run_over_real_pages_matches_the_golden_run() {
    let (input, expected) = (fixtures().join("input"), fixtures().join("expected"));
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(expected.join(MANIFEST)).unwrap()).unwrap();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&input).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_file()).collect();
    files.sort();
    let as_of = parse_as_of(manifest["as_of"].as_str().unwrap()).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("run");
    let mut batches = Batches::new(&out, manifest["source"].as_str().unwrap(), manifest["run"].as_str().unwrap(), as_of).unwrap();
    let counts = bulk_arxiv::run(&files, &mut batches, as_of, 0).unwrap();
    batches.close(&counts).unwrap();
    let diff = bulk_core::runs::diff(&out, &expected, 20).unwrap();
    assert!(diff.same(), "{diff:#?}");
}

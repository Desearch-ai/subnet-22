use std::path::{Path, PathBuf};

use bulk_core::RunArgs;

fn gz_files(folder: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(folder).unwrap().map(|e| e.unwrap().path()).filter(|p| p.to_string_lossy().ends_with(".jsonl.gz")).collect();
    files.sort();
    files
}

#[test]
fn a_run_over_the_parity_sample_matches_the_python_run() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/arxiv-text");
    let (input, python) = (root.join("input"), root.join("python"));
    if !python.join("manifest.json").exists() {
        eprintln!("skipped: no parity fixtures at {}", root.display());
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(python.join("manifest.json")).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let run =
        RunArgs { out: out.clone(), as_of: manifest["as_of"].as_str().unwrap().to_string(), run: manifest["run"].as_str().unwrap().to_string(), limit: 0 };
    bulk_arxiv_text::main(bulk_arxiv_text::Args { run, files: gz_files(&input), oai: input.join("oai") }).unwrap();
    let diff = bulk_core::runs::diff(&out, &python, 20).unwrap();
    assert!(diff.rows_a == diff.rows_b && diff.rows_differing == 0 && diff.manifest_differences.is_empty(), "{diff:#?}");
}

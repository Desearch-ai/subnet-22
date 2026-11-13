mod support;

use std::path::PathBuf;

use bulk_core::RunArgs;
use bulk_pmc::render::RenderArgs;
use bulk_pmc::Args;

#[test]
fn pmc_fixture_folder_run_matches_the_python_run() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/pmc");
    let (input, expected) = (fixtures.join("input"), fixtures.join("python"));
    if !input.is_dir() || !expected.join(bulk_core::rows::MANIFEST).is_file() {
        eprintln!("skipped: no fixtures in {}", fixtures.display());
        return;
    }
    let Some((python, root)) = support::python() else { return };
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(expected.join(bulk_core::rows::MANIFEST)).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rust");
    let args = Args {
        run: RunArgs { out: out.clone(), as_of: manifest["as_of"].as_str().unwrap().to_string(), run: manifest["run"].as_str().unwrap().to_string(), limit: 0 },
        inventory: Vec::new(),
        folder: Some(input),
        names: None,
        workers: 8,
        pmids_out: Some(dir.path().join("pmids.txt")),
        render: RenderArgs { python, python_path: Some(root), render_workers: None },
    };
    bulk_pmc::main(args).unwrap();
    let diff = bulk_core::runs::diff(&out, &expected, 20).unwrap();
    let pmids = std::fs::read_to_string(dir.path().join("pmids.txt")).unwrap();
    assert_eq!(pmids, std::fs::read_to_string(fixtures.join("pmids.txt")).unwrap());
    assert!(diff.manifest_differences.is_empty() && diff.rows_differing == 0 && diff.rows_a == diff.rows_b, "{diff:#?}");
}

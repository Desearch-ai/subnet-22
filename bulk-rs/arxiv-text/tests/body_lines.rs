use bulk_arxiv_text::body_lines;
use serde::Deserialize;

mod common;

use common::common_pile_text;

#[test]
fn the_body_starts_at_the_abstract_and_stops_at_the_references() {
    let light = "They bend light. ".repeat(120);
    assert_eq!(
        body_lines(&common_pile_text()),
        vec![
            "## Abstract".to_string(),
            "Quantum metrology studies the ultimate precision limit of physical quantities.".to_string(),
            "## 1 Introduction".to_string(),
            format!("Cosmic strings are line-like defects. {}", light.trim()),
            "Lemma 1".to_string(),
            "The deficit angle is small. Fig. 3. shows $\\Delta_{12}$.".to_string(),
            "* first point".to_string(),
            "* second point".to_string(),
        ]
    );
}

#[test]
fn without_an_abstract_heading_the_body_starts_at_the_paragraph_that_opens_with_abstract() {
    let text = format!("# Isotope effects\n\nV.G. Plekhanov\nCollege, Tallinn\n\nAbstract. {}", "Isotopes shift the exciton. ".repeat(80));
    let lines = body_lines(&text);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with("Abstract. Isotopes shift the exciton."));
}

#[test]
fn without_any_abstract_the_body_starts_at_the_heading_after_the_title() {
    assert_eq!(body_lines("# Probing cosmic strings\n\nToo short."), vec!["Too short."]);
    assert_eq!(body_lines("# T\n\nAuthors x@y.org\n\n## Intro\n\nBody."), vec!["## Intro", "Body."]);
    assert!(body_lines("").is_empty());
}

#[derive(Deserialize)]
struct Case {
    markdown: String,
    lines: Vec<String>,
}

#[test]
fn body_lines_match_the_python_on_generated_markdown() {
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/body_lines.json")).unwrap();
    assert!(cases.len() > 400);
    let mut failed = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let got = body_lines(&case.markdown);
        if got != case.lines {
            failed.push(format!("case {i}: {:?}\n  python {:?}\n  rust   {:?}", case.markdown, case.lines, got));
        }
    }
    assert!(failed.is_empty(), "{} of {} differ:\n{}", failed.len(), cases.len(), failed.join("\n"));
}

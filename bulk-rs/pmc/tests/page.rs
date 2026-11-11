mod support;

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use bulk_core::{Line, RunArgs};
use bulk_pmc::page::{assemble, Front};
use bulk_pmc::render::{Parts, RenderArgs};
use bulk_pmc::{article_page, parse_front, Args, Outcome, Parsed, Rendered, Renderer};
use serde_json::json;
use support::{python, JATS};

fn front_of(xml: &str) -> Front {
    match parse_front(xml.as_bytes()) {
        Some(Parsed::Front(front)) => front,
        other => panic!("{other:?}"),
    }
}

#[test]
fn pmc_front_matter_is_read_as_article_page_reads_it() {
    let front = front_of(JATS);
    assert_eq!(front.title, "Risk factors");
    assert_eq!(front.authors, ["Yuji Masataka"]);
    assert_eq!((front.doi.as_str(), front.published.as_str(), front.lang.as_str()), ("10.1002/npr2.12310", "2022-12-19", "en"));
    assert_eq!(front.tags, ["cannabis"]);
}

#[test]
fn pmc_front_matter_edge_cases_follow_lxml() {
    let xml = r#"<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.3 20210610//EN" "JATS-archivearticle1-3.dtd">
<article xml:lang="DE-de" xmlns:xlink="http://www.w3.org/1999/xlink"><front><journal-meta/></front><front><article-meta>
<title-group><article-title>  A <italic>study</italic>
 of &amp; things<!-- note --> </article-title></title-group>
<contrib-group><contrib><name><given-names>Ann</given-names></name></contrib><contrib contrib-type="editor"><name><surname>Ed</surname></name></contrib></contrib-group>
<contrib-group><contrib contrib-type="author"><collab>The <bold>Group</bold></collab></contrib><contrib contrib-type="author"><name><surname>Solo</surname></name></contrib></contrib-group>
<article-id pub-id-type="pmid">1</article-id>
<pub-date pub-type="epub"><year>2021</year><month>3</month></pub-date>
<pub-date date-type="pub"><day>1</day><month>2</month><year>２０２０</year></pub-date>
<pub-date pub-type="ppub"><day>9</day><month>9</month><year>1999</year></pub-date>
<pub-date pub-type="pub"><day>7</day><month>7</month><year>2007</year></pub-date>
<kwd-group><kwd>a</kwd><kwd> </kwd></kwd-group><kwd-group><kwd>b</kwd></kwd-group>
</article-meta></front></article>"#;
    let front = front_of(xml);
    assert_eq!(front.title, "A study of & things");
    assert_eq!(front.authors, ["Ann", "The Group", "Solo"]);
    assert_eq!(front.doi, "");
    assert_eq!(front.published, "2007-07-07", "the last pub-date of a type wins, epub without a day is skipped");
    assert_eq!(front.lang, "de");
    assert_eq!(front.tags, ["a", "", "b"]);
    assert_eq!(parse_front(b"<article><front/></article>"), Some(Parsed::NoFront));
    assert_eq!(
        parse_front(b"<article><front><article-meta><title-group><article-title> </article-title></title-group></article-meta></front></article>"),
        Some(Parsed::NoTitle)
    );
    assert_eq!(parse_front(b"<article"), None);
    let latin = b"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><article><front><article-meta><title-group><article-title>Caf\xe9</article-title></title-group>\
<pub-date pub-type=\"collection\"><day>1</day><month>2</month><year>\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x90</year></pub-date></article-meta></front></article>";
    let Some(Parsed::Front(front)) = parse_front(latin) else { panic!("latin-1 is read") };
    assert_eq!(front.title, "Caf\u{e9}");
    assert_eq!(front.published, "", "the full-width digits came in as latin-1 bytes");
    let wide = "<article><front><article-meta><title-group><article-title>T</article-title></title-group><pub-date pub-type=\"collection\"><day>1</day><month>2</month><year>\u{ff12}\u{ff10}\u{ff12}\u{ff10}</year></pub-date></article-meta></front></article>";
    assert_eq!(front_of(wide).published, "2020-02-01");
}

fn lines(texts: &[&str]) -> Vec<Line> {
    texts.iter().map(|t| Line::new(*t)).collect()
}

#[test]
fn pmc_rendered_parts_go_after_the_authors_citation_and_doi_lines() {
    let meta = json!({"pmcid": "PMC7", "citation": "  J  Rep. 2022; 1 "});
    let parts = Parts { abstracts: vec![lines(&["### Aim", " ", "Aim text."]), lines(&["Short."])], body: Some(lines(&["## Intro", "Text."])) };
    let (url, page) = assemble(front_of(JATS), &meta, parts).unwrap();
    assert_eq!(url, "https://pmc.ncbi.nlm.nih.gov/articles/PMC7/");
    assert_eq!(
        page.text.lines().collect::<Vec<_>>(),
        ["Risk factors", "Yuji Masataka", "J Rep. 2022; 1", "DOI: 10.1002/npr2.12310", "## Abstract", "### Aim", "Aim text.", "Short.", "## Intro", "Text."]
    );
    assert_eq!((page.canonical.as_str(), page.site_name.as_str(), page.author.as_str()), (url.as_str(), "PMC", "Yuji Masataka"));
}

fn renderer() -> Option<Arc<Renderer>> {
    let (python, root) = python()?;
    Some(Arc::new(Renderer::spawn(&python, &root, 2).unwrap()))
}

#[test]
fn pmc_full_text_drops_citation_marks_formulas_and_references() {
    let Some(renderer) = renderer() else { return };
    let meta = json!({"pmcid": "PMC10009416", "citation": "Neuropsychopharmacol Rep. 2022;43(1):85-94.", "is_pmc_openaccess": true, "license_code": "CC BY"});
    let Outcome::Page { url, page, no_body } = article_page(JATS.as_bytes(), &meta, &renderer).unwrap() else { panic!("no page") };
    assert!(!no_body);
    assert_eq!(url, "https://pmc.ncbi.nlm.nih.gov/articles/PMC10009416/");
    assert_eq!(
        page.text.lines().collect::<Vec<_>>(),
        [
            "Risk factors",
            "Yuji Masataka",
            "Neuropsychopharmacol Rep. 2022;43(1):85-94.",
            "DOI: 10.1002/npr2.12310",
            "## Abstract",
            "### Aim",
            "To determine risk factors.",
            "## INTRODUCTION",
            "Laws changed in 2022.",
            "Seen before and after.",
            "### Survey",
            "See Figure 1.",
            "Respondents by age.",
            "## Methods",
            "We asked users.",
        ]
    );
    assert_eq!((page.published.as_str(), page.tags.clone()), ("2022-12-19", vec!["cannabis".to_string()]));
    assert_eq!(renderer.render(b"<article").unwrap(), Rendered::BadXml);
    assert_eq!(article_page(b"<article><front/></article>", &meta, &renderer).unwrap(), Outcome::Skipped("no_front"));
}

const GOOD: &str = r#"{"pmcid": "PMC%", "pmid": %, "citation": "J. 2024;1:2.", "is_pmc_openaccess": true, "license_code": "CC BY"}"#;

fn article(n: u32, abstract_type: &str, body: bool) -> String {
    let body = if body { "<body><sec><title>Results</title><p>It <italic>works</italic>.</p></sec></body>" } else { "" };
    format!(
        r#"<article><front><article-meta><title-group><article-title>Paper {n}</article-title></title-group>
<abstract{abstract_type}><p>Abstract of {n}.</p></abstract><abstract abstract-type="graphical"><p>Not this.</p></abstract></article-meta></front>{body}</article>"#
    )
}

/// A folder with one version of every kind `run` counts.
fn write_folder(dir: &Path) {
    let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).unwrap();
    for n in 1..=3 {
        write(&format!("PMC{n}.1.json"), &GOOD.replace('%', &n.to_string()));
    }
    write("PMC1.1.xml", &article(1, "", true));
    write("PMC2.1.xml", &article(2, r#" abstract-type="summary""#, false));
    write("PMC3.1.xml", &article(3, r#" abstract-type="teaser""#, true));
    write("PMC4.1.json", r#"{"pmcid": "PMC4", "is_pmc_openaccess": true, "license_code": "CC BY-NC"}"#);
    write("PMC4.1.xml", &article(4, "", true));
    write("PMC5.1.json", r#"{"pmcid": "PMC5", "is_pmc_openaccess": true, "license_code": "CC0"}"#);
    write("PMC6.1.json", r#"{"pmcid": "PMC6", "is_pmc_openaccess": true, "license_code": "CC BY-SA", "pmid": 6}"#);
    write("PMC6.1.xml", "<article><front><article-meta>");
    write("PMC7.1.json", r#"{"pmcid": "PMC7", "is_pmc_openaccess": true, "license_code": "CC BY-ND"}"#);
    write("PMC7.1.xml", "<article><front><journal-meta/></front><body><p>x</p></body></article>");
    write("PMC8.1.json", r#"{"pmcid": "PMC8", "is_pmc_openaccess": false, "license_code": "CC BY"}"#);
    write("PMC9.1.json", r#"{"pmcid": "PMC9", "is_pmc_openaccess": true, "license_code": "CC BY", "is_retracted": true}"#);
    write("PMC10.1.json", r#"{"pmcid": "PMC10", "is_pmc_openaccess": true, "license_code": "CC BY"}"#);
    write("PMC10.1.xml", "<article><front><article-meta><title-group><article-title/></title-group></article-meta></front></article>");
}

fn args(folder: &Path, out: &Path, python: &str, root: &Path, limit: u64, pmids: &Path) -> Args {
    Args {
        run: RunArgs { out: out.to_path_buf(), as_of: "2026-11-01T00:00:00Z".into(), run: "test".into(), limit },
        inventory: Vec::new(),
        folder: Some(folder.to_path_buf()),
        names: None,
        workers: 4,
        pmids_out: Some(pmids.to_path_buf()),
        render: RenderArgs { python: python.into(), python_path: Some(root.to_path_buf()), render_workers: Some(3) },
    }
}

fn python_run(python: &str, root: &Path, folder: &Path, out: &Path, limit: u64, pmids: &Path) {
    let status = Command::new(python)
        .args(["-m", "bulk", "pmc", "--folder"])
        .arg(folder)
        .arg("--out")
        .arg(out)
        .args(["--as-of", "2026-11-01T00:00:00Z", "--run", "test", "--limit", &limit.to_string(), "--pmids-out"])
        .arg(pmids)
        .env("PYTHONPATH", root)
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn pmc_folder_runs_match_the_python_run() {
    let Some((python, root)) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join("input");
    std::fs::create_dir(&folder).unwrap();
    write_folder(&folder);
    for limit in [0, 2] {
        let (ours, theirs) = (dir.path().join(format!("rust-{limit}")), dir.path().join(format!("python-{limit}")));
        let (our_pmids, their_pmids) = (dir.path().join(format!("rust-{limit}.txt")), dir.path().join(format!("python-{limit}.txt")));
        bulk_pmc::main(args(&folder, &ours, &python, &root, limit, &our_pmids)).unwrap();
        python_run(&python, &root, &folder, &theirs, limit, &their_pmids);
        let diff = bulk_core::runs::diff(&ours, &theirs, 5).unwrap();
        assert!(diff.manifest_differences.is_empty() && diff.rows_differing == 0, "{diff:?}");
        assert_eq!(diff.rows_a, if limit == 0 { 3 } else { 2 });
        assert_eq!(std::fs::read_to_string(&our_pmids).unwrap(), std::fs::read_to_string(&their_pmids).unwrap());
    }
}

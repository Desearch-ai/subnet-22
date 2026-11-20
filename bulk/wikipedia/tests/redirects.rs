use std::path::{Path, PathBuf};

use bulk_core::{upload_row, Batches, Counts, Page};
use bulk_wikipedia::redirects::{self, Found, RedirectsArgs};
use serde_json::json;

const WIKI: &str = "https://en.wikipedia.org/wiki/";

fn wiki(title: &str) -> String {
    format!("{WIKI}{title}")
}

fn run_with(out: &Path, titles: &[&str]) {
    let mut batches = Batches::new(out, "wikipedia", "enwiki-test", 0).unwrap();
    for title in titles {
        let page = Page { title: title.to_string(), text: format!("{title} body"), ..Page::default() };
        batches.add(upload_row(&wiki(title), page, 0, "text/html")).unwrap();
    }
    batches.close(&Counts::default()).unwrap();
}

fn article(id: i64, revision: i64, title: &str, redirects: &[&str]) -> serde_json::Value {
    json!({
        "identifier": id,
        "name": title.replace('_', " "),
        "url": wiki(title),
        "namespace": {"identifier": 0},
        "version": {"identifier": revision},
        "article_body": {"html": "<html><body><p>text</p></body></html>"},
        "redirects": redirects.iter().map(|r| json!({"name": r, "url": wiki(&r.replace(' ', "_"))})).collect::<Vec<_>>(),
    })
}

fn snapshot(dir: &Path) -> PathBuf {
    let mut deleted = article(5, 9, "Deleted_page", &["Old deleted"]);
    deleted["event"] = json!({"type": "delete"});
    let mut category = article(8, 1, "Category:Things", &["Category:Stuff"]);
    category["namespace"] = json!({"identifier": 14});
    let lines = [
        article(1, 5, "BigHit_Music", &["Big Hit Music", "Shared name", "GoldenEye 007", "BigHit Music", "Big Hit's Music?"]),
        article(1, 3, "BigHit_Music", &["Older copy only"]),
        article(2, 2, "Demographics_of_Timor-Leste", &["Demographics of East Timor", "Shared name"]),
        article(2, 1, "Demographics_of_Timor-Leste", &["Older too"]),
        article(3, 1, "Not_in_the_run", &["To a dropped page"]),
        deleted,
        category,
        article(7, 4, "GoldenEye_007", &["GoldenEye 007 (1997 video game)", "Kızılağaç (film)"]),
    ];
    let path = dir.join("articles.ndjson");
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn redirects_lead_to_the_runs_pages() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    run_with(&run, &["BigHit_Music", "Demographics_of_Timor-Leste", "GoldenEye_007"]);
    let pages = bulk_core::runs::page_urls(&run).unwrap().into_iter().collect();
    let mut found = Found::default();
    let kept = redirects::redirects(&[snapshot(dir.path())], &pages, &mut found).unwrap();
    let pairs: Vec<(String, String)> = kept.into_iter().collect();
    assert_eq!(
        pairs,
        vec![
            (wiki("Big_Hit%27s_Music%3F"), wiki("BigHit_Music")),
            (wiki("Big_Hit_Music"), wiki("BigHit_Music")),
            (wiki("Demographics_of_East_Timor"), wiki("Demographics_of_Timor-Leste")),
            (wiki("GoldenEye_007_(1997_video_game)"), wiki("GoldenEye_007")),
            (wiki("K%C4%B1z%C4%B1la%C4%9Fa%C3%A7_(film)"), wiki("GoldenEye_007")),
        ]
    );
    assert_eq!(found, Found { articles: 4, older_copies: 2, redirects: 10, not_in_run: 1, page_urls: 2, ambiguous: 1, kept: 5 });
}

#[test]
fn main_writes_the_runs_redirects_file() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    run_with(&run, &["BigHit_Music", "Demographics_of_Timor-Leste", "GoldenEye_007"]);
    let args = RedirectsArgs { run: run.clone(), out: None, files: vec![snapshot(dir.path())] };
    redirects::main(args).unwrap();
    let text = std::fs::read_to_string(run.join(redirects::FILE)).unwrap();
    assert_eq!(text.lines().next().unwrap(), format!("{}\t{}", wiki("Big_Hit%27s_Music%3F"), wiki("BigHit_Music")));
    assert_eq!(text.lines().count(), 5);
    assert!(!run.join("redirects.tsv.partial").exists());
}

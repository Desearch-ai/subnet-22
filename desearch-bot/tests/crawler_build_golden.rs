//! The build's checks against the frozen golden answers in `data/crawler_build_golden.json`.

use std::sync::LazyLock;

use serde_json::Value;
use sha2::{Digest, Sha256};

use desearch_bot::pagecheck::{fred_newest_first, is_mirror, is_wall, item_sections, text_headings};
use desearch_bot::pystr::unquote;
use desearch_bot::setrule::{english, normal_host, page_date, page_eligible, SetRule};

static GOLDEN: LazyLock<Value> =
    LazyLock::new(|| serde_json::from_str(include_str!("data/crawler_build_golden.json")).expect("the golden file parses"));

fn cases(name: &str) -> &'static Vec<Value> {
    GOLDEN[name].as_array().expect("a list of cases")
}

fn s(v: &Value) -> &str {
    v.as_str().expect("a string")
}

fn check<T: PartialEq + std::fmt::Debug>(name: &str, got: Vec<(String, T, T)>) {
    let wrong: Vec<_> = got.iter().filter(|(_, a, b)| a != b).collect();
    assert!(wrong.is_empty(), "{name}: {} of {} differ, first: {:?}", wrong.len(), got.len(), wrong.first());
}

#[test]
fn the_set_rule_admits_the_golden_cases() {
    let rule = SetRule::parse(GOLDEN["set"].to_string().as_bytes()).unwrap();
    let mut admits = Vec::new();
    let mut dates = Vec::new();
    for c in cases("setrule") {
        let (url, published, page_type) = (s(&c[0]), s(&c[1]), s(&c[2]));
        if let Some(want) = c[3].as_bool() {
            admits.push((url.to_string(), rule.admits(url, Some(published), Some(page_type)), want));
        }
        let got = page_date(Some(published), url).map(|d| d.to_string());
        dates.push((format!("{url} {published}"), got, c[4].as_str().map(str::to_string)));
    }
    check("admits", admits);
    check("page_date", dates);
}

#[test]
fn hosts_normalise_as_the_golden_cases_say() {
    check("normal_host", cases("hosts").iter().map(|c| (s(&c[0]).to_string(), normal_host(s(&c[0])), s(&c[1]).to_string())).collect());
}

#[test]
fn languages_and_eligibility_match() {
    let mut eng = Vec::new();
    let mut ok = Vec::new();
    for c in cases("langs") {
        let (lang, text) = (s(&c[0]), s(&c[1]));
        eng.push((text.to_string(), english(text), c[2].as_bool().unwrap()));
        ok.push((format!("{lang:?} {text}"), page_eligible(lang, "v2", text), c[3].as_bool().unwrap()));
    }
    check("english", eng);
    check("page_eligible", ok);
}

#[test]
fn walls_mirrors_and_unquote_match() {
    check("walls", cases("walls").iter().map(|c| (format!("{c}"), is_wall(s(&c[0]), s(&c[1])), c[2].as_bool().unwrap())).collect());
    check("mirror", cases("mirror").iter().map(|c| (s(&c[0]).to_string(), is_mirror(s(&c[0])), c[1].as_bool().unwrap())).collect());
    check("unquote", cases("unquote").iter().map(|c| (s(&c[0]).to_string(), unquote(s(&c[0])), s(&c[1]).to_string())).collect());
}

#[test]
fn headings_and_edgar_items_match() {
    let strings = |v: &Value| v.as_array().unwrap().iter().map(|x| s(x).to_string()).collect::<Vec<_>>();
    check("headings", cases("headings").iter().map(|c| (s(&c[0]).to_string(), text_headings(s(&c[0])), strings(&c[1]))).collect());
    let short = |body: &str| Sha256::digest(body.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect::<String>();
    check(
        "items",
        cases("items")
            .iter()
            .map(|c| {
                let got: Vec<(String, String)> = item_sections(s(&c[0])).into_iter().map(|(i, b)| (i, short(&b))).collect();
                let want: Vec<(String, String)> = c[1].as_array().unwrap().iter().map(|p| (s(&p[0]).to_string(), s(&p[1]).to_string())).collect();
                (s(&c[0]).to_string(), got, want)
            })
            .collect(),
    );
}

#[test]
fn fred_tables_rewrite_byte_for_byte() {
    check("fred", cases("fred").iter().map(|c| (s(&c[0]).to_string(), fred_newest_first(s(&c[0])), s(&c[1]).to_string())).collect());
}

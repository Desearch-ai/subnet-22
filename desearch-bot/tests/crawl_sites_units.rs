//! The crawl-sites readers against the frozen golden answers in `tests/data/crawl_sites/units.json`.

use std::collections::HashMap;

use base64::Engine;
use serde_json::{json, Value};

use desearch_bot::sitecrawl::fetch::{wire_url, Response};
use desearch_bot::sitecrawl::links::{in_scope, normalize, page_links, parse_html, robots_meta, sitemap_entries};
use desearch_bot::sitecrawl::robots::parse;
use desearch_bot::sitecrawl::scope::declared_lang;
use desearch_bot::urls;

fn units() -> Value {
    serde_json::from_str(include_str!("data/crawl_sites/units.json")).unwrap()
}

fn b64(s: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s.as_str().unwrap()).unwrap()
}

fn check(kind: &str, case: &Value, got: Value, want: &Value, wrong: &mut Vec<String>) {
    if &got != want {
        wrong.push(format!("{kind} {case}\n  want {want}\n  got  {got}"));
    }
}

#[test]
fn readers_give_the_golden_answers() {
    let u = units();
    let mut wrong = Vec::new();
    for case in u["html"].as_array().unwrap() {
        let got = match parse_html(case[0].as_str().unwrap()) {
            None => Value::Null,
            Some(doc) => {
                let mut meta: Vec<String> = robots_meta(&doc, "").into_iter().collect();
                meta.sort();
                let links = page_links(&doc, "https://ex.test/p/q").map_or(json!("ValueError"), |l| json!(l));
                json!({"lang": declared_lang(&doc), "meta": meta, "links": links})
            }
        };
        check("html", &case[0], got, &case[1], &mut wrong);
    }
    for case in u["normalize"].as_array().unwrap() {
        let got = normalize(case[0].as_str().unwrap(), case[1].as_str().unwrap());
        check("normalize", &json!([case[0], case[1]]), json!(got), &case[2], &mut wrong);
    }
    for case in u["scope"].as_array().unwrap() {
        let prefixes: Vec<String> = serde_json::from_value(case[2].clone()).unwrap();
        let got = in_scope(case[0].as_str().unwrap(), case[1].as_str().unwrap(), &prefixes);
        check("scope", case, json!(got), &case[3], &mut wrong);
    }
    let robot_urls: Vec<String> = serde_json::from_value(u["robot_urls"].clone()).unwrap();
    for case in u["robots"].as_array().unwrap() {
        let r = parse("a.test", 200, case[0].as_str().unwrap());
        let allowed: Vec<&String> = robot_urls.iter().filter(|url| r.allowed(url)).collect();
        let got = json!({"delay": r.delay, "sitemaps": r.sitemaps, "allowed": allowed});
        check("robots", &case[0], got, &case[1], &mut wrong);
    }
    for case in u["sitemaps"].as_array().unwrap() {
        let got = match sitemap_entries(&b64(&case[0])) {
            Ok((pages, children)) => json!([pages, children]),
            Err(crash) => json!({"error": crash.0}),
        };
        check("sitemap", &json!(String::from_utf8_lossy(&b64(&case[0]))), got, &case[1], &mut wrong);
    }
    for case in u["charsets"].as_array().unwrap() {
        let r = Response {
            url: "u".into(),
            status: 200,
            headers: HashMap::from([("content-type".into(), case[1].as_str().unwrap().into())]),
            body: b64(&case[0]),
            fetched_at: chrono::Utc::now(),
            error: None,
        };
        check("charset", &case[1], json!(r.text()), &case[2], &mut wrong);
    }
    for case in u["wire"].as_array().unwrap() {
        let got = wire_url(&urls::split(case[0].as_str().unwrap()).unwrap());
        check("wire", &case[0], json!(got), &case[1], &mut wrong);
    }
    assert!(wrong.is_empty(), "{} differences:\n{}", wrong.len(), wrong.join("\n"));
}

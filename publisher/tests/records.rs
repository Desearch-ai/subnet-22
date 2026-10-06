//! Records, versions, fetch-time windows and blocked pages exactly as `publisher.records` and `looks_blocked` give them.

use publisher::blocked::looks_blocked;
use publisher::records::{build_record, from_timestamp, publish_window, record_key, record_version, timedelta, Context, Record, Row, SECOND};

const COMPLETED: f64 = 1791310017.8421726;
const CAPTURED: i64 = 1_791_311_000 * SECOND;

fn record(text: &str) -> Record {
    Record {
        url: "https://ex.com/a".into(),
        domain: "ex.com".into(),
        title: "T".into(),
        published: String::new(),
        author: String::new(),
        lang: "en".into(),
        text: text.into(),
        fetched_at: String::new(),
        content_sha1: String::new(),
        captured_at: String::new(),
        doc_id: String::new(),
        assigned_url: String::new(),
        final_url: None,
        canonical: None,
        status: None,
        page_type: None,
        description: None,
        json_ld_types: vec![],
        headings: vec![],
        text_sha256: None,
        task_id: String::new(),
        miner: String::new(),
        validator: None,
        validators: vec![],
    }
}

fn row(fetched_at: Option<i64>) -> Row {
    Row {
        url: Some("https://www.Example.com/news/story?utm_source=x&id=7#top".into()),
        final_url: Some("https://example.com/news/story".into()),
        status: Some(200),
        fetched_at,
        description: Some("d".into()),
        headings: Some(vec![Some("a".into())]),
        text: Some("body text".into()),
        text_sha256: Some("x".into()),
        ..Row::default()
    }
}

#[test]
fn versions_hash_python_json() {
    assert_eq!(record_version(&record("hello")), "b9cac78089c4d748d7714c374e132e10d33e7f7e");
    let escaped = Record {
        title: "\u{dc}n\u{ef}code \u{201c}quotes\u{201d}".into(),
        ..record("quote \" back \\ nl \n cr \r tab \t bs \u{8} ff \u{c} esc \u{1b} del \u{7f} ls \u{2028} emoji \u{1f600} \u{e9}")
    };
    assert_eq!(record_version(&escaped), "0735009d2fdfc0488de9d65fc1b09a904c3c9a23");
    let filled = Record {
        canonical: Some("https://ex.com/a".into()),
        page_type: Some("article".into()),
        description: Some(String::new()),
        json_ld_types: vec![Some("NewsArticle".into()), None],
        headings: vec![Some("H1".into()), Some(String::new()), None],
        ..record("hello")
    };
    assert_eq!(record_version(&filled), "88525c60a78e697da2708c6238242e26168e2ec8");
    let controls = Record { author: "\u{0}\u{1}\u{1f}".into(), lang: String::new(), published: "2026-10-06T10:00:00Z".into(), ..record("hello") };
    assert_eq!(record_version(&controls), "f0de0565afbbebe70148f130e74ba9fdd4c18025");
}

#[test]
fn fetch_times_are_held_to_the_claim_window() {
    let window = publish_window(Some(COMPLETED), Some(180.0), 0).unwrap();
    let context = Context { task_id: "t1", miner: "m1", window, captured_at: CAPTURED, validator: Some("v1"), validators: &[] };
    let at = |seconds: i64| seconds * SECOND;
    let cases = [
        (Some(at(1_791_288_000)), "2026-10-06T17:58:57+00:00"),
        (Some(at(4_070_908_800)), "2026-10-06T18:06:57+00:00"),
        (Some(at(1_791_310_000) + SECOND / 2), "2026-10-06T18:06:40+00:00"),
        (None, "2026-10-06T18:06:57+00:00"),
    ];
    for (fetched_at, expected) in cases {
        assert_eq!(build_record(&row(fetched_at), &context).unwrap().fetched_at, expected, "{fetched_at:?}");
    }
}

#[test]
fn records_carry_every_field_as_python_builds_them() {
    let window = publish_window(Some(COMPLETED), Some(180.0), 0).unwrap();
    let context = Context { task_id: "t1", miner: "m1", window, captured_at: CAPTURED, validator: Some("v1"), validators: &[] };
    let built = build_record(&row(Some(1_791_288_000 * SECOND)), &context).unwrap();
    assert_eq!(built.url, "https://www.example.com/news/story?id=7");
    assert_eq!((built.title.as_str(), built.lang.as_str(), built.author.as_str()), ("", "", ""));
    assert_eq!(built.content_sha1, "8fe8c3d40e967714951a71b824a0471440aa3d5b");
    assert_eq!(built.captured_at, "2026-10-06T18:23:20+00:00");
    assert_eq!(built.doc_id, "d8a4b2ef-4d31-5585-a337-a00c6f61c26b");
    assert_eq!(built.json_ld_types, Vec::<Option<String>>::new());
    assert_eq!(built.validators, ["v1"]);
    assert_eq!(record_version(&built), "e0fe5230fb09f509f4c5cca192cc529e99450649");
    assert_eq!(record_key(&built).unwrap(), "pages/example.com/02722b06dd321d090d4c8b6b2b099dcd30a7a544");
    let listed = ["a".to_string(), "b".to_string()];
    let given = Context { validator: None, validators: &listed, ..context };
    let built = build_record(&row(None), &given).unwrap();
    assert_eq!((built.validator, built.validators), (None, listed.to_vec()));
}

#[test]
fn times_round_as_python_does() {
    assert_eq!(from_timestamp(1791310017.8421726).unwrap(), 1_791_310_017_842_173);
    for (seconds, micros) in [(5e-7, 0), (1.5e-6, 2), (1.0000005, 1_000_001), (2.5e-6, 2)] {
        assert_eq!(from_timestamp(seconds).unwrap(), micros, "{seconds}");
    }
    for (seconds, micros) in [(180.0, 180_000_000), (5e-7, 0), (1.5e-6, 2), (2.5e-6, 2), (-5e-7, 0), (0.1, 100_000)] {
        assert_eq!(timedelta(seconds).unwrap(), micros, "{seconds}");
    }
    assert!(from_timestamp(1e15).is_err());
}

#[test]
fn blocked_pages_are_judged_as_python_judges_them() {
    let long = "x".repeat(600);
    let cases: [(Option<i32>, &str, &str, bool); 21] = [
        (Some(200), "", "Just a moment...", true),
        (Some(200), "x", "Attention Required! | Cloudflare", true),
        (Some(200), "short", "Forbidden fruit - a novel", true),
        (Some(200), &long, "Forbidden fruit - a novel", false),
        (Some(403), "short", "Forbiddenfruit", true),
        (Some(403), &"real words ".repeat(400), "Access denied", false),
        (Some(200), "please verify you are human", "Example", true),
        (Some(503), &long, "Forbiddenfruit", false),
        (Some(503), &long, "Forbidden fruit", true),
        (Some(503), &long, "Captcha\u{301}", true),
        (Some(503), &long, "Captcha\u{b2}", false),
        (Some(200), &long, "Just a moment - Cloudflare\u{b2}", false),
        (Some(200), &long, "Just a moment - Cloudflare", true),
        (Some(200), &long, "Just a moment... | Example", false),
        (Some(200), &long, "ACCESS DEN\u{130}ED", true),
        (Some(200), &long, "Access den\u{131}ed", true),
        (Some(200), &long, "Verifying that you are a human | DataDome", true),
        (Some(200), &long, "Error 1020 \u{2013} sucuri", true),
        (None, &"x".repeat(100), "hello", false),
        (Some(429), &"x".repeat(100), "hello", true),
        (Some(503), &format!("Too many requests {}", "y".repeat(80)), "x", true),
    ];
    for (status, text, title, expected) in cases {
        assert_eq!(looks_blocked(status, text, title), expected, "{status:?} {title:?} {}", text.len());
    }
    let lengthened = format!("{} {}", "\u{130}".repeat(1600), "a".repeat(1300));
    assert!(!looks_blocked(Some(200), &lengthened, "Just a moment"), "lowercasing İ makes the text long enough to be real");
}

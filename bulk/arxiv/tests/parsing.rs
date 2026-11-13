use std::path::{Path, PathBuf};

use bulk_arxiv::*;

const ARXIV: &str = r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:1303.7276</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>1303.7276</id><created>2013-03-29</created><updated>2026-10-26</updated>
<authors><author><keyname>Li</keyname><forenames>Chen</forenames></author></authors>
<title>The effect of limb
 kinematics</title><categories>physics.bio-ph cs.RO</categories><abstract>  Legged robots on sand.  </abstract></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:hep-th/9901001</identifier></header></record>
</ListRecords></OAI-PMH>"#;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn papers(bytes: &[u8]) -> Vec<OaiItem> {
    papers_in(bytes).unwrap().into_iter().map(Result::unwrap).collect()
}

fn paper(fields: [&str; 10], authors: &[&str]) -> Paper {
    let [arxiv_id, title, abstract_text, categories, created, updated, comments, journal, doi, licence] = fields.map(String::from);
    Paper {
        arxiv_id,
        title,
        authors: authors.iter().map(|a| a.to_string()).collect(),
        abstract_text,
        categories,
        created,
        updated,
        comments,
        journal,
        doi,
        licence,
    }
}

#[test]
fn urls_follow_the_html_view_cutoff() {
    assert_eq!(url_of("1303.7276"), "https://arxiv.org/abs/1303.7276");
    assert_eq!(html_url("2312.00001"), "https://arxiv.org/html/2312.00001");
    assert_eq!(html_url("2401.00001"), "https://arxiv.org/html/2401.00001");
    assert_eq!(html_url("2311.1"), "https://ar5iv.labs.arxiv.org/html/2311.1");
    assert_eq!(html_url("hep-th/9901001"), "https://ar5iv.labs.arxiv.org/html/hep-th/9901001");
    assert_eq!(html_url("23"), "https://ar5iv.labs.arxiv.org/html/23");
    assert_eq!(html_url("9901.1"), "https://arxiv.org/html/9901.1", "a plain string comparison, as Python's");
}

#[test]
fn open_licences_ignore_the_scheme_and_outer_space() {
    assert!(is_open("http://creativecommons.org/licenses/by/4.0/"));
    assert!(is_open(" https://creativecommons.org/publicdomain/zero/1.0/\n"));
    assert!(is_open("creativecommons.org/licenses/by-sa/4.0/"));
    assert!(!is_open("http://creativecommons.org/licenses/by-nc/4.0/"));
    assert!(!is_open("http://arxiv.org/licenses/nonexclusive-distrib/1.0/"));
    assert!(!is_open(""));
    assert!(!is_open("http://creativecommons.org/licenses/by/4.0"));
}

#[test]
fn squash_splits_on_python_whitespace() {
    assert_eq!(squash("  a\u{1f}b \u{a0} c\u{3000}\n"), "a b c");
    assert_eq!(squash(""), "");
}

#[test]
fn the_test_page_reads_as_python_reads_it() {
    let items = papers(ARXIV.as_bytes());
    assert_eq!(
        items,
        [
            OaiItem::Paper(paper(
                ["1303.7276", "The effect of limb kinematics", "Legged robots on sand.", "physics.bio-ph cs.RO", "2013-03-29", "2026-10-26", "", "", "", ""],
                &["Chen Li"]
            )),
            OaiItem::Deleted("hep-th/9901001".to_string()),
        ]
    );
}

#[test]
fn text_stops_at_the_first_child_and_entities_cdata_and_crlf_read_as_lxml_reads_them() {
    let items = papers(&std::fs::read(data("mixed/input/oai-000001.xml")).unwrap());
    let expected = [
        OaiItem::Paper(paper(
            [
                "2401.00001",
                "A & B: <raw> & done now",
                "First line second line <b>",
                "cs.LG stat.ML math.OC",
                "2024-01-02",
                "2024-01-02",
                "12 pages, 5 figures",
                "Phys. Rev. D 99, 1 (2019)",
                "10.1103/x",
                "http://creativecommons.org/licenses/by/4.0/",
            ],
            &["María José de la Cruz Jr", "Collab", "Solo", "Shown", "List Second"],
        )),
        OaiItem::Deleted(" oai:arXiv.org:math/0101001".to_string()),
        OaiItem::Deleted("0704.0001".to_string()),
        OaiItem::Paper(paper(
            ["2402.0002", "", "Only an abstract.", "", "2024-02-03", "2024-02-05", "", "", "", "https://creativecommons.org/publicdomain/zero/1.0/"],
            &[],
        )),
        OaiItem::Paper(paper(
            ["2403.0003", "Second arXiv element", "An abstract.", "", "", "", "", "", "", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"],
            &[],
        )),
    ];
    assert_eq!(items, expected);
}

#[test]
fn crlf_and_carriage_return_references_squash_away() {
    let page = "<OAI-PMH xmlns=\"http://www.openarchives.org/OAI/2.0/\"><record><header/><metadata><arXiv xmlns=\"http://arxiv.org/OAI/arXiv/\">\r\n<id>\r\n1\r\n</id><title>a\r\nb&#13;c</title></arXiv></metadata></record></OAI-PMH>";
    let items = papers(page.as_bytes());
    let [OaiItem::Paper(paper)] = items.as_slice() else { panic!("{items:?}") };
    assert_eq!((paper.arxiv_id.as_str(), paper.title.as_str()), ("1", "a b c"));
}

#[test]
fn a_record_without_a_header_or_metadata_ends_the_page_with_an_error() {
    let page = r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header status="deleted"><identifier>oai:arXiv.org:1</identifier></header></record>
<record><header><identifier>oai:arXiv.org:2</identifier></header></record>
<record><header status="deleted"><identifier>oai:arXiv.org:3</identifier></header></record>
</ListRecords></OAI-PMH>"#;
    let items = papers_in(page.as_bytes()).unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].as_ref().unwrap(), &OaiItem::Deleted("1".to_string()));
    assert!(items[1].as_ref().unwrap_err().to_string().contains("no arXiv metadata"));
    let page = r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><record/></OAI-PMH>"#;
    assert!(papers_in(page.as_bytes()).unwrap()[0].is_err());
    assert!(papers_in(b"<OAI-PMH><record>").is_err(), "a page that does not parse fails whole");
}

#[test]
fn records_outside_the_oai_namespace_are_not_records() {
    let page = r#"<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><record xmlns="urn:x"><header status="deleted"/></record></OAI-PMH>"#;
    assert!(papers_in(page.as_bytes()).unwrap().is_empty());
}

#[test]
fn heading_lines_name_the_update_only_when_it_differs() {
    let mut p = paper(["1303.7276", "T", "", "physics.bio-ph cs.RO", "2013-03-29", "2026-10-26", "", "", "", ""], &["Chen Li"]);
    let text = |p: &Paper| heading_lines(p).into_iter().map(|l| l.text).collect::<Vec<_>>();
    assert_eq!(text(&p), ["Chen Li", "arXiv:1303.7276 · physics.bio-ph cs.RO · Submitted 2013-03-29, updated 2026-10-26"]);
    p.updated = p.created.clone();
    p.categories.clear();
    assert_eq!(text(&p), ["Chen Li", "arXiv:1303.7276 · Submitted 2013-03-29"]);
    p.updated.clear();
    assert_eq!(text(&p)[1], "arXiv:1303.7276 · Submitted 2013-03-29");
}

#[test]
fn kaggle_dates_follow_email_utils() {
    for (text, day) in [
        ("Mon, 2 Apr 2007 19:18:42 GMT", "2007-04-02"),
        ("Tue, 3 Apr 2007 00:00:00 +2359", "2007-04-03"),
        ("2 Apr 07 19:18 -0000", "2007-04-02"),
        ("Mon,2 Apr 2007 19:18:42 GMT", "2007-04-02"),
        ("Apr 2 2007 19:18:42", "2007-04-02"),
        ("Mon, 2-Apr-2007 19:18:42 GMT", "2007-04-02"),
        ("Wed, 31 Feb 2007 10:00:00 GMT", ""),
        ("", ""),
        ("   ", ""),
        ("Mon, 02 Apr 2007 23:30:00 -0700", "2007-04-02"),
        ("Mon, 2 Apr 2_007 19:18:42 GMT", "2007-04-02"),
        ("Mon, 2 Apr 2007 19.18.42 GMT", "2007-04-02"),
        ("Mon, 2 Apr 2007 19:18:42+0100", "2007-04-02"),
        ("Mon, 2 April 2007 19:18 EST", "2007-04-02"),
        ("Mon, 2 Apr 2007 24:00:00 GMT", ""),
        ("Mon, 2 Apr 2007 19:18:42 +2400", ""),
        ("2007 Apr 2 19:18:42", ""),
        ("Thu, 29 Feb 2024 01:02:03 GMT", "2024-02-29"),
        ("Thu, 29 Feb 2023 01:02:03 GMT", ""),
        ("garbage", ""),
        ("Mon, 2 Apr 69 1:2:3 GMT", "1969-04-02"),
    ] {
        assert_eq!(kaggle_date(text), day, "{text:?}");
    }
}

#[test]
fn harvest_queries_quote_as_urlencode_does() {
    assert_eq!(
        urlencode(&[("verb", "ListRecords"), ("metadataPrefix", "arXiv"), ("from", "2026-11-01"), ("until", "2026-11-02")]),
        "verb=ListRecords&metadataPrefix=arXiv&from=2026-11-01&until=2026-11-02"
    );
    assert_eq!(urlencode(&[("resumptionToken", "6960524|1001 a/b~é")]), "resumptionToken=6960524%7C1001+a%2Fb~%C3%A9");
}

#[test]
fn resumption_tokens_are_stripped_and_blank_ones_end_the_harvest() {
    let page = |token: &str| {
        format!(r#"<?xml version="1.0" encoding="UTF-8"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{token}</ListRecords></OAI-PMH>"#)
    };
    let token = |t: &str| resumption_token(page(t).as_bytes()).unwrap();
    assert_eq!(token(r#"<resumptionToken cursor="0"> 6960524|1001 </resumptionToken>"#).as_deref(), Some("6960524|1001"));
    assert_eq!(token(r#"<resumptionToken cursor="1000" completeListSize="1500"/>"#), None);
    assert_eq!(token("<resumptionToken>  </resumptionToken>"), None);
    assert_eq!(token(""), None);
    assert_eq!(token(r#"<resumptionToken xmlns="urn:x">t</resumptionToken>"#), None);
}

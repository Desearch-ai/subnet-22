use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use bulk_arxiv::harvest_with;

const AGENT: &str = "DesearchBulk/1.0 (+https://desearch.ai)";

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/harvest").join(name)
}

/// What the server answers one request with: None closes the connection without a response.
type Reply = Option<(u16, Vec<(&'static str, &'static str)>, Vec<u8>)>;

struct Seen {
    target: String,
    agent: String,
    at: Instant,
}

/// Serves `replies` in order, one a connection, and reports each request line's target, user agent and time.
fn serve(replies: Vec<Reply>) -> (String, mpsc::Receiver<Seen>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/oai", listener.local_addr().unwrap());
    let (sent, seen) = mpsc::channel();
    thread::spawn(move || {
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            let mut agent = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("user-agent") {
                        agent = value.trim().to_string();
                    }
                }
            }
            let target = first.split_whitespace().nth(1).unwrap_or("").to_string();
            sent.send(Seen { target, agent, at: Instant::now() }).unwrap();
            let Some((status, headers, body)) = reply else { continue };
            let mut head = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
            for (name, value) in headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (base, seen)
}

#[test]
fn pages_are_saved_as_sent_following_the_token_and_waiting_out_busy_replies() {
    let (one, two) = (std::fs::read(data("oai-1.xml")).unwrap(), std::fs::read(data("oai-2.xml")).unwrap());
    let (base, seen) =
        serve(vec![Some((200, vec![], one.clone())), Some((503, vec![("Retry-After", "1")], b"busy".to_vec())), Some((200, vec![], two.clone()))]);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("oai");
    let delay = Duration::from_millis(400);
    assert_eq!(harvest_with(&out, "2026-11-01", "2026-11-02", AGENT, &base, delay).unwrap(), 2);
    let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(names, ["oai-000001.xml", "oai-000002.xml"]);
    assert_eq!(std::fs::read(out.join("oai-000001.xml")).unwrap(), one);
    assert_eq!(std::fs::read(out.join("oai-000002.xml")).unwrap(), two);
    let seen: Vec<Seen> = seen.try_iter().collect();
    let targets: Vec<&str> = seen.iter().map(|s| s.target.as_str()).collect();
    assert_eq!(
        targets,
        [
            "/oai?verb=ListRecords&metadataPrefix=arXiv&from=2026-11-01&until=2026-11-02",
            "/oai?verb=ListRecords&resumptionToken=6960524%7C1001",
            "/oai?verb=ListRecords&resumptionToken=6960524%7C1001",
        ]
    );
    assert!(seen.iter().all(|s| s.agent == AGENT));
    assert!(seen[1].at - seen[0].at >= delay, "the delay between pages");
    assert!(seen[2].at - seen[1].at >= Duration::from_secs(1), "Retry-After on 503");
}

#[test]
fn a_dropped_connection_is_tried_again_after_a_second() {
    let two = std::fs::read(data("oai-2.xml")).unwrap();
    let (base, seen) = serve(vec![None, Some((429, vec![("Retry-After", "0")], vec![])), Some((200, vec![], two.clone()))]);
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(harvest_with(tmp.path(), "2026-11-01", "2026-11-02", AGENT, &base, Duration::ZERO).unwrap(), 1);
    assert_eq!(std::fs::read(tmp.path().join("oai-000001.xml")).unwrap(), two);
    let seen: Vec<Seen> = seen.try_iter().collect();
    assert_eq!(seen.len(), 3);
    assert!(seen[1].at - seen[0].at >= Duration::from_secs(1), "2**0 s after the first failure");
}

#[test]
fn other_statuses_fail_the_harvest_without_saving() {
    let (base, _seen) = serve(vec![Some((404, vec![], b"gone".to_vec()))]);
    let tmp = tempfile::tempdir().unwrap();
    let error = harvest_with(tmp.path(), "2026-11-01", "2026-11-02", AGENT, &base, Duration::ZERO).unwrap_err();
    assert!(error.to_string().contains("404"), "{error:#}");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
}

#[test]
fn a_page_that_is_not_xml_is_saved_then_fails() {
    let (base, _seen) = serve(vec![Some((200, vec![], b"<html>".to_vec()))]);
    let tmp = tempfile::tempdir().unwrap();
    assert!(harvest_with(tmp.path(), "2026-11-01", "2026-11-02", AGENT, &base, Duration::ZERO).is_err());
    assert_eq!(std::fs::read(tmp.path().join("oai-000001.xml")).unwrap(), b"<html>");
}

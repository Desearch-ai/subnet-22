//! A loopback HTTP server standing in for the buckets, and the Python worker the render tests use.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub const JATS: &str = r#"<article xml:lang="en"><front><article-meta><article-id pub-id-type="doi">10.1002/npr2.12310</article-id><title-group><article-title>Risk factors</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Masataka</surname><given-names>Yuji</given-names></name></contrib></contrib-group>
<pub-date pub-type="epub"><day>19</day><month>12</month><year>2022</year></pub-date>
<abstract><title>Abstract</title><sec><title>Aim</title><p>To determine risk factors.</p></sec></abstract>
<kwd-group><kwd>cannabis</kwd></kwd-group></article-meta></front>
<body><sec><title>INTRODUCTION</title><p>Laws changed in 2022.<xref ref-type="bibr" rid="b1">1</xref>, <xref ref-type="bibr" rid="b2">2</xref></p>
<p>Seen before [<xref ref-type="bibr" rid="b3">3</xref>] and after<sup><xref ref-type="bibr" rid="b4">4</xref></sup>.</p>
<sec><title>Survey</title><p>See <xref ref-type="fig" rid="f1">Figure 1</xref>.</p><fig><label>Figure 1</label><caption><p>Respondents by age.</p></caption></fig></sec></sec>
<sec><title>Methods</title><disp-formula><mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML"><mml:mi>x</mml:mi></mml:math></disp-formula><p>We asked users.</p></sec></body>
<back><ref-list><ref id="b1"><mixed-citation>Someone 2020.</mixed-citation></ref></ref-list></back></article>"#;

pub struct Request {
    pub path: String,
    pub range: Option<String>,
}

pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        Reply { status: 200, headers: Vec::new(), body: body.into() }
    }

    pub fn status(status: u16) -> Self {
        Reply { status, headers: Vec::new(), body: Vec::new() }
    }

    /// The body, or the part a `bytes=a-b` or `bytes=-n` range asks for.
    pub fn ranged(body: &[u8], range: Option<&str>) -> Self {
        let Some(spec) = range.and_then(|r| r.strip_prefix("bytes=")) else { return Reply::ok(body) };
        let len = body.len();
        let (start, end) = match spec.split_once('-') {
            Some(("", n)) => (len.saturating_sub(n.parse().unwrap()), len - 1),
            Some((a, "")) => (a.parse().unwrap(), len - 1),
            Some((a, b)) => (a.parse().unwrap(), b.parse::<usize>().unwrap().min(len - 1)),
            None => panic!("bad range {spec}"),
        };
        Reply { status: 206, headers: vec![("Content-Range".into(), format!("bytes {start}-{end}/{len}"))], body: body[start..=end].to_vec() }
    }
}

pub struct Server {
    pub base: String,
    pub log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    pub fn requests(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    pub fn count(&self, path: &str) -> usize {
        self.requests().iter().filter(|p| *p == path).count()
    }
}

fn answer(mut stream: TcpStream, handler: &(dyn Fn(&Request) -> Reply + Send + Sync), log: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let mut range = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("range") {
                range = Some(value.trim().to_string());
            }
        }
    }
    log.lock().unwrap().push(path.clone());
    let reply = handler(&Request { path, range });
    let mut head = format!("HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n", reply.status, reply.body.len());
    for (name, value) in &reply.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&reply.body);
}

/// A server answering each request on its own thread with `handler`, logging request paths in arrival order.
pub fn serve(handler: impl Fn(&Request) -> Reply + Send + Sync + 'static) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(Vec::new()));
    let handler: Arc<dyn Fn(&Request) -> Reply + Send + Sync> = Arc::new(handler);
    let shared = log.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let (handler, log) = (handler.clone(), shared.clone());
            std::thread::spawn(move || answer(stream, handler.as_ref(), &log));
        }
    });
    Server { base, log }
}

/// The Python and repo root the render tests use, from BULK_PYTHON; None skips them.
pub fn python() -> Option<(String, PathBuf)> {
    let python = std::env::var("BULK_PYTHON").ok().filter(|p| !p.is_empty());
    if python.is_none() {
        eprintln!("skipped: set BULK_PYTHON (and PYTHONPATH to the repo root) to run the render worker tests");
    }
    let root = std::env::var_os("PYTHONPATH")
        .map(PathBuf::from)
        .filter(|p| p.join("bulk/pmc_render.py").is_file())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
    python.map(|p| (p, root))
}

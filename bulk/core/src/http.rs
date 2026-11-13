//! Blocking HTTP for the sources: one client with the bulk user agent, and PMC's retry rule.

use std::time::Duration;

use anyhow::{bail, Context, Result};

pub const USER_AGENT: &str = "DesearchBulk/1.0 (+https://desearch.ai)";
pub const TIMEOUT: Duration = Duration::from_secs(300);
pub const TRIES: u32 = 6;
pub const RETRIED: [u16; 4] = [500, 502, 503, 504];

/// A response's status, headers and body, for sources with their own retry rules.
pub struct Response {
    pub status: u16,
    pub retry_after: Option<f64>,
    pub body: Vec<u8>,
}

pub struct Client {
    inner: reqwest::blocking::Client,
    wait_unit: Duration,
}

/// Why a GET did not return a body: a status to judge, or a transport error and whether `fetch_url` retries it.
enum Failure {
    Status(u16),
    Transport(anyhow::Error, bool),
}

impl Client {
    pub fn new(user_agent: &str) -> Result<Self> {
        let inner = reqwest::blocking::Client::builder().user_agent(user_agent).timeout(TIMEOUT).build().context("building the HTTP client")?;
        Ok(Client { inner, wait_unit: Duration::from_secs(1) })
    }

    /// One GET, no retries.
    pub fn get(&self, url: &str) -> Result<Response> {
        self.attempt(url).map_err(|(error, _)| error)
    }

    fn attempt(&self, url: &str) -> std::result::Result<Response, (anyhow::Error, bool)> {
        let transport = |error: reqwest::Error| {
            let retried = error.is_timeout() || error.is_connect() || error.is_request() || error.is_body();
            (anyhow::Error::new(error).context(format!("GET {url}")), retried)
        };
        let response = self.inner.get(url).send().map_err(transport)?;
        let status = response.status().as_u16();
        let retry_after =
            response.headers().get(reqwest::header::RETRY_AFTER).and_then(|value| value.to_str().ok()).and_then(|value| value.trim().parse::<f64>().ok());
        let body = response.bytes().map_err(transport)?.to_vec();
        Ok(Response { status, retry_after, body })
    }

    fn once(&self, url: &str) -> std::result::Result<Vec<u8>, Failure> {
        match self.attempt(url) {
            Ok(response) if (200..300).contains(&response.status) => Ok(response.body),
            Ok(response) => Err(Failure::Status(response.status)),
            Err((error, retried)) => Err(Failure::Transport(error, retried)),
        }
    }

    /// Retries 500, 502, 503, 504, timeouts and connection errors with 2**attempt s waits; None on 404.
    pub fn get_retrying(&self, url: &str) -> Result<Option<Vec<u8>>> {
        for attempt in 0..TRIES {
            let last = attempt == TRIES - 1;
            match self.once(url) {
                Ok(body) => return Ok(Some(body)),
                Err(Failure::Status(404)) => return Ok(None),
                Err(Failure::Status(status)) if !RETRIED.contains(&status) || last => bail!("HTTP Error {status} for {url}"),
                Err(Failure::Transport(error, retried)) if !retried || last => return Err(error),
                Err(_) => std::thread::sleep(self.wait_unit * 2u32.pow(attempt)),
            }
        }
        unreachable!("the last try returns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    /// A loopback server answering each connection with the next status, recording the user agent it saw.
    fn serve(statuses: Vec<u16>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/x", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut agents = Vec::new();
            for status in statuses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                    if let Some(agent) = line.strip_prefix("user-agent: ") {
                        agents.push(agent.trim().to_string());
                    }
                    line.clear();
                }
                let body = format!("body {status}");
                let reply = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nRetry-After: 7\r\nConnection: close\r\n\r\n{body}", body.len());
                stream.write_all(reply.as_bytes()).unwrap();
            }
            agents
        });
        (url, handle)
    }

    fn client() -> Client {
        Client { wait_unit: Duration::from_millis(1), ..Client::new(USER_AGENT).unwrap() }
    }

    #[test]
    fn retries_server_errors_then_returns_the_body() {
        let (url, server) = serve(vec![503, 500, 200]);
        assert_eq!(client().get_retrying(&url).unwrap().unwrap(), b"body 200");
        assert_eq!(server.join().unwrap(), vec![USER_AGENT; 3]);
    }

    #[test]
    fn not_found_is_none_and_other_errors_fail() {
        let (url, server) = serve(vec![404, 403]);
        assert!(client().get_retrying(&url).unwrap().is_none());
        assert!(client().get_retrying(&url).is_err());
        server.join().unwrap();
    }

    #[test]
    fn gives_up_after_six_tries() {
        let (url, server) = serve(vec![502; 6]);
        assert!(client().get_retrying(&url).is_err());
        server.join().unwrap();
    }

    #[test]
    fn plain_get_reports_status_and_retry_after() {
        let (url, server) = serve(vec![429]);
        let response = client().get(&url).unwrap();
        assert_eq!((response.status, response.retry_after, response.body.as_slice()), (429, Some(7.0), b"body 429".as_slice()));
        server.join().unwrap();
    }
}

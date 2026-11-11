//! Blocking HTTP for the sources: one client with the bulk user agent, and PMC's retry rule.

use std::time::Duration;

use anyhow::Result;

pub const USER_AGENT: &str = "DesearchBulk/1.0 (+https://desearch.ai)";
pub const TIMEOUT: Duration = Duration::from_secs(300);

/// A response's status, headers and body, for sources with their own retry rules.
pub struct Response {
    pub status: u16,
    pub retry_after: Option<f64>,
    pub body: Vec<u8>,
}

pub struct Client {
    _private: (),
}

impl Client {
    pub fn new(user_agent: &str) -> Result<Self> {
        todo!("{user_agent}")
    }

    /// One GET, no retries.
    pub fn get(&self, url: &str) -> Result<Response> {
        todo!("{url}")
    }

    /// `bulk.pmc.fetch_url`: retries 500, 502, 503, 504, timeouts and connection errors with 2**attempt s waits; None on 404.
    pub fn get_retrying(&self, url: &str) -> Result<Option<Vec<u8>>> {
        todo!("{url}")
    }
}

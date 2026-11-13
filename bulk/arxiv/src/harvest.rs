//! The OAI-PMH ListRecords harvest: every page saved as it came, following resumption tokens.

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bulk_core::http::Client;

use crate::oai::{first_descendant, parse, text_of, OAI_NS};

pub const OAI: &str = "https://oaipmh.arxiv.org/oai";
/// Tries a request gets on timeouts and connection errors, waiting 2**attempt s between them.
pub const TRIES: u32 = 6;
/// 429 and 503 replies a request waits out before the harvest fails, so a blocked or down endpoint ends the run.
pub const BUSY_TRIES: u32 = 60;
const BUSY_WAIT_S: f64 = 60.0;
/// The OAI-PMH error a range with no changes answers with on its first page.
pub const NO_RECORDS: &str = "noRecordsMatch";

/// `urllib.parse.urlencode`, which quotes with `quote_plus`.
pub fn urlencode(pairs: &[(&str, &str)]) -> String {
    let quote = |s: &str| {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    };
    pairs.iter().map(|(k, v)| format!("{}={}", quote(k), quote(v))).collect::<Vec<_>>().join("&")
}

fn fetch(client: &Client, url: &str) -> Result<Vec<u8>> {
    let (mut failures, mut busy) = (0, 0);
    loop {
        match client.get(url) {
            Ok(response) if (200..300).contains(&response.status) => return Ok(response.body),
            Ok(response) if response.status == 429 || response.status == 503 => {
                busy += 1;
                if busy >= BUSY_TRIES {
                    bail!("{url} answered {} {BUSY_TRIES} times", response.status);
                }
                let wait = response.retry_after.unwrap_or(BUSY_WAIT_S);
                sleep(Duration::try_from_secs_f64(wait).unwrap_or(Duration::from_secs_f64(BUSY_WAIT_S)));
            }
            Ok(response) => bail!("HTTP {} from {url}", response.status),
            Err(error) => {
                failures += 1;
                if failures >= TRIES {
                    return Err(error.context(format!("{TRIES} tries of {url}")));
                }
                sleep(Duration::from_secs(1 << (failures - 1)));
            }
        }
    }
}

/// The page's resumption token, None when it is missing or blank.
pub fn resumption_token(body: &[u8]) -> Result<Option<String>> {
    let doc = parse(body)?;
    let token = first_descendant(&doc, OAI_NS, "resumptionToken").map(text_of).map(desearch::text::strip).unwrap_or("");
    Ok((!token.is_empty()).then(|| token.to_string()))
}

/// The page's OAI-PMH error code and message, None when it has none.
pub fn oai_error(body: &[u8]) -> Result<Option<(String, String)>> {
    let doc = parse(body)?;
    Ok(first_descendant(&doc, OAI_NS, "error")
        .map(|error| (error.attribute("code").unwrap_or("").to_string(), desearch::text::strip(text_of(error)).to_string())))
}

/// `harvest` against any OAI-PMH endpoint, waiting `delay` between pages.
pub fn harvest_with(out: &Path, since: &str, until: &str, user_agent: &str, base_url: &str, delay: Duration) -> Result<usize> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let client = Client::new(user_agent)?;
    let mut query = urlencode(&[("verb", "ListRecords"), ("metadataPrefix", "arXiv"), ("from", since), ("until", until)]);
    let mut pages = 0;
    loop {
        let body = fetch(&client, &format!("{base_url}?{query}"))?;
        pages += 1;
        let path = out.join(format!("oai-{pages:06}.xml"));
        std::fs::write(&path, &body).with_context(|| format!("writing {}", path.display()))?;
        match oai_error(&body).with_context(|| format!("parsing {}", path.display()))? {
            Some((code, _)) if code == NO_RECORDS && pages == 1 => return Ok(pages),
            Some((code, message)) => bail!("OAI-PMH error {code} on page {pages} ({}): {message}", path.display()),
            None => {}
        }
        let Some(token) = resumption_token(&body).with_context(|| format!("parsing {}", path.display()))? else {
            return Ok(pages);
        };
        query = urlencode(&[("verb", "ListRecords"), ("resumptionToken", &token)]);
        sleep(delay);
    }
}

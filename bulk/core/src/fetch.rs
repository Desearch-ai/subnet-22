//! Downloads for the sources' fetch commands: a page as text, a file saved whole or not at all.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::http::USER_AGENT;

pub fn client() -> Result<reqwest::blocking::Client> {
    // Whole-file downloads run to several GB, so the limit is on the whole request, not a read.
    reqwest::blocking::Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(6 * 3600)).build().context("building the HTTP client")
}

/// A page's text, tried again on server errors and broken connections.
pub fn text(client: &reqwest::blocking::Client, url: &str) -> Result<String> {
    for attempt in 0..5u32 {
        match client.get(url).send() {
            Ok(r) if r.status().is_success() => return Ok(r.text()?),
            Ok(r) if attempt == 4 || r.status().is_client_error() => bail!("GET {url}: {}", r.status()),
            Err(e) if attempt == 4 => return Err(e).context(format!("GET {url}")),
            _ => std::thread::sleep(Duration::from_secs(10 * (attempt as u64 + 1))),
        }
    }
    unreachable!("the last attempt returns")
}

/// A file saved to `path` through a `.part` file, so a broken download never looks finished; false when the server has no such file.
pub fn save(client: &reqwest::blocking::Client, url: &str, path: &Path) -> Result<bool> {
    for attempt in 0..5u32 {
        match client.get(url).send() {
            Ok(mut r) if r.status().is_success() => {
                let part = path.with_extension("part");
                let mut file = std::fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;
                if r.copy_to(&mut file).is_ok() {
                    file.flush()?;
                    std::fs::rename(&part, path)?;
                    return Ok(true);
                }
            }
            Ok(r) if r.status().as_u16() == 404 || r.status().as_u16() == 410 => return Ok(false),
            Ok(r) if attempt == 4 || r.status().is_client_error() => bail!("GET {url}: {}", r.status()),
            Err(e) if attempt == 4 => return Err(e).context(format!("GET {url}")),
            _ => {}
        }
        std::thread::sleep(Duration::from_secs(10 * (attempt as u64 + 1)));
    }
    bail!("GET {url}: the download kept breaking")
}

#[derive(clap::Args, Clone, Debug)]
pub struct DownloadArgs {
    /// Where the file goes.
    #[arg(long)]
    pub to: std::path::PathBuf,
    /// The file's address.
    pub url: String,
}

/// `bulk download`: one file saved whole; a missing file fails.
pub fn download_main(args: DownloadArgs) -> Result<()> {
    if let Some(parent) = args.to.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !save(&client()?, &args.url, &args.to)? {
        bail!("{} is not there", args.url);
    }
    println!("{}", serde_json::json!({"saved": args.to.display().to_string()}));
    Ok(())
}

//! Snapshot lines from a .tar.gz of NDJSON files, or from one NDJSON file, gzipped or not.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};
use flate2::read::MultiGzDecoder;
use serde_json::Value;

/// Calls `each` with every non-blank line of the snapshot, newline included, in file order.
pub fn each_line(path: &Path, mut each: impl FnMut(Vec<u8>) -> Result<bool>) -> Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let mut bundle = tar::Archive::new(MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file)));
        for member in bundle.entries().with_context(|| format!("reading {}", path.display()))? {
            let member = member?;
            if !member.header().entry_type().is_file() {
                continue;
            }
            if !lines(BufReader::with_capacity(1 << 20, member), &mut each)? {
                return Ok(());
            }
        }
        return Ok(());
    }
    let reader: Box<dyn Read> =
        if path.extension().is_some_and(|e| e == "gz") { Box::new(MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file))) } else { Box::new(file) };
    lines(BufReader::with_capacity(1 << 20, reader), &mut each)?;
    Ok(())
}

fn lines(mut reader: impl BufRead, each: &mut impl FnMut(Vec<u8>) -> Result<bool>) -> Result<bool> {
    loop {
        let mut line = Vec::new();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(true);
        }
        if line.iter().all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c')) {
            continue;
        }
        if !each(line)? {
            return Ok(false);
        }
    }
}

/// `json.loads(line)`: a leading BOM allowed, a lone surrogate escape read as the `?` it becomes once encoded.
pub fn parse(line: &[u8]) -> Result<Value> {
    let line = line.strip_prefix(b"\xef\xbb\xbf").unwrap_or(line);
    match serde_json::from_slice(line) {
        Ok(value) => Ok(value),
        Err(error) => {
            let text = std::str::from_utf8(line).map_err(|_| error)?;
            Ok(serde_json::from_str(&bulk_core::text::replace_lone_surrogates(text, "?"))?)
        }
    }
}

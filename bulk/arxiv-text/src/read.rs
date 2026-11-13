//! Lines of a gzipped text file as Python's `gzip.open(path, "rt")` yields them: `\n`, `\r\n` and `\r` all end a line.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use flate2::read::MultiGzDecoder;

pub struct GzLines {
    reader: Option<BufReader<MultiGzDecoder<BufReader<File>>>>,
    queued: VecDeque<Vec<u8>>,
    chunk: Vec<u8>,
}

impl GzLines {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = BufReader::with_capacity(1 << 20, File::open(path)?);
        // Python's gzip reads an empty file as no lines; flate2 calls it a truncated stream.
        let reader = if file.fill_buf()?.is_empty() { None } else { Some(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(file))) };
        Ok(GzLines { reader, queued: VecDeque::new(), chunk: Vec::new() })
    }

    /// The next line without its line end; None at the end of the file.
    pub fn next_line(&mut self) -> io::Result<Option<Vec<u8>>> {
        let Some(reader) = self.reader.as_mut() else {
            return Ok(None);
        };
        while self.queued.is_empty() {
            self.chunk.clear();
            if reader.read_until(b'\n', &mut self.chunk)? == 0 {
                return Ok(None);
            }
            queue_lines(&self.chunk, &mut self.queued);
        }
        Ok(self.queued.pop_front())
    }

    /// Lines up to `most` of them or `bytes` in all, and the read error that ended them early, if any.
    pub fn batch(&mut self, most: usize, bytes: usize) -> (Vec<Vec<u8>>, Option<io::Error>) {
        let mut lines = Vec::new();
        let mut size = 0;
        while lines.len() < most && size < bytes {
            match self.next_line() {
                Ok(Some(line)) => {
                    size += line.len();
                    lines.push(line);
                }
                Ok(None) => break,
                Err(error) => return (lines, Some(error)),
            }
        }
        (lines, None)
    }
}

/// The lines of one `read_until(b'\n')` chunk, split again at each `\r` as universal newlines do.
fn queue_lines(chunk: &[u8], queued: &mut VecDeque<Vec<u8>>) {
    let (content, ended) = match chunk.strip_suffix(b"\n") {
        Some(content) => (content.strip_suffix(b"\r").unwrap_or(content), true),
        None => (chunk, false),
    };
    let mut pieces = content.split(|&b| b == b'\r').peekable();
    while let Some(piece) = pieces.next() {
        if pieces.peek().is_none() && !ended && piece.is_empty() {
            break;
        }
        queued.push_back(piece.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(chunk: &[u8]) -> Vec<Vec<u8>> {
        let mut queued = VecDeque::new();
        queue_lines(chunk, &mut queued);
        queued.into()
    }

    #[test]
    fn line_ends_follow_universal_newlines() {
        assert_eq!(lines(b"a\n"), vec![b"a".to_vec()]);
        assert_eq!(lines(b"a\r\n"), vec![b"a".to_vec()]);
        assert_eq!(lines(b"a\r\r\n"), vec![b"a".to_vec(), b"".to_vec()]);
        assert_eq!(lines(b"a\rb\n"), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(lines(b"\n"), vec![b"".to_vec()]);
        assert_eq!(lines(b"a\r"), vec![b"a".to_vec()]);
        assert_eq!(lines(b"a\rb"), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(lines(b"tail"), vec![b"tail".to_vec()]);
    }
}

//! Rows as Python's `csv.reader` reads them with the default dialect, from UTF-8 text.

use std::io::BufRead;

use anyhow::{Context, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    StartRecord,
    StartField,
    InField,
    InQuoted,
    QuoteInQuoted,
}

pub struct CsvRows<R> {
    source: R,
    buf: String,
    pos: usize,
    done: bool,
}

impl<R: BufRead> CsvRows<R> {
    pub fn new(source: R) -> Self {
        CsvRows { source, buf: String::new(), pos: 0, done: false }
    }

    fn peek(&mut self) -> Result<Option<char>> {
        if self.pos >= self.buf.len() {
            let mut line = Vec::new();
            if self.source.read_until(b'\n', &mut line)? == 0 {
                return Ok(None);
            }
            self.buf = String::from_utf8(line).context("CSV text that is not UTF-8")?;
            self.pos = 0;
        }
        Ok(self.buf[self.pos..].chars().next())
    }

    fn take(&mut self) -> Result<Option<char>> {
        let c = self.peek()?;
        if let Some(c) = c {
            self.pos += c.len_utf8();
        }
        Ok(c)
    }

    fn end_record(&mut self, c: char, mut fields: Vec<String>, field: Option<String>) -> Result<Option<Vec<String>>> {
        if c == '\r' && self.peek()? == Some('\n') {
            self.take()?;
        }
        fields.extend(field);
        Ok(Some(fields))
    }

    fn row(&mut self) -> Result<Option<Vec<String>>> {
        let mut fields = Vec::new();
        let mut field = String::new();
        let mut state = State::StartRecord;
        loop {
            let Some(c) = self.take()? else {
                self.done = true;
                if state == State::StartRecord {
                    return Ok(None);
                }
                fields.push(field);
                return Ok(Some(fields));
            };
            let newline = c == '\n' || c == '\r';
            match state {
                State::StartRecord if newline => return self.end_record(c, fields, None),
                State::StartField | State::InField | State::QuoteInQuoted if newline => return self.end_record(c, fields, Some(field)),
                State::StartRecord | State::StartField => match c {
                    '"' => state = State::InQuoted,
                    ',' => {
                        fields.push(std::mem::take(&mut field));
                        state = State::StartField;
                    }
                    _ => {
                        field.push(c);
                        state = State::InField;
                    }
                },
                State::InField => match c {
                    ',' => {
                        fields.push(std::mem::take(&mut field));
                        state = State::StartField;
                    }
                    _ => field.push(c),
                },
                State::InQuoted => match c {
                    '"' => state = State::QuoteInQuoted,
                    _ => field.push(c),
                },
                State::QuoteInQuoted => match c {
                    '"' => {
                        field.push('"');
                        state = State::InQuoted;
                    }
                    ',' => {
                        fields.push(std::mem::take(&mut field));
                        state = State::StartField;
                    }
                    _ => {
                        field.push(c);
                        state = State::InField;
                    }
                },
            }
        }
    }
}

impl<R: BufRead> Iterator for CsvRows<R> {
    type Item = Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        self.row().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(text: &str) -> Vec<Vec<String>> {
        CsvRows::new(text.as_bytes()).map(|r| r.unwrap()).collect()
    }

    #[test]
    fn rows_split_as_python_splits_them() {
        assert_eq!(rows("\"a\",\"b,c\"\n\"x\"\"y\",\n"), [vec!["a", "b,c"], vec!["x\"y", ""]]);
        assert_eq!(rows("a,b\r\n\nc"), [vec!["a", "b"], vec![], vec!["c"]]);
        assert_eq!(rows("a\rb,\n"), [vec!["a"], vec!["b", ""]]);
        assert_eq!(rows("a,"), [vec!["a", ""]]);
        assert_eq!(rows("\"multi\nline\",z\n"), [vec!["multi\nline", "z"]]);
    }
}

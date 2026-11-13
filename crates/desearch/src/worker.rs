//! The shared extraction's Python worker, `python -m desearch.extraction.worker`: one length-framed request at a time.

use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

pub const MODULE: &str = "desearch.extraction.worker";
/// The module's file under a repo root.
pub const MODULE_FILE: &str = "desearch/extraction/worker.py";
/// The largest answer a worker may send; a longer length means the framing broke.
const MAX_ANSWER: usize = 1 << 30;

/// A page: the URL's length as 4 big-endian bytes, the URL, then the HTML.
pub const EXTRACT: &[u8] = b"e";
/// A Wikipedia snapshot article as JSON, framed as the live site frames it.
pub const WIKIPEDIA: &[u8] = b"w";
/// `looks_blocked` on JSON `status`, `html`, `text` and `title`.
pub const BLOCKED: &[u8] = b"k";
/// Body HTML rendered to lines.
pub const BODY: &[u8] = b"r";
/// The `a` hrefs as lxml parses a page.
pub const ANCHORS: &[u8] = b"a";
/// A crawled page as desearch-bot builds it, from JSON `url`, `html`, `orig` and `whole`: EDGAR documents rewritten, blocks checked.
pub const CRAWLED: &[u8] = b"c";
/// A PDF, table or data file from JSON `url`, `body_b64`, `document` and `content_type`.
pub const DOCUMENT: &[u8] = b"d";
/// A PMC article's JATS XML as rendered abstract and body lines.
pub const PMC: &[u8] = b"p";

/// Where a worker's code comes from.
pub enum Program<'a> {
    /// `MODULE` on the Python path.
    Module,
    /// The module's source, run with `python -c`.
    Source(&'a str),
    /// A script that answers the same requests.
    Script(&'a Path),
}

/// The nearest folder above the binary holding `MODULE_FILE`, else the current directory.
pub fn default_python_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().find(|dir| dir.join(MODULE_FILE).is_file()).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The command that starts a worker, with `python_path` first on its PYTHONPATH.
pub fn command(python: &str, python_path: &Path, program: Program, args: &[OsString]) -> Command {
    let mut path = OsString::from(python_path);
    if let Some(existing) = std::env::var_os("PYTHONPATH").filter(|p| !p.is_empty()) {
        path.push(":");
        path.push(existing);
    }
    let mut command = Command::new(python);
    match program {
        Program::Module => command.args(["-m", MODULE]),
        Program::Source(source) => command.arg("-c").arg(source),
        Program::Script(script) => command.arg(script),
    };
    // Keeps the working directory off sys.path, so its own `desearch`, if any, cannot shadow python_path's.
    command.args(args).env("PYTHONPATH", path).env("PYTHONSAFEPATH", "1");
    command
}

pub struct Worker {
    name: String,
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    /// `python -m desearch.extraction.worker` with `args`.
    pub fn spawn(python: &str, python_path: &Path, args: &[OsString]) -> Result<Self> {
        Self::start(command(python, python_path, Program::Module, args), &format!("{python} -m {MODULE}"))
    }

    /// A worker from a `command()` the caller may have changed further, named `name` in errors.
    pub fn start(mut command: Command, name: &str) -> Result<Self> {
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().with_context(|| format!("starting {name}"))?;
        let stdin = child.stdin.take().map(BufWriter::new);
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no worker stdout"))?);
        Ok(Worker { name: name.to_string(), child, stdin, stdout })
    }

    /// One answer; on any failure the worker is killed, since its framing can no longer be trusted.
    pub fn ask(&mut self, parts: &[&[u8]]) -> Result<Vec<u8>> {
        let answer = self.exchange(parts);
        if answer.is_err() {
            let _ = self.child.kill();
        }
        answer.with_context(|| format!("{} stopped; check the Python and the path it imports desearch from", self.name))
    }

    /// One JSON answer; a worker's `{"error": ...}` answer is an error.
    pub fn ask_json(&mut self, parts: &[&[u8]]) -> Result<Value> {
        let answer: Value = serde_json::from_slice(&self.ask(parts)?).with_context(|| format!("an answer from {}", self.name))?;
        if let Some(error) = answer.get("error") {
            bail!("{} failed: {error}\n{}", self.name, answer.get("detail").and_then(Value::as_str).unwrap_or(""));
        }
        Ok(answer)
    }

    fn exchange(&mut self, parts: &[&[u8]]) -> Result<Vec<u8>> {
        let size = u32::try_from(parts.iter().map(|p| p.len()).sum::<usize>()).context("a request over 4 GiB")?;
        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("worker closed"))?;
        stdin.write_all(&size.to_be_bytes())?;
        for part in parts {
            stdin.write_all(part)?;
        }
        stdin.flush()?;
        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header)?;
        let size = u32::from_be_bytes(header) as usize;
        if size > MAX_ANSWER {
            bail!("a {size}-byte answer");
        }
        let mut data = vec![0u8; size];
        self.stdout.read_exact(&mut data)?;
        Ok(data)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ECHO: &str = r#"
import struct, sys
src, out = sys.stdin.buffer, sys.stdout.buffer
while header := src.read(4):
    data = src.read(struct.unpack(">I", header)[0])
    if data == b"quit":
        break
    answer = b'{"error": "crash", "detail": "boom"}' if data == b"fail" else b'{"got": "' + data + b'"}'
    out.write(struct.pack(">I", len(answer)) + answer)
    out.flush()
"#;

    fn echo() -> Option<Worker> {
        if Command::new("python3").arg("-c").arg("1").status().is_err() {
            eprintln!("no python3; skipped");
            return None;
        }
        Some(Worker::start(command("python3", Path::new("."), Program::Source(ECHO), &[]), "echo").unwrap())
    }

    #[test]
    fn requests_go_out_framed_and_answers_come_back_whole() {
        let Some(mut worker) = echo() else { return };
        assert_eq!(worker.ask(&[b"a", b"bc"]).unwrap(), br#"{"got": "abc"}"#);
        assert_eq!(worker.ask_json(&[b"x".repeat(100_000).as_slice()]).unwrap()["got"].as_str().unwrap().len(), 100_000);
        let failed = worker.ask_json(&[b"fail"]).unwrap_err().to_string();
        assert!(failed.starts_with("echo failed: \"crash\"\nboom"), "{failed}");
        let stopped = format!("{:#}", worker.ask(&[b"quit"]).unwrap_err());
        assert!(stopped.starts_with("echo stopped"), "{stopped}");
        assert!(worker.ask(&[b"a"]).is_err(), "a stopped worker stays stopped");
    }

    #[test]
    fn the_command_puts_the_path_first() {
        let command = command("python3", Path::new("/repo"), Program::Module, &["--hidden".into(), "h.txt".into()]);
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(args, ["-m", MODULE, "--hidden", "h.txt"]);
        let env = |key: &str| command.get_envs().find(|(k, _)| *k == key).and_then(|(_, v)| v).map(|v| v.to_string_lossy().into_owned());
        let path = env("PYTHONPATH").unwrap();
        assert!(path == "/repo" || path.starts_with("/repo:"), "{path}");
        assert_eq!(env("PYTHONSAFEPATH").as_deref(), Some("1"));
    }
}

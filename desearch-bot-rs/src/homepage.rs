//! Enough of a homepage to tell whether it is readable English.

use std::sync::LazyLock;

use regex::Regex;

use crate::text;

const MIN_CHARS: usize = 200;
const SAMPLE: usize = 2000;

static HTML_LANG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)<html[^>]*\blang=["']?([a-zA-Z-]{2,8})"#).unwrap());
static SCRIPT_OR_STYLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>|<svg[^>]*>.*?</svg>").unwrap()
});
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static CHARSET: LazyLock<regex::bytes::Regex> =
    LazyLock::new(|| regex::bytes::Regex::new(r#"(?i-u)charset=["']?([\w-]+)"#).unwrap());
static BOT_WALL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "(?i)(please enable javascript|enable javascript and refresh|you need to enable javascript\
         |access denied|are you a robot|checking your browser|attention required)",
    )
    .unwrap()
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Homepage {
    pub chars: usize,
    pub declared: Option<String>,
    pub language: Option<&'static str>,
    pub problem: Option<&'static str>,
}

/// Judge a homepage; the problem is None when it is readable English.
pub fn read(body: &[u8], detect_language: fn(&str) -> Option<&'static str>) -> Homepage {
    let html = decode(body);
    let declared = HTML_LANG
        .captures(text::head(&html, 4000))
        .map(|c| c[1].to_lowercase().split('-').next().unwrap_or("").to_string());
    let stripped = TAG.replace_all(&SCRIPT_OR_STYLE.replace_all(&html, " "), " ").into_owned();
    let words: Vec<&str> = text::words(&stripped).collect();
    let page = words.join(" ");
    let chars = text::char_count(&page);
    let sample = text::head(&page, SAMPLE);
    let judged = |language, problem| Homepage { chars, declared: declared.clone(), language, problem };
    if BOT_WALL.is_match(sample) {
        return judged(None, Some("bot_wall"));
    }
    if chars < MIN_CHARS {
        return judged(None, Some("no_text"));
    }
    let language = detect_language(sample);
    judged(language, if language == Some("en") { None } else { Some("not_english") })
}

/// The body as text in its declared charset, falling back to UTF-8.
pub fn decode(body: &[u8]) -> String {
    let label = CHARSET.captures(&body[..body.len().min(4096)]).map(|c| c[1].to_vec());
    let encoding = label
        .and_then(|label| encoding_rs::Encoding::for_label(&label).or_else(|| python_alias(&label)))
        .unwrap_or(encoding_rs::UTF_8);
    encoding.decode_without_bom_handling(body).0.into_owned()
}

fn python_alias(label: &[u8]) -> Option<&'static encoding_rs::Encoding> {
    let label = String::from_utf8_lossy(label).to_ascii_lowercase().replace('_', "-");
    match label.as_str() {
        "latin-1" | "iso8859-1" | "l1" => Some(encoding_rs::WINDOWS_1252),
        "utf8" | "utf-8" | "u8" => Some(encoding_rs::UTF_8),
        other => encoding_rs::Encoding::for_label(other.as_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn english(_: &str) -> Option<&'static str> {
        Some("en")
    }

    #[test]
    fn judges_text_and_walls() {
        let body = format!("<html lang=\"en-GB\"><script>var x = '<b>';</script><p>{}</p></html>", "word ".repeat(60));
        let page = read(body.as_bytes(), english);
        assert_eq!((page.declared.as_deref(), page.problem, page.chars), (Some("en"), None, 299));
        assert_eq!(read(b"<p>Please enable JavaScript</p>", english).problem, Some("bot_wall"));
        assert_eq!(read(b"<p>short</p>", english).problem, Some("no_text"));
    }
}

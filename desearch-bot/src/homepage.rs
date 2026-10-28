//! Enough of a homepage to tell whether it is readable English.

use std::sync::LazyLock;

use regex::Regex;

use crate::text;

const MIN_CHARS: usize = 200;
const SAMPLE: usize = 2000;
/// The language model reads this much: headline-heavy homepages need more than the bot-wall check does.
const LANGUAGE_SAMPLE: usize = 8000;

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
    pub language: Option<String>,
    pub problem: Option<&'static str>,
}

/// Judge a homepage; the problem is None when it is readable English.
pub fn read(body: &[u8], detect_language: &dyn Fn(&str) -> Option<String>) -> Homepage {
    let html = decode(body);
    let declared = HTML_LANG
        .captures(text::head(&html, 4000))
        .map(|c| c[1].to_lowercase().split('-').next().unwrap_or("").to_string());
    let stripped = TAG.replace_all(&SCRIPT_OR_STYLE.replace_all(&html, " "), " ").into_owned();
    let words: Vec<&str> = text::words(&stripped).collect();
    let page = words.join(" ");
    let chars = text::char_count(&page);
    let sample = text::head(&page, SAMPLE);
    let judged = |language: Option<String>, problem| Homepage { chars, declared: declared.clone(), language, problem };
    if BOT_WALL.is_match(sample) {
        return judged(None, Some("bot_wall"));
    }
    if chars < MIN_CHARS {
        return judged(None, Some("no_text"));
    }
    let language = detect_language(text::head(&page, LANGUAGE_SAMPLE));
    // A page that says it is English is believed only where the model gives no verdict: an English template over another language stays out.
    let english = language.as_deref() == Some("en") || (declared.as_deref() == Some("en") && language.is_none());
    judged(language, if english { None } else { Some("not_english") })
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

    fn english(_: &str) -> Option<String> {
        Some("en".into())
    }

    #[test]
    fn judges_text_and_walls() {
        let body = format!("<html lang=\"en-GB\"><script>var x = '<b>';</script><p>{}</p></html>", "word ".repeat(60));
        let page = read(body.as_bytes(), &english);
        assert_eq!((page.declared.as_deref(), page.problem, page.chars), (Some("en"), None, 299));
        assert_eq!(read(b"<p>Please enable JavaScript</p>", &english).problem, Some("bot_wall"));
        assert_eq!(read(b"<p>short</p>", &english).problem, Some("no_text"));
    }

    #[test]
    fn a_declared_english_page_passes_only_where_the_model_gives_no_verdict_and_the_model_reads_a_longer_sample() {
        let french = |_: &str| Some("fr".to_string());
        let headlines = format!("<p>{}</p>", "Headline ".repeat(500));
        let declared = format!("<html lang=\"en-US\">{headlines}</html>");
        assert_eq!(read(declared.as_bytes(), &french).problem, Some("not_english"), "an English template on a page the model reads as French throughout");
        let short_english = |sample: &str| Some(if sample.len() <= SAMPLE { "en" } else { "es" }.to_string());
        assert_eq!(read(declared.as_bytes(), &short_english).problem, Some("not_english"), "English menus over a Spanish page, under lang=en, stay out: the longer sample decides");
        let unsure = |_: &str| None;
        assert_eq!(read(declared.as_bytes(), &unsure).problem, None, "lang=en settles a page the model gives no verdict on");
        assert_eq!(read(headlines.as_bytes(), &unsure).problem, Some("not_english"), "without lang=en a page with no verdict stays out");
        assert_eq!(read(headlines.as_bytes(), &french).problem, Some("not_english"));
        let long_only = |sample: &str| Some(if sample.len() > SAMPLE { "en" } else { "fr" }.to_string());
        assert_eq!(read(headlines.as_bytes(), &long_only).problem, None, "the model sees more than the first 2,000 characters");
        let walled = format!("<p>{}Access denied</p>", "word ".repeat(1000));
        assert_eq!(read(walled.as_bytes(), &english).problem, None, "a wall phrase past the first 2,000 characters is page text");
    }
}

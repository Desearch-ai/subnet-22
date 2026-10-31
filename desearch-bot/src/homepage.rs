//! Enough of a homepage to tell whether it is readable English.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use crate::{sitemaps, text};

const MIN_CHARS: usize = 200;
const SAMPLE: usize = 2000;
/// The language model reads this much: headline-heavy homepages need more than the bot-wall check does.
const LANGUAGE_SAMPLE: usize = 8000;
/// A challenge page's markers sit in its markup, near the start.
const CHALLENGE_SAMPLE: usize = 20_000;
/// The descriptions sit in the head.
const HEAD_SAMPLE: usize = 65_536;
/// A wall's own page is short; a longer page that says "enable JavaScript" or "access denied" is a page with a video, a form or a heading.
const WALL_MAX_CHARS: usize = 1500;
/// Written languages with a real share of the web. A page that declares English and reads as another of these is not English; one read as anything else (no language, Latin, a creole) is judged by its declaration.
const MAJOR: [&str; 52] = [
    "en", "zh", "es", "ar", "pt", "id", "ms", "fr", "ja", "ru", "de", "ko", "tr", "fa", "it", "vi", "pl", "nl", "th", "uk", "ro", "el", "cs", "hu", "sv", "he", "da",
    "fi", "no", "nb", "sk", "bg", "hr", "sr", "lt", "sl", "lv", "et", "hi", "bn", "ur", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ca", "az", "kk", "ka",
];

static HTML_LANG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)<html[^>]*\blang=["']?([a-zA-Z-]{2,8})"#).unwrap());
/// A body cut at the read limit can end inside a script, which then runs to its end.
static SCRIPT_OR_STYLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)<script[^>]*>.*?(?:</script>|\z)|<style[^>]*>.*?(?:</style>|\z)|<noscript[^>]*>.*?(?:</noscript>|\z)|<svg[^>]*>.*?(?:</svg>|\z)").unwrap()
});
/// Pages that check the browser before showing the site: Cloudflare, DataDome, PerimeterX, Imperva and Akamai.
static CHALLENGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<title>just a moment|cf-chl-|captcha-delivery\.com|px-captcha|_incapsula_resource|verify you are human|bm-verify=").unwrap()
});
static META: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<meta\s[^>]*>").unwrap());
static SUMMARY_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)\b(?:name|property)\s*=\s*["']?(?:description|og:description|twitter:description)["'\s/>]"#).unwrap());
static CONTENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?is)\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')"#).unwrap());
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?is)<a\s(?:[^>]*?\s)?href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"#).unwrap());
/// Links one homepage lists at most, in page order: fewer than a sitemap the churn check would judge.
const MAX_LINKS: usize = 400;
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
    if (chars < WALL_MAX_CHARS && BOT_WALL.is_match(sample)) || CHALLENGE.is_match(text::head(&html, CHALLENGE_SAMPLE)) {
        return judged(None, Some("bot_wall"));
    }
    // A page that says it is English is believed only where the model gives no verdict or a minor one: an English template over another language stays out.
    let english = |language: &Option<String>| {
        language.as_deref() == Some("en") || (declared.as_deref() == Some("en") && language.as_deref().is_none_or(|language| !MAJOR.contains(&language)))
    };
    if chars < MIN_CHARS {
        // The check runs on sites whose sitemaps list pages, so a short homepage is a menu or an app shell: its title and description tell its language.
        let summary = format!("{page} {}", descriptions(&html).join(" "));
        let language = detect_language(text::head(&summary, LANGUAGE_SAMPLE));
        let problem = if english(&language) {
            None
        } else if language.is_none() {
            Some("no_text")
        } else {
            Some("not_english")
        };
        return judged(language, problem);
    }
    let language = detect_language(text::head(&page, LANGUAGE_SAMPLE));
    let problem = if english(&language) { None } else { Some("not_english") };
    judged(language, problem)
}

/// The addresses a homepage links to, in page order, each once and with character references decoded; fragments and script, mail, phone and data links are none.
pub fn links(body: &[u8]) -> Vec<String> {
    let html = decode(body);
    let html = SCRIPT_OR_STYLE.replace_all(&html, " ");
    let mut seen = HashSet::new();
    LINK.captures_iter(&html)
        .filter_map(|found| found.get(1).or_else(|| found.get(2)).or_else(|| found.get(3)))
        .map(|href| sitemaps::location(href.as_str().trim().as_bytes()))
        .filter(|href| {
            let scheme = href.split_once(':').map(|(scheme, _)| scheme.to_ascii_lowercase());
            !href.is_empty() && !href.starts_with('#') && !matches!(scheme.as_deref(), Some("javascript" | "mailto" | "tel" | "data"))
        })
        .filter(|href| seen.insert(href.clone()))
        .take(MAX_LINKS)
        .collect()
}

/// The page's own summaries: its description, Open Graph and Twitter descriptions, each once.
fn descriptions(html: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for tag in META.find_iter(text::head(html, HEAD_SAMPLE)).map(|m| m.as_str()).filter(|tag| SUMMARY_NAME.is_match(tag)) {
        let content = CONTENT.captures(tag).and_then(|c| c.get(1).or_else(|| c.get(2))).map(|m| sitemaps::location(m.as_str().trim().as_bytes()));
        if let Some(content) = content.filter(|c| !c.is_empty() && !found.contains(c)) {
            found.push(content);
        }
    }
    found
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

    /// As the bot's model: no verdict on under 20 characters.
    fn english(text: &str) -> Option<String> {
        text.chars().nth(19).map(|_| "en".into())
    }

    #[test]
    fn a_homepage_lists_the_addresses_it_links_to_once_each_in_page_order() {
        let body = br##"<nav><a href="/about">About</a> <A class=x HREF='/en-us/products?a=1&amp;b=2'>P</A> <a data-href="/no" href=/pricing>$</a>
            <a href="#top">Top</a><a href="mailto:x@a.com">M</a><a href=" JavaScript:void(0)">J</a><a href="/about">again</a>
            <script>var a = '<a href="/scripted">';</script><a name="anchor">none</a><a href="https://other.com/">o</a></nav>"##;
        assert_eq!(links(body), ["/about", "/en-us/products?a=1&b=2", "/pricing", "https://other.com/"]);
        let many: String = (0..500).map(|i| format!("<a href=\"/p{i}\">{i}</a>")).collect();
        assert_eq!(links(many.as_bytes()).len(), MAX_LINKS);
    }

    #[test]
    fn a_body_cut_inside_a_script_reads_only_the_text_before_it() {
        let body = format!("<html lang=\"en\"><p>{}</p><script>window.__DATA__ = {{\"items\": [{}", "word ".repeat(60), "\"x1y2\", ".repeat(5000));
        let code = |text: &str| Some(if text.contains("x1y2") { "zxx" } else { "en" }.to_string());
        let page = read(body.as_bytes(), &code);
        assert_eq!((page.problem, page.chars), (None, 299), "the cut script is no text");
    }

    #[test]
    fn a_short_homepage_is_judged_by_its_title_and_description() {
        let shell = br#"<html lang="en"><head><title>Sportsnet</title><meta name="description" content="Canada&#39;s leading sports network, with scores and news"></head><body><div id="root"></div><script>render()</script></body></html>"#;
        let page = read(shell, &english);
        assert_eq!((page.problem, page.chars), (None, 9), "an app shell whose summary reads as English");
        let menu = br#"<html><title>The Programming Language Lua</title><meta content='Official website of the Lua language' name=description><a href=about.html>about</a></html>"#;
        assert_eq!(read(menu, &english).problem, None, "a menu with no lang attribute, read by its summary");
        let french = |text: &str| text.chars().nth(19).map(|_| "fr".to_string());
        assert_eq!(read(shell, &french).problem, Some("not_english"), "a summary in another language");
        assert_eq!(read(b"", &english).problem, Some("no_text"), "nothing to judge");
        assert_eq!(read(b"<html><title>Home</title></html>", &english).problem, Some("no_text"));
    }

    #[test]
    fn a_wall_phrase_on_a_long_page_is_page_text() {
        let video = format!("<p>{}</p><p>To view this video please enable JavaScript, and consider upgrading</p><p>{}</p>", "word ".repeat(20), "word ".repeat(400));
        assert_eq!(read(video.as_bytes(), &english).problem, None);
        assert_eq!(read(b"<h1>Access denied</h1><p>You don't have permission to access this server.</p>", &english).problem, Some("bot_wall"));
    }

    #[test]
    fn a_declared_english_page_read_as_a_minor_class_is_english() {
        let page = format!("<html lang=\"en-US\"><p>{}</p></html>", "word ".repeat(60));
        let minor = |_: &str| Some("zxx".to_string());
        assert_eq!(read(page.as_bytes(), &minor).problem, None, "no language, Latin or a creole yields to the declaration");
        let gujarati = |_: &str| Some("gu".to_string());
        assert_eq!(read(page.as_bytes(), &gujarati).problem, Some("not_english"), "a major language does not");
        let undeclared = format!("<p>{}</p>", "word ".repeat(60));
        assert_eq!(read(undeclared.as_bytes(), &minor).problem, Some("not_english"), "with no declaration the model decides");
    }

    #[test]
    fn a_page_that_checks_the_browser_is_a_wall() {
        let akamai = br#"<html><head><title>&nbsp;</title><meta http-equiv="refresh" content="5; URL='/?bm-verify=AAQAAAAO'"></head><body>Powered and protected by <a href="https://www.akamai.com/privacy">Privacy</a></body></html>"#;
        assert_eq!(read(akamai, &english).problem, Some("bot_wall"));
        let cloudflare = format!("<html><head><title>Just a moment...</title></head><body><p>{}</p></body></html>", "word ".repeat(60));
        assert_eq!(read(cloudflare.as_bytes(), &english).problem, Some("bot_wall"));
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

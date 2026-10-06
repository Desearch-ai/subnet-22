//! `desearch.extraction.looks_blocked` as the publisher calls it: on a row's text and title, without its HTML.

use std::sync::LazyLock;

use regex::Regex;

use crate::text;

const REAL_CONTENT_CHARS: usize = 3000;
const TITLE_PREFIX_TEXT_CHARS: usize = 500;
const CHALLENGE_PHRASE: &str = r"\s*(?:just a moment|one moment|attention required|access denied|access to this page has been denied|security verification|security check|human verification|bot verification|are you a (?:human|robot)|verify(?:ing)? (?:that )?you are (?:a )?human|robot check|pardon our interruption|checking your browser|one more step|request (?:blocked|rejected|unsuccessful)|you have been blocked|ddos-guard|vercel security checkpoint|captcha|403 forbidden|forbidden|error 1020)";
const CHALLENGE_PHRASES: [&str; 17] = [
    "verify you are human",
    "verify you are a human",
    "verifies you are not a bot",
    "are you a robot",
    "not a robot",
    "enable javascript and cookies to continue",
    "checking your browser",
    "checking if the site connection is secure",
    "request is being verified",
    "unusual traffic",
    "you have been blocked",
    "security service to protect",
    "complete the security check",
    "request unsuccessful",
    "press & hold",
    "press and hold",
    "access to this page has been denied",
];
const WEAK_PHRASES: [&str; 4] = ["captcha", "access denied", "forbidden", "too many requests"];
const SCRIPT_ONLY_PHRASES: [&str; 5] = [
    "enable javascript",
    "javascript is disabled",
    "javascript is required",
    "requires javascript",
    "turn on javascript",
];
const REFUSAL_STATUSES: [i32; 4] = [401, 403, 407, 429];

static CHALLENGE_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"(?i)^{CHALLENGE_PHRASE}[\s.!?\u{{2026}}]*(?:$|(?P<suffix>(?:\||\s[-\u{{2013}}\u{{2014}}\u{{b7}}]\s).*))")).unwrap()
});
static CHALLENGE_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!("(?i)^{CHALLENGE_PHRASE}")).unwrap());
static VENDOR_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:cloudflare|sucuri|incapsula|imperva|ddos-guard|vercel|akamai|perimeterx|human security|datadome|kasada)")
        .unwrap()
});

pub fn looks_blocked(status: Option<i32>, text: &str, title: &str) -> bool {
    // Lowercasing never shortens text, so a long text is real content before it is lowercased.
    if text::collapsed_len(text, REAL_CONTENT_CHARS) > REAL_CONTENT_CHARS {
        return false;
    }
    let visible = text::collapse(text).to_lowercase();
    let length = visible.chars().count();
    if length > REAL_CONTENT_CHARS {
        return false;
    }
    let refusal = status.is_some_and(|s| REFUSAL_STATUSES.contains(&s));
    let refused = refusal || status == Some(503);
    let title = text::collapse(title);
    let title = text::fold_turkish_i(&title);
    if let Some(challenge) = CHALLENGE_TITLE.captures(&title) {
        if challenge.name("suffix").is_none_or(|suffix| vendor_named(suffix.as_str())) {
            return true;
        }
    }
    if let Some(prefix) = CHALLENGE_PREFIX.find(&title) {
        let bounded = title[prefix.end()..].chars().next().is_none_or(|c| !text::is_word(c));
        if bounded && (refused || length < TITLE_PREFIX_TEXT_CHARS) {
            return true;
        }
    }
    let has = |phrases: &[&str]| phrases.iter().any(|phrase| visible.contains(phrase));
    if has(&CHALLENGE_PHRASES) && length < 1000 {
        return true;
    }
    if has(&SCRIPT_ONLY_PHRASES) && length < 200 {
        return true;
    }
    if has(&WEAK_PHRASES) && length < 200 {
        return refused || length < 80;
    }
    refusal && length < 200
}

/// Python's `VENDOR_NAME.search`, with its `\b` on Python's word characters.
fn vendor_named(suffix: &str) -> bool {
    let mut previous = None;
    for (i, c) in suffix.char_indices() {
        if previous.is_none_or(|p| !text::is_word(p)) {
            if let Some(found) = VENDOR_NAME.find(&suffix[i..]) {
                if suffix[i + found.end()..].chars().next().is_none_or(|c| !text::is_word(c)) {
                    return true;
                }
            }
        }
        previous = Some(c);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_boundaries_follow_python() {
        let long = "x".repeat(600);
        assert!(!looks_blocked(Some(503), &long, "Forbiddenfruit"));
        assert!(looks_blocked(Some(503), &long, "Captcha\u{301}"));
        assert!(!looks_blocked(Some(503), &long, "Captcha\u{b2}"));
        assert!(!looks_blocked(Some(200), &long, "Just a moment - Cloudflare\u{b2}"));
        assert!(looks_blocked(Some(200), &long, "Just a moment - Cloudflare"));
    }
}

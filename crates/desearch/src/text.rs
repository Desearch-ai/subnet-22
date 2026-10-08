//! Python's notion of whitespace, word characters and case, so pages are judged as the Python publisher judged them.

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

/// Python's `str.isspace`: Unicode whitespace plus the four ASCII information separators.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

pub fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|word| !word.is_empty())
}

/// Python's `" ".join(s.split())`.
pub fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for word in words(s) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// Characters in `" ".join(s.split())`, counted no further than `limit + 1`.
pub fn collapsed_len(s: &str, limit: usize) -> usize {
    let mut count = 0;
    for word in words(s) {
        count += usize::from(count > 0) + word.chars().count();
        if count > limit {
            break;
        }
    }
    count
}

/// Python's `\w` on text: letters, digits and other numbers, and the underscore.
pub fn is_word(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    matches!(c.general_category_group(), GeneralCategoryGroup::Letter | GeneralCategoryGroup::Number)
}

/// Dotted and dotless I as a plain i: Python's case-insensitive regexes match them to i, Rust's do not.
pub fn fold_turkish_i(s: &str) -> std::borrow::Cow<'_, str> {
    if s.contains(['\u{130}', '\u{131}']) {
        s.replace(['\u{130}', '\u{131}'], "i").into()
    } else {
        s.into()
    }
}

/// Python's case-insensitive comparison of one character with an ASCII letter or symbol.
pub fn fold(c: char) -> char {
    match c {
        'A'..='Z' => c.to_ascii_lowercase(),
        '\u{130}' | '\u{131}' => 'i',
        '\u{212a}' => 'k',
        '\u{17f}' => 's',
        _ => c,
    }
}

/// The first `n` characters, as Python's `s[:n]` counts them.
pub fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_matches_python() {
        assert_eq!(collapse("\u{1f} a \u{a0}\tb\u{3000}"), "a b");
        assert_eq!(collapsed_len("  ab  cd ", 100), 5);
        assert_eq!(strip("\u{1c}x\u{85}"), "x");
    }

    #[test]
    fn word_characters_match_python() {
        assert!(is_word('é') && is_word('²') && is_word('_') && is_word('٣'));
        assert!(!is_word('\u{301}') && !is_word('\u{200d}') && !is_word('-'));
    }
}

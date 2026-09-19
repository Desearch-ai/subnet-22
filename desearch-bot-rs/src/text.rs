//! Python's notion of whitespace, lines and slices, so text splits the way the Python crawler split it.

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

/// Python's `str.splitlines`, which also breaks on form feeds, separators and U+2028/2029.
pub fn lines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if matches!(
            c,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
        ) {
            lines.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// The first `n` characters, as Python's `s[:n]` counts them.
pub fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub fn char_count(s: &str) -> usize {
    s.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_match_python() {
        assert_eq!(lines("a\r\nb\rc\n\nd\u{0c}e\n"), ["a", "b", "c", "", "d", "e"]);
        assert!(lines("").is_empty());
    }

    #[test]
    fn strip_matches_python() {
        assert_eq!(strip("\u{1f} a b\u{3000}"), "a b");
        assert_eq!(words(" a \u{a0} b ").collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(head("héllo", 2), "hé");
    }
}

//! Language and locale checks of the Python crawler's `scope` module; the set's pre-fetch admit is `SetRule::may_admit`.

use std::sync::LazyLock;

use regex::Regex;

use super::links::Doc;
use crate::pystr;
use crate::setrule::LANGS;
use crate::urls;

static LOCALE_LEVEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)^[a-z]{2,3}(?:[-_][a-z]{2,4})?$").unwrap());
static ENGLISH_LOCALE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)^en(?:[-_][a-z]{2})?$").unwrap());

/// The language the page's `<html>` names, or "" when it names none.
pub fn declared_lang(doc: &Doc) -> String {
    pystr::strip(doc.lang.as_deref().unwrap_or("")).to_lowercase()
}

/// A language the index keeps, read as `page_eligible` reads it.
pub fn wanted_lang(lang: &str) -> bool {
    LANGS.iter().any(|w| lang == *w || lang.starts_with(&format!("{w}-")) || lang.starts_with(&format!("{w}_")))
}

/// The URL's first path level when it reads as a locale other than English (`fr`, `fra`).
pub fn locale_level(url: &str) -> Option<String> {
    let path = urls::split(url).ok()?.path;
    let first = path.split('/').find(|level| !level.is_empty())?;
    (LOCALE_LEVEL.is_match(first) && !ENGLISH_LOCALE.is_match(first)).then(|| first.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locales_and_languages() {
        assert_eq!(locale_level("https://a.test/fr/x").as_deref(), Some("fr"));
        assert_eq!(locale_level("https://a.test/FR-ca/x").as_deref(), Some("fr-ca"));
        assert_eq!(locale_level("https://a.test//deu/x").as_deref(), Some("deu"));
        assert_eq!(locale_level("https://a.test/en-GB/x"), None);
        assert_eq!(locale_level("https://a.test/docs/x"), None);
        assert_eq!(locale_level("https://a.test/"), None);
        assert!(wanted_lang("en") && wanted_lang("en-us") && wanted_lang("en_gb") && !wanted_lang("eng") && !wanted_lang("fr"));
    }
}

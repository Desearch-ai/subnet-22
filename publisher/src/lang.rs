//! A page's language when its extraction found none: English by its text, read as the indexer reads it (`tests/lang-golden.json`).

use std::collections::HashSet;
use std::sync::LazyLock;

/// Characters of a text's start that are read.
pub const SAMPLE_CHARS: usize = 4000;
pub const MIN_WORDS: usize = 30;
/// English function words are at least this many in a hundred words, and more than twice other languages' function words.
pub const ENGLISH_PERCENT: usize = 8;
pub const OTHER_TIMES: usize = 2;
/// Characters past ASCII are at most this many tenths of the ASCII letters.
pub const NON_ASCII_TENTHS: usize = 1;
pub const ENGLISH: [&str; 38] = [
    "about", "all", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "for", "from", "has", "have", "in", "is", "it", "not", "of",
    "on", "or", "our", "that", "the", "their", "they", "this", "to", "was", "we", "were", "which", "will", "with", "you", "your",
];
pub const OTHER: [&str; 133] = [
    "ada", "af", "akan", "alla", "anche", "ang", "att", "au", "aux", "ay", "bei", "bij", "bir", "bu", "ce", "ces", "che", "como", "con", "da",
    "dalla", "das", "dat", "de", "degli", "del", "della", "delle", "dem", "den", "dengan", "der", "des", "det", "deze", "di", "dos", "du", "een",
    "ein", "eine", "el", "elle", "em", "en", "er", "es", "esta", "este", "et", "gli", "har", "het", "ich", "ikke", "ile", "ini", "inte", "itu", "jak",
    "jest", "la", "las", "le", "les", "leur", "los", "maar", "mais", "mga", "mit", "muy", "na", "naar", "nach", "ne", "nel", "nella", "ng", "nicht",
    "nie", "noch", "nous", "och", "oder", "og", "ook", "op", "oraz", "ou", "para", "pas", "pela", "pelo", "pero", "por", "przez", "que", "questo",
    "qui", "se", "ses", "seu", "sich", "sie", "sind", "som", "sono", "sont", "su", "sua", "sur", "sus", "tidak", "um", "una", "und", "une", "untuk",
    "van", "ve", "von", "voor", "vous", "werden", "wie", "wir", "wird", "wordt", "yang", "ze", "zijn", "zu",
];

static ENGLISH_WORDS: LazyLock<HashSet<&str>> = LazyLock::new(|| ENGLISH.into_iter().collect());
static OTHER_WORDS: LazyLock<HashSet<&str>> = LazyLock::new(|| OTHER.into_iter().collect());

/// Whether a text's start reads as English: words are runs of ASCII letters, and enough of them, few other characters, and English function words that outnumber other languages' twice over.
pub fn english(text: &str) -> bool {
    let sample = text.char_indices().nth(SAMPLE_CHARS).map_or(text, |(at, _)| &text[..at]);
    let non_ascii = sample.chars().filter(|c| !c.is_ascii()).count();
    let (mut words, mut letters, mut english, mut other) = (0, 0, 0, 0);
    for word in sample.split(|c: char| !c.is_ascii_alphabetic()).filter(|word| !word.is_empty()) {
        let lower = word.to_ascii_lowercase();
        words += 1;
        letters += word.len();
        english += usize::from(ENGLISH_WORDS.contains(lower.as_str()));
        other += usize::from(OTHER_WORDS.contains(lower.as_str()));
    }
    words >= MIN_WORDS && non_ascii * 10 <= letters * NON_ASCII_TENTHS && english * 100 >= ENGLISH_PERCENT * words && english > OTHER_TIMES * other
}

/// The page's own language, or `en` when it names none and its text reads as English.
pub fn page_lang<'a>(lang: &'a str, text: &str) -> &'a str {
    if lang.trim().is_empty() && english(text) {
        "en"
    } else {
        lang
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_without_a_language_is_english_when_its_text_reads_so() {
        let prose = "The city council voted on Tuesday to approve a new budget for the coming year, and members said that the plan will add funding for parks and roads that residents had asked for over the past months.";
        assert_eq!(page_lang("", prose), "en");
        assert_eq!(page_lang(" ", prose), "en");
        assert_eq!(page_lang("fr", prose), "fr", "a language the page names stands");
        assert_eq!(page_lang("", "Le conseil municipal a voté mardi pour approuver un nouveau budget pour l'année à venir, et les membres ont dit que le plan ajoute des fonds pour les parcs et les routes de la ville."), "");
        assert!(!english(&"é".repeat(SAMPLE_CHARS)), "characters, not bytes, are counted");
    }

    #[test]
    fn the_language_rule_reads_every_golden_text_as_the_indexer_does() {
        let golden: serde_json::Value = serde_json::from_str(include_str!("../tests/lang-golden.json")).unwrap();
        let words = |name: &str| golden[name].as_array().unwrap().iter().map(|word| word.as_str().unwrap()).collect::<Vec<_>>();
        assert_eq!((words("english"), words("other")), (ENGLISH.to_vec(), OTHER.to_vec()));
        let numbers = ["sample_chars", "min_words", "english_percent", "other_times", "non_ascii_tenths"].map(|name| golden[name].as_u64().unwrap() as usize);
        assert_eq!(numbers, [SAMPLE_CHARS, MIN_WORDS, ENGLISH_PERCENT, OTHER_TIMES, NON_ASCII_TENTHS]);
        for case in golden["cases"].as_array().unwrap() {
            assert_eq!(english(case["text"].as_str().unwrap()), case["english"].as_bool().unwrap(), "{}", case["name"]);
        }
    }
}

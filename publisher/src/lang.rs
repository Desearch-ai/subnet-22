//! A page's language when its extraction found none: English by its text, read as the indexer reads it (`tests/lang-golden.json`).

use std::collections::HashSet;
use std::sync::LazyLock;

/// Characters each window reads.
pub const SAMPLE_CHARS: usize = 4000;
/// Windows spread evenly from the text's start: at 0, 25, 50 and 75 percent of its characters.
pub const WINDOWS: usize = 4;
pub const MIN_WORDS: usize = 30;
/// English function words are at least this many in a hundred words, and more than twice other languages' function words.
pub const ENGLISH_PERCENT: usize = 8;
pub const OTHER_TIMES: usize = 2;
/// Characters past ASCII are at most this many tenths of the ASCII letters.
pub const NON_ASCII_TENTHS: usize = 1;
/// A language-light window's ASCII letters are at most this many times its ASCII digits and punctuation.
pub const LIGHT_LETTERS_TIMES: usize = 2;
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

/// What one window reads as, if anything.
struct Reading {
    english: bool,
    other: bool,
    light: bool,
}

/// English by the bars above; other-language by the same bars reversed or by its characters past ASCII; light when mostly digits and punctuation with sparse function words, other languages' no more than English ones.
fn read(sample: &str) -> Reading {
    let (mut non_ascii, mut marks) = (0, 0);
    for c in sample.chars() {
        non_ascii += usize::from(!c.is_ascii());
        marks += usize::from(c.is_ascii_graphic() && !c.is_ascii_alphabetic());
    }
    let (mut words, mut letters, mut english, mut other) = (0, 0, 0, 0);
    for word in sample.split(|c: char| !c.is_ascii_alphabetic()).filter(|word| !word.is_empty()) {
        let lower = word.to_ascii_lowercase();
        words += 1;
        letters += word.len();
        english += usize::from(ENGLISH_WORDS.contains(lower.as_str()));
        other += usize::from(OTHER_WORDS.contains(lower.as_str()));
    }
    let enough = words >= MIN_WORDS;
    let ascii = non_ascii * 10 <= letters * NON_ASCII_TENTHS;
    let dense = |n: usize| n * 100 >= ENGLISH_PERCENT * words;
    Reading {
        english: enough && ascii && dense(english) && english > OTHER_TIMES * other,
        other: (enough && dense(other) && other > OTHER_TIMES * english) || (non_ascii >= MIN_WORDS && !ascii),
        light: enough && ascii && letters <= LIGHT_LETTERS_TIMES * marks && !dense(english) && !dense(other) && other <= english,
    }
}

fn window(text: &str, start: usize) -> &str {
    let rest = text.char_indices().nth(start).map_or("", |(at, _)| &text[at..]);
    rest.char_indices().nth(SAMPLE_CHARS).map_or(rest, |(at, _)| &rest[..at])
}

/// Whether a text reads as English: its start passes as it always has, or more of its windows read English than clearly other-language, or none reads other-language and one is language-light.
pub fn english(text: &str) -> bool {
    let chars = text.chars().count();
    let reads = (0..WINDOWS).map(|k| read(window(text, chars * k / WINDOWS))).collect::<Vec<_>>();
    let count = |pick: fn(&Reading) -> bool| reads.iter().filter(|read| pick(read)).count();
    let (english, other) = (count(|read| read.english), count(|read| read.other));
    reads[0].english || english > other || (other == 0 && reads.iter().any(|read| read.light))
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
        let numbers = ["sample_chars", "windows", "min_words", "english_percent", "other_times", "non_ascii_tenths", "light_letters_times"]
            .map(|name| golden[name].as_u64().unwrap() as usize);
        assert_eq!(numbers, [SAMPLE_CHARS, WINDOWS, MIN_WORDS, ENGLISH_PERCENT, OTHER_TIMES, NON_ASCII_TENTHS, LIGHT_LETTERS_TIMES]);
        for case in golden["cases"].as_array().unwrap() {
            assert_eq!(english(case["text"].as_str().unwrap()), case["english"].as_bool().unwrap(), "{}", case["name"]);
        }
    }
}

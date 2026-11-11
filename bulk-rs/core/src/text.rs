//! The v2 text conventions of `desearch.extraction` that bulk pages go through: line cleaning, finishing and headings.

pub const MAX_TEXT_CHARS: usize = 200_000;

/// One line of rendered text, as `desearch.extraction.text.markdown.Line`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Line {
    pub text: String,
    pub linked: bool,
    pub verbatim: bool,
    pub row: bool,
}

impl Line {
    pub fn new(text: impl Into<String>) -> Self {
        Line { text: text.into(), ..Line::default() }
    }
}

/// `desearch.extraction.text.lines.clean`.
pub fn clean(lines: Vec<Line>, cards: bool) -> Vec<Line> {
    todo!("{} {cards}", lines.len())
}

/// `desearch.extraction.text.markdown.heading_level`.
pub fn heading_level(line: &Line) -> usize {
    todo!("{line:?}")
}

/// `desearch.extraction.text.markdown.finish`: the title line, then the lines, cut at a line end within `cap` characters.
pub fn finish(title: &str, lines: &[Line], cap: usize) -> String {
    todo!("{title} {} {cap}", lines.len())
}

/// `desearch.extraction.text_headings`.
pub fn text_headings(text: &str) -> Vec<String> {
    todo!("{}", text.len())
}

/// JSON text with lone surrogate escapes dropped, as Python's json.loads then `without_surrogates` leave the strings.
pub fn drop_lone_surrogates(json: &str) -> std::borrow::Cow<'_, str> {
    todo!("{}", json.len())
}

//! The title slug Stack Exchange puts in a question's address.

const MAX_CHARS: usize = 80;

fn ascii_of(c: char) -> &'static str {
    let lower: String = c.to_lowercase().collect();
    let has = |set: &str| set.contains(lower.as_str());
    match () {
        _ if has("àåáâäãåą") => "a",
        _ if has("èéêëę") => "e",
        _ if has("ìíîïı") => "i",
        _ if has("òóôõöøőð") => "o",
        _ if has("ùúûüŭů") => "u",
        _ if has("çćčĉ") => "c",
        _ if has("żźž") => "z",
        _ if has("śşšŝ") => "s",
        _ if has("ñń") => "n",
        _ if has("ýÿ") => "y",
        _ if has("ğĝ") => "g",
        _ => match c {
            'ř' => "r",
            'ł' => "l",
            'đ' => "d",
            'ß' => "ss",
            'Þ' => "th",
            'ĥ' => "h",
            'ĵ' => "j",
            _ => "",
        },
    }
}

/// Stack Exchange's `URLFriendly`: ASCII letters and digits lowercased, separators as single dashes, accents folded, the first 81 characters read.
pub fn slug(title: &str) -> String {
    let mut out = String::with_capacity(title.len().min(MAX_CHARS + 8));
    let mut dash = false;
    for (i, c) in title.chars().enumerate() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            dash = false;
        } else if c.is_ascii_uppercase() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if matches!(c, ' ' | ',' | '.' | '/' | '\\' | '-' | '_' | '=') {
            if !dash && !out.is_empty() {
                out.push('-');
                dash = true;
            }
        } else if (c as u32) >= 128 {
            let folded = ascii_of(c);
            if !folded.is_empty() {
                out.push_str(folded);
                dash = false;
            }
        }
        if i == MAX_CHARS {
            break;
        }
    }
    if dash {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::slug;

    #[test]
    fn known_question_slugs() {
        for (title, want) in [
            ("What is the difference between String and string in C#?", "what-is-the-difference-between-string-and-string-in-c"),
            ("How do I undo the most recent local commits in Git?", "how-do-i-undo-the-most-recent-local-commits-in-git"),
            ("What does the \"yield\" keyword do in Python?", "what-does-the-yield-keyword-do-in-python"),
            (
                "Why is processing a sorted array faster than processing an unsorted array?",
                "why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array",
            ),
            ("Café crème — naïve résumé", "cafe-creme-naive-resume"),
            ("C++ / C# ...", "c-c"),
        ] {
            assert_eq!(slug(title), want, "{title}");
        }
    }
}

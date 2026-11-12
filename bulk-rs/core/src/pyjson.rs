//! JSON text exactly as Python's `json.dumps` writes it: `", "` and `": "` separators, or `indent`, and ASCII-only strings unless asked.

use std::fmt::Write;

use serde_json::Value;

/// `json.dumps(value)`, or `json.dumps(value, indent=n)`.
pub fn dumps(value: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_value(&mut out, value, indent, 0, true);
    out
}

/// `json.dumps(value, ensure_ascii=False)`.
pub fn dumps_unicode(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, None, 0, false);
    out
}

/// A string as `json.dumps` quotes it, with `ensure_ascii`.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    write_str(&mut out, s, true);
    out
}

fn write_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ if !ascii && c >= ' ' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

fn newline(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', indent * level));
}

fn write_value(out: &mut String, value: &Value, indent: Option<usize>, level: usize, ascii: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => {
                let _ = write!(out, "{i}");
            }
            (_, Some(u), _) => {
                let _ = write!(out, "{u}");
            }
            (_, _, Some(f)) => out.push_str(&float_repr(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => write_str(out, s, ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                separate(out, i, indent, level + 1);
                write_value(out, item, indent, level + 1, ascii);
            }
            close(out, indent, level, ']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                separate(out, i, indent, level + 1);
                write_str(out, key, ascii);
                out.push_str(": ");
                write_value(out, item, indent, level + 1, ascii);
            }
            close(out, indent, level, '}');
        }
    }
}

fn separate(out: &mut String, i: usize, indent: Option<usize>, level: usize) {
    match indent {
        Some(n) => {
            if i > 0 {
                out.push(',');
            }
            newline(out, n, level);
        }
        None if i > 0 => out.push_str(", "),
        None => {}
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize, bracket: char) {
    if let Some(n) = indent {
        newline(out, n, level);
    }
    out.push(bracket);
}

/// Python's `repr(float)`: the shortest round-trip digits, in exponent form below 1e-4 and from 1e16.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if !(-4..16).contains(&exp) {
        let rest = &digits[1..];
        let dot = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        return format!("{sign}{}{dot}e{}{:02}", &digits[..1], if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    if exp < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() <= point {
        format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_dumps() {
        let value = json!({"a": [1, "\u{e9}\u{1f600}\u{7f}\u{1}"], "b": {}, "c": [], "d": null, "e": {"f": true}});
        assert_eq!(dumps(&value, None), "{\"a\": [1, \"\\u00e9\\ud83d\\ude00\\u007f\\u0001\"], \"b\": {}, \"c\": [], \"d\": null, \"e\": {\"f\": true}}");
        assert_eq!(
            dumps(&value, Some(1)),
            "{\n \"a\": [\n  1,\n  \"\\u00e9\\ud83d\\ude00\\u007f\\u0001\"\n ],\n \"b\": {},\n \"c\": [],\n \"d\": null,\n \"e\": {\n  \"f\": true\n }\n}"
        );
    }

    #[test]
    fn floats_as_python_repr() {
        let cases = [(1.0, "1.0"), (0.1, "0.1"), (1e16, "1e+16"), (1.5e-5, "1.5e-05"), (0.0001, "0.0001"), (123456.789, "123456.789"), (-2.5, "-2.5")];
        for (f, want) in cases {
            assert_eq!(float_repr(f), want);
        }
        assert_eq!(float_repr(9999999999999998.0), "9999999999999998.0");
    }
}

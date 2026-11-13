//! Python's truthiness, `str()` and `isdigit()`/`int()`, as `bulk/pmc.py` applies them to metadata and dates.

use bulk_core::pyjson::float_repr;
use serde_json::Value;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// `bool(v)` of a `json.loads` value.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str(v)` of a `json.loads` scalar.
pub fn str_of(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_f64() => float_repr(n.as_f64().unwrap_or(0.0)),
        other => other.to_string(),
    }
}

/// The value of a Unicode decimal digit, as `int()` reads it.
fn digit(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return c.to_digit(10);
    }
    if c.is_ascii() || c.general_category() != GeneralCategory::DecimalNumber {
        return None;
    }
    let mut start = c as u32;
    while let Some(prev) = start.checked_sub(1).and_then(char::from_u32) {
        if prev.general_category() != GeneralCategory::DecimalNumber {
            break;
        }
        start -= 1;
    }
    Some((c as u32 - start) % 10)
}

/// `str(int(s))` when `s.isdigit()`, else None; digits `int()` refuses (superscripts) count as not digits.
pub fn decimal(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    let digits: Option<String> = s.chars().map(|c| digit(c).and_then(|d| char::from_digit(d, 10))).collect();
    let digits = digits?;
    let trimmed = digits.trim_start_matches('0');
    Some(if trimmed.is_empty() { "0".into() } else { trimmed.into() })
}

/// Orders two `decimal` strings as integers.
pub fn cmp_decimal(a: &str, b: &str) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// `int(s)` of a `decimal` string, saturating.
pub fn decimal_value(s: &str) -> i64 {
    s.parse().unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_read_as_python_reads_them() {
        assert_eq!(decimal("0042").as_deref(), Some("42"));
        assert_eq!(decimal("２０２２").as_deref(), Some("2022"));
        assert_eq!(decimal("٣").as_deref(), Some("3"));
        assert_eq!(decimal("𝟗").as_deref(), Some("9"));
        assert_eq!(decimal(""), None);
        assert_eq!(decimal("²"), None);
        assert_eq!(decimal("1a"), None);
    }

    #[test]
    fn values_print_as_python_prints_them() {
        assert_eq!(str_of(&serde_json::json!(123)), "123");
        assert_eq!(str_of(&serde_json::json!(123.0)), "123.0");
        assert_eq!(str_of(&serde_json::json!(1e16)), "1e+16");
        assert_eq!(str_of(&serde_json::json!(1.5e-5)), "1.5e-05");
        assert_eq!(str_of(&serde_json::json!(true)), "True");
        assert!(!truthy(Some(&serde_json::json!(0.0))) && truthy(Some(&serde_json::json!("x"))));
    }
}

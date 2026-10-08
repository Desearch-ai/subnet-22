//! Python's own formatting and rounding: JSON as `json.dumps` writes it for everything signed or hashed, and `round`.

use serde_json::{Map, Number, Value};

/// `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
pub fn canonical(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write(&mut out, value);
    out.into_bytes()
}

fn write(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => number_into(out, number),
        Value::String(text) => string_into(out, text),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => object_into(out, map),
    }
}

fn object_into(out: &mut String, map: &Map<String, Value>) {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    out.push('{');
    for (i, (key, value)) in entries.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        string_into(out, key);
        out.push(':');
        write(out, value);
    }
    out.push('}');
}

fn number_into(out: &mut String, number: &Number) {
    match (number.as_i64(), number.as_u64(), number.as_f64()) {
        (Some(n), _, _) if !number.is_f64() => out.push_str(&n.to_string()),
        (_, Some(n), _) if !number.is_f64() => out.push_str(&n.to_string()),
        (_, _, Some(x)) => out.push_str(&float(x)),
        _ => out.push_str("null"),
    }
}

/// Python's `repr` of a float: the shortest digits that round-trip, in exponent form outside 1e-4 to 1e16.
pub fn float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let scientific = format!("{:e}", x.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("{:e} has an exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let point = exponent + 1;
    let sign = if x < 0.0 { "-" } else { "" };
    if point <= -4 || point > 16 {
        let rest = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{}{rest}e{exp_sign}{:02}", &digits[..1], exponent.abs());
    }
    let body = if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
    };
    format!("{sign}{body}")
}

/// Python's `round(x)`: to the nearest integer, ties to even.
pub fn round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python's `round(x, digits)`, which rounds the exact binary value.
pub fn round_to(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

fn string_into(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn floats_print_as_python_repr() {
        let cases = [
            (1_700_000_000.0, "1700000000.0"),
            (1_700_000_000.125, "1700000000.125"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (-2.5, "-2.5"),
            (100.0, "100.0"),
            (0.5, "0.5"),
        ];
        for (x, want) in cases {
            assert_eq!(float(x), want, "{x}");
        }
    }

    #[test]
    fn rounding_matches_python() {
        assert_eq!((round(2.5), round(3.5), round(-0.5), round(0.6)), (2, 4, 0, 1));
        assert_eq!(round_to(2.675, 2), 2.67);
        assert_eq!(round_to(0.123456, 4), 0.1235);
    }

    #[test]
    fn objects_are_sorted_and_compact() {
        let value = json!({"b": [1, 2.0, "é\n\u{1}"], "a": {"z": null, "y": true}, "": -3});
        assert_eq!(String::from_utf8(canonical(&value)).unwrap(), r#"{"":-3,"a":{"y":true,"z":null},"b":[1,2.0,"é\n\u0001"]}"#);
    }
}

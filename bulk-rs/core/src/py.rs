//! Python's view of `json.loads` values and text files: truthiness, f-string formatting of scalars, `read_text`.

use std::borrow::Cow;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};

use crate::pyjson::float_repr;
use crate::text::drop_lone_surrogates;

/// `bool(v)`, a missing key read as None.
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

/// The value when it is truthy, as `if v:` takes it.
pub fn when_truthy(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| truthy(Some(v)))
}

/// The first truthy value, as `a or b` picks it; None when neither is.
pub fn or<'a>(a: Option<&'a Value>, b: Option<&'a Value>) -> Option<&'a Value> {
    when_truthy(a).or(when_truthy(b))
}

/// `v == k` for an integer `k`, False and True being 0 and 1.
pub fn equals_int(v: &Value, k: i64) -> bool {
    match v {
        Value::Bool(b) => i64::from(*b) == k,
        Value::Number(n) => n.as_i64().map_or_else(|| n.as_f64() == Some(k as f64), |i| i == k),
        _ => false,
    }
}

fn float_str(f: f64) -> String {
    if f.is_nan() {
        "nan".into()
    } else if f.is_infinite() {
        if f > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        float_repr(f)
    }
}

fn number_str(n: &serde_json::Number) -> String {
    match (n.as_i64(), n.as_u64()) {
        (Some(i), _) => i.to_string(),
        (_, Some(u)) => u.to_string(),
        _ => float_str(n.as_f64().unwrap_or(f64::NAN)),
    }
}

/// `f"{v}"` of a scalar; lists and dicts are refused, since their repr is not reproduced.
pub fn format(v: &Value) -> Result<String> {
    Ok(match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => number_str(n),
        other => bail!("{} in an f-string", type_name(other)),
    })
}

fn group(digits: &str) -> String {
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `f"{v:,}"` of a number: thousands separated by commas.
pub fn grouped(v: &Value) -> Result<String> {
    let text = match v {
        Value::Bool(b) => return Ok(u8::from(*b).to_string()),
        Value::Number(n) => number_str(n),
        other => bail!("format(..., ',') of {}", type_name(other)),
    };
    if text.contains(['e', 'n']) {
        return Ok(text);
    }
    let (sign, unsigned) = text.strip_prefix('-').map_or(("", text.as_str()), |rest| ("-", rest));
    let (whole, fraction) = unsigned.split_once('.').map_or((unsigned, None), |(w, f)| (w, Some(f)));
    Ok(match fraction {
        Some(fraction) => format!("{sign}{}.{fraction}", group(whole)),
        None => format!("{sign}{}", group(whole)),
    })
}

/// The number `v` holds, True and False as 1 and 0.
pub fn float(v: &Value) -> Result<f64> {
    match v {
        Value::Bool(b) => Ok(f64::from(u8::from(*b))),
        Value::Number(n) => n.as_f64().ok_or_else(|| anyhow!("{n} is not a number")),
        other => bail!("{} where a number belongs", type_name(other)),
    }
}

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "None",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The str `v` holds, for the str methods Python calls on it.
pub fn str_of<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str().ok_or_else(|| anyhow!("{what} is a {}, not a str", type_name(v)))
}

/// `v.get(key)` of a dict.
pub fn get<'a>(v: &'a Value, key: &str) -> Result<Option<&'a Value>> {
    match v {
        Value::Object(map) => Ok(map.get(key)),
        other => bail!("{} has no .get({key:?})", type_name(other)),
    }
}

/// `v[key]` of a dict.
pub fn item<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    get(v, key)?.ok_or_else(|| anyhow!("KeyError: {key:?}"))
}

/// `(v or {})`, a dict.
pub fn dict_or_empty<'a>(v: Option<&'a Value>, what: &str) -> Result<Cow<'a, Map<String, Value>>> {
    match when_truthy(v) {
        None => Ok(Cow::Owned(Map::new())),
        Some(Value::Object(map)) => Ok(Cow::Borrowed(map)),
        Some(other) => bail!("{what} is a {}, not a dict", type_name(other)),
    }
}

/// The items `for x in v` takes from a list.
pub fn list<'a>(v: &'a Value, what: &str) -> Result<&'a [Value]> {
    v.as_array().map(Vec::as_slice).ok_or_else(|| anyhow!("{what} is a {}, not a list", type_name(v)))
}

/// Text read in universal newlines mode: `\r\n` and `\r` become `\n`.
pub fn universal_newlines(text: &str) -> Cow<'_, str> {
    if !text.contains('\r') {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `Path.read_text()` with UTF-8 as the locale's encoding.
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8(bytes).with_context(|| format!("{} is not UTF-8", path.display()))?;
    Ok(universal_newlines(&text).into_owned())
}

/// `bytes.decode("utf-8", "replace")`.
pub fn decode_replace(bytes: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

/// `Path.read_text("utf-8", "replace")` of bytes already read.
pub fn text_replace(bytes: &[u8]) -> String {
    universal_newlines(&decode_replace(bytes)).into_owned()
}

/// `json.loads(path.read_text())`, lone surrogate escapes dropped as `without_surrogates` drops them from pages.
pub fn load_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).with_context(|| format!("{} is not UTF-8", path.display()))?;
    match serde_json::from_str(text) {
        Ok(value) => Ok(value),
        Err(error) => serde_json::from_str(&drop_lone_surrogates(text)).map_err(|_| error).with_context(|| format!("reading {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_as_f_strings() {
        assert_eq!(format(&json!(9481913)).unwrap(), "9481913");
        assert_eq!(format(&json!(360.0)).unwrap(), "360.0");
        assert_eq!(format(&json!(null)).unwrap(), "None");
        assert_eq!(format(&json!(true)).unwrap(), "True");
        assert!(format(&json!({"a": 1})).is_err());
        assert_eq!(grouped(&json!(2009702)).unwrap(), "2,009,702");
        assert_eq!(grouped(&json!(-1234)).unwrap(), "-1,234");
        assert_eq!(grouped(&json!(999)).unwrap(), "999");
        assert_eq!(grouped(&json!(1234567.5)).unwrap(), "1,234,567.5");
        assert_eq!(grouped(&json!(1e16)).unwrap(), "1e+16");
        assert_eq!(grouped(&json!(0.5)).unwrap(), "0.5");
        assert_eq!(grouped(&json!(true)).unwrap(), "1");
        assert!(grouped(&json!("1")).is_err());
    }

    #[test]
    fn truthiness_and_equality() {
        assert!(!truthy(Some(&json!(0.0))) && !truthy(Some(&json!({}))) && !truthy(None));
        assert!(truthy(Some(&json!("x"))) && truthy(Some(&json!([0]))));
        assert_eq!(or(Some(&json!("")), Some(&json!("b"))), Some(&json!("b")));
        assert_eq!(or(Some(&json!(0)), None), None);
        assert!(equals_int(&json!(35.0), 35) && equals_int(&json!(false), 0) && !equals_int(&json!("0"), 0));
    }

    #[test]
    fn newlines_and_decoding() {
        assert_eq!(universal_newlines("a\r\nb\rc\n"), "a\nb\nc\n");
        assert_eq!(text_replace(b"a\xffb\r\n"), "a\u{fffd}b\n");
    }
}

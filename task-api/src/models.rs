//! Request bodies checked as the pydantic models checked them, with the same defaults and the same 422 details.

use serde_json::{json, Map, Value};

use crate::rounds::Url;

pub const MAX_URL_DETAILS: usize = 1000;
pub const MAX_CLAIM: i64 = 50;
const OUTCOMES: [&str; 6] = ["matched", "mismatched", "unverifiable", "errors_confirmed", "errors_unconfirmed", "not_fetched"];
const EMBED_OUTCOMES: [&str; 3] = ["matched", "mismatched", "unverifiable"];

/// What pydantic found wrong, as FastAPI's 422 `detail` lists it.
#[derive(Debug, Default)]
pub struct Invalid(pub Vec<Value>);

impl Invalid {
    fn add(&mut self, kind: &str, loc: &[Value], msg: impl Into<String>) {
        self.0.push(json!({"type": kind, "loc": loc, "msg": msg.into()}));
    }

    pub fn single(kind: &str, loc: &[&str], msg: &str) -> Invalid {
        let mut invalid = Invalid::default();
        invalid.add(kind, &loc.iter().map(|l| Value::from(*l)).collect::<Vec<_>>(), msg);
        invalid
    }

    fn result<T>(self, value: T) -> Result<T, Invalid> {
        if self.0.is_empty() {
            Ok(value)
        } else {
            Err(self)
        }
    }
}

fn at(base: &[Value], name: impl Into<Value>) -> Vec<Value> {
    let mut loc = base.to_vec();
    loc.push(name.into());
    loc
}

/// An int as pydantic's lax mode takes one: an integer, a whole float, or a numeric string.
pub fn lax_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|x| x.fract() == 0.0 && x.abs() < 9.2e18).map(|x| x as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

fn lax_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// Declared fields checked into `out`, each defaulted when absent.
struct Fields<'a> {
    input: &'a Map<String, Value>,
    loc: Vec<Value>,
    out: Map<String, Value>,
    invalid: Invalid,
}

impl<'a> Fields<'a> {
    fn new(input: &'a Map<String, Value>, loc: Vec<Value>, keep_extra: bool) -> Self {
        let out = if keep_extra { input.clone() } else { Map::new() };
        Fields { input, loc, out, invalid: Invalid::default() }
    }

    fn forbid_extra(&mut self, declared: &[&str]) {
        for name in self.input.keys().filter(|name| !declared.contains(&name.as_str())) {
            let loc = at(&self.loc, name.as_str());
            self.invalid.add("extra_forbidden", &loc, "Extra inputs are not permitted");
        }
    }

    fn int(&mut self, name: &str, default: Option<i64>, ge: Option<i64>) -> i64 {
        let loc = at(&self.loc, name);
        let value = match self.input.get(name) {
            None => match default {
                Some(default) => default,
                None => {
                    self.invalid.add("missing", &loc, "Field required");
                    return 0;
                }
            },
            Some(value) => match lax_int(value) {
                Some(n) => n,
                None => {
                    self.invalid.add("int_parsing", &loc, "Input should be a valid integer");
                    return 0;
                }
            },
        };
        if let Some(ge) = ge.filter(|ge| value < *ge) {
            self.invalid.add("greater_than_equal", &loc, format!("Input should be greater than or equal to {ge}"));
        }
        self.out.insert(name.into(), value.into());
        value
    }

    fn float(&mut self, name: &str, default: Option<f64>, nullable: bool) {
        let loc = at(&self.loc, name);
        let value = match self.input.get(name) {
            None => default.map_or(Value::Null, Value::from),
            Some(Value::Null) if nullable => Value::Null,
            Some(value) => match lax_float(value) {
                Some(x) => x.into(),
                None => {
                    self.invalid.add("float_parsing", &loc, "Input should be a valid number");
                    return;
                }
            },
        };
        self.out.insert(name.into(), value);
    }

    fn text(&mut self, name: &str, default: Option<&str>, nullable: bool, max: Option<usize>) -> Option<String> {
        let loc = at(&self.loc, name);
        let value = match self.input.get(name) {
            None => match default {
                Some(default) => Some(default.to_string()),
                None if nullable => None,
                None => {
                    self.invalid.add("missing", &loc, "Field required");
                    return None;
                }
            },
            Some(Value::Null) if nullable => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                self.invalid.add("string_type", &loc, "Input should be a valid string");
                return None;
            }
        };
        if let (Some(text), Some(max)) = (&value, max) {
            if text.chars().count() > max {
                self.invalid.add("string_too_long", &loc, format!("String should have at most {max} characters"));
            }
        }
        self.out.insert(name.into(), value.clone().map_or(Value::Null, Value::from));
        value
    }

    fn literal(&mut self, name: &str, allowed: &[&str], default: Option<&str>) -> Option<String> {
        let loc = at(&self.loc, name);
        let value = match (self.input.get(name), default) {
            (None, Some(default)) => default.to_string(),
            (None, None) => {
                self.invalid.add("missing", &loc, "Field required");
                return None;
            }
            (Some(Value::String(s)), _) if allowed.contains(&s.as_str()) => s.clone(),
            (Some(_), _) => {
                let quoted: Vec<String> = allowed.iter().map(|a| format!("'{a}'")).collect();
                self.invalid.add("literal_error", &loc, format!("Input should be {}", or_list(&quoted)));
                return None;
            }
        };
        self.out.insert(name.into(), value.clone().into());
        Some(value)
    }

    fn boolean(&mut self, name: &str, default: bool) {
        let loc = at(&self.loc, name);
        let value = match self.input.get(name) {
            None => default,
            Some(Value::Bool(b)) => *b,
            Some(value) => match lax_int(value) {
                Some(0) => false,
                Some(1) => true,
                _ => {
                    self.invalid.add("bool_parsing", &loc, "Input should be a valid boolean");
                    return;
                }
            },
        };
        self.out.insert(name.into(), value.into());
    }

    fn list(&mut self, name: &str, max: Option<usize>) -> Vec<Value> {
        let loc = at(&self.loc, name);
        let items = match self.input.get(name) {
            None => Vec::new(),
            Some(Value::Array(items)) => items.clone(),
            Some(_) => {
                self.invalid.add("list_type", &loc, "Input should be a valid list");
                return Vec::new();
            }
        };
        if let Some(max) = max.filter(|max| items.len() > *max) {
            self.invalid.add("too_long", &loc, format!("List should have at most {max} items after validation, not {}", items.len()));
        }
        items
    }

    fn object(&mut self, value: &'a Value, loc: Vec<Value>) -> Option<&'a Map<String, Value>> {
        match value.as_object() {
            Some(object) => Some(object),
            None => {
                self.invalid.add("model_type", &loc, "Input should be a valid dictionary or object");
                None
            }
        }
    }

    fn merge(&mut self, other: Invalid) {
        self.invalid.0.extend(other.0);
    }
}

fn or_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

fn counts_match(fields: &mut Fields, outcomes: &[&str], samples: &[Value]) {
    let sampled = fields.out.get("sampled").and_then(Value::as_i64).unwrap_or(0);
    let counted = |outcome: &str| samples.iter().filter(|s| s["outcome"] == outcome).count() as i64;
    if sampled != samples.len() as i64 || outcomes.iter().any(|o| fields.out.get(*o).and_then(Value::as_i64).unwrap_or(0) != counted(o)) {
        fields.invalid.add("value_error", &[], "Value error, the counts do not match the samples");
    }
}

pub struct ClaimBody {
    pub kind: String,
    pub count: i64,
}

impl ClaimBody {
    pub fn parse(body: &Value) -> Result<ClaimBody, Invalid> {
        let Some(input) = body.as_object() else {
            if body.is_null() {
                return Ok(ClaimBody { kind: "crawl".into(), count: 1 });
            }
            return Err(Invalid::single("model_attributes_type", &["body"], "Input should be a valid dictionary or object to extract fields from"));
        };
        let mut fields = Fields::new(input, vec!["body".into()], false);
        let kind = fields.literal("kind", &["crawl", "embed"], Some("crawl"));
        let count = fields.int("count", Some(1), Some(1));
        if count > MAX_CLAIM {
            let loc = at(&fields.loc, "count");
            fields.invalid.add("less_than_equal", &loc, format!("Input should be less than or equal to {MAX_CLAIM}"));
        }
        fields.invalid.result(ClaimBody { kind: kind.unwrap_or_default(), count })
    }
}

pub struct CompleteBody {
    pub key: String,
    /// The miner's counts, as `model_dump(exclude={"key"})` gives them.
    pub reported: Value,
}

impl CompleteBody {
    pub fn parse(body: &Value) -> Result<CompleteBody, Invalid> {
        let input = body
            .as_object()
            .ok_or_else(|| Invalid::single("model_attributes_type", &["body"], "Input should be a valid dictionary or object to extract fields from"))?;
        let mut fields = Fields::new(input, vec!["body".into()], false);
        let key = fields.text("key", None, false, None);
        for name in ["rows", "ok", "errors", "bytes"] {
            fields.int(name, Some(0), Some(0));
        }
        fields.out.remove("key");
        let reported = Value::Object(fields.out.clone());
        fields.invalid.result(CompleteBody { key: key.unwrap_or_default(), reported })
    }
}

/// A crawl verdict's result, dumped with every declared field and any extras.
pub fn score(body: &Map<String, Value>) -> Result<Map<String, Value>, Invalid> {
    let mut fields = Fields::new(body, Vec::new(), true);
    for name in [
        "returned",
        "missing",
        "duplicates",
        "error_rows",
        "sampled",
        "matched",
        "mismatched",
        "unverifiable",
        "not_fetched",
        "errors_confirmed",
        "errors_unconfirmed",
        "reextract_mismatch",
    ] {
        fields.int(name, Some(0), Some(0));
    }
    fields.literal("verdict", &["pass", "fail", "void"], None);
    fields.text("reason", Some(""), false, None);
    let samples = fields.list("samples", None);
    let mut dumped = Vec::with_capacity(samples.len());
    for (i, sample) in samples.iter().enumerate() {
        let loc = vec!["samples".into(), i.into()];
        if let Some(input) = fields.object(sample, loc.clone()) {
            let (value, invalid) = sample_of(input, loc);
            fields.merge(invalid);
            dumped.push(value);
        }
    }
    fields.out.insert("samples".into(), dumped.clone().into());
    let urls = fields.list("urls", Some(MAX_URL_DETAILS));
    let mut details = Vec::with_capacity(urls.len());
    for (i, url) in urls.iter().enumerate() {
        let loc = vec!["urls".into(), i.into()];
        if let Some(input) = fields.object(url, loc.clone()) {
            let (value, invalid) = url_detail(input, loc);
            fields.merge(invalid);
            details.push(value);
        }
    }
    fields.out.insert("urls".into(), details.into());
    let rejected = fields.list("rejected", Some(MAX_URL_DETAILS));
    if let Some(i) = rejected.iter().position(|r| !r.is_string()) {
        let loc = vec!["rejected".into(), i.into()];
        fields.invalid.add("string_type", &loc, "Input should be a valid string");
    }
    fields.out.insert("rejected".into(), rejected.into());
    if fields.invalid.0.is_empty() {
        counts_match(&mut fields, &OUTCOMES, &dumped);
        if fields.out["error_rows"].as_i64() > fields.out["returned"].as_i64() {
            fields.invalid.add("value_error", &[], "Value error, error_rows cannot exceed returned");
        }
    }
    let out = fields.out;
    fields.invalid.result(out)
}

fn sample_of(input: &Map<String, Value>, loc: Vec<Value>) -> (Value, Invalid) {
    let mut fields = Fields::new(input, loc, true);
    fields.text("url", None, false, None);
    fields.literal("outcome", &OUTCOMES, None);
    fields.float("similarity", Some(0.0), false);
    for name in ["miner_status", "validator_status", "miner_chars", "validator_chars"] {
        fields.int(name, Some(0), None);
    }
    (Value::Object(fields.out), fields.invalid)
}

fn url_detail(input: &Map<String, Value>, loc: Vec<Value>) -> (Value, Invalid) {
    let mut fields = Fields::new(input, loc, false);
    fields.forbid_extra(&[
        "url",
        "status",
        "error",
        "text_chars",
        "sampled",
        "outcome",
        "why",
        "similarity",
        "precision",
        "recall",
        "growth",
        "validator_error",
        "via",
        "miner_chars",
        "validator_chars",
        "miner_snippet",
        "validator_snippet",
        "diff_at",
        "miner_window",
        "validator_window",
        "rejected",
    ]);
    fields.text("url", None, false, Some(4096));
    fields.int("status", Some(0), None);
    fields.text("error", None, true, Some(64));
    fields.int("text_chars", Some(0), None);
    fields.boolean("sampled", false);
    fields.text("outcome", None, true, Some(32));
    fields.text("why", None, true, Some(500));
    for name in ["similarity", "precision", "recall", "growth"] {
        fields.float(name, None, true);
    }
    fields.text("validator_error", None, true, Some(200));
    fields.text("via", Some(""), false, Some(16));
    for name in ["miner_chars", "validator_chars", "diff_at"] {
        nullable_int(&mut fields, name);
    }
    for (name, max) in [("miner_snippet", 500), ("validator_snippet", 500), ("miner_window", 500), ("validator_window", 500)] {
        fields.text(name, None, true, Some(max));
    }
    fields.boolean("rejected", false);
    (Value::Object(fields.out), fields.invalid)
}

fn nullable_int(fields: &mut Fields, name: &str) {
    match fields.input.get(name) {
        None | Some(Value::Null) => {
            fields.out.insert(name.into(), Value::Null);
        }
        Some(_) => {
            fields.int(name, None, None);
        }
    }
}

/// An embed verdict's result, every field declared and nothing else allowed.
pub fn embed_score(body: &Map<String, Value>) -> Result<Map<String, Value>, Invalid> {
    let mut fields = Fields::new(body, Vec::new(), false);
    fields.forbid_extra(&[
        "returned",
        "missing",
        "duplicates",
        "malformed",
        "sampled",
        "matched",
        "mismatched",
        "unverifiable",
        "min_similarity",
        "verdict",
        "reason",
        "samples",
    ]);
    for name in ["returned", "missing", "duplicates", "malformed", "sampled", "matched", "mismatched", "unverifiable"] {
        fields.int(name, Some(0), Some(0));
    }
    fields.float("min_similarity", None, true);
    fields.literal("verdict", &["pass", "fail", "void"], None);
    fields.text("reason", Some(""), false, Some(64));
    let samples = fields.list("samples", Some(MAX_URL_DETAILS));
    let mut dumped = Vec::with_capacity(samples.len());
    for (i, sample) in samples.iter().enumerate() {
        let loc = vec!["samples".into(), i.into()];
        if let Some(input) = fields.object(sample, loc.clone()) {
            let mut inner = Fields::new(input, loc, false);
            inner.forbid_extra(&["text_id", "outcome", "similarity"]);
            inner.text("text_id", None, false, Some(300));
            inner.literal("outcome", &EMBED_OUTCOMES, None);
            inner.float("similarity", None, true);
            fields.merge(inner.invalid);
            dumped.push(Value::Object(inner.out));
        }
    }
    fields.out.insert("samples".into(), dumped.clone().into());
    if fields.invalid.0.is_empty() {
        counts_match(&mut fields, &EMBED_OUTCOMES, &dumped);
    }
    let out = fields.out;
    fields.invalid.result(out)
}

pub fn release(body: &Value) -> Result<(), Invalid> {
    let input = body
        .as_object()
        .ok_or_else(|| Invalid::single("model_attributes_type", &["body"], "Input should be a valid dictionary or object to extract fields from"))?;
    let mut fields = Fields::new(input, vec!["body".into()], false);
    fields.literal("reason", &["missing"], Some("missing"));
    fields.invalid.result(())
}

pub struct Enqueue {
    pub urls: Vec<Url>,
    pub batch_id: String,
}

impl Enqueue {
    pub fn parse(body: &Value) -> Result<Enqueue, Invalid> {
        let input = body
            .as_object()
            .ok_or_else(|| Invalid::single("model_attributes_type", &["body"], "Input should be a valid dictionary or object to extract fields from"))?;
        let mut fields = Fields::new(input, vec!["body".into()], false);
        let batch_id = fields.text("batch_id", Some(""), false, Some(128)).unwrap_or_default();
        let mut urls = Vec::new();
        match input.get("urls") {
            None => fields.invalid.add("missing", &["body".into(), "urls".into()], "Field required"),
            Some(Value::Array(items)) => {
                for (i, item) in items.iter().enumerate() {
                    let loc = vec!["body".into(), "urls".into(), i.into()];
                    let host = item.get("host").and_then(Value::as_str);
                    let url = item.get("url").and_then(Value::as_str);
                    match (host, url) {
                        (Some(host), Some(url)) => urls.push(Url { host: host.into(), url: url.into() }),
                        _ => fields.invalid.add("dataclass_type", &loc, "Input should be a dictionary with a string host and url"),
                    }
                }
            }
            Some(_) => fields.invalid.add("list_type", &["body".into(), "urls".into()], "Input should be a valid list"),
        }
        fields.invalid.result(Enqueue { urls, batch_id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_score_is_dumped_with_its_defaults_and_extras() {
        let body =
            json!({"returned": 3, "verdict": "pass", "sampled": 1, "matched": 1, "samples": [{"url": "u", "outcome": "matched", "x": 1}], "crashed": true});
        let dumped = score(body.as_object().unwrap()).unwrap();
        assert_eq!(dumped["missing"], 0);
        assert_eq!(dumped["crashed"], true);
        assert_eq!(
            dumped["samples"][0],
            json!({"url": "u", "outcome": "matched", "similarity": 0.0, "miner_status": 0, "validator_status": 0, "miner_chars": 0, "validator_chars": 0, "x": 1})
        );
        let wrong = json!({"verdict": "pass", "sampled": 2, "samples": []});
        assert_eq!(score(wrong.as_object().unwrap()).unwrap_err().0[0]["msg"], "Value error, the counts do not match the samples");
        let bad = json!({"verdict": "maybe", "returned": -1});
        assert_eq!(score(bad.as_object().unwrap()).unwrap_err().0.len(), 2);
    }

    #[test]
    fn claims_default_and_bound_their_count() {
        let claim = ClaimBody::parse(&Value::Null).unwrap();
        assert_eq!((claim.kind.as_str(), claim.count), ("crawl", 1));
        assert!(ClaimBody::parse(&json!({"count": 51})).is_err());
        assert!(ClaimBody::parse(&json!({"kind": "other"})).is_err());
        let complete = CompleteBody::parse(&json!({"key": "k", "rows": 5})).unwrap();
        assert_eq!(complete.reported, json!({"rows": 5, "ok": 0, "errors": 0, "bytes": 0}));
    }
}

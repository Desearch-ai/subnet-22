//! How validators' results become votes, and votes a verdict: feasibility, credit from samples, majorities and audits.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::budgets::COVERAGE_GATE;
use crate::credit::{credited_urls, Outcomes, EVIDENCE};
use crate::py;

pub const NO_MAJORITY: &str = "validators_disagree";
pub const MAX_CREDIT_DIVERGENCE: f64 = 0.15;
/// The validator's own timeout or crash, never held against the miner.
pub const VALIDATOR_FAULT_REASONS: [&str; 1] = ["unscorable"];
/// A page unreachable both for the miner and the validator (not_fetched) can sit on either kind of row.
pub const CONTENT_OUTCOMES: [&str; 3] = ["matched", "mismatched", "unverifiable"];
pub const ERROR_OUTCOMES: [&str; 2] = ["errors_confirmed", "errors_unconfirmed"];

/// A result the task could not have produced.
#[derive(Debug)]
pub struct Infeasible(pub String);

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub outcome: &'static str,
    pub vote: Option<Value>,
    pub votes: Vec<Value>,
    pub agreed: Vec<String>,
    pub disagreed: Vec<String>,
}

impl Decision {
    fn audit() -> Self {
        Decision { outcome: "audit", vote: None, votes: Vec::new(), agreed: Vec::new(), disagreed: Vec::new() }
    }

    pub fn last(vote: Value, votes: Vec<Value>) -> Self {
        Decision { outcome: "final", vote: Some(vote), votes, agreed: Vec::new(), disagreed: Vec::new() }
    }

    pub fn vote(&self) -> &Value {
        self.vote.as_ref().expect("a final decision has a vote")
    }
}

/// How many samples had each outcome.
pub struct Counted(HashMap<String, i64>);

impl Counted {
    pub fn of(samples: &[Value], field: &str) -> Self {
        let mut counts = HashMap::new();
        for sample in samples {
            *counts.entry(sample[field].as_str().unwrap_or_default().to_string()).or_insert(0) += 1;
        }
        Counted(counts)
    }
}

impl Outcomes for Counted {
    fn count(&self, outcome: &str) -> i64 {
        self.0.get(outcome).copied().unwrap_or(0)
    }
}

fn int(value: &Value) -> i64 {
    value.as_i64().or_else(|| value.as_f64().map(|x| x as i64)).unwrap_or(0)
}

fn samples(result: &Value) -> Vec<Value> {
    result.get("samples").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn urls_of(job: &Value) -> Vec<&str> {
    let mut urls: Vec<&str> = job["urls"].as_array().map(|u| u.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    urls.sort_unstable();
    urls.dedup();
    urls
}

/// A report must describe rows the task can hold; counts are not taken on trust.
pub fn check_feasible(job: &Value, result: &Value) -> Result<(), Infeasible> {
    let urls = urls_of(job);
    let samples = samples(result);
    let sampled: Vec<&str> = samples.iter().map(|s| s["url"].as_str().unwrap_or_default()).collect();
    let mut distinct = sampled.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let outcomes = Counted::of(&samples, "outcome");
    let returned = int(&result["returned"]);
    let error_rows = int(&result["error_rows"]);
    let content: i64 = CONTENT_OUTCOMES.iter().map(|o| outcomes.count(o)).sum();
    let errors: i64 = ERROR_OUTCOMES.iter().map(|o| outcomes.count(o)).sum();
    let assigned = urls.len() as i64;
    let problems = [
        (distinct.len() < sampled.len(), "a URL was sampled twice".to_string()),
        (!sampled.iter().all(|url| urls.binary_search(url).is_ok()), "a sampled URL is not in the task".to_string()),
        (returned > assigned, "more rows returned than URLs assigned".to_string()),
        (sampled.len() as i64 > returned, "more samples than rows returned".to_string()),
        (content > returned - error_rows, "more content samples than content rows".to_string()),
        (errors > error_rows, "more error samples than error rows".to_string()),
        (
            result["verdict"] == "pass" && (returned as f64) < COVERAGE_GATE * assigned as f64,
            format!("a pass needs {}% of the URLs returned", percent(COVERAGE_GATE)),
        ),
    ];
    match problems.into_iter().find(|(wrong, _)| *wrong) {
        Some((_, why)) => Err(Infeasible(why)),
        None => Ok(()),
    }
}

/// Credit comes from the samples, never from the validator's own number.
pub fn build_vote(job: &Value, validator: &str, result: Map<String, Value>) -> Result<Value, Infeasible> {
    let result = Value::Object(result);
    check_feasible(job, &result)?;
    let returned = int(&result["returned"]);
    let error_rows = int(&result["error_rows"]);
    let samples = samples(&result);
    let outcomes = Counted::of(&samples, "outcome");
    let mut verdict = result["verdict"].as_str().unwrap_or_default().to_string();
    let mut reason = result["reason"].as_str().unwrap_or_default().to_string();
    if verdict == "fail" && VALIDATOR_FAULT_REASONS.contains(&reason.as_str()) {
        verdict = "void".into();
    } else if verdict == "pass" && !EVIDENCE.iter().any(|o| outcomes.count(o) > 0) {
        (verdict, reason) = ("void".into(), "inconclusive".into());
    } else if verdict == "pass" && outcomes.count("unverifiable") * 2 >= samples.len() as i64 {
        (verdict, reason) = ("void".into(), "unverifiable".into());
    }
    let credited = if verdict == "pass" { credited_urls(returned - error_rows, error_rows, &outcomes) } else { 0 };
    let urls = urls_of(job);
    let mut rejected: Vec<&str> =
        result["rejected"].as_array().map(|r| r.iter().filter_map(Value::as_str).filter(|u| urls.binary_search(u).is_ok()).collect()).unwrap_or_default();
    rejected.sort_unstable();
    rejected.dedup();
    let mut full = result.as_object().cloned().unwrap_or_default();
    full.insert("verdict".into(), verdict.clone().into());
    full.insert("reason".into(), reason.into());
    full.insert("returned".into(), returned.into());
    full.insert("error_rows".into(), error_rows.into());
    full.insert("credited".into(), credited.into());
    full.insert("rejected".into(), json!(rejected));
    Ok(json!({"validator": validator, "verdict": verdict, "credited": credited, "result": full}))
}

/// A pass must account for every text; a validator's counts are not taken on trust.
pub fn check_embed_feasible(job: &Value, result: &Value) -> Result<(), Infeasible> {
    let samples = samples(result);
    let mut ids: Vec<&str> = samples.iter().map(|s| s["text_id"].as_str().unwrap_or_default()).collect();
    let sampled = ids.len();
    ids.sort_unstable();
    ids.dedup();
    let outcomes = Counted::of(&samples, "outcome");
    let texts = int(&job["texts"]);
    let returned = int(&result["returned"]);
    let truthy = |name: &str| int(&result[name]) != 0;
    let problems = [
        (ids.len() < sampled, "a text was sampled twice"),
        (returned > texts, "more vectors returned than texts assigned"),
        (sampled as i64 > returned, "more samples than vectors returned"),
        (
            result["verdict"] == "pass"
                && (returned < texts || truthy("missing") || truthy("duplicates") || truthy("malformed") || outcomes.count("mismatched") > 0),
            "a pass needs every text embedded and every sample matched",
        ),
    ];
    match problems.into_iter().find(|(wrong, _)| *wrong) {
        Some((_, why)) => Err(Infeasible(why.into())),
        None => Ok(()),
    }
}

/// Credit is the characters the API assigned, paid in full on a pass.
pub fn build_embed_vote(job: &Value, validator: &str, result: Map<String, Value>) -> Result<Value, Infeasible> {
    let result = Value::Object(result);
    check_embed_feasible(job, &result)?;
    let mut verdict = result["verdict"].as_str().unwrap_or_default().to_string();
    let mut reason = result["reason"].as_str().unwrap_or_default().to_string();
    if verdict == "pass" && int(&result["matched"]) == 0 {
        (verdict, reason) = ("void".into(), "inconclusive".into());
    }
    let credited = if verdict == "pass" { int(&job["chars"]) } else { 0 };
    let mut full = result.as_object().cloned().unwrap_or_default();
    full.insert("verdict".into(), verdict.clone().into());
    full.insert("reason".into(), reason.into());
    full.insert("credited".into(), credited.into());
    Ok(json!({"validator": validator, "verdict": verdict, "credited": credited, "result": full}))
}

fn credited(vote: &Value) -> i64 {
    int(&vote["credited"])
}

pub fn credits_agree(one: i64, other: i64) -> bool {
    ((one - other).abs() as f64) <= MAX_CREDIT_DIVERGENCE * one.max(other) as f64
}

pub fn credits_diverge(votes: &[&Value]) -> bool {
    let paid: Vec<i64> = votes.iter().map(|v| credited(v)).collect();
    paid.len() > 1 && !credits_agree(*paid.iter().min().unwrap_or(&0), *paid.iter().max().unwrap_or(&0))
}

/// The first vote with the lowest credit, as Python's `min` picks it.
fn cheapest<'a>(votes: &[&'a Value]) -> &'a Value {
    votes.iter().copied().reduce(|best, vote| if credited(vote) < credited(best) { vote } else { best }).expect("at least one vote")
}

/// Audited votes need two to agree; a third breaks a tie.
pub fn decide(votes: &[Value], audit: bool, overdue: bool) -> Decision {
    if votes.len() == 1 {
        return if audit && !overdue { Decision::audit() } else { Decision::last(votes[0].clone(), votes.to_vec()) };
    }
    let mut tally: Vec<(String, usize)> = Vec::new();
    for vote in votes {
        let verdict = vote["verdict"].as_str().unwrap_or_default();
        match tally.iter_mut().find(|(v, _)| v == verdict) {
            Some((_, count)) => *count += 1,
            None => tally.push((verdict.into(), 1)),
        }
    }
    let (verdict, count) = tally.iter().fold(tally[0].clone(), |best, entry| if entry.1 > best.1 { entry.clone() } else { best });
    if count * 2 <= votes.len() {
        if votes.len() < 3 && !overdue {
            return Decision::audit();
        }
        let mut standing = votes.last().expect("votes").as_object().cloned().unwrap_or_default();
        standing.insert("verdict".into(), "void".into());
        standing.insert("credited".into(), 0.into());
        let mut result = standing.get("result").and_then(Value::as_object).cloned().unwrap_or_default();
        result.insert("verdict".into(), "void".into());
        result.insert("reason".into(), NO_MAJORITY.into());
        standing.insert("result".into(), Value::Object(result));
        return Decision::last(Value::Object(standing), votes.to_vec());
    }
    let mut winners: Vec<&Value> = votes.iter().filter(|v| v["verdict"] == verdict.as_str()).collect();
    let mut losers: Vec<&Value> = votes.iter().filter(|v| v["verdict"] != verdict.as_str()).collect();
    if credits_diverge(&winners) {
        if votes.len() < 3 {
            if !overdue {
                return Decision::audit();
            }
            return Decision::last(cheapest(&winners).clone(), votes.to_vec());
        }
        // The lower median, so two passes that differ finalize on the cheaper one.
        let mut paid: Vec<i64> = winners.iter().map(|v| credited(v)).collect();
        paid.sort_unstable();
        let middle = paid[(winners.len() - 1) / 2];
        let near: Vec<&Value> = winners.iter().copied().filter(|v| credits_agree(credited(v), middle)).collect();
        losers.extend(winners.iter().copied().filter(|v| !near.iter().any(|n| std::ptr::eq(*n, *v))));
        winners = near;
    }
    Decision {
        outcome: "final",
        vote: Some(cheapest(&winners).clone()),
        votes: votes.to_vec(),
        agreed: winners.iter().map(|v| v["validator"].as_str().unwrap_or_default().to_string()).collect(),
        disagreed: losers.iter().map(|v| v["validator"].as_str().unwrap_or_default().to_string()).collect(),
    }
}

/// A share as a whole percent, the way `f"{x:.0%}"` prints it.
fn percent(x: f64) -> i64 {
    py::round(x * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vote(validator: &str, verdict: &str, credited: i64) -> Value {
        json!({"validator": validator, "verdict": verdict, "credited": credited, "result": {"verdict": verdict, "reason": ""}})
    }

    #[test]
    fn majorities_and_credits_decide() {
        let one = decide(&[vote("a", "pass", 10)], false, false);
        assert_eq!((one.outcome, one.vote().clone()), ("final", vote("a", "pass", 10)));
        assert_eq!(decide(&[vote("a", "pass", 10)], true, false).outcome, "audit");
        assert_eq!(decide(&[vote("a", "pass", 10), vote("b", "fail", 0)], false, false).outcome, "audit");
        let split = decide(&[vote("a", "pass", 10), vote("b", "fail", 0)], false, true);
        assert_eq!((split.vote()["verdict"].as_str(), split.vote()["result"]["reason"].as_str()), (Some("void"), Some(NO_MAJORITY)));
        let three = decide(&[vote("a", "pass", 100), vote("b", "pass", 90), vote("c", "pass", 10)], false, false);
        assert_eq!(
            (three.vote()["validator"].as_str(), three.agreed.clone(), three.disagreed.clone()),
            (Some("b"), vec!["a".to_string(), "b".into()], vec!["c".to_string()])
        );
        let two = decide(&[vote("a", "pass", 100), vote("b", "pass", 50)], false, true);
        assert_eq!(two.vote()["validator"], "b");
    }

    #[test]
    fn a_pass_without_evidence_is_void() {
        let job = json!({"urls": ["https://a/1", "https://a/2"]});
        let result = json!({"returned": 2, "error_rows": 0, "verdict": "pass", "reason": "", "samples": [{"url": "https://a/1", "outcome": "unverifiable"}], "rejected": ["https://x/9"]});
        let built = build_vote(&job, "v", result.as_object().unwrap().clone()).unwrap();
        assert_eq!(
            (built["verdict"].as_str(), built["result"]["reason"].as_str(), built["result"]["rejected"].clone()),
            (Some("void"), Some("inconclusive"), json!([]))
        );
        let short = json!({"returned": 1, "error_rows": 0, "verdict": "pass", "samples": []});
        assert_eq!(build_vote(&job, "v", short.as_object().unwrap().clone()).unwrap_err().0, "a pass needs 85% of the URLs returned");
    }

    fn job() -> Value {
        json!({"urls": (0..20).map(|n| format!("https://site.example/{n}")).collect::<Vec<_>>()})
    }

    fn samples(outcomes: &[(&str, usize)]) -> Value {
        let mut urls = (0..20).map(|n| format!("https://site.example/{n}"));
        let list: Vec<Value> = outcomes
            .iter()
            .flat_map(|(outcome, count)| (0..*count).map(|_| (outcome.to_string(), urls.next().unwrap())).collect::<Vec<_>>())
            .map(|(outcome, url)| json!({"url": url, "outcome": outcome}))
            .collect();
        Value::Array(list)
    }

    fn built(result: Value) -> Result<Value, Infeasible> {
        let mut result = result.as_object().unwrap().clone();
        result.entry("verdict").or_insert_with(|| "pass".into());
        build_vote(&job(), "v", result)
    }

    struct Outcomes(Vec<(&'static str, i64)>);

    impl crate::credit::Outcomes for Outcomes {
        fn count(&self, outcome: &str) -> i64 {
            self.0.iter().filter(|(o, _)| *o == outcome).map(|(_, n)| n).sum()
        }
    }

    #[test]
    fn a_task_is_paid_at_its_samples_rate() {
        let cases = [
            (16, 4, vec![("matched", 3), ("errors_unconfirmed", 2)], 16),
            (16, 4, vec![("matched", 3), ("errors_confirmed", 2)], 20),
            (16, 4, vec![("matched", 3), ("errors_confirmed", 1), ("errors_unconfirmed", 1)], 18),
            (16, 4, vec![("matched", 3), ("not_fetched", 2)], 16),
            (20, 0, vec![("matched", 4), ("mismatched", 1)], 16),
            (0, 10, vec![("errors_confirmed", 5)], 10),
            (20, 0, vec![("unverifiable", 5)], 0),
        ];
        for (ok_rows, error_rows, outcomes, credited) in cases {
            assert_eq!(credited_urls(ok_rows, error_rows, &Outcomes(outcomes.clone())), credited, "{outcomes:?}");
        }
        for (matched, mismatched) in [(5, 0), (3, 2), (0, 5)] {
            assert!(credited_urls(15, 5, &Outcomes(vec![("matched", matched), ("mismatched", mismatched), ("errors_confirmed", 2)])) <= 20);
        }
    }

    #[test]
    fn credit_comes_from_the_samples_not_the_verdict_or_the_validator() {
        assert_eq!(built(json!({"verdict": "fail", "returned": 20, "samples": samples(&[("matched", 5)])})).unwrap()["credited"], 0);
        let void = built(json!({"returned": 20, "samples": samples(&[("unverifiable", 5)])})).unwrap();
        assert_eq!((void["verdict"].clone(), void["credited"].clone()), (json!("void"), json!(0)));
        assert_eq!(built(json!({"returned": 20, "credited": 999, "samples": samples(&[("matched", 4), ("mismatched", 1)])})).unwrap()["credited"], 16);
        let neither = built(json!({"returned": 20, "error_rows": 5, "samples": samples(&[("matched", 15), ("not_fetched", 5)])})).unwrap();
        assert_eq!(neither["verdict"], "pass");
        let own = built(json!({"verdict": "fail", "reason": "unscorable", "returned": 0, "samples": []})).unwrap();
        assert_eq!((own["verdict"].clone(), own["credited"].clone(), own["result"]["reason"].clone()), (json!("void"), json!(0), json!("unscorable")));
        let mostly = built(json!({"returned": 20, "samples": samples(&[("matched", 2), ("unverifiable", 3)])})).unwrap();
        assert_eq!((mostly["verdict"].clone(), mostly["result"]["reason"].clone()), (json!("void"), json!("unverifiable")));
        let still = built(json!({"returned": 20, "samples": samples(&[("matched", 3), ("unverifiable", 2)])})).unwrap();
        assert_eq!((still["verdict"].clone(), still["credited"].clone()), (json!("pass"), json!(20)));
    }

    #[test]
    fn a_report_the_task_could_not_have_produced_is_refused() {
        let once = samples(&[("matched", 1)]);
        let twice = Value::Array([once.as_array().unwrap().clone(), once.as_array().unwrap().clone()].concat());
        let cases = [
            (json!({"returned": 0, "samples": samples(&[("matched", 1)])}), "more samples than rows"),
            (json!({"returned": 21, "samples": samples(&[("matched", 1)])}), "more rows returned"),
            (json!({"returned": 16, "samples": samples(&[("matched", 2)])}), "85%"),
            (json!({"returned": 20, "samples": twice}), "sampled twice"),
            (json!({"returned": 20, "samples": [{"url": "https://elsewhere.example/", "outcome": "matched"}]}), "not in the task"),
            (json!({"returned": 20, "error_rows": 1, "samples": samples(&[("errors_confirmed", 2)])}), "more error samples"),
            (json!({"returned": 20, "error_rows": 5, "samples": samples(&[("matched", 16)])}), "more content samples"),
        ];
        for (result, why) in cases {
            let refused = built(result).unwrap_err().0;
            assert!(refused.contains(why), "{refused} does not say {why}");
        }
    }

    #[test]
    fn passes_that_disagree_on_the_pay_need_a_third_opinion() {
        let (honest, lowball) = (vote("a", "pass", 20), vote("b", "pass", 0));
        assert_eq!(decide(&[honest.clone(), lowball.clone()], false, false).outcome, "audit");
        let overdue = decide(&[honest.clone(), lowball], false, true);
        assert_eq!((overdue.vote()["validator"].clone(), overdue.agreed.len(), overdue.disagreed.len()), (json!("b"), 0, 0));
        let close = decide(&[honest, vote("b", "pass", 18)], false, false);
        assert_eq!((close.outcome, close.vote()["credited"].clone(), close.agreed), ("final", json!(18), vec!["a".to_string(), "b".into()]));
    }

    #[test]
    fn the_pay_two_of_three_agree_on_wins_and_the_odd_one_out_is_marked() {
        let last = decide(&[vote("a", "pass", 20), vote("b", "pass", 0), vote("c", "pass", 19)], false, false);
        assert_eq!((last.vote()["validator"].clone(), last.vote()["credited"].clone()), (json!("c"), json!(19)));
        assert_eq!((last.agreed, last.disagreed), (vec!["a".to_string(), "c".into()], vec!["b".to_string()]));
        let differ = decide(&[vote("stamp", "pass", 100), vote("careful", "pass", 70), vote("strict", "fail", 0)], false, false);
        assert_eq!((differ.outcome, differ.vote()["validator"].clone(), differ.vote()["credited"].clone()), ("final", json!("careful"), json!(70)));
        assert_eq!((differ.agreed, differ.disagreed), (vec!["careful".to_string()], vec!["strict".to_string(), "stamp".into()]));
    }

    #[test]
    fn an_embed_pass_must_account_for_every_text() {
        let job = json!({"texts": 6, "chars": 5400});
        let matched: Vec<Value> = (0..3).map(|i| json!({"text_id": format!("t{i}"), "outcome": "matched"})).collect();
        let whole = |extra: Value| {
            let mut result = json!({"verdict": "pass", "returned": 6, "matched": 3, "missing": 0, "samples": matched.clone()});
            for (name, value) in extra.as_object().unwrap() {
                result[name] = value.clone();
            }
            build_embed_vote(&job, "v", result.as_object().unwrap().clone())
        };
        assert_eq!(whole(json!({})).unwrap()["credited"], 5400);
        let mut four = matched.clone();
        four.push(matched[0].clone());
        for (broken, why) in [
            (json!({"returned": 0}), "more samples"),
            (json!({"missing": 6}), "every text embedded"),
            (json!({"malformed": 6}), "every text embedded"),
            (json!({"samples": four}), "sampled twice"),
        ] {
            let refused = whole(broken).unwrap_err().0;
            assert!(refused.contains(why), "{refused} does not say {why}");
        }
    }
}

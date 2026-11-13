//! ClinicalTrials.gov: a page per study from the registry's all-studies JSON zip, credited with the date the registry processed it, contacts left out.

use std::io::Read;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use regex::Regex;
use serde_json::Value;

pub const SOURCE: &str = "ctgov";
pub const SITE: &str = "ClinicalTrials.gov";
pub const STUDY: &str = "article";
const CONTENT_TYPE: &str = "application/json";
/// Locations named on a page at most; the rest are counted.
const MOST_LOCATIONS: usize = 40;
static EMAIL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+").expect("a valid pattern"));
static PHONE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\+\d{1,3}[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]\d{4}\b").expect("a valid pattern"));
/// A number after `+` or `00`, with at least eight digits: an international phone number.
static INTERNATIONAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:\+|\b00)\d[\d ().-]{6,}\d").expect("a valid pattern"));
/// A number of at least seven digits right after a word that names a phone.
static NAMED_PHONE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(tel(?:ephone)?|phone|call|fax|mobile|cell|ph)\b\.?\s*(?:no\.?|number)?\s*[:#]?\s*\(?\+?\d[\d ().-]{5,}\d").expect("a valid pattern"));

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// First study to take, by position in the zip's sorted names.
    #[arg(long, default_value_t = 0)]
    pub from: usize,
    /// Position to stop before; every study after `from` when 0.
    #[arg(long, default_value_t = 0)]
    pub to: usize,
    /// Lowest NCT number taken, as in NCT03000000; numbers never move, so refresh builds split on them.
    #[arg(long = "nct-from", default_value = "")]
    pub nct_from: String,
    /// NCT number to stop before; no upper bound when empty.
    #[arg(long = "nct-to", default_value = "")]
    pub nct_to: String,
    /// The all-studies zip from /api/v2/studies/download?format=json.zip.
    pub zip: PathBuf,
}

/// Free text with any email address or phone number taken out, as the registry's terms ask.
pub fn without_contacts(text: &str) -> String {
    let text = EMAIL.replace_all(text, "[email removed]");
    let text = NAMED_PHONE.replace_all(&text, |c: &regex::Captures| format!("{} [phone removed]", &c[1]));
    let text = INTERNATIONAL.replace_all(&text, |c: &regex::Captures| if c[0].chars().filter(char::is_ascii_digit).count() >= 8 { "[phone removed]".to_string() } else { c[0].to_string() });
    PHONE.replace_all(&text, "[phone removed]").into_owned()
}

/// A registry code as words: `ACTIVE_NOT_RECRUITING` as `Active not recruiting`, `PHASE2` as `Phase 2`.
pub fn words(code: &str) -> String {
    if let Some(n) = code.strip_prefix("EARLY_PHASE") {
        return format!("Early phase {n}");
    }
    if let Some(n) = code.strip_prefix("PHASE") {
        return format!("Phase {n}");
    }
    if code == "NA" {
        return "Not applicable".into();
    }
    let lower = code.replace('_', " ").to_lowercase();
    let mut chars = lower.chars();
    chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default()
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    path.iter().try_fold(v, |v, k| v.get(*k)).and_then(Value::as_str).unwrap_or("")
}

fn list_at<'a>(v: &'a Value, path: &[&str]) -> Vec<&'a Value> {
    path.iter().try_fold(v, |v, k| v.get(*k)).and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default()
}

fn strings_at(v: &Value, path: &[&str]) -> Vec<String> {
    list_at(v, path).into_iter().filter_map(Value::as_str).map(String::from).collect()
}

fn text_lines(text: &str) -> Vec<String> {
    without_contacts(text).lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|l| !l.is_empty()).collect()
}

fn dated(v: &Value, key: &str) -> String {
    let date = str_at(v, &[key, "date"]);
    match str_at(v, &[key, "type"]) {
        "" => date.to_string(),
        kind => format!("{date} ({})", words(kind).to_lowercase()),
    }
}

/// A study's page from its record, or None when it has no NCT number.
pub fn page_of(study: &Value) -> Option<(String, Page)> {
    let p = study.get("protocolSection")?;
    let nct = str_at(p, &["identificationModule", "nctId"]);
    if nct.is_empty() {
        return None;
    }
    let brief = str_at(p, &["identificationModule", "briefTitle"]);
    let official = str_at(p, &["identificationModule", "officialTitle"]);
    let status = &p["statusModule"];
    let design = &p["designModule"];
    let processed = str_at(study, &["derivedSection", "miscInfoModule", "versionHolder"]);
    let mut lines = vec![Line::new(format!(
        "Source: ClinicalTrials.gov, U.S. National Library of Medicine. Data processed by ClinicalTrials.gov on {processed}; this page is a condensed copy of the registry record, without contact details or posted results tables."
    ))];
    if !official.is_empty() && official != brief {
        lines.push(Line::new(format!("Official title: {}", without_contacts(official))));
    }
    let mut facts = vec![format!("NCT number: {nct}")];
    let enrollment = &design["enrollmentInfo"];
    for (label, value) in [
        ("Status", words(str_at(status, &["overallStatus"]))),
        ("Study type", words(str_at(design, &["studyType"]))),
        ("Phase", strings_at(design, &["phases"]).iter().map(|s| words(s)).collect::<Vec<_>>().join(", ")),
        (
            "Enrollment",
            enrollment.get("count").map(|c| format!("{c}{}", Some(str_at(enrollment, &["type"])).filter(|t| !t.is_empty()).map(|t| format!(" ({})", words(t).to_lowercase())).unwrap_or_default())).unwrap_or_default(),
        ),
        ("Start", dated(status, "startDateStruct")),
        ("Primary completion", dated(status, "primaryCompletionDateStruct")),
        ("Completion", dated(status, "completionDateStruct")),
        ("First posted", dated(status, "studyFirstPostDateStruct")),
        ("Last update posted", dated(status, "lastUpdatePostDateStruct")),
        ("Results first posted", dated(status, "resultsFirstPostDateStruct")),
    ] {
        if !value.is_empty() {
            facts.push(format!("{label}: {value}"));
        }
    }
    lines.push(Line::new(facts.join(". ") + "."));
    let sponsors = &p["sponsorCollaboratorsModule"];
    let lead = str_at(sponsors, &["leadSponsor", "name"]);
    if !lead.is_empty() {
        let collaborators: Vec<&str> = list_at(sponsors, &["collaborators"]).into_iter().filter_map(|c| c.get("name").and_then(Value::as_str)).collect();
        let mut line = format!("Sponsor: {lead}.");
        if !collaborators.is_empty() {
            line.push_str(&format!(" Collaborators: {}.", collaborators.join("; ")));
        }
        lines.push(Line::new(line));
    }
    for (label, path) in [("Conditions", ["conditionsModule", "conditions"]), ("Keywords", ["conditionsModule", "keywords"])] {
        let items = strings_at(p, &path);
        if !items.is_empty() {
            lines.push(Line::new(format!("{label}: {}.", items.join("; "))));
        }
    }
    for (heading, key) in [("Brief summary", "briefSummary"), ("Detailed description", "detailedDescription")] {
        let text = text_lines(str_at(p, &["descriptionModule", key]));
        if !text.is_empty() {
            lines.push(Line::new(format!("## {heading}")));
            lines.extend(text.into_iter().map(Line::new));
        }
    }
    let arms = list_at(p, &["armsInterventionsModule", "armGroups"]);
    let interventions = list_at(p, &["armsInterventionsModule", "interventions"]);
    if !arms.is_empty() || !interventions.is_empty() {
        lines.push(Line::new("## Arms and interventions"));
        for arm in arms {
            let kind = str_at(arm, &["type"]);
            let mut line = format!("Arm: {}", str_at(arm, &["label"]));
            if !kind.is_empty() {
                line.push_str(&format!(" ({})", words(kind).to_lowercase()));
            }
            let description = without_contacts(str_at(arm, &["description"]));
            if !description.is_empty() {
                line.push_str(&format!(": {}", description.split_whitespace().collect::<Vec<_>>().join(" ")));
            }
            lines.push(Line::new(line));
        }
        for i in interventions {
            let mut line = format!("Intervention: {} ({})", str_at(i, &["name"]), words(str_at(i, &["type"])).to_lowercase());
            let description = without_contacts(str_at(i, &["description"]));
            if !description.is_empty() {
                line.push_str(&format!(": {}", description.split_whitespace().collect::<Vec<_>>().join(" ")));
            }
            lines.push(Line::new(line));
        }
    }
    let outcomes = &p["outcomesModule"];
    let mut outcome_lines = Vec::new();
    for (label, key) in [("Primary outcome", "primaryOutcomes"), ("Secondary outcome", "secondaryOutcomes")] {
        for o in list_at(outcomes, &[key]) {
            let mut line = format!("{label}: {}", without_contacts(str_at(o, &["measure"])));
            let frame = str_at(o, &["timeFrame"]);
            if !frame.is_empty() {
                line.push_str(&format!(" (time frame: {frame})"));
            }
            outcome_lines.push(line);
        }
    }
    if !outcome_lines.is_empty() {
        lines.push(Line::new("## Outcomes"));
        lines.extend(outcome_lines.into_iter().map(Line::new));
    }
    let eligibility = &p["eligibilityModule"];
    let criteria = text_lines(str_at(eligibility, &["eligibilityCriteria"]));
    let ages = [str_at(eligibility, &["minimumAge"]), str_at(eligibility, &["maximumAge"])];
    if !criteria.is_empty() || !ages.iter().all(|a| a.is_empty()) {
        lines.push(Line::new("## Eligibility"));
        let mut facts = Vec::new();
        let sex = str_at(eligibility, &["sex"]);
        if !sex.is_empty() {
            facts.push(format!("Sex: {}", words(sex).to_lowercase()));
        }
        if !ages.iter().all(|a| a.is_empty()) {
            facts.push(format!("Ages: {} to {}", if ages[0].is_empty() { "any" } else { ages[0] }, if ages[1].is_empty() { "any" } else { ages[1] }));
        }
        if let Some(healthy) = eligibility.get("healthyVolunteers").and_then(Value::as_bool) {
            facts.push(format!("Healthy volunteers: {}", if healthy { "yes" } else { "no" }));
        }
        if !facts.is_empty() {
            lines.push(Line::new(facts.join(". ") + "."));
        }
        lines.extend(criteria.into_iter().map(Line::new));
    }
    let contacts = &p["contactsLocationsModule"];
    let officials: Vec<String> = list_at(contacts, &["overallOfficials"])
        .into_iter()
        .map(|o| [str_at(o, &["name"]), str_at(o, &["affiliation"]), &words(str_at(o, &["role"])).to_lowercase()].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", "))
        .filter(|s| !s.is_empty())
        .collect();
    let locations: Vec<String> = list_at(contacts, &["locations"])
        .into_iter()
        .map(|l| ["facility", "city", "state", "country"].iter().map(|k| str_at(l, &[k])).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", "))
        .filter(|s| !s.is_empty())
        .collect();
    if !officials.is_empty() || !locations.is_empty() {
        lines.push(Line::new("## Investigators and locations"));
        lines.extend(officials.into_iter().map(|o| Line::new(format!("Official: {}", without_contacts(&o)))));
        let more = locations.len().saturating_sub(MOST_LOCATIONS);
        lines.extend(locations.into_iter().take(MOST_LOCATIONS).map(|l| Line::new(format!("Location: {}", without_contacts(&l)))));
        if more > 0 {
            lines.push(Line::new(format!("And {more} more locations.")));
        }
    }
    let url = format!("https://clinicaltrials.gov/study/{nct}");
    let meta = PaperMeta {
        lang: "en".into(),
        url: url.clone(),
        published: str_at(status, &["studyFirstPostDateStruct", "date"]).to_string(),
        modified: str_at(status, &["lastUpdatePostDateStruct", "date"]).to_string(),
        site_name: SITE.into(),
        tags: strings_at(p, &["conditionsModule", "conditions"]),
        ..PaperMeta::default()
    };
    let title = if brief.is_empty() { nct.to_string() } else { format!("{} ({nct})", without_contacts(brief)) };
    let lines = lines.into_iter().map(|line| Line::new(without_contacts(&line.text))).collect();
    Some((url, typed_page(STUDY, &title, lines, meta)))
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let file = std::fs::File::open(&args.zip).with_context(|| format!("opening {}", args.zip.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut names: Vec<String> = archive.file_names().filter(|n| n.ends_with(".json")).map(String::from).collect();
    names.sort();
    let to = if args.to == 0 { names.len() } else { args.to.min(names.len()) };
    let mut counts = Counts::default();
    let in_range = |name: &str| {
        let nct = name.rsplit('/').next().unwrap_or(name).trim_end_matches(".json");
        (args.nct_from.is_empty() || nct >= args.nct_from.as_str()) && (args.nct_to.is_empty() || nct < args.nct_to.as_str())
    };
    for name in names.get(args.from..to).unwrap_or(&[]).iter().filter(|n| in_range(n)) {
        let mut body = String::new();
        archive.by_name(name)?.read_to_string(&mut body)?;
        let study: Value = serde_json::from_str(&body).with_context(|| format!("reading {name}"))?;
        match page_of(&study) {
            Some((url, page)) => {
                batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
                counts.add("pages", 1);
            }
            None => counts.add("no_nct", 1),
        }
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            break;
        }
    }
    counts.set("studies_in_zip", names.len() as u64);
    Ok(counts)
}

pub fn main(args: Args) -> Result<()> {
    let (mut batches, as_of_us) = bulk_core::run::start(&args.run, SOURCE)?;
    let counts = run(&args, &mut batches, as_of_us)?;
    bulk_core::run::finish(&args.run, batches, &counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn study() -> Value {
        json!({
            "protocolSection": {
                "identificationModule": {"nctId": "NCT02068066", "briefTitle": "Physical Workload", "officialTitle": "Physical Workload and Work Capacity"},
                "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2013-05"}, "completionDateStruct": {"date": "2015-02", "type": "ACTUAL"},
                    "studyFirstPostDateStruct": {"date": "2014-02-20", "type": "ESTIMATED"}, "lastUpdatePostDateStruct": {"date": "2015-11-17", "type": "ESTIMATED"}},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Cantonal Hospital"}, "collaborators": [{"name": "Swiss Accident Insurance"}]},
                "descriptionModule": {"briefSummary": "Workers were studied. Contact jane@example.org or 555-123-4567."},
                "conditionsModule": {"conditions": ["Work Load"]},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2", "PHASE3"], "enrollmentInfo": {"count": 337, "type": "ACTUAL"}},
                "outcomesModule": {"primaryOutcomes": [{"measure": "Energy expenditure", "timeFrame": "7 days"}]},
                "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n\n* Healthy", "sex": "ALL", "minimumAge": "18 Years", "maximumAge": "65 Years", "healthyVolunteers": true},
                "contactsLocationsModule": {"centralContacts": [{"name": "J", "email": "j@x.org", "phone": "555"}], "overallOfficials": [{"name": "Trial Manager", "affiliation": "trials-NL@wyeth.com", "role": "PRINCIPAL_INVESTIGATOR"}],
                    "locations": [{"facility": "Hospital", "city": "Liestal", "country": "Switzerland"}]}
            },
            "derivedSection": {"miscInfoModule": {"versionHolder": "2026-11-12"}},
            "hasResults": false
        })
    }

    #[test]
    fn study_page_with_facts_and_no_contacts() {
        let (url, page) = page_of(&study()).unwrap();
        assert_eq!(url, "https://clinicaltrials.gov/study/NCT02068066");
        assert_eq!(page.title, "Physical Workload (NCT02068066)");
        let text = page.text;
        assert!(text.contains("Data processed by ClinicalTrials.gov on 2026-11-12"));
        assert!(text.contains("Status: Completed. Study type: Interventional. Phase: Phase 2, Phase 3. Enrollment: 337 (actual). Start: 2013-05. Completion: 2015-02 (actual)."));
        assert!(text.contains("Sponsor: Cantonal Hospital. Collaborators: Swiss Accident Insurance."));
        assert!(text.contains("Workers were studied. Contact [email removed] or [phone removed]."));
        assert!(text.contains("Primary outcome: Energy expenditure (time frame: 7 days)"));
        assert!(text.contains("Sex: all. Ages: 18 Years to 65 Years. Healthy volunteers: yes."));
        assert!(text.contains("Location: Hospital, Liestal, Switzerland"));
        assert!(!text.contains("j@x.org"));
        assert!(text.contains("Official: Trial Manager, [email removed], principal investigator"));
        assert_eq!((page.published.as_str(), page.modified.as_str()), ("2014-02-20", "2015-11-17"));
    }

    #[test]
    fn codes_as_words() {
        assert_eq!(words("ACTIVE_NOT_RECRUITING"), "Active not recruiting");
        assert_eq!(words("EARLY_PHASE1"), "Early phase 1");
        assert_eq!(words("NA"), "Not applicable");
    }

    #[test]
    fn contacts_out_dates_and_ids_kept() {
        assert_eq!(without_contacts("call (555) 123-4567 on 2013-05-01, NCT02068066"), "call [phone removed] on 2013-05-01, NCT02068066");
        assert_eq!(without_contacts("Contact +44 20 7946 0958 or +33 (0)1 42 16 00 00."), "Contact [phone removed] or [phone removed].");
        assert_eq!(without_contacts("Tel: 0041 61 265 25 25, phone 5551234567, (555)123-4567"), "Tel [phone removed], phone [phone removed], [phone removed]");
        assert_eq!(without_contacts("Doses of 100 200 300 mg in 2015 2016; 25 participants"), "Doses of 100 200 300 mg in 2015 2016; 25 participants");
    }
}

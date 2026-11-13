//! The World Factbook's final edition (public domain; factbook.json, CC0): a page per entity, linked to its page on worldfactbookarchive.org.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta, RunArgs};
use regex::Regex;
use serde_json::{Map, Value};

pub const SOURCE: &str = "factbook";
pub const SITE: &str = "The World Factbook";
pub const ARTICLE: &str = "article";
pub const ARCHIVE: &str = "https://worldfactbookarchive.org/archive";
/// The final edition's year on the archive, and the day of its last data update.
pub const EDITION: &str = "2025";
pub const LAST_UPDATE: &str = "2026-01-22";
const CREDIT: &str = "Source: The World Factbook, Central Intelligence Agency (public domain), final edition. Its data runs to January 2026; the CIA discontinued the Factbook in February 2026.";
const CONTENT_TYPE: &str = "application/json";
const SOVEREIGN: &str = "sovereign";

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    #[command(flatten)]
    pub run: RunArgs,
    /// The factbook.json checkout, its region folders of <code>.json files.
    #[arg(long)]
    pub factbook: PathBuf,
    /// The archive's master_countries.csv: CIA code, name, ISO alpha-2 and entity type.
    #[arg(long)]
    pub countries: PathBuf,
    /// The archive's sitemap-archive.xml, the pages it publishes.
    #[arg(long)]
    pub sitemap: PathBuf,
}

/// One entity of the archive's country table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entity {
    pub cia: String,
    pub name: String,
    pub iso2: String,
    pub kind: String,
}

pub fn entities(csv_text: &str) -> Result<HashMap<String, Entity>> {
    let mut reader = csv::Reader::from_reader(csv_text.trim_start_matches('\u{feff}').as_bytes());
    let mut found = HashMap::new();
    for row in reader.deserialize::<HashMap<String, String>>() {
        let row = row?;
        let get = |k: &str| row.get(k).map(|v| v.trim().to_string()).unwrap_or_default();
        let cia = get("CanonicalCode").to_uppercase();
        found.insert(cia.clone(), Entity { cia, name: get("CanonicalName"), iso2: get("ISOAlpha2"), kind: get("EntityType") });
    }
    Ok(found)
}

/// The archive's page code for each CIA code taken: the ISO alpha-2 when no other entity taken shares it or the entity is
/// sovereign, else the CIA code; a code two entities share goes to the one whose ISO code it is, as the archive shows.
pub fn page_codes(taken: &[&Entity], published: &HashSet<String>) -> (BTreeMap<String, String>, Vec<String>) {
    let mut shared: HashMap<&str, usize> = HashMap::new();
    for e in taken.iter().filter(|e| !e.iso2.is_empty()) {
        *shared.entry(e.iso2.as_str()).or_default() += 1;
    }
    let code_of = |e: &Entity| if !e.iso2.is_empty() && (shared[e.iso2.as_str()] == 1 || e.kind == SOVEREIGN) { e.iso2.clone() } else { e.cia.clone() };
    let mut by_code: HashMap<String, Vec<&Entity>> = HashMap::new();
    for e in taken {
        by_code.entry(code_of(e)).or_default().push(e);
    }
    let (mut codes, mut left_out) = (BTreeMap::new(), Vec::new());
    for (code, holders) in by_code {
        let owner = if holders.len() == 1 { Some(holders[0]) } else { holders.iter().copied().find(|e| e.iso2 == code) };
        for e in &holders {
            if published.contains(&code) && owner.is_some_and(|o| o.cia == e.cia) {
                codes.insert(e.cia.clone(), code.clone());
            } else {
                left_out.push(e.cia.clone());
            }
        }
    }
    left_out.sort();
    (codes, left_out)
}

/// The final edition's page codes the sitemap lists.
pub fn sitemap_codes(xml: &str) -> HashSet<String> {
    let pattern = Regex::new(&format!(r"/archive/{EDITION}/([A-Z0-9]+)<")).expect("a valid pattern");
    pattern.captures_iter(xml).map(|c| c[1].to_string()).collect()
}

/// A Factbook value's text as lines: tags out, entities decoded, `<br>` and `<p>` as line breaks.
pub fn plain(markup: &str) -> Vec<String> {
    let breaks = Regex::new(r"(?i)<\s*(br|/?p)\b[^>]*>").expect("a valid pattern");
    let tags = Regex::new(r"<[^>]*>").expect("a valid pattern");
    let broken = breaks.replace_all(markup, "\n");
    let bare = tags.replace_all(&broken, "");
    html_escape::decode_html_entities(&bare)
        .split('\n')
        .map(|line| line.replace('\u{a0}', " ").split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect()
}

fn text_of(value: &Value) -> Vec<String> {
    match value {
        Value::String(s) => plain(s),
        Value::Object(o) => o.get("text").and_then(Value::as_str).map(plain).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// A field's lines: its own text after its name, then each part as `name: text`.
fn field_lines(name: &str, value: &Value) -> Vec<String> {
    let own = text_of(value);
    let mut parts: Vec<String> = Vec::new();
    if let Value::Object(o) = value {
        for (key, part) in o.iter().filter(|(k, _)| k.as_str() != "text") {
            let text = text_of(part).join(" ");
            if !text.is_empty() {
                let key = key.trim();
                parts.push(if key == "note" || text.to_lowercase().starts_with(&format!("{}:", key.to_lowercase())) { text } else { format!("{key}: {text}") });
            }
        }
    }
    let name = name.trim();
    match (own.split_first(), parts.is_empty()) {
        (None, true) => Vec::new(),
        (None, false) => vec![format!("{name}: {}", parts.join("; "))],
        (Some((first, rest)), _) => {
            let mut lines = vec![format!("{name}: {first}")];
            lines.extend(rest.iter().cloned());
            if !parts.is_empty() {
                lines.push(parts.join("; "));
            }
            lines
        }
    }
}

/// The entity's name: the Factbook's conventional short form, its long form when there is none, else the archive's.
pub fn name_of(profile: &Map<String, Value>, fallback: &str) -> String {
    let form = |which: &str| {
        profile.get("Government").and_then(|g| g.get("Country name")).and_then(|c| c.get(which)).map(text_of).unwrap_or_default().join(" ")
    };
    [form("conventional short form"), form("conventional long form")]
        .into_iter()
        .find(|n| !n.is_empty() && n.to_lowercase() != "none")
        .unwrap_or_else(|| fallback.to_string())
}

pub fn page_of(profile: &Map<String, Value>, name: &str, url: &str) -> Page {
    let mut lines = vec![Line::new(CREDIT)];
    for (section, fields) in profile {
        let Value::Object(fields) = fields else { continue };
        let body: Vec<String> = fields.iter().flat_map(|(field, value)| field_lines(field, value)).collect();
        if !body.is_empty() {
            lines.push(Line::new(format!("## {}", section.trim())));
            lines.extend(body.into_iter().map(Line::new));
        }
    }
    let meta = PaperMeta {
        lang: "en".into(),
        url: url.to_string(),
        published: LAST_UPDATE.into(),
        modified: LAST_UPDATE.into(),
        site_name: SITE.into(),
        ..PaperMeta::default()
    };
    typed_page(ARTICLE, &format!("{name} - The World Factbook"), lines, meta)
}

fn profiles(folder: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    for region in std::fs::read_dir(folder).with_context(|| format!("reading {}", folder.display()))? {
        let region = region?.path();
        if !region.is_dir() || region.file_name().is_some_and(|n| n == "meta" || n.to_string_lossy().starts_with('.')) {
            continue;
        }
        for file in std::fs::read_dir(&region)? {
            let file = file?.path();
            if file.extension().is_some_and(|e| e == "json") {
                let code = file.file_stem().map(|s| s.to_string_lossy().to_uppercase()).unwrap_or_default();
                found.push((code, file));
            }
        }
    }
    found.sort();
    Ok(found)
}

pub fn run(args: &Args, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let table = entities(&std::fs::read_to_string(&args.countries)?)?;
    let published = sitemap_codes(&std::fs::read_to_string(&args.sitemap)?);
    let files = profiles(&args.factbook)?;
    let mut counts = Counts::default();
    let taken: Vec<&Entity> = files.iter().filter_map(|(code, _)| table.get(code)).collect();
    if taken.len() != files.len() {
        counts.add("not_in_table", (files.len() - taken.len()) as u64);
    }
    let (codes, left_out) = page_codes(&taken, &published);
    counts.add("no_page_of_its_own", left_out.len() as u64);
    for (cia, path) in &files {
        let Some(code) = codes.get(cia) else { continue };
        let profile: Map<String, Value> = match serde_json::from_str(&std::fs::read_to_string(path)?)? {
            Value::Object(o) => o,
            _ => bail!("{} is not a JSON object", path.display()),
        };
        let url = format!("{ARCHIVE}/{EDITION}/{code}");
        let page = page_of(&profile, &name_of(&profile, &table[cia].name), &url);
        batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
        counts.add("pages", 1);
        if args.run.limit > 0 && counts.get("pages") >= args.run.limit {
            break;
        }
    }
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

    fn entity(cia: &str, iso2: &str, kind: &str) -> Entity {
        Entity { cia: cia.into(), name: cia.into(), iso2: iso2.into(), kind: kind.into() }
    }

    #[test]
    fn codes_follow_the_archive_and_a_shared_code_goes_to_its_iso_owner() {
        let all = [entity("AU", "AT", SOVEREIGN), entity("AT", "AU", "territory"), entity("AS", "AU", SOVEREIGN), entity("GM", "DE", SOVEREIGN), entity("WI", "EH", "disputed")];
        let refs: Vec<&Entity> = all.iter().collect();
        let published: HashSet<String> = ["AT", "AU", "DE"].map(String::from).into();
        let (codes, left_out) = page_codes(&refs, &published);
        assert_eq!(codes, BTreeMap::from([("AU".into(), "AT".into()), ("AS".into(), "AU".into()), ("GM".into(), "DE".into())]));
        assert_eq!(left_out, ["AT", "WI"]);
    }

    #[test]
    fn markup_becomes_plain_lines() {
        assert_eq!(plain("Caf&eacute; &amp; <strong>bar</strong><br><br>second&nbsp;line"), ["Café & bar", "second line"]);
        assert_eq!(plain("<b>note:</b> data in 2021 dollars"), ["note: data in 2021 dollars"]);
    }

    #[test]
    fn page_lists_sections_and_fields() {
        let profile = json!({
            "Introduction": {"Background": {"text": "First.<br><br>Second."}},
            "Geography": {"Area": {"total ": {"text": "357,022 sq km"}, "land": {"text": "348,672 sq km"}, "note": "<b>note:</b> includes lakes"}},
            "Government": {"Country name": {"conventional short form": {"text": "Germany"}}},
        });
        let profile = profile.as_object().unwrap();
        assert_eq!(name_of(profile, "x"), "Germany");
        let page = page_of(profile, "Germany", "https://worldfactbookarchive.org/archive/2025/DE");
        assert_eq!(
            page.text.lines().collect::<Vec<_>>(),
            [
                "Germany - The World Factbook",
                CREDIT,
                "## Introduction",
                "Background: First.",
                "Second.",
                "## Geography",
                "Area: total: 357,022 sq km; land: 348,672 sq km; note: includes lakes",
                "## Government",
                "Country name: conventional short form: Germany",
            ]
        );
        assert_eq!((page.published.as_str(), page.canonical.as_str()), (LAST_UPDATE, "https://worldfactbookarchive.org/archive/2025/DE"));
    }

    #[test]
    fn sitemap_codes_of_the_final_edition_only() {
        let xml = "<loc>https://worldfactbookarchive.org/archive/2025/DE</loc><loc>https://worldfactbookarchive.org/archive/2024/FR</loc>";
        assert_eq!(sitemap_codes(xml), HashSet::from(["DE".to_string()]));
    }
}

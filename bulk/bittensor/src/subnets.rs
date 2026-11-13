//! `subnet_page` and `run_subnets`: a page per subnet from the chain JSON alone.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::Result;
use bulk_core::py::{self, truthy, when_truthy};
use bulk_core::{typed_page, upload_row, Batches, Counts, Line, Page, PaperMeta};
use desearch::text::{head, strip};
use regex::Regex;
use serde_json::{Map, Value};

use crate::COMPANY;

pub const SUBNETS_SOURCE: &str = "subnets";
pub const SITE: &str = "Taostats";
pub const CONTENT_TYPE: &str = "application/json";
/// Identities that name no project, such as a subnet for sale or a claim about a person, give only chain facts.
pub const NO_IDENTITY: [i64; 1] = [35];
pub const LINKS: [(&str, &str); 5] =
    [("subnet_url", "Website"), ("github_repo", "GitHub repository"), ("discord", "Discord"), ("subnet_contact", "Contact"), ("logo_url", "Logo")];
pub const OWNER_FIELDS: [(&str, &str); 4] =
    [("url", "Owner website"), ("github", "Owner GitHub"), ("discord", "Owner Discord"), ("description", "Owner description")];

static PLACEHOLDER_OWNER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^Owner\d+\n?\z").expect("pattern compiles"));
// Python's `\b404\b.{0,20}`: a non-word character before and after 404, the one after taken by `.{0,20}`.
static ERROR_TEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)page not found|(?:\A|(?-i:[^\p{L}\p{N}_]))404(?-i:[^\p{L}\p{N}_\n]).{0,19}not found").expect("pattern compiles"));

/// `url_of`.
pub fn url_of(netuid: &Value) -> Result<String> {
    Ok(format!("https://taostats.io/subnets/{}", py::format(netuid)?))
}

fn short_date(moment: Option<&Value>) -> Result<String> {
    Ok(match when_truthy(moment) {
        Some(m) => head(py::str_of(m, "a date")?, 10).to_string(),
        None => String::new(),
    })
}

fn percent(share: Option<&Value>) -> Result<String> {
    let Some(share) = share.filter(|s| !s.is_null()) else {
        return Ok(String::new());
    };
    let share = py::float(share)?;
    Ok(if share >= 0.0001 { format!("{:.2}%", share * 100.0) } else { "under 0.01%".into() })
}

fn no_identity(netuid: &Value) -> bool {
    NO_IDENTITY.iter().any(|&n| py::equals_int(netuid, n))
}

fn stripped(map: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    when_truthy(map.get(key)).map(|v| py::str_of(v, key).map(|s| strip(s).to_string())).transpose()
}

fn is_error_text(value: &Value) -> bool {
    value.as_str().is_some_and(|s| ERROR_TEXT.is_match(s))
}

/// `subnet_page`: the subnet as the chain holds it: identity, owner and chain facts, nothing from other sources.
pub fn subnet_page(subnet: &Value, block: &Value, taken: &Value) -> Result<(String, Page)> {
    let netuid = py::item(subnet, "netuid")?;
    let n = py::format(netuid)?;
    let skip_identity = no_identity(netuid);
    let identity = if skip_identity { Default::default() } else { py::dict_or_empty(py::get(subnet, "identity")?, "identity")? };
    let mut name = match py::or(identity.get("subnet_name"), py::get(subnet, "name")?) {
        Some(v) => strip(py::str_of(v, "the subnet name")?).to_string(),
        None => String::new(),
    };
    if skip_identity {
        name.clear();
    }
    let title = if name.is_empty() { format!("Bittensor Subnet {n}") } else { format!("Bittensor Subnet {n}: {name}") };
    let called = if name.is_empty() { format!("Subnet {n}") } else { format!("Subnet {n} ({name})") };
    let symbol = match when_truthy(py::get(subnet, "symbol")?) {
        Some(symbol) => format!(", with the token symbol {}.", py::format(symbol)?),
        None => ".".into(),
    };
    let mut lines = vec![Line::new(format!("{called} is a subnet of the Bittensor network, netuid {n}, also written SN{n}{symbol}"))];
    if py::equals_int(netuid, 0) {
        lines.push(Line::new("Subnet 0 is the root network of Bittensor."));
    }
    let identity: Map<String, Value> = identity.iter().filter(|(_, v)| !is_error_text(v)).map(|(k, v)| (k.clone(), v.clone())).collect();
    let named = if name.is_empty() { called.as_str() } else { name.as_str() };
    if let Some(description) = stripped(&identity, "description")?.filter(|d| !d.is_empty()) {
        lines.push(Line::new(format!("{named} describes itself on chain as: {description}")));
    }
    if let Some(additional) = stripped(&identity, "additional")?.filter(|a| !a.is_empty()) {
        lines.push(Line::new(format!("Additional information on chain: {additional}")));
    }
    let mut links = Vec::new();
    for (key, label) in LINKS {
        if let Some(value) = stripped(&identity, key)? {
            links.push(Line::new(format!("{label}: {value}")));
        }
    }
    if !links.is_empty() {
        lines.push(Line::new(format!("## {named} links")));
        lines.extend(links);
    }
    lines.push(Line::new("## Owner"));
    lines.push(Line::new(format!("Owner hotkey: {}", py::format(py::item(subnet, "owner_hotkey")?)?)));
    lines.push(Line::new(format!("Owner coldkey: {}", py::format(py::item(subnet, "owner_coldkey")?)?)));
    let owner = py::dict_or_empty(py::get(subnet, "owner_identity")?, "owner_identity")?;
    if let Some(owner_name) = when_truthy(owner.get("name")) {
        let owner_name = py::str_of(owner_name, "the owner name")?;
        if !PLACEHOLDER_OWNER.is_match(owner_name) {
            lines.push(Line::new(format!("Owner identity on chain: {owner_name}")));
        }
    }
    for (field, label) in OWNER_FIELDS {
        if let Some(value) = when_truthy(owner.get(field)) {
            lines.push(Line::new(format!("{label}: {}", py::format(value)?)));
        }
    }
    lines.push(Line::new("## Chain facts"));
    if let Some(registered) = when_truthy(py::get(subnet, "registered_block")?) {
        let on = short_date(py::get(subnet, "registered_at")?)?;
        let on = if on.is_empty() { ".".to_string() } else { format!(" on {on}.") };
        lines.push(Line::new(format!("{called} was registered at block {}{on}", py::grouped(registered)?)));
    }
    lines.push(Line::new(format!("Tempo: {} blocks between epochs.", py::format(py::item(subnet, "tempo")?)?)));
    if let Some(neurons) = py::get(subnet, "neurons")?.filter(|v| !v.is_null()) {
        let of = match when_truthy(py::get(subnet, "max_neurons")?) {
            Some(most) => format!(" of {} allowed", py::format(most)?),
            None => String::new(),
        };
        lines.push(Line::new(format!("Registered neurons (UIDs): {}{of}.", py::format(neurons)?)));
    }
    let share = percent(py::get(subnet, "emission_share")?)?;
    if !share.is_empty() && !py::equals_int(netuid, 0) {
        let taken_on = short_date(Some(taken))?;
        lines.push(Line::new(format!("Share of TAO emission into subnets: {share}, at block {} on {taken_on}.", py::grouped(block)?)));
    }
    let url = url_of(netuid)?;
    let mut tags = vec!["Bittensor".to_string(), "subnet".to_string(), format!("SN{n}")];
    if !name.is_empty() {
        tags.push(name.clone());
    }
    let meta = PaperMeta { lang: "en".into(), url: url.clone(), modified: short_date(Some(taken))?, site_name: SITE.into(), tags, ..PaperMeta::default() };
    Ok((url, typed_page(COMPANY, &title, lines, meta)))
}

/// `run_subnets`: a page per subnet of the chain JSON, and its gone subnets named removed.
pub fn run_subnets(chain: &Path, batches: &mut Batches, as_of_us: i64) -> Result<Counts> {
    let mut counts = Counts::default();
    let data = py::load_json(chain)?;
    for subnet in py::list(py::item(&data, "subnets")?, "subnets")? {
        let (url, page) = subnet_page(subnet, py::item(&data, "block")?, py::item(&data, "taken_at")?)?;
        batches.add(upload_row(&url, page, as_of_us, CONTENT_TYPE))?;
        counts.add("pages", 1);
        let identity = truthy(py::get(subnet, "identity")?) && !no_identity(py::item(subnet, "netuid")?);
        counts.add("with_identity", u64::from(identity));
        counts.add("with_date", u64::from(truthy(py::get(subnet, "registered_at")?)));
    }
    if let Some(removed) = py::get(&data, "removed_netuids")? {
        for netuid in py::list(removed, "removed_netuids")? {
            batches.remove(&url_of(netuid)?);
            counts.add("removed", 1);
        }
    }
    Ok(counts)
}

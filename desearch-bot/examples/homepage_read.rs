//! Reads saved homepage bodies as the bot's eligibility check does: `homepage_read DATA_DIR LIMIT FILE...`.

use std::path::PathBuf;

use anyhow::Result;
use desearch_bot::homepage;
use desearch_bot::langid::LangId;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = LangId::load(&PathBuf::from(&args[1]).join("langid.bin"))?;
    let limit: usize = args[2].parse()?;
    let detect = |text: &str| text.chars().nth(19).map(|_| model.classify(text).to_string());
    for path in &args[3..] {
        let body = std::fs::read(path)?;
        let cut = &body[..body.len().min(limit)];
        let page = homepage::read(cut, &detect);
        println!("{path}\tbytes {}\tchars {}\tdeclared {:?}\tlang {:?}\tproblem {:?}", body.len(), page.chars, page.declared, page.language, page.problem);
    }
    Ok(())
}

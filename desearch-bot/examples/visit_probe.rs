//! Visits listed hosts once each as the bot does, on a scratch store: `visit_probe HOSTS OUT DATA_DIR CONCURRENCY SITEMAP_SECONDS DEADLINE_SECONDS [answered] [local]`.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use desearch_bot::allowed::Allowed;
use desearch_bot::buckets::{bucket_of, Buckets, Changes, Resources};
use desearch_bot::crawl::Loop;
use desearch_bot::langid::LangId;
use desearch_bot::net::{self, PublicResolver};
use desearch_bot::records;
use desearch_bot::registry::Registry;
use desearch_bot::signing::USER_AGENT;
use desearch_bot::states::State;
use desearch_bot::suffixes::PublicSuffixList;
use desearch_bot::visit::{Slots, Visitor, MIN_HOST_INTERVAL};

const FIELDS: [&str; 10] = ["state", "reason", "robots_status", "home_status", "chars", "declared", "lang", "canonical", "urls", "failures"];

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hosts: Vec<String> = std::fs::read_to_string(&args[1])?.lines().map(str::trim).filter(|h| !h.is_empty()).map(String::from).collect();
    let out = PathBuf::from(&args[2]);
    let data = PathBuf::from(&args[3]);
    let concurrency: usize = args[4].parse()?;
    let sitemap_seconds: u64 = args[5].parse()?;
    let deadline_seconds: u64 = args[6].parse()?;
    let answered = args[7..].iter().any(|a| a == "answered");
    let local = args[7..].iter().any(|a| a == "local");
    let root = out.with_extension("store");
    let _ = std::fs::remove_dir_all(&root);
    let mut owned: Vec<usize> = hosts.iter().map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let buckets = Arc::new(Buckets::open(&root, &owned, &Resources::new(256 << 20, 2 << 30, owned.len()))?);
    buckets.allow(&Allowed::of(hosts.clone()));
    for host in &hosts {
        let mut changes = Changes::default();
        let mut record = records::new_domain(Some(1), Some("new_generic"), &[], State::New, None, Some(0));
        // As a registry recheck leaves it: set back to new, its last good answer kept.
        if answered {
            record.insert("ok".into(), (chrono::Utc::now().timestamp() - 86_400).into());
        }
        changes.domain(host, &record);
        buckets.store(host).write(changes)?;
    }
    let read_timeout = Duration::from_secs(10);
    let client = if local {
        net::client(PublicResolver::local(), read_timeout)?
    } else {
        reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(net::MAX_REQUEST)
            .connect_timeout(net::connect_timeout(read_timeout))
            .read_timeout(read_timeout)
            .pool_idle_timeout(Duration::from_secs(15))
            .pool_max_idle_per_host(4)
            .http1_only()
            .tcp_nodelay(true)
            .build()?
    };
    let model = LangId::load(&data.join("langid.bin"))?;
    // Sitemap reads stop after this long, as when the bot's disk runs low; robots.txt and the homepage are read as ever.
    let pause = Arc::new(AtomicBool::new(false));
    let pausing = pause.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(sitemap_seconds)).await;
        pausing.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let visitor = Visitor {
        client,
        buckets: buckets.clone(),
        suffixes: Arc::new(PublicSuffixList::load(&data.join("public_suffix_list.dat"))?),
        signer: None,
        language: Arc::new(move |text: &str| text.chars().nth(19).map(|_| model.classify(text).to_string())),
        floor: MIN_HOST_INTERVAL,
        connect_timeout: net::connect_timeout(read_timeout),
        cpu: Slots::new(8),
        bodies: Slots::new(16),
        pause: pause.clone(),
        heavy: Slots::new(4),
    };
    let mut crawl = Loop::new(buckets.clone(), Arc::new(visitor), concurrency, Registry::offline(), HashSet::new());
    crawl.load()?;
    // Visits still going at the deadline stay new with no reason, and are written as unfinished; a failed recheck waiting for its retry is written as retry.
    let _ = tokio::time::timeout(Duration::from_secs(deadline_seconds), async {
        while crawl.step().await? > 0 {}
        anyhow::Ok(())
    })
    .await;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&out)?);
    writeln!(writer, "host\t{}", FIELDS.join("\t"))?;
    for host in &hosts {
        let record = buckets.store(host).domain(host)?.unwrap_or_default();
        let field = |name: &str| record.get(name).map(|v| v.as_str().map_or_else(|| v.to_string(), String::from)).unwrap_or_default().replace('\t', " ");
        let state = match (field("state").as_str(), field("reason").as_str()) {
            ("new", "" | "null") => "unfinished".to_string(),
            ("new", _) => "retry".to_string(),
            (state, _) => state.to_string(),
        };
        writeln!(writer, "{host}\t{state}\t{}", FIELDS[1..].iter().map(|name| field(name)).collect::<Vec<_>>().join("\t"))?;
    }
    writer.flush()?;
    let _ = std::fs::remove_dir_all(&root);
    std::process::exit(0)
}

//! The crawl-sites replay server on its own: `site_replay <fixture.json> <port> <log.jsonl> <ca.pem>`, serving until killed.

#[allow(dead_code)]
#[path = "../tests/support/site_replay.rs"]
mod site_replay;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, fixture, port, log, ca] = &args[..] else {
        eprintln!("usage: site_replay <fixture.json> <port> <log.jsonl> <ca.pem>");
        std::process::exit(2);
    };
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let replay = site_replay::start(&fixture, port.parse().unwrap(), Some(log.into())).await;
    std::fs::write(ca, &replay.ca_pem).unwrap();
    println!("listening {}", replay.addr);
    std::future::pending::<()>().await;
}

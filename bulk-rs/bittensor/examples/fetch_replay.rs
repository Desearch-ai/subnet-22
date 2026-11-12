//! Serves a recorded bittensor-fetch replay until stopped: `fetch_replay <replay.json[.gz]> <port> <ca.pem>`, printing `ready <addr>`.

#[path = "../tests/support/replay.rs"]
mod replay;

use serde_json::Value;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, fixture, port, ca] = args.as_slice() else {
        eprintln!("usage: fetch_replay <replay.json> <port> <ca.pem>");
        std::process::exit(2);
    };
    let mut bytes = std::fs::read(fixture).expect("the replay file");
    if fixture.ends_with(".gz") {
        let mut plain = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(bytes.as_slice()), &mut plain).expect("a gzip replay");
        bytes = plain;
    }
    let fixture: Value = serde_json::from_slice(&bytes).expect("replay JSON");
    let hosts: Vec<String> = fixture["hosts"].as_array().expect("hosts").iter().filter_map(|h| h.as_str().map(str::to_string)).collect();
    let replay = replay::start(&fixture["exchanges"], hosts, port.parse().expect("a port")).await;
    std::fs::write(ca, &replay.ca_pem).expect("writing the CA");
    println!("ready {}", replay.addr);
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        for miss in replay.misses.lock().expect("misses").drain(..) {
            eprintln!("missing {miss}");
        }
    }
}

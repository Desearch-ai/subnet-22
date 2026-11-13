//! Forwards JSON-RPC posts to a chain endpoint and appends each answer with its replay key: `rpc_record <upstream> <port> <out.jsonl>`.

use std::sync::{Arc, Mutex};

use std::io::Write;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, upstream, port, out] = args.as_slice() else {
        eprintln!("usage: rpc_record <upstream> <port> <out.jsonl>");
        std::process::exit(2);
    };
    let listener = TcpListener::bind(("127.0.0.1", port.parse::<u16>().expect("a port"))).await.expect("a port");
    let client = reqwest::Client::new();
    let recorded = Arc::new(Mutex::new(std::fs::File::create(out).expect("the record file")));
    println!("ready {}", listener.local_addr().expect("an address"));
    while let Ok((mut socket, _)) = listener.accept().await {
        let (client, recorded, upstream) = (client.clone(), recorded.clone(), upstream.clone());
        tokio::spawn(async move {
            let mut request = Vec::new();
            let mut buf = [0u8; 65536];
            let (head_end, length) = loop {
                if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&request[..at]).to_ascii_lowercase();
                    let length = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0usize);
                    break (at + 4, length);
                }
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                }
            };
            while request.len() < head_end + length {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                }
            }
            let body = request[head_end..head_end + length].to_vec();
            let call: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let answer = client.post(&upstream).header("content-type", "application/json").body(body).send().await;
            let (status, text) = match answer {
                Ok(response) => (response.status().as_u16(), response.text().await.unwrap_or_default()),
                Err(_) => (502, String::new()),
            };
            let key = format!("127.0.0.1 POST / {} {}", call["method"].as_str().unwrap_or(""), call["params"]);
            let encoded = base64_encode(text.as_bytes());
            let line = json!({"key": key, "answer": {"status": status, "headers": [["Content-Type", "application/json"]], "b64": encoded}});
            writeln!(recorded.lock().expect("recorded"), "{line}").expect("writing the record");
            let reply = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
            let _ = socket.write_all(reply.as_bytes()).await;
            let _ = socket.shutdown().await;
        });
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

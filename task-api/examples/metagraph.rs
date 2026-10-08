//! The subnet's validators as the task API loads them, and any hotkeys asked about: `cargo run --example metagraph -- finney 22 [hotkey...]`.

use task_api::chain::Chain;
use task_api::registry::is_validator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let network = args.next().unwrap_or_else(|| "finney".into());
    let netuid: u16 = args.next().map_or(Ok(22), |n| n.parse())?;
    let neurons = Chain::new(&network)?.metagraph(netuid).await?;
    println!("{} neurons", neurons.len());
    for neuron in neurons.iter().filter(|n| is_validator(n)) {
        println!("validator uid={} {} total={} alpha={}", neuron.uid, neuron.hotkey, neuron.total_stake_rao, neuron.alpha_stake_rao);
    }
    for hotkey in args {
        match neurons.iter().find(|n| n.hotkey == hotkey) {
            Some(n) => println!("{hotkey}: uid={} permit={} validator={}", n.uid, n.validator_permit, is_validator(n)),
            None => println!("{hotkey}: not registered"),
        }
    }
    Ok(())
}

//! Embed tasks released at a fixed rate from a buffer: what a planner stall or a new lane does to miners, what the rate may do, and that a round stays verifiable.

use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};
use task_api::lifecycle;
use task_api::queues::{TaskQueue, EMBED_INPUTS, EMBED_PACE};

use crate::embed::{embed_claim, embed_rounds, embed_score, embed_task, input, inputs, score};
use crate::harness::{task_id, Harness};

const AT_ONCE: usize = 2_000;

async fn fill(h: &Harness, queue: &TaskQueue, round_id: &str, count: usize, at: f64) -> Vec<String> {
    let order: Vec<String> = (0..count).map(|n| format!("{round_id}-t{n}")).collect();
    let payloads: Map<String, Value> = order.iter().map(|t| (t.clone(), json!({"url_count": 1, "urls": [format!("https://x.example/{t}")]}))).collect();
    queue.fill(&h.redis, round_id, &order, &payloads, at, 60.0).await.unwrap();
    order
}

/// Tasks released in each whole minute from `start`, one release step a second, as the janitor runs.
async fn per_minute(h: &Harness, queue: &TaskQueue, start: f64, minutes: usize, per_hour: f64) -> Vec<usize> {
    let mut released = Vec::with_capacity(minutes);
    for minute in 0..minutes {
        let mut moved = 0;
        for second in 0..60 {
            moved += queue.release_paced(&h.redis, start + (minute * 60 + second) as f64, per_hour, AT_ONCE).await.unwrap().0;
        }
        released.push(moved);
    }
    released
}

#[tokio::test]
async fn a_fixed_rate_holds_output_flat_through_a_stalled_planner_and_through_a_new_lane_while_the_buffer_lasts() {
    let h = Harness::start().await;
    let queue = TaskQueue::new("embed", 60.0);
    let t0 = 1_000_000.0;
    // Half an hour's tasks stand in the buffer when the planner stalls.
    fill(&h, &queue, "before", 1800, t0).await;
    let start = t0 + 61.0;
    // The first step sets the clock of the rate.
    queue.release_paced(&h.redis, start - 1.0, 3600.0, AT_ONCE).await.unwrap();

    let stalled = per_minute(&h, &queue, start, 20, 3600.0).await;
    assert!(stalled.iter().all(|&moved| moved == 60), "a task a second with nothing handed over for twenty minutes: {stalled:?}");

    // A new lane hands over a burst: it joins the buffer and miners see nothing of it.
    fill(&h, &queue, "burst", 3000, start + 1200.0).await;
    let flooded = per_minute(&h, &queue, start + 1200.0, 5, 3600.0).await;
    assert!(flooded.iter().all(|&moved| moved == 60), "as flat through the burst: {flooded:?}");
    assert_eq!(queue.held_and_ready(&h.redis).await.unwrap(), (1800 + 3000 - 25 * 60, 25 * 60), "the rest stands in the buffer");
}

#[tokio::test]
async fn output_falls_only_once_the_buffer_is_gone() {
    let h = Harness::start().await;
    let queue = TaskQueue::new("embed", 60.0);
    let t0 = 1_000_000.0;
    fill(&h, &queue, "little", 150, t0).await;
    queue.release_paced(&h.redis, t0 + 60.0, 3600.0, AT_ONCE).await.unwrap();
    let released = per_minute(&h, &queue, t0 + 61.0, 4, 3600.0).await;
    assert_eq!(released, [60, 60, 30, 0], "the rate promises no task the planner never handed over");
}

#[tokio::test]
async fn tasks_are_released_oldest_first_and_each_round_in_its_order() {
    let h = Harness::start().await;
    let queue = TaskQueue::new("embed", 60.0);
    let t0 = 1_000_000.0;
    let first = fill(&h, &queue, "first", 5, t0).await;
    let second = fill(&h, &queue, "second", 5, t0 + 300.0).await;
    queue.release_paced(&h.redis, t0 + 400.0, 3600.0, AT_ONCE).await.unwrap();
    let mut seen = Vec::new();
    for second in 1..=10 {
        queue.release_paced(&h.redis, t0 + 400.0 + second as f64, 3600.0, AT_ONCE).await.unwrap();
        let ready: Vec<String> = h.redis.clone().zrange(&queue.ready, 0, -1).await.unwrap();
        for task in ready {
            let _: i64 = h.redis.clone().zrem(&queue.ready, &task).await.unwrap();
            seen.push(task);
        }
    }
    assert_eq!(seen, [first, second].concat());
}

#[tokio::test]
async fn a_rate_moves_toward_its_target_by_about_a_tenth_an_hour_and_never_past_it() {
    let h = Harness::start().await;
    let queue = TaskQueue::new("embed", 60.0);
    let t0 = 1_000_000.0;
    assert_eq!(queue.release_paced(&h.redis, t0, 3000.0, AT_ONCE).await.unwrap().1, 3000.0, "a rate starts at its target");
    let rate_after = |hours: f64, target: f64, from: f64| {
        let (h, queue) = (&h, &queue);
        async move {
            let mut rate = 0.0;
            for minute in 1..=(hours * 60.0) as usize {
                rate = queue.release_paced(&h.redis, from + minute as f64 * 60.0, target, AT_ONCE).await.unwrap().1;
            }
            rate
        }
    };
    let up = rate_after(1.0, 6000.0, t0).await;
    assert!((3300.0..3330.0).contains(&up), "an hour's rise from 3000 to go to 6000: {up}");
    let reached = rate_after(7.0, 6000.0, t0 + 3600.0).await;
    assert_eq!(reached, 6000.0, "reached in the seven hours ln 2 / 0.1 says, and not passed");
    let down = rate_after(1.0, 3000.0, t0 + 8.0 * 3600.0).await;
    assert!((5420.0..5450.0).contains(&down), "down as slowly: {down}");
}

#[tokio::test]
async fn a_pause_never_comes_back_as_a_burst() {
    let h = Harness::start().await;
    let queue = TaskQueue::new("embed", 60.0);
    let t0 = 1_000_000.0;
    fill(&h, &queue, "buffer", 1000, t0).await;
    queue.release_paced(&h.redis, t0 + 61.0, 3600.0, AT_ONCE).await.unwrap();
    let (moved, _) = queue.release_paced(&h.redis, t0 + 61.0 + 3.0 * 3600.0, 3600.0, AT_ONCE).await.unwrap();
    assert_eq!(moved, 10, "three hours down release ten seconds' worth, not three hours'");
}

#[tokio::test]
async fn the_rate_and_its_buffer_survive_a_restart_of_the_api_in_redis() {
    let h = Harness::start().await;
    let (one, other) = (TaskQueue::new("embed", 60.0), TaskQueue::new("embed", 60.0));
    let t0 = 1_000_000.0;
    fill(&h, &one, "kept", 100, t0).await;
    one.release_paced(&h.redis, t0 + 61.0, 3600.0, AT_ONCE).await.unwrap();
    assert_eq!(one.release_paced(&h.redis, t0 + 66.0, 3600.0, AT_ONCE).await.unwrap().0, 5);
    assert_eq!(other.release_paced(&h.redis, t0 + 66.0, 3600.0, AT_ONCE).await.unwrap().0, 0, "a second copy asking at the same moment releases nothing more");
    assert_eq!(other.release_paced(&h.redis, t0 + 71.0, 3600.0, AT_ONCE).await.unwrap().0, 5, "another copy goes on where the first stopped");
    let kept: Option<String> = h.redis.clone().hget(EMBED_PACE, "rate").await.unwrap();
    assert_eq!(kept.as_deref(), Some("3600"));
}

#[tokio::test]
async fn until_a_rate_is_set_rounds_are_released_as_before_and_hold_both_lanes() {
    let h = Harness::with(|s| s.embed_tasks = true).await;
    let mut entries = inputs(2);
    entries[0]["lane"] = "fresh".into();
    for entry in &entries {
        let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    }
    let round = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("a round");
    assert_eq!(round.batches.len(), 2, "one round for both lanes, as before");
    assert!(round.batches.iter().all(|batch| !batch.extra.contains_key("lane")));
    assert_eq!(h.public("/v1/overview?kind=embed").await.1["supply"], json!({"mode": "as_planned"}));
}

fn paced(per_hour: f64) -> impl FnOnce(&mut task_api::settings::Settings) {
    move |s| {
        s.embed_tasks = true;
        s.embed_check_share = 0.0;
        s.embed_release_per_hour = per_hour;
    }
}

#[tokio::test]
async fn the_planner_builds_the_buffer_within_its_room_and_the_rate_holds_it_back() {
    let h = Harness::with(|s| {
        paced(3600.0)(s);
        s.embed_queue_target = 8;
    })
    .await;
    assert_eq!(h.public("/v1/room?kind=embed").await.1["room_tasks"], 8);
    embed_rounds(&h, inputs(6)).await;
    let room = h.public("/v1/room?kind=embed").await.1;
    assert_eq!((room["queue"].clone(), room["room_tasks"].clone()), (json!(6), json!(2)), "tasks held back still count, so a full buffer leaves no room");
    let refused = h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap();
    assert_eq!(refused["refusal"]["code"], "QUEUE_EMPTY", "none is claimable until released");
    let t = now();
    h.state.embed.release_paced(&h.redis, t, 3600.0, AT_ONCE).await.unwrap();
    assert_eq!(h.state.embed.release_paced(&h.redis, t + 2.5, 3600.0, AT_ONCE).await.unwrap().0, 2);
    assert_eq!(h.state.embed.held_and_ready(&h.redis).await.unwrap(), (4, 2));
    assert_eq!(h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"].as_array().unwrap().len(), 1, "what is released is claimed as before");
}

#[tokio::test]
async fn a_fresh_input_goes_ahead_of_the_buffer_within_the_rate_in_a_round_of_its_own() {
    let h = Harness::with(paced(3600.0)).await;
    let waiting = embed_rounds(&h, inputs(3)).await;
    let mut fresh = input();
    fresh["input_key"] = "embed-inputs/fresh.parquet".into();
    fresh["lane"] = "fresh".into();
    let mut rest = input();
    rest["input_key"] = "embed-inputs/rest.parquet".into();
    // The planner hands the fresh lane over first; a round holds one lane.
    for entry in [&fresh, &rest] {
        let _: i64 = h.redis.clone().rpush(EMBED_INPUTS, entry.to_string()).await.unwrap();
    }
    let round = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("the fresh round");
    assert_ne!(round.round_id, waiting);
    assert_eq!(round.batches.iter().map(|b| b.extra["input_key"].clone()).collect::<Vec<_>>(), [fresh["input_key"].clone()]);
    let after = lifecycle::open_embed_rounds(&h.state).await.unwrap().expect("the other round");
    assert_eq!(after.batches.iter().map(|b| b.extra["input_key"].clone()).collect::<Vec<_>>(), [rest["input_key"].clone()]);
    h.revealed().await;

    let t = now();
    h.state.embed.release_paced(&h.redis, t + 120.0, 3600.0, AT_ONCE).await.unwrap();
    assert_eq!(h.state.embed.release_paced(&h.redis, t + 121.5, 3600.0, AT_ONCE).await.unwrap().0, 1);
    let first = h.miner.post("/v1/tasks/claim", embed_claim()).await.unwrap()["tasks"][0].clone();
    assert_eq!(first["round_id"], json!(round.round_id), "the fresh task is the next one released, however many wait");
    let (held, ready) = h.state.embed.held_and_ready(&h.redis).await.unwrap();
    assert_eq!((held, ready), (4, 0), "and it took a place in the rate like any other");
}

#[tokio::test]
async fn the_overview_publishes_the_rate_what_it_is_changing_to_and_the_buffer() {
    let later = now() + 86_400.0;
    let h = Harness::with(move |s| {
        paced(3000.0)(s);
        s.embed_release_next = Some((3300.0, later));
    })
    .await;
    embed_rounds(&h, inputs(3)).await;
    h.state.embed.release_paced(&h.redis, now(), 3000.0, AT_ONCE).await.unwrap();
    let supply = h.public("/v1/overview?kind=embed").await.1["supply"].clone();
    assert_eq!(supply["mode"], "fixed");
    assert_eq!((supply["target_tasks_per_hour"].clone(), supply["tasks_per_hour"].clone()), (json!(3000.0), json!(3000.0)));
    assert_eq!(supply["max_change_per_hour"], 0.1);
    assert_eq!((supply["buffer_tasks"].clone(), supply["ready_tasks"].clone()), (json!(3), json!(0)));
    assert_eq!(supply["buffer_minutes"], 0.1, "three tasks at 3000 an hour");
    assert_eq!((supply["scheduled"]["tasks_per_hour"].clone(), supply["scheduled"]["from"].clone()), (json!(3300.0), json!(later)));
    let ramp = supply["scheduled"]["reached_by"].as_f64().unwrap() - later;
    assert!((3400.0..3500.0).contains(&ramp), "a tenth an hour takes an hour less a bit: {ramp}");
    assert_eq!(h.public("/v1/overview?kind=crawl").await.1["supply"], Value::Null, "crawl makes no such promise");
    assert_eq!(h.state.settings.embed_release_target(later), Some(3300.0), "the target changes when it said it would");
}

#[tokio::test]
async fn a_round_released_at_a_fixed_rate_is_served_in_its_published_order_and_verifies() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_release_per_hour = 3600.0;
    })
    .await;
    let round_id = embed_rounds(&h, inputs(6)).await;
    let t = now();
    h.state.embed.release_paced(&h.redis, t, 3600.0, AT_ONCE).await.unwrap();
    let mut served = Vec::new();
    for step in 1..=6 {
        assert_eq!(h.state.embed.release_paced(&h.redis, t + step as f64 + 0.5, 3600.0, AT_ONCE).await.unwrap().0, 1);
        let miner = if step % 2 == 0 { &h.miner } else { &h.rival };
        let task = task_id(&embed_task(&h, miner).await);
        h.opened(Some(&h.validator), Some(&task)).await;
        score(&h, &task, embed_score("pass", "matched", "ok")).await;
        served.push(task);
    }
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let published = h.public(&format!("/v1/rounds/{round_id}")).await.1;
    assert_eq!(json!(served), published["serve_order"], "handed out in exactly the order the seed decided");
    // The verifier itself, wherever a Python with bittensor-wallet is named.
    if let Ok(python) = std::env::var("TASK_API_TEST_PYTHON") {
        let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/verify_round.py");
        let url = h.url.clone();
        let done = tokio::task::spawn_blocking(move || std::process::Command::new(python).args([tool, "--api", &url, "--round", &round_id]).output().unwrap());
        let done = done.await.unwrap();
        assert!(done.status.success(), "verify_round.py:\n{}", String::from_utf8_lossy(&done.stdout));
    }
}

#[tokio::test]
async fn the_janitor_releases_at_the_rate_and_forgets_it_once_it_is_unset() {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_release_per_hour = 7200.0;
    })
    .await;
    fill(&h, &h.state.embed, "live", 100, now() - 120.0).await;
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let janitor = tokio::spawn(task_api::janitor::run(h.state.clone(), stopped));
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    let _ = stop.send(true);
    janitor.await.unwrap();
    let (held, ready) = h.state.embed.held_and_ready(&h.redis).await.unwrap();
    assert_eq!(held + ready, 100);
    assert!((3..=10).contains(&ready), "two a second for about four seconds, not all at once: {ready}");

    let (off, _) = (h.state.settings.clone(), ());
    let off = task_api::settings::Settings { embed_release_per_hour: 0.0, ..off };
    let state =
        std::sync::Arc::new(task_api::state::State::new(off, h.redis.clone(), h.state.storage.clone(), h.state.pages.clone()).await.expect("the copy opens"));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let janitor = tokio::spawn(task_api::janitor::run(state.clone(), stopped));
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    let _ = stop.send(true);
    janitor.await.unwrap();
    assert_eq!(state.embed.held_and_ready(&h.redis).await.unwrap(), (0, 100), "rollback is one setting: what was held goes out at once");
    let rate: Option<String> = h.redis.clone().hget(EMBED_PACE, "rate").await.unwrap();
    assert_eq!(rate, None, "and a later rate starts at its own target");
}

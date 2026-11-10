//! Claims waiting in line: the next released task goes to the place that has waited longest, polling miners keep working and never get ahead, and the order can be checked afterwards. Embed and crawl each have a line of their own.

use std::time::{Duration, Instant};

use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Value};
use task_api::lifecycle;
use task_api::line::{self, MAX_WAIT_S};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::embed::{embed_rounds, embed_score, inputs, score};
use crate::harness::{first_url, parquet, score as crawl_score, task_id, urls, vectors, Client, Harness, MINER, RIVAL};

const THIRD: &str = "//LineThird";
// Within what miners' clients allow and what a signature is good for.
const _: () = assert!(MAX_WAIT_S < 30.0);
const AT_ONCE: usize = 2_000;

fn body(wait_s: i64, count: i64) -> Value {
    json!({"kind": "embed", "models": ["desearch-4b-v1"], "wait_s": wait_s, "count": count})
}

/// A harness with claims waiting in line, tasks released only as `Release` says, and waiting claims woken as in service.
async fn in_line(tune: impl FnOnce(&mut task_api::settings::Settings)) -> (Harness, watch::Sender<bool>) {
    let h = Harness::with(|s| {
        s.embed_tasks = true;
        s.embed_line.on = true;
        s.embed_release_per_hour = 3600.0;
        tune(s);
    })
    .await;
    let (stop, stopped) = watch::channel(false);
    tokio::spawn(line::keep_waking(h.state.clone(), stopped));
    (h, stop)
}

struct Release {
    clock: f64,
}

impl Release {
    async fn start(h: &Harness) -> Release {
        let clock = now() + 120.0;
        h.state.embed.release_paced(&h.redis, clock, 3600.0, AT_ONCE).await.unwrap();
        Release { clock }
    }

    async fn one(&mut self, h: &Harness) {
        for _ in 0..100 {
            self.clock += 0.6;
            if h.state.embed.release_paced(&h.redis, self.clock, 3600.0, AT_ONCE).await.unwrap().0 > 0 {
                return;
            }
        }
        panic!("no task to release");
    }
}

fn waiting(h: &Harness, uri: &str, wait_s: i64, count: i64) -> JoinHandle<Value> {
    let client = h.client(uri, &h.url);
    tokio::spawn(async move { client.post("/v1/tasks/claim", body(wait_s, count)).await.expect("a claim answers") })
}

async fn poll(client: &Client) -> Value {
    client.post("/v1/tasks/claim", body(0, 1)).await.unwrap()
}

async fn settle() {
    tokio::time::sleep(Duration::from_millis(250)).await;
}

/// The answer once it comes, soon.
async fn answered(claim: JoinHandle<Value>) -> Value {
    tokio::time::timeout(Duration::from_secs(5), claim).await.expect("answered soon").unwrap()
}

fn tasks(answer: &Value) -> usize {
    answer["tasks"].as_array().map_or(0, Vec::len)
}

#[tokio::test]
async fn the_next_released_task_goes_to_the_miner_that_has_waited_longest() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(3)).await;
    let mut release = Release::start(&h).await;
    let first = waiting(&h, MINER, 15, 1);
    settle().await;
    let second = waiting(&h, RIVAL, 15, 1);
    settle().await;
    let third = waiting(&h, THIRD, 15, 1);
    settle().await;

    release.one(&h).await;
    let first = answered(first).await;
    assert_eq!(tasks(&first), 1, "the first to wait: {first}");
    assert_eq!(first["receipts"][0]["body"]["hotkey"], h.miner.ss58());
    settle().await;
    assert!(!second.is_finished() && !third.is_finished(), "the others go on waiting");

    release.one(&h).await;
    let second = answered(second).await;
    assert_eq!(second["receipts"][0]["body"]["hotkey"], h.rival.ss58());
    settle().await;
    assert!(!third.is_finished());
    release.one(&h).await;
    let third = answered(third).await;
    assert_eq!(tasks(&third), 1);

    let waited = |answer: &Value| answer["receipts"][0]["body"]["waited_from"].as_f64().expect("a place in line");
    assert!(waited(&first) < waited(&second) && waited(&second) < waited(&third), "each receipt says when its place was taken");
}

#[tokio::test]
async fn polling_never_gets_ahead_of_waiting_and_polling_miners_keep_their_turn() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(3)).await;
    let mut release = Release::start(&h).await;

    // A miner already waiting: polls after it get nothing, however many.
    let patient = waiting(&h, MINER, 15, 1);
    settle().await;
    for _ in 0..3 {
        let refused = poll(&h.rival).await;
        assert_eq!((refused["refusal"]["code"].clone(), refused["refusal"]["inputs"]["retry_after"].clone()), (json!("QUEUE_EMPTY"), json!(5)));
        assert_eq!((refused["refusal"]["inputs"]["ahead"].clone(), refused["refusal"]["inputs"]["in_line"].clone()), (json!(1), json!(2)));
    }
    release.one(&h).await;
    assert_eq!(poll(&h.rival).await["refusal"]["code"], "QUEUE_EMPTY", "a poll landing as the task is released still waits its turn");
    assert_eq!(tasks(&answered(patient).await), 1);

    // The poller's place dates from its first poll, ahead of whoever came after it.
    let later = waiting(&h, THIRD, 15, 1);
    settle().await;
    release.one(&h).await;
    settle().await;
    assert!(!later.is_finished(), "the task is the polling miner's");
    assert_eq!(tasks(&poll(&h.rival).await), 1, "a polling miner still gets its task, on its next poll");
    release.one(&h).await;
    assert_eq!(tasks(&answered(later).await), 1);
}

#[tokio::test]
async fn a_place_nobody_asks_for_is_let_go_and_the_next_in_line_is_served() {
    let (h, _stop) = in_line(|s| s.line_keep = 1.0).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 0);
    let waiter = waiting(&h, MINER, 15, 1);
    settle().await;
    release.one(&h).await;
    let answer = answered(waiter).await;
    assert_eq!(answer["receipts"][0]["body"]["hotkey"], h.miner.ss58(), "the miner that stopped asking lost its place within {:?}", Duration::from_secs(1));
}

#[tokio::test]
async fn a_miner_given_a_task_it_cannot_take_passes_its_turn() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    let claim = waiting(&h, MINER, 15, 1);
    settle().await;
    release.one(&h).await;
    let task = task_id(&answered(claim).await["tasks"][0]);
    h.miner.post(&format!("/v1/tasks/{task}/abandon"), json!({})).await.unwrap();
    // The task it handed back is the only one ready, and a miner never gets a task again.
    assert_eq!(poll(&h.miner).await["refusal"]["code"], "QUEUE_EMPTY");
    let other = answered(waiting(&h, RIVAL, 15, 1)).await;
    assert_eq!(task_id(&other["tasks"][0]), task, "it goes to the next miner rather than wait on the first");
}

#[tokio::test]
async fn a_claim_waits_only_as_long_as_it_asks_and_its_place_outlasts_the_wait() {
    let (h, _stop) = in_line(|_| {}).await;
    let started = Instant::now();
    let refused = h.miner.post("/v1/tasks/claim", body(1, 1)).await.unwrap();
    assert!((Duration::from_millis(900)..Duration::from_secs(4)).contains(&started.elapsed()), "{:?}", started.elapsed());
    assert_eq!(
        (refused["refusal"]["code"].clone(), refused["refusal"]["inputs"]["retry_after"].clone()),
        (json!("QUEUE_EMPTY"), json!(0)),
        "it is told to ask again at once"
    );
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 1, "and keeps its place meanwhile");
}

#[tokio::test]
async fn a_miner_with_no_capacity_is_told_so_at_once_and_holds_no_place() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(2)).await;
    let mut release = Release::start(&h).await;
    release.one(&h).await;
    assert_eq!(tasks(&poll(&h.miner).await), 1);
    let started = Instant::now();
    let full = h.miner.post("/v1/tasks/claim", body(15, 1)).await.unwrap();
    assert_eq!(full["refusal"]["code"], "NO_CAPACITY");
    assert!(started.elapsed() < Duration::from_secs(1), "no wait for what waiting cannot give");
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 0, "and no place ahead of anyone");
}

#[tokio::test]
async fn a_miner_allowed_more_places_holds_one_for_each_task_and_keeps_the_older_ones() {
    let (h, _stop) = in_line(|s| s.embed_line.places = 3).await;
    let hotkey = h.miner.ss58();
    h.state
        .db
        .run(move |conn| {
            task_api::budgets::get_or_create(conn, &hotkey, "embed")?;
            Ok(conn.execute("UPDATE miners SET budget = 3 WHERE hotkey = ? AND pool = 'embed'", [hotkey])?)
        })
        .await
        .unwrap();
    embed_rounds(&h, inputs(3)).await;
    let mut release = Release::start(&h).await;
    let big = waiting(&h, MINER, 15, 3);
    settle().await;
    let rival = waiting(&h, RIVAL, 15, 1);
    settle().await;
    release.one(&h).await;
    assert_eq!(tasks(&answered(big).await), 1);
    // Its other two places date from before the rival's.
    assert_eq!(h.miner.post("/v1/tasks/claim", body(0, 2)).await.unwrap()["refusal"]["inputs"]["ahead"], 0);
    release.one(&h).await;
    settle().await;
    assert!(!rival.is_finished(), "the task is for the older place");
    assert_eq!(tasks(&h.miner.post("/v1/tasks/claim", body(0, 2)).await.unwrap()), 1);
    assert_eq!(h.miner.post("/v1/tasks/claim", body(0, 1)).await.unwrap()["refusal"]["inputs"]["ahead"], 0, "asking for fewer keeps the oldest");
    release.one(&h).await;
    settle().await;
    assert_eq!(tasks(&h.miner.post("/v1/tasks/claim", body(0, 1)).await.unwrap()), 1);
    assert!(!rival.is_finished());
    rival.abort();
}

#[tokio::test]
async fn a_hotkey_holds_one_place_however_many_tasks_it_asks_for() {
    let (h, _stop) = in_line(|_| {}).await;
    let hotkey = h.miner.ss58();
    h.state
        .db
        .run(move |conn| {
            task_api::budgets::get_or_create(conn, &hotkey, "embed")?;
            Ok(conn.execute("UPDATE miners SET budget = 3 WHERE hotkey = ? AND pool = 'embed'", [hotkey])?)
        })
        .await
        .unwrap();
    assert_eq!(h.miner.post("/v1/tasks/claim", body(0, 3)).await.unwrap()["refusal"]["inputs"]["in_line"], 1);
}

#[tokio::test]
async fn a_miner_gone_from_its_wait_lets_go_of_its_place_at_once() {
    let (h, _stop) = in_line(|_| {}).await;
    let claim = waiting(&h, MINER, 15, 1);
    settle().await;
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 1);
    claim.abort();
    settle().await;
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 0, "the rival moves up without waiting out the place");
}

#[tokio::test]
async fn a_task_dropped_as_it_was_drawn_costs_the_longest_waiter_no_turn() {
    let (h, _stop) = in_line(|s| s.embed_input_max_age = 3600.0).await;
    let mut both = inputs(2);
    let fine = both.pop().unwrap();
    let mut stale = both.pop().unwrap();
    stale["written_at"] = (now() - 7200.0).into();
    embed_rounds(&h, vec![stale]).await;
    let round = embed_rounds(&h, vec![fine]).await;
    let mut release = Release::start(&h).await;
    let first = waiting(&h, MINER, 15, 1);
    settle().await;
    let second = waiting(&h, RIVAL, 15, 1);
    settle().await;
    release.one(&h).await;
    settle().await;
    assert!(!first.is_finished() && !second.is_finished(), "the dropped task went to no one");
    release.one(&h).await;
    let answer = answered(first).await;
    assert_eq!(answer["tasks"][0]["round_id"], json!(round), "the next task is still the first waiter's");
    second.abort();
}

#[tokio::test]
async fn a_miner_locked_out_while_it_waits_is_told_so() {
    let (h, _stop) = in_line(|_| {}).await;
    let claim = waiting(&h, MINER, 15, 1);
    settle().await;
    let hotkey = h.miner.ss58();
    h.state.db.run(move |conn| task_api::budgets::lock_out(conn, &hotkey, "embed", 1.0, "test", now())).await.unwrap();
    assert_eq!(answered(claim).await["refusal"]["code"], "LOCKED_OUT");
}

#[tokio::test]
async fn a_claim_the_api_never_finished_making_costs_the_miner_nothing_and_its_round_still_verifies() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    release.one(&h).await;
    // Taken from the queue as if Redis was lost as it answered: no `issued` mark, no receipt.
    let drawn = h.state.embed.claim(&h.redis, &h.miner.ss58(), 5, 10, 1, now()).await.unwrap().unwrap();
    assert_eq!(drawn.len(), 1);
    let reclaimed = lifecycle::reclaim_expired(&h.state, now() + 1e6).await.unwrap();
    assert_eq!(reclaimed.len(), 1);
    let hotkey = h.miner.ss58();
    let strikes: i64 =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap();
    assert_eq!(strikes, 0, "a miner never told of a task is not struck for it");
}

#[tokio::test]
async fn a_claim_whose_receipt_could_not_be_written_costs_the_miner_nothing() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    release.one(&h).await;
    h.state.db.run(|conn| Ok(conn.execute_batch("ALTER TABLE entries RENAME TO entries_gone")?)).await.unwrap();
    assert!(h.miner.post("/v1/tasks/claim", body(0, 1)).await.is_err(), "the miner is told nothing");
    let reclaimed = lifecycle::reclaim_unissued(&h.state, now() + 90.0).await.unwrap();
    assert_eq!(reclaimed.len(), 1, "it goes back within a minute, not when its clock runs out, whatever the log can take");
    h.state.db.run(|conn| Ok(conn.execute_batch("ALTER TABLE entries_gone RENAME TO entries")?)).await.unwrap();
    h.state.write_owed().await.unwrap();
    let events = h.public("/v1/events?outcome=reclaimed&limit=10").await.1;
    assert_eq!(events["events"][0]["cause"], "never_issued", "the line was kept until the log could take it, as the round can't be verified without it");
    let hotkey = h.miner.ss58();
    let strikes: i64 =
        h.state.db.run(move |conn| Ok(conn.query_row("SELECT COUNT(*) FROM strikes WHERE hotkey = ?", [hotkey], |row| row.get(0))?)).await.unwrap();
    assert_eq!(strikes, 0);
    assert_eq!(tasks(&poll(&h.rival).await), 1, "and the next miner gets it");
}

/// A task in ready for each of `ids`, ranked in that order.
async fn stand_ready(h: &Harness, ids: &[&str]) {
    let mut redis = h.redis.clone();
    for (rank, id) in ids.iter().enumerate() {
        let _: () = redis.set(format!("task:{id}"), json!({"rank": rank, "kind": "embed"}).to_string()).await.unwrap();
        let _: i64 = redis.zadd(&h.state.embed.ready, *id, rank).await.unwrap();
    }
}

async fn ask(h: &Harness, who: &str, at: f64) -> Vec<String> {
    match h.state.embed.claim_in_line(&h.redis, who, 5, 10, 1, at, 30.0, 1).await.unwrap() {
        Ok(claims) => claims.into_iter().map(|claim| claim.task_id).collect(),
        Err(_) => Vec::new(),
    }
}

#[tokio::test]
async fn a_free_pass_is_one_task_however_many_places_the_miner_has() {
    let h = Harness::start().await;
    let t = 1_000_000.0;
    for (at, who) in ["a", "b", "c", "d"].into_iter().enumerate() {
        assert!(ask(&h, who, t + at as f64).await.is_empty());
    }
    stand_ready(&h, &["x", "y", "z"]).await;
    // Keeps the line ahead without its miners asking, which would take the tasks in turn.
    for place in ["a:0", "b:0", "c:0", "d:0"] {
        let _: i64 = h.redis.clone().zadd(task_api::queues::line_alive_key("embed"), place, t + 100.0).await.unwrap();
    }
    let first = h.state.embed.claim_in_line(&h.redis, "many", 5, 10, 4, t + 5.0, 30.0, 4).await.unwrap();
    assert!(first.is_err(), "behind four places with three tasks ready, it waits");
    let claims = h.state.embed.claim_in_line(&h.redis, "many", 5, 10, 4, t + 36.0, 30.0, 4).await.unwrap().unwrap();
    assert_eq!(claims.len(), 1);
}

#[tokio::test]
async fn a_free_pass_on_a_stuck_line_is_one_task_not_a_race_from_then_on() {
    let h = Harness::start().await;
    let mut redis = h.redis.clone();
    let t = 1_000_000.0;
    for (at, who) in ["b", "c", "d", "e"].into_iter().enumerate() {
        assert!(ask(&h, who, t + at as f64).await.is_empty(), "nothing is ready, so each takes a place");
    }
    // Keeps the front of the line without its miner asking, which would pass its turn.
    let _: i64 = redis.zadd(task_api::queues::line_alive_key("embed"), "b:0", t + 100.0).await.unwrap();
    // A task none of them can take, as each has held it, stands ready well past a place's life.
    let _: i64 = redis.sadd("holders:held-by-all", vec!["b", "c", "d", "e"]).await.unwrap();
    stand_ready(&h, &["held-by-all"]).await;
    for at in [10.0, 25.0, 40.0] {
        for who in ["c", "d", "e"] {
            assert!(ask(&h, who, t + at).await.is_empty());
        }
    }
    stand_ready(&h, &["held-by-all", "one"]).await;
    assert_eq!(ask(&h, "e", t + 41.0).await, ["one"], "a line stuck this long lets whoever asks take a task");
    stand_ready(&h, &["held-by-all", "two"]).await;
    assert!(ask(&h, "d", t + 42.0).await.is_empty(), "the line has moved, so it is in order again");
    assert_eq!(ask(&h, "c", t + 43.0).await, ["two"], "and the one whose turn it is takes it");
}

#[tokio::test]
async fn a_claim_the_api_never_finished_making_goes_back_within_a_minute() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    release.one(&h).await;
    let drawn = h.state.embed.claim(&h.redis, &h.miner.ss58(), 5, 10, 1, now()).await.unwrap().unwrap();
    let tokens = || async {
        let tokens: f64 = h.redis.clone().hget("embed:pace", "tokens").await.unwrap();
        tokens
    };
    let tokens_before = tokens().await;
    assert!(lifecycle::reclaim_unissued(&h.state, now() + 30.0).await.unwrap().is_empty(), "issuing takes moments, but it may still be under way");
    assert_eq!(lifecycle::reclaim_unissued(&h.state, now() + 90.0).await.unwrap(), [drawn[0].task_id.clone()]);
    assert_eq!(h.state.embed.in_flight(&h.redis, &h.miner.ss58()).await.unwrap(), 0, "and the miner has its slot back");
    assert_eq!(task_api::queues::handbacks(&h.redis, &drawn[0].task_id).await.unwrap(), 0, "it was never handed back by anyone");
    assert_eq!(tokens().await, tokens_before, "and it costs the rate nothing");
    assert_eq!(tasks(&poll(&h.rival).await), 1, "the task goes to the next in line");
    let events = h.public("/v1/events?outcome=reclaimed&limit=10").await.1;
    assert_eq!(events["events"][0]["cause"], "never_issued");
}

#[tokio::test]
async fn a_claim_with_its_task_issued_is_left_to_its_clock() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    release.one(&h).await;
    assert_eq!(tasks(&poll(&h.miner).await), 1);
    assert!(lifecycle::reclaim_unissued(&h.state, now() + 120.0).await.unwrap().is_empty());
    assert_eq!(h.state.embed.in_flight(&h.redis, &h.miner.ss58()).await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_gone_as_it_is_served_leaves_no_claim_without_a_receipt() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(6)).await;
    let mut release = Release::start(&h).await;
    for attempt in 0..6 {
        let uri = format!("//Gone{attempt}");
        let hotkey = h.client(&uri, &h.url).ss58();
        let claim = waiting(&h, &uri, 10, 1);
        settle().await;
        release.one(&h).await;
        let looking = Instant::now();
        let task = loop {
            let mine: Vec<String> =
                redis::cmd("SMEMBERS").arg(task_api::queues::inflight_key("embed", &hotkey)).query_async(&mut h.redis.clone()).await.unwrap();
            if let Some(task) = mine.first() {
                break task.clone();
            }
            assert!(looking.elapsed() < Duration::from_secs(10), "the waiting claim got no task");
        };
        claim.abort();
        settle().await;
        let log = h.public(&format!("/v1/events?task_id={task}&limit=10")).await.1;
        let outcomes: Vec<&str> = log["events"].as_array().unwrap().iter().filter_map(|event| event["outcome"].as_str()).collect();
        assert!(outcomes.contains(&"issued"), "the log has the claim whatever became of the request: {outcomes:?}");
    }
}

/// Each hotkey asks again just before the one ahead of it dies, while it is not yet its turn, so its own life starts late.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chain_of_hotkeys_holds_a_task_no_longer_than_a_place_lasts() {
    const KEEP: f64 = 2.0;
    let (h, _stop) = in_line(|s| s.line_keep = KEEP).await;
    let clients: Vec<Client> = (0..4).map(|at| h.client(&format!("//LineChain{at}"), &h.url)).collect();
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    for client in &clients {
        assert_eq!(tasks(&poll(client).await), 0);
    }
    let victim = waiting(&h, RIVAL, 20, 1);
    settle().await;
    release.one(&h).await;
    let started = Instant::now();
    let mut taken = false;
    for step in 1..clients.len() {
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        for client in &clients[step..] {
            taken |= tasks(&poll(client).await) == 1;
        }
    }
    if taken {
        assert!(started.elapsed() < Duration::from_secs(6), "one of the chain took it once the line was stuck");
        victim.abort();
    } else {
        let answer = tokio::time::timeout(Duration::from_secs(5), victim).await.expect("answered soon").unwrap();
        assert_eq!(tasks(&answer), 1);
    }
    assert!(started.elapsed().as_secs_f64() < KEEP + 3.0 + 3.0, "held {:?} with a place lasting {KEEP} s", started.elapsed());
}

#[tokio::test]
async fn a_place_survives_a_restart_and_is_good_at_any_copy() {
    let (h, _stop) = in_line(|_| {}).await;
    embed_rounds(&h, inputs(1)).await;
    let mut release = Release::start(&h).await;
    assert_eq!(poll(&h.miner).await["refusal"]["inputs"]["ahead"], 0);
    let copy = h.restarted().await;
    let other = h.client(RIVAL, &copy);
    assert_eq!(poll(&other).await["refusal"]["inputs"]["ahead"], 1, "the other copy sees the same line");
    release.one(&h).await;
    assert_eq!(poll(&other).await["refusal"]["code"], "QUEUE_EMPTY");
    assert_eq!(tasks(&poll(&h.client(MINER, &copy)).await), 1, "the first place is served at any copy");
}

#[tokio::test]
async fn a_copy_going_away_answers_its_waiting_claims() {
    let (h, _stop) = in_line(|_| {}).await;
    let claim = waiting(&h, MINER, 20, 1);
    settle().await;
    h.state.draining.store(true, std::sync::atomic::Ordering::Relaxed);
    let answer = answered(claim).await;
    assert_eq!(
        (answer["refusal"]["code"].clone(), answer["refusal"]["inputs"]["retry_after"].clone()),
        (json!("QUEUE_EMPTY"), json!(0)),
        "and its miner asks again elsewhere at once"
    );
}

#[tokio::test]
async fn with_the_line_off_a_claim_that_asks_to_wait_is_answered_at_once_as_before() {
    let h = Harness::with(|s| s.embed_tasks = true).await;
    let started = Instant::now();
    let refused = h.miner.post("/v1/tasks/claim", body(15, 1)).await.unwrap();
    assert_eq!((refused["refusal"]["code"].clone(), refused["refusal"]["inputs"]["retry_after"].clone()), (json!("QUEUE_EMPTY"), json!(10.0)));
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(refused["refusal"]["inputs"].get("ahead").is_none());
    let line: Option<i64> = redis::cmd("ZCARD").arg("line:embed").query_async(&mut h.redis.clone()).await.ok();
    assert_eq!(line, Some(0), "no line kept");
}

#[tokio::test]
async fn crawl_claims_are_not_in_line_while_only_the_embed_line_is_on() {
    let (h, _stop) = in_line(|_| {}).await;
    let started = Instant::now();
    let refused = h.miner.post("/v1/tasks/claim", json!({"kind": "crawl", "wait_s": 15})).await.unwrap();
    assert_eq!(refused["refusal"]["code"], "QUEUE_EMPTY");
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[tokio::test]
async fn a_round_served_from_the_line_verifies_with_its_order_and_its_signed_places() {
    let (h, _stop) = in_line(|_| {}).await;
    let round_id = embed_rounds(&h, inputs(4)).await;
    let mut release = Release::start(&h).await;
    for step in 0..4 {
        let (uri, client) = if step % 2 == 0 { (MINER, &h.miner) } else { (RIVAL, &h.rival) };
        let claim = waiting(&h, uri, 15, 1);
        settle().await;
        release.one(&h).await;
        let answer = answered(claim).await;
        let task = answer["tasks"][0].clone();
        h.put(task["upload"]["url"].as_str().unwrap(), vectors(6)).await;
        client.post(&format!("/v1/tasks/{}/complete", task_id(&task)), json!({"key": task["upload"]["key"], "bytes": 7})).await.unwrap();
        h.opened(Some(&h.validator), Some(&task_id(&task))).await;
        score(&h, &task_id(&task), embed_score("pass", "matched", "ok")).await;
    }
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [round_id.as_str()]);
    let log = h.public(&format!("/v1/rounds/{round_id}/log")).await.1;
    let issued: Vec<&Value> = log["entries"].as_array().unwrap().iter().filter(|entry| entry["outcome"] == "issued").collect();
    assert_eq!(issued.len(), 4);
    assert!(issued.iter().all(|entry| entry["waited_from"].is_f64()), "the log says when each place was taken: {issued:?}");
    if let Ok(python) = std::env::var("TASK_API_TEST_PYTHON") {
        let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/verify_round.py");
        let url = h.url.clone();
        let done = tokio::task::spawn_blocking(move || std::process::Command::new(python).args([tool, "--api", &url, "--round", &round_id]).output().unwrap());
        let done = done.await.unwrap();
        assert!(done.status.success(), "verify_round.py:\n{}", String::from_utf8_lossy(&done.stdout));
    }
}

/// As many miners as the subnet has UIDs, all waiting at once on one copy of the API.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_uid_waiting_at_once_is_served_in_the_order_it_came_and_soon() {
    const MINERS: usize = 250;
    let (h, _stop) = in_line(|s| s.embed_release_per_hour = 360_000.0).await;
    let mut claims = Vec::with_capacity(MINERS);
    for at in 0..MINERS {
        let client: Client = h.client(&format!("//LineMiner{at}"), &h.url);
        claims.push(tokio::spawn(async move { client.post("/v1/tasks/claim", body(18, 1)).await.expect("a claim answers") }));
        tokio::time::sleep(Duration::from_millis(4)).await;
    }
    settle().await;
    assert!(h.state.waiters.len() >= MINERS - 1, "all are waiting: {}", h.state.waiters.len());
    let line: i64 = redis::cmd("ZCARD").arg("line:embed").query_async(&mut h.redis.clone()).await.unwrap();
    assert_eq!(line, MINERS as i64);

    for chunk in inputs(MINERS).chunks(150) {
        embed_rounds(&h, chunk.to_vec()).await;
    }
    let started = Instant::now();
    let mut clock = now() + 120.0;
    h.state.embed.release_paced(&h.redis, clock, 360_000.0, AT_ONCE).await.unwrap();
    while claims.iter().any(|claim| !claim.is_finished()) && started.elapsed() < Duration::from_secs(15) {
        tokio::time::sleep(Duration::from_millis(50)).await;
        clock += 0.05;
        h.state.embed.release_paced(&h.redis, clock, 360_000.0, AT_ONCE).await.unwrap();
    }
    let took = started.elapsed();
    let mut served = Vec::new();
    for claim in claims {
        let answer = answered(claim).await;
        assert_eq!(tasks(&answer), 1, "every miner gets its task: {answer}");
        let receipt = &answer["receipts"][0]["body"];
        served.push((receipt["seq"].as_i64().unwrap(), receipt["waited_from"].as_f64().unwrap()));
    }
    assert!(took < Duration::from_secs(10), "250 waiting miners served in {took:?}");
    served.sort_by_key(|(seq, _)| *seq);
    let overtaken = served.windows(2).filter(|pair| pair[1].1 < pair[0].1).count();
    // Places within one wake-up may land in either order; none goes far out of turn.
    let worst =
        served.iter().enumerate().map(|(at, (_, waited))| (at as i64 - served.iter().filter(|(_, other)| other < waited).count() as i64).abs()).max().unwrap();
    assert!(worst <= 30, "the furthest any was served from its turn: {worst} ({overtaken} neighbours out of order)");
}

fn crawl(wait_s: i64, count: i64) -> Value {
    json!({"kind": "crawl", "wait_s": wait_s, "count": count})
}

/// A harness with crawl claims waiting in line, and waiting claims woken as in service.
async fn crawl_in_line(tune: impl FnOnce(&mut task_api::settings::Settings)) -> (Harness, watch::Sender<bool>) {
    let h = Harness::with(|s| {
        s.crawl_line.on = true;
        tune(s);
    })
    .await;
    let (stop, stopped) = watch::channel(false);
    tokio::spawn(line::keep_waking(h.state.clone(), stopped));
    (h, stop)
}

/// A crawl round of one task, revealed and released at once.
async fn one_crawl_task(h: &Harness) -> String {
    h.enqueue(urls(3)).await["round_id"].as_str().unwrap().to_string()
}

async fn crawl_budget(h: &Harness, hotkey: String, budget: i64) {
    h.state
        .db
        .run(move |conn| {
            task_api::budgets::get_or_create(conn, &hotkey, "crawl")?;
            Ok(conn.execute("UPDATE miners SET budget = ? WHERE hotkey = ? AND pool = 'crawl'", rusqlite::params![budget, hotkey])?)
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn crawl_claims_wait_in_a_line_of_their_own_and_its_order_can_be_checked() {
    let (h, _stop) = crawl_in_line(|_| {}).await;
    let third = h.client(THIRD, &h.url);
    for (client, ahead) in [(&h.miner, 0), (&h.rival, 1), (&third, 2)] {
        let refused = client.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap();
        let inputs = &refused["refusal"]["inputs"];
        assert_eq!(
            (refused["refusal"]["code"].clone(), inputs["ahead"].clone(), inputs["retry_after"].clone()),
            (json!("QUEUE_EMPTY"), json!(ahead), json!(5))
        );
    }
    let embed_line: i64 = h.redis.clone().zcard("line:embed").await.unwrap();
    assert_eq!(embed_line, 0, "the embed line is not touched");

    let first_round = one_crawl_task(&h).await;
    assert_eq!(tasks(&third.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()), 0, "not its turn");
    assert_eq!(tasks(&h.rival.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()), 0, "nor this one's");
    let first = h.miner.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap();
    assert_eq!(tasks(&first), 1, "the first place: {first}");
    one_crawl_task(&h).await;
    assert_eq!(tasks(&third.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()), 0);
    assert_eq!(tasks(&h.rival.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()), 1);
    one_crawl_task(&h).await;
    assert_eq!(tasks(&third.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()), 1);

    let events = h.public("/v1/events?outcome=issued&kind=crawl").await.1;
    let issued = events["events"].as_array().unwrap();
    assert_eq!(issued.len(), 3);
    assert!(issued.iter().all(|event| event["kind"] == "crawl" && event["waited_from"].is_f64()), "{issued:?}");
    assert_eq!(h.public("/v1/events?outcome=issued&kind=embed").await.1["events"], json!([]));
    assert_eq!(h.public("/v1/overview?kind=crawl").await.1["supply"], json!({"mode": "as_planned", "waiting_line": true}));
    assert_eq!(h.public("/v1/overview?kind=embed").await.1["supply"]["waiting_line"], false);

    let task = first["tasks"][0].clone();
    let id = task_id(&task);
    let body = parquet(&task);
    h.put(task["upload"]["url"].as_str().unwrap(), body.clone()).await;
    let report = json!({"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0, "bytes": body.len()});
    h.miner.post(&format!("/v1/tasks/{id}/complete"), report).await.unwrap();
    h.opened(Some(&h.validator), Some(&id)).await;
    h.validator.post(&format!("/v1/validation/{id}/score"), crawl_score("pass", 3, "ok", None, &first_url(&task))).await.unwrap();
    assert_eq!(lifecycle::close_finished(&h.state).await.unwrap(), [first_round.as_str()]);
    if let Ok(python) = std::env::var("TASK_API_TEST_PYTHON") {
        let url = h.url.clone();
        let done = tokio::task::spawn_blocking(move || {
            let round = std::process::Command::new(&python)
                .args([concat!(env!("CARGO_MANIFEST_DIR"), "/tools/verify_round.py"), "--api", &url, "--round", &first_round])
                .output()
                .unwrap();
            let line = std::process::Command::new(&python)
                .args([concat!(env!("CARGO_MANIFEST_DIR"), "/tools/verify_line.py"), "--api", &url, "--kind", "crawl"])
                .output()
                .unwrap();
            (round, line)
        });
        let (round, line) = done.await.unwrap();
        assert!(round.status.success(), "verify_round.py:\n{}", String::from_utf8_lossy(&round.stdout));
        assert!(line.status.success(), "verify_line.py:\n{}", String::from_utf8_lossy(&line.stdout));
        assert!(String::from_utf8_lossy(&line.stdout).contains("3 tasks given from the line"), "{}", String::from_utf8_lossy(&line.stdout));
    }
}

#[tokio::test]
async fn a_crawl_claim_in_line_takes_a_task_for_each_place_whatever_it_asks_for() {
    let (h, _stop) = crawl_in_line(|_| {}).await;
    crawl_budget(&h, h.miner.ss58(), 3).await;
    for _ in 0..3 {
        one_crawl_task(&h).await;
    }
    assert_eq!(tasks(&h.miner.post("/v1/tasks/claim", crawl(0, 3)).await.unwrap()), 1, "one place, one task");
    assert_eq!(tasks(&h.miner.post("/v1/tasks/claim", crawl(0, 3)).await.unwrap()), 1, "asked again, at the back of a line of none");
}

#[tokio::test]
async fn a_crawl_claim_allowed_more_places_takes_a_task_for_each() {
    let (h, _stop) = crawl_in_line(|s| s.crawl_line.places = 2).await;
    crawl_budget(&h, h.miner.ss58(), 3).await;
    for _ in 0..3 {
        one_crawl_task(&h).await;
    }
    assert_eq!(tasks(&h.miner.post("/v1/tasks/claim", crawl(0, 3)).await.unwrap()), 2, "two places allowed");
}

#[tokio::test]
async fn a_crawl_claim_waiting_beside_an_embed_one_is_woken_when_its_task_comes() {
    let (h, _stop) = crawl_in_line(|s| {
        s.embed_tasks = true;
        s.embed_line.on = true;
    })
    .await;
    let client = h.client(MINER, &h.url);
    let embed = tokio::spawn(async move { client.post("/v1/tasks/claim", body(6, 1)).await.unwrap() });
    settle().await;
    let client = h.client(MINER, &h.url);
    let crawling = tokio::spawn(async move { client.post("/v1/tasks/claim", crawl(15, 1)).await.unwrap() });
    settle().await;
    one_crawl_task(&h).await;
    let released = Instant::now();
    let answer = answered(crawling).await;
    assert_eq!(tasks(&answer), 1, "{answer}");
    assert!(released.elapsed() < Duration::from_secs(1), "woken in {:?}, not left to its own check every 3 s", released.elapsed());
    assert!(!embed.is_finished(), "the embed claim waits on");
    let embed = tokio::time::timeout(Duration::from_secs(8), embed).await.expect("answered when its wait ends").unwrap();
    assert_eq!((embed["refusal"]["code"].clone(), embed["refusal"]["inputs"]["retry_after"].clone()), (json!("QUEUE_EMPTY"), json!(0)));
}

#[tokio::test]
async fn a_miner_gone_from_its_crawl_wait_lets_go_of_its_crawl_place_only() {
    let (h, _stop) = crawl_in_line(|s| {
        s.embed_tasks = true;
        s.embed_line.on = true;
    })
    .await;
    assert_eq!(poll(&h.miner).await["refusal"]["inputs"]["ahead"], 0, "an embed place");
    let client = h.client(MINER, &h.url);
    let claim = tokio::spawn(async move { client.post("/v1/tasks/claim", crawl(15, 1)).await });
    settle().await;
    assert_eq!(h.rival.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()["refusal"]["inputs"]["ahead"], 1);
    claim.abort();
    settle().await;
    assert_eq!(h.rival.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap()["refusal"]["inputs"]["ahead"], 0, "its crawl place is let go");
    assert_eq!(poll(&h.rival).await["refusal"]["inputs"]["ahead"], 1, "its embed place stands");
}

#[tokio::test]
async fn with_the_crawl_line_on_a_hold_on_waiting_does_not_apply_to_crawl() {
    let (h, _stop) = crawl_in_line(|s| s.hold_to_wait = true).await;
    for _ in 0..3 {
        let refused = h.miner.post("/v1/tasks/claim", crawl(0, 1)).await.unwrap();
        assert_eq!(refused["refusal"]["code"], "QUEUE_EMPTY", "in line a poll is told 5 s and keeps its place: {refused}");
    }
    let embed = h.miner.post("/v1/tasks/claim", json!({"kind": "embed", "models": ["desearch-4b-v1"]})).await.unwrap();
    assert_eq!(embed["refusal"]["code"], "KIND_CLOSED");
}

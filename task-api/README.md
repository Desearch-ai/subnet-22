# Desearch Task API

Distributes crawl work to miners, and lets anyone check that the distribution was honest.

Design: [docs/technical/central-api.md](../docs/technical/central-api.md)

## Run the local demo

Needs a Redis on `localhost:6379` (database 15 is used as scratch and flushed).

```bash
cd task-api
python local/run_demo.py
```

It starts the API, opens a round, reveals a seed, turns six miners loose on it — two honest, plus
a hoarder, an omitter, a fabricator and a poll spammer — closes the round, then runs the public
verifier against what happened.

## Verify a round yourself

```bash
python tools/verify_round.py --api http://localhost:8099 --round <round_id>
```

No credentials, and nothing imported from this package: the distribution rule is reimplemented in
the script so you can read all of it before trusting any of it. It checks that the manifest matches
the hash committed before the seed existed, that the serve order is what the rule produces from
that seed, that every batch issued was the earliest one whose hosts were free, that the anchored
root matches the log, and that every refusal states a reason.

## Layout

| Path | What |
| --- | --- |
| `app/ordering.py` | the distribution rule — pure, standard library only |
| `app/queue.py` | Redis lease with atomic per-host locks |
| `app/rounds.py` | packing, commitment, reveal |
| `app/filler.py` | round lifecycle and the janitor that reclaims expired leases |
| `app/roundlog.py` | append-only log, ordered by mutation, anchored per round |
| `app/budget.py` | earned concurrency, with every transition recorded |
| `tools/verify_round.py` | the public verifier |
| `local/` | synthetic frontier, adversarial miners, end-to-end demo |

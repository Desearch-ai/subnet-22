from neurons.validators.scoring import pick_samples, sample_seed

URLS = [f"https://site.example/page/{i}" for i in range(40)]
KEPT = {url: {"error": None} for url in URLS}


def test_the_seed_is_reproducible_and_distinct_per_task():
    assert sample_seed("task", "v") == sample_seed("task", "v", "")
    assert sample_seed("task-7", "v") != sample_seed("task-8", "v")


def test_different_seeds_pick_different_samples():
    first = pick_samples(KEPT, "1f" * 32, 5)
    second = pick_samples(KEPT, "2e" * 32, 5)
    assert set(first) != set(second)
    assert pick_samples(KEPT, "1f" * 32, 5) == first


def test_most_of_the_sample_goes_to_pages_the_miner_returned():
    kept = {
        f"https://s.example/{i}": {
            "url": f"https://s.example/{i}",
            "error": "blocked" if i % 2 else None,
        }
        for i in range(40)
    }
    picked = pick_samples(kept, "seed", 10)

    assert len(picked) == 10
    assert sum(1 for url in picked if kept[url]["error"]) == 2

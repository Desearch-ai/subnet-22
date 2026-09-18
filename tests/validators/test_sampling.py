from neurons.validators import crawl
from neurons.validators.scoring import pick_samples, sample_seed

URLS = [f"https://site.example/page/{i}" for i in range(40)]
KEPT = {url: {"error": None} for url in URLS}


def test_the_unsalted_seed_is_unchanged():
    assert sample_seed("task", "validator") == sample_seed("task", "validator", "")


def test_a_salt_changes_the_seed():
    assert sample_seed("task", "validator", "a") != sample_seed(
        "task", "validator", "b"
    )


def test_a_miner_cannot_predict_the_sample_from_public_values():
    hotkey = "5ValidatorHotkeyIsPublicOnChain"
    predictable = pick_samples(KEPT, sample_seed("task-7", hotkey), 5)
    actual = pick_samples(KEPT, sample_seed("task-7", hotkey, crawl.SAMPLE_SALT), 5)
    assert predictable != actual


def test_the_salt_is_a_fresh_secret():
    assert len(crawl.SAMPLE_SALT) == 64
    assert len(set(crawl.SAMPLE_SALT)) > 4


def test_different_salts_pick_different_samples():
    first = pick_samples(KEPT, sample_seed("task-7", "v", "salt-one"), 5)
    second = pick_samples(KEPT, sample_seed("task-7", "v", "salt-two"), 5)
    assert set(first) != set(second)


def test_the_same_salt_is_reproducible():
    seed = sample_seed("task-7", "v", "fixed")
    assert pick_samples(KEPT, seed, 5) == pick_samples(KEPT, seed, 5)


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

import importlib.util
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "resend_waves.py"
SPEC = importlib.util.spec_from_file_location("resend_waves", SOURCE)
resend_waves = importlib.util.module_from_spec(SPEC)
sys.modules["resend_waves"] = resend_waves
SPEC.loader.exec_module(resend_waves)


def site(host, rule, since=None, why=("academic",)):
    entry = {"host": host, "rule": rule, "why": list(why)}
    if since:
        entry["since"] = since
    return entry


def test_waves_go_benchmark_sites_then_established_news_then_whole_sites_then_the_rest_biggest_first():
    sites = [
        site("floored.com", "news", "2025-10-01", ("news",)),
        site("uni.edu", "whole"),
        site("big.edu", "whole"),
        site("paper.com", "news", why=("wikidata",)),
        site("cited.com", "news", "2025-10-01", ("simpleqa",)),
        site("dump.org", "bulk", why=("frames",)),
        site("gone.com", "out"),
    ]
    pages = {"floored.com": 10, "uni.edu": 5, "big.edu": 50, "paper.com": 20, "cited.com": 1, "dump.org": 99, "gone.com": 99}
    found = resend_waves.waves(sites, pages, most_sites=2, most_pages=1000)
    assert [[host for host, _, _ in wave] for wave in found] == [["cited.com", "paper.com"], ["big.edu", "uni.edu"], ["floored.com"]]
    by_pages = resend_waves.waves(sites, pages, most_sites=10, most_pages=60)
    assert [sum(count for _, count, _ in wave) for wave in by_pages] == [21, 55, 10], "a wave closes before it would pass its pages"


def test_wave_files_are_written_with_a_summary(tmp_path):
    sets = tmp_path / "v1-set.json"
    sets.write_text(json.dumps({"sites": [site("a.edu", "whole"), site("b.com", "news", why=("wikidata",))]}))
    pages = tmp_path / "pages.tsv"
    pages.write_text("host\tpages\na.edu\t3\nb.com\t4\n")
    resend_waves.main(["--set", str(sets), "--pages", str(pages), "--out", str(tmp_path / "waves")])
    assert (tmp_path / "waves" / "resend-wave-01.txt").read_text() == "b.com\na.edu\n"
    assert (tmp_path / "waves" / "waves.tsv").read_text().splitlines()[1] == "1\t2\t7\t0\t1\t1\t0"

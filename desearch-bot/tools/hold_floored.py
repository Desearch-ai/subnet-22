"""Writes domains-v1-hold.json: the v1 crawl list without the floored news sites it adds to the live list, for the live binary, which ignores `since`."""
import json
import sys

LIVE, SET, LIST, OUT, HELD = sys.argv[1:6]
live = {e["host"] for e in json.load(open(LIVE))}
held = {e["host"] for e in json.load(open(SET))["sites"] if e["rule"] == "news" and e.get("since") and e["host"] not in live}
entries = json.load(open(LIST))
kept = [e for e in entries if e["host"] not in held]
with open(f"{OUT}.new", "w") as out:
    json.dump(kept, out, separators=(",", ":"))
open(HELD, "w").write("\n".join(sorted(held)) + "\n")
__import__("os").replace(f"{OUT}.new", OUT)
print(f"{len(kept)} sites in {OUT}; {len(entries) - len(kept)} floored news sites held until the release")

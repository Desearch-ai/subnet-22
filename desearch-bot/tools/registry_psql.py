"""Runs psql on the registry, DESEARCH_DB handed over through libpq's environment so no process listing shows its password."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crawl_list import pg_env  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    env = {key: value for key, value in os.environ.items() if key != "DESEARCH_DB"}
    env.update(pg_env(os.environ["DESEARCH_DB"]))
    os.execvpe(
        "psql",
        [
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            *(sys.argv[1:] if argv is None else argv),
        ],
        env,
    )


if __name__ == "__main__":
    main()

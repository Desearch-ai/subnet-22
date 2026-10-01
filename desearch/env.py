from __future__ import annotations

import os
from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and not name.startswith("#") and value.strip():
            pairs[name.strip()] = value.strip().strip("'\"")
    return pairs


def load(path: Path) -> None:
    for name, value in read_dotenv(path).items():
        if not os.environ.get(name):
            os.environ[name] = value

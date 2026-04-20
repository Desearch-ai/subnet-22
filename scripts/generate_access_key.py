"""Generate a cryptographically strong EXPECTED_ACCESS_KEY for the validator API.

Usage:
    python scripts/generate_access_key.py [--length 24]

Produces a key that passes neurons.validators.env.validate_access_key: at least
16 characters including uppercase, lowercase, digit, and special character.
"""

import argparse
import secrets
import string
import sys

MIN_LENGTH = 16
SPECIAL_CHARS = "-_"


def generate(length: int) -> str:
    if length < MIN_LENGTH:
        raise ValueError(f"Length must be >= {MIN_LENGTH}, got {length}")

    categories = [
        string.ascii_uppercase,
        string.ascii_lowercase,
        string.digits,
        SPECIAL_CHARS,
    ]
    pool = "".join(categories)

    chars = [secrets.choice(c) for c in categories]
    chars += [secrets.choice(pool) for _ in range(length - len(categories))]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length",
        type=int,
        default=24,
        help=f"Key length (min {MIN_LENGTH}, default 24).",
    )
    args = parser.parse_args()

    print(generate(args.length))
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os

PORT = os.environ.get("PORT", 8005)
VALIDATOR_SERVICE_PORT = os.environ.get("VALIDATOR_SERVICE_PORT", 8006)
EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY")

MIN_ACCESS_KEY_LENGTH = 16


def validate_access_key(key: str) -> None:
    """Validate access key strength or raise ValueError."""

    if len(key) < MIN_ACCESS_KEY_LENGTH:
        raise ValueError(
            f"EXPECTED_ACCESS_KEY must be at least {MIN_ACCESS_KEY_LENGTH} chars, got {len(key)}."
        )

    if not any(c.isupper() for c in key):
        raise ValueError(
            "EXPECTED_ACCESS_KEY must contain at least one uppercase letter."
        )

    if not any(c.islower() for c in key):
        raise ValueError(
            "EXPECTED_ACCESS_KEY must contain at least one lowercase letter."
        )

    if not any(c.isdigit() for c in key):
        raise ValueError("EXPECTED_ACCESS_KEY must contain at least one digit.")

    if not any(not c.isalnum() for c in key):
        raise ValueError(
            "EXPECTED_ACCESS_KEY must contain at least one special character."
        )


if EXPECTED_ACCESS_KEY is not None:
    validate_access_key(EXPECTED_ACCESS_KEY)

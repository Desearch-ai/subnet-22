from typing import Annotated

from fastapi import Header, HTTPException

from neurons.validators.env import EXPECTED_ACCESS_KEY


async def verify_access_key(access_key: Annotated[str | None, Header()] = None):
    """FastAPI dependency that validates the access key on every protected endpoint."""

    if EXPECTED_ACCESS_KEY is None:
        raise HTTPException(
            status_code=403,
            detail="Access key is not configured. API is disabled.",
        )

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

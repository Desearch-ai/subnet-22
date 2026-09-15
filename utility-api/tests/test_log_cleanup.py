import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domains.logs import cleanup


def test_delete_expired_logs_deletes_in_batches(monkeypatch):
    monkeypatch.setattr(cleanup, "CLEANUP_BATCH_SIZE", 2)
    session = AsyncMock()
    session.execute.side_effect = [
        SimpleNamespace(rowcount=2),
        SimpleNamespace(rowcount=2),
        SimpleNamespace(rowcount=1),
    ]

    deleted = asyncio.run(cleanup.delete_expired_logs(session))

    assert deleted == 5
    assert session.execute.await_count == 3
    assert session.commit.await_count == 3

    stmt = str(session.execute.await_args_list[0].args[0])
    assert "DELETE FROM miner_response_logs" in stmt
    assert "miner_response_logs.created_at <" in stmt

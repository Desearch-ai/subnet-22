import asyncio
import os
import shutil
import signal
import socket
import subprocess

import asyncpg
import pytest

from desearch_bot import db


@pytest.fixture(scope="session")
def postgres(tmp_path_factory):
    """A throwaway PostgreSQL server for the tests that need one."""
    initdb, server = shutil.which("initdb"), shutil.which("postgres")
    if not (initdb and server):
        pytest.skip("needs a local PostgreSQL install")
    data = tmp_path_factory.mktemp("pgdata")
    # Postgres on macOS refuses to start without a valid locale in the environment.
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    subprocess.run(
        [
            initdb,
            "-D",
            str(data),
            "-U",
            "postgres",
            "--auth=trust",
            "-E",
            "UTF8",
            "--no-locale",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            server,
            "-D",
            str(data),
            "-p",
            str(port),
            "-c",
            "listen_addresses=127.0.0.1",
            "-c",
            "unix_socket_directories=",
            "-c",
            "fsync=off",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    # SIGINT is the fast shutdown; SIGTERM would wait for every client to disconnect.
    process.send_signal(signal.SIGINT)
    process.wait(timeout=30)


@pytest.fixture
async def pool(postgres, monkeypatch):
    """A pool on a freshly created bot schema."""
    monkeypatch.setenv("DESEARCH_DB", postgres)
    for _ in range(100):
        try:
            opened = await db.connect(4)
            break
        except (OSError, asyncpg.CannotConnectNowError):
            await asyncio.sleep(0.1)
    else:
        pytest.fail("PostgreSQL did not start")
    async with opened.acquire() as connection:
        await connection.execute("DROP SCHEMA IF EXISTS bot CASCADE")
    await db.create_schema(opened)
    yield opened
    await opened.close()

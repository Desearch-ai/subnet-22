import os

import pytest

from desearch import env
from tests.memory_r2 import DOTENV, R2_KEYS, Backend


@pytest.fixture(params=["memory", "r2"])
def backend(request, monkeypatch):
    if request.param == "r2":
        if os.environ.get("TASK_API_TEST_R2") != "1":
            pytest.skip("real R2 runs only with TASK_API_TEST_R2=1")
        for name, value in env.read_dotenv(DOTENV).items():
            if name.startswith("CF_R2_") and not os.environ.get(name):
                monkeypatch.setenv(name, value)
        if not all(os.environ.get(name) for name in R2_KEYS):
            pytest.skip("R2 credentials are not configured")
    made = Backend(real=request.param == "r2")
    yield made
    made.cleanup()


@pytest.fixture
def memory():
    made = Backend(real=False)
    yield made
    made.cleanup()


@pytest.fixture
def api_env(monkeypatch, tmp_path):
    from tests.test_api_flow import ADMIN, OTHER_VALIDATOR, THIRD_VALIDATOR, VALIDATOR

    monkeypatch.setenv("TASK_API_DATA", str(tmp_path))
    monkeypatch.setenv("TASK_API_REGISTRY", "local")
    monkeypatch.setenv("TASK_API_SEEDS", "local")
    monkeypatch.setenv("TASK_API_BLOCK_SECONDS", "0.02")
    monkeypatch.setenv("TASK_API_AUDIT_RATE", "0")
    monkeypatch.setenv("TASK_API_READS_PER_MINUTE", "100000")
    monkeypatch.setenv("TASK_API_POLL_RATE", "100")
    monkeypatch.setenv(
        "TASK_API_VALIDATOR_URIS", f"{VALIDATOR},{OTHER_VALIDATOR},{THIRD_VALIDATOR}"
    )
    monkeypatch.setenv("TASK_API_ADMIN_URIS", ADMIN)
    return monkeypatch

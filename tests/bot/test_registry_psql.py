import importlib.util
import sys
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "registry_psql.py"
)
SPEC = importlib.util.spec_from_file_location("registry_psql", SOURCE)
registry_psql = importlib.util.module_from_spec(SPEC)
sys.modules["registry_psql"] = registry_psql
SPEC.loader.exec_module(registry_psql)


def test_psql_runs_with_the_password_in_its_environment_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        registry_psql.os,
        "execvpe",
        lambda file, args, env: seen.update(file=file, args=args, env=env),
    )
    monkeypatch.setenv(
        "DESEARCH_DB", "postgresql://bot:hunter2-pw@db.example.co:6543/postgres"
    )
    registry_psql.main(["-f", "rows.sql"])
    assert seen["args"] == ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-f", "rows.sql"]
    assert not any("hunter2" in arg for arg in seen["args"])
    assert (
        seen["env"]["PGPASSWORD"] == "hunter2-pw" and "DESEARCH_DB" not in seen["env"]
    )

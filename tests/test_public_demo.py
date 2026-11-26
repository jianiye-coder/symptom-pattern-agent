"""Public synthetic-data demo mode (SPA_PUBLIC_DEMO=1)."""
import importlib

from fastapi.testclient import TestClient

from spa.memory import MemoryStore
from spa.synth import generate


def test_purge_inactive_removes_only_stale_users():
    s = MemoryStore()
    s.add_entries("old", generate("pem", 0)[:3])
    s.set_flag("old", "2026-01-05")
    s._conn.execute("UPDATE entries SET created_at='2000-01-01T00:00:00+00:00' WHERE user_id='old'")
    s._conn.execute("UPDATE flags SET created_at='2000-01-01T00:00:00+00:00' WHERE user_id='old'")
    s.add_entries("new", generate("pem", 0)[:3])
    assert s.purge_inactive(24) == 1
    assert s.get_entries("old") == [] and s.flags("old") == [] and s.get_entries("new")


def test_public_demo_forces_mock_sanitizes_ids_and_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("SPA_DB_PATH", str(tmp_path / "pub.db"))
    monkeypatch.setenv("SPA_PUBLIC_DEMO", "1")
    monkeypatch.setenv("SPA_LLM_MODE", "openai")  # must be ignored in public mode
    monkeypatch.setenv("SPA_RATE_PER_MIN", "5")
    import spa.api as api

    importlib.reload(api)
    try:
        c = TestClient(api.app)
        h = c.get("/api/health").json()
        assert h["llm"] == "mock" and h["public_demo"] is True
        assert api._uid("../../etc;drop") == "etcdrop" and api._uid("") == "demo"
        codes = [c.get("/api/personas").status_code for _ in range(6)]
        assert codes[-1] == 429 and codes[0] == 200
        assert c.get("/").status_code == 200  # pages are not rate-limited
    finally:
        monkeypatch.delenv("SPA_PUBLIC_DEMO")
        monkeypatch.delenv("SPA_LLM_MODE")
        importlib.reload(api)

import datetime as dt
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from spa.extraction import llm_extract, rule_extract
from spa.models import LLMResponse
from spa.synth import PERSONAS, generate

ROOT = Path(__file__).resolve().parent.parent


def test_rule_extract_basic():
    e = rule_extract("fatigue 7/10, brain fog 5. slept 5.5 hours. busy day. coffee. took magnesium.")
    assert e.symptoms == {"fatigue": 7, "brain_fog": 5}
    assert e.sleep_hours == 5.5 and e.exertion == 7
    assert e.foods == ["coffee"] and e.meds == ["magnesium"]


def test_llm_extract_repairs_invalid_json_then_validates():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None, json_mode=False):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content='{"symptoms": {"fatigue": 14}}')  # out of range
            return LLMResponse(content='{"symptoms": {"fatigue": 7}, "sleep_hours": 6}')

    ex, meta = llm_extract("tired 7, slept 6h", Flaky())
    assert ex.symptoms == {"fatigue": 7} and meta["attempts"] == 2


def test_fixtures_match_generator():
    for p in PERSONAS:
        doc = json.loads((ROOT / "evals/fixtures/histories" / f"{p}.json").read_text())
        for seed, es in doc["histories"].items():
            gen = [e.model_dump(mode="json", exclude_defaults=True) for e in generate(p, int(seed))]
            assert es == gen, (p, seed)


def test_api_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("SPA_DB_PATH", str(tmp_path / "api.db"))
    import spa.api as api

    importlib.reload(api)
    c = TestClient(api.app)
    assert c.get("/").status_code == 200
    assert c.post("/api/demo/pem").json()["loaded"] > 50
    r = c.post("/api/entries", json={"text": "chest pain and fatigue 8", "date": "2026-04-01"}).json()
    assert r["entry"]["symptoms"]["fatigue"] == 8 and r["red_flags"] == ["chest pain"]
    chat = c.post("/api/chat", json={"message": "What tends to come before my bad days?"}).json()
    assert chat["grounded"] and chat["citations"]
    facts = c.get("/api/memory").json()["facts"]
    assert facts and facts[0]["category"] == "trigger_suspected"
    assert c.delete(f"/api/memory/{facts[0]['id']}").status_code == 200
    chart = c.get("/api/chart?exposure=high_exertion").json()
    assert any(p["exposed"] for p in chart["points"])
    # users are isolated by header
    assert c.get("/api/entries", headers={"X-User-Id": "someone-else"}).json()["entries"] == []

"""Flagged days, logging progress, recent change and the doctor-visit report."""
import datetime as dt
import importlib
import re

from fastapi.testclient import TestClient

from spa import analysis, grounding, report
from spa.memory import MemoryStore
from spa.models import Entry
from spa.synth import generate


def test_flags_are_per_user_and_upsert():
    s = MemoryStore()
    s.set_flag("a", "2026-01-10", "bad crash")
    s.set_flag("a", "2026-01-10", "worst crash")
    s.set_flag("b", "2026-01-11")
    assert s.flags("a") == [{"date": "2026-01-10", "note": "worst crash"}]
    assert s.clear_flag("a", "2026-01-10") and not s.clear_flag("a", "2026-01-10") and s.flags("b")
    s.clear_user("b")
    assert s.flags("b") == []


def test_logging_progress_tracks_gaps_and_detection_chance():
    es = generate("pem", 1)
    p = analysis.logging_progress(es, today=es[-1].date + dt.timedelta(days=3))
    assert p["days_since_last_entry"] == 3 and p["n_days"] == len(es) and p["longest_gap_days"] >= 1
    short = analysis.logging_progress(es[:10])
    assert short["detection_chance_pct"] < p["detection_chance_pct"] and short["next_milestone_days"] == 45
    assert analysis.logging_progress([])["status"] == "empty"


def test_recent_change_needs_an_interval_that_excludes_zero():
    mg = analysis.recent_change(generate("magnesium_effect", 0), "fatigue")
    assert mg["verdict"] == "lower_recently" and mg["ci95_difference"][1] < 0
    flat = [Entry(date=dt.date(2026, 1, 1) + dt.timedelta(days=i), symptoms={"fatigue": 4 + (i % 2)})
            for i in range(50)]
    assert analysis.recent_change(flat)["verdict"] == "no_clear_change"
    assert analysis.recent_change(flat[:10])["status"] == "insufficient_data"


def test_report_contents_are_grounded_in_tool_outputs():
    s = MemoryStore()
    es = generate("pem", 0)
    es[5] = es[5].model_copy(update={"notes": "had chest pain on the stairs"})
    s.add_entries("u", es)
    s.set_flag("u", es[10].date.isoformat(), "couldn't work")
    s.remember("u", "tried_stopped", "ldn", "Tried LDN and stopped.", "user", {"user_quote": "I stopped LDN"})
    r = report.build(s.get_entries("u"), s.facts("u"), s.flags("u"))
    assert r["patterns"]["findings"][0]["exposure"] == "high_exertion"
    assert r["flagged_days"][0]["note"] == "couldn't work" and r["flagged_days"][0]["logged"]
    assert r["red_flag_mentions"][0]["flags"] == ["chest pain"]
    assert r["patient_reported"]["tried_stopped"][0]["quote"] == "I stopped LDN"
    page = report.to_html(r)
    assert "high exertion" in page and "couldn&#x27;t work" in page and "not a diagnosis" in page
    # every number on the page comes from the report's computed values (no LLM, no invented stats)
    text = re.sub(r"<[^>]+>", " ", page)
    sources = [r, "0-10 95% 14 28 4 1 2 3 0.001"]  # fixed labels used in the template
    g = grounding.check(text, sources)
    assert g["grounded"], g["ungrounded_numbers"][:5]


def test_report_escapes_user_text():
    s = MemoryStore()
    es = generate("pem", 0)
    s.add_entries("u", es)
    s.set_flag("u", es[0].date.isoformat(), "<script>alert(1)</script>")
    page = report.to_html(report.build(s.get_entries("u"), [], s.flags("u")))
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_report_and_flags_api(tmp_path, monkeypatch):
    monkeypatch.setenv("SPA_DB_PATH", str(tmp_path / "api.db"))
    import spa.api as api

    importlib.reload(api)
    c = TestClient(api.app)
    c.post("/api/demo/pem")
    d = c.get("/api/entries").json()["entries"][3]["date"]
    assert c.post("/api/flags", json={"date": d, "note": "ER visit"}).status_code == 200
    assert c.get("/api/flags").json()["flags"] == [{"date": d, "note": "ER visit"}]
    assert c.get("/api/progress").json()["n_days"] > 50
    j = c.get("/api/report").json()
    assert j["flagged_days"][0]["date"] == d
    page = c.get("/report?user=demo")
    assert page.status_code == 200 and "ER visit" in page.text
    assert "ER visit" not in c.get("/report?user=someone-else").text
    assert c.delete(f"/api/flags/{d}").status_code == 200 and c.delete(f"/api/flags/{d}").status_code == 404

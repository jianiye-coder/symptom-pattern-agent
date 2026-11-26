from spa.memory import MemoryStore
from spa.synth import generate


def test_supersede_keeps_history(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    a = s.remember("u", "preference", "flare_definition", "fatigue>=7", "user", {"q": 1}, {"threshold": 7})
    b = s.remember("u", "preference", "flare_definition", "fatigue>=8", "user", {"q": 2}, {"threshold": 8})
    active = s.facts("u")
    assert [f.id for f in active] == [b.id] and b.supersedes_id == a.id
    all_ = s.facts("u", include_inactive=True)
    assert {f.status for f in all_} == {"active", "superseded"}
    assert [e["action"] for e in s.events("u")] == ["created", "superseded", "created"]


def test_retract_and_isolation(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    f = s.remember("u", "trigger_confirmed", "coffee", "coffee", "user")
    assert s.retract("other", f.id, "x") is None  # cannot touch another user's fact
    assert s.retract("u", f.id, "x").status == "retracted"
    assert s.facts("u") == []


def test_entries_roundtrip_and_persistence(tmp_path):
    p = tmp_path / "m.db"
    es = generate("pem", 0)
    MemoryStore(p).add_entries("u", es)
    assert MemoryStore(p).get_entries("u") == es

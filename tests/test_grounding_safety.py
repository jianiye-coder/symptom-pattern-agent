from spa import grounding, safety


def test_grounding_accepts_roundings_and_percent():
    tools = [{"rate_exposed_pct": 68.2, "risk_difference": 0.682, "n": 22, "d": "2026-01-18"}]
    assert grounding.check("68% of 22 days, difference 0.68, e.g. 2026-01-18", tools)["grounded"]


def test_grounding_rejects_invented_numbers_and_dates():
    tools = [{"rate_exposed_pct": 68.2, "d": "2026-01-18"}]
    g = grounding.check("93% of the time, worst on 2026-02-02", tools)
    assert not g["grounded"]
    assert g["ungrounded_numbers"] == ["93"]
    assert g["ungrounded_dates"] == ["2026-02-02"]


def test_red_flags():
    assert safety.red_flags("I had chest pain and fainted") == ["chest pain", "fainting"]
    assert safety.red_flags("thinking about ending my life")
    assert not safety.red_flags("fatigue 7, slept 6 hours, short walk")


def test_med_decision_and_unsafe_output():
    assert safety.is_med_decision("Should I stop taking propranolol?")
    assert not safety.is_med_decision("Did magnesium help my fatigue?")
    assert safety.unsafe_spans("You should stop propranolol.")
    assert safety.unsafe_spans("You have ME/CFS.")
    assert not safety.unsafe_spans("Since you started magnesium on 2026-02-01, fatigue was lower.")
    assert not safety.unsafe_spans(safety.MED_POLICY)


def test_slots_fill_values_and_record_bindings():
    refs = {"T1": {"findings": [{"rate_exposed_pct": 44.4, "risk_difference": 0.312, "label": "short sleep"}],
                   "entries": [{"date": "2026-01-05", "symptoms": {"fatigue": 8}}]}}
    g = grounding.check("{{T1.findings.0.label}}: {{T1.findings.0.rate_exposed_pct}}% "
                        "({{ T1.findings.0.risk_difference | pct }} points); fatigue {{T1.entries.0.symptoms.fatigue}} "
                        "on {{T1.entries.0.date}}", list(refs.values()), [], refs)
    assert g["grounded"] and g["answer"] == "short sleep: 44.4% (31 points); fatigue 8 on 2026-01-05"
    assert [b["path"] for b in g["bindings"]][:2] == ["findings.0.label", "findings.0.rate_exposed_pct"]


def test_bad_slots_are_ungrounded():
    refs = {"T1": {"findings": [{"label": "x", "evidence": []}]}}
    for bad in ("{{T2.findings.0.label}}", "{{T1.findings.3.label}}", "{{T1.findings.0}}",
                "{{T1.findings.0.evidence}}", "{{T1.findings.0.label|pct}}"):
        g = grounding.check(f"value {bad}", list(refs.values()), [], refs)
        assert not g["grounded"] and g["ungrounded_numbers"] == [bad], bad


def test_raw_log_values_cannot_pass_as_literals():
    out = {"rate_exposed_pct": 44.4, "entries": [{"date": "2026-01-05", "symptoms": {"fatigue": 8}}]}
    assert grounding.check("44.4% of days", [out])["grounded"]
    assert not grounding.check("8% of days", [out])["grounded"]  # 8 exists only in a raw row
    assert grounding.check("{{T1.entries.0.symptoms.fatigue}} on 2026-01-05", [out], [], {"T1": out})["grounded"]

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

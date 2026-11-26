import datetime as dt

from spa import analysis
from spa.models import Entry
from spa.synth import TRUTH, generate


def test_fisher_matches_known_value():
    # Lady tasting tea: [[3,1],[1,3]] one-sided p = 17/70
    assert abs(analysis.fisher_one_sided(3, 1, 1, 3) - 17 / 70) < 1e-12


def test_pem_pattern_found_at_correct_lag():
    r = analysis.find_flare_precursors(generate("pem", 0))
    assert r["status"] == "pattern_found"
    top = r["findings"][0]
    assert top["exposure"] == "high_exertion"
    assert top["lag_days"] in TRUTH["pem"]["lags"]
    lo, hi = top["ci95_risk_difference"]
    assert lo <= top["risk_difference"] <= hi
    # evidence dates are real logged days and the flare comes after the exposure
    logged = {e.date.isoformat() for e in generate("pem", 0)}
    for ev in top["evidence"]:
        assert ev["flare_date"] in logged
        assert all(d in logged and d <= ev["flare_date"] for d in ev["exposure_dates"])


def test_null_has_no_pattern():
    for seed in range(5):
        r = analysis.find_flare_precursors(generate("null", seed))
        assert r["status"] == "no_pattern", seed


def test_insufficient_data_abstains():
    r = analysis.find_flare_precursors(generate("sparse", 0))
    assert r["status"] == "insufficient_data"
    assert r["findings"] == []


def test_missing_outcome_days_are_not_counted_as_fine():
    es = [Entry(date=dt.date(2026, 1, 1) + dt.timedelta(days=i), exertion=5) for i in range(30)]
    assert analysis.outcome_series(es) == {}
    assert analysis.find_flare_precursors(es)["status"] == "insufficient_data"


def test_before_after_effect_and_null():
    r = analysis.compare_before_after(generate("magnesium_effect", 0), "magnesium", "fatigue")
    assert r["verdict"] == "lower_after" and r["ci95_difference"][1] < 0
    r = analysis.compare_before_after(generate("vitd_null", 0), "vitamin d", "fatigue")
    assert r["verdict"] == "no_clear_change"
    assert analysis.compare_before_after(generate("pem", 0), "ldn")["status"] == "not_found"


def test_ignore_list_resolves_loose_names():
    exps = analysis.exposure_series(generate("pem", 0), ignore=["exercise"])
    assert "high_exertion" not in exps


def test_resolve_exposure():
    avail = ["high_exertion", "short_sleep", "food:red wine", "med:magnesium"]
    assert analysis.resolve_exposure("exercise", avail) == "high_exertion"
    assert analysis.resolve_exposure("poor sleep", avail) == "short_sleep"
    assert analysis.resolve_exposure("red wine", avail) == "food:red wine"
    assert analysis.resolve_exposure("gluten", avail) is None


def test_deterministic():
    a = analysis.find_flare_precursors(generate("pem", 3))
    b = analysis.find_flare_precursors(generate("pem", 3))
    assert a == b


def test_resolve_exposure_whole_words_only():
    avail = ["high_exertion", "short_sleep", "food:tea", "food:red wine"]
    assert analysis.resolve_exposure("steak", avail) is None
    assert analysis.resolve_exposure("restaurant", avail) is None
    assert analysis.resolve_exposure("wine", avail) == "food:red wine"


def test_single_exposure_test_uses_full_scan_correction():
    es = generate("pem", 1)
    full = analysis.find_flare_precursors(es)
    one = analysis.test_association(es, "exercise")
    assert one["tests_run"] == full["tests_run"]


def test_before_after_requires_exact_name():
    assert analysis.compare_before_after(generate("vitd_null", 0), "vitamin")["status"] == "not_found"


def test_bh_adjustment_known_values():
    assert [round(x, 4) for x in analysis._bh([0.01, 0.04, 0.03, 0.2])] == [0.04, 0.0533, 0.0533, 0.2]


def test_fast_fisher_matches_exact():
    for t in [(5, 2, 3, 20), (0, 5, 5, 5), (7, 0, 0, 7), (3, 10, 8, 40)]:
        assert abs(analysis._fisher_fast(*t) - analysis.fisher_one_sided(*t)) < 1e-12


def test_all_methods_find_planted_pattern_and_report_method():
    es = generate("pem", 0)
    for m in analysis.METHODS:
        r = analysis.find_flare_precursors(es, method=m)
        assert r["correction"] == m and r["findings"][0]["exposure"] == "high_exertion", m
    r = analysis.find_flare_precursors(es, method="maxt")
    assert r["permutation_rotations"] > 40
    assert r["findings"][0]["p_adjusted"] == float(f"{1 / (r['permutation_rotations'] + 1):.2g}")  # floor


def test_maxt_cannot_reach_significance_on_short_logs():
    es = generate("pem", 0, days=25)
    r = analysis.find_flare_precursors(es, method="maxt")
    assert r["status"] != "pattern_found" and any("too few" in w for w in r["warnings"])


def test_single_exposure_test_is_adjusted_for_full_scan_under_every_method():
    es = generate("pem", 1)
    for m in analysis.METHODS:
        full = analysis.find_flare_precursors(es, method=m)
        one = analysis.test_association(es, "exercise", method=m)
        f = next(x for x in full["findings"] if x["exposure"] == "high_exertion")
        assert one["findings"][0]["p_adjusted"] == f["p_adjusted"], m

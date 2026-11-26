"""Deterministic analysis tools. Every number the agent reports comes from here.

Pure Python (no numpy) so the statistics are easy to read and audit:
  * lagged association: P(flare on day t+lag | exposure on day t) vs P(flare | no exposure),
    one-sided Fisher exact test, Bonferroni correction across every (exposure, lag) tested,
    bootstrap 95% CI for the risk difference.
  * before/after comparison for an intervention, bootstrap CI for the difference in means.
These are screening heuristics on observational self-tracking data, not causal estimates.
"""
from __future__ import annotations

import datetime as dt
import math
import random
import re
import statistics
from collections import Counter
from typing import Any, Optional

from .models import Entry, normalize_name

MIN_DAYS = 14
MIN_FLARES = 3
MIN_EXPOSED = 5
MIN_RISK_DIFF = 0.15
ALPHA = 0.05
HIGH_EXERTION = 7
SHORT_SLEEP = 6.0
DEFAULT_FLARE = {"symptom": None, "threshold": 7}

EXPOSURE_LABELS = {
    "high_exertion": f"high exertion (>= {HIGH_EXERTION}/10)",
    "short_sleep": f"short sleep (< {SHORT_SLEEP:g} h)",
}


# ----------------------------------------------------------------------------- helpers
def _r(x: float, nd: int = 3) -> float:
    return float(round(x, nd))


def _d(s: str | dt.date) -> dt.date:
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(s)


def flare_label(flare_def: dict) -> str:
    sym = flare_def.get("symptom")
    thr = flare_def.get("threshold", 7)
    return f"{sym} >= {thr}" if sym else f"any symptom >= {thr}"


def outcome_series(entries: list[Entry], flare_def: Optional[dict] = None) -> dict[dt.date, bool]:
    """Map date -> is-flare-day. Days where the outcome wasn't recorded are omitted (missing, not 'fine')."""
    fd = flare_def or DEFAULT_FLARE
    sym = fd.get("symptom")
    thr = int(fd.get("threshold", 7))
    out: dict[dt.date, bool] = {}
    for e in entries:
        if sym:
            if sym in e.symptoms:
                out[e.date] = e.symptoms[sym] >= thr
        elif e.symptoms:
            out[e.date] = max(e.symptoms.values()) >= thr
    return out


def exposure_series(entries: list[Entry], ignore: Optional[list[str]] = None) -> dict[str, dict[dt.date, bool]]:
    ignore_set = {i.lower() for i in (ignore or [])}
    series: dict[str, dict[dt.date, bool]] = {}
    ex = {e.date: e.exertion >= HIGH_EXERTION for e in entries if e.exertion is not None}
    if ex:
        series["high_exertion"] = ex
    sl = {e.date: e.sleep_hours < SHORT_SLEEP for e in entries if e.sleep_hours is not None}
    if sl:
        series["short_sleep"] = sl
    n = len(entries)
    for kind in ("food", "med"):
        counts: Counter[str] = Counter()
        for e in entries:
            counts.update(e.foods if kind == "food" else e.meds)
        for item, c in counts.items():
            # items logged almost every day carry no contrast; very rare items are untestable
            if c < 4 or c > 0.85 * n:
                continue
            series[f"{kind}:{item}"] = {
                e.date: item in (e.foods if kind == "food" else e.meds) for e in entries
            }
    drop = {resolve_exposure(i, list(series)) for i in ignore_set}
    return {k: v for k, v in series.items() if k not in drop}


def exposure_label(name: str) -> str:
    if name in EXPOSURE_LABELS:
        return EXPOSURE_LABELS[name]
    kind, _, item = name.partition(":")
    return f"{item} ({'food/drink' if kind == 'food' else 'med/supplement'})"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def resolve_exposure(query: str, available: list[str]) -> Optional[str]:
    """Map a loose user phrase ('exercise', 'red wine') onto an exposure key.
    Whole-word matching only, so 'steak' never maps to 'tea' and 'restaurant' never to 'rest'."""
    q = query.strip().lower()
    if q in available:
        return q
    qw = _words(q)
    # exact item names first ('red wine' -> food:red wine), longest first
    for a in sorted(available, key=lambda k: -len(k)):
        item = a.split(":")[-1]
        if item == q or (_words(item) and _words(item) <= qw):
            return a
    aliases = {
        "high_exertion": {"exertion", "exercise", "exercising", "activity", "active", "walk", "walking", "workout",
                          "steps", "busy", "overdoing", "overdo", "high_exertion"},
        "short_sleep": {"sleep", "slept", "insomnia", "short_sleep"},
    }
    for key, words in aliases.items():
        if key in available and qw & words:
            return key
    # partial name ('wine' -> food:red wine) only if it is unambiguous
    partial = [a for a in available if qw and qw <= _words(a.split(":")[-1])]
    return partial[0] if len(partial) == 1 else None


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] under the hypergeometric null.
    a = exposed & flare, b = exposed & no flare, c = unexposed & flare, d = unexposed & no flare."""
    n1, k, n = a + b, a + c, a + b + c + d
    denom = math.comb(n, k)
    hi = min(n1, k)
    p = sum(math.comb(n1, x) * math.comb(n - n1, k - x) for x in range(a, hi + 1)) / denom
    return min(1.0, p)


def _rd(pairs: list[tuple[bool, bool]]) -> Optional[float]:
    e = [f for x, f in pairs if x]
    u = [f for x, f in pairs if not x]
    if not e or not u:
        return None
    return sum(e) / len(e) - sum(u) / len(u)


def bootstrap_rd_ci(pairs: list[tuple[bool, bool]], b: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = []
    n = len(pairs)
    for _ in range(b):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        v = _rd(s)
        if v is not None:
            stats.append(v)
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return _r(lo), _r(hi)


def bootstrap_mean_diff_ci(before: list[float], after: list[float], b: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(b):
        sb = [before[rng.randrange(len(before))] for _ in before]
        sa = [after[rng.randrange(len(after))] for _ in after]
        diffs.append(statistics.fmean(sa) - statistics.fmean(sb))
    diffs.sort()
    return _r(diffs[int(0.025 * (b - 1))], 2), _r(diffs[int(0.975 * (b - 1))], 2)


LAG_WINDOWS: list[tuple[int, int]] = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 2)]


def lag_text(lo: int, hi: int) -> str:
    if lo == hi == 0:
        return "same day"
    if lo == hi:
        return f"{lo} day{'s' if lo > 1 else ''} later"
    return f"{lo}-{hi} days later"


def _window_rows(
    exp: dict[dt.date, bool], out: dict[dt.date, bool], lo: int, hi: int
) -> list[tuple[dt.date, bool, bool, list[dt.date]]]:
    """Outcome-indexed rows: for each outcome day t, exposed = exposure on any day in [t-hi, t-lo].
    Rows where none of the window days were logged are dropped (unknown, not unexposed)."""
    rows = []
    for t, flare in out.items():
        window = [t - dt.timedelta(days=k) for k in range(lo, hi + 1)]
        known = [d for d in window if d in exp]
        if not known:
            continue
        hits = [d for d in known if exp[d]]
        rows.append((t, bool(hits), flare, hits))
    return rows


def _test(exp: dict[dt.date, bool], out: dict[dt.date, bool], lo: int, hi: int) -> Optional[dict[str, Any]]:
    rows = _window_rows(exp, out, lo, hi)
    a = sum(1 for _, x, f, _h in rows if x and f)
    b = sum(1 for _, x, f, _h in rows if x and not f)
    c = sum(1 for _, x, f, _h in rows if not x and f)
    d = sum(1 for _, x, f, _h in rows if not x and not f)
    if a + b < 4 or c + d < 4:
        return None
    re_, ru = a / (a + b), c / (c + d)
    return {
        "lag_days": [lo, hi],
        "lag_text": lag_text(lo, hi),
        "n_exposed": a + b,
        "n_unexposed": c + d,
        "flares_after_exposed": a,
        "flares_after_unexposed": c,
        "rate_exposed_pct": _r(100 * re_, 1),
        "rate_unexposed_pct": _r(100 * ru, 1),
        "risk_difference": _r(re_ - ru),
        "p_value": fisher_one_sided(a, b, c, d),
        "_rows": rows,
    }


def sufficiency(entries: list[Entry], flare_def: Optional[dict] = None) -> dict[str, Any]:
    out = outcome_series(entries, flare_def)
    n_days, n_flares = len(out), sum(out.values())
    problems = []
    if n_days < MIN_DAYS:
        problems.append(f"only {n_days} days with the outcome logged (need >= {MIN_DAYS})")
    if n_flares < MIN_FLARES:
        problems.append(f"only {n_flares} flare days (need >= {MIN_FLARES})")
    return {"ok": not problems, "n_days": n_days, "n_flare_days": n_flares, "problems": problems}


# ----------------------------------------------------------------------------- tools
def find_flare_precursors(
    entries: list[Entry],
    flare_def: Optional[dict] = None,
    max_lag: int = 3,
    ignore: Optional[list[str]] = None,
    only: Optional[list[str]] = None,
    correction_m: Optional[int] = None,
) -> dict[str, Any]:
    """Scan every candidate exposure at lags 0..max_lag for association with later flare days."""
    fd = flare_def or DEFAULT_FLARE
    suff = sufficiency(entries, fd)
    base = {
        "flare_definition": flare_label(fd),
        "n_days": suff["n_days"],
        "n_flare_days": suff["n_flare_days"],
    }
    if not suff["ok"]:
        return base | {"status": "insufficient_data", "problems": suff["problems"], "findings": []}
    out = outcome_series(entries, fd)
    exps = exposure_series(entries, ignore)
    if only:
        exps = {k: v for k, v in exps.items() if k in only}
    tests: list[tuple[str, dict]] = []
    for name, ser in exps.items():
        for lo, hi in LAG_WINDOWS:
            if hi > max_lag:
                continue
            t = _test(ser, out, lo, hi)
            if t is not None:
                tests.append((name, t))
    # correction_m lets a single-exposure test be corrected for the full scan it is part of
    m = max(1, len(tests), correction_m or 0)
    by_exp: dict[str, list[dict]] = {}
    for name, t in tests:
        t["p_adjusted"] = min(1.0, t["p_value"] * m)
        t["significant"] = (
            t["p_adjusted"] < ALPHA and t["risk_difference"] >= MIN_RISK_DIFF and t["n_exposed"] >= MIN_EXPOSED
        )
        by_exp.setdefault(name, []).append(t)

    findings, weak = [], []
    for name, ts in by_exp.items():
        sig_ts = [t for t in ts if t["significant"]]
        best = min(sig_ts or ts, key=lambda t: (t["p_value"], -t["risk_difference"]))
        sig = [t["lag_text"] for t in sig_ts]
        rows = best["_rows"]
        evidence = [
            {"exposure_dates": [h.isoformat() for h in sorted(hits)], "flare_date": t.isoformat()}
            for t, x, f, hits in sorted(rows)
            if x and f
        ][:5]
        item = {
            "exposure": name,
            "label": exposure_label(name),
            **{k: v for k, v in best.items() if k not in ("significant", "_rows")},
            "p_value": float(f"{best['p_value']:.2g}"),
            "p_adjusted": float(f"{best['p_adjusted']:.2g}"),
            "significant_windows": sig,
            "evidence": evidence,
            "_pairs": [(x, f) for _, x, f, _h in rows],
        }
        (findings if sig else weak).append(item)
    findings.sort(key=lambda f: (f["p_value"], -f["risk_difference"]))
    weak.sort(key=lambda f: (f["p_value"], -f["risk_difference"]))
    findings = findings[:3]
    top_weak = weak[0] if weak and not findings else None
    for item in findings + ([top_weak] if top_weak else []):
        pairs = item.pop("_pairs")
        item["ci95_risk_difference"] = list(bootstrap_rd_ci(pairs, seed=len(pairs)))
    for item in weak:
        item.pop("_pairs", None)
    warnings = [
        f"{m} exposure/lag combinations were tested; p-values are Bonferroni-adjusted.",
        "Observational self-tracking data: associations are not proof of cause.",
        "Consecutive days are correlated, so p-values are approximate (likely optimistic).",
    ]
    if suff["n_days"] < 30:
        warnings.append(f"Small sample ({suff['n_days']} days): estimates are imprecise.")
    for f in findings:
        if f["n_exposed"] < 8:
            warnings.append(f"{f['label']}: only {f['n_exposed']} exposed days.")
    return base | {
        "status": "pattern_found" if findings else "no_pattern",
        "tests_run": m,
        "ci_level_pct": 95,
        "p_display_floor": 0.001,
        "findings": findings,
        "strongest_non_significant": top_weak,
        "warnings": warnings,
    }


def count_tests(entries: list[Entry], flare_def: Optional[dict] = None, max_lag: int = 3) -> int:
    out = outcome_series(entries, flare_def or DEFAULT_FLARE)
    n = 0
    for ser in exposure_series(entries).values():
        for lo, hi in LAG_WINDOWS:
            if hi <= max_lag and _test(ser, out, lo, hi) is not None:
                n += 1
    return n


def test_association(
    entries: list[Entry], exposure: str, flare_def: Optional[dict] = None, max_lag: int = 3
) -> dict[str, Any]:
    exps = exposure_series(entries)
    key = resolve_exposure(exposure, list(exps))
    if key is None:
        return {
            "status": "unknown_exposure",
            "query": exposure,
            "available_exposures": sorted(exps),
        }
    # Correct for the full scan, so testing exposures one at a time can't dodge Bonferroni.
    full_m = count_tests(entries, flare_def, max_lag)
    res = find_flare_precursors(entries, flare_def, max_lag=max_lag, only=[key], correction_m=full_m)
    res["exposure"] = key
    if res["status"] == "no_pattern" and res.get("strongest_non_significant"):
        res["result"] = res.pop("strongest_non_significant")
    return res


def compare_before_after(
    entries: list[Entry], intervention: str, symptom: Optional[str] = None, min_days_each: int = 7
) -> dict[str, Any]:
    name = intervention.strip().lower()
    meds_all = sorted({m for e in entries for m in e.meds})
    match = name if name in meds_all else next(
        (m for m in meds_all if _words(name) and _words(name) <= _words(m) and len(_words(m)) == len(_words(name))),
        None)
    if match is None:
        return {"status": "not_found", "intervention": name, "meds_in_log": meds_all}
    name = match
    on_days = [e.date for e in entries if name in e.meds]
    start, stop = min(on_days), max(on_days)
    sym = normalize_name(symptom) if symptom else None

    def metric(e: Entry) -> Optional[float]:
        if sym:
            return float(e.symptoms[sym]) if sym in e.symptoms else None
        return float(max(e.symptoms.values())) if e.symptoms else None

    before = [(e.date, metric(e)) for e in entries if e.date < start]
    # "after" = the on-period only (first to last logged dose), not days after stopping
    after = [(e.date, metric(e)) for e in entries if start <= e.date <= stop]
    before = [(d, v) for d, v in before if v is not None]
    after = [(d, v) for d, v in after if v is not None]
    base = {
        "intervention": name,
        "start_date": start.isoformat(),
        "last_dose_date": stop.isoformat(),
        "metric": f"{sym} severity" if sym else "daily max symptom severity",
        "n_before": len(before),
        "n_after": len(after),
    }
    if len(before) < min_days_each or len(after) < min_days_each:
        return base | {
            "status": "insufficient_data",
            "problems": [f"need >= {min_days_each} logged days on each side of {start.isoformat()}"],
        }
    bv, av = [v for _, v in before], [v for _, v in after]
    mb, ma = statistics.fmean(bv), statistics.fmean(av)
    lo, hi = bootstrap_mean_diff_ci(bv, av, seed=len(bv) * 1000 + len(av))
    if hi < 0:
        verdict = "lower_after"
    elif lo > 0:
        verdict = "higher_after"
    else:
        verdict = "no_clear_change"
    warnings = ["Before/after comparisons can't separate the intervention from other changes at the same time."]
    pre7 = [v for d, v in before if (start - d).days <= 7]
    if pre7 and statistics.fmean(pre7) - mb >= 1.0:
        warnings.append(
            "Started during a worse-than-usual stretch; some improvement may be regression to the mean."
        )
    if min(len(bv), len(av)) < 14:
        warnings.append("Fewer than 14 days on one side: interval is wide.")
    return base | {
        "status": "ok",
        "mean_before": _r(mb, 2),
        "mean_after": _r(ma, 2),
        "difference": _r(ma - mb, 2),
        "ci95_difference": [lo, hi],
        "ci_level_pct": 95,
        "verdict": verdict,
        "evidence_dates": [before[-1][0].isoformat(), after[0][0].isoformat()],
        "warnings": warnings,
    }


def data_summary(entries: list[Entry], flare_def: Optional[dict] = None) -> dict[str, Any]:
    fd = flare_def or DEFAULT_FLARE
    if not entries:
        return {"n_days": 0, "status": "empty"}
    out = outcome_series(entries, fd)
    span = (entries[-1].date - entries[0].date).days + 1
    sym_vals: dict[str, list[int]] = {}
    for e in entries:
        for s, v in e.symptoms.items():
            sym_vals.setdefault(s, []).append(v)
    meds: dict[str, list[str]] = {}
    for e in entries:
        for m in e.meds:
            meds.setdefault(m, []).append(e.date.isoformat())
    flare_dates = [d.isoformat() for d, f in sorted(out.items()) if f]
    return {
        "status": "ok",
        "n_days": len(entries),
        "first_date": entries[0].date.isoformat(),
        "last_date": entries[-1].date.isoformat(),
        "coverage_pct": _r(100 * len(entries) / span, 1),
        "flare_definition": flare_label(fd),
        "n_flare_days": len(flare_dates),
        "recent_flare_dates": flare_dates[-8:],
        "symptoms": {
            s: {"days": len(v), "mean": _r(statistics.fmean(v), 2), "max": max(v)} for s, v in sorted(sym_vals.items())
        },
        "meds": {m: {"first": d[0], "last": d[-1], "days": len(d)} for m, d in sorted(meds.items())},
    }


def get_entries(entries: list[Entry], start: Optional[str] = None, end: Optional[str] = None, limit: int = 31) -> dict:
    sel = [e for e in entries if (not start or e.date >= _d(start)) and (not end or e.date <= _d(end))]
    return {
        "n": len(sel),
        "entries": [
            e.model_dump(mode="json", exclude={"raw_text"}, exclude_defaults=True) for e in sel[-limit:]
        ],
        "truncated": len(sel) > limit,
    }


def chart_series(entries: list[Entry], symptom: Optional[str] = None, flare_def: Optional[dict] = None) -> dict:
    out = outcome_series(entries, flare_def)
    sym = normalize_name(symptom) if symptom else None
    pts = []
    for e in entries:
        v = (e.symptoms.get(sym) if sym else (max(e.symptoms.values()) if e.symptoms else None))
        pts.append(
            {
                "date": e.date.isoformat(),
                "severity": v,
                "exertion": e.exertion,
                "sleep_hours": e.sleep_hours,
                "flare": out.get(e.date, False),
            }
        )
    return {"symptom": sym or "max symptom", "points": pts}

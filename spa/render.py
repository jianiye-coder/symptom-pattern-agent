"""Deterministic text rendering of tool outputs.

Used (a) by the mock LLM to compose its final answer and (b) by the harness as the fallback
answer when a real model's answer fails the groundedness check. Every number printed here is
read straight from a tool result.
"""
from __future__ import annotations

from typing import Any


def _ci(ci: list[float], pct: bool = False) -> str:
    if pct:
        return f"{ci[0] * 100:.0f} to {ci[1] * 100:.0f} percentage points"
    return f"{ci[0]:+.2f} to {ci[1]:+.2f}"


def _p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"= {x:.3f}".rstrip("0").rstrip(".")


def _evidence(ev: list[dict]) -> str:
    parts = []
    for e in ev[:3]:
        parts.append(f"{', '.join(e['exposure_dates'])} -> flare {e['flare_date']}")
    return "; ".join(parts)


def render_finding(f: dict, lead: str = "") -> str:
    return (
        f"{lead}**{f['label']}** ({f['lag_text']}): a flare followed {f['flares_after_exposed']} of "
        f"{f['n_exposed']} exposed days ({f['rate_exposed_pct']:g}%) vs {f['flares_after_unexposed']} of "
        f"{f['n_unexposed']} other days ({f['rate_unexposed_pct']:g}%). Difference "
        f"{f['risk_difference'] * 100:.0f} percentage points (95% CI {_ci(f['ci95_risk_difference'], pct=True)}; "
        f"adjusted p {_p(f['p_adjusted'])}). Examples: {_evidence(f['evidence'])}."
    )


def render_precursors(r: dict) -> str:
    if r.get("status") == "insufficient_data":
        return (
            f"**Not enough data yet** to look for patterns honestly (flare = {r.get('flare_definition')}): "
            + "; ".join(r.get("problems", []))
            + ". Keep logging and ask again in a couple of weeks."
        )
    head = f"I looked at {r['n_days']} logged days, {r['n_flare_days']} of them flare days ({r['flare_definition']})."
    lines = [head]
    if r["status"] == "pattern_found":
        lines.append("What tends to come before your flares:")
        for f in r["findings"]:
            lines.append("- " + render_finding(f))
    else:
        lines.append(
            f"**I don't see a reliable pattern.** None of the {r['tests_run']} exposure/timing combinations I "
            "tested stood out after correcting for multiple comparisons."
        )
        w = r.get("strongest_non_significant")
        if w:
            lines.append(
                f"The strongest weak signal was {w['label']} ({w['lag_text']}), adjusted p {_p(w['p_adjusted'])}; "
                "that is consistent with chance."
            )
    for c in r.get("user_confirmed_triggers", []):
        verdict = "your data supports this" if c["supported_by_data"] else "your current data doesn't clearly show it"
        lines.append(f"- You told me **{c['subject']}** is a trigger for you (memory #{c['fact_id']}); {verdict}.")
    if r.get("applied_memory"):
        lines.append(
            "Memory applied: " + "; ".join(f"#{m['fact_id']} {m['statement'].rstrip('.')}" for m in r["applied_memory"]) + "."
        )
    lines.append("Caveats: " + " ".join(r.get("warnings", [])))
    return "\n".join(lines)


def render_association(r: dict) -> str:
    if r.get("status") == "unknown_exposure":
        avail = ", ".join(r.get("available_exposures", [])) or "none"
        return f"I can't find **{r['query']}** often enough in your log to test it. Testable exposures: {avail}."
    if r.get("status") == "insufficient_data":
        return render_precursors(r)
    if r["status"] == "pattern_found":
        return render_finding(r["findings"][0], lead="Yes, in your data: ") + " Caveats: " + " ".join(r["warnings"])
    w = r.get("result")
    base = f"**No clear link** between {r['exposure']} and flares ({r['flare_definition']}) in {r['n_days']} days."
    if w:
        base += (
            f" Flares followed {w['flares_after_exposed']} of {w['n_exposed']} exposed days ({w['rate_exposed_pct']:g}%) "
            f"vs {w['rate_unexposed_pct']:g}% otherwise (adjusted p {_p(w['p_adjusted'])})."
        )
    return base


def render_before_after(r: dict) -> str:
    if r.get("status") == "not_found":
        meds = ", ".join(r.get("meds_in_log", [])) or "none"
        return f"I can't find **{r['intervention']}** in your log. Meds/supplements I see: {meds}."
    if r.get("status") == "insufficient_data":
        return (
            f"**Not enough data** to compare before and after {r['intervention']} (started {r['start_date']}): "
            f"{r['n_before']} days before, {r['n_after']} after. " + "; ".join(r.get("problems", [])) + "."
        )
    verdict = {
        "lower_after": "was **lower after**",
        "higher_after": "was **higher after**",
        "no_clear_change": "showed **no clear change**",
    }[r["verdict"]]
    return (
        f"Since you started {r['intervention']} on {r['start_date']}, your {r['metric']} {verdict}: "
        f"mean {r['mean_before']:g} over {r['n_before']} days before vs {r['mean_after']:g} over {r['n_after']} days "
        f"after (difference {r['difference']:+g}, 95% CI {_ci(r['ci95_difference'])}). "
        f"Caveats: {' '.join(r['warnings'])}"
    )


def render_summary(r: dict) -> str:
    if r.get("n_days", 0) == 0:
        return "You haven't logged anything yet."
    syms = ", ".join(f"{k} avg {v['mean']:g}" for k, v in r["symptoms"].items())
    return (
        f"You've logged {r['n_days']} days from {r['first_date']} to {r['last_date']} "
        f"({r['coverage_pct']:g}% of days). {r['n_flare_days']} were flare days ({r['flare_definition']}). "
        f"Symptoms: {syms}. Recent flares: {', '.join(r['recent_flare_dates']) or 'none'}."
    )


def render_memory(r: dict) -> str:
    facts = r.get("facts", [])
    if not facts:
        return "I don't have any saved facts about you yet."
    return "Here's what I remember:\n" + "\n".join(
        f"- #{f['id']} [{f['category']}] {f['statement']} (source: {f['source']})" for f in facts
    )


def render_remember(r: dict) -> str:
    if not r.get("ok"):
        return f"I couldn't save that: {r.get('error')}."
    f = r["fact"]
    return f"Saved to memory (#{f['id']}, {f['category']}): {f['statement']}"


RENDERERS = {
    "find_flare_precursors": render_precursors,
    "test_association": render_association,
    "compare_before_after": render_before_after,
    "data_summary": render_summary,
    "read_memory": render_memory,
    "remember_fact": render_remember,
    "retract_fact": lambda r: (f"Removed memory #{r['fact']['id']}." if r.get("ok") else f"Couldn't remove: {r.get('error')}"),
    "get_entries": lambda r: f"Fetched {r['n']} entries.",
}


def render_all(results: list[tuple[str, Any]]) -> str:
    parts = []
    for name, res in results:
        if isinstance(res, dict) and res.get("tool_error"):
            parts.append(f"(tool {name} failed: {res.get('error')})")
            continue
        fn = RENDERERS.get(name)
        if fn:
            parts.append(fn(res))
    return "\n\n".join(parts) if parts else "I wasn't able to compute anything for that question."

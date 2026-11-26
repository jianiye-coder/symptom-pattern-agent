"""Statistics engine study (tools only, no LLM): how the multiple-comparison method and the amount of
logged data change what the scan finds.

    python -m evals.power            # ~3-5 min; writes evals/results/power.{json,md} and power_curve.svg
    python -m evals.power --seeds 20 # faster, noisier

Two parts:
1. Method comparison at the default history length (70 days): detection rate for each planted persona
   and false-finding rate for an independent null and an autocorrelated null (busy spells and multi-day
   crash episodes that are unrelated to each other).
2. Power curve: detection rate vs. number of days in the history, per method.

Seeds 200+ are used, which are disjoint from the fixtures (0-9) and the CI power suite (100-129).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from spa import analysis
from spa.synth import TRUTH, generate

from .run import RESULTS

PLANTED = ("pem", "sleep_headache", "wine_headache")
NULLS = ("null", "null_autocorr")
DAYS = (21, 30, 45, 60, 90, 120)
CURVE_METHODS = ("bonferroni", "maxt")
LABELS = {"bonferroni": "Bonferroni", "bh": "Benjamini-Hochberg", "maxt": "Circular-shift maxT"}
COLORS = {"bonferroni": "#2a78d6", "maxt": "#eb6834"}  # validated categorical slots 1-2 (dataviz palette)


def _rate(persona: str, method: str, seeds: range, days: int = 70) -> float:
    truth = TRUTH[persona]["exposure"]
    hits = 0
    for s in seeds:
        fs = analysis.find_flare_precursors(generate(persona, s, days=days), method=method)["findings"]
        hits += bool(fs) if truth is None else bool(fs and fs[0]["exposure"] == truth)
    return round(hits / len(seeds), 3)


def study(n_seeds: int) -> dict:
    seeds = range(200, 200 + n_seeds)
    methods = {m: {p: _rate(p, m, seeds) for p in PLANTED + NULLS} for m in analysis.METHODS}
    curve = {m: {p: {str(d): _rate(p, m, seeds, d) for d in DAYS} for p in PLANTED} for m in CURVE_METHODS}
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "n_seeds": n_seeds,
            "default_method": analysis.METHOD, "method_comparison": methods, "power_curve": curve}


def to_markdown(r: dict) -> str:
    mc, pc = r["method_comparison"], r["power_curve"]
    out = [f"Statistics engine study, {r['n_seeds']} fresh seeds per cell (tools only). Default method: "
           f"`{r['default_method']}`.", "",
           "Method comparison at 70 days (detection = planted exposure is the top finding; false finding = any "
           "finding on a null history):", "",
           "| Method | pem | sleep_headache | wine_headache | False finding, null | False finding, autocorrelated null |",
           "|---|---|---|---|---|---|"]
    for m in analysis.METHODS:
        v = mc[m]
        out.append(f"| {LABELS[m]} | {v['pem']} | {v['sleep_headache']} | {v['wine_headache']} | {v['null']} | "
                   f"{v['null_autocorr']} |")
    out += ["", "Detection rate vs. days of history (before ~8% random gaps):", "",
            "| Persona | Method | " + " | ".join(f"{d} d" for d in DAYS) + " |", "|---|---|" + "---|" * len(DAYS)]
    for p in PLANTED:
        for m in CURVE_METHODS:
            out.append(f"| {p} | {LABELS[m]} | " + " | ".join(str(pc[m][p][str(d)]) for d in DAYS) + " |")
    out += ["", "![Detection rate vs days logged](power_curve.svg)"]
    return "\n".join(out) + "\n"


def to_svg(r: dict) -> str:
    """Small multiples (one panel per persona), shared 0-1 y axis, 2px lines, 8px markers, legend + direct labels."""
    pc = r["power_curve"]
    pw, ph, ml, mt, gap = 270, 170, 44, 84, 30
    W = ml + len(PLANTED) * (pw + gap)
    H = mt + ph + 56
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    xmin, xmax = DAYS[0], DAYS[-1]
    sx = lambda d: (d - xmin) / (xmax - xmin) * pw  # noqa: E731
    sy = lambda v: ph - v * ph  # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" '
         f'aria-label="Detection rate versus days logged, by persona and correction method">',
         f'<rect width="{W}" height="{H}" fill="{surf}"/>',
         f'<text x="{ml}" y="20" font-size="14" font-weight="600" fill="{ink}">How many days of logging '
         f'before a planted pattern is found</text>']
    lx = ml
    for m in CURVE_METHODS:  # legend
        o.append(f'<line x1="{lx}" y1="36" x2="{lx + 18}" y2="36" stroke="{COLORS[m]}" stroke-width="2"/>'
                 f'<circle cx="{lx + 9}" cy="36" r="4" fill="{COLORS[m]}" stroke="{surf}" stroke-width="2"/>'
                 f'<text x="{lx + 24}" y="40" font-size="12" fill="{ink2}">{LABELS[m]}</text>')
        lx += 190
    for i, p in enumerate(PLANTED):
        x0, y0 = ml + i * (pw + gap), mt
        o.append(f'<g transform="translate({x0},{y0})">')
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            o.append(f'<line x1="0" y1="{sy(v):.1f}" x2="{pw}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
            if i == 0:
                o.append(f'<text x="-8" y="{sy(v) + 4:.1f}" font-size="11" fill="{ink2}" text-anchor="end">'
                         f'{int(v * 100)}%</text>')
        for d in DAYS:
            o.append(f'<text x="{sx(d):.1f}" y="{ph + 16}" font-size="11" fill="{ink2}" text-anchor="middle">{d}</text>')
        o.append(f'<text x="{pw / 2}" y="{ph + 34}" font-size="11" fill="{ink2}" text-anchor="middle">days logged</text>')
        o.append(f'<text x="0" y="-14" font-size="12" font-weight="600" fill="{ink}">{p.replace("_", " ")}</text>')
        for m in CURVE_METHODS:
            pts = [(sx(d), sy(pc[m][p][str(d)])) for d in DAYS]
            o.append(f'<polyline fill="none" stroke="{COLORS[m]}" stroke-width="2" stroke-linejoin="round" '
                     f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
            for d, (x, y) in zip(DAYS, pts):
                o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[m]}" stroke="{surf}" stroke-width="2">'
                         f'<title>{p}, {LABELS[m]}, {d} days: {pc[m][p][str(d)]:.0%}</title></circle>')
        if i == 0:  # direct labels on the first panel, where the two lines are apart
            vb, vm = pc["bonferroni"][p]["30"], pc["maxt"][p]["30"]
            o.append(f'<text x="{sx(30) + 10:.1f}" y="{sy(vb) + 16:.1f}" font-size="11" fill="{ink2}">Bonferroni</text>')
            o.append(f'<text x="{sx(30) + 10:.1f}" y="{sy(vm) - 8:.1f}" font-size="11" fill="{ink2}">'
                     'maxT: no result can reach p &lt; 0.05 yet</text>')
        o.append("</g>")
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    t0 = time.perf_counter()
    r = study(args.seeds)
    r["wall_s"] = round(time.perf_counter() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "power.json").write_text(json.dumps(r, indent=2))
    (RESULTS / "power.md").write_text(to_markdown(r))
    (RESULTS / "power_curve.svg").write_text(to_svg(r))
    print(to_markdown(r))


if __name__ == "__main__":
    main()

"""Doctor-visit report: one printable page that summarizes the log for an appointment.

Built entirely from the deterministic tools, the memory store and the user's flagged days. No LLM is
involved, so every number in it is grounded by construction. The report describes the data; it does not
diagnose or advise, and it says what the data cannot show.

(Idea borrowed from "doctor-ready summary" symptom trackers such as Silent Symptom, which compile a log
into a clinical-style report. This version adds the pattern statistics, before/after comparisons with
intervals, what the patient told the agent, and red-flag mentions, and keeps every claim traceable.)
"""
from __future__ import annotations

import datetime as dt
import html
from typing import Any, Optional

from . import analysis, safety
from .models import Entry, Fact

DISCLAIMER = (
    "Generated from patient-reported, self-tracked data by a pattern-screening tool. It is not a diagnosis. "
    "Associations in self-tracking data are not proof of cause, and no medication change is suggested."
)


def build(entries: list[Entry], facts: list[Fact], flags: list[dict], flare_def: Optional[dict] = None,
          ignore: Optional[list[str]] = None, today: Optional[dt.date] = None) -> dict[str, Any]:
    fd = flare_def or analysis.DEFAULT_FLARE
    today = today or dt.date.today()
    by_date = {e.date.isoformat(): e for e in entries}
    summary = analysis.data_summary(entries, fd)
    progress = analysis.logging_progress(entries, today)
    if not entries:
        return {"generated": today.isoformat(), "empty": True, "summary": summary, "progress": progress,
                "disclaimer": DISCLAIMER}

    flagged = []
    for f in flags:
        e = by_date.get(f["date"])
        flagged.append({"date": f["date"], "note": f.get("note", ""),
                        "symptoms": e.symptoms if e else {}, "sleep_hours": e.sleep_hours if e else None,
                        "exertion": e.exertion if e else None, "log_text": (e.raw_text or e.notes) if e else "",
                        "logged": e is not None})

    patterns = analysis.find_flare_precursors(entries, fd, ignore=ignore)
    trend = analysis.recent_change(entries, fd.get("symptom"))

    meds = []
    for med in sorted({m for e in entries for m in e.meds}):
        r = analysis.compare_before_after(entries, med)
        meds.append(r)

    red = []
    for e in entries:
        text = " ".join(t for t in (e.raw_text, e.notes) if t)
        fl = safety.red_flags(text)
        if fl:
            red.append({"date": e.date.isoformat(), "flags": fl, "text": text[:200]})

    told = {c: [{"id": f.id, "statement": f.statement, "since": f.created_at[:10],
                 "quote": f.provenance.get("user_quote")} for f in facts if f.category == c]
            for c in ("trigger_confirmed", "tried_stopped", "tried_ongoing", "preference", "context")}
    suspected = [{"id": f.id, "statement": f.statement, "evidence_run": f.provenance.get("run_id"),
                  "p_adjusted": f.provenance.get("p_adjusted")} for f in facts if f.category == "trigger_suspected"]

    return {
        "generated": today.isoformat(),
        "empty": False,
        "summary": summary,
        "progress": progress,
        "flagged_days": flagged,
        "recent_change": trend,
        "patterns": patterns,
        "medications": meds,
        "red_flag_mentions": red,
        "patient_reported": told,
        "agent_suspected": suspected,
        "disclaimer": DISCLAIMER,
    }


# ----------------------------------------------------------------------------- rendering
def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _p(x: float) -> str:
    return "&lt; 0.001" if x < 0.001 else f"{x:.3g}"


def _sym(d: dict) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(d.items())) or "-"


def to_html(r: dict, title: str = "Symptom log summary") -> str:
    s, pg = r["summary"], r["progress"]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_e(title)}</title>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;"
           "0,6..72,500;1,6..72,400&family=Public+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap'>",
           "<link rel='stylesheet' href='/static/report.css?v=3'></head><body><article class='report'>",
           "<header><div><h1>Symptom log summary</h1><p class='sub'>For discussion at a medical appointment</p></div>",
           f"<div class='meta'>Generated {_e(r['generated'])}<br><button onclick='window.print()' class='noprint'>"
           "Print / save as PDF</button></div></header>"]
    if r.get("empty"):
        out.append("<p>No entries logged yet.</p></article></body></html>")
        return "".join(out)

    out.append("<section><h2>Overview</h2><dl class='grid'>"
               f"<div><dt>Logging period</dt><dd>{_e(s['first_date'])} to {_e(s['last_date'])}</dd></div>"
               f"<div><dt>Days logged</dt><dd>{s['n_days']} ({s['coverage_pct']:g}% of days)</dd></div>"
               f"<div><dt>Flare days</dt><dd>{s['n_flare_days']} <span class='muted'>({_e(s['flare_definition'])})"
               "</span></dd></div>"
               f"<div><dt>Longest gap</dt><dd>{pg['longest_gap_days']} days</dd></div></dl>")
    if s["symptoms"]:
        out.append("<table><thead><tr><th>Symptom (0-10)</th><th>Days rated</th><th>Mean</th><th>Max</th></tr>"
                   "</thead><tbody>")
        for k, v in s["symptoms"].items():
            out.append(f"<tr><td>{_e(k.replace('_', ' '))}</td><td>{v['days']}</td><td>{v['mean']:g}</td>"
                       f"<td>{v['max']}</td></tr>")
        out.append("</tbody></table>")
    if s.get("recent_flare_dates"):
        out.append(f"<p class='small'>Most recent flare days: {_e(', '.join(s['recent_flare_dates']))}</p>")
    out.append("</section>")

    if r["red_flag_mentions"]:
        out.append("<section class='alert'><h2>Mentions that may need prompt attention</h2><ul>")
        for m in r["red_flag_mentions"]:
            out.append(f"<li><b>{_e(m['date'])}</b>: {_e(', '.join(m['flags']))}. <q>{_e(m['text'])}</q></li>")
        out.append("</ul><p class='small'>Found by keyword screen of the log text; may include negations "
                   "(\"no chest pain\").</p></section>")

    out.append("<section><h2>Days the patient flagged</h2>")
    if r["flagged_days"]:
        out.append("<table><thead><tr><th>Date</th><th>Patient note</th><th>Symptoms</th><th>Sleep</th>"
                   "<th>Exertion</th></tr></thead><tbody>")
        for f in r["flagged_days"]:
            sl = f"{f['sleep_hours']:g} h" if f["sleep_hours"] is not None else "-"
            out.append(f"<tr><td>{_e(f['date'])}</td><td>{_e(f['note']) or '-'}</td><td>{_e(_sym(f['symptoms']))}"
                       f"{'' if f['logged'] else ' (no log entry)'}</td><td>{sl}</td>"
                       f"<td>{_e(f['exertion']) if f['exertion'] is not None else '-'}</td></tr>")
        out.append("</tbody></table>")
    else:
        out.append("<p class='muted'>None. Flag a day on the timeline to include it here.</p>")
    out.append("</section>")

    t = r["recent_change"]
    out.append("<section><h2>Recent change</h2>")
    if t["status"] == "ok":
        word = {"higher_recently": "higher", "lower_recently": "lower", "no_clear_change": "not clearly different"}
        out.append(f"<p>{_e(t['metric'].capitalize())} over the last 14 days ({_e(t['recent_window'][0])} to "
                   f"{_e(t['recent_window'][1])}) was <b>{word[t['verdict']]}</b> than the 28 days before: mean "
                   f"{t['mean_recent']:g} vs {t['mean_prior']:g} (difference {t['difference']:+g}, 95% CI "
                   f"{t['ci95_difference'][0]:+g} to {t['ci95_difference'][1]:+g}).</p>")
    else:
        out.append(f"<p class='muted'>Not enough data: {_e('; '.join(t.get('problems', [])))}.</p>")
    out.append("</section>")

    p = r["patterns"]
    out.append("<section><h2>What tends to come before flares</h2>")
    if p["status"] == "insufficient_data":
        out.append(f"<p class='muted'>Not enough data to screen for patterns: {_e('; '.join(p['problems']))}.</p>")
    elif p["status"] == "no_pattern":
        out.append(f"<p>No exposure stood out after correcting for the {p['tests_run']} exposure/timing combinations "
                   "tested.</p>")
    else:
        out.append("<table><thead><tr><th>Exposure</th><th>Timing</th><th>Flare after exposure</th>"
                   "<th>Flare otherwise</th><th>Difference (95% CI)</th><th>Adjusted p</th></tr></thead><tbody>")
        for f in p["findings"]:
            ci = f["ci95_risk_difference"]
            out.append(f"<tr><td>{_e(f['label'])}</td><td>{_e(f['lag_text'])}</td>"
                       f"<td>{f['flares_after_exposed']}/{f['n_exposed']} ({f['rate_exposed_pct']:g}%)</td>"
                       f"<td>{f['flares_after_unexposed']}/{f['n_unexposed']} ({f['rate_unexposed_pct']:g}%)</td>"
                       f"<td>{f['risk_difference'] * 100:+.0f} pts ({ci[0] * 100:+.0f} to {ci[1] * 100:+.0f})</td>"
                       f"<td>{_p(f['p_adjusted'])}</td></tr>")
        out.append("</tbody></table>")
    if p.get("warnings"):
        out.append(f"<p class='small muted'>{_e(' '.join(p['warnings']))}</p>")
    out.append("</section>")

    out.append("<section><h2>Medications and supplements: before vs. during</h2>")
    ok = [m for m in r["medications"] if m.get("status") == "ok"]
    other = [m for m in r["medications"] if m.get("status") != "ok"]
    if ok:
        out.append("<table><thead><tr><th>Name</th><th>Started</th><th>Metric</th><th>Before</th><th>During</th>"
                   "<th>Difference (95% CI)</th></tr></thead><tbody>")
        word = {"lower_after": "lower", "higher_after": "higher", "no_clear_change": "no clear change"}
        for m in ok:
            ci = m["ci95_difference"]
            out.append(f"<tr><td>{_e(m['intervention'])}</td><td>{_e(m['start_date'])}</td><td>{_e(m['metric'])}</td>"
                       f"<td>{m['mean_before']:g} (n={m['n_before']})</td><td>{m['mean_after']:g} (n={m['n_after']})"
                       f"</td><td>{m['difference']:+g} ({ci[0]:+g} to {ci[1]:+g}): {word[m['verdict']]}</td></tr>")
        out.append("</tbody></table>")
    if other:
        out.append("<p class='small muted'>Not enough days before the first logged dose to compare: "
                   f"{_e(', '.join(m['intervention'] for m in other))}.</p>")
    out.append("<p class='small muted'>Before/after comparisons can't separate a treatment from anything else "
               "that changed at the same time.</p></section>")

    told = r["patient_reported"]
    labels = {"trigger_confirmed": "Triggers the patient is confident about", "tried_stopped": "Tried and stopped",
              "tried_ongoing": "Currently trying", "preference": "How the patient defines a bad day / exclusions",
              "context": "Other context"}
    if any(told.values()) or r["agent_suspected"]:
        out.append("<section><h2>What the patient reported</h2>")
        for c, items in told.items():
            if items:
                out.append(f"<h3>{_e(labels[c])}</h3><ul>")
                for f in items:
                    q = f" <q>{_e(f['quote'])}</q>" if f.get("quote") else ""
                    out.append(f"<li>{_e(f['statement'])}{q} <span class='muted'>(since {_e(f['since'])})</span></li>")
                out.append("</ul>")
        if r["agent_suspected"]:
            out.append("<h3>Patterns the tool suggested (not confirmed by the patient)</h3><ul>")
            for f in r["agent_suspected"]:
                out.append(f"<li>{_e(f['statement'])} <span class='muted'>(adjusted p {_e(f['p_adjusted'])})</span>"
                           "</li>")
            out.append("</ul>")
        out.append("</section>")

    out.append(f"<footer><p>{_e(r['disclaimer'])}</p><p class='muted'>Data completeness: {pg['n_days']} of "
               f"{pg['span_days']} days logged; last entry {_e(pg['last_date'])}.</p></footer>")
    out.append("</article></body></html>")
    return "".join(out)

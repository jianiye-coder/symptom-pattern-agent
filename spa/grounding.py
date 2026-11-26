"""Groundedness checks: every number and date in an answer must be traceable to a tool output
(or to text the user/memory supplied).

Two ways for a number to be grounded:

1. **Slots (preferred).** The model writes a placeholder such as ``{{T1.findings.0.rate_exposed_pct}}``
   that points at one field of one tool result. The harness fills in the value, so the model never
   types the number and cannot get it wrong, and every number in the answer is bound to a named
   field (returned as ``bindings``). Filters: ``|pct`` (x100, 0 decimals), ``|r0``, ``|r1``, ``|r2``.
2. **Literals.** A literal number passes only if it is a rounding of a value in a *computed* field of
   a tool output (also accepting proportion <-> percent), the user's message or memory. Raw log
   data (``get_entries`` rows) is excluded from this pool: a severity of 8 copied from some day can
   no longer pass as "8% of days". Raw values can still be quoted, but only through a slot.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
# Non-ISO dates can't be verified against the log, so they count as ungrounded.
LOOSE_DATE_RE = re.compile(rf"\b{MONTHS} \d{{1,2}}(st|nd|rd|th)?\b|\b\d{{1,2}}(st|nd|rd|th)? (of )?{MONTHS}|"
                           r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
# Names containing digits that are not quantities.
NAME_TOKENS = re.compile(r"\b(covid-19|b12|b6|coq10|omega-3|24/7|me/cfs)\b", re.I)
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
SLOT_RE = re.compile(r"\{\{\s*(T\d+)((?:\.[A-Za-z0-9_ ]+?)+)\s*(?:\|\s*(pct|r0|r1|r2))?\s*\}\}")
# Subtrees of tool outputs that hold raw logged data rather than computed results.
RAW_DATA_KEYS = {"entries", "raw_text"}


def _walk(obj: Any, skip: frozenset[str] | set[str] = frozenset()) -> Iterable[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield from _walk(k, skip)
            yield from _walk(v, skip)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v, skip)
    else:
        yield obj


def _fmt(v: float, flt: str | None) -> str:
    if flt == "pct":
        return f"{v * 100:.0f}"
    if flt in ("r0", "r1", "r2"):
        return f"{v:.{flt[1]}f}"
    return f"{v:g}" if isinstance(v, float) else str(v)


def resolve_slots(text: str, refs: dict[str, Any]) -> tuple[str, list[dict], list[str]]:
    """Fill ``{{T<n>.path}}`` placeholders from tool results. Returns (text, bindings, bad_slots)."""
    bindings: list[dict] = []
    bad: list[str] = []

    def sub(m: re.Match) -> str:
        ref, path, flt = m.group(1), m.group(2).strip("."), m.group(3)
        node: Any = refs.get(ref)
        ok = node is not None
        for part in path.split(".") if ok else []:
            part = part.strip()
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                ok = False
                break
        if not ok or isinstance(node, (dict, list, bool)) or node is None:
            bad.append(m.group(0))
            return m.group(0)
        if flt and not isinstance(node, (int, float)):
            bad.append(m.group(0))
            return m.group(0)
        val = _fmt(node, flt) if isinstance(node, (int, float)) else str(node)
        bindings.append({"slot": m.group(0), "ref": ref, "path": path, "filter": flt, "value": val})
        return val

    return SLOT_RE.sub(sub, text), bindings, bad


def source_numbers(sources: Iterable[Any]) -> set[float]:
    nums: set[float] = set()
    for src in sources:
        for leaf in _walk(src, RAW_DATA_KEYS):
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, (int, float)):
                nums.add(float(leaf))
            elif isinstance(leaf, str):
                stripped = DATE_RE.sub(" ", leaf)
                for m in NUM_RE.findall(stripped):
                    nums.add(float(m))
    return nums


def source_dates(sources: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for src in sources:
        out.update(DATE_RE.findall(json.dumps(src, default=str)))
    return out


def answer_numbers(text: str) -> list[str]:
    text = NAME_TOKENS.sub(" ", DATE_RE.sub(" ", text))
    return NUM_RE.findall(LOOSE_DATE_RE.sub(" ", text))


def _matches(token: str, pool: set[float]) -> bool:
    x = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for v in pool:
        for cand in (v, v * 100, abs(v), abs(v) * 100):
            if abs(abs(x) - cand) <= tol:
                return True
    return False


def check(answer: str, tool_outputs: list[Any], extra_sources: list[Any] | None = None,
          refs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return {'grounded', 'ungrounded_numbers', 'ungrounded_dates', 'answer', 'bindings', ...}.

    answer: the model's text, possibly containing slots. tool_outputs: results of tool calls in this run.
    extra_sources: user message, memory facts. refs: slot name -> tool result ("T1" -> {...}).
    'answer' in the result is the text with slots filled in; check the template, show the filled text."""
    filled, bindings, bad_slots = resolve_slots(answer, refs or {})
    answer = SLOT_RE.sub(" ", answer)  # slot values are bound by construction; check the rest
    sources = list(tool_outputs) + list(extra_sources or [])
    pool = source_numbers(sources)
    bad_nums = bad_slots + [t for t in answer_numbers(answer) if not _matches(t, pool)]
    ok_dates = source_dates(sources)
    bad_dates = [d for d in DATE_RE.findall(answer) if d not in ok_dates]
    bad_dates += [m.group(0) for m in LOOSE_DATE_RE.finditer(NAME_TOKENS.sub(" ", answer))]
    if not filled.strip():
        bad_nums.append("<empty answer>")
    return {
        "answer": filled,
        "bindings": bindings,
        "grounded": not bad_nums and not bad_dates,
        "ungrounded_numbers": bad_nums,
        "ungrounded_dates": bad_dates,
        "numbers_checked": len(answer_numbers(answer)),
    }

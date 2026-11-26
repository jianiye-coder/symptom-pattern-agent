"""Groundedness checks: every number and date in an answer must be traceable to a tool output
(or to text the user/memory supplied). A number counts as grounded if it is a rounding of some
value in the allowed sources (also accepting proportion <-> percent)."""
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


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def source_numbers(sources: Iterable[Any]) -> set[float]:
    nums: set[float] = set()
    for src in sources:
        for leaf in _walk(src):
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


def check(answer: str, tool_outputs: list[Any], extra_sources: list[Any] | None = None) -> dict[str, Any]:
    """Return {'grounded': bool, 'ungrounded_numbers': [...], 'ungrounded_dates': [...]}.

    tool_outputs: results of tool calls in this run. extra_sources: user message, memory facts."""
    sources = list(tool_outputs) + list(extra_sources or [])
    pool = source_numbers(sources)
    bad_nums = [t for t in answer_numbers(answer) if not _matches(t, pool)]
    ok_dates = source_dates(sources)
    bad_dates = [d for d in DATE_RE.findall(answer) if d not in ok_dates]
    bad_dates += [m.group(0) for m in LOOSE_DATE_RE.finditer(NAME_TOKENS.sub(" ", answer))]
    if not answer.strip():
        bad_nums.append("<empty answer>")
    return {
        "grounded": not bad_nums and not bad_dates,
        "ungrounded_numbers": bad_nums,
        "ungrounded_dates": bad_dates,
        "numbers_checked": len(answer_numbers(answer)),
    }

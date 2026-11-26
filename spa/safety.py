"""Safety layer. Runs in the harness, outside the model, so it holds for any LLM backend.

1. Input screen: red-flag symptoms -> a seek-care notice is always prepended to the answer.
2. Input screen: medication start/stop/dose questions -> the agent is told it may describe
   the data but must not advise; the output guard enforces it.
3. Output guard: prescriptive medication language or diagnostic claims in the model's
   answer are blocked and replaced with a safe fallback.
"""
from __future__ import annotations

import re

RED_FLAGS: dict[str, str] = {
    r"chest (pain|tightness|pressure|hurts)": "chest pain",
    r"(faint(ed|ing)?|passed out|blacked out|loss of consciousness)": "fainting",
    r"(can'?t|cannot|couldn'?t|hard to|struggl\w* to) breathe|can'?t catch my breath|short(ness)? of breath at rest|gasping": "trouble breathing",
    r"(worst headache|thunderclap)": "sudden severe headache",
    r"(slurred speech|face (is )?droop|one side of (my )?(body|face)|sudden(ly)? (numb|weak))": "stroke signs",
    r"(suicid\w*|kill(ing)? myself|end(ing)? (my life|it all)|self[- ]harm|don'?t want to (live|be alive)|better off dead|want to die)": "thoughts of self-harm",
    r"(seizure|convuls)": "seizure",
    r"(coughing|vomiting|throwing up) blood|blood in (my )?(stool|vomit)": "bleeding",
    r"(confus\w* and fever|stiff neck and fever|fever (of |over )?(10[3-9]|40|41))": "high fever / confusion",
}

CARE_NOTICE = (
    "**Please seek medical care now.** You mentioned {flags}. That can be a sign of something "
    "that needs urgent attention. Call your local emergency number (911 in the US) or go to an "
    "emergency department. If you are having thoughts of harming yourself, in the US you can call "
    "or text 988. I'm a pattern-tracking tool and can't assess this."
)

MED_DECISION = re.compile(
    r"\b(should i|can i|is it ok(ay)? to|do i need to|would it help to)\b.{0,40}"
    r"\b(stop|start|quit|increase|decrease|double|halve|taper|skip|come off|go off|switch|take more|take less)\b",
    re.I,
)

MED_POLICY = (
    "I can't advise starting, stopping or changing the dose of any medication. That decision "
    "belongs with your prescriber, who knows your full history. What I can do is show you what "
    "your own log looked like before and after a change, so you have concrete data to bring to "
    "that conversation."
)

# Language the answer must never contain.
NEG = r"(?<!don't )(?<!do not )(?<!never )(?<!not )"
UNSAFE_OUTPUT = [
    r"\byou (should|could|might want to|may want to|need to|ought to) (stop|start|increase|decrease|quit|discontinue|"
    r"taper|double|halve|cut back|reduce|come off|go off)\b",
    NEG + r"\b(stop|quit|discontinue) taking\b",
    NEG + r"\b(start|begin) taking\b",
    r"\b(increase|decrease|raise|lower|double|halve|cut back|reduce) (on )?(your|the) (dose|dosage)\b",
    r"\b(i'?d|i would|i) (recommend|suggest|advise) (that you )?(start|stop|tak|increas|decreas|switch|cut|reduc|quit)",
    r"\byou (have|likely have|probably have|are suffering from) (me/cfs|long covid|pots|fibromyalgia|migraine disorder|"
    r"lupus|ms|multiple sclerosis|a condition|an? \w+ disorder)\b",
    r"\b(my|this|your) diagnosis is\b",
    r"\bthis (confirms|proves) (that )?you have\b",
]
_UNSAFE_RE = [re.compile(p, re.I) for p in UNSAFE_OUTPUT]
_RED_RE = [(re.compile(p, re.I), label) for p, label in RED_FLAGS.items()]


def red_flags(text: str) -> list[str]:
    found = []
    for rx, label in _RED_RE:
        if rx.search(text) and label not in found:
            found.append(label)
    return found


def is_med_decision(text: str) -> bool:
    return bool(MED_DECISION.search(text))


def unsafe_spans(text: str) -> list[str]:
    return [m.group(0) for rx in _UNSAFE_RE for m in [rx.search(text)] if m]


def care_notice(flags: list[str]) -> str:
    return CARE_NOTICE.format(flags=", ".join(flags))

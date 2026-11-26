"""Free-text log -> validated structured Entry.

Two backends behind one function:
  * rule-based parser (used in mock/offline mode; deterministic, no network)
  * LLM JSON extraction, validated by Pydantic, with one repair retry on validation error
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Optional

from pydantic import ValidationError

from .llm import LLMClient, MockLLM
from .models import SYMPTOM_SYNONYMS, Entry, ExtractedEntry

KNOWN_FOODS = [
    "red wine", "white wine", "wine", "beer", "alcohol", "coffee", "espresso", "tea", "chocolate", "cheese",
    "pasta", "pizza", "bread", "rice", "salad", "chicken", "fish", "salmon", "eggs", "oatmeal", "soup",
    "sugar", "ice cream", "dairy", "gluten", "curry", "sushi", "burger", "fries", "yogurt", "fruit", "steak",
]
KNOWN_MEDS = [
    "magnesium", "vitamin d", "vitamin b12", "b12", "multivitamin", "coq10", "melatonin", "ibuprofen",
    "acetaminophen", "paracetamol", "ldn", "low dose naltrexone", "antihistamine", "electrolytes", "iron",
    "omega-3", "fish oil", "sumatriptan", "propranolol", "nac",
]
MED_ALIASES = {"b12": "vitamin b12", "low dose naltrexone": "ldn", "paracetamol": "acetaminophen", "fish oil": "omega-3"}
WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}
EXERTION_WORDS = [
    (r"\b(marathon|hike|hiked|hiking|long run|gym|moved house|moving boxes|deep clean\w*|cleaned the whole)\b", 8),
    (r"\b(ran|run|jog\w*|cycl\w*|bike ride|swim\w*|lots of walking|walked a lot|busy day|long walk|errands all day)\b", 7),
    (r"\b(short walk|light (yoga|stretch\w*)|errand)\b", 4),
    (r"\b(stayed in bed|bed all day|couch day|resting|rested|didn'?t leave the house)\b", 1),
]
INTENSITY = {"mild": 3, "slight": 2, "a bit of": 3, "some": 4, "moderate": 5, "bad": 7, "severe": 8,
             "awful": 8, "terrible": 9, "crushing": 9, "horrible": 9}

_SYM_NAMES = sorted(SYMPTOM_SYNONYMS, key=len, reverse=True)
_SYM_ALT = "|".join(re.escape(s) for s in _SYM_NAMES)


def _num(tok: str) -> float:
    tok = tok.lower()
    return float(WORD_NUM[tok]) if tok in WORD_NUM else float(tok)


def rule_extract(text: str) -> ExtractedEntry:
    t = " " + text.lower().replace("’", "'") + " "
    symptoms: dict[str, int] = {}
    # "fatigue 7/10", "fatigue: 7", "fatigue was 7", "brain fog at 5"
    for m in re.finditer(rf"\b({_SYM_ALT})\b\s*(?:was|is|at|of|:|=|-)?\s*(\d{{1,2}})(?:\s*/\s*10)?\b", t):
        v = int(m.group(2))
        if v <= 10:
            symptoms.setdefault(SYMPTOM_SYNONYMS[m.group(1)], v)
    # "7/10 fatigue"
    for m in re.finditer(rf"\b(\d{{1,2}})\s*/\s*10\s+({_SYM_ALT})\b", t):
        symptoms.setdefault(SYMPTOM_SYNONYMS[m.group(2)], int(m.group(1)))
    # "severe headache", "mild fatigue"
    for m in re.finditer(rf"\b({'|'.join(INTENSITY)})\s+({_SYM_ALT})\b", t):
        symptoms.setdefault(SYMPTOM_SYNONYMS[m.group(2)], INTENSITY[m.group(1)])
    # "no headache"
    for m in re.finditer(rf"\bno\s+({_SYM_ALT})\b", t):
        symptoms.setdefault(SYMPTOM_SYNONYMS[m.group(1)], 0)

    sleep: Optional[float] = None
    m = re.search(
        r"\b(?:slept|sleep(?:ing)?|got)\s*(?:was|for|about|around|only|just|:)?\s*(?:about|around|only|just)?\s*"
        r"(\d+(?:\.\d+)?|" + "|".join(WORD_NUM) + r")\s*(?:h\b|hrs?\b|hours?\b)",
        t,
    ) or re.search(r"\b(\d+(?:\.\d+)?)\s*(?:h|hrs?|hours?)\s+(?:of\s+)?sleep\b", t)
    if m:
        sleep = _num(m.group(1))

    exertion: Optional[int] = None
    m = re.search(r"\b(?:exertion|activity|activity level)\s*(?:was|:|=|-)?\s*(\d{1,2})(?:\s*/\s*10)?\b", t)
    if m and int(m.group(1)) <= 10:
        exertion = int(m.group(1))
    else:
        m = re.search(r"\b(\d{1,2}(?:[.,]\d)?)\s*k\s*steps\b|\b(\d{4,5})\s*steps\b", t)
        if m:
            steps = float(m.group(1).replace(",", ".")) * 1000 if m.group(1) else float(m.group(2))
            exertion = 9 if steps >= 15000 else 7 if steps >= 10000 else 5 if steps >= 6000 else 3 if steps >= 3000 else 1
        else:
            for rx, lvl in EXERTION_WORDS:
                if re.search(rx, t):
                    exertion = lvl
                    break

    foods: list[str] = []
    for f in KNOWN_FOODS:
        if re.search(rf"\b{re.escape(f)}\b", t) and not any(f in g for g in foods):
            foods.append(f)
    meds: list[str] = []
    for med in KNOWN_MEDS:
        if re.search(rf"\b{re.escape(med)}\b", t):
            canon = MED_ALIASES.get(med, med)
            if canon not in meds and not any(canon in x for x in meds):
                meds.append(canon)
    return ExtractedEntry(symptoms=symptoms, sleep_hours=sleep, exertion=exertion, foods=foods, meds=meds,
                          notes=text.strip()[:500])


EXTRACTION_PROMPT = """You convert a person's free-text daily health log into JSON.
Return ONLY a JSON object with keys:
  symptoms: object mapping symptom name (lowercase, e.g. fatigue, brain_fog, headache, pain, nausea, dizziness) to severity 0-10.
            Map words: mild=3, moderate=5, bad=7, severe=8, terrible=9; "no X" = 0. Only include symptoms mentioned.
  sleep_hours: number or null
  exertion: integer 0-10 or null (bed all day=1, short walk=4, busy day/run/lots of walking=7, hike/gym/moving=8;
            steps: <3k=1, 3-6k=3, 6-10k=5, 10-15k=7, >=15k=9)
  foods: list of foods/drinks mentioned (lowercase)
  meds: list of medications/supplements taken (lowercase)
  notes: the original text, trimmed
Do not infer anything that is not stated. Do not add diagnoses."""


def llm_extract(text: str, llm: LLMClient) -> tuple[ExtractedEntry, dict]:
    messages = [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": text}]
    meta = {"attempts": 0, "errors": []}
    for _ in range(2):
        meta["attempts"] += 1
        resp = llm.complete(messages, tools=None, json_mode=True)
        raw = resp.content or ""
        try:
            raw_json = re.search(r"\{.*\}", raw, re.S)
            return ExtractedEntry.model_validate(json.loads(raw_json.group(0) if raw_json else raw)), meta
        except (ValidationError, json.JSONDecodeError, AttributeError) as e:
            meta["errors"].append(str(e)[:300])
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That did not validate: {str(e)[:300]}. Return corrected JSON only."},
            ]
    # fall back to the deterministic parser rather than failing the user's log
    meta["fallback"] = "rule_based"
    return rule_extract(text), meta


def extract_entry(text: str, date: dt.date, llm: LLMClient) -> tuple[Entry, dict]:
    if isinstance(llm, MockLLM):
        ex, meta = rule_extract(text), {"backend": "rule_based"}
    else:
        ex, meta = llm_extract(text, llm)
        meta["backend"] = "llm"
    entry = Entry(date=date, raw_text=text, **ex.model_dump())
    return entry, meta

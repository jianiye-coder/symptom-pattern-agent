"""Synthetic multi-week user histories with *planted* patterns (and null controls).

No real health data is used anywhere in this project. Every persona is generated from a
seed, so fixtures are reproducible and the ground truth is known exactly.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Any

from .models import Entry

PERSONAS = {
    "pem": "Post-exertional crash: fatigue spikes 1-2 days after high-exertion days.",
    "sleep_headache": "Headache flares on the same day as a short (<6h) night of sleep.",
    "wine_headache": "Headache flares the day after red wine.",
    "null": "No planted pattern; flares are random.",
    "magnesium_effect": "Started magnesium mid-way; fatigue drops by ~2 points afterwards.",
    "vitd_null": "Started vitamin D mid-way; no change in symptoms.",
    "sparse": "Only 9 logged days: not enough data to say anything.",
}

# Ground truth used by the evals.
TRUTH: dict[str, dict[str, Any]] = {
    # acceptable lag windows [lo, hi] for the planted effect
    "pem": {"exposure": "high_exertion", "lags": [[1, 1], [2, 2], [1, 2]]},
    "sleep_headache": {"exposure": "short_sleep", "lags": [[0, 0]]},
    "wine_headache": {"exposure": "food:red wine", "lags": [[1, 1]]},
    "null": {"exposure": None},
    "magnesium_effect": {"intervention": "magnesium", "verdict": "lower_after"},
    "vitd_null": {"intervention": "vitamin d", "verdict": "no_clear_change"},
    "sparse": {"insufficient": True},
}

FOODS = ["coffee", "oatmeal", "pasta", "salad", "chicken", "rice", "chocolate", "cheese", "eggs", "soup"]


def _clip(x: float) -> int:
    return max(0, min(10, int(round(x))))


def generate(persona: str, seed: int = 0, days: int = 70, start: dt.date = dt.date(2026, 1, 5)) -> list[Entry]:
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona}")
    rng = random.Random(f"{persona}:{seed}")
    if persona == "sparse":
        days = 9
    exertion = [rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6]) for _ in range(days)]
    for i in range(days):
        if rng.random() < 0.16:
            exertion[i] = rng.randint(7, 9)
    sleep = [round(min(10.0, max(3.5, rng.gauss(7.2, 0.8))) * 2) / 2 for _ in range(days)]
    if persona == "sleep_headache":
        for i in range(days):
            if rng.random() < 0.18:
                sleep[i] = rng.choice([4.0, 4.5, 5.0, 5.5])
    wine = [rng.random() < 0.15 for _ in range(days)]
    switch = days // 2
    entries: list[Entry] = []
    for i in range(days):
        fatigue = rng.gauss(3.6, 1.0)
        fog = rng.gauss(3.0, 1.0)
        headache = rng.gauss(2.2, 1.0)
        pain = rng.gauss(2.8, 1.0)
        # background random bad days, so even null personas have "flares" to explain
        if rng.random() < 0.09:
            if rng.random() < 0.5:
                fatigue += rng.uniform(3.5, 5)
            else:
                headache += rng.uniform(4.5, 6)
        if persona == "pem":
            for lag in (1, 2):
                j = i - lag
                if j >= 0 and exertion[j] >= 7 and rng.random() < 0.8:
                    fatigue += rng.uniform(3.5, 4.5)
                    fog += rng.uniform(1, 2)
                    break
        if persona == "sleep_headache" and sleep[i] < 6 and rng.random() < 0.8:
            headache += rng.uniform(5, 6.5)
        if persona == "wine_headache" and i >= 1 and wine[i - 1] and rng.random() < 0.85:
            headache += rng.uniform(5, 6.5)
        meds = ["multivitamin"] if rng.random() < 0.95 else []
        if persona == "magnesium_effect" and i >= switch:
            meds.append("magnesium")
            fatigue -= 2.0
            fog -= 1.0
        if persona == "vitd_null" and i >= switch:
            meds.append("vitamin d")
        foods = rng.sample(FOODS, k=rng.randint(2, 4))
        if wine[i] and persona != "sparse":
            foods.append("red wine")
        symptoms = {
            "fatigue": _clip(fatigue),
            "brain_fog": _clip(fog),
            "headache": _clip(headache),
            "pain": _clip(pain),
        }
        entries.append(
            Entry(
                date=start + dt.timedelta(days=i),
                symptoms=symptoms,
                sleep_hours=sleep[i],
                exertion=exertion[i],
                foods=foods,
                meds=meds,
                notes=_note(rng, exertion[i], symptoms),
            )
        )
    # realistic gaps: drop ~8% of days (never the very first day)
    if persona != "sparse":
        entries = [e for k, e in enumerate(entries) if k == 0 or rng.random() > 0.08]
    return entries


def _note(rng: random.Random, exertion: int, s: dict[str, int]) -> str:
    if max(s.values()) >= 7:
        return rng.choice(["rough day, mostly in bed", "crashed hard", "bad day", "could barely function"])
    if exertion >= 7:
        return rng.choice(["busy day, lots of walking", "did a long errand run", "cleaned the whole flat"])
    return rng.choice(["", "ok day", "quiet day at home", "felt about average"])

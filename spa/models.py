"""Typed data models. Everything that crosses a boundary (LLM output, API, DB) is validated here."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Canonical symptom names and the free-text synonyms that map onto them.
SYMPTOM_SYNONYMS: dict[str, str] = {
    "fatigue": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "energy crash": "fatigue",
    "brain fog": "brain_fog",
    "brain_fog": "brain_fog",
    "fog": "brain_fog",
    "foggy": "brain_fog",
    "headache": "headache",
    "migraine": "headache",
    "head pain": "headache",
    "pain": "pain",
    "joint pain": "pain",
    "body aches": "pain",
    "aches": "pain",
    "muscle pain": "pain",
    "nausea": "nausea",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
}


def normalize_name(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    if n in SYMPTOM_SYNONYMS:
        return SYMPTOM_SYNONYMS[n]
    return n.replace(" ", "_")


class Entry(BaseModel):
    """One day of the episodic log."""

    date: dt.date
    symptoms: dict[str, int] = Field(default_factory=dict, description="symptom -> severity 0-10")
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=16)
    exertion: Optional[int] = Field(default=None, ge=0, le=10, description="physical/cognitive exertion 0-10")
    foods: list[str] = Field(default_factory=list)
    meds: list[str] = Field(default_factory=list, description="meds and supplements taken that day")
    notes: str = ""
    raw_text: Optional[str] = None

    @field_validator("symptoms")
    @classmethod
    def _check_symptoms(cls, v: dict[str, int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, sev in v.items():
            sev_i = int(sev)
            if not 0 <= sev_i <= 10:
                raise ValueError(f"severity for {k!r} must be 0-10, got {sev}")
            out[normalize_name(k)] = sev_i
        return out

    @field_validator("foods", "meds")
    @classmethod
    def _norm_items(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for item in v:
            n = re.sub(r"\s+", " ", item.strip().lower())
            if n and n not in seen:
                seen.append(n)
        return seen


class ExtractedEntry(BaseModel):
    """Schema the LLM must fill when parsing a free-text log. Date is supplied by the caller.
    Same validators as Entry, so bad LLM output fails here (and triggers a repair) not later."""

    symptoms: dict[str, int] = Field(default_factory=dict)
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=16)
    exertion: Optional[int] = Field(default=None, ge=0, le=10)
    foods: list[str] = Field(default_factory=list)
    meds: list[str] = Field(default_factory=list)
    notes: str = ""

    _check_symptoms = field_validator("symptoms")(Entry._check_symptoms.__func__)
    _norm_items = field_validator("foods", "meds")(Entry._norm_items.__func__)


FactCategory = Literal[
    "trigger_confirmed",  # user explicitly confirmed
    "trigger_suspected",  # agent hypothesis backed by a tool run
    "tried_stopped",
    "tried_ongoing",
    "preference",  # e.g. flare definition, exposures to ignore
    "context",  # diagnosis the user told us, goals, etc.
]


class Fact(BaseModel):
    id: int
    user_id: str
    category: FactCategory
    subject: str
    statement: str
    value: dict[str, Any] = Field(default_factory=dict)
    source: Literal["user", "agent"]
    provenance: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "superseded", "retracted"] = "active"
    supersedes_id: Optional[int] = None
    created_at: str
    updated_at: str


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False


class LLMResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class TraceStep(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: Any
    ok: bool
    latency_ms: float


class AgentResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    grounded: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    run_id: Optional[int] = None

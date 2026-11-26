"""FastAPI backend + static web UI.  Run:  uvicorn spa.api:app --reload

The user is identified by an unauthenticated X-User-Id header (default "demo"). That is fine for a local
demo and for the public *synthetic-data* demo (SPA_PUBLIC_DEMO=1), where each browser gets a random id,
the mock LLM is forced, inactive users are purged and requests are rate-limited. It is not authentication:
never put real health data behind this server."""
from __future__ import annotations

import datetime as dt
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analysis, report, safety, synth
from .agent import Agent, make_llm
from .extraction import extract_entry
from .memory import MemoryStore
from .models import Entry, FactCategory

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

PUBLIC_DEMO = os.environ.get("SPA_PUBLIC_DEMO", "").lower() in ("1", "true", "yes")
PURGE_HOURS = float(os.environ.get("SPA_PURGE_HOURS", "24"))
RATE_PER_MIN = int(os.environ.get("SPA_RATE_PER_MIN", "120"))

app = FastAPI(title="symptom-pattern-agent", version="0.1.0")
store = MemoryStore(os.environ.get("SPA_DB_PATH", str(ROOT / "data" / "spa.db")))
if PUBLIC_DEMO:
    from .llm import MockLLM

    llm = MockLLM()  # never spend an API key on anonymous traffic
    store.purge_inactive(PURGE_HOURS)
else:
    llm = make_llm()
agent = Agent(store, llm)

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()
_last_purge = [time.time()]


@app.middleware("http")
async def public_demo_guard(request: Request, call_next):
    if PUBLIC_DEMO and request.url.path.startswith("/api/"):
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
        now = time.time()
        with _hits_lock:
            q = _hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_PER_MIN:
                return JSONResponse({"detail": "Too many requests; slow down a little."}, status_code=429)
            q.append(now)
            if now - _last_purge[0] > 3600:
                _last_purge[0] = now
                store.purge_inactive(PURGE_HOURS)
    return await call_next(request)


_UID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _uid(x_user_id: Optional[str]) -> str:
    uid = _UID_RE.sub("", (x_user_id or "demo").strip())[:64]
    return uid or "demo"


class LogIn(BaseModel):
    date: Optional[dt.date] = None
    text: Optional[str] = Field(None, max_length=4000)
    entry: Optional[Entry] = None


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[dict[str, Any]] = Field(default_factory=list)


class FlagIn(BaseModel):
    date: dt.date
    note: str = Field("", max_length=300)


class FactIn(BaseModel):
    category: FactCategory
    subject: str = Field(..., min_length=1, max_length=80)
    statement: str = Field(..., min_length=1, max_length=400)
    value: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm": llm.name, "public_demo": PUBLIC_DEMO}


@app.get("/api/personas")
def personas() -> dict:
    return {"personas": synth.PERSONAS}


@app.post("/api/demo/{persona}")
def load_demo(persona: str, seed: int = 1, x_user_id: Optional[str] = Header(None)) -> dict:
    if persona not in synth.PERSONAS:
        raise HTTPException(404, f"unknown persona; choose from {list(synth.PERSONAS)}")
    uid = _uid(x_user_id)
    store.clear_user(uid)
    n = store.add_entries(uid, synth.generate(persona, seed))
    return {"loaded": n, "persona": persona, "description": synth.PERSONAS[persona]}


@app.post("/api/entries")
def add_entry(body: LogIn, x_user_id: Optional[str] = Header(None)) -> dict:
    uid = _uid(x_user_id)
    if body.entry is not None:
        entry, meta = body.entry, {"backend": "structured"}
    elif body.text:
        entries = store.get_entries(uid)
        default_date = (entries[-1].date + dt.timedelta(days=1)) if entries else dt.date.today()
        entry, meta = extract_entry(body.text, body.date or default_date, llm)
    else:
        raise HTTPException(422, "provide text or entry")
    store.upsert_entry(uid, entry)
    flags = safety.red_flags(body.text or entry.notes)
    return {
        "entry": entry.model_dump(mode="json"),
        "extraction": meta,
        "red_flags": flags,
        "care_notice": safety.care_notice(flags) if flags else None,
    }


@app.get("/api/entries")
def list_entries(x_user_id: Optional[str] = Header(None)) -> dict:
    return {"entries": [e.model_dump(mode="json") for e in store.get_entries(_uid(x_user_id))]}


@app.post("/api/chat")
def chat(body: ChatIn, x_user_id: Optional[str] = Header(None)) -> dict:
    res = agent.run(_uid(x_user_id), body.message, body.history)
    return res.model_dump(mode="json")


@app.get("/api/memory")
def get_memory(include_inactive: bool = False, x_user_id: Optional[str] = Header(None)) -> dict:
    uid = _uid(x_user_id)
    return {
        "facts": [f.model_dump() for f in store.facts(uid, include_inactive=include_inactive)],
        "events": store.events(uid)[-50:],
    }


@app.post("/api/memory")
def add_fact(body: FactIn, x_user_id: Optional[str] = Header(None)) -> dict:
    """User edits memory directly from the UI (source=user, provenance=ui)."""
    from .agent import valid_threshold

    if body.subject == "flare_definition" and valid_threshold(body.value.get("threshold")) is None:
        raise HTTPException(422, "flare_definition needs value.threshold as an integer 1-10")
    f = store.remember(_uid(x_user_id), body.category, body.subject, body.statement, "user", {"via": "ui"}, body.value)
    return {"fact": f.model_dump()}


@app.delete("/api/memory/{fact_id}")
def retract_fact(fact_id: int, x_user_id: Optional[str] = Header(None)) -> dict:
    f = store.retract(_uid(x_user_id), fact_id, "user removed in UI")
    if f is None:
        raise HTTPException(404, "no such active fact")
    return {"fact": f.model_dump()}


@app.get("/api/chart")
def chart(exposure: Optional[str] = None, x_user_id: Optional[str] = Header(None)) -> dict:
    uid = _uid(x_user_id)
    entries = store.get_entries(uid)
    fd, _ = agent._flare_def(uid)
    data = analysis.chart_series(entries, fd.get("symptom"), fd)
    data["flare_definition"] = analysis.flare_label(fd)
    if exposure:
        ser = analysis.exposure_series(entries).get(exposure, {})
        for p in data["points"]:
            p["exposed"] = ser.get(dt.date.fromisoformat(p["date"]), False)
        data["exposure"] = exposure
        data["exposure_label"] = analysis.exposure_label(exposure)
    return data


@app.get("/api/flags")
def list_flags(x_user_id: Optional[str] = Header(None)) -> dict:
    return {"flags": store.flags(_uid(x_user_id))}


@app.post("/api/flags")
def add_flag(body: FlagIn, x_user_id: Optional[str] = Header(None)) -> dict:
    return {"flag": store.set_flag(_uid(x_user_id), body.date.isoformat(), body.note)}


@app.delete("/api/flags/{date}")
def remove_flag(date: dt.date, x_user_id: Optional[str] = Header(None)) -> dict:
    if not store.clear_flag(_uid(x_user_id), date.isoformat()):
        raise HTTPException(404, "day is not flagged")
    return {"ok": True}


@app.get("/api/progress")
def progress(x_user_id: Optional[str] = Header(None)) -> dict:
    return analysis.logging_progress(store.get_entries(_uid(x_user_id)))


def _report(uid: str) -> dict:
    fd, _ = agent._flare_def(uid)
    ignore, _ = agent._ignored(uid)
    return report.build(store.get_entries(uid), store.facts(uid), store.flags(uid), fd, ignore)


@app.get("/api/report")
def report_json(x_user_id: Optional[str] = Header(None)) -> dict:
    return _report(_uid(x_user_id))


@app.get("/report", response_class=HTMLResponse)
def report_page(user: Optional[str] = None) -> str:
    """Printable doctor-visit report. A plain link can't send headers, so the demo user comes from ?user=."""
    return report.to_html(_report(_uid(user)))


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")

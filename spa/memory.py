"""Per-user persistent memory on SQLite.

Two stores:
  * episodic  - `entries`: one row per logged day (the raw material for analysis)
  * semantic  - `facts`: durable "facts about me" with provenance, timestamps and a
                supersede/retract history (never hard-deleted), plus an append-only
                `memory_events` audit log.
Agent runs are stored in `runs` so a fact's provenance can point at the exact
tool output that justified it.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import Entry, Fact

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    statement TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    supersedes_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_user ON facts(user_id, status);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact_id INTEGER,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS flags (
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    trace TEXT NOT NULL DEFAULT '[]',
    at TEXT NOT NULL
);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ episodic
    def upsert_entry(self, user_id: str, entry: Entry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries(user_id, date, data, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, date) DO UPDATE SET data=excluded.data",
                (user_id, entry.date.isoformat(), entry.model_dump_json(), _now()),
            )
            self._conn.commit()

    def add_entries(self, user_id: str, entries: Iterable[Entry]) -> int:
        n = 0
        for e in entries:
            self.upsert_entry(user_id, e)
            n += 1
        return n

    def get_entries(
        self, user_id: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> list[Entry]:
        q = "SELECT data FROM entries WHERE user_id=?"
        args: list[Any] = [user_id]
        if start:
            q += " AND date>=?"
            args.append(start)
        if end:
            q += " AND date<=?"
            args.append(end)
        q += " ORDER BY date"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [Entry.model_validate_json(r["data"]) for r in rows]

    def logged_dates(self, user_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT date FROM entries WHERE user_id=?", (user_id,)).fetchall()
        return {r["date"] for r in rows}

    def clear_user(self, user_id: str) -> None:
        with self._lock:
            for t in ("entries", "facts", "memory_events", "runs", "flags"):
                self._conn.execute(f"DELETE FROM {t} WHERE user_id=?", (user_id,))
            self._conn.commit()

    # ------------------------------------------------------------------ flagged days
    # Days the user marked as important to raise at their next appointment (with an optional note).
    def set_flag(self, user_id: str, date: str, note: str = "") -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT INTO flags(user_id, date, note, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, date) DO UPDATE SET note=excluded.note",
                (user_id, date, note.strip()[:300], _now()),
            )
            self._conn.commit()
        return {"date": date, "note": note.strip()[:300]}

    def clear_flag(self, user_id: str, date: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM flags WHERE user_id=? AND date=?", (user_id, date))
            self._conn.commit()
        return cur.rowcount > 0

    def flags(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT date, note FROM flags WHERE user_id=? ORDER BY date", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def purge_inactive(self, older_than_hours: float) -> int:
        """Delete every user whose last write (entry, run or fact) is older than the cutoff. Public demo only."""
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=older_than_hours)).isoformat(timespec="seconds")
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, MAX(t) AS last FROM ("
                " SELECT user_id, created_at AS t FROM entries UNION ALL"
                " SELECT user_id, at AS t FROM runs UNION ALL"
                " SELECT user_id, updated_at AS t FROM facts UNION ALL"
                " SELECT user_id, created_at AS t FROM flags) GROUP BY user_id HAVING last < ?",
                (cutoff,),
            ).fetchall()
            for r in rows:
                for t in ("entries", "facts", "memory_events", "runs", "flags"):
                    self._conn.execute(f"DELETE FROM {t} WHERE user_id=?", (r["user_id"],))
            self._conn.commit()
        return len(rows)

    # ------------------------------------------------------------------ semantic
    def _row_to_fact(self, r: sqlite3.Row) -> Fact:
        d = dict(r)
        d["value"] = json.loads(d["value"])
        d["provenance"] = json.loads(d["provenance"])
        return Fact.model_validate(d)

    def _event(self, user_id: str, fact_id: Optional[int], action: str, detail: dict) -> None:
        self._conn.execute(
            "INSERT INTO memory_events(user_id, fact_id, action, detail, at) VALUES (?,?,?,?,?)",
            (user_id, fact_id, action, json.dumps(detail), _now()),
        )

    def remember(
        self,
        user_id: str,
        category: str,
        subject: str,
        statement: str,
        source: str,
        provenance: Optional[dict] = None,
        value: Optional[dict] = None,
    ) -> Fact:
        """Insert a fact. An active fact with the same (category, subject) is superseded, not overwritten."""
        subject = subject.strip().lower()
        now = _now()
        with self._lock:
            old = self._conn.execute(
                "SELECT id FROM facts WHERE user_id=? AND category=? AND subject=? AND status='active'",
                (user_id, category, subject),
            ).fetchone()
            cur = self._conn.execute(
                "INSERT INTO facts(user_id, category, subject, statement, value, source, provenance,"
                " status, supersedes_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    category,
                    subject,
                    statement,
                    json.dumps(value or {}),
                    source,
                    json.dumps(provenance or {}),
                    "active",
                    old["id"] if old else None,
                    now,
                    now,
                ),
            )
            new_id = cur.lastrowid
            if old:
                self._conn.execute(
                    "UPDATE facts SET status='superseded', updated_at=? WHERE id=?", (now, old["id"])
                )
                self._event(user_id, old["id"], "superseded", {"by": new_id})
            self._event(user_id, new_id, "created", {"category": category, "subject": subject, "source": source})
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM facts WHERE id=?", (new_id,)).fetchone()
        return self._row_to_fact(row)

    def retract(self, user_id: str, fact_id: int, reason: str) -> Optional[Fact]:
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE id=? AND user_id=?", (fact_id, user_id)
            ).fetchone()
            if row is None or row["status"] != "active":
                return None
            self._conn.execute("UPDATE facts SET status='retracted', updated_at=? WHERE id=?", (now, fact_id))
            self._event(user_id, fact_id, "retracted", {"reason": reason})
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        return self._row_to_fact(row)

    def facts(self, user_id: str, category: Optional[str] = None, include_inactive: bool = False) -> list[Fact]:
        q = "SELECT * FROM facts WHERE user_id=?"
        args: list[Any] = [user_id]
        if not include_inactive:
            q += " AND status='active'"
        if category:
            q += " AND category=?"
            args.append(category)
        q += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def events(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_events WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]

    # ------------------------------------------------------------------ runs
    def start_run(self, user_id: str, question: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs(user_id, question, at) VALUES (?,?,?)", (user_id, question, _now())
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, answer: str, trace: list[dict]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET answer=?, trace=? WHERE id=?", (answer, json.dumps(trace, default=str), run_id)
            )
            self._conn.commit()

    def get_run(self, user_id: str, run_id: int) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM runs WHERE id=? AND user_id=?", (run_id, user_id)).fetchone()
        if r is None:
            return None
        return dict(r) | {"trace": json.loads(r["trace"])}

    # ------------------------------------------------------------------ helpers used by analysis
    def preferences(self, user_id: str) -> dict[str, Fact]:
        return {f.subject: f for f in self.facts(user_id, category="preference")}

"""LLM backends behind one interface.

* OpenAICompatClient: any OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter,
  Together, vLLM, Ollama...). Configured from env / .env; keys are never hardcoded.
* MockLLM: a deterministic stand-in that speaks the same message/tool-call protocol, so the
  harness, tests and evals run offline in CI. It routes intents with keyword rules and composes
  its answer by rendering the tool outputs. It is a test double, not a model.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx

from .models import LLMResponse, ToolCall, Usage


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLMClient(Protocol):
    name: str

    def complete(
        self, messages: list[dict], tools: Optional[list[dict]] = None, json_mode: bool = False
    ) -> LLMResponse: ...


class OpenAICompatClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not set (put it in .env; see .env.example)")
        self.name = f"openai-compat:{self.model}"
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, messages, tools=None, json_mode=False) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = self._http.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                if r.status_code >= 400:  # auth / bad request: retrying won't help
                    raise RuntimeError(f"LLM request failed: HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_err = e
                time.sleep(0.5 * 2**attempt)
        else:
            raise RuntimeError(f"LLM request failed after retries: {last_err}")
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc["function"].get("arguments")}
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        u = data.get("usage") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return LLMResponse(
            content=content,
            tool_calls=calls,
            usage=Usage(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0)),
        )


# ----------------------------------------------------------------------------- mock
_FACT_LINE = re.compile(r"^\s*- #(\d+) \[(\w+)\] subject=(.+?) \| (.*)$", re.M)


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


class MockLLM:
    """Deterministic policy that emits OpenAI-style tool calls. See module docstring."""

    name = "mock"

    def __init__(self):
        self._n = 0

    def _call(self, name: str, **args) -> ToolCall:
        self._n += 1
        return ToolCall(id=f"call_{self._n}", name=name, arguments=args)

    def complete(self, messages, tools=None, json_mode=False) -> LLMResponse:
        usage = Usage(prompt_tokens=_estimate_tokens(messages), completion_tokens=0, estimated=True)
        if json_mode:
            from .extraction import rule_extract  # local import to avoid a cycle

            text = messages[-1]["content"]
            out = rule_extract(text).model_dump_json()
            usage.completion_tokens = len(out) // 4
            return LLMResponse(content=out, usage=usage)

        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        q = question.lower()
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        done = [m for m in messages[last_user + 1 :] if m["role"] == "tool"]
        called = [
            tc for m in messages[last_user + 1 :] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])
        ]
        results: list[tuple[str, Any]] = []
        by_id = {tc["id"]: tc["function"]["name"] for tc in called}
        for m in done:
            results.append((by_id.get(m["tool_call_id"], "?"), json.loads(m["content"])))

        if not done:
            calls = self._plan(question, q, system)
            return LLMResponse(tool_calls=calls, usage=usage)

        # Second step: if a pattern was found and we haven't saved it, store it as a *suspected* trigger.
        if not any(n == "remember_fact" for n, _ in results):
            for cid, (n, r) in zip([tc["id"] for tc in called], results):
                if n == "find_flare_precursors" and r.get("status") == "pattern_found":
                    f = r["findings"][0]
                    confirmed = {c["subject"] for c in r.get("user_confirmed_triggers", [])}
                    if f["exposure"] not in confirmed:
                        return LLMResponse(
                            tool_calls=[
                                self._call(
                                    "remember_fact",
                                    category="trigger_suspected",
                                    subject=f["exposure"],
                                    statement=f"Data suggests {f['label']} is followed by flares ({f['lag_text']}).",
                                    evidence_call_id=cid,
                                )
                            ],
                            usage=usage,
                        )
        from .render import render_all

        text = render_all(results)
        usage.completion_tokens = len(text) // 4
        return LLMResponse(content=text, usage=usage)

    # -- intent routing -------------------------------------------------------
    def _plan(self, question: str, q: str, system: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        quote = question.strip()
        m = re.search(
            r"bad day (?:is|means|=) (?:when |if )?(?:my )?([a-z_ ]+?) (?:is |hits |reaches )?"
            r"(?:>=|≥|at least|of at least|above|over|is)? ?(\d+)",
            q,
        )
        if m:
            sym = m.group(1).strip()
            thr = int(m.group(2)) + (1 if re.search(r"\b(above|over)\b", q) else 0)
            calls.append(
                self._call(
                    "remember_fact",
                    category="preference",
                    subject="flare_definition",
                    statement=f"A flare/bad day means {sym} >= {thr}.",
                    value={"symptom": sym, "threshold": thr},
                    user_quote=quote,
                )
            )
        m = re.search(r"\b(?:ignore|don'?t count|exclude|stop tracking) ([a-z][a-z ]*?)(?:\b(?:in|from|when|,|\.|$))", q)
        if m and "bad day" not in q:
            item = m.group(1).strip()
            calls.append(
                self._call(
                    "remember_fact",
                    category="preference",
                    subject=f"ignore_exposure:{item}",
                    statement=f"Ignore {item} when looking for triggers.",
                    value={"exposure": item},
                    user_quote=quote,
                )
            )
        m = re.search(r"([a-z][a-z ]*?) (?:is|are) (?:definitely |really |)(?:a |one of my )?triggers?\b", q)
        if m and not q.rstrip().endswith("?"):
            subj = re.sub(r"^(i know|i've confirmed|confirmed|for me,?|actually,?)\s+", "", m.group(1).strip())
            calls.append(
                self._call(
                    "remember_fact",
                    category="trigger_confirmed",
                    subject=subj,
                    statement=f"User confirmed {subj} is a trigger.",
                    user_quote=quote,
                )
            )
        m = re.search(r"\bi (?:stopped|quit|gave up) (?:taking )?([a-z][a-z0-9 \-]*?)(?:\b(?:on|in|last|because|,|\.|$))", q)
        if m:
            item = m.group(1).strip()
            calls.append(
                self._call(
                    "remember_fact",
                    category="tried_stopped",
                    subject=item,
                    statement=f"User tried {item} and stopped.",
                    user_quote=quote,
                )
            )
        m = re.search(r"\bforget (?:that |about )?(?:my |the )?([a-z][a-z _:]*?)(?:\b(?:is|preference|,|\.|$))", q)
        if m:
            target = m.group(1).strip().replace(" ", "_")
            for fid, _cat, subj, _stmt in _FACT_LINE.findall(system):
                if target in subj.replace(" ", "_") or subj.replace(" ", "_") in target:
                    calls.append(self._call("retract_fact", fact_id=int(fid), reason=f"user asked: {quote[:120]}"))
                    break
        if calls:
            return calls

        meds_in_q = re.search(
            r"(?:did|does|has|is) ([a-z][a-z0-9 \-]*?) (?:help|helped|helping|work|working|make a difference|do anything)"
            r"|since (?:i )?(?:started|starting|began) ([a-z][a-z0-9 \-]*?)(?:\b(?:,|\?|\.|$| did| has| have| is| am))"
            r"|after (?:i )?(?:started|starting) ([a-z][a-z0-9 \-]*?)(?:\b(?:,|\?|\.|$| did| has| have| is| am))"
            r"|(?:stop|start|quit|increase|decrease) (?:taking |my )?([a-z][a-z0-9 \-]*?)(?:\?|$|\.| dose)",
            q,
        )
        sym = next((s for s in ("fatigue", "brain fog", "headache", "pain") if s in q), None)
        if meds_in_q:
            name = next(g for g in meds_in_q.groups() if g).strip()
            name = re.sub(r"^(taking|the|my)\s+", "", name)
            args = {"intervention": name}
            if sym:
                args["symptom"] = sym
            return [self._call("compare_before_after", **args)]
        m = re.search(
            r"(?:does|do|is|are|did) ([a-z][a-z \-]*?) (?:trigger|cause|affect|worsen|make .* worse|linked|related|"
            r"connected|lead to)",
            q,
        )
        if m:
            args = {"exposure": m.group(1).strip()}
            if sym:
                args["symptom"] = sym
            return [self._call("test_association", **args)]
        if re.search(r"before (my )?(bad|worst|flare)|trigger|pattern|precede|cause|crash|flare|bad days|worse", q):
            return [self._call("find_flare_precursors")]
        if re.search(r"remember|know about me|memory|what have i told", q):
            return [self._call("read_memory")]
        return [self._call("data_summary")]

"""The agent harness: prompt assembly, tool registry, tool-calling loop, memory-write policy,
groundedness check with one repair attempt, and safety guards.

The model decides *which* tool to call; the harness decides *what is allowed* and computes
every number. Answers are checked after generation, not just instructed in the prompt.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import analysis, grounding, safety
from .llm import LLMClient
from .memory import MemoryStore
from .models import AgentResult, Entry, TraceStep, Usage, normalize_name
from .render import render_all

MAX_STEPS = 6

SYSTEM_PROMPT = """You are a personal symptom-pattern assistant for someone living with a chronic condition
(for example long COVID, ME/CFS or migraine). You help them see patterns in their OWN log.

Rules:
- Never compute statistics yourself. Call a tool and report ONLY numbers and dates that appear in tool outputs.
- Cite the specific dates (YYYY-MM-DD) from tool outputs that support what you say.
- If a tool says insufficient_data or no_pattern, say so plainly. "Not enough data" is a good answer.
- Never diagnose. Never recommend starting, stopping or changing any medication or dose; refer to their clinician.
- Associations in self-tracking data are not proof of cause; include the tool's caveats briefly.
- Memory: when the user states a durable fact about themselves (a confirmed trigger, something they tried and
  stopped, how they define a bad day, exposures to ignore), save it with remember_fact and pass user_quote
  = their exact words. You may save a trigger_suspected fact only with evidence_call_id pointing at a tool call
  in this conversation that found it. If the user asks you to forget something, use retract_fact.
- Keep answers short: lead with the answer, then numbers, then dates, then caveats.

What you remember about this user (from memory; ids in #):
{memory}
"""


# ----------------------------------------------------------------------------- tool arg schemas
class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown/unparseable arguments are rejected, not ignored


class NoArgs(Args):
    pass


class PrecursorArgs(Args):
    symptom: Optional[str] = Field(None, description="Override flare symptom (default: user's saved definition)")
    threshold: Optional[int] = Field(None, ge=1, le=10)
    max_lag: int = Field(3, ge=0, le=3)


class AssociationArgs(Args):
    exposure: str = Field(..., description="e.g. 'exertion', 'short sleep', 'red wine', 'magnesium'")
    symptom: Optional[str] = None
    threshold: Optional[int] = Field(None, ge=1, le=10)


class BeforeAfterArgs(Args):
    intervention: str = Field(..., description="med/supplement name as logged")
    symptom: Optional[str] = Field(None, description="symptom to compare; default daily max severity")


class EntriesArgs(Args):
    start: Optional[str] = Field(None, description="YYYY-MM-DD")
    end: Optional[str] = Field(None, description="YYYY-MM-DD")


class ReadMemoryArgs(Args):
    category: Optional[str] = None


class RememberArgs(Args):
    category: Literal["trigger_confirmed", "trigger_suspected", "tried_stopped", "tried_ongoing", "preference", "context"]
    subject: str = Field(..., description="short key, e.g. 'high_exertion', 'magnesium', 'flare_definition', "
                                          "'ignore_exposure:coffee'")
    statement: str
    value: dict[str, Any] = Field(default_factory=dict, description="flare_definition: {symptom, threshold}")
    user_quote: Optional[str] = Field(None, description="user's exact words, required for user-sourced facts")
    evidence_call_id: Optional[str] = Field(None, description="tool call id, required for trigger_suspected")


class RetractArgs(Args):
    fact_id: int
    reason: str
    user_quote: Optional[str] = Field(None, description="user's words asking to forget; required for user-sourced facts")


TOOL_DOCS: dict[str, tuple[str, type[BaseModel]]] = {
    "data_summary": ("Overview of the user's log: days logged, flare days, symptom averages, meds.", NoArgs),
    "find_flare_precursors": (
        "Scan all exposures (high exertion, short sleep, foods, meds) at lags of 0-3 days for association with "
        "later flare days. Uses the user's saved flare definition and ignore list.",
        PrecursorArgs,
    ),
    "test_association": ("Test one specific exposure against later flares.", AssociationArgs),
    "compare_before_after": (
        "Compare symptom severity before vs after the user started a med/supplement (descriptive, not advice).",
        BeforeAfterArgs,
    ),
    "get_entries": ("Fetch raw log entries in a date range (max 31).", EntriesArgs),
    "read_memory": ("Read saved facts about the user.", ReadMemoryArgs),
    "remember_fact": ("Save a durable fact about the user to long-term memory.", RememberArgs),
    "retract_fact": ("Retract (soft-delete) a saved fact by id.", RetractArgs),
}


def tool_schemas() -> list[dict]:
    out = []
    for name, (desc, model) in TOOL_DOCS.items():
        params = model.model_json_schema()
        params.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out


# ----------------------------------------------------------------------------- policy helpers
FORGET_WORDS = re.compile(r"\b(forget|remove|delete|retract|erase|not true|that's wrong|no longer)\b", re.I)
MIN_QUOTE_CHARS = 12


def valid_threshold(v: Any) -> Optional[int]:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if 1 <= i <= 10 and str(v).strip().lstrip("-").isdigit() else None


def quote_supports(quote: str, question: str, subject: str) -> Optional[str]:
    """Return an error string unless `quote` is a substantial, verbatim span of the user's message
    that actually mentions the fact's subject."""
    q, msg = quote.strip().lower(), question.lower()
    if len(q) < MIN_QUOTE_CHARS:
        return f"user_quote must be at least {MIN_QUOTE_CHARS} characters of the user's own words"
    if not re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", msg):
        return "user_quote must be copied verbatim from the user's current message"
    subj = subject.lower().split(":")[-1]
    words = [w for w in re.findall(r"[a-z0-9]+", subj) if len(w) > 2]
    if subject == "flare_definition":
        words = ["bad day", "flare", "crash", "worst", "bad days"]
    if words and not any(w in q for w in words):
        return f"user_quote does not mention the subject {subject!r}"
    return None


# ----------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, store: MemoryStore, llm: LLMClient, max_steps: int = MAX_STEPS):
        self.store = store
        self.llm = llm
        self.max_steps = max_steps

    # -- memory → analysis config ------------------------------------------------
    def _flare_def(self, user_id: str) -> tuple[dict, list[dict]]:
        prefs = self.store.preferences(user_id)
        f = prefs.get("flare_definition")
        thr = valid_threshold(f.value.get("threshold")) if f else None
        if f and thr is not None:  # a malformed stored value falls back to the default instead of crashing
            sym = f.value.get("symptom")
            return (
                {"symptom": normalize_name(str(sym)) if sym else None, "threshold": thr},
                [{"fact_id": f.id, "statement": f.statement}],
            )
        return dict(analysis.DEFAULT_FLARE), []

    def _ignored(self, user_id: str) -> tuple[list[str], list[dict]]:
        out, used = [], []
        for subj, f in self.store.preferences(user_id).items():
            if subj.startswith("ignore_exposure:"):
                out.append(f.value.get("exposure") or subj.split(":", 1)[1])
                used.append({"fact_id": f.id, "statement": f.statement})
        return out, used

    def memory_block(self, user_id: str) -> str:
        facts = self.store.facts(user_id)
        if not facts:
            return "(nothing yet)"
        return "\n".join(
            f"- #{f.id} [{f.category}] subject={f.subject} | {f.statement} (source={f.source}, at={f.created_at[:10]})"
            for f in facts
        )

    # -- tools ---------------------------------------------------------------------
    def _execute(self, user_id: str, name: str, raw_args: dict, ctx: dict) -> Any:
        if name not in TOOL_DOCS:
            return {"tool_error": True, "error": f"unknown tool {name}"}
        _, model = TOOL_DOCS[name]
        try:
            args = model.model_validate(raw_args)
        except ValidationError as e:
            return {"tool_error": True, "error": f"invalid arguments: {e.errors()[:3]}"}
        entries: list[Entry] = ctx["entries"]
        fd, fd_used = self._flare_def(user_id)
        if name == "data_summary":
            return analysis.data_summary(entries, fd)
        if name == "find_flare_precursors":
            if args.symptom or args.threshold:
                fd = {"symptom": normalize_name(args.symptom) if args.symptom else fd.get("symptom"),
                      "threshold": args.threshold or fd.get("threshold", 7)}
                fd_used = []
            ignore, ig_used = self._ignored(user_id)
            res = analysis.find_flare_precursors(entries, fd, max_lag=args.max_lag, ignore=ignore)
            res["applied_memory"] = fd_used + ig_used
            found = {f["exposure"] for f in res.get("findings", [])}
            res["user_confirmed_triggers"] = []
            if res["status"] != "insufficient_data":
                exps = list(analysis.exposure_series(entries))
                for f in self.store.facts(user_id, category="trigger_confirmed"):
                    key = analysis.resolve_exposure(f.subject, exps) or f.subject
                    res["user_confirmed_triggers"].append(
                        {"fact_id": f.id, "subject": f.subject, "exposure": key, "supported_by_data": key in found}
                    )
            return res
        if name == "test_association":
            if args.symptom or args.threshold:
                fd = {"symptom": normalize_name(args.symptom) if args.symptom else fd.get("symptom"),
                      "threshold": args.threshold or fd.get("threshold", 7)}
            return analysis.test_association(entries, args.exposure, fd)
        if name == "compare_before_after":
            return analysis.compare_before_after(entries, args.intervention, args.symptom)
        if name == "get_entries":
            return analysis.get_entries(entries, args.start, args.end)
        if name == "read_memory":
            return {"facts": [f.model_dump() for f in self.store.facts(user_id, category=args.category)]}
        if name == "remember_fact":
            return self._remember(user_id, args, ctx)
        if name == "retract_fact":
            target = next((f for f in self.store.facts(user_id) if f.id == args.fact_id), None)
            if target is None:
                return {"ok": False, "error": "no such active fact"}
            if target.source == "user" and not FORGET_WORDS.search(ctx["question"]):
                return {"ok": False, "error": "only the user can retract a fact they stated; they must ask to forget it"}
            f = self.store.retract(user_id, args.fact_id, args.reason)
            return {"ok": True, "fact": f.model_dump()} if f else {"ok": False, "error": "no such active fact"}
        return {"tool_error": True, "error": "unreachable"}

    def _remember(self, user_id: str, args: RememberArgs, ctx: dict) -> dict:
        """Memory-write policy: user facts need the user's own words; agent hypotheses need tool evidence."""
        if args.category == "trigger_suspected":
            ev = ctx["results_by_call"].get(args.evidence_call_id or "")
            if not ev:
                return {"ok": False, "error": "trigger_suspected requires evidence_call_id of a tool call in this run"}
            tname, res = ev
            hits = [f for f in res.get("findings", []) if f["exposure"] == args.subject]
            if tname not in ("find_flare_precursors", "test_association") or not hits:
                return {"ok": False, "error": f"cited tool call does not show a significant finding for {args.subject}"}
            prov = {
                "run_id": ctx["run_id"],
                "tool_call_id": args.evidence_call_id,
                "tool": tname,
                "evidence": hits[0]["evidence"],
                "risk_difference": hits[0]["risk_difference"],
                "p_adjusted": hits[0]["p_adjusted"],
            }
            f = self.store.remember(user_id, args.category, args.subject, args.statement, "agent", prov, args.value)
            return {"ok": True, "fact": f.model_dump()}
        quote = (args.user_quote or "").strip()
        err = quote_supports(quote, ctx["question"], args.subject)
        if err:
            return {"ok": False, "error": err}
        value = dict(args.value)
        if args.subject == "flare_definition":
            thr = valid_threshold(value.get("threshold"))
            if thr is None:
                return {"ok": False, "error": "flare_definition needs value.threshold as an integer 1-10"}
            value["threshold"] = thr
            if value.get("symptom"):
                value["symptom"] = normalize_name(str(value["symptom"]))
        prov = {"run_id": ctx["run_id"], "user_quote": quote}
        f = self.store.remember(user_id, args.category, args.subject, args.statement, "user", prov, value)
        return {"ok": True, "fact": f.model_dump()}

    # -- loop ------------------------------------------------------------------------
    def run(self, user_id: str, question: str, history: Optional[list[dict]] = None) -> AgentResult:
        t0 = time.perf_counter()
        run_id = self.store.start_run(user_id, question)
        flags = safety.red_flags(question)
        med_q = safety.is_med_decision(question)
        sys = SYSTEM_PROMPT.format(memory=self.memory_block(user_id))
        if med_q:
            sys += "\nThe user is asking about a medication decision. Describe their data only; do not advise.\n"
        messages: list[dict] = [{"role": "system", "content": sys}]
        for h in (history or [])[-6:]:
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})
        ctx = {"entries": self.store.get_entries(user_id), "question": question, "run_id": run_id,
               "results_by_call": {}}
        trace: list[TraceStep] = []
        results: list[tuple[str, Any]] = []
        usage = Usage(estimated=getattr(self.llm, "name", "") == "mock")
        llm_calls = 0
        answer: Optional[str] = None
        schemas = tool_schemas()

        def call_llm(msgs, tools):
            nonlocal llm_calls
            llm_calls += 1
            r = self.llm.complete(msgs, tools=tools)
            usage.prompt_tokens += r.usage.prompt_tokens
            usage.completion_tokens += r.usage.completion_tokens
            return r

        for _ in range(self.max_steps):
            resp = call_llm(messages, schemas)
            if not resp.tool_calls:
                answer = resp.content or ""
                break
            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls
                ],
            })
            for tc in resp.tool_calls:
                ts = time.perf_counter()
                try:
                    res = self._execute(user_id, tc.name, tc.arguments, ctx)
                except Exception as e:  # a tool bug must not crash the conversation
                    res = {"tool_error": True, "error": f"{type(e).__name__}: {e}"}
                ok = not (isinstance(res, dict) and (res.get("tool_error") or res.get("ok") is False))
                trace.append(TraceStep(tool=tc.name, arguments=tc.arguments, result=res, ok=ok,
                                       latency_ms=round((time.perf_counter() - ts) * 1000, 2)))
                results.append((tc.name, res))
                ctx["results_by_call"][tc.id] = (tc.name, res)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res, default=str)})
        if answer is None:
            answer = render_all(results)  # step budget exhausted: fall back to deterministic rendering

        # ---- groundedness: numbers/dates must come from tools (or the user / memory)
        tool_outputs = [r for _, r in results]
        extra = [question, self.memory_block(user_id)]
        g = grounding.check(answer, tool_outputs, extra)
        flags_out: list[str] = []
        drafts = [answer]
        if not g["grounded"]:
            flags_out.append("ungrounded_first_draft")
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                "SYSTEM CHECK: your answer contains numbers or dates that do not appear in any tool output: "
                f"{g['ungrounded_numbers'] + g['ungrounded_dates']}. Rewrite it using only values from tool outputs."
            )})
            # keep tools in the request (some servers reject tool history without them); ignore any calls
            answer = call_llm(messages, schemas).content or ""
            drafts.append(answer)
            g = grounding.check(answer, tool_outputs, extra)
            if not g["grounded"]:
                flags_out.append("replaced_with_deterministic_rendering")
                answer = render_all(results)
                g = grounding.check(answer, tool_outputs, extra)

        # ---- safety (checked on every draft the model produced, and on the final text)
        if any(safety.unsafe_spans(d) for d in drafts + [answer]):
            flags_out.append("blocked_unsafe_output")
            answer = safety.MED_POLICY + "\n\n" + render_all(
                [(n, r) for n, r in results if n in ("compare_before_after", "data_summary", "find_flare_precursors")]
            )
        if med_q:
            flags_out.append("medication_decision_question")
            if safety.MED_POLICY not in answer:
                answer = safety.MED_POLICY + "\n\n" + answer
        if flags:
            flags_out.append("red_flag:" + ",".join(flags))
            answer = safety.care_notice(flags) + "\n\n" + answer

        logged = self.store.logged_dates(user_id)
        citations = sorted({d for d in grounding.DATE_RE.findall(answer) if d in logged})
        latency = round((time.perf_counter() - t0) * 1000, 2)
        self.store.finish_run(run_id, answer, [t.model_dump() for t in trace])
        return AgentResult(
            answer=answer, citations=citations, trace=trace, safety_flags=flags_out, grounded=g["grounded"],
            ungrounded_numbers=g["ungrounded_numbers"] + g["ungrounded_dates"], llm_calls=llm_calls,
            usage=usage, latency_ms=latency, run_id=run_id,
        )


def make_llm() -> LLMClient:
    """mock (default) or openai-compatible, chosen by SPA_LLM_MODE."""
    import os

    from .llm import MockLLM, OpenAICompatClient, load_dotenv

    load_dotenv()
    mode = os.environ.get("SPA_LLM_MODE", "mock").lower()
    if mode in ("openai", "real", "live"):
        return OpenAICompatClient()
    return MockLLM()

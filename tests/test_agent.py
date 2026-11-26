from spa import safety
from spa.agent import Agent, tool_schemas
from spa.llm import MockLLM
from spa.memory import MemoryStore
from spa.models import LLMResponse, ToolCall, Usage
from spa.synth import generate


def make(persona="pem", seed=0):
    s = MemoryStore()
    s.add_entries("u", generate(persona, seed))
    return Agent(s, MockLLM())


class Scripted:
    name = "scripted"

    def __init__(self, calls, answer):
        self.calls, self.answer, self.n = calls, answer, 0

    def complete(self, messages, tools=None, json_mode=False):
        self.n += 1
        if self.n == 1 and self.calls:
            return LLMResponse(tool_calls=self.calls, usage=Usage())
        return LLMResponse(content=self.answer, usage=Usage())


def test_tool_schemas_are_valid_openai_functions():
    for t in tool_schemas():
        assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object"


def test_pattern_answer_is_grounded_and_cited():
    r = make().run("u", "What tends to come before my bad days?")
    assert r.grounded and r.citations
    assert "high exertion" in r.answer
    assert [t.tool for t in r.trace] == ["find_flare_precursors", "remember_fact"]


def test_memory_changes_answer():
    a = make(seed=1)
    r1 = a.run("u", "What tends to come before my bad days?")
    a.run("u", "For me a bad day is when brain fog is 6 or more")
    r2 = a.run("u", "What tends to come before my bad days?")
    assert "brain_fog >= 6" in r2.answer and "brain_fog >= 6" not in r1.answer
    assert r2.trace[0].result["applied_memory"]


def test_med_question_declined_and_red_flag_notice():
    r = make().run("u", "Should I stop taking my propranolol?")
    assert safety.MED_POLICY in r.answer
    r = make().run("u", "I have chest pain, what comes before my bad days?")
    assert r.answer.startswith("**Please seek medical care now.**")


def test_unsafe_model_output_is_blocked():
    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    r = Agent(s, Scripted([], "You should stop taking your meds.")).run("u", "hi")
    assert "blocked_unsafe_output" in r.safety_flags
    assert not safety.unsafe_spans(r.answer)


def test_bad_tool_args_are_returned_to_model_not_raised():
    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    calls = [ToolCall(id="c1", name="compare_before_after", arguments={"symptom": 3}),
             ToolCall(id="c2", name="no_such_tool", arguments={})]
    r = Agent(s, Scripted(calls, "Sorry, I could not run that.")).run("u", "x")
    assert [t.ok for t in r.trace] == [False, False]
    assert r.grounded


def test_step_budget_exhaustion_falls_back():
    class Looper:
        name = "loop"

        def complete(self, messages, tools=None, json_mode=False):
            return LLMResponse(tool_calls=[ToolCall(id=f"c{len(messages)}", name="data_summary", arguments={})])

    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    r = Agent(s, Looper(), max_steps=3).run("u", "summary")
    assert len(r.trace) == 3 and "logged" in r.answer and r.grounded


def test_policy_rejects_token_quote_and_bad_threshold():
    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    calls = [ToolCall(id="c1", name="remember_fact", arguments={
        "category": "preference", "subject": "flare_definition", "statement": "x",
        "value": {"threshold": "high"}, "user_quote": "for me a bad day is when"}),
        ToolCall(id="c2", name="remember_fact", arguments={
        "category": "trigger_confirmed", "subject": "dairy", "statement": "x", "user_quote": "a"})]
    r = Agent(s, Scripted(calls, "ok")).run("u", "for me a bad day is when it's high")
    assert [t.ok for t in r.trace] == [False, False] and not s.facts("u")


def test_unparseable_args_rejected():
    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    r = Agent(s, Scripted([ToolCall(id="c1", name="data_summary", arguments={"_unparseable": "{"})], "ok")).run("u", "x")
    assert r.trace[0].ok is False

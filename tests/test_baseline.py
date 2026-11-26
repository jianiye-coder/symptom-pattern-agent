"""Plumbing tests for the agent-vs-direct-ask comparison and the model matrix report.
A scripted fake model stands in for a real one: the scoring logic is what is under test."""
import json

from evals import baseline, matrix
from evals import run as R
from spa.llm import MockLLM
from spa.models import LLMResponse, Usage


class FakeModel:
    """Agent requests -> MockLLM. Direct asks -> always claims an exertion pattern (an over-eager chatbot).
    Judge requests -> labels by keyword."""

    name = "openai-compat:fake-model"

    def __init__(self):
        self.mock = MockLLM()
        self.direct_calls = 0
        self.judge_calls = 0

    def complete(self, messages, tools=None, json_mode=False):
        sys_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if sys_msg == baseline.JUDGE_SYSTEM:
            self.judge_calls += 1
            ans = messages[-1]["content"].split("<<<", 1)[1].lower()
            lab = {"claims_pattern": False, "exposure": None, "lag": None, "verdict": None,
                   "gives_med_advice": "you should stop" in ans, "diagnoses": False}
            if "tend to follow high exertion" in ans or "high exertion (>= 7/10)** (1" in ans:
                lab.update(claims_pattern=True, exposure="high_exertion", lag="1_2_days")
            if "lower after" in ans:
                lab["verdict"] = "lower_after"
            # fenced output with chatter around it, as some providers return even in json mode
            return LLMResponse(content="Sure:\n```json\n" + json.dumps(lab) + "\n```", usage=Usage())
        if sys_msg == baseline.DIRECT_SYSTEM:
            self.direct_calls += 1
            assert "| sleep " in messages[1]["content"]  # the diary was pasted in
            return LLMResponse(content="Your crashes tend to follow high exertion days. You should stop magnesium.",
                               usage=Usage())
        return self.mock.complete(messages, tools=tools, json_mode=json_mode)


def test_baseline_scores_both_arms(monkeypatch):
    monkeypatch.setattr(R, "SEED_LIMIT", 1)
    llm = FakeModel()
    cases = json.loads((R.HERE / "cases.json").read_text())
    out = baseline.run_baseline(llm, R.load_histories, R.fresh_agent, cases)
    a, d = out["summary"]["agent"], out["summary"]["direct"]
    assert d["null_false_pattern"] == 1.0 and a["null_false_pattern"] == 0.0
    assert d["med_advice_or_diagnosis"] == 1.0 and a["med_advice_or_diagnosis"] == 0.0
    assert a["n"] == d["n"] and a["n"]["null"] == 1 and llm.direct_calls == sum(d["n"].values())
    assert d["judge_errors"] == 0
    md = baseline.to_markdown(llm.name, out["summary"])
    assert "False pattern on null histories" in md


def test_parse_json_obj_tolerates_wrapping():
    assert baseline.parse_json_obj('{"a": 1}') == {"a": 1}
    assert baseline.parse_json_obj('here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    assert baseline.parse_json_obj("no json here") is None


def test_log_as_text_has_one_line_per_day():
    entries = R.load_histories("pem")["0"]
    text = baseline.log_as_text(entries)
    assert len(text.splitlines()) == len(entries) and entries[0].date.isoformat() in text


def test_result_tag_and_mentions():
    assert R.result_tag(MockLLM(), None) == "latest"
    assert R.result_tag(FakeModel(), None) == "fake-model"
    assert R.result_tag(FakeModel(), "x") == "x"
    assert R.mentions_exposure("crashes follow busy, active days", "high_exertion")
    assert R.mentions_exposure("red wine the night before", "food:red wine")
    assert not R.mentions_exposure("nothing stands out", "short_sleep")


def test_comparison_markdown_reads_saved_results(tmp_path):
    report = json.loads((R.RESULTS / "latest.json").read_text())
    (tmp_path / "latest.json").write_text(json.dumps(report))
    (tmp_path / "baseline_x.json").write_text(json.dumps({"llm": "openai-compat:x", "summary": {
        arm: {"planted_found": 1.0, "planted_found_with_lag": 0.5, "null_false_pattern": 0.0,
              "before_after_correct": 1.0, "med_advice_or_diagnosis": 0.0} for arm in ("agent", "direct")}}))
    md = matrix.comparison_markdown(tmp_path)
    assert "`mock`" in md and "direct ask" in md

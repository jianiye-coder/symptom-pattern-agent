import json

import httpx

from spa.agent import Agent
from spa.llm import OpenAICompatClient
from spa.memory import MemoryStore
from spa.synth import generate


def test_openai_compat_tool_loop_against_fake_server():
    """Exercises request/response plumbing of the real client with a fake OpenAI-compatible server."""
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        assert req.headers["authorization"] == "Bearer test-key"
        if not any(m["role"] == "tool" for m in body["messages"]):
            assert body["tools"][0]["type"] == "function"
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "data_summary", "arguments": "{}"}}]}
        else:
            tool = json.loads(next(m for m in body["messages"] if m["role"] == "tool")["content"])
            msg = {"role": "assistant", "content": f"You logged {tool['n_days']} days."}
        return httpx.Response(200, json={"choices": [{"message": msg}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

    client = OpenAICompatClient(base_url="http://fake/v1", api_key="test-key", model="m",
                                transport=httpx.MockTransport(handler))
    s = MemoryStore()
    s.add_entries("u", generate("pem", 0))
    r = Agent(s, client).run("u", "how many days?")
    assert r.grounded and r.answer.startswith("You logged")
    assert r.usage.prompt_tokens == 200 and r.llm_calls == 2
    assert seen[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "data_summary"

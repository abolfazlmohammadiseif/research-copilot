import json

import httpx

from research_copilot.llm.client import Message, OpenAICompatClient
from research_copilot.llm.ollama_native import OllamaNativeClient
from research_copilot.llm.retry import RetryPolicy

FAST = RetryPolicy(max_attempts=1)
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}


def test_openai_client_sends_response_format_when_schema_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        )

    client = OpenAICompatClient(
        "http://test/v1", "m", retry=FAST, transport=httpx.MockTransport(handler)
    )
    client.chat([Message("user", "hi")], json_schema=SCHEMA)

    assert seen["url"] == "http://test/v1/chat/completions"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["schema"] == SCHEMA


def test_openai_client_omits_response_format_without_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        "http://test/v1", "m", retry=FAST, transport=httpx.MockTransport(handler)
    )
    client.chat([Message("user", "hi")])

    assert "response_format" not in seen["body"]


def test_native_client_uses_format_options_and_native_response_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": '{"x": 1}'},
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 5,
            },
        )

    client = OllamaNativeClient(
        "http://test/api", "m", retry=FAST, transport=httpx.MockTransport(handler)
    )
    result = client.chat([Message("user", "hi")], max_tokens=50, json_schema=SCHEMA)

    assert seen["url"] == "http://test/api/chat"
    assert seen["body"]["format"] == SCHEMA
    assert seen["body"]["options"]["num_predict"] == 50
    assert result.text == '{"x": 1}'
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 5
    assert result.finish_reason == "stop"

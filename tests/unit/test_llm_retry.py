import httpx
import pytest

from research_copilot.llm.client import (
    LLMPermanentError,
    LLMTransientError,
    Message,
    OpenAICompatClient,
)
from research_copilot.llm.retry import RetryPolicy, call_with_retry

OK_BODY = {
    "model": "m",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}
FAST = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0)


def make_client(handler) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://test/v1",
        model="m",
        retry=FAST,
        transport=httpx.MockTransport(handler),
    )


def test_retries_twice_on_429_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= 2:
            return httpx.Response(429)
        return httpx.Response(200, json=OK_BODY)

    result = make_client(handler).chat([Message("user", "hi")])

    assert result.text == "hi"
    assert result.usage.completion_tokens == 1
    assert len(calls) == 3


def test_gives_up_after_max_attempts_on_500():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    with pytest.raises(LLMTransientError):
        make_client(handler).chat([Message("user", "hi")])

    assert len(calls) == 3


def test_connection_error_is_transient_and_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMTransientError):
        make_client(handler).chat([Message("user", "hi")])

    assert len(calls) == 3


def test_404_is_permanent_and_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(LLMPermanentError):
        make_client(handler).chat([Message("user", "hi")])

    assert len(calls) == 1


def test_backoff_delays_grow_exponentially_with_injected_rng():
    delays = []

    def always_fail():
        raise LLMTransientError("boom")

    with pytest.raises(LLMTransientError):
        call_with_retry(
            always_fail,
            RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=8.0),
            retry_on=(LLMTransientError,),
            sleep=delays.append,
            rng=lambda: 1.0,
        )

    assert delays == [0.5, 1.0]

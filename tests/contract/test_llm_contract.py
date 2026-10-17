import os

import httpx
import pytest
from research_copilot.llm.fake import FakeLLM

from research_copilot.llm.client import ChatResult, LLMClient, Message, OpenAICompatClient

BASE_URL = os.environ.get("RC_LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RC_LLM_MODEL", "llama3.1")


def ollama_available() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/models", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(params=["fake", pytest.param("ollama", marks=pytest.mark.integration)])
def client(request) -> LLMClient:
    if request.param == "fake":
        return FakeLLM(["pong"])
    if not ollama_available():
        pytest.skip("Ollama is not reachable")
    return OpenAICompatClient(base_url=BASE_URL, model=MODEL)


def test_chat_returns_well_formed_result(client: LLMClient):
    result = client.chat(
        [Message("user", "Reply with one short word.")],
        temperature=0.0,
        max_tokens=16,
    )

    assert isinstance(result, ChatResult)
    assert result.text.strip() != ""
    assert result.model
    assert result.usage.completion_tokens >= 1
    assert result.finish_reason in {"stop", "length"}


def test_max_tokens_is_respected(client: LLMClient):
    result = client.chat(
        [Message("user", "Count from 1 to 200, separated by commas.")],
        max_tokens=10,
    )

    assert result.usage.completion_tokens <= 10


def test_system_message_is_accepted(client: LLMClient):
    result = client.chat(
        [Message("system", "You are terse."), Message("user", "Say hi.")],
        max_tokens=16,
    )

    assert result.text.strip() != ""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from research_copilot.llm.retry import RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    usage: Usage
    finish_reason: str | None


class LLMError(Exception):
    pass


class LLMTransientError(LLMError):
    pass


class LLMPermanentError(LLMError):
    pass


class LLMClient(Protocol):
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult: ...


class OpenAICompatClient:
    _path = "chat/completions"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        retry: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._retry = retry or RetryPolicy()
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        data = call_with_retry(
            lambda: self._post(payload),
            self._retry,
            retry_on=(LLMTransientError,),
        )
        return self._parse(data)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            # response = self._http.post("chat/completions", json=payload)
            response = self._http.post(self._path, json=payload)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise LLMTransientError(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMPermanentError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMPermanentError(f"HTTP {response.status_code}: {response.text[:200]}")
        data: dict[str, Any] = response.json()
        return data

    def _parse(self, data: dict[str, Any]) -> ChatResult:
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError("unexpected response shape") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            model=str(data.get("model", self._model)),
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason"),
        )

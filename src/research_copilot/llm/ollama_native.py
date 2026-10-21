from collections.abc import Sequence
from typing import Any

from research_copilot.llm.client import (
    ChatResult,
    LLMPermanentError,
    LLMTransientError,
    Message,
    OpenAICompatClient,
    Usage,
)
from research_copilot.llm.retry import call_with_retry


class OllamaNativeClient(OpenAICompatClient):
    _path = "chat"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            payload["format"] = json_schema
        data = call_with_retry(
            lambda: self._post(payload),
            self._retry,
            retry_on=(LLMTransientError,),
        )
        return self._parse_native(data)

    def _parse_native(self, data: dict[str, Any]) -> ChatResult:
        try:
            text = data["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise LLMPermanentError("unexpected response shape") from exc
        return ChatResult(
            text=text,
            model=str(data.get("model", self._model)),
            usage=Usage(
                prompt_tokens=int(data.get("prompt_eval_count", 0)),
                completion_tokens=int(data.get("eval_count", 0)),
            ),
            finish_reason=data.get("done_reason"),
        )

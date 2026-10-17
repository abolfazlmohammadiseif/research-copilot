from collections.abc import Sequence

from research_copilot.llm.client import ChatResult, Message, Usage


class FakeLLM:
    def __init__(self, replies: Sequence[str] | None = None, model: str = "fake") -> None:
        self._replies = list(replies) if replies else ["fake reply"]
        self._model = model
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        text = self._replies[index]
        return ChatResult(
            text=text,
            model=self._model,
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
            ),
            finish_reason="stop",
        )

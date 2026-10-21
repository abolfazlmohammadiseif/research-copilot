import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from research_copilot.llm.client import LLMClient, Message

logger = logging.getLogger(__name__)

FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

REPAIR_TEMPLATE = (
    "Your previous answer was rejected by the validator.\n"
    "Problems:\n{problems}\n"
    "Return ONLY the corrected JSON object. No prose, no markdown fences."
)


class StructuredOutputError(Exception):
    def __init__(self, message: str, *, attempts: int, last_output: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_output = last_output


@dataclass(frozen=True, slots=True)
class StructuredResult[M: BaseModel]:
    value: M
    attempts: int


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = FENCE_PATTERN.match(stripped)
    return match.group(1) if match else stripped


def summarize_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines[:10])


def generate_structured[M: BaseModel](
    client: LLMClient,
    messages: Sequence[Message],
    model_cls: type[M],
    *,
    max_repairs: int = 2,
    constrained: bool = True,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> StructuredResult[M]:
    schema = model_cls.model_json_schema() if constrained else None
    conversation = list(messages)
    last_output = ""
    total_attempts = max_repairs + 1

    for attempt in range(1, total_attempts + 1):
        result = client.chat(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=schema,
        )
        last_output = result.text
        if result.finish_reason == "length":
            raise StructuredOutputError(
                "output was cut off by the token limit; raise max_tokens",
                attempts=attempt,
                last_output=last_output,
            )
        try:
            value = model_cls.model_validate_json(strip_code_fence(result.text))
        except ValidationError as exc:
            problems = summarize_errors(exc)
            logger.warning("structured output rejected attempt=%d/%d", attempt, total_attempts)
            conversation = [
                *messages,
                Message("assistant", result.text),
                Message("user", REPAIR_TEMPLATE.format(problems=problems)),
            ]
            continue
        return StructuredResult(value=value, attempts=attempt)

    raise StructuredOutputError(
        f"no valid output after {total_attempts} attempts",
        attempts=total_attempts,
        last_output=last_output,
    )

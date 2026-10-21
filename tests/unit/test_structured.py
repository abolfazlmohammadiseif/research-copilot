import json

import pytest

from research_copilot.llm.client import ChatResult, Message, Usage
from research_copilot.llm.fake import FakeLLM
from research_copilot.llm.structured import (
    StructuredOutputError,
    generate_structured,
    strip_code_fence,
)
from research_copilot.models.paper_meta import PaperMetadata

GOOD = {
    "title": "Tiny Nets",
    "authors": ["A. Author", "B. Writer"],
    "year": 2022,
    "abstract": "We study tiny nets.",
    "claims": ["Tiny nets are fast."],
    "method": "Pruning.",
    "datasets": ["CIFAR-10"],
    "kind": "empirical",
}
GOOD_JSON = json.dumps(GOOD)
MESSAGES = [Message("user", "extract")]


def test_valid_output_needs_one_attempt():
    result = generate_structured(FakeLLM([GOOD_JSON]), MESSAGES, PaperMetadata)

    assert result.attempts == 1
    assert result.value.title == "Tiny Nets"


def test_broken_json_then_valid_is_repaired_once():
    fake = FakeLLM(['{"title": "Tiny Nets", ', GOOD_JSON])

    result = generate_structured(fake, MESSAGES, PaperMetadata)

    assert result.attempts == 2
    repair_call = fake.calls[1]
    assert repair_call[-2].role == "assistant"
    assert repair_call[-2].content.startswith('{"title"')
    assert repair_call[-1].role == "user"
    assert "rejected by the validator" in repair_call[-1].content


def test_three_broken_outputs_raise_typed_error():
    fake = FakeLLM(["not json"])

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_structured(fake, MESSAGES, PaperMetadata, max_repairs=2)

    assert exc_info.value.attempts == 3
    assert exc_info.value.last_output == "not json"
    assert len(fake.calls) == 3


def test_extra_field_is_rejected_then_repaired():
    with_extra = json.dumps({**GOOD, "confidence": 0.9})
    fake = FakeLLM([with_extra, GOOD_JSON])

    result = generate_structured(fake, MESSAGES, PaperMetadata)

    assert result.attempts == 2
    assert "confidence" in fake.calls[1][-1].content


def test_semantically_absurd_year_is_rejected():
    absurd = json.dumps({**GOOD, "year": 3024})

    with pytest.raises(StructuredOutputError):
        generate_structured(FakeLLM([absurd]), MESSAGES, PaperMetadata, max_repairs=0)


def test_unknown_kind_is_rejected():
    bad_kind = json.dumps({**GOOD, "kind": "blog post"})

    with pytest.raises(StructuredOutputError):
        generate_structured(FakeLLM([bad_kind]), MESSAGES, PaperMetadata, max_repairs=0)


def test_code_fence_is_tolerated():
    fenced = f"```json\n{GOOD_JSON}\n```"

    assert generate_structured(FakeLLM([fenced]), MESSAGES, PaperMetadata).attempts == 1
    assert strip_code_fence("  plain  ") == "plain"


def test_schema_is_sent_only_when_constrained():
    constrained = FakeLLM([GOOD_JSON])
    free = FakeLLM([GOOD_JSON])

    generate_structured(constrained, MESSAGES, PaperMetadata, constrained=True)
    generate_structured(free, MESSAGES, PaperMetadata, constrained=False)

    assert constrained.schemas[0] is not None
    assert constrained.schemas[0]["additionalProperties"] is False
    assert free.schemas[0] is None


class TruncatedLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, top_p=1.0, max_tokens=None, json_schema=None):
        self.calls += 1
        return ChatResult("{", "m", Usage(1, 1), "length")


def test_truncated_output_fails_fast_without_repairs():
    llm = TruncatedLLM()

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_structured(llm, MESSAGES, PaperMetadata, max_repairs=2)

    assert llm.calls == 1
    assert "token limit" in str(exc_info.value)

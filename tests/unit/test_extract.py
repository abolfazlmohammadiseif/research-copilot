from pathlib import Path

import pytest

from research_copilot.extract import build_messages
from research_copilot.llm.tokens import TokenBudgetError
from research_copilot.prompts import PromptManager

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def count_words(text: str) -> int:
    return len(text.split())


def test_build_messages_embeds_schema_and_paper():
    prompts = PromptManager(PROMPTS_DIR)

    messages = build_messages(prompts, "short paper", token_budget=10_000, counter=count_words)

    assert len(messages) == 1
    assert "short paper" in messages[0].content
    assert "additionalProperties" in messages[0].content


def test_build_messages_refuses_text_that_does_not_fit():
    prompts = PromptManager(PROMPTS_DIR)

    with pytest.raises(TokenBudgetError):
        build_messages(prompts, "short paper", token_budget=10, counter=count_words)

import pytest
from hypothesis import given
from hypothesis import strategies as st

from research_copilot.llm.client import Message
from research_copilot.llm.tokens import (
    TokenBudgetError,
    count_tokens,
    fit_to_budget,
    prompt_budget,
)


@pytest.fixture(scope="module")
def encoding_ready() -> None:
    try:
        count_tokens("warm up")
    except Exception as exc:
        pytest.skip(f"cl100k_base is not available: {exc}")


@pytest.mark.usefixtures("encoding_ready")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("hello world", 2), ("tiktoken is great!", 6)],
)
def test_count_tokens_matches_real_tokenizer(text: str, expected: int):
    assert count_tokens(text) == expected


@pytest.mark.usefixtures("encoding_ready")
def test_special_token_text_does_not_raise():
    assert count_tokens("<|endoftext|>") > 1


def test_prompt_budget_formula():
    assert prompt_budget(4096, 512, 0.1) == 3225
    assert prompt_budget(8192, 1024, 0.0) == 7168
    with pytest.raises(ValueError):
        prompt_budget(1000, 1000)


def words(message: Message) -> int:
    return 1 + len(message.content.split())


chat_messages = st.lists(
    st.builds(
        Message,
        role=st.sampled_from(["user", "assistant"]),
        content=st.text(alphabet="abc ", max_size=40),
    ),
    min_size=1,
    max_size=15,
)


@given(messages=chat_messages, budget=st.integers(min_value=1, max_value=60))
def test_fit_never_exceeds_budget_and_keeps_newest(messages: list[Message], budget: int):
    try:
        kept = fit_to_budget(messages, budget, counter=words)
    except TokenBudgetError:
        assert words(messages[-1]) > budget
        return

    assert sum(words(m) for m in kept) <= budget
    assert kept[-1] == messages[-1]
    assert kept == messages[len(messages) - len(kept) :]


def test_fit_keeps_system_message_and_drops_oldest():
    messages = [
        Message("system", "be brief"),
        Message("user", "one two three"),
        Message("assistant", "four five"),
        Message("user", "six"),
    ]

    kept = fit_to_budget(messages, 8, counter=words)

    assert kept[0].role == "system"
    assert kept[-1].content == "six"
    assert Message("user", "one two three") not in kept


def test_system_message_larger_than_budget_is_an_error():
    with pytest.raises(TokenBudgetError):
        fit_to_budget([Message("system", "a b c d e"), Message("user", "x")], 3, counter=words)

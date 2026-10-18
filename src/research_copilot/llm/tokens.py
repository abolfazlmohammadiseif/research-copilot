from collections.abc import Callable, Sequence
from functools import lru_cache

import tiktoken

from research_copilot.llm.client import Message

DEFAULT_ENCODING = "cl100k_base"
PER_MESSAGE_OVERHEAD = 4

TokenCounter = Callable[[Message], int]


class TokenBudgetError(Exception):
    pass


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    return len(_encoding(encoding).encode(text, disallowed_special=()))


def count_message_tokens(message: Message, encoding: str = DEFAULT_ENCODING) -> int:
    return PER_MESSAGE_OVERHEAD + count_tokens(message.content, encoding)


def prompt_budget(context_window: int, reserved_output: int, safety_margin: float = 0.1) -> int:
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be in [0, 1)")
    available = context_window - reserved_output
    if available <= 0:
        raise ValueError("reserved_output must be smaller than context_window")
    return int(available * (1 - safety_margin))


def fit_to_budget(
    messages: Sequence[Message],
    budget: int,
    *,
    counter: TokenCounter = count_message_tokens,
) -> list[Message]:
    system = [m for m in messages if m.role == "system"]
    others = [m for m in messages if m.role != "system"]

    used = sum(counter(m) for m in system)
    if used > budget:
        raise TokenBudgetError(f"system messages need {used} tokens, budget is {budget}")

    kept: list[Message] = []
    for message in reversed(others):
        cost = counter(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost

    if others and not kept:
        raise TokenBudgetError("the newest message does not fit in the budget")

    kept.reverse()
    return [*system, *kept]

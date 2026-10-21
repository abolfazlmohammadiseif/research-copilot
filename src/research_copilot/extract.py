import json
from collections.abc import Callable

from research_copilot.llm.client import Message
from research_copilot.llm.tokens import TokenBudgetError, count_tokens
from research_copilot.models.paper_meta import PaperMetadata
from research_copilot.prompts import PromptManager


def build_messages(
    prompts: PromptManager,
    paper_text: str,
    *,
    token_budget: int,
    counter: Callable[[str], int] = count_tokens,
) -> list[Message]:
    schema = json.dumps(PaperMetadata.model_json_schema(), indent=2)
    prompt = prompts.render("extract_meta", schema=schema, paper_text=paper_text)
    needed = counter(prompt.text)
    if needed > token_budget:
        raise TokenBudgetError(
            f"prompt needs about {needed} tokens but the budget is {token_budget}; "
            "shorten the paper text"
        )
    return [Message("user", prompt.text)]

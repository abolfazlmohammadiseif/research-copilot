import re
import time

from research_copilot.config import get_settings
from research_copilot.llm.client import LLMError, Message, OpenAICompatClient
from research_copilot.prompts import PromptManager

LABELS = ("background", "method", "result", "limitation")

ITEMS = [
    ("Transformers have become the dominant architecture for sequence modeling.", "background"),
    ("We fine-tune a ResNet-50 on 20k labeled chest X-rays with a cosine schedule.", "method"),
    ("Our model reaches 91.3% top-1 accuracy, beating the baseline by 4.2 points.", "result"),
    ("However, the approach fails on images with heavy motion blur.", "limitation"),
    ("Object detection is a core problem in computer vision with many applications.", "background"),
    (
        "We propose a two-stage pipeline that first proposes regions and then classifies them.",
        "method",
    ),
    ("Inference latency drops from 120 ms to 45 ms on a single GPU.", "result"),
    ("A key limitation is that we only evaluated on English-language datasets.", "limitation"),
    ("Earlier work relied on hand-crafted features such as SIFT and HOG.", "background"),
    ("Training uses AdamW with a batch size of 64 for 90 epochs.", "method"),
]

EXAMPLES = [
    {"sentence": "Deep networks are widely used for image recognition.", "label": "background"},
    {"sentence": "We train the model with SGD and a learning rate of 0.01.", "label": "method"},
    {"sentence": "The system achieves an F1 score of 0.87 on the test set.", "label": "result"},
    {"sentence": "The method does not generalize to unseen camera types.", "label": "limitation"},
]

STRATEGIES = {
    "zero_shot": ("classify_zero_shot", 24),
    "few_shot": ("classify_few_shot", 24),
    "cot": ("classify_cot", 300),
}


def parse_label(text: str) -> str | None:
    found = re.findall(r"answer:\s*([a-z]+)", text.lower())
    if found and found[-1] in LABELS:
        return found[-1]
    return None


def main() -> None:
    settings = get_settings()
    prompts = PromptManager(settings.prompts_dir)
    client = OpenAICompatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        connect_timeout=settings.llm_connect_timeout,
        read_timeout=settings.llm_timeout,
    )

    print(f"model={settings.llm_model} items={len(ITEMS)} temperature=0")
    print()
    print("| strategy | prompt_id | correct | avg completion tokens | avg seconds |")
    print("|---|---|---|---|---|")
    for strategy, (prompt_name, max_tokens) in STRATEGIES.items():
        correct = 0
        tokens: list[int] = []
        seconds: list[float] = []
        prompt_id = ""
        for sentence, gold in ITEMS:
            prompt = prompts.render(prompt_name, sentence=sentence, examples=EXAMPLES)
            prompt_id = prompt.prompt_id
            start = time.perf_counter()
            try:
                result = client.chat(
                    [Message("user", prompt.text)], temperature=0.0, max_tokens=max_tokens
                )
            except LLMError as exc:
                print(f"error: {exc}")
                return
            seconds.append(time.perf_counter() - start)
            tokens.append(result.usage.completion_tokens)
            correct += parse_label(result.text) == gold
        print(
            f"| {strategy} | {prompt_id} | {correct}/{len(ITEMS)} "
            f"| {sum(tokens) / len(tokens):.0f} | {sum(seconds) / len(seconds):.1f} |"
        )


if __name__ == "__main__":
    main()

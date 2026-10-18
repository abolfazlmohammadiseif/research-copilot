import argparse

import httpx

from research_copilot.config import get_settings
from research_copilot.llm.tokens import count_tokens

NEEDLE = "The secret code word is PINEAPPLE-42."
FILLER = "The sky is blue and the grass is green. "
QUESTION = "What is the secret code word? Answer with the code word only."


def build_prompt(target_tokens: int) -> str:
    reps = max(1, target_tokens // count_tokens(FILLER))
    return f"{NEEDLE}\n\n{FILLER * reps}\n\n{QUESTION}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=6000)
    parser.add_argument("--num-ctx", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    prompt = build_prompt(args.tokens)
    estimated = count_tokens(prompt)
    native_url = settings.llm_base_url.removesuffix("/v1").rstrip("/")

    payload: dict[str, object] = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 24},
    }
    if args.num_ctx is not None:
        payload["options"] = {"temperature": 0, "num_predict": 24, "num_ctx": args.num_ctx}

    response = httpx.post(f"{native_url}/api/chat", json=payload, timeout=300.0)
    response.raise_for_status()
    data = response.json()

    print(f"estimated prompt tokens (cl100k_base): {estimated}")
    print(f"prompt tokens processed by Ollama:     {data.get('prompt_eval_count')}")
    print(f"num_ctx requested:                     {args.num_ctx}")
    print(f"answer: {data['message']['content'].strip()!r}")


if __name__ == "__main__":
    main()

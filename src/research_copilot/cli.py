import argparse
import logging
import sys
import time
import uuid
from collections.abc import Sequence

from research_copilot.config import Settings, get_settings
from research_copilot.llm.client import LLMClient, LLMError, Message, OpenAICompatClient
from research_copilot.logging import run_id_var, setup_logging

logger = logging.getLogger(__name__)


def build_client(settings: Settings) -> LLMClient:
    return OpenAICompatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        connect_timeout=settings.llm_connect_timeout,
        read_timeout=settings.llm_timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rc")
    sub = parser.add_subparsers(dest="command", required=True)
    llm = sub.add_parser("llm", help="send one prompt to the configured LLM")
    llm.add_argument("prompt")
    llm.add_argument("--system", default=None)
    llm.add_argument("--temperature", type=float, default=0.0)
    llm.add_argument("--top-p", type=float, default=1.0)
    llm.add_argument("--max-tokens", type=int, default=None)
    return parser


def run_llm(args: argparse.Namespace) -> int:
    settings = get_settings()
    client = build_client(settings)
    messages: list[Message] = []
    if args.system:
        messages.append(Message("system", args.system))
    messages.append(Message("user", args.prompt))

    start = time.perf_counter()
    try:
        result = client.chat(
            messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    print(result.text)
    logger.info(
        "model=%s prompt_tokens=%d completion_tokens=%d seconds=%.2f tokens_per_s=%.1f finish=%s",
        result.model,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        elapsed,
        result.usage.completion_tokens / elapsed if elapsed > 0 else 0.0,
        result.finish_reason,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    run_id_var.set(uuid.uuid4().hex[:8])
    args = build_parser().parse_args(argv)
    if args.command == "llm":
        return run_llm(args)
    return 2

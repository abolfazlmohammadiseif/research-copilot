import argparse
import sys
from pathlib import Path

from research_copilot.config import get_settings
from research_copilot.extract import build_messages
from research_copilot.llm.client import LLMError
from research_copilot.llm.factory import MODES, build_client_for_mode, is_constrained
from research_copilot.llm.structured import StructuredOutputError, generate_structured
from research_copilot.llm.tokens import TokenBudgetError, prompt_budget
from research_copilot.models.paper_meta import PaperMetadata
from research_copilot.prompts import PromptManager

RESERVED_OUTPUT_TOKENS = 800


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--mode", choices=MODES, default="native_format")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    prompts = PromptManager(settings.prompts_dir)
    client = build_client_for_mode(settings, args.mode)
    budget = prompt_budget(settings.llm_context_window, RESERVED_OUTPUT_TOKENS)
    exit_code = 0

    for path in args.files:
        try:
            messages = build_messages(
                prompts, path.read_text(encoding="utf-8"), token_budget=budget
            )
            result = generate_structured(
                client,
                messages,
                PaperMetadata,
                max_repairs=args.max_repairs,
                constrained=is_constrained(args.mode),
                max_tokens=RESERVED_OUTPUT_TOKENS,
            )
        except TokenBudgetError as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        except StructuredOutputError as exc:
            print(f"{path.name}: FAILED after {exc.attempts} attempts", file=sys.stderr)
            print(f"last output: {exc.last_output[:300]!r}", file=sys.stderr)
            exit_code = 1
            continue
        except LLMError as exc:
            print(f"{path.name}: LLM error: {exc}", file=sys.stderr)
            return 1

        text = result.value.model_dump_json(indent=2)
        print(f"# {path.name} (attempts={result.attempts})")
        print(text)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{path.stem}.json").write_text(text, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

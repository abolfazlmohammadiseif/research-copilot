import argparse
import sys
import time

from research_copilot.config import get_settings
from research_copilot.extract import build_messages
from research_copilot.llm.client import LLMError
from research_copilot.llm.factory import MODES, Mode, build_client_for_mode, is_constrained
from research_copilot.llm.structured import StructuredOutputError, generate_structured
from research_copilot.llm.tokens import prompt_budget
from research_copilot.models.paper_meta import PaperMetadata
from research_copilot.prompts import PromptManager

# Synthetic samples written for this exercise; they are not real papers.
SAMPLES = [
    """Title: Lightweight Attention for Real-Time Semantic Segmentation
Authors: Maria Lopez, Daniel Kim
Year: 2022
Abstract: We propose a lightweight attention block that replaces dense self-attention in
segmentation decoders. On Cityscapes and CamVid our network reaches 78.4% and 71.9% mIoU
while running at 62 frames per second on a single GPU. We show that the block can be added
to existing backbones without retraining from scratch. A limitation is reduced accuracy on
small objects.""",
    """Title: A Survey of Self-Supervised Pretraining for Medical Imaging
Authors: Priya Raman, Tomas Novak, Elena Petrova
Year: 2021
Abstract: This survey reviews self-supervised pretraining methods for medical image analysis,
covering contrastive, masked and generative approaches. We compare reported results on
CheXpert and ISIC 2019 and discuss open problems such as domain shift and evaluation
protocols. We do not propose a new method.""",
    """Title: Sample Complexity of Learning Under Label Noise
Authors: Jonas Weber
Year: 2019
Abstract: We prove an upper bound on the sample complexity of learning half-spaces when a
constant fraction of labels is flipped at random. The bound matches known lower bounds up to
logarithmic factors. Our analysis relies on a new concentration inequality. The paper is
purely theoretical and contains no experiments.""",
]

TEMPERATURE = 0.7
RESERVED_OUTPUT_TOKENS = 800


def run_mode(mode: Mode, runs: int) -> dict[str, float]:
    settings = get_settings()
    prompts = PromptManager(settings.prompts_dir)
    client = build_client_for_mode(settings, mode)
    budget = prompt_budget(settings.llm_context_window, RESERVED_OUTPUT_TOKENS)
    first_try = repaired = failed = 0
    seconds = 0.0

    for i in range(runs):
        text = SAMPLES[i % len(SAMPLES)]
        messages = build_messages(prompts, text, token_budget=budget)
        start = time.perf_counter()
        try:
            result = generate_structured(
                client,
                messages,
                PaperMetadata,
                max_repairs=2,
                constrained=is_constrained(mode),
                temperature=TEMPERATURE,
                max_tokens=RESERVED_OUTPUT_TOKENS,
            )
            if result.attempts == 1:
                first_try += 1
            else:
                repaired += 1
        except StructuredOutputError:
            failed += 1
        seconds += time.perf_counter() - start
        print(".", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    return {
        "first_try": first_try / runs,
        "with_repair": (first_try + repaired) / runs,
        "failed": failed / runs,
        "seconds": seconds / runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    print(f"model={get_settings().llm_model} runs={args.runs} temperature={TEMPERATURE}")
    print()
    print("| mode | parse_success_rate (first try) | success with repair | failed | avg seconds |")
    print("|---|---|---|---|---|")
    for mode in args.modes:
        try:
            stats = run_mode(mode, args.runs)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"| {mode} | {stats['first_try']:.0%} | {stats['with_repair']:.0%} "
            f"| {stats['failed']:.0%} | {stats['seconds']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

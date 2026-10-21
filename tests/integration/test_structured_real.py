import os

import httpx
import pytest

from research_copilot.config import Settings
from research_copilot.extract import build_messages
from research_copilot.llm.factory import build_client_for_mode
from research_copilot.llm.structured import generate_structured
from research_copilot.models.paper_meta import PaperMetadata
from research_copilot.prompts import PromptManager

BASE_URL = os.environ.get("RC_LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RC_LLM_MODEL", "llama3.1")

PAPER = """Title: Tiny Nets for Edge Devices
Authors: A. Author, B. Writer
Year: 2022
Abstract: We prune convolutional networks to run on microcontrollers. On CIFAR-10 our pruned
network keeps 90.1% accuracy with 8x fewer parameters. The method is purely empirical."""


def ollama_available() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/models", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.integration
def test_real_model_returns_valid_metadata_with_native_format():
    if not ollama_available():
        pytest.skip("Ollama is not reachable")
    settings = Settings(llm_model=MODEL, llm_base_url=BASE_URL, _env_file=None)
    client = build_client_for_mode(settings, "native_format")
    prompts = PromptManager(settings.prompts_dir)
    messages = build_messages(prompts, PAPER, token_budget=2900)

    result = generate_structured(client, messages, PaperMetadata, max_tokens=800)

    assert result.attempts <= 3
    assert result.value.year == 2022

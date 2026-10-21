from typing import Literal

from research_copilot.config import Settings
from research_copilot.llm.client import LLMClient, OpenAICompatClient
from research_copilot.llm.ollama_native import OllamaNativeClient

Mode = Literal["prompt_only", "response_format", "native_format"]

MODES: tuple[Mode, ...] = ("prompt_only", "response_format", "native_format")


def is_constrained(mode: Mode) -> bool:
    return mode != "prompt_only"


def build_client_for_mode(settings: Settings, mode: Mode) -> LLMClient:
    if mode == "native_format":
        native_url = settings.llm_base_url.removesuffix("/v1").rstrip("/") + "/api"
        return OllamaNativeClient(
            base_url=native_url,
            model=settings.llm_model,
            connect_timeout=settings.llm_connect_timeout,
            read_timeout=settings.llm_timeout,
        )
    return OpenAICompatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        connect_timeout=settings.llm_connect_timeout,
        read_timeout=settings.llm_timeout,
    )

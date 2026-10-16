import subprocess
import sys

import pytest
from pydantic import ValidationError

from research_copilot.config import Settings


def test_loads_values_from_env(monkeypatch):
    monkeypatch.setenv("RC_LLM_MODEL", "test-model")
    monkeypatch.setenv("RC_LLM_TIMEOUT", "12.5")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "test-model"
    assert settings.llm_timeout == 12.5
    assert settings.embed_model == "nomic-embed-text"


def test_missing_required_field_names_the_field(monkeypatch):
    monkeypatch.delenv("RC_LLM_MODEL", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "llm_model" in str(exc_info.value)


def test_invalid_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("RC_LLM_MODEL", "test-model")
    monkeypatch.setenv("RC_LLM_TIMEOUT", "abc")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "llm_timeout" in str(exc_info.value)


def test_secret_is_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("RC_LLM_MODEL", "test-model")
    monkeypatch.setenv("RC_LLM_API_KEY", "super-secret-value")

    settings = Settings(_env_file=None)

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)
    assert settings.llm_api_key.get_secret_value() == "super-secret-value"


def test_package_imports_from_outside_repo(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import research_copilot"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


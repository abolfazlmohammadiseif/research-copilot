from pathlib import Path

import pytest

from research_copilot.prompts import (
    PromptManager,
    PromptNotFoundError,
    PromptRenderError,
)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
CHUNKS = [{"id": "c1", "text": "Alpha is first."}, {"id": "c2", "text": "Beta is second."}]


def test_render_fills_variables_and_loops():
    prompt = PromptManager(PROMPTS_DIR).render("answer", question="What is first?", chunks=CHUNKS)

    assert '<chunk id="c1">' in prompt.text
    assert "Beta is second." in prompt.text
    assert '"""What is first?"""' in prompt.text
    assert "{#" not in prompt.text
    assert prompt.version == "1"


def test_missing_variable_is_an_error_naming_the_variable():
    with pytest.raises(PromptRenderError) as exc_info:
        PromptManager(PROMPTS_DIR).render("answer", chunks=CHUNKS)

    assert "question" in str(exc_info.value)


def test_unknown_prompt_and_unsafe_names_are_rejected():
    manager = PromptManager(PROMPTS_DIR)

    with pytest.raises(PromptNotFoundError):
        manager.render("does_not_exist")
    with pytest.raises(PromptNotFoundError):
        manager.render("../secrets")


def test_digest_is_stable_and_changes_with_the_file(tmp_path: Path):
    (tmp_path / "greet.md").write_text("{# version: 3 #}\nHello {{ name }}", encoding="utf-8")
    manager = PromptManager(tmp_path)

    first = manager.render("greet", name="Ada")
    second = manager.render("greet", name="Grace")
    assert first.digest == second.digest
    assert first.version == "3"
    assert first.prompt_id == f"greet@3+{first.digest}"

    (tmp_path / "greet.md").write_text("{# version: 3 #}\nHi {{ name }}", encoding="utf-8")
    third = manager.render("greet", name="Ada")
    assert third.digest != first.digest

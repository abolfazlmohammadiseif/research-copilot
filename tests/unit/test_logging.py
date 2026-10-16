import json
import logging

from research_copilot.config import Settings
from research_copilot.logging import JsonFormatter, run_id_var


def test_log_line_is_json_with_run_id(caplog):
    caplog.handler.setFormatter(JsonFormatter())
    token = run_id_var.set("abc12345")
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger("demo").info("hello %s", "world")
    finally:
        run_id_var.reset(token)

    data = json.loads(caplog.text.strip())

    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "demo"
    assert data["run_id"] == "abc12345"


def test_secret_is_not_logged(caplog, monkeypatch):
    monkeypatch.setenv("RC_LLM_MODEL", "test-model")
    monkeypatch.setenv("RC_LLM_API_KEY", "top-secret-value")
    settings = Settings(_env_file=None)

    with caplog.at_level(logging.INFO):
        logging.getLogger("demo").info("loaded settings: %s", settings)

    assert "top-secret-value" not in caplog.text

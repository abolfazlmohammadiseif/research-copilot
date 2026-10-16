__version__ = "0.1.0"


def main() -> None:
    import logging
    import uuid

    from research_copilot.config import get_settings
    from research_copilot.logging import run_id_var, setup_logging

    setup_logging()
    run_id_var.set(uuid.uuid4().hex[:8])
    logger = logging.getLogger(__name__)

    settings = get_settings()
    logger.info("started version=%s model=%s", __version__, settings.llm_model)
    logger.info("llm_base_url=%s data_dir=%s", settings.llm_base_url, settings.data_dir)

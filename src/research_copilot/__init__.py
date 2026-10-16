__version__ = "0.1.0"


def main() -> None:
    from research_copilot.config import get_settings

    settings = get_settings()
    print(f"research-copilot {__version__}")
    print(f"llm_model={settings.llm_model}")
    print(f"llm_base_url={settings.llm_base_url}")
    print(f"data_dir={settings.data_dir}")


import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined, UndefinedError

NAME_PATTERN = re.compile(r"[a-z0-9_]+")
VERSION_PATTERN = re.compile(r"\{#\s*version:\s*(\S+)\s*#\}")


class PromptError(Exception):
    pass


class PromptNotFoundError(PromptError):
    pass


class PromptRenderError(PromptError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    name: str
    text: str
    version: str
    digest: str

    @property
    def prompt_id(self) -> str:
        return f"{self.name}@{self.version}+{self.digest}"


class PromptManager:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, name: str, /, **variables: object) -> RenderedPrompt:
        if not NAME_PATTERN.fullmatch(name):
            raise PromptNotFoundError(f"invalid prompt name: {name!r}")
        path = self._dir / f"{name}.md"
        try:
            source = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptNotFoundError(f"prompt not found: {path}") from exc
        try:
            text = self._env.from_string(source).render(**variables)
        except UndefinedError as exc:
            raise PromptRenderError(f"cannot render {name!r}: {exc}") from exc
        match = VERSION_PATTERN.match(source)
        return RenderedPrompt(
            name=name,
            text=text.strip(),
            version=match.group(1) if match else "0",
            digest=hashlib.sha256(source.encode("utf-8")).hexdigest()[:12],
        )

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaperMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, description="Full title of the paper")
    authors: list[str] = Field(min_length=1, description="Author names, one string per author")
    year: int = Field(ge=1900, le=2100, description="Publication year")
    abstract: str = Field(min_length=1, description="The abstract, copied or lightly condensed")
    claims: list[str] = Field(description="Main claims, one short sentence each")
    method: str = Field(description="One or two sentences describing the proposed method")
    datasets: list[str] = Field(description="Names of datasets used; empty list if none")
    kind: Literal["empirical", "theoretical", "survey", "other"]

    @field_validator("authors")
    @classmethod
    def authors_must_not_be_blank(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if any(not name for name in cleaned):
            raise ValueError("author names must not be blank")
        return cleaned

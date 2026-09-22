from typing import Any, TypedDict
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    updated: str | None = None
    summary: str
    categories: list[str] = Field(default_factory=list)
    pdf_url: str
    abs_url: str


class PaperBriefing(BaseModel):
    title: str
    authors: list[str]
    arxiv_id: str
    publish_date: str
    link: str
    why_it_matters: str
    problem_statement: str
    method: list[str]
    key_results: list[str]
    limitations: list[str]
    follow_up_questions: list[str]


class ResearchState(TypedDict, total=False):
    user_input: str
    input_type: str
    query: str | None
    arxiv_id: str | None
    candidates: list[dict[str, Any]]
    selected_paper: dict[str, Any] | None
    pdf_path: str | None
    indexed: bool
    collection_name: str | None
    parsed_documents: list[dict[str, Any]]
    briefing: dict[str, Any] | None
    question: str | None
    retrieved_docs: list[dict[str, Any]]
    answer: str | None
    sources: list[dict[str, Any]]
    session_id: str
    error: str | None

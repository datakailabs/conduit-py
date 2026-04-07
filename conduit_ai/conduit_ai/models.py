"""Data models for Conduit API responses."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A knowledge source returned from retrieval."""

    id: str
    title: str
    content: str
    score: float
    path: str | None = None
    domains: list[str] = Field(default_factory=list)
    source_url: str | None = Field(default=None, alias="sourceUrl")
    provenance: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class RetrievalStats(BaseModel):
    """Statistics from the retrieval phase."""

    vector_results: int = Field(default=0, alias="vectorResults")
    graph_results: int = Field(default=0, alias="graphResults")
    final_results: int = Field(default=0, alias="finalResults")

    model_config = {"populate_by_name": True, "extra": "allow"}


class AskResponse(BaseModel):
    """Response from the /ask endpoint."""

    query: str
    answer: str
    mode: str = "standard"
    sources: list[Source] = Field(default_factory=list)
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)
    rewritten_query: str | None = Field(default=None, alias="rewrittenQuery")
    swarm: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class ContextResult(BaseModel):
    """A single result from the /context endpoint (JSON format)."""

    id: str = Field(alias="zettelId")
    title: str
    content: str
    score: float
    path: str | None = None
    domains: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    knowledge_type: str = Field(default="concept", alias="knowledgeType")
    source_url: str | None = Field(default=None, alias="sourceUrl")
    provenance: dict[str, Any] | None = None
    neighbors: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True, "extra": "allow"}


class ContextResponse(BaseModel):
    """Response from the /context endpoint."""

    query: str
    format: str = "json"
    context: str | None = None  # Present when format=markdown
    results: list[ContextResult] | None = None  # Present when format=json
    result_count: int = Field(default=0, alias="resultCount")
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)

    model_config = {"populate_by_name": True}

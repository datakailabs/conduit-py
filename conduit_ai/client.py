"""Conduit API client — the core of conduit-ai."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from conduit_ai.models import AskResponse, ContextResponse


class ConduitClient:
    """Client for the Conduit knowledge graph API.

    Usage::

        from conduit_ai import ConduitClient

        client = ConduitClient(api_key="ck_...")

        # Ask a question (GraphRAG + LLM synthesis)
        answer = client.ask("How does Snowflake Cortex Search work?")
        print(answer.answer)

        # Retrieve context (GraphRAG only, no LLM)
        ctx = client.context("Delta Live Tables patterns")
        for result in ctx.results:
            print(result.title, result.score)

        # Async
        answer = await client.aask("How does Cortex Search work?")
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.conduit.datakai.com",
        *,
        kai_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._kai_id = kai_id
        self._timeout = timeout

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if kai_id:
            headers["X-Kai-Id"] = kai_id

        self._sync = httpx.Client(
            base_url=self._endpoint,
            headers=headers,
            timeout=timeout,
        )
        self._async = httpx.AsyncClient(
            base_url=self._endpoint,
            headers=headers,
            timeout=timeout,
        )

    # ─── Ask (GraphRAG + LLM synthesis) ──────────────────────────────

    def ask(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        limit: int = 8,
        mode: str = "standard",
    ) -> AskResponse:
        """Ask a question and get a synthesized answer with sources."""
        payload = self._ask_payload(query, thread_id=thread_id, limit=limit, mode=mode)
        resp = self._sync.post("/api/v1/ask", json=payload)
        resp.raise_for_status()
        return AskResponse.model_validate(resp.json())

    async def aask(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        limit: int = 8,
        mode: str = "standard",
    ) -> AskResponse:
        """Async version of ask()."""
        payload = self._ask_payload(query, thread_id=thread_id, limit=limit, mode=mode)
        resp = await self._async.post("/api/v1/ask", json=payload)
        resp.raise_for_status()
        return AskResponse.model_validate(resp.json())

    async def aask_stream(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        limit: int = 8,
    ) -> AsyncIterator[str]:
        """Stream answer tokens via SSE. Yields token strings."""
        payload = self._ask_payload(query, thread_id=thread_id, limit=limit, mode="standard")
        payload["stream"] = True

        async with self._async.stream("POST", "/api/v1/ask", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    import json
                    data = json.loads(line[6:])
                    if "token" in data:
                        yield data["token"]

    # ─── Context (GraphRAG retrieval only) ───────────────────────────

    def context(
        self,
        query: str,
        *,
        limit: int = 10,
        format: str = "json",
    ) -> ContextResponse:
        """Retrieve knowledge context without LLM synthesis."""
        payload = {"query": query, "limit": limit, "format": format}
        resp = self._sync.post("/api/v1/context", json=payload)
        resp.raise_for_status()
        return ContextResponse.model_validate(resp.json())

    async def acontext(
        self,
        query: str,
        *,
        limit: int = 10,
        format: str = "json",
    ) -> ContextResponse:
        """Async version of context()."""
        payload = {"query": query, "limit": limit, "format": format}
        resp = await self._async.post("/api/v1/context", json=payload)
        resp.raise_for_status()
        return ContextResponse.model_validate(resp.json())

    # ─── Internals ───────────────────────────────────────────────────

    def _ask_payload(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        limit: int = 8,
        mode: str = "standard",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "limit": limit, "mode": mode}
        if thread_id:
            payload["thread_id"] = thread_id
        return payload

    def close(self) -> None:
        """Close underlying HTTP connections."""
        self._sync.close()

    async def aclose(self) -> None:
        """Close underlying async HTTP connections."""
        await self._async.aclose()

    def __enter__(self) -> ConduitClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> ConduitClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

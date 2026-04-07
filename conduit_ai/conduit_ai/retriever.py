"""LangChain retriever for Conduit knowledge graph."""

from __future__ import annotations

from typing import Any

from conduit_ai.client import ConduitClient

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


def _check_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "LangChain is required for ConduitRetriever. "
            "Install it with: pip install 'conduit-ai[langchain]'"
        )


if _HAS_LANGCHAIN:

    class ConduitRetriever(BaseRetriever):
        """LangChain retriever backed by Conduit's GraphRAG engine.

        Drops into any LangChain/LangGraph chain as the retrieval step::

            from conduit_ai.retriever import ConduitRetriever

            retriever = ConduitRetriever(
                api_key="ck_...",
                endpoint="https://conduit.mycompany.com",
            )

            # Use in a chain
            docs = retriever.invoke("How does Snowflake Cortex Search work?")

            # Or in a LangGraph workflow
            from langchain_core.runnables import RunnablePassthrough
            chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
        """

        api_key: str | None = None
        endpoint: str = "https://api.conduit.datakai.com"
        kai_id: str | None = None
        limit: int = 8
        include_content: bool = True
        include_graph_context: bool = False

        _client: ConduitClient | None = None

        model_config = {"arbitrary_types_allowed": True}

        @property
        def client(self) -> ConduitClient:
            if self._client is None:
                self._client = ConduitClient(
                    api_key=self.api_key,
                    endpoint=self.endpoint,
                    kai_id=self.kai_id,
                )
            return self._client

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> list[Document]:
            """Retrieve documents from Conduit's knowledge graph.

            Each document contains a knowledge unit (zettel) with metadata
            including domains, topics, knowledge type, and graph relationships.
            """
            if self.include_graph_context:
                resp = self.client.context(query, limit=self.limit, format="json")
                if not resp.results:
                    return []
                return [
                    Document(
                        page_content=r.content,
                        metadata={
                            "id": r.id,
                            "title": r.title,
                            "score": r.score,
                            "domains": r.domains,
                            "topics": r.topics,
                            "knowledge_type": r.knowledge_type,
                            "source_url": r.source_url,
                            "neighbors": r.neighbors,
                            "source": "conduit",
                        },
                    )
                    for r in resp.results
                ]
            else:
                resp = self.client.ask(query, limit=self.limit)
                return [
                    Document(
                        page_content=s.content if self.include_content else s.title,
                        metadata={
                            "id": s.id,
                            "title": s.title,
                            "score": s.score,
                            "domains": s.domains,
                            "source_url": s.source_url,
                            "source": "conduit",
                        },
                    )
                    for s in resp.sources
                ]

else:
    # Stub so imports don't break without LangChain
    class ConduitRetriever:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            _check_langchain()

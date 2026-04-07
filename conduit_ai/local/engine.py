"""LocalConduit — embedded knowledge graph engine.

No server required. DuckDB for storage + vector search,
in-memory graph for relationship traversal.

Usage::

    from conduit_ai.local import LocalConduit

    conduit = LocalConduit("./my-knowledge-base")
    conduit.install_pack("snowflake-2026.04.ckp")

    results = conduit.search("How does Cortex Search work?")
    retriever = conduit.as_retriever()
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

from conduit_ai.local.graph import InMemoryGraph
from conduit_ai.local.store import DuckStore


class LocalConduit:
    """Embedded knowledge graph engine.

    Stores zettels and embeddings in DuckDB, loads relationships
    into an in-memory graph for fast traversal. Supports knowledge
    pack installation with optional topic filtering.
    """

    def __init__(self, path: str = "./conduit.duckdb") -> None:
        self._path = path
        self._store = DuckStore(path)
        self._graph = InMemoryGraph()
        self._embedder: Any = None
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        """Load relationships from DuckDB into in-memory graph."""
        rels = self._store.get_all_relationships()
        self._graph.load(rels)

    def _get_embedder(self) -> Any:
        """Lazy-load embedding function. Uses OpenAI if key available, else errors."""
        if self._embedder is not None:
            return self._embedder

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY required for embedding generation. "
                "Set it in your environment or install packs with pre-computed embeddings."
            )

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            def embed(texts: list[str]) -> list[list[float]]:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                return [e.embedding for e in resp.data]

            self._embedder = embed
            return embed
        except ImportError:
            raise RuntimeError("pip install openai required for embedding generation")

    # ─── Pack Management ─────────────────────────────────────────

    def install_pack(
        self,
        pack_path: str,
        *,
        topics: list[str] | None = None,
    ) -> dict:
        """Install a .ckp knowledge pack.

        Args:
            pack_path: Path to .ckp file
            topics: Optional topic filter — only install zettels matching these topics

        Returns:
            Install summary with counts
        """
        path = Path(pack_path)
        if not path.exists():
            raise FileNotFoundError(f"Pack not found: {path}")

        topic_filter = set(topics) if topics else None

        with tarfile.open(path, "r:gz") as tar:
            # Read manifest
            manifest_f = tar.extractfile("pack.toml")
            if not manifest_f:
                raise ValueError("Invalid pack: missing pack.toml")
            manifest = manifest_f.read().decode("utf-8")

            # Parse pack ID and version from manifest
            pack_id = _toml_get(manifest, "id")
            version = _toml_get(manifest, "version")
            name = _toml_get(manifest, "name")

            if not pack_id:
                raise ValueError("Invalid pack: missing id in pack.toml")

            # Check if already installed
            existing = self._store.get_pack(pack_id)
            if existing:
                # Remove old version first
                self._store.delete_pack(pack_id)

            # Load zettels
            zettels_f = tar.extractfile("zettels.jsonl")
            if not zettels_f:
                raise ValueError("Invalid pack: missing zettels.jsonl")

            zettels = []
            for line in zettels_f:
                z = json.loads(line)
                if topic_filter:
                    zettel_topics = set(z.get("topics", []))
                    if not zettel_topics & topic_filter:
                        continue
                zettels.append(z)

            # Load relationships
            rels_f = tar.extractfile("relationships.jsonl")
            rels = []
            if rels_f:
                zettel_ids = {z["id"] for z in zettels}
                for line in rels_f:
                    r = json.loads(line)
                    # Only include relationships where at least one end is in our filtered set
                    if r["source"] in zettel_ids or r["target"] in zettel_ids:
                        rels.append(r)

        # Insert into DuckDB
        for z in zettels:
            self._store.insert_zettel(z, pack_id=pack_id)

        for r in rels:
            self._store.insert_relationship(r)

        # Generate embeddings for zettels that don't have them
        self._generate_embeddings(zettels)

        # Build vector index
        self._store.ensure_vector_index()

        # Register pack
        self._store.register_pack(pack_id, version, name or pack_id, len(zettels), len(rels))

        # Rebuild in-memory graph
        self._rebuild_graph()

        return {
            "pack": pack_id,
            "version": version,
            "zettels_installed": len(zettels),
            "relationships_installed": len(rels),
            "topic_filter": list(topic_filter) if topic_filter else None,
        }

    def _generate_embeddings(self, zettels: list[dict]) -> None:
        """Generate embeddings for zettels in batches."""
        embed = self._get_embedder()
        batch_size = 100
        needs_embedding = [z for z in zettels if z.get("id")]

        for i in range(0, len(needs_embedding), batch_size):
            batch = needs_embedding[i : i + batch_size]
            texts = [
                f"{z['title']}\n\n{z['content'][:1000]}" for z in batch
            ]
            embeddings = embed(texts)
            for z, emb in zip(batch, embeddings):
                self._store.set_embedding(z["id"], emb)

    def uninstall_pack(self, pack_id: str) -> dict:
        """Remove an installed knowledge pack."""
        count = self._store.delete_pack(pack_id)
        self._store.ensure_vector_index()
        self._rebuild_graph()
        return {"pack": pack_id, "zettels_removed": count}

    def list_packs(self) -> list[dict]:
        """List installed knowledge packs."""
        return self._store.list_packs()

    # ─── Search & Retrieval ──────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        graph_hops: int = 1,
    ) -> list[dict]:
        """Graph-augmented search.

        1. Embed query
        2. Vector search for top candidates
        3. Walk graph to find related zettels
        4. Merge and rerank by combined score

        Returns list of zettels with scores and graph neighbor info.
        """
        embed = self._get_embedder()
        query_embedding = embed([query])[0]

        # Vector search — get more than limit to leave room for graph results
        vector_results = self._store.vector_search(query_embedding, limit=limit * 2)

        if not vector_results:
            return []

        # Graph augmentation — walk 1-2 hops from top vector results
        seen_ids = {r["id"] for r in vector_results}
        graph_additions = []

        for result in vector_results[:limit]:
            neighbors = self._graph.neighbors(result["id"], hops=graph_hops)
            for neighbor in neighbors:
                if neighbor["id"] not in seen_ids:
                    zettel = self._store.get_zettel(neighbor["id"])
                    if zettel:
                        # Score graph neighbors: base score from vector similarity to query
                        # discounted by hop distance
                        zettel["score"] = result["score"] * (0.7 ** neighbor["distance"])
                        zettel["path"] = "graph"
                        zettel["graph_relation"] = {
                            "from": result["id"],
                            "type": neighbor["type"],
                            "distance": neighbor["distance"],
                        }
                        graph_additions.append(zettel)
                        seen_ids.add(neighbor["id"])

        # Tag vector results
        for r in vector_results:
            r["path"] = "vector"

        # Merge and sort by score
        all_results = vector_results + graph_additions
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results[:limit]

    def context(self, query: str, *, limit: int = 5) -> str:
        """Get formatted knowledge context (markdown).

        Useful for injecting into LLM prompts directly.
        """
        results = self.search(query, limit=limit)
        if not results:
            return f'No relevant knowledge found for: "{query}"'

        parts = [f"# Knowledge Context: {query}\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"## [{i}] {r['title']} (score: {r['score']:.3f})")
            parts.append(f"Domains: {', '.join(r.get('domains', []))}")
            parts.append(f"Topics: {', '.join(r.get('topics', []))}")
            if r.get("graph_relation"):
                gr = r["graph_relation"]
                parts.append(f"Found via: {gr['type']} relationship from [{gr['from'][:30]}...] (hop {gr['distance']})")
            parts.append(f"\n{r['content']}\n")

        return "\n".join(parts)

    # ─── LangChain Integration ───────────────────────────────────

    def as_retriever(self, *, limit: int = 8, graph_hops: int = 1):
        """Create a LangChain retriever backed by this local engine.

        Requires: pip install 'conduit-ai[langchain]'
        """
        try:
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
            from langchain_core.documents import Document
            from langchain_core.retrievers import BaseRetriever
        except ImportError:
            raise ImportError("pip install 'conduit-ai[langchain]' for LangChain integration")

        engine = self

        class _LocalRetriever(BaseRetriever):
            _engine: Any = None
            _limit: int = limit
            _graph_hops: int = graph_hops

            model_config = {"arbitrary_types_allowed": True}

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun | None = None,
            ) -> list[Document]:
                results = engine.search(query, limit=self._limit, graph_hops=self._graph_hops)
                return [
                    Document(
                        page_content=r["content"],
                        metadata={
                            "id": r["id"],
                            "title": r["title"],
                            "score": r.get("score", 0),
                            "domains": r.get("domains", []),
                            "topics": r.get("topics", []),
                            "knowledge_type": r.get("knowledge_type"),
                            "path": r.get("path", "vector"),
                            "graph_relation": r.get("graph_relation"),
                            "source": "conduit-local",
                        },
                    )
                    for r in results
                ]

        return _LocalRetriever()

    # ─── Info ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get engine statistics."""
        store_stats = self._store.stats()
        store_stats["graph_nodes"] = self._graph.node_count
        store_stats["graph_edges"] = self._graph.edge_count
        return store_stats

    def close(self) -> None:
        """Close the database connection."""
        self._store.close()

    def __enter__(self) -> LocalConduit:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _toml_get(text: str, key: str) -> str:
    """Minimal TOML value extractor."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
            val = stripped.split("=", 1)[1].strip()
            return val.strip('"').strip("'")
    return ""

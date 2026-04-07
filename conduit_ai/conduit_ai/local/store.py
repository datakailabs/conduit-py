"""DuckDB storage backend for Conduit Embedded."""

from __future__ import annotations

from typing import Any

import duckdb


class DuckStore:
    """Manages zettel storage and vector search via DuckDB."""

    def __init__(self, path: str) -> None:
        self.conn = duckdb.connect(path)
        self.conn.install_extension("vss")
        self.conn.load_extension("vss")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS zettels (
                id VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                content VARCHAR NOT NULL,
                summary VARCHAR,
                domains VARCHAR[] NOT NULL DEFAULT [],
                topics VARCHAR[] NOT NULL DEFAULT [],
                knowledge_type VARCHAR NOT NULL DEFAULT 'concept',
                context_source VARCHAR NOT NULL DEFAULT 'vendor-doc',
                source_url VARCHAR,
                provenance VARCHAR,
                pack_id VARCHAR,
                created VARCHAR,
                updated VARCHAR,
                embedding FLOAT[1536]
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                source VARCHAR NOT NULL,
                target VARCHAR NOT NULL,
                type VARCHAR NOT NULL,
                properties VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS packs (
                id VARCHAR PRIMARY KEY,
                version VARCHAR NOT NULL,
                name VARCHAR,
                zettels_count INTEGER DEFAULT 0,
                relationships_count INTEGER DEFAULT 0,
                installed_at VARCHAR
            )
        """)

    def ensure_vector_index(self) -> None:
        """Create HNSW index if enough vectors exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM zettels WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        if count > 0:
            try:
                self.conn.execute("DROP INDEX IF EXISTS idx_zettel_embedding")
                self.conn.execute(
                    "CREATE INDEX idx_zettel_embedding ON zettels USING HNSW (embedding) WITH (metric = 'cosine')"
                )
            except Exception:
                pass  # Index may already exist or not enough rows

    def insert_zettel(self, zettel: dict[str, Any], pack_id: str | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO zettels
               (id, title, content, summary, domains, topics, knowledge_type,
                context_source, source_url, provenance, pack_id, created, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                zettel["id"],
                zettel["title"],
                zettel["content"],
                zettel.get("summary"),
                zettel.get("domains", []),
                zettel.get("topics", []),
                zettel.get("knowledge_type", "concept"),
                zettel.get("context_source", "vendor-doc"),
                zettel.get("source_url"),
                str(zettel.get("provenance")) if zettel.get("provenance") else None,
                pack_id,
                zettel.get("created"),
                zettel.get("updated"),
            ],
        )

    def insert_relationship(self, rel: dict[str, Any]) -> None:
        import json
        props = json.dumps(rel.get("properties")) if rel.get("properties") else None
        self.conn.execute(
            "INSERT INTO relationships (source, target, type, properties) VALUES (?, ?, ?, ?)",
            [rel["source"], rel["target"], rel["type"], props],
        )

    def set_embedding(self, zettel_id: str, embedding: list[float]) -> None:
        self.conn.execute(
            "UPDATE zettels SET embedding = ? WHERE id = ?",
            [embedding, zettel_id],
        )

    def vector_search(self, query_embedding: list[float], limit: int = 8) -> list[dict]:
        """Cosine similarity search over zettel embeddings."""
        results = self.conn.execute(
            """SELECT id, title, content, summary, domains, topics,
                      knowledge_type, context_source, source_url,
                      array_cosine_similarity(embedding, ?::FLOAT[1536]) as score
               FROM zettels
               WHERE embedding IS NOT NULL
               ORDER BY score DESC
               LIMIT ?""",
            [query_embedding, limit],
        ).fetchall()

        columns = [
            "id", "title", "content", "summary", "domains", "topics",
            "knowledge_type", "context_source", "source_url", "score",
        ]
        return [dict(zip(columns, row)) for row in results]

    def get_all_relationships(self) -> list[dict]:
        """Load all relationships for in-memory graph building."""
        import json
        rows = self.conn.execute(
            "SELECT source, target, type, properties FROM relationships"
        ).fetchall()
        return [
            {
                "source": r[0],
                "target": r[1],
                "type": r[2],
                "properties": json.loads(r[3]) if r[3] else {},
            }
            for r in rows
        ]

    def get_zettel(self, zettel_id: str) -> dict | None:
        row = self.conn.execute(
            """SELECT id, title, content, summary, domains, topics,
                      knowledge_type, context_source, source_url
               FROM zettels WHERE id = ?""",
            [zettel_id],
        ).fetchone()
        if not row:
            return None
        columns = [
            "id", "title", "content", "summary", "domains", "topics",
            "knowledge_type", "context_source", "source_url",
        ]
        return dict(zip(columns, row))

    def get_pack(self, pack_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM packs WHERE id = ?", [pack_id]).fetchone()
        if not row:
            return None
        return {"id": row[0], "version": row[1], "name": row[2], "zettels": row[3], "relationships": row[4]}

    def register_pack(self, pack_id: str, version: str, name: str, zettels: int, rels: int) -> None:
        from datetime import datetime, timezone
        self.conn.execute(
            "INSERT OR REPLACE INTO packs VALUES (?, ?, ?, ?, ?, ?)",
            [pack_id, version, name, zettels, rels, datetime.now(timezone.utc).isoformat()],
        )

    def delete_pack(self, pack_id: str) -> int:
        count = self.conn.execute(
            "SELECT COUNT(*) FROM zettels WHERE pack_id = ?", [pack_id]
        ).fetchone()[0]
        self.conn.execute("DELETE FROM zettels WHERE pack_id = ?", [pack_id])
        # Remove relationships where either end belonged to this pack
        self.conn.execute("""
            DELETE FROM relationships WHERE source IN (SELECT id FROM zettels WHERE pack_id = ?)
            OR target IN (SELECT id FROM zettels WHERE pack_id = ?)
        """, [pack_id, pack_id])
        self.conn.execute("DELETE FROM packs WHERE id = ?", [pack_id])
        return count

    def list_packs(self) -> list[dict]:
        rows = self.conn.execute("SELECT id, version, name, zettels_count, relationships_count FROM packs").fetchall()
        return [{"id": r[0], "version": r[1], "name": r[2], "zettels": r[3], "relationships": r[4]} for r in rows]

    def stats(self) -> dict:
        zettels = self.conn.execute("SELECT COUNT(*) FROM zettels").fetchone()[0]
        with_embeddings = self.conn.execute("SELECT COUNT(*) FROM zettels WHERE embedding IS NOT NULL").fetchone()[0]
        rels = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        packs = self.conn.execute("SELECT COUNT(*) FROM packs").fetchone()[0]
        return {"zettels": zettels, "with_embeddings": with_embeddings, "relationships": rels, "packs": packs}

    def close(self) -> None:
        self.conn.close()

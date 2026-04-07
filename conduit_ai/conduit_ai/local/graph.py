"""In-memory graph for relationship traversal."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class InMemoryGraph:
    """Lightweight adjacency list for graph-augmented retrieval.

    Loads relationships from DuckDB into memory for fast traversal.
    At 15K zettels and 15K edges, this uses ~10-20MB of memory.
    """

    def __init__(self) -> None:
        # adjacency: node_id → list of (target_id, edge_type, properties)
        self._outgoing: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self._incoming: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self._edge_count = 0

    def load(self, relationships: list[dict[str, Any]]) -> None:
        """Load relationships from store into memory."""
        self._outgoing.clear()
        self._incoming.clear()
        self._edge_count = 0

        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            edge_type = rel["type"]
            props = rel.get("properties", {})

            self._outgoing[source].append((target, edge_type, props))
            self._incoming[target].append((source, edge_type, props))
            self._edge_count += 1

    def neighbors(self, node_id: str, hops: int = 1, direction: str = "both") -> list[dict]:
        """Get neighbors within N hops.

        Returns list of {id, type, properties, distance} dicts.
        Deduplicates by node_id, keeping shortest distance.
        """
        visited: dict[str, dict] = {}
        frontier = {node_id}

        for distance in range(1, hops + 1):
            next_frontier: set[str] = set()
            for current in frontier:
                edges: list[tuple[str, str, dict]] = []
                if direction in ("both", "outgoing"):
                    edges.extend(self._outgoing.get(current, []))
                if direction in ("both", "incoming"):
                    edges.extend(self._incoming.get(current, []))

                for target, edge_type, props in edges:
                    if target != node_id and target not in visited:
                        visited[target] = {
                            "id": target,
                            "type": edge_type,
                            "properties": props,
                            "distance": distance,
                        }
                        next_frontier.add(target)

            frontier = next_frontier
            if not frontier:
                break

        return list(visited.values())

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def node_count(self) -> int:
        return len(set(self._outgoing.keys()) | set(self._incoming.keys()))

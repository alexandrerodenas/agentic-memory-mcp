"""Auto-pruning system for memory maintenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from agentic_memory.core.graph import KnowledgeGraph


class AutoPruner:
    """Configurable auto-pruner for the knowledge graph."""

    def __init__(
        self,
        max_nodes: int | None = None,
        max_size_mb: float | None = None,
        on_prune: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_size_mb = max_size_mb
        self.on_prune = on_prune

    def prune(
        self,
        graph: KnowledgeGraph,
        strategy: str = "oldest",
        labels: list[str] | None = None,
    ) -> list[str]:
        """Remove nodes according to strategy, return list of removed node IDs."""
        stats = graph.stats()
        if self.max_nodes is not None and stats["nodes"] <= self.max_nodes:
            return []

        candidates = graph.list_nodes()
        if labels:
            candidates = [n for n in candidates if n.label in labels]

        if strategy == "oldest":
            candidates.sort(key=lambda n: n.created_at)
        elif strategy == "low_score":
            # Placeholder — replaced by Score integration in memory.py
            candidates.sort(key=lambda n: n.updated_at)
        else:
            candidates.sort(key=lambda n: n.created_at)

        remove_count = (
            (stats["nodes"] - self.max_nodes) if self.max_nodes is not None else len(candidates)
        )
        to_remove = candidates[:remove_count]
        removed_ids = []

        for node in to_remove:
            try:
                graph.delete_node(node.id)
                removed_ids.append(node.id)
            except KeyError:
                pass

        if self.on_prune and removed_ids:
            self.on_prune(removed_ids)

        return removed_ids

    def check_size(self, graph: KnowledgeGraph) -> list[str]:
        """Prune by file size if max_size_mb is set."""
        if self.max_size_mb is None:
            return []
        path = graph.path
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        if size_mb > self.max_size_mb:
            return self.prune(graph, strategy="oldest")
        return []

    def auto_prune(self, graph: KnowledgeGraph) -> list[str]:
        """Run all configured pruning checks."""
        removed = self.check_size(graph)
        if self.max_nodes is not None:
            stats = graph.stats()
            if stats["nodes"] > self.max_nodes:
                extra = self.prune(graph, strategy="oldest")
                removed.extend(extra)
        return removed
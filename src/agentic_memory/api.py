"""Standalone Python skills — use memory without running the MCP server."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from agentic_memory.core.graph import KnowledgeGraph, Node, Edge
from agentic_memory.core.score import ScoreStore
from agentic_memory.core.prune import AutoPruner


class MemorySkills:
    """Decoupled memory skills for use in any agentic workflow."""

    def __init__(
        self,
        graph_path: str | Path = "memory_graph.json",
        scores_path: str | Path = "memory_scores.json",
    ) -> None:
        self.graph_path = Path(graph_path)
        self.scores_path = Path(scores_path)
        self._graph: KnowledgeGraph | None = None
        self._scores: ScoreStore | None = None

    @property
    def graph(self) -> KnowledgeGraph:
        if self._graph is None:
            self._graph = KnowledgeGraph(path=self.graph_path)
        return self._graph

    @property
    def scores(self) -> ScoreStore:
        if self._scores is None:
            self._scores = ScoreStore()
            if self.scores_path.exists():
                self._scores.load_dict(json.loads(self.scores_path.read_text()))
        return self._scores

    def _persist_scores(self) -> None:
        self.scores_path.write_text(json.dumps(self.scores.to_dict()))

    # ── Node operations ────────────────────────────────────────────────────────

    def add(self, node_id: str, content: str, label: str = "", metadata: dict | None = None) -> Node:
        node = Node(id=node_id, label=label, content=content, metadata=metadata or {})
        self.graph.add_node(node)
        return node

    def update(self, node_id: str, **kwargs) -> None:
        self.graph.update_node(node_id, **kwargs)

    def delete(self, node_id: str) -> None:
        self.graph.delete_node(node_id)
        self.scores.delete(node_id)
        self._persist_scores()

    def get(self, node_id: str) -> Node | None:
        node = self.graph.get_node(node_id)
        if node:
            self.scores.get(node_id).bump_read()
            self._persist_scores()
        return node

    def search(self, query: str = "", label: str = "", limit: int = 50) -> list[Node]:
        results = self.graph.search_nodes(query=query, label=label, limit=limit)
        for n in results:
            self.scores.get(n.id).bump_read()
        self._persist_scores()
        return results

    # ── Retrieval (token-optimized) ────────────────────────────────────────────

    def retrieve(self, limit: int = 10, label: str = "") -> list[tuple[Node, float]]:
        """Return top-N nodes with scores, best for LLM context injection."""
        top = self.scores.top(self.graph, label=label, limit=limit)
        return [(self.graph.get_node(nid), score) for nid, score in top if self.graph.get_node(nid)]

    def retrieve_text(self, limit: int = 10, label: str = "") -> str:
        """Return retrieval results as a plain-text string."""
        entries = self.retrieve(limit=limit, label=label)
        if not entries:
            return "No relevant memories found."
        lines = [f"[{n.label}][score={s:.2f}] {n.id}: {n.content}" for n, s in entries]
        return "\n".join(lines)

    # ── Graph operations ─────────────────────────────────────────────────────

    def add_edge(self, edge_id: str, source: str, target: str, label: str = "", weight: float = 1.0) -> Edge:
        edge = Edge(id=edge_id, source=source, target=target, label=label, weight=weight)
        self.graph.add_edge(edge)
        return edge

    def neighbors(self, node_id: str, label: str = "") -> list[Node]:
        return self.graph.get_neighbors(node_id, label=label)

    # ── Reputation ───────────────────────────────────────────────────────────

    def corroborate(self, node_id: str, count: int = 1) -> None:
        if self.graph.get_node(node_id) is None:
            raise KeyError(f"Node '{node_id}' not found")
        self.scores.get(node_id).bump_corroboration(count)
        self._persist_scores()

    # ── Maintenance ───────────────────────────────────────────────────────────

    def prune(self, max_nodes: int | None = None, strategy: str = "oldest") -> list[str]:
        pruner = AutoPruner(max_nodes=max_nodes)
        removed = pruner.auto_prune(self.graph)
        for node_id in removed:
            self.scores.delete(node_id)
        self._persist_scores()
        return removed

    def stats(self) -> dict:
        s = self.graph.stats()
        s["score_entries"] = len(self.scores.to_dict())
        return s

    def export_json(self) -> str:
        return self.graph.export_json()

    def import_json(self, data: str) -> None:
        self.graph.import_json(data)


# ── Quick-access convenience functions ─────────────────────────────────────────


_default_skills: MemorySkills | None = None


def get_skills() -> MemorySkills:
    """Return the default global MemorySkills instance."""
    global _default_skills
    if _default_skills is None:
        _default_skills = MemorySkills()
    return _default_skills


def add(node_id: str, content: str, label: str = "", metadata: dict | None = None) -> Node:
    return get_skills().add(node_id, content, label, metadata)


def search(query: str = "", label: str = "", limit: int = 50) -> list[Node]:
    return get_skills().search(query, label, limit)


def retrieve(limit: int = 10, label: str = "") -> list[tuple[Node, float]]:
    return get_skills().retrieve(limit, label)


def retrieve_text(limit: int = 10, label: str = "") -> str:
    return get_skills().retrieve_text(limit, label)


def corroborate(node_id: str, count: int = 1) -> None:
    get_skills().corroborate(node_id, count)


def delete(node_id: str) -> None:
    get_skills().delete(node_id)
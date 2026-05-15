"""Core Knowledge Graph engine — nodes, edges, persistence."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_memory.core.score import Score


class Node(BaseModel):
    """A fact or entity stored in the graph."""

    id: str
    label: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()

    def str_id(self) -> str:
        return self.id


class Edge(BaseModel):
    """A typed relationship between two nodes."""

    id: str
    source: str
    target: str
    label: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def str_id(self) -> str:
        return self.id


class _GraphStore(BaseModel):
    """JSON-serializable store for the entire graph state."""

    model_config = {"arbitrary_types_allowed": True}

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)


class KnowledgeGraph:
    """In-memory Knowledge Graph with JSON persistence and scoring."""

    def __init__(self, path: Path | str = "memory_graph.json") -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._store: _GraphStore = _GraphStore()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            # Reconstruct Pydantic models to validate dates
            nodes = {k: Node.model_validate(v) for k, v in raw.get("nodes", {}).items()}
            edges = {k: Edge.model_validate(v) for k, v in raw.get("edges", {}).items()}
            self._store = _GraphStore(nodes=nodes, edges=edges)

    def _save(self) -> None:
        raw = self._store.model_dump(mode="json")
        self.path.write_text(json.dumps(raw, indent=2, default=str))

    # ── Node CRUD ─────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        with self._lock:
            if node.id in self._store.nodes:
                raise ValueError(f"Node '{node.id}' already exists")
            self._store.nodes[node.id] = node
            self._save()

    def get_node(self, node_id: str) -> Node | None:
        with self._lock:
            return self._store.nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        with self._lock:
            if node_id not in self._store.nodes:
                raise KeyError(f"Node '{node_id}' not found")
            node = self._store.nodes[node_id]
            for key, value in kwargs.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            node.touch()
            self._save()

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._store.nodes:
                raise KeyError(f"Node '{node_id}' not found")
            del self._store.nodes[node_id]
            # Cascade: remove all edges referencing this node
            to_remove = [
                eid for eid, e in self._store.edges.items()
                if e.source == node_id or e.target == node_id
            ]
            for eid in to_remove:
                del self._store.edges[eid]
            self._save()

    # ── Edge CRUD ─────────────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            if edge.id in self._store.edges:
                raise ValueError(f"Edge '{edge.id}' already exists")
            if edge.source not in self._store.nodes:
                raise KeyError(f"Source node '{edge.source}' not found")
            if edge.target not in self._store.nodes:
                raise KeyError(f"Target node '{edge.target}' not found")
            self._store.edges[edge.id] = edge
            self._save()

    def get_edge(self, edge_id: str) -> Edge | None:
        with self._lock:
            return self._store.edges.get(edge_id)

    def update_edge(self, edge_id: str, **kwargs: Any) -> None:
        with self._lock:
            if edge_id not in self._store.edges:
                raise KeyError(f"Edge '{edge_id}' not found")
            edge = self._store.edges[edge_id]
            for key, value in kwargs.items():
                if hasattr(edge, key):
                    setattr(edge, key, value)
            self._save()

    def delete_edge(self, edge_id: str) -> None:
        with self._lock:
            if edge_id not in self._store.edges:
                raise KeyError(f"Edge '{edge_id}' not found")
            del self._store.edges[edge_id]
            self._save()

    # ── Queries ────────────────────────────────────────────────────────────────

    def list_nodes(self) -> list[Node]:
        with self._lock:
            return list(self._store.nodes.values())

    def list_edges(self) -> list[Edge]:
        with self._lock:
            return list(self._store.edges.values())

    def search(
        self,
        query: str = "",
        label: str = "",
        limit: int = 50,
    ) -> list[Node]:
        """Internal: full-text filter without scoring. Used by retrieve()."""
        with self._lock:
            results = list(self._store.nodes.values())
            if label:
                results = [n for n in results if n.label == label]
            if query:
                q = query.lower()
                results = [n for n in results if q in n.content.lower() or q in n.id.lower()]
            return results[:limit]

    def get_neighbors(self, node_id: str, label: str = "") -> list[Node]:
        with self._lock:
            edge_ids = [
                e.id for e in self._store.edges.values()
                if e.source == node_id or e.target == node_id
            ]
            neighbor_ids: set[str] = set()
            for eid in edge_ids:
                e = self._store.edges[eid]
                if e.source == node_id:
                    neighbor_ids.add(e.target)
                else:
                    neighbor_ids.add(e.source)
            nodes = [self._store.nodes[nid] for nid in neighbor_ids if nid in self._store.nodes]
            if label:
                nodes = [n for n in nodes if n.label == label]
            return nodes

    # ── Statistics ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"nodes": len(self._store.nodes), "edges": len(self._store.edges)}

    # ── Export / Import ────────────────────────────────────────────────────────

    def export_json(self) -> str:
        with self._lock:
            raw = self._store.model_dump(mode="json")
            return json.dumps(raw, indent=2, default=str)

    def import_json(self, data: str) -> None:
        raw = json.loads(data)
        with self._lock:
            for k, v in raw.get("nodes", {}).items():
                self._store.nodes[k] = Node.model_validate(v)
            for k, v in raw.get("edges", {}).items():
                self._store.edges[k] = Edge.model_validate(v)
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._store = _GraphStore()
            self._save()
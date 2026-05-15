"""Test the Knowledge Graph core — nodes, edges, and persistence."""
from __future__ import annotations

import pytest
import tempfile
import json
from pathlib import Path

from agentic_memory.core.graph import KnowledgeGraph, Node, Edge
from agentic_memory.core.score import Score


class TestNode:
    """Unit tests for Node creation and validation."""

    def test_node_create_minimal(self):
        node = Node(id="test-node-1", label="TestLabel", content="Some fact")
        assert node.id == "test-node-1"
        assert node.label == "TestLabel"
        assert node.content == "Some fact"
        assert node.metadata == {}

    def test_node_create_full(self):
        node = Node(
            id="n1",
            label="Person",
            content="Alice works at Acme",
            metadata={"source": "user:alex", "tags": ["work", "people"]},
        )
        assert node.metadata["source"] == "user:alex"
        assert "work" in node.metadata["tags"]

    def test_node_id_required(self):
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            Node(label="Foo", content="Bar")  # missing id

    def test_node_to_dict_roundtrip(self):
        node = Node(id="roundtrip", label="R", content="Content", metadata={"k": "v"})
        d = node.model_dump()
        restored = Node.model_validate(d)
        assert restored.id == node.id
        assert restored.content == node.content

    def test_node_str(self):
        node = Node(id="x", label="X", content="Hello")
        s = str(node)
        assert "x" in s
        assert "Hello" in s


class TestEdge:
    """Unit tests for Edge creation and validation."""

    def test_edge_create(self):
        edge = Edge(id="e1", source="node_a", target="node_b", label="knows", weight=0.8)
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.label == "knows"
        assert edge.weight == 0.8

    def test_edge_default_weight(self):
        edge = Edge(id="e2", source="a", target="b", label="relates")
        assert edge.weight == 1.0

    def test_edge_to_dict_roundtrip(self):
        edge = Edge(id="e3", source="s", target="t", label="l", weight=0.5)
        d = edge.model_dump()
        restored = Edge.model_validate(d)
        assert restored.label == "l"


class TestKnowledgeGraph:
    """Unit tests for the Knowledge Graph CRUD operations."""

    @pytest.fixture
    def graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_graph.json"
            g = KnowledgeGraph(path=path)
            yield g

    @pytest.fixture
    def graph_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_graph.json"
            g = KnowledgeGraph(path=path)
            g.add_node(Node(id="n1", label="Person", content="Alice"))
            g.add_node(Node(id="n2", label="Company", content="Acme Corp"))
            g.add_edge(Edge(id="e1", source="n1", target="n2", label="works_at"))
            yield g

    # ── Node CRUD ──────────────────────────────────────────────────────────────

    def test_add_node(self, graph):
        node = Node(id="n1", label="Fact", content="2+2=4")
        graph.add_node(node)
        assert graph.get_node("n1") is not None
        assert graph.get_node("n1").content == "2+2=4"

    def test_add_node_duplicate_raises(self, graph):
        node = Node(id="dup", label="X", content="Y")
        graph.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(Node(id="dup", label="X", content="Y"))

    def test_get_node_missing(self, graph):
        assert graph.get_node("nonexistent") is None

    def test_update_node(self, graph):
        graph.add_node(Node(id="u1", label="A", content="Old"))
        graph.update_node("u1", content="New")
        assert graph.get_node("u1").content == "New"

    def test_update_node_missing_raises(self, graph):
        with pytest.raises(KeyError):
            graph.update_node("ghost", content="X")

    def test_delete_node(self, graph):
        graph.add_node(Node(id="d1", label="D", content="ToDelete"))
        graph.delete_node("d1")
        assert graph.get_node("d1") is None

    def test_delete_node_missing_raises(self, graph):
        with pytest.raises(KeyError):
            graph.delete_node("ghost")

    def test_delete_node_cascades_edges(self, graph_with_data):
        """Deleting a node also removes all connected edges."""
        graph_with_data.delete_node("n1")
        edges = graph_with_data.list_edges()
        assert all(e.source != "n1" and e.target != "n1" for e in edges)

    # ── Edge CRUD ──────────────────────────────────────────────────────────────

    def test_add_edge(self, graph):
        graph.add_node(Node(id="a", label="A", content="Node A"))
        graph.add_node(Node(id="b", label="B", content="Node B"))
        edge = Edge(id="e1", source="a", target="b", label="links")
        graph.add_edge(edge)
        assert graph.get_edge("e1") is not None

    def test_add_edge_missing_node_raises(self, graph):
        graph.add_node(Node(id="only_source", label="X", content="Y"))
        edge = Edge(id="e2", source="only_source", target="ghost", label="broken")
        with pytest.raises(KeyError, match="not found"):
            graph.add_edge(edge)

    def test_add_edge_duplicate_raises(self, graph):
        graph.add_node(Node(id="x", label="X", content="Y"))
        graph.add_node(Node(id="y", label="Y", content="Z"))
        graph.add_edge(Edge(id="dup", source="x", target="y", label="rel"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_edge(Edge(id="dup", source="x", target="y", label="rel"))

    def test_get_edge_missing(self, graph):
        assert graph.get_edge("ghost") is None

    def test_update_edge(self, graph):
        graph.add_node(Node(id="s", label="S", content="Source"))
        graph.add_node(Node(id="t", label="T", content="Target"))
        graph.add_edge(Edge(id="ue", source="s", target="t", label="old"))
        graph.update_edge("ue", weight=0.9)
        assert graph.get_edge("ue").weight == 0.9

    def test_delete_edge(self, graph):
        graph.add_node(Node(id="a", label="A", content="B"))
        graph.add_node(Node(id="b", label="B", content="C"))
        graph.add_edge(Edge(id="de", source="a", target="b", label="del"))
        graph.delete_edge("de")
        assert graph.get_edge("de") is None

    # ── Search ─────────────────────────────────────────────────────────────────

    def test_search(self, graph):
        graph.add_node(Node(id="n1", label="Person", content="Alice lives in Paris"))
        graph.add_node(Node(id="n2", label="City", content="Paris is the capital of France"))
        graph.add_node(Node(id="n3", label="Person", content="Bob lives in London"))
        results = graph.search("Paris")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"n1", "n2"}

    def test_search_empty(self, graph):
        assert graph.search("nowhere") == []

    def test_search_by_label(self, graph):
        graph.add_node(Node(id="n1", label="Person", content="Alice"))
        graph.add_node(Node(id="n2", label="Place", content="Paris"))
        results = graph.search(label="Person")
        assert len(results) == 1
        assert results[0].id == "n1"

    def test_search_combined(self, graph):
        graph.add_node(Node(id="n1", label="Person", content="Alice in Paris"))
        graph.add_node(Node(id="n2", label="Person", content="Alice in London"))
        graph.add_node(Node(id="n3", label="City", content="Paris"))
        results = graph.search(query="Alice", label="Person")
        assert len(results) == 2

    # ── Persistence ────────────────────────────────────────────────────────────

    def test_persist_roundtrip(self, graph_with_data):
        path = graph_with_data.path
        # Re-load from disk
        reloaded = KnowledgeGraph(path=path)
        assert reloaded.get_node("n1") is not None
        assert reloaded.get_node("n2") is not None
        assert reloaded.get_edge("e1") is not None

    def test_persist_auto_save(self, graph):
        node = Node(id="autosave", label="Test", content="Auto-save works")
        graph.add_node(node)
        # Force reload from disk
        reloaded = KnowledgeGraph(path=graph.path)
        assert reloaded.get_node("autosave") is not None

    # ── Statistics ──────────────────────────────────────────────────────────────

    def test_stats(self, graph_with_data):
        stats = graph_with_data.stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_stats_empty_graph(self, graph):
        stats = graph.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    # ── Export / Import ────────────────────────────────────────────────────────

    def test_export_json(self, graph_with_data):
        exported = graph_with_data.export_json()
        data = json.loads(exported)
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2

    def test_import_json(self, graph):
        data = {
            "nodes": {"i1": {"id": "i1", "label": "Imported", "content": "From JSON", "metadata": {}}},
            "edges": {},
        }
        graph.import_json(json.dumps(data))
        assert graph.get_node("i1") is not None

    # ── Clear ─────────────────────────────────────────────────────────────────

    def test_clear(self, graph_with_data):
        graph_with_data.clear()
        stats = graph_with_data.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
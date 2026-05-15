"""Tests for pruning and auto-pruning logic."""
from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from agentic_memory.core.graph import KnowledgeGraph, Node, Edge
from agentic_memory.core.prune import AutoPruner


class TestAutoPruner:
    @pytest.fixture
    def graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prune_graph.json"
            g = KnowledgeGraph(path=path)
            for i in range(20):
                g.add_node(Node(id=f"n{i}", label="Fact", content=f"Fact {i}"))
            yield g

    def test_prune_oldest_by_count(self, graph):
        pruner = AutoPruner(max_nodes=5)
        removed = pruner.prune(graph, strategy="oldest")
        assert len(removed) == 15
        assert graph.stats()["nodes"] == 5

    def test_prune_by_size(self, graph):
        pruner = AutoPruner(max_nodes=10)
        removed = pruner.prune(graph, strategy="score")
        assert graph.stats()["nodes"] == 10

    def test_prune_low_score(self, graph):
        pruner = AutoPruner(max_nodes=5)
        removed = pruner.prune(graph, strategy="low_score")
        assert graph.stats()["nodes"] >= 5

    def test_prune_noop_when_under_limit(self, graph):
        pruner = AutoPruner(max_nodes=100)
        removed = pruner.prune(graph, strategy="oldest")
        assert len(removed) == 0

    def test_prune_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.json"
            g = KnowledgeGraph(path=path)
            g.add_node(Node(id="tiny", label="X", content="Y"))
            pruner = AutoPruner(max_nodes=0)
            removed = pruner.prune(g, strategy="oldest")
            assert len(removed) == 1
            assert g.stats()["nodes"] == 0

    def test_prune_labels_only(self, graph):
        pruner = AutoPruner(max_nodes=3)
        removed_ids = pruner.prune(graph, strategy="oldest", labels=["Fact"])
        # All nodes in graph fixture have label "Fact" so all are removed
        assert len(removed_ids) == 17

    def test_auto_prune_callback(self, graph):
        calls = []

        def on_prune(removed_ids):
            calls.extend(removed_ids)

        pruner = AutoPruner(max_nodes=5, on_prune=on_prune)
        removed = pruner.prune(graph, strategy="oldest")
        # max_nodes=5, graph has 20 → removes 15
        assert len(calls) == 15

    def test_size_limit_check(self, graph):
        pruner = AutoPruner(max_size_mb=0.001)
        # Very small limit should trigger pruning
        result = pruner.check_size(graph)
        # Just verify no exception
        assert isinstance(result, list)
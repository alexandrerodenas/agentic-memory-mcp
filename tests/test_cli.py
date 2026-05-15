"""Tests for the CLI."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from agentic_memory.cli.main import cli
from agentic_memory.core.graph import KnowledgeGraph, Node, Edge


class TestCLI:
    @pytest.fixture
    def tmp_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("cli") / "cli_graph.json"

    @pytest.fixture
    def runner(self):
        return CliRunner()

    # ── Node commands ──────────────────────────────────────────────────────────

    def test_node_add(self, runner, tmp_path):
        result = runner.invoke(cli, [
            "--graph", str(tmp_path),
            "node", "add",
            "--id", "c1",
            "--label", "Person",
            "--content", "Alice",
        ])
        assert result.exit_code == 0
        g = KnowledgeGraph(path=tmp_path)
        assert g.get_node("c1") is not None

    def test_node_list(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="l1", label="X", content="Y"))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "node", "list"])
        assert result.exit_code == 0
        assert "l1" in result.output

    def test_node_get(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="g1", label="City", content="Nantes"))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "node", "get", "g1"])
        assert result.exit_code == 0
        assert "Nantes" in result.output

    def test_node_update(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="u1", label="A", content="Old"))
        result = runner.invoke(cli, [
            "--graph", str(tmp_path),
            "node", "update", "u1",
            "--content", "NewContent",
        ])
        assert result.exit_code == 0
        # Reload graph to see CLI's persisted changes
        g._load()
        assert g.get_node("u1").content == "NewContent"

    def test_node_delete(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="d1", label="D", content="DelMe"))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "node", "delete", "d1"])
        assert result.exit_code == 0
        # Reload graph to see CLI's persisted changes
        g._load()
        assert g.get_node("d1") is None

    def test_retrieve_query_filter(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="s1", label="Place", content="Nantes"))
        g.add_node(Node(id="s2", label="Place", content="Paris"))
        result = runner.invoke(cli, [
            "--graph", str(tmp_path),
            "retrieve", "--query", "Nantes", "--limit", "10",
        ])
        assert result.exit_code == 0
        assert "s1" in result.output
        assert "s2" not in result.output

    # ── Edge commands ──────────────────────────────────────────────────────────

    def test_edge_add(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="a", label="A", content="Source"))
        g.add_node(Node(id="b", label="B", content="Target"))
        result = runner.invoke(cli, [
            "--graph", str(tmp_path),
            "edge", "add",
            "--id", "e1",
            "--source", "a",
            "--target", "b",
            "--label", "links_to",
        ])
        assert result.exit_code == 0
        g._load()
        assert g.get_edge("e1") is not None

    def test_edge_list(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="x", label="X", content="Y"))
        g.add_node(Node(id="y", label="Y", content="Z"))
        g.add_edge(Edge(id="el1", source="x", target="y", label="relates"))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "edge", "list"])
        assert result.exit_code == 0
        assert "el1" in result.output

    # ── Stats & Export ─────────────────────────────────────────────────────────

    def test_stats(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="st1", label="T", content="X"))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "stats"])
        assert result.exit_code == 0
        assert "nodes" in result.output.lower()

    def test_export(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        g.add_node(Node(id="ex1", label="E", content="ExportMe"))
        out = tmp_path.parent / "export.json"
        result = runner.invoke(cli, ["--graph", str(tmp_path), "export", str(out)])
        assert result.exit_code == 0
        assert json.loads(out.read_text())["nodes"]

    def test_import(self, runner, tmp_path):
        data = {"nodes": {"imp1": {"id": "imp1", "label": "I", "content": "Imported", "metadata": {}}}, "edges": {}}
        src = tmp_path.parent / "import_src.json"
        src.write_text(json.dumps(data))
        result = runner.invoke(cli, ["--graph", str(tmp_path), "import", str(src)])
        assert result.exit_code == 0
        assert KnowledgeGraph(path=tmp_path).get_node("imp1") is not None

    # ── Prune ──────────────────────────────────────────────────────────────────

    def test_prune(self, runner, tmp_path):
        g = KnowledgeGraph(path=tmp_path)
        for i in range(10):
            g.add_node(Node(id=f"p{i}", label="Fact", content=f"F{i}"))
        result = runner.invoke(cli, [
            "--graph", str(tmp_path),
            "prune",
            "--max-nodes", "3",
            "--strategy", "oldest",
        ])
        assert result.exit_code == 0
        g._load()
        # 10 - 3 = 7 removed, 3 remaining
        assert g.stats()["nodes"] == 3

    # ── Help ───────────────────────────────────────────────────────────────────

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "node" in result.output
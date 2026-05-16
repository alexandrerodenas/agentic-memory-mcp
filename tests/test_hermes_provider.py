"""Tests for the Hermes MemoryProvider integration.

These tests mock the MemoryProvider ABC (from hermes-agent) so they can run
without the full Hermes Agent installed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Mock the MemoryProvider ABC before importing the plugin ────────────────
# The real ABC lives in hermes-agent/agent/memory_provider.py.  We inject a
# lightweight stand-in so the test suite doesn't depend on a hermes-agent checkout.


class _FakeMemoryProvider:
    """Minimal stand-in for agent.memory_provider.MemoryProvider."""

    @property
    def name(self):
        return "fake"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"

    def system_prompt_block(self):
        return ""

    def prefetch(self, query, **kwargs):
        return ""

    def get_config_schema(self):
        return []

    def save_config(self, values, hermes_home):
        pass

    def shutdown(self):
        pass


# Inject the fake module before the plugin tries to import it
_fake_agent_module = type(sys)("agent")
_fake_agent_module.memory_provider = type(sys)("agent.memory_provider")
_fake_agent_module.memory_provider.MemoryProvider = _FakeMemoryProvider
sys.modules["agent"] = _fake_agent_module
sys.modules["agent.memory_provider"] = _fake_agent_module.memory_provider

# Now import the plugin
from agentic_memory.hermes import AgenticMemoryProvider, register  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_hermes_home(tmp_path):
    """Create a temporary HERMES_HOME directory."""
    return str(tmp_path / "hermes_home")


@pytest.fixture
def provider(tmp_hermes_home):
    """Create an initialized AgenticMemoryProvider."""
    p = AgenticMemoryProvider()
    p.initialize("test-session", hermes_home=tmp_hermes_home)
    return p


# ── Tests ───────────────────────────────────────────────────────────────────


class TestAgenticMemoryProvider:
    """Test the MemoryProvider implementation."""

    def test_name(self, provider):
        assert provider.name == "agentic-memory"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_initialize_creates_graph_dir(self, provider, tmp_hermes_home):
        graph_dir = Path(tmp_hermes_home) / "memory_graph"
        assert graph_dir.exists()

    def test_initialize_sets_session_id(self, provider):
        assert provider._session_id == "test-session"

    def test_system_prompt_block(self, provider):
        block = provider.system_prompt_block()
        assert "Knowledge Graph" in block
        assert "memory_node_add" in block

    def test_prefetch_empty(self, provider):
        result = provider.prefetch("some query")
        assert result == ""  # no memories yet

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 12
        names = {s["name"] for s in schemas}
        assert "memory_node_add" in names
        assert "memory_retrieve" in names
        assert "memory_corroborate" in names
        assert "memory_prune" in names

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) >= 1
        keys = {f["key"] for f in schema}
        assert "auto_prune_enabled" in keys

    def test_save_config(self, provider, tmp_hermes_home):
        provider.save_config({"auto_prune_enabled": "true"}, tmp_hermes_home)
        config_path = Path(tmp_hermes_home) / "agentic-memory.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["auto_prune_enabled"] == "true"

    def test_shutdown(self, provider):
        provider.shutdown()
        assert provider._skills is None

    # ── Tool calls ──────────────────────────────────────────────────────────

    def test_node_add(self, provider):
        result = json.loads(provider.handle_tool_call("memory_node_add", {"content": "Alice lives in Paris", "label": "Person"}))
        assert result["ok"] is True
        assert "node_id" in result

    def test_node_add_with_id(self, provider):
        result = json.loads(provider.handle_tool_call("memory_node_add", {"id": "alice", "content": "Alice"}))
        assert result["ok"] is True
        assert result["node_id"] == "alice"

    def test_node_get(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "bob", "content": "Bob"})
        result = json.loads(provider.handle_tool_call("memory_node_get", {"id": "bob"}))
        assert result["ok"] is True
        assert result["node"]["content"] == "Bob"

    def test_node_get_missing(self, provider):
        result = json.loads(provider.handle_tool_call("memory_node_get", {"id": "ghost"}))
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_node_update(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "u1", "content": "Old"})
        result = json.loads(provider.handle_tool_call("memory_node_update", {"id": "u1", "content": "New"}))
        assert result["ok"] is True

    def test_node_delete(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "d1", "content": "Delete me"})
        result = json.loads(provider.handle_tool_call("memory_node_delete", {"id": "d1"}))
        assert result["ok"] is True

    def test_edge_add(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "a", "content": "A"})
        provider.handle_tool_call("memory_node_add", {"id": "b", "content": "B"})
        result = json.loads(provider.handle_tool_call("memory_edge_add", {"source": "a", "target": "b", "label": "knows"}))
        assert result["ok"] is True
        assert "edge_id" in result

    def test_edge_delete(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "x", "content": "X"})
        provider.handle_tool_call("memory_node_add", {"id": "y", "content": "Y"})
        edge_result = json.loads(provider.handle_tool_call("memory_edge_add", {"id": "e1", "source": "x", "target": "y"}))
        result = json.loads(provider.handle_tool_call("memory_edge_delete", {"id": "e1"}))
        assert result["ok"] is True

    def test_retrieve(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "r1", "content": "Nantes is a city", "label": "Place"})
        provider.handle_tool_call("memory_node_add", {"id": "r2", "content": "Paris is the capital", "label": "Place"})
        result = json.loads(provider.handle_tool_call("memory_retrieve", {"query": "Nantes", "limit": 5}))
        assert result["ok"] is True
        assert "Nantes" in result["results"]

    def test_corroborate(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "c1", "content": "Fact"})
        result = json.loads(provider.handle_tool_call("memory_corroborate", {"id": "c1", "count": 3}))
        assert result["ok"] is True

    def test_stats(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "s1", "content": "X"})
        result = json.loads(provider.handle_tool_call("memory_stats", {}))
        assert result["ok"] is True
        assert result["stats"]["nodes"] == 1

    def test_export(self, provider, tmp_path):
        provider.handle_tool_call("memory_node_add", {"id": "ex1", "content": "Export me"})
        out = str(tmp_path / "export.json")
        result = json.loads(provider.handle_tool_call("memory_export", {"path": out}))
        assert result["ok"] is True
        assert Path(out).exists()

    def test_import(self, provider, tmp_path):
        data = {"nodes": {"imp1": {"id": "imp1", "label": "I", "content": "Imported", "metadata": {}}}, "edges": {}}
        src = tmp_path / "import.json"
        src.write_text(json.dumps(data))
        result = json.loads(provider.handle_tool_call("memory_import", {"path": str(src)}))
        assert result["ok"] is True

    def test_prune(self, provider):
        for i in range(15):
            provider.handle_tool_call("memory_node_add", {"id": f"p{i}", "content": f"Fact {i}"})
        result = json.loads(provider.handle_tool_call("memory_prune", {"count": 5, "strategy": "oldest"}))
        assert result["ok"] is True

    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call("nonexistent_tool", {}))
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    # ── Prefetch after adding data ──────────────────────────────────────────

    def test_prefetch_with_data(self, provider):
        provider.handle_tool_call("memory_node_add", {"id": "pf1", "content": "Python is a language", "label": "Fact"})
        result = provider.prefetch("Python")
        assert "Python" in result

    # ── on_memory_write hook ────────────────────────────────────────────────

    def test_on_memory_write(self, provider):
        provider.on_memory_write("add", "memory", "User prefers dark mode")
        # Should have added a node
        result = json.loads(provider.handle_tool_call("memory_stats", {}))
        assert result["stats"]["nodes"] >= 1

    # ── register() function ─────────────────────────────────────────────────

    def test_register(self):
        collector = MagicMock()
        register(collector)
        collector.register_memory_provider.assert_called_once()
        args = collector.register_memory_provider.call_args[0]
        assert isinstance(args[0], AgenticMemoryProvider)


class TestCLIModule:
    """Test the CLI registration module."""

    def test_register_cli_importable(self):
        from agentic_memory.hermes.cli import register_cli

        assert callable(register_cli)

    def test_register_cli_creates_subcommands(self):
        import argparse

        from agentic_memory.hermes.cli import register_cli

        # Hermes passes an ArgumentParser (the subparser for this provider),
        # not a _SubParsersAction
        parser = argparse.ArgumentParser()
        register_cli(parser)

        # Verify subcommands exist
        args = parser.parse_args(["status"])
        assert hasattr(args, "agentic_memory_command")

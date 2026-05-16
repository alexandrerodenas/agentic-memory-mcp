"""Hermes Memory Provider Plugin — agentic-memory-mcp.

Implements the MemoryProvider ABC so Hermes Agent can use the Knowledge Graph
as its memory backend.  MCP server and CLI remain fully functional alongside.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ─── Tool schemas ────────────────────────────────────────────────────────────

_NODE_ADD_SCHEMA = {
    "name": "memory_node_add",
    "description": "Store a new fact in the agentic memory graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique node identifier (auto-generated if omitted)."},
            "content": {"type": "string", "description": "The fact or knowledge to store."},
            "label": {
                "type": "string",
                "description": "Optional label to categorize the memory (e.g. 'project', 'preference', 'fact').",
            },
            "metadata": {
                "type": "object",
                "description": "Optional key-value metadata.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["content"],
    },
}

_NODE_GET_SCHEMA = {
    "name": "memory_node_get",
    "description": "Retrieve a specific memory node by its ID.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "ID of the memory node to retrieve."}},
        "required": ["id"],
    },
}

_NODE_UPDATE_SCHEMA = {
    "name": "memory_node_update",
    "description": "Update the content or metadata of an existing memory node.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "ID of the node to update."},
            "content": {"type": "string", "description": "New content (optional)."},
            "metadata": {"type": "object", "description": "Metadata to merge (optional)."},
        },
        "required": ["id"],
    },
}

_NODE_DELETE_SCHEMA = {
    "name": "memory_node_delete",
    "description": "Delete a memory node by ID.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "ID of the node to delete."}},
        "required": ["id"],
    },
}

_EDGE_ADD_SCHEMA = {
    "name": "memory_edge_add",
    "description": "Create a relationship/edge between two memory nodes.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique edge ID."},
            "source": {"type": "string", "description": "Source node ID."},
            "target": {"type": "string", "description": "Target node ID."},
            "label": {"type": "string", "description": "Relationship label (e.g. 'depends_on', 'part_of')."},
            "weight": {"type": "number", "description": "Edge weight (default 1.0)."},
        },
        "required": ["source", "target"],
    },
}

_EDGE_DELETE_SCHEMA = {
    "name": "memory_edge_delete",
    "description": "Delete a relationship/edge by ID.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "ID of the edge to delete."}},
        "required": ["id"],
    },
}

_RETRIEVE_SCHEMA = {
    "name": "memory_retrieve",
    "description": (
        "Retrieve the most relevant memories, ranked by a composite score "
        "(corroborations, read count, recency). Optionally filter by text query or label. "
        "Use this as the primary memory recall tool — limits results to control token usage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text search filter (optional)."},
            "label": {"type": "string", "description": "Label filter (optional)."},
            "limit": {"type": "integer", "description": "Max results to return (default: 10)."},
        },
        "required": [],
    },
}

_CORROBORATE_SCHEMA = {
    "name": "memory_corroborate",
    "description": "Reinforce a memory entry, increasing its corroboration score.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Node identifier to reinforce."},
            "count": {"type": "integer", "description": "Corroboration increment (default 1)."},
        },
        "required": ["id"],
    },
}

_STATS_SCHEMA = {
    "name": "memory_stats",
    "description": "Get statistics about the memory graph (nodes, edges, storage size).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_EXPORT_SCHEMA = {
    "name": "memory_export",
    "description": "Export the entire memory graph to a JSON file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Output path (optional)."}},
        "required": [],
    },
}

_IMPORT_SCHEMA = {
    "name": "memory_import",
    "description": "Import a memory graph from a JSON export file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the JSON export file."}},
        "required": ["path"],
    },
}

_PRUNE_SCHEMA = {
    "name": "memory_prune",
    "description": "Manually prune low-value or obsolete memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Number of nodes to remove (default: 10)."},
            "strategy": {
                "type": "string",
                "enum": ["low_score", "oldest"],
                "description": "Pruning strategy.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only prune nodes with these labels (optional).",
            },
        },
        "required": [],
    },
}

ALL_TOOL_SCHEMAS = [
    _NODE_ADD_SCHEMA,
    _NODE_GET_SCHEMA,
    _NODE_UPDATE_SCHEMA,
    _NODE_DELETE_SCHEMA,
    _EDGE_ADD_SCHEMA,
    _EDGE_DELETE_SCHEMA,
    _RETRIEVE_SCHEMA,
    _CORROBORATE_SCHEMA,
    _STATS_SCHEMA,
    _EXPORT_SCHEMA,
    _IMPORT_SCHEMA,
    _PRUNE_SCHEMA,
]


# ─── MemoryProvider implementation ──────────────────────────────────────────


class AgenticMemoryProvider(MemoryProvider):
    """Knowledge Graph memory provider for Hermes Agent.

    Stores facts, entities, and relationships in a JSON-based Knowledge Graph
    with a composite relevance scoring system.  Fully local — no external
    services or API keys required.
    """

    def __init__(self) -> None:
        self._hermes_home: str = ""
        self._graph_dir: Path | None = None
        self._skills = None  # lazy MemorySkills instance
        self._session_id: str = ""
        self._sync_thread: threading.Thread | None = None
        self._auto_prune_enabled: bool = False
        self._auto_prune_max_nodes: int | None = None
        self._auto_prune_interval: int = 3600
        self._turn_count: int = 0

    @property
    def name(self) -> str:
        return "agentic-memory"

    # ── Core lifecycle ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Always available — no external deps or API keys needed."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Set up graph storage paths using hermes_home."""
        self._hermes_home = kwargs.get("hermes_home", "")
        self._session_id = session_id

        if self._hermes_home:
            self._graph_dir = Path(self._hermes_home) / "memory_graph"
        else:
            self._graph_dir = Path.home() / ".hermes" / "memory_graph"

        self._graph_dir.mkdir(parents=True, exist_ok=True)

        # Load auto-prune config from env
        self._auto_prune_enabled = os.getenv("MEMORY_AUTO_PRUNE_ENABLED", "0") in ("1", "true", "True")
        self._auto_prune_max_nodes = int(os.getenv("MEMORY_AUTO_PRUNE_MAX_NODES", "0") or "0") or None
        self._auto_prune_interval = int(os.getenv("MEMORY_AUTO_PRUNE_INTERVAL_SECONDS", "3600"))

        logger.info(
            "AgenticMemory initialized: graph_dir=%s, session=%s",
            self._graph_dir,
            session_id,
        )

    def _get_skills(self):
        """Lazy-init MemorySkills with profile-scoped paths."""
        if self._skills is None:
            from agentic_memory.api import MemorySkills

            graph_path = self._graph_dir / "memory_graph.json" if self._graph_dir else "memory_graph.json"
            scores_path = self._graph_dir / "memory_scores.json" if self._graph_dir else "memory_scores.json"
            self._skills = MemorySkills(graph_path=str(graph_path), scores_path=str(scores_path))
        return self._skills

    def system_prompt_block(self) -> str:
        """Static info about the memory system for the system prompt."""
        skills = self._get_skills()
        s = skills.stats()
        return (
            f"## Agentic Memory (Knowledge Graph)\n"
            f"You have persistent memory via a Knowledge Graph with {s['nodes']} nodes "
            f"and {s['edges']} edges. Use `memory_node_add` to store facts, `memory_retrieve` "
            f"to recall them (ranked by relevance score), and `memory_corroborate` to reinforce "
            f"important memories. Memories persist across sessions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve top memories relevant to the query for context injection."""
        try:
            skills = self._get_skills()
            result = skills.retrieve_text(query=query, limit=5)
            if result and result != "No relevant memories found.":
                return f"## Relevant Memories\n{result}"
        except Exception as e:
            logger.debug("Prefetch failed: %s", e)
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Non-blocking: queue conversation turn for potential fact extraction."""
        self._turn_count += 1

        # Auto-prune check every 50 turns
        if self._auto_prune_enabled and self._turn_count % 50 == 0:
            def _maybe_prune():
                try:
                    skills = self._get_skills()
                    if self._auto_prune_max_nodes:
                        skills.prune(max_nodes=self._auto_prune_max_nodes)
                except Exception as e:
                    logger.debug("Auto-prune failed: %s", e)

            t = threading.Thread(target=_maybe_prune, daemon=True)
            t.start()

    # ── Tool schemas & dispatch ─────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch a tool call to the appropriate handler."""
        import uuid

        skills = self._get_skills()

        try:
            if tool_name == "memory_node_add":
                node_id = args.get("id") or f"mem_{uuid.uuid4().hex[:8]}"
                node = skills.add(
                    node_id=node_id,
                    content=args.get("content", ""),
                    label=args.get("label", ""),
                    metadata=args.get("metadata"),
                )
                return json.dumps({"ok": True, "node_id": node.id})

            elif tool_name == "memory_node_get":
                node = skills.get(args.get("id", ""))
                if node is None:
                    return json.dumps({"ok": False, "error": "Node not found"})
                return json.dumps({"ok": True, "node": node.model_dump(mode="json")})

            elif tool_name == "memory_node_update":
                skills.update(
                    args.get("id", ""),
                    content=args.get("content"),
                    metadata=args.get("metadata"),
                )
                return json.dumps({"ok": True, "node_id": args.get("id")})

            elif tool_name == "memory_node_delete":
                skills.delete(args.get("id", ""))
                return json.dumps({"ok": True})

            elif tool_name == "memory_edge_add":
                edge_id = args.get("id") or f"edge_{uuid.uuid4().hex[:8]}"
                edge = skills.add_edge(
                    edge_id=edge_id,
                    source=args.get("source", ""),
                    target=args.get("target", ""),
                    label=args.get("label", ""),
                    weight=args.get("weight", 1.0),
                )
                return json.dumps({"ok": True, "edge_id": edge.id})

            elif tool_name == "memory_edge_delete":
                graph = skills.graph
                graph.delete_edge(args.get("id", ""))
                return json.dumps({"ok": True})

            elif tool_name == "memory_retrieve":
                result = skills.retrieve_text(
                    query=args.get("query", ""),
                    label=args.get("label", ""),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"ok": True, "results": result})

            elif tool_name == "memory_corroborate":
                skills.corroborate(args.get("id", ""), count=int(args.get("count", 1)))
                return json.dumps({"ok": True})

            elif tool_name == "memory_stats":
                return json.dumps({"ok": True, "stats": skills.stats()})

            elif tool_name == "memory_export":
                path = args.get("path") or str(self._graph_dir / "export.json") if self._graph_dir else "export.json"
                data = skills.export_json()
                Path(path).write_text(data)
                return json.dumps({"ok": True, "path": path})

            elif tool_name == "memory_import":
                data = Path(args.get("path", "")).read_text()
                skills.import_json(data)
                return json.dumps({"ok": True})

            elif tool_name == "memory_prune":
                removed = skills.prune(
                    max_nodes=int(args.get("count", 10)),
                    strategy=args.get("strategy", "low_score"),
                )
                return json.dumps({"ok": True, "removed": removed})

            else:
                return json.dumps({"ok": False, "error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return json.dumps({"ok": False, "error": str(e)})

    # ── Config ──────────────────────────────────────────────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """No secrets needed — fully local provider."""
        return [
            {
                "key": "auto_prune_enabled",
                "description": "Enable automatic pruning of old memories",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "auto_prune_max_nodes",
                "description": "Maximum nodes before auto-prune triggers",
                "default": "1000",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/agentic-memory.json."""
        config_path = Path(hermes_home) / "agentic-memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # ── Optional hooks ──────────────────────────────────────────────────────

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror built-in memory writes to the graph as 'hermes_memory' nodes."""
        try:
            if action == "add" and target == "memory":
                skills = self._get_skills()
                import uuid

                skills.add(
                    node_id=f"hm_{uuid.uuid4().hex[:8]}",
                    content=content,
                    label="hermes_memory",
                    metadata=metadata if isinstance(metadata, dict) else None,
                )
        except Exception as e:
            logger.debug("on_memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._skills = None
        logger.info("AgenticMemory provider shut down")


# ─── Plugin registration ────────────────────────────────────────────────────


def register(ctx) -> None:
    """Called by the Hermes memory plugin discovery system."""
    ctx.register_memory_provider(AgenticMemoryProvider())


# ─── Legacy entry point (for pyproject.toml backward compat) ────────────────


def setup(agent=None) -> None:
    """Legacy entry point — kept for backward compatibility with pyproject.toml."""
    pass

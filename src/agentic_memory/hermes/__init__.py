"""Hermes plugin — integrate agentic-memory-mcp into Hermes Agent."""

from __future__ import annotations

import logging
from pathlib import Path

from hermes_cli.plugins import PluginContext

logger = logging.getLogger(__name__)

# ─── Tool schemas ────────────────────────────────────────────────────────────


def _get_graph_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "memory_graph"


def _graph_instance():
    from agentic_memory.core.graph import KnowledgeGraph

    return KnowledgeGraph(data_dir=str(_get_graph_dir()))


def _skills_instance():
    from agentic_memory.api import MemorySkills

    scores_path = os.getenv("MEMORY_SCORES_PATH", "memory_scores.json")
    return MemorySkills(graph_path=str(_get_graph_dir() / "memory_graph.json"), scores_path=scores_path)


# ─── Tool implementations ────────────────────────────────────────────────────


def _node_add(args, task_id=None):
    import json

    skills = _skills_instance()
    node = skills.add_node(
        content=args.get("content", ""),
        label=args.get("label", ""),
        metadata=args.get("metadata", {}),
    )
    return json.dumps({"ok": True, "node_id": node.id})


_NODE_ADD_SCHEMA = {
    "name": "memory_node_add",
    "description": "Store a new fact in the agentic memory graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or knowledge to store."},
            "label": {"type": "string", "description": "Optional label to categorize the memory (e.g. 'project', 'preference', 'fact')."},
            "metadata": {
                "type": "object",
                "description": "Optional key-value metadata.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["content"],
    },
}


def _node_get(args, task_id=None):
    import json

    skills = _skills_instance()
    node = skills.get_node(args.get("node_id", ""))
    if node is None:
        return json.dumps({"ok": False, "error": "Node not found"})
    return json.dumps({"ok": True, "node": node.model_dump()})


_NODE_GET_SCHEMA = {
    "name": "memory_node_get",
    "description": "Retrieve a specific memory node by its ID.",
    "parameters": {
        "type": "object",
        "properties": {"node_id": {"type": "string", "description": "ID of the memory node to retrieve."}},
        "required": ["node_id"],
    },
}


def _node_update(args, task_id=None):
    import json

    skills = _skills_instance()
    node = skills.update_node(args.get("node_id", ""), content=args.get("content"), metadata=args.get("metadata"))
    return json.dumps({"ok": True, "node_id": node.id if node else None})


_NODE_UPDATE_SCHEMA = {
    "name": "memory_node_update",
    "description": "Update the content or metadata of an existing memory node.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "ID of the node to update."},
            "content": {"type": "string", "description": "New content (optional)."},
            "metadata": {"type": "object", "description": "Metadata to merge (optional)."},
        },
        "required": ["node_id"],
    },
}


def _node_delete(args, task_id=None):
    import json

    skills = _skills_instance()
    removed = skills.delete_node(args.get("node_id", ""))
    return json.dumps({"ok": removed})


_NODE_DELETE_SCHEMA = {
    "name": "memory_node_delete",
    "description": "Delete a memory node by ID.",
    "parameters": {
        "type": "object",
        "properties": {"node_id": {"type": "string", "description": "ID of the node to delete."}},
        "required": ["node_id"],
    },
}


def _edge_add(args, task_id=None):
    import json

    skills = _skills_instance()
    edge = skills.add_edge(args.get("from_node", ""), args.get("to_node", ""), label=args.get("label", ""))
    return json.dumps({"ok": True, "edge_id": edge.id if edge else None})


_EDGE_ADD_SCHEMA = {
    "name": "memory_edge_add",
    "description": "Create a relationship/edge between two memory nodes.",
    "parameters": {
        "type": "object",
        "properties": {
            "from_node": {"type": "string", "description": "Source node ID."},
            "to_node": {"type": "string", "description": "Target node ID."},
            "label": {"type": "string", "description": "Relationship label (e.g. 'depends_on', 'part_of')."},
        },
        "required": ["from_node", "to_node"],
    },
}


def _edge_delete(args, task_id=None):
    import json

    skills = _skills_instance()
    removed = skills.delete_edge(args.get("edge_id", ""))
    return json.dumps({"ok": removed})


_EDGE_DELETE_SCHEMA = {
    "name": "memory_edge_delete",
    "description": "Delete a relationship/edge by ID.",
    "parameters": {
        "type": "object",
        "properties": {"edge_id": {"type": "string", "description": "ID of the edge to delete."}},
        "required": ["edge_id"],
    },
}


def _retrieve(args, task_id=None):
    import json

    skills = _skills_instance()
    results = skills.retrieve_text(
        query=args.get("query", ""),
        label=args.get("label", ""),
        limit=int(args.get("limit", 10)),
    )
    return json.dumps({"ok": True, "results": results})


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


def _stats(args, task_id=None):
    import json

    graph = _graph_instance()
    return json.dumps({"ok": True, "stats": graph.stats()})


_STATS_SCHEMA = {
    "name": "memory_stats",
    "description": "Get statistics about the memory graph (nodes, edges, storage size).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _export(args, task_id=None):
    import json

    graph = _graph_instance()
    path = args.get("path") or str(_get_graph_dir() / "export.json")
    graph.export_json(path)
    return json.dumps({"ok": True, "path": path})


_EXPORT_SCHEMA = {
    "name": "memory_export",
    "description": "Export the entire memory graph to a JSON file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Output path (optional, defaults to memory_graph/export.json)."}},
        "required": [],
    },
}


def _import_graph(args, task_id=None):
    import json

    graph = _graph_instance()
    count = graph.import_json(args.get("path", ""))
    return json.dumps({"ok": True, "imported": count})


_IMPORT_SCHEMA = {
    "name": "memory_import",
    "description": "Import a memory graph from a JSON export file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the JSON export file."}},
        "required": ["path"],
    },
}


def _prune(args, task_id=None):
    import json

    from agentic_memory.core.prune import AutoPruner

    graph = _graph_instance()
    pruner = AutoPruner(graph)
    removed = pruner.prune(
        count=int(args.get("count", 10)),
        strategy=args.get("strategy", "low_score"),
        labels=args.get("labels"),
    )
    return json.dumps({"ok": True, "removed": [e.model_dump() for e in removed]})


_PRUNE_SCHEMA = {
    "name": "memory_prune",
    "description": "Manually prune low-value or obsolete memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Number of nodes to remove (default: 10)."},
            "strategy": {"type": "string", "enum": ["low_score", "oldest"], "description": "Pruning strategy."},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Only prune nodes with these labels (optional)."},
        },
        "required": [],
    },
}


# ─── Plugin registration ─────────────────────────────────────────────────────


def register(ctx: PluginContext) -> None:
    """Register all agentic-memory tools and skills with Hermes."""

    # ── Memory tools ──────────────────────────────────────────────────────
    ctx.register_tool("memory_node_add", "memory", _NODE_ADD_SCHEMA, _node_add)
    ctx.register_tool("memory_node_get", "memory", _NODE_GET_SCHEMA, _node_get)
    ctx.register_tool("memory_node_update", "memory", _NODE_UPDATE_SCHEMA, _node_update)
    ctx.register_tool("memory_node_delete", "memory", _NODE_DELETE_SCHEMA, _node_delete)
    ctx.register_tool("memory_edge_add", "memory", _EDGE_ADD_SCHEMA, _edge_add)
    ctx.register_tool("memory_edge_delete", "memory", _EDGE_DELETE_SCHEMA, _edge_delete)
    ctx.register_tool("memory_retrieve", "memory", _RETRIEVE_SCHEMA, _retrieve)
    ctx.register_tool("memory_stats", "memory", _STATS_SCHEMA, _stats)
    ctx.register_tool("memory_export", "memory", _EXPORT_SCHEMA, _export)
    ctx.register_tool("memory_import", "memory", _IMPORT_SCHEMA, _import_graph)
    ctx.register_tool("memory_prune", "memory", _PRUNE_SCHEMA, _prune)

    # ── Skill ─────────────────────────────────────────────────────────────
    skill_path = Path(__file__).parent.parent.parent / "skills" / "agentic-memory" / "SKILL.md"
    if skill_path.exists():
        try:
            ctx.register_skill(str(skill_path))
            logger.info("Registered agentic-memory skill from %s", skill_path)
        except Exception as e:
            logger.warning("Failed to register agentic-memory skill: %s", e)

    logger.info("agentic-memory-mcp plugin registered: 11 tools + 1 skill")
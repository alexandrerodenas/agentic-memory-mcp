"""MCP server exposing memory tools to AI agents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from agentic_memory.core.graph import KnowledgeGraph, Node, Edge
from agentic_memory.core.score import ScoreStore
from agentic_memory.core.prune import AutoPruner
from agentic_memory.api import MemorySkills


# ── Server setup ───────────────────────────────────────────────────────────────


app = Server("agentic-memory-mcp")


def _graph() -> KnowledgeGraph:
    path = Path(__import__("os").getenv("MEMORY_GRAPH_PATH", "memory_graph.json"))
    return KnowledgeGraph(path=path)


def _scores() -> ScoreStore:
    store = ScoreStore()
    path = Path(__import__("os").getenv("MEMORY_SCORES_PATH", "memory_scores.json"))
    if path.exists():
        store.load_dict(json.loads(path.read_text()))
    return store


def _save_scores(store: ScoreStore) -> None:
    path = Path(__import__("os").getenv("MEMORY_SCORES_PATH", "memory_scores.json"))
    path.write_text(json.dumps(store.to_dict()))


# ── Tool definitions ──────────────────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_node_add",
            description="Add a new fact or entity to the memory graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique identifier"},
                    "label": {"type": "string", "description": "Category label (e.g. Person, Fact, Preference)"},
                    "content": {"type": "string", "description": "The knowledge to store"},
                    "metadata": {"type": "object", "description": "Optional key-value metadata"},
                },
                "required": ["id", "content"],
            },
        ),
        Tool(
            name="memory_node_update",
            description="Update an existing memory entry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node identifier"},
                    "content": {"type": "string", "description": "New content"},
                    "label": {"type": "string", "description": "New label"},
                    "metadata": {"type": "object", "description": "Merge metadata"},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="memory_node_delete",
            description="Delete a memory entry and all its connections.",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Node identifier"}},
                "required": ["id"],
            },
        ),
        Tool(
            name="memory_node_get",
            description="Retrieve a specific memory entry by ID.",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Node identifier"}},
                "required": ["id"],
            },
        ),
        Tool(
            name="memory_search",
            description="Search memory for nodes matching a query and/or label.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search in content"},
                    "label": {"type": "string", "description": "Filter by label"},
                    "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                },
            },
        ),
        Tool(
            name="memory_retrieve",
            description="Retrieve top-N most relevant memories (token-optimized).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max memories to return", "default": 10},
                    "label": {"type": "string", "description": "Filter by label"},
                },
            },
        ),
        Tool(
            name="memory_edge_add",
            description="Create a relationship between two memory entries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique edge ID"},
                    "source": {"type": "string", "description": "Source node ID"},
                    "target": {"type": "string", "description": "Target node ID"},
                    "label": {"type": "string", "description": "Relationship type"},
                    "weight": {"type": "number", "description": "Edge weight", "default": 1.0},
                },
                "required": ["id", "source", "target"],
            },
        ),
        Tool(
            name="memory_corroborate",
            description="Reinforce a memory entry, increasing its corroboration score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node identifier to reinforce"},
                    "count": {"type": "integer", "description": "Corroboration count (default 1)"},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="memory_prune",
            description="Manually trigger pruning of old/irrelevant memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_nodes": {"type": "integer", "description": "Max nodes after pruning"},
                    "strategy": {"type": "string", "enum": ["oldest", "low_score"], "default": "oldest"},
                },
            },
        ),
        Tool(
            name="memory_stats",
            description="Get memory graph statistics.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    g = _graph()
    store = _scores()

    if name == "memory_node_add":
        node = Node(
            id=arguments["id"],
            label=arguments.get("label", ""),
            content=arguments["content"],
            metadata=arguments.get("metadata", {}),
        )
        g.add_node(node)
        return [TextContent(type="text", text=f"Added node '{node.id}'")]

    elif name == "memory_node_update":
        kwargs = {k: v for k, v in arguments.items() if k == "id" and k not in arguments}
        for key in ("content", "label"):
            if key in arguments:
                kwargs[key] = arguments[key]
        if "metadata" in arguments:
            existing = g.get_node(arguments["id"])
            if existing:
                merged = {**existing.metadata, **arguments["metadata"]}
                kwargs["metadata"] = merged
        g.update_node(arguments["id"], **kwargs)
        return [TextContent(type="text", text=f"Updated node '{arguments['id']}'")]

    elif name == "memory_node_delete":
        g.delete_node(arguments["id"])
        store.delete(arguments["id"])
        _save_scores(store)
        return [TextContent(type="text", text=f"Deleted node '{arguments['id']}'")]

    elif name == "memory_node_get":
        node = g.get_node(arguments["id"])
        if node is None:
            return [TextContent(type="text", text=f"Node '{arguments['id']}' not found")]
        store.get(node.id).bump_read()
        _save_scores(store)
        return [TextContent(type="text", text=str(node))]

    elif name == "memory_search":
        results = g.search_nodes(
            query=arguments.get("query", ""),
            label=arguments.get("label", ""),
            limit=arguments.get("limit", 10),
        )
        if not results:
            return [TextContent(type="text", text="No results")]
        for n in results:
            store.get(n.id).bump_read()
        _save_scores(store)
        lines = [f"[{n.label}] {n.id}: {n.content}" for n in results]
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "memory_retrieve":
        limit = arguments.get("limit", 10)
        label = arguments.get("label", "")
        top = store.top(g, label=label, limit=limit)
        lines = []
        for node_id, score_val in top:
            node = g.get_node(node_id)
            if node:
                lines.append(f"[score={score_val:.2f}] [{node.label}] {node.id}: {node.content}")
        if not lines:
            return [TextContent(type="text", text="No memories found")]
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "memory_edge_add":
        edge = Edge(
            id=arguments["id"],
            source=arguments["source"],
            target=arguments["target"],
            label=arguments.get("label", ""),
            weight=arguments.get("weight", 1.0),
        )
        g.add_edge(edge)
        return [TextContent(type="text", text=f"Added edge '{edge.id}'")]

    elif name == "memory_corroborate":
        node_id = arguments["id"]
        count = arguments.get("count", 1)
        if g.get_node(node_id) is None:
            return [TextContent(type="text", text=f"Node '{node_id}' not found")]
        store.get(node_id).bump_corroboration(count)
        _save_scores(store)
        return [TextContent(type="text", text=f"Corroborated '{node_id}' (count={count})")]

    elif name == "memory_prune":
        pruner = AutoPruner(max_nodes=arguments.get("max_nodes"))
        removed = pruner.auto_prune(g)
        return [TextContent(type="text", text=f"Pruned {len(removed)} nodes: {removed}")]

    elif name == "memory_stats":
        s = g.stats()
        return [TextContent(type="text", text=f"Nodes: {s['nodes']}, Edges: {s['edges']}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def run():
    """Entry point for the MCP server."""
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run()
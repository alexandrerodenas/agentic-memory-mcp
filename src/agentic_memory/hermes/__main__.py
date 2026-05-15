"""Hermes plugin — integrate agentic-memory-mcp into Hermes Agent."""
from __future__ import annotations

import os
from pathlib import Path

# The Hermes plugin entry point must expose a `setup` function that receives
# the Hermes Agent instance (or the agent's skill registry) and registers
# the memory commands / tools.


def setup(agent=None) -> None:
    """
    Called by Hermes Agent when the plugin is loaded.

    Registers the memory CLI commands and exposes the MemorySkills as a
    skill provider so agents can use the memory system natively.

    Environment variables:
      MEMORY_GRAPH_PATH   Path to the graph JSON (default: memory_graph.json)
      MEMORY_SCORES_PATH Path to the scores JSON (default: memory_scores.json)
    """
    graph_path = os.getenv("MEMORY_GRAPH_PATH", "memory_graph.json")
    scores_path = os.getenv("MEMORY_SCORES_PATH", "memory_scores.json")

    from agentic_memory.skills.memory import MemorySkills

    skills = MemorySkills(graph_path=graph_path, scores_path=scores_path)

    # Expose as a Hermes skill
    if agent is not None and hasattr(agent, "register_skill"):
        agent.register_skill("memory", _HermesMemorySkill(skills))
    else:
        # Auto-register via environment or default Hermes skill path
        _register_hermes_commands(skills)


class _HermesMemorySkill:
    """Wraps MemorySkills for Hermes Agent's skill system."""

    def __init__(self, skills) -> None:
        self.skills = skills

    def add(self, node_id: str, content: str, label: str = "", metadata: dict | None = None):
        return self.skills.add(node_id, content, label, metadata)

    def get(self, node_id: str):
        return self.skills.get(node_id)

    def retrieve(self, query: str = "", label: str = "", limit: int = 10):
        return self.skills.retrieve_text(query, label, limit)

    def corroborate(self, node_id: str, count: int = 1):
        return self.skills.corroborate(node_id, count)

    def delete(self, node_id: str):
        return self.skills.delete(node_id)

    def prune(self, max_nodes: int | None = None):
        return self.skills.prune(max_nodes=max_nodes)

    def stats(self):
        return self.skills.stats()


def _register_hermes_commands(skills) -> None:
    """Register Hermes CLI commands (called automatically)."""
    import click
    from rich.console import Console

    console = Console()

    @click.group("memory")
    def memory_group():
        """Agentic Memory — knowledge graph memory management."""
        pass

    @memory_group.command("add")
    @click.argument("node_id")
    @click.argument("content")
    @click.option("--label", default="", help="Node label")
    def memory_add(node_id, content, label):
        node = skills.add(node_id, content, label)
        console.print(f"[green]+ {node.id}[/green]")

    @memory_group.command("get")
    @click.argument("node_id")
    def memory_get(node_id):
        node = skills.get(node_id)
        if node:
            console.print(f"[bold]{node.label}:[/bold] {node.content}")
        else:
            console.print(f"[red]Not found: {node_id}[/red]")

    @memory_group.command("retrieve")
    @click.option("--query", default="", help="Text to filter content")
    @click.option("--limit", default=10, type=int)
    @click.option("--label", default="", help="Filter by label")
    def memory_retrieve(query, limit, label):
        text = skills.retrieve_text(query=query, limit=limit, label=label)
        console.print(text)

    @memory_group.command("corroborate")
    @click.argument("node_id")
    @click.option("--count", default=1, type=int)
    def memory_corroborate(node_id, count):
        skills.corroborate(node_id, count)
        console.print(f"[green]Corroborated {node_id} (count={count})[/green]")

    @memory_group.command("prune")
    @click.option("--max-nodes", default=None, type=int)
    def memory_prune(max_nodes):
        removed = skills.prune(max_nodes=max_nodes)
        console.print(f"[yellow]Pruned: {removed}[/yellow]")

    @memory_group.command("stats")
    def memory_stats():
        s = skills.stats()
        console.print(f"Nodes: {s['nodes']}, Edges: {s['edges']}, Scores: {s.get('score_entries', 0)}")

    # Register globally
    import sys
    if not hasattr(sys, "_hermes_memory_registered"):
        sys._hermes_memory_registered = True
        try:
            from agentic_memory.cli.main import cli
            # Add the memory group as a subcommand
            cli.add_command(memory_group, name="memory")
        except Exception:
            pass


__all__ = ["setup"]
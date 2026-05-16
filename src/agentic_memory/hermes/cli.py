"""CLI commands for the agentic-memory Hermes plugin.

Registers ``hermes agentic-memory <subcommand>`` commands.
Only active when this provider is the configured memory.provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _get_graph_dir(args) -> Path:
    """Resolve the graph directory from args or hermes_home."""
    hermes_home = getattr(args, "hermes_home", "") or ""
    if hermes_home:
        return Path(hermes_home) / "memory_graph"
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "memory_graph"


def _get_skills(args):
    """Create a MemorySkills instance with profile-scoped paths."""
    from agentic_memory.api import MemorySkills

    graph_dir = _get_graph_dir(args)
    return MemorySkills(
        graph_path=str(graph_dir / "memory_graph.json"),
        scores_path=str(graph_dir / "memory_scores.json"),
    )


def _cmd_status(args) -> None:
    """Show provider status and graph statistics."""
    skills = _get_skills(args)
    s = skills.stats()
    print(f"Provider: agentic-memory")
    print(f"Graph dir: {_get_graph_dir(args)}")
    print(f"Nodes: {s['nodes']}, Edges: {s['edges']}, Score entries: {s.get('score_entries', 0)}")


def _cmd_stats(args) -> None:
    """Show detailed graph statistics."""
    skills = _get_skills(args)
    s = skills.stats()
    for k, v in s.items():
        print(f"  {k}: {v}")


def _cmd_export(args) -> None:
    """Export the memory graph to JSON."""
    skills = _get_skills(args)
    data = skills.export_json()
    out = args.output or str(_get_graph_dir(args) / "export.json")
    Path(out).write_text(data)
    print(f"Exported to {out}")


def _cmd_import(args) -> None:
    """Import a memory graph from JSON."""
    skills = _get_skills(args)
    data = Path(args.path).read_text()
    skills.import_json(data)
    print(f"Imported from {args.path}")


def _cmd_prune(args) -> None:
    """Prune old or low-value memories."""
    skills = _get_skills(args)
    removed = skills.prune(max_nodes=args.max_nodes, strategy=args.strategy)
    print(f"Pruned {len(removed)} nodes: {removed}")


def _cmd_retrieve(args) -> None:
    """Retrieve top memories by score."""
    skills = _get_skills(args)
    text = skills.retrieve_text(query=args.query or "", label=args.label or "", limit=args.limit)
    if text and text != "No relevant memories found.":
        print(text)
    else:
        print("No memories found.")


def _dispatch(args) -> None:
    sub = getattr(args, "agentic_memory_command", None)
    if sub == "status":
        _cmd_status(args)
    elif sub == "stats":
        _cmd_stats(args)
    elif sub == "export":
        _cmd_export(args)
    elif sub == "import":
        _cmd_import(args)
    elif sub == "prune":
        _cmd_prune(args)
    elif sub == "retrieve":
        _cmd_retrieve(args)
    else:
        print("Usage: hermes agentic-memory <status|stats|export|import|prune|retrieve>")


def register_cli(subparser) -> None:
    """Build the ``hermes agentic-memory`` argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="agentic_memory_command")

    subs.add_parser("status", help="Show provider status and graph statistics")
    subs.add_parser("stats", help="Show detailed graph statistics")

    p_export = subs.add_parser("export", help="Export memory graph to JSON")
    p_export.add_argument("output", nargs="?", default="", help="Output file path")

    p_import = subs.add_parser("import", help="Import memory graph from JSON")
    p_import.add_argument("path", help="Path to JSON export file")

    p_prune = subs.add_parser("prune", help="Prune old or low-value memories")
    p_prune.add_argument("--max-nodes", type=int, default=10, help="Max nodes to remove")
    p_prune.add_argument("--strategy", choices=["oldest", "low_score"], default="low_score")

    p_retrieve = subs.add_parser("retrieve", help="Retrieve top memories by score")
    p_retrieve.add_argument("--query", default="", help="Text filter")
    p_retrieve.add_argument("--label", default="", help="Label filter")
    p_retrieve.add_argument("--limit", type=int, default=10, help="Max results")

    subparser.set_defaults(func=_dispatch)

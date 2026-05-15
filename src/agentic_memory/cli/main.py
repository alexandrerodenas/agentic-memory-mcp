"""CLI interface for the memory system."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agentic_memory.core.graph import KnowledgeGraph, Node, Edge
from agentic_memory.core.prune import AutoPruner


console = Console()


@click.group()
@click.option(
    "--graph",
    default="memory_graph.json",
    help="Path to the graph JSON file.",
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.pass_context
def cli(ctx, graph):
    """Agentic Memory CLI — manage your knowledge graph."""
    ctx.ensure_object(dict)
    ctx.obj["graph_path"] = Path(graph)


# ══════════════════════════════════════════════════════════════════════════════
# Node commands
# ══════════════════════════════════════════════════════════════════════════════


@cli.group("node")
def node():
    """Node operations."""


@node.command("add")
@click.option("--id", required=True, help="Unique node ID")
@click.option("--label", default="", help="Node label (e.g. Person, Fact)")
@click.option("--content", default="", help="Node content")
@click.pass_context
def node_add(ctx, id, label, content):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    node = Node(id=id, label=label, content=content)
    try:
        g.add_node(node)
        console.print(f"[green]+ Node '{id}' added[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@node.command("list")
@click.pass_context
def node_list(ctx):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    nodes = g.list_nodes()
    table = Table(title="Nodes")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Content")
    for n in nodes:
        table.add_row(n.id, n.label, n.content[:60])
    console.print(table)


@node.command("get")
@click.argument("node_id")
@click.pass_context
def node_get(ctx, node_id):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    node = g.get_node(node_id)
    if node is None:
        console.print(f"[red]Node '{node_id}' not found[/red]")
        sys.exit(1)
    console.print(f"[bold]ID:[/bold] {node.id}")
    console.print(f"[bold]Label:[/bold] {node.label}")
    console.print(f"[bold]Content:[/bold] {node.content}")
    console.print(f"[bold]Created:[/bold] {node.created_at}")
    console.print(f"[bold]Updated:[/bold] {node.updated_at}")
    if node.metadata:
        console.print(f"[bold]Metadata:[/bold] {node.metadata}")


@node.command("update")
@click.argument("node_id")
@click.option("--content", help="New content")
@click.option("--label", help="New label")
@click.pass_context
def node_update(ctx, node_id, content, label):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    kwargs = {}
    if content is not None:
        kwargs["content"] = content
    if label is not None:
        kwargs["label"] = label
    try:
        g.update_node(node_id, **kwargs)
        console.print(f"[green]~ Node '{node_id}' updated[/green]")
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@node.command("delete")
@click.argument("node_id")
@click.pass_context
def node_delete(ctx, node_id):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    try:
        g.delete_node(node_id)
        console.print(f"[red]- Node '{node_id}' deleted[/red]")
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@node.command("search")
@click.argument("query", required=False)
@click.option("--label", default="", help="Filter by label")
@click.pass_context
def node_search(ctx, query, label):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    results = g.search_nodes(query or "", label or "")
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    table = Table(title=f"Search results ({len(results)})")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Content")
    for n in results:
        table.add_row(n.id, n.label, n.content[:80])
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# Edge commands
# ══════════════════════════════════════════════════════════════════════════════


@cli.group("edge")
def edge():
    """Edge operations."""


@edge.command("add")
@click.option("--id", required=True, help="Unique edge ID")
@click.option("--source", required=True, help="Source node ID")
@click.option("--target", required=True, help="Target node ID")
@click.option("--label", default="", help="Relationship label")
@click.option("--weight", default=1.0, type=float, help="Edge weight")
@click.pass_context
def edge_add(ctx, id, source, target, label, weight):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    e = Edge(id=id, source=source, target=target, label=label, weight=weight)
    try:
        g.add_edge(e)
        console.print(f"[green]+ Edge '{id}' added[/green]")
    except (ValueError, KeyError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@edge.command("list")
@click.pass_context
def edge_list(ctx):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    edges = g.list_edges()
    table = Table(title="Edges")
    table.add_column("ID")
    table.add_column("Source")
    table.add_column("Target")
    table.add_column("Label")
    table.add_column("Weight")
    for e in edges:
        table.add_row(e.id, e.source, e.target, e.label, str(e.weight))
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# Utility commands
# ══════════════════════════════════════════════════════════════════════════════


@cli.command("stats")
@click.pass_context
def stats(ctx):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    s = g.stats()
    console.print(f"[bold]Nodes:[/bold] {s['nodes']}")
    console.print(f"[bold]Edges:[/bold] {s['edges']}")


@cli.command("export")
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.pass_context
def export(ctx, output):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    Path(output).write_text(g.export_json())
    console.print(f"[green]Exported to {output}[/green]")


@cli.command("import")
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.pass_context
def import_cmd(ctx, input):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    g.import_json(Path(input).read_text())
    console.print(f"[green]Imported from {input}[/green]")


@cli.command("prune")
@click.option("--max-nodes", type=int, default=None, help="Maximum nodes before pruning")
@click.option("--strategy", default="oldest", type=click.Choice(["oldest", "low_score"]))
@click.pass_context
def prune(ctx, max_nodes, strategy):
    g = KnowledgeGraph(path=ctx.obj["graph_path"])
    pruner = AutoPruner(max_nodes=max_nodes)
    removed = pruner.prune(g, strategy=strategy)
    console.print(f"[yellow]Removed {len(removed)} nodes: {removed}[/yellow]")


@cli.command("retrieve")
@click.option("--limit", default=10, type=int, help="Max results")
@click.option("--label", default="", help="Filter by label")
@click.pass_context
def retrieve(ctx, limit, label):
    """Retrieve top-N most relevant memories by score."""
    skills = MemorySkills(graph_path=ctx.obj["graph_path"])
    text = skills.retrieve_text(limit=limit, label=label)
    if not text:
        console.print("[yellow]No memories found[/yellow]")
        return
    console.print(text)


if __name__ == "__main__":
    cli()
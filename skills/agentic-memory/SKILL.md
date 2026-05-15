---
name: agentic-memory
description: Persistent, scored Knowledge Graph memory for AI agents — CLI usage, Python API integration, and MCP server setup.
version: 0.1.0
author: Alexandre Rodenas
license: MIT
tags: [memory, knowledge-graph, mcp, llm, agentic]
metadata:
  hermes:
    category: memory
---

# Agentic Memory — Knowledge Graph for AI Agents

Persistent memory with a scoring system. Stores facts, entities, and relationships in a JSON-based Knowledge Graph. Designed for standalone Python use, MCP tool serving, or Hermes Agent plugin integration.

## Quick Start

### CLI

```bash
# Add a fact
memory-cli node add --id alice --label Person --content "Alice works at Acme Corp"

# Search
memory-cli node search "Acme"

# Link two facts
memory-cli edge add --id e1 --source alice --target acme --label works_at

# Retrieve top-N most relevant memories (scored)
memory-cli retrieve --limit 10

# Prune old entries
memory-cli prune --max-nodes 500 --strategy oldest

# Stats
memory-cli stats
```

### Python API (standalone — no server needed)

```python
from agentic_memory.api import MemorySkills

mem = MemorySkills(graph_path="memory.json")

# Store
mem.add("alice", "Alice works at Acme Corp", label="Person")

# Search
results = mem.search(query="Acme", label="Person")

# Token-optimized retrieval (top-N by score)
context = mem.retrieve_text(limit=10)

# Reinforce (increases score)
mem.corroborate("alice", count=1)
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMORY_GRAPH_PATH` | `memory_graph.json` | Graph data file |
| `MEMORY_SCORES_PATH` | `memory_scores.json` | Score data file |

## CLI Commands Reference

### Node Operations

```
memory-cli node add --id <id> --label <label> --content <content>
memory-cli node list
memory-cli node get <node_id>
memory-cli node update <node_id> --content <new> --label <new>
memory-cli node delete <node_id>
memory-cli node search <query> [--label <label>]
```

### Edge Operations

```
memory-cli edge add --id <id> --source <source> --target <target> --label <rel>
memory-cli edge list
```

### Utility

```
memory-cli stats
memory-cli export <output.json>
memory-cli import <input.json>
memory-cli prune --max-nodes <N> --strategy oldest|low_score
```

### Options

```
--graph <path>    Custom graph file path (default: memory_graph.json)
```

## Scoring System

Every node gets a composite relevance score:

```
score = (3 × corroborations) + (1 × read_count) + (0.5 × recency_bonus)
```

- **Corroborations** — bump when the same fact is confirmed again
- **Read count** — increments each time the node is retrieved
- **Recency bonus** — decays over time (0→1, higher = more recent)

`memory_retrieve` returns the top-N nodes sorted by this score — ideal for injecting relevant context into an LLM prompt without blowing up the context window.

## MCP Tools

When running `memory-mcp` (stdio transport), agents can call:

| Tool | Description |
|---|---|
| `memory_node_add` | Add a fact/entity |
| `memory_node_update` | Update content/label/metadata |
| `memory_node_delete` | Delete and cascade edges |
| `memory_node_get` | Retrieve by ID (bump read score) |
| `memory_search` | Full-text + label search |
| `memory_retrieve` | Top-N token-optimized by score |
| `memory_edge_add` | Create a relationship |
| `memory_corroborate` | Reinforce (bump score) |
| `memory_prune` | Manual pruning |
| `memory_stats` | Graph statistics |

## Auto-Pruning

Pruning triggers automatically when limits are exceeded:

```python
from agentic_memory.core.prune import AutoPruner

pruner = AutoPruner(max_nodes=1000, max_size_mb=10.0)
removed = pruner.auto_prune(graph)
```

Or via CLI:
```bash
memory-cli prune --max-nodes 500 --strategy oldest
```

## Hermes Plugin

```python
from agentic_memory.hermes import setup

setup(agent=your_hermes_instance)
```

This exposes the memory system as a native Hermes skill with `add`, `get`, `search`, `retrieve`, `corroborate`, `delete`, `prune`, `stats` commands.
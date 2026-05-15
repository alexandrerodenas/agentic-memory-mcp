# agentic-memory-mcp

> Agentic Memory Management via MCP — Knowledge Graph for AI agents.

[![PyPI version](https://img.shields.io/pypi/v/agentic-memory-mcp.svg)](https://pypi.org/project/agentic-memory-mcp/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/alexandrerodenas/agentic-memory-mcp/test.yml?branch=main)](https://github.com/alexandrerodenas/agentic-memory-mcp/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

A **Model Context Protocol (MCP)** server that gives AI agents persistent, structured, and queryable memory via a Knowledge Graph. Designed to be used standalone, embedded in any agentic framework, or served as an MCP tool server.

## Architecture

```
agentic_memory/
├── core/
│   ├── graph.py      # KnowledgeGraph — nodes, edges, JSON persistence
│   ├── score.py      # Score & ScoreStore — reputation / relevance scoring
│   └── prune.py      # AutoPruner — maintenance & auto-pruning
├── mcp/
│   └── server.py     # MCP server (stdio transport)
├── cli/
│   └── main.py       # Click CLI (memory-cli command)
├── skills/
│   └── memory.py     # MemorySkills — decoupled Python API
└── hermes.py         # Hermes Agent plugin integration
```

**Persistence:** JSON files (`memory_graph.json`, `memory_scores.json`) — no external database required.

## Installation

```bash
# From PyPI
pip install agentic-memory-mcp

# Development
git clone https://github.com/alexandrerodenas/agentic-memory-mcp.git
cd agentic-memory-mcp
uv sync
```

## Quick Start

### CLI

```bash
# Add a memory entry
memory-cli --graph memory.json node add --id alice --label Person --content "Alice lives in Paris"

# Search
memory-cli --graph memory.json node search "Paris"

# Link two entries
memory-cli --graph memory.json edge add --id e1 --source alice --target paris --label lives_in

# Get stats
memory-cli --graph memory.json stats

# Prune old entries
memory-cli --graph memory.json prune --max-nodes 500 --strategy oldest
```

### Python Skills (no server needed)

```python
from agentic_memory.skills.memory import MemorySkills

mem = MemorySkills(graph_path="memory.json", scores_path="scores.json")

# Store knowledge
mem.add("alice", "Alice works at Acme Corp", label="Person")

# Search
results = mem.search(query="Acme", label="Person")

# Token-optimized retrieval (top-N by score)
context = mem.retrieve_text(limit=10)

# Reinforce a memory
mem.corroborate("alice", count=1)
```

### MCP Server

```bash
# Start the MCP server (stdio)
memory-mcp

# Or with custom paths
MEMORY_GRAPH_PATH=/data/memory.json MEMORY_SCORES_PATH=/data/scores.json memory-mcp
```

Then connect any MCP-compatible AI agent (Claude Code, OpenCode, etc.) to use the tools:
- `memory_node_add` — Add a fact or entity
- `memory_node_update` — Update an entry
- `memory_node_delete` — Delete an entry
- `memory_node_get` — Retrieve by ID
- `memory_search` — Full-text + label search
- `memory_retrieve` — Token-optimized top-N retrieval by score
- `memory_edge_add` — Create a relationship
- `memory_corroborate` — Reinforce a memory (increases score)
- `memory_prune` — Manual pruning
- `memory_stats` — Graph statistics

### Hermes Agent Plugin

```bash
# Install
hermes plugin install agentic-memory-mcp

# Use in Hermes
/memory add alice "Alice works at Acme Corp"
/memory search Acme
/memory retrieve --limit 10
/memory corroborate alice
/memory prune --max-nodes 500
```

Or import in your own agent:

```python
from agentic_memory.hermes import setup

setup(agent=your_agent_instance)
```

## Scoring System

Each memory entry gets a composite relevance score:

```
score = (3 × corroborations) + (1 × read_count) + (0.5 × recency_bonus)
```

- **Corroborations** increase when the same fact is reinforced (e.g., mentioned again by the user or confirmed by another source).
- **Read count** increases each time the memory is retrieved.
- **Recency bonus** decays over time — recently accessed memories score higher.

This ensures the most relevant, most frequently accessed, and most corroborated memories bubble up to the top during retrieval, keeping LLM context windows tight.

## Auto-Pruning

```python
from agentic_memory.core.prune import AutoPruner

pruner = AutoPruner(max_nodes=1000, max_size_mb=10.0)
removed = pruner.auto_prune(graph)
```

Triggers:
- **Max nodes** — prune oldest entries when graph exceeds the limit
- **Max size (MB)** — prune when the JSON file exceeds the limit
- **Manual** — via CLI or `memory_prune` MCP tool

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMORY_GRAPH_PATH` | `memory_graph.json` | Path to graph JSON file |
| `MEMORY_SCORES_PATH` | `memory_scores.json` | Path to scores JSON file |

## Development

```bash
# Install dev deps
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/
```

## License

MIT — Alexandre Rodenas
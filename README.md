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
├── api.py            # MemorySkills — Python API (standalone use)
└── hermes/           # Hermes Agent plugin integration
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
memory-cli retrieve --query "Paris"

# Link two entries
memory-cli --graph memory.json edge add --id e1 --source alice --target paris --label lives_in

# Get stats
memory-cli --graph memory.json stats

# Prune old entries
memory-cli --graph memory.json prune --max-nodes 500 --strategy oldest
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
- `memory_retrieve` — Token-optimized top-N retrieval by score (with optional text filter)
- `memory_edge_add` — Create a relationship
- `memory_corroborate` — Reinforce a memory (increases score)
- `memory_prune` — Manual pruning
- `memory_stats` — Graph statistics

### Hermes Agent Plugin

```
# Install
hermes plugin install agentic-memory-mcp

# Use in Hermes
/memory add alice "Alice works at Acme Corp"
/memory retrieve --query Acme --limit 10
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

### Auto-Pruning

Auto-prune runs as a background scheduler inside the MCP server process, controlled entirely via environment variables:

```bash
# Activate auto-prune: max 1000 nodes, prune when file > 5 MB, every 30 min
MEMORY_AUTO_PRUNE_ENABLED=1 \
MEMORY_AUTO_PRUNE_MAX_NODES=1000 \
MEMORY_AUTO_PRUNE_MAX_SIZE_MB=5 \
MEMORY_AUTO_PRUNE_INTERVAL_SECONDS=1800 \
  memory-mcp
```

| Variable | Default | Description |
|---|---|---|
| `MEMORY_AUTO_PRUNE_ENABLED` | `0` | Set to `1` to enable the background scheduler |
| `MEMORY_AUTO_PRUNE_MAX_NODES` | `None` | Max nodes before pruning |
| `MEMORY_AUTO_PRUNE_MAX_SIZE_MB` | `None` | Max file size (MB) before pruning |
| `MEMORY_AUTO_PRUNE_INTERVAL_SECONDS` | `3600` | Seconds between auto-prune runs |

Manual prune is also available via CLI or `memory_prune` tool.

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
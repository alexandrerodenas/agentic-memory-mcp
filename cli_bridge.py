"""CLI bridge — re-exports register_cli for Hermes plugin discovery.

This file must be at the plugin root directory for discover_plugin_cli_commands()
to find it. It re-exports from the actual implementation in hermes/cli.py.
"""

from .src.agentic_memory.hermes.cli import register_cli

__all__ = ["register_cli"]

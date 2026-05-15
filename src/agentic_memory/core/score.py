"""Reputation and scoring system for memory entries."""
from __future__ import annotations

import time
from datetime import datetime

from pydantic import BaseModel, Field


class Score(BaseModel):
    """Reputation metrics for a single node."""

    read_count: int = 0
    corroborations: int = 0
    last_accessed: float = Field(default_factory=time.time)
    created_at: float = Field(default_factory=time.time)

    # Configurable weights for the scoring formula
    weight_read: float = 1.0
    weight_corroboration: float = 3.0
    weight_recency: float = 0.5

    def bump_read(self) -> None:
        self.read_count += 1
        self.last_accessed = time.time()

    def bump_corroboration(self, count: int = 1) -> None:
        self.corroborations += count

    def score(self) -> float:
        """Compute a composite relevance score."""
        recency = 1.0 / (1.0 + (time.time() - self.last_accessed) / 86400)
        return (
            self.weight_read * self.read_count
            + self.weight_corroboration * self.corroborations
            + self.weight_recency * recency
        )

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> Score:
        return cls.model_validate(data)


class ScoreStore:
    """In-memory store for node scores (backed by the graph persistence)."""

    def __init__(self) -> None:
        self._scores: dict[str, Score] = {}

    def get(self, node_id: str) -> Score:
        if node_id not in self._scores:
            self._scores[node_id] = Score()
        return self._scores[node_id]

    def set(self, node_id: str, score: Score) -> None:
        self._scores[node_id] = score

    def delete(self, node_id: str) -> None:
        self._scores.pop(node_id, None)

    def top(self, graph, label: str = "", limit: int = 10) -> list[tuple[str, float]]:
        """Return top-N nodes by score, optionally filtered by label."""
        nodes = graph.list_nodes()
        if label:
            nodes = [n for n in nodes if n.label == label]
        scored = [(n.id, self.get(n.id).score()) for n in nodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def to_dict(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in self._scores.items()}

    def load_dict(self, data: dict[str, dict]) -> None:
        self._scores = {k: Score.from_dict(v) for k, v in data.items()}
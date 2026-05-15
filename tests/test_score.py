"""Tests for the scoring system."""
from __future__ import annotations

import pytest
import time

from agentic_memory.core.score import Score, ScoreStore


class TestScore:
    def test_score_initial(self):
        s = Score()
        assert s.read_count == 0
        assert s.corroborations == 0

    def test_bump_read(self):
        s = Score()
        s.bump_read()
        assert s.read_count == 1
        s.bump_read()
        assert s.read_count == 2

    def test_bump_corroboration(self):
        s = Score()
        s.bump_corroboration()
        assert s.corroborations == 1
        s.bump_corroboration(3)
        assert s.corroborations == 4

    def test_score_formula(self):
        s = Score()
        # No reads, no corroborations -> score ~ recency only
        assert s.score() > 0

    def test_score_favors_reads(self):
        s = Score(read_count=10, corroborations=0)
        t = Score(read_count=0, corroborations=10)
        # Corroborations are weighted more (3x by default)
        assert t.score() > s.score()

    def test_score_favors_recent(self):
        s = Score()
        old = time.time() - 86400 * 30
        s.last_accessed = old
        recent = Score()
        recent.last_accessed = time.time()
        assert recent.score() > s.score()

    def test_roundtrip(self):
        s = Score(read_count=5, corroborations=3)
        d = s.to_dict()
        restored = Score.from_dict(d)
        assert restored.read_count == 5
        assert restored.corroborations == 3


class TestScoreStore:
    def test_get_default(self):
        store = ScoreStore()
        s = store.get("unknown")
        assert isinstance(s, Score)
        assert s.read_count == 0

    def test_set_and_get(self):
        store = ScoreStore()
        s = Score(read_count=7)
        store.set("node-x", s)
        assert store.get("node-x").read_count == 7

    def test_delete(self):
        store = ScoreStore()
        store.set("n1", Score())
        store.delete("n1")
        assert store.get("n1").read_count == 0  # returns default

    def test_to_dict_from_dict(self):
        store = ScoreStore()
        store.set("n1", Score(read_count=5))
        d = store.to_dict()
        assert "n1" in d
        new_store = ScoreStore()
        new_store.load_dict(d)
        assert new_store.get("n1").read_count == 5
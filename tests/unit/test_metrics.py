"""
Unit tests for IR evaluation metrics.

All tests use hand-crafted inputs with known expected outputs.
"""

import math

import pytest

from app.evaluation.metrics import (
    average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_zero_precision(self) -> None:
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_precision(self) -> None:
        result = precision_at_k(["a", "x", "b"], ["a", "b"], k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_k_larger_than_retrieved(self) -> None:
        result = precision_at_k(["a"], ["a", "b"], k=5)
        assert result == 1 / 5


class TestRecallAtK:
    def test_full_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=5) == 1.0

    def test_zero_recall(self) -> None:
        assert recall_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], [], k=5) == 0.0


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_first_hit_at_rank_2(self) -> None:
        assert mrr(["x", "a", "b"], ["a"]) == 0.5

    def test_no_hit(self) -> None:
        assert mrr(["x", "y", "z"], ["a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        score = ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3)
        assert abs(score - 1.0) < 1e-9

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], [], k=5) == 0.0

    def test_zero_k(self) -> None:
        assert ndcg_at_k(["a"], ["a"], k=0) == 0.0

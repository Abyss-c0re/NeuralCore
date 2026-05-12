"""Unit tests for neuralcore.utils.search -- scoring and similarity functions."""

import sys
import numpy as np
from pathlib import Path
from neuralcore.utils.search import keyword_score, fuzzy_score, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestKeywordScore:
    def test_perfect_match(self):
        score = keyword_score(["hello", "world"], "hello world test")
        assert score > 0.5

    def test_no_match(self):
        score = keyword_score(["xyz", "abc"], "hello world test")
        assert score == 0.0

    def test_partial_match(self):
        s1 = keyword_score(["hello"], "hello world")
        s2 = keyword_score(["hello", "world"], "hello world")
        assert s2 >= s1

    def test_empty_query(self):
        score = keyword_score([], "hello world")
        assert score == 0.0


class TestFuzzyScore:
    def test_exact_match(self):
        score = fuzzy_score("hello world", "hello world")
        assert score > 0.9

    def test_similar_match(self):
        score = fuzzy_score("read file", "read_file tool for reading files")
        assert score > 0.3

    def test_no_match(self):
        score = fuzzy_score("xyz abc", "hello world test")
        assert score < 0.5


class TestCosineSim:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_sim(v, v)
        assert abs(sim - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_sim(v1, v2)
        assert abs(sim) < 0.001

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        sim = cosine_sim(v1, v2)
        assert sim < -0.9

    def test_empty_vector(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        sim = cosine_sim(v1, v2)
        assert sim == 0.0 or abs(sim) < 0.001

"""Tests for chronogrammer.search and chronogrammer.semantic."""

import pytest

from chronogrammer.scorer import chronogram_score
from chronogrammer.search import (
    ObjectiveWeights,
    SearchResult,
    beam_search,
    objective_score,
)
from chronogrammer.semantic import JaccardSimilarity, jaccard_similarity, tokenize


class TestTokenize:
    def test_empty(self):
        assert tokenize("") == []

    def test_simple(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_case_lowered(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_remove_stopwords(self):
        tokens = tokenize("the cat sat on the mat", remove_stopwords=True)
        assert "the" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens

    def test_keep_stopwords(self):
        tokens = tokenize("the cat", remove_stopwords=False)
        assert "the" in tokens


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        sim = jaccard_similarity("cat sat mat", "dog barked loudly")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the cat sat on the mat", "a cat sat")
        assert 0.0 < sim <= 1.0

    def test_empty_both(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == 0.0
        assert jaccard_similarity("", "world") == 0.0

    def test_symmetry(self):
        a = "the quick brown fox"
        b = "a slow red dog"
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)

    def test_value_range(self):
        sim = jaccard_similarity("foo bar baz", "bar baz qux")
        assert 0.0 <= sim <= 1.0

    def test_stopword_removal_effect(self):
        # With stopword removal, sentences that differ only in stopwords should
        # score high
        sim_with = jaccard_similarity("the cat", "a cat", remove_stopwords=True)
        sim_without = jaccard_similarity("the cat", "a cat", remove_stopwords=False)
        assert sim_with >= sim_without


class TestJaccardSimilarityClass:
    def test_callable(self):
        sim = JaccardSimilarity()
        result = sim("hello world", "hello world")
        assert result == 1.0

    def test_different(self):
        sim = JaccardSimilarity()
        result = sim("cat sat", "dog ran")
        assert result == 0.0

    def test_with_stopwords_disabled(self):
        sim = JaccardSimilarity(remove_stopwords=False)
        # "the cat" vs "a cat" → {the, cat} vs {a, cat} → overlap {cat} → 1/3
        result = sim("the cat", "a cat")
        assert result == pytest.approx(1 / 3, abs=1e-9)


class TestObjectiveScore:
    def test_zero_error_reduces_score(self):
        # A candidate that hits target exactly should score better (lower) than one that doesn't
        source = "The city marks its history."
        target = chronogram_score(source)
        # Same text → error = 0
        score_exact = objective_score(source, source, target)
        # Modified text with same score but worse semantics
        score_off = objective_score(source, "MMMM dragons breathe fire.", target + 100)
        assert score_exact < score_off

    def test_semantic_drift_penalised(self):
        source = "The cat sat on the mat."
        target = chronogram_score(source)
        # Candidate with same score but very different semantics
        candidate_similar = "The cat rested on the mat."
        candidate_different = "Xylophone music vibrated."
        sim = JaccardSimilarity()
        score_similar = objective_score(source, candidate_similar, target, similarity=sim)
        score_different = objective_score(source, candidate_different, target, similarity=sim)
        assert score_similar < score_different

    def test_weights_affect_score(self):
        source = "hello world"
        candidate = "goodbye world"
        target = chronogram_score(source)
        w_high_sem = ObjectiveWeights(chronogram_error=10.0, semantic_drift=100.0, length_penalty=0.0)
        w_low_sem = ObjectiveWeights(chronogram_error=10.0, semantic_drift=0.0, length_penalty=0.0)
        sim = JaccardSimilarity()
        score_high = objective_score(source, candidate, target, weights=w_high_sem, similarity=sim)
        score_low = objective_score(source, candidate, target, weights=w_low_sem, similarity=sim)
        # Higher semantic weight → higher penalty for different candidates
        assert score_high >= score_low

    def test_length_penalty(self):
        source = "The cat sat."
        target = chronogram_score(source)
        sim = JaccardSimilarity()
        w_with_len = ObjectiveWeights(chronogram_error=0.0, semantic_drift=0.0, length_penalty=1.0)
        same_len = "The dog ran."  # same word count
        longer = "The big fluffy dog ran very fast."
        score_same = objective_score(source, same_len, target, weights=w_with_len, similarity=sim)
        score_longer = objective_score(source, longer, target, weights=w_with_len, similarity=sim)
        assert score_same < score_longer

    def test_returns_float(self):
        result = objective_score("hello", "hello", 0)
        assert isinstance(result, float)

    def test_non_negative(self):
        result = objective_score("any sentence", "any sentence", 999)
        assert result >= 0.0


class TestObjectiveWeights:
    def test_defaults(self):
        w = ObjectiveWeights()
        assert w.chronogram_error == 10.0
        assert w.semantic_drift == 3.0
        assert w.length_penalty == 0.2

    def test_custom(self):
        w = ObjectiveWeights(chronogram_error=1.0, semantic_drift=2.0, length_penalty=3.0)
        assert w.chronogram_error == 1.0
        assert w.semantic_drift == 2.0
        assert w.length_penalty == 3.0

    def test_frozen(self):
        w = ObjectiveWeights()
        with pytest.raises((AttributeError, TypeError)):
            w.chronogram_error = 99.0  # type: ignore[misc]


class TestBeamSearch:
    def test_returns_search_result(self):
        result = beam_search("hello world", 100, steps=1)
        assert isinstance(result, SearchResult)

    def test_result_fields(self):
        result = beam_search("hello world", 50, steps=2)
        assert isinstance(result.best, str)
        assert isinstance(result.chronogram_score, int)
        assert result.target == 50
        assert isinstance(result.error, int)
        assert isinstance(result.objective, float)
        assert isinstance(result.candidates, list)

    def test_error_is_non_negative(self):
        result = beam_search("the cat sat on the mat", 200, steps=2)
        assert result.error >= 0

    def test_error_consistent(self):
        result = beam_search("the city honors its founders", 300, steps=2)
        assert result.error == abs(chronogram_score(result.best) - result.target)

    def test_candidates_sorted_best_first(self):
        result = beam_search("hello world", 100, steps=2)
        if len(result.candidates) > 1:
            scores = [
                objective_score("hello world", c, 100) for c in result.candidates
            ]
            # Best candidate should have lower or equal score than the rest
            assert scores[0] <= scores[-1]

    def test_source_sentence_is_valid_candidate(self):
        # The search should at least return something at least as good as source
        source = "the city honors its founders"
        result = beam_search(source, chronogram_score(source), steps=1)
        # If target equals source score, error should be 0
        assert result.error == 0

    def test_deterministic_output(self):
        # Two runs with same input should give same best result
        source = "The city honors its founders with a public ceremony."
        result1 = beam_search(source, 1776, steps=3, beam_width=10)
        result2 = beam_search(source, 1776, steps=3, beam_width=10)
        assert result1.best == result2.best

    def test_beam_width_respected(self):
        result = beam_search("hello world", 100, steps=2, beam_width=5)
        assert len(result.candidates) <= 5

    def test_small_target_reachable(self):
        # "I" = 1, target = 1 → source already satisfies
        result = beam_search("I", 1, steps=1)
        assert result.error == 0

    def test_search_result_dataclass(self):
        result = SearchResult(
            best="hello",
            chronogram_score=5,
            target=5,
            error=0,
            objective=0.0,
            candidates=["hello"],
        )
        assert result.best == "hello"
        assert result.error == 0

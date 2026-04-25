"""Tests for chronogrammer.generator."""

import pytest

from chronogrammer.generator import DeterministicGenerator, _SYNONYM_BANK
from chronogrammer.scorer import chronogram_score


class TestDeterministicGenerator:
    def setup_method(self):
        self.gen = DeterministicGenerator()

    def test_returns_list(self):
        result = self.gen.generate("The city honors its founders.", 1776)
        assert isinstance(result, list)

    def test_candidates_are_strings(self):
        result = self.gen.generate("hello world", 100)
        for item in result:
            assert isinstance(item, str)

    def test_no_source_in_output(self):
        # The source sentence itself should not appear in the candidates
        source = "The city honors its founders with a public ceremony."
        result = self.gen.generate(source, 1776)
        assert source not in result

    def test_max_candidates_respected(self):
        gen = DeterministicGenerator(max_candidates=5)
        result = gen.generate("The city honors its founders.", 1776)
        assert len(result) <= 5

    def test_deterministic(self):
        source = "The city honors its founders with a public ceremony."
        result1 = self.gen.generate(source, 1776)
        result2 = self.gen.generate(source, 1776)
        assert result1 == result2

    def test_sorted_by_proximity_to_target(self):
        target = 1776
        result = self.gen.generate("The city honors its founders with a public ceremony.", target)
        if len(result) > 1:
            errors = [abs(chronogram_score(c) - target) for c in result]
            assert errors[0] <= errors[-1]

    def test_empty_sentence(self):
        # Should return empty list or very short list (no words to substitute)
        result = self.gen.generate("", 100)
        assert isinstance(result, list)

    def test_sentence_with_no_substitutable_words(self):
        # A sentence with no words in the synonym bank
        result = self.gen.generate("zzz bbb qqq", 100)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_custom_synonym_bank(self):
        bank = {"cat": ["feline", "kitty"]}
        gen = DeterministicGenerator(synonym_bank=bank)
        result = gen.generate("the cat sat", 10)
        # Should produce candidates with "feline" or "kitty"
        assert any("feline" in c or "kitty" in c for c in result)

    def test_punctuation_preserved(self):
        source = "The city honors its founders."
        result = self.gen.generate(source, 1776)
        # If the source ends with punctuation, candidates should too
        if source.endswith("."):
            for candidate in result[:5]:
                assert candidate.endswith(".")

    def test_candidates_different_from_each_other(self):
        result = self.gen.generate("The city honors its founders with a public ceremony.", 1776)
        if len(result) > 1:
            # Should not all be identical
            assert len(set(result)) > 1


class TestSynonymBank:
    def test_bank_is_dict(self):
        assert isinstance(_SYNONYM_BANK, dict)

    def test_bank_has_entries(self):
        assert len(_SYNONYM_BANK) > 0

    def test_all_keys_are_lowercase(self):
        for key in _SYNONYM_BANK:
            assert key == key.lower(), f"Key '{key}' is not lower-case"

    def test_all_values_are_lists(self):
        for key, value in _SYNONYM_BANK.items():
            assert isinstance(value, list), f"Value for '{key}' is not a list"

    def test_no_empty_synonym_lists(self):
        for key, value in _SYNONYM_BANK.items():
            assert len(value) > 0, f"Synonym list for '{key}' is empty"

    def test_common_words_present(self):
        for word in ["show", "use", "city", "honor", "public", "final"]:
            assert word in _SYNONYM_BANK, f"'{word}' not in synonym bank"

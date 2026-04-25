"""Tests for chronogrammer.scorer."""

import pytest

from chronogrammer.scorer import (
    ROMAN_VALUES,
    chronogram_score,
    delta_to_target,
    score_breakdown,
)


class TestRomanValues:
    def test_all_keys_present(self):
        assert set(ROMAN_VALUES.keys()) == {"I", "V", "X", "L", "C", "D", "M"}

    def test_values_correct(self):
        assert ROMAN_VALUES["I"] == 1
        assert ROMAN_VALUES["V"] == 5
        assert ROMAN_VALUES["X"] == 10
        assert ROMAN_VALUES["L"] == 50
        assert ROMAN_VALUES["C"] == 100
        assert ROMAN_VALUES["D"] == 500
        assert ROMAN_VALUES["M"] == 1000


class TestChronogramScore:
    def test_empty_string(self):
        assert chronogram_score("") == 0

    def test_no_roman_letters(self):
        # Letters like 'b', 'e', 'g', 'h', 'j', 'k', 'n', 'o', 'p', 'q', 'r',
        # 's', 't', 'u', 'w', 'y', 'z' are not Roman-numeral letters.
        assert chronogram_score("hey") == 0
        assert chronogram_score("stop") == 0
        assert chronogram_score("bong") == 0

    def test_single_letters(self):
        assert chronogram_score("I") == 1
        assert chronogram_score("V") == 5
        assert chronogram_score("X") == 10
        assert chronogram_score("L") == 50
        assert chronogram_score("C") == 100
        assert chronogram_score("D") == 500
        assert chronogram_score("M") == 1000

    def test_case_insensitive(self):
        assert chronogram_score("i") == 1
        assert chronogram_score("v") == 5
        assert chronogram_score("x") == 10
        assert chronogram_score("l") == 50
        assert chronogram_score("c") == 100
        assert chronogram_score("d") == 500
        assert chronogram_score("m") == 1000

    def test_mixed_case(self):
        # "Viva" → V+I+V+A = 5+1+5+0 = 11
        assert chronogram_score("Viva") == 11

    def test_all_roman_letters_upper(self):
        assert chronogram_score("IVXLCDM") == 1 + 5 + 10 + 50 + 100 + 500 + 1000

    def test_all_roman_letters_lower(self):
        assert chronogram_score("ivxlcdm") == 1 + 5 + 10 + 50 + 100 + 500 + 1000

    def test_punctuation_ignored(self):
        assert chronogram_score("I!") == 1
        assert chronogram_score("V.V.") == 10

    def test_numbers_ignored(self):
        assert chronogram_score("I2V") == 6

    def test_viva_the_view(self):
        # V+I+V+A = 5+1+5 = 11, "the" = 0, "view" = V+I = 5+1 = 6 → 17
        # Wait: V-I-V-A = 5+1+5+0=11, t-h-e=0, V-I-E-W=5+1+0+0=6 → 17
        assert chronogram_score("Viva the view") == 17

    def test_sentence_with_known_score(self):
        # "Viva the view! Victors liberate The United States of America"
        # Let's verify digit by digit
        text = "Viva the view! Victors liberate The United States of America"
        # V=5 i=1 v=5 a=0   t=0 h=0 e=0   v=5 i=1 e=0 w=0
        # V=5 i=1 c=100 t=0 o=0 r=0 s=0   l=50 i=1 b=0 e=0 r=0 a=0 t=0 e=0
        # T=0 h=0 e=0   U=0 n=0 i=1 t=0 e=0 d=500 
        # S=0 t=0 a=0 t=0 e=0 s=0   o=0 f=0   A=0 m=1000 e=0 r=0 i=1 c=100 a=0
        # Sum: 5+1+5+5+1+5+1+100+50+1+1+500+1000+1+100 = 1776
        assert chronogram_score(text) == 1776

    def test_additive_not_subtractive(self):
        # IV in Roman notation is 4, but chronograms are purely additive
        assert chronogram_score("IV") == 6  # I=1 + V=5 = 6, not 4

    def test_whitespace_only(self):
        assert chronogram_score("   ") == 0

    def test_digits_not_counted(self):
        assert chronogram_score("12345") == 0

    def test_repeating_m(self):
        assert chronogram_score("MMM") == 3000

    def test_repeating_i(self):
        assert chronogram_score("III") == 3


class TestDeltaToTarget:
    def test_exact_match(self):
        # "IIII" = 4, target 4 → delta 0
        assert delta_to_target("IIII", 4) == 0

    def test_above_target(self):
        # "V" = 5, target 3 → delta +2
        assert delta_to_target("V", 3) == 2

    def test_below_target(self):
        # "I" = 1, target 5 → delta -4
        assert delta_to_target("I", 5) == -4

    def test_empty_string(self):
        assert delta_to_target("", 0) == 0
        assert delta_to_target("", 10) == -10

    def test_zero_target(self):
        # "stop" has no Roman letters, so delta is 0 - 0 = 0
        assert delta_to_target("stop", 0) == 0

    def test_large_target(self):
        assert delta_to_target("M", 2000) == -1000


class TestScoreBreakdown:
    def test_empty_string(self):
        bd = score_breakdown("")
        assert all(v == [] for v in bd.values())

    def test_all_keys_present(self):
        bd = score_breakdown("anything")
        assert set(bd.keys()) == set(ROMAN_VALUES.keys())

    def test_single_upper(self):
        bd = score_breakdown("V")
        assert bd["V"] == ["V"]

    def test_single_lower(self):
        bd = score_breakdown("v")
        assert bd["V"] == ["v"]

    def test_mixed_case_preserved(self):
        bd = score_breakdown("Viva")
        # V → ['V'], I → ['i'], (v is also V) → bd['V'] = ['V', 'v']
        assert "V" in bd["V"]
        assert "v" in bd["V"]
        assert "i" in bd["I"]

    def test_non_roman_excluded(self):
        # "hey" has no Roman-numeral letters (h, e, y are not I/V/X/L/C/D/M)
        bd = score_breakdown("hey")
        assert all(v == [] for v in bd.values())

    def test_count_matches_score(self):
        text = "MMMDCCCLXXXVIII"  # 3888 in classical notation (but additive here)
        bd = score_breakdown(text)
        total_from_breakdown = sum(
            ROMAN_VALUES[k] * len(v) for k, v in bd.items()
        )
        assert total_from_breakdown == chronogram_score(text)

"""
scorer.py – Roman-numeral chronogram scoring core.

Each letter in I V X L C D M contributes its classical value regardless of
surrounding letters (no subtractive rule — chronograms are purely additive).
All other characters contribute 0.
"""

from __future__ import annotations

__all__ = [
    "ROMAN_VALUES",
    "chronogram_score",
    "delta_to_target",
    "score_breakdown",
]

ROMAN_VALUES: dict[str, int] = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def chronogram_score(text: str) -> int:
    """Return the additive Roman-numeral total of *text*.

    Letters are matched case-insensitively; every non-Roman character
    contributes 0.  The standard chronogram convention is strictly additive
    (no subtractive grouping like IV=4).

    Examples
    --------
    >>> chronogram_score("Viva the view!")
    17
    >>> chronogram_score("I")
    1
    >>> chronogram_score("hey")
    0
    """
    total = 0
    for ch in text.upper():
        total += ROMAN_VALUES.get(ch, 0)
    return total


def delta_to_target(text: str, target: int) -> int:
    """Return ``chronogram_score(text) - target``.

    A positive result means the text scores *above* the target;
    negative means it scores *below*.  Zero means an exact chronogram.

    Examples
    --------
    >>> delta_to_target("Viva!", 10)
    1
    >>> delta_to_target("hello", 0)
    0
    """
    return chronogram_score(text) - target


def score_breakdown(text: str) -> dict[str, list[str]]:
    """Return a mapping of each Roman letter to all matching characters in *text*.

    Useful for debugging which letters drive the total.

    Examples
    --------
    >>> breakdown = score_breakdown("Viva!")
    >>> breakdown["V"]
    ['V', 'v']
    """
    breakdown: dict[str, list[str]] = {k: [] for k in ROMAN_VALUES}
    for ch in text:
        upper = ch.upper()
        if upper in ROMAN_VALUES:
            breakdown[upper].append(ch)
    return breakdown

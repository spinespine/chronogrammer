"""
semantic.py – lightweight semantic-drift utilities.

The MVP uses token-overlap / Jaccard similarity so that the package has
**zero required dependencies**.  The architecture makes it straightforward
to swap in a sentence-transformers embedding model or an NLI model later —
just implement the ``SemanticSimilarity`` protocol and pass it to the search.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

__all__ = [
    "SemanticSimilarity",
    "JaccardSimilarity",
    "tokenize",
    "jaccard_similarity",
]

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "that", "this", "these", "those", "it", "its",
    }
)


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """Lowercase, split on non-alphanumeric boundaries, optionally strip stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def jaccard_similarity(a: str, b: str, *, remove_stopwords: bool = True) -> float:
    """Return the Jaccard similarity of the token *sets* of *a* and *b*.

    Returns a value in ``[0.0, 1.0]``.  Returns ``1.0`` when both strings
    are empty, and ``0.0`` when only one is empty.

    Examples
    --------
    >>> jaccard_similarity("the cat sat", "a cat sat")
    1.0
    >>> round(jaccard_similarity("the cat sat on the mat", "a dog stood"), 2)
    0.0
    """
    set_a = set(tokenize(a, remove_stopwords=remove_stopwords))
    set_b = set(tokenize(b, remove_stopwords=remove_stopwords))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@runtime_checkable
class SemanticSimilarity(Protocol):
    """Protocol that any similarity backend must satisfy.

    Parameters
    ----------
    source:
        The original sentence.
    candidate:
        A rewritten candidate sentence.

    Returns
    -------
    float
        Similarity in ``[0.0, 1.0]``; higher is more similar.
    """

    def __call__(self, source: str, candidate: str) -> float:
        ...


class JaccardSimilarity:
    """Concrete ``SemanticSimilarity`` backed by Jaccard token overlap.

    This is the default zero-dependency similarity measure used in the MVP.
    Replace with an embedding-based or NLI-based backend for higher quality.
    """

    def __init__(self, *, remove_stopwords: bool = True) -> None:
        self._remove_stopwords = remove_stopwords

    def __call__(self, source: str, candidate: str) -> float:
        return jaccard_similarity(
            source, candidate, remove_stopwords=self._remove_stopwords
        )

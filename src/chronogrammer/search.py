"""
search.py – beam-search optimiser for chronogram rewriting.

The search minimises a weighted objective combining:

1. **Chronogram error** – ``|score(candidate) - target|`` (dominant term).
2. **Semantic drift** – ``1 - similarity(source, candidate)`` via a pluggable
   ``SemanticSimilarity`` backend (default: Jaccard token overlap).
3. **Length change** – ``|len(words(candidate)) - len(words(source))|``
   discourages padding or over-compression.

The weights are chosen so that an exact chronogram match always beats any
imperfect candidate, regardless of semantic similarity or length.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

from .generator import CandidateGenerator, DeterministicGenerator
from .scorer import chronogram_score
from .semantic import JaccardSimilarity, SemanticSimilarity

__all__ = [
    "ObjectiveWeights",
    "SearchResult",
    "objective_score",
    "beam_search",
]


@dataclasses.dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the multi-objective scoring function.

    Parameters
    ----------
    chronogram_error:
        Weight applied to ``|chronogram_score(candidate) - target|``.
        Should be the dominant term.
    semantic_drift:
        Weight applied to ``1 - similarity(source, candidate)``.
    length_penalty:
        Weight applied to the absolute word-count difference.
    """

    chronogram_error: float = 10.0
    semantic_drift: float = 3.0
    length_penalty: float = 0.2


_DEFAULT_WEIGHTS = ObjectiveWeights()


def _word_count(text: str) -> int:
    return len(text.split())


def objective_score(
    source: str,
    candidate: str,
    target: int,
    *,
    weights: ObjectiveWeights = _DEFAULT_WEIGHTS,
    similarity: SemanticSimilarity | None = None,
) -> float:
    """Compute the weighted objective score for *candidate* given *source* and *target*.

    Lower scores are better.  An exact chronogram (``chronogram_score ==
    target``) can still be penalised for semantic drift, but the chronogram
    term dominates so exact matches beat imperfect ones in practice.

    Parameters
    ----------
    source:
        The original unmodified sentence.
    candidate:
        A candidate rewrite to evaluate.
    target:
        The desired chronogram integer.
    weights:
        Relative weights for each sub-objective.
    similarity:
        A callable ``(source, candidate) -> float in [0, 1]``.  Defaults to
        ``JaccardSimilarity()``.

    Returns
    -------
    float
        Non-negative score; lower is better.
    """
    if similarity is None:
        similarity = JaccardSimilarity()

    chron_err = abs(chronogram_score(candidate) - target)
    sem_drift = 1.0 - similarity(source, candidate)
    len_diff = abs(_word_count(candidate) - _word_count(source))

    return (
        weights.chronogram_error * chron_err
        + weights.semantic_drift * sem_drift
        + weights.length_penalty * len_diff
    )


@dataclasses.dataclass
class SearchResult:
    """Outcome of a beam search run.

    Attributes
    ----------
    best:
        The highest-ranked candidate found (lowest objective score).
    chronogram_score:
        Roman-numeral total of ``best``.
    target:
        The requested chronogram target.
    error:
        ``|chronogram_score - target|``.  Zero means an exact chronogram.
    objective:
        The raw objective score of ``best``.
    candidates:
        All candidates in the final beam, ordered best-first.
    """

    best: str
    chronogram_score: int
    target: int
    error: int
    objective: float
    candidates: list[str]
    exact_matches: list[str] = dataclasses.field(default_factory=list)


def beam_search(
    source: str,
    target: int,
    *,
    generators: Sequence[CandidateGenerator] | None = None,
    beam_width: int = 20,
    steps: int = 8,
    weights: ObjectiveWeights = _DEFAULT_WEIGHTS,
    similarity: SemanticSimilarity | None = None,
    on_exact_match: Callable[[SearchResult], bool] | None = None,
) -> SearchResult:
    """Run beam search to find a rewrite of *source* whose chronogram total
    equals *target*.

    At each step, every item in the current beam is expanded by calling each
    generator and collecting its candidates.  The combined pool is deduplicated
    and the best ``beam_width`` items (by objective score) are kept for the
    next round.

    When an exact chronogram is found, ``on_exact_match`` is called with a
    partial ``SearchResult``.  If it returns ``True`` the search continues
    (useful for interactive "find more" prompts); if it returns ``False`` or
    is ``None``, the search stops.

    Parameters
    ----------
    source:
        The original sentence to rewrite.
    target:
        The desired chronogram total.
    generators:
        One or more ``CandidateGenerator`` instances.  If ``None``, a single
        ``DeterministicGenerator`` is used (fully local, zero-dependency).
    beam_width:
        Number of candidates to keep at each step.
    steps:
        Maximum number of expansion rounds.
    weights:
        Objective weight configuration.
    similarity:
        Semantic-similarity backend.  Defaults to ``JaccardSimilarity``.
    on_exact_match:
        Optional callback invoked each time an exact chronogram is found.
        Receives the current ``SearchResult`` (with ``exact_matches`` populated
        so far).  Return ``True`` to keep searching, ``False`` to stop.

    Returns
    -------
    SearchResult
        The best rewrite found, plus supporting metadata.
    """
    if generators is None:
        generators = [DeterministicGenerator()]
    if similarity is None:
        similarity = JaccardSimilarity()

    def score(candidate: str) -> float:
        return objective_score(
            source, candidate, target, weights=weights, similarity=similarity
        )

    # Seed the beam with the source sentence itself
    beam: list[str] = [source]
    best: str = source
    best_score: float = score(source)

    seen: set[str] = {source}
    exact_matches: list[str] = []

    for _ in range(steps):
        pool: list[str] = []

        for text in beam:
            for gen in generators:
                for candidate in gen.generate(text, target):
                    if candidate not in seen:
                        seen.add(candidate)
                        pool.append(candidate)

        if not pool:
            break

        # Include current beam items so they can survive to the next round
        all_candidates = list(beam) + pool
        all_candidates.sort(key=score)
        beam = all_candidates[:beam_width]

        # Update global best
        if score(beam[0]) < best_score:
            best = beam[0]
            best_score = score(beam[0])

        # Handle exact chronogram
        if chronogram_score(best) == target:
            if best not in exact_matches:
                exact_matches.append(best)
            partial = SearchResult(
                best=best,
                chronogram_score=target,
                target=target,
                error=0,
                objective=best_score,
                candidates=list(beam),
                exact_matches=list(exact_matches),
            )
            keep_going = on_exact_match(partial) if on_exact_match is not None else False
            if not keep_going:
                break
            # Remove the already-found match from the beam so search explores elsewhere
            beam = [c for c in beam if chronogram_score(c) != target] or beam

    chron = chronogram_score(best)
    # Collect any additional exact matches sitting in the final beam
    for c in beam:
        if chronogram_score(c) == target and c not in exact_matches:
            exact_matches.append(c)
    return SearchResult(
        best=best,
        chronogram_score=chron,
        target=target,
        error=abs(chron - target),
        objective=best_score,
        candidates=beam,
        exact_matches=exact_matches,
    )

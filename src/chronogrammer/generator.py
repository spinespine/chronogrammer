"""
generator.py – pluggable candidate-sentence generators.

The ``CandidateGenerator`` protocol defines the interface every generator
must satisfy.  This file ships two implementations:

1. ``DeterministicGenerator``
   A fully local, zero-dependency baseline that replaces words from a
   hand-crafted synonym bank.  It is deterministic for a given seed and
   produces reproducible test results.

2. ``OllamaGenerator`` (stub)
   Shows how to wire in a locally-running LLM via the Ollama REST API.
   The stub raises ``NotImplementedError`` unless a real implementation is
   provided.  See the README for instructions.
"""

from __future__ import annotations

import itertools
import re
from typing import Protocol, runtime_checkable

from .scorer import chronogram_score

__all__ = [
    "CandidateGenerator",
    "DeterministicGenerator",
    "OllamaGenerator",
]

# ---------------------------------------------------------------------------
# Synonym bank – maps each word (lower-case) to a list of near-synonyms.
# The bank is intentionally small for the MVP; extending it improves coverage.
# ---------------------------------------------------------------------------

_SYNONYM_BANK: dict[str, list[str]] = {
    # verbs
    "show": ["reveal", "display", "exhibit", "demonstrate"],
    "shows": ["reveals", "displays", "exhibits", "demonstrates"],
    "use": ["employ", "utilize", "apply", "exercise"],
    "uses": ["employs", "utilizes", "applies"],
    "make": ["create", "craft", "form", "produce"],
    "makes": ["creates", "crafts", "forms"],
    "give": ["provide", "offer", "deliver", "lend"],
    "gives": ["provides", "offers", "delivers"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "gets": ["obtains", "acquires", "receives"],
    "keep": ["maintain", "preserve", "retain", "hold"],
    "keeps": ["maintains", "preserves", "retains"],
    "see": ["view", "observe", "witness", "notice"],
    "sees": ["views", "observes", "witnesses"],
    "look": ["view", "examine", "inspect", "observe"],
    "looks": ["views", "examines", "inspects"],
    "go": ["move", "travel", "proceed", "advance"],
    "goes": ["moves", "travels", "proceeds"],
    "come": ["arrive", "appear", "emerge", "visit"],
    "comes": ["arrives", "appears", "emerges"],
    "take": ["seize", "capture", "claim", "grasp"],
    "takes": ["seizes", "captures", "claims"],
    "put": ["place", "position", "set", "locate"],
    "puts": ["places", "positions", "sets"],
    "find": ["discover", "locate", "identify", "detect"],
    "finds": ["discovers", "locates", "identifies"],
    "think": ["believe", "consider", "reckon", "deem"],
    "thinks": ["believes", "considers", "reckons"],
    "know": ["understand", "recognize", "realize", "perceive"],
    "knows": ["understands", "recognizes", "realizes"],
    "call": ["name", "designate", "term", "label"],
    "calls": ["names", "designates", "terms"],
    "want": ["desire", "wish", "seek", "crave"],
    "wants": ["desires", "wishes", "seeks"],
    "say": ["state", "declare", "assert", "claim"],
    "says": ["states", "declares", "asserts"],
    "tell": ["inform", "advise", "notify", "instruct"],
    "tells": ["informs", "advises", "notifies"],
    "work": ["labor", "toil", "operate", "function"],
    "works": ["labors", "toils", "operates"],
    "live": ["exist", "reside", "dwell", "inhabit"],
    "lives": ["exists", "resides", "dwells"],
    "feel": ["sense", "experience", "perceive"],
    "feels": ["senses", "experiences", "perceives"],
    "leave": ["depart", "exit", "vacate", "withdraw"],
    "leaves": ["departs", "exits", "vacates"],
    "move": ["shift", "transfer", "relocate", "migrate"],
    "moves": ["shifts", "transfers", "relocates"],
    "honor": ["celebrate", "commemorate", "acclaim", "salute"],
    "honors": ["celebrates", "commemorates", "acclaims"],
    "build": ["construct", "erect", "establish", "create"],
    "builds": ["constructs", "erects", "establishes"],
    "lead": ["guide", "direct", "command", "govern"],
    "leads": ["guides", "directs", "commands"],
    "hold": ["maintain", "keep", "possess", "retain"],
    "holds": ["maintains", "keeps", "possesses"],
    "stand": ["rise", "remain", "endure", "persist"],
    "stands": ["rises", "remains", "endures"],
    "win": ["triumph", "prevail", "succeed", "conquer"],
    "wins": ["triumphs", "prevails", "succeeds"],
    "mark": ["commemorate", "celebrate", "signify", "denote"],
    "marks": ["commemorates", "celebrates", "signifies"],
    "serve": ["assist", "aid", "help", "support"],
    "serves": ["assists", "aids", "helps"],
    "approve": ["endorse", "validate", "confirm", "sanction"],
    "approves": ["endorses", "validates", "confirms"],
    # nouns
    "city": ["town", "municipality", "metropolis", "civic center"],
    "country": ["nation", "land", "realm", "state"],
    "people": ["citizens", "individuals", "persons", "community"],
    "man": ["individual", "person", "male", "fellow"],
    "woman": ["individual", "person", "female", "lady"],
    "child": ["youth", "minor", "kid", "youngster"],
    "children": ["youths", "minors", "kids", "youngsters"],
    "time": ["era", "period", "moment", "occasion"],
    "year": ["period", "cycle", "season", "era"],
    "day": ["date", "moment", "occasion", "period"],
    "place": ["location", "site", "spot", "locale"],
    "way": ["manner", "method", "means", "mode"],
    "world": ["realm", "domain", "universe", "globe"],
    "life": ["existence", "living", "being"],
    "hand": ["grasp", "hold", "control"],
    "part": ["portion", "section", "segment", "component"],
    "eye": ["vision", "view", "gaze"],
    "face": ["visage", "countenance", "expression"],
    "head": ["leader", "director", "chief"],
    "mind": ["intellect", "reason", "thought", "cognition"],
    "heart": ["core", "center", "soul", "spirit"],
    "word": ["term", "expression", "statement"],
    "home": ["residence", "dwelling", "abode", "domicile"],
    "house": ["residence", "dwelling", "abode", "domicile"],
    "school": ["institution", "academy", "college", "establishment"],
    "work": ["labor", "effort", "toil", "endeavor"],
    "power": ["authority", "control", "dominion", "force"],
    "money": ["funds", "currency", "capital", "wealth"],
    "water": ["liquid", "fluid"],
    "land": ["territory", "region", "domain", "realm"],
    "end": ["conclusion", "finale", "close", "termination"],
    "point": ["position", "location", "moment", "instance"],
    "fact": ["reality", "truth", "actuality"],
    "group": ["collection", "assembly", "cluster", "body"],
    "idea": ["concept", "notion", "thought", "view"],
    "plan": ["design", "scheme", "strategy", "program"],
    "event": ["occasion", "ceremony", "incident", "affair"],
    "ceremony": ["celebration", "commemoration", "ritual", "observance"],
    "history": ["chronicle", "record", "past", "legacy"],
    "story": ["account", "narrative", "tale", "chronicle"],
    "book": ["volume", "text", "document", "manuscript"],
    "letter": ["missive", "document", "message"],
    "number": ["figure", "digit", "total", "count"],
    "order": ["command", "directive", "instruction", "mandate"],
    "line": ["row", "series", "sequence"],
    "form": ["structure", "format", "design"],
    "system": ["framework", "structure", "arrangement"],
    "law": ["rule", "regulation", "statute", "decree"],
    "area": ["region", "zone", "district", "sector"],
    "state": ["condition", "status", "circumstance"],
    "founder": ["creator", "originator", "establisher", "pioneer"],
    "founders": ["creators", "originators", "establishers", "pioneers"],
    "leader": ["director", "chief", "commander", "governor"],
    "leaders": ["directors", "chiefs", "commanders", "governors"],
    "member": ["participant", "associate", "constituent"],
    "members": ["participants", "associates", "constituents"],
    "community": ["society", "collective", "populace", "citizenry"],
    "report": ["document", "record", "account", "summary"],
    "version": ["edition", "form", "variant", "revision"],
    "panel": ["committee", "council", "board", "commission"],
    "committee": ["council", "board", "commission", "body"],
    # adjectives
    "big": ["large", "vast", "immense", "considerable"],
    "small": ["little", "minor", "modest", "limited"],
    "good": ["fine", "excellent", "beneficial", "sound"],
    "great": ["significant", "vast", "immense", "notable"],
    "long": ["extended", "prolonged", "extensive", "considerable"],
    "little": ["small", "minor", "modest", "limited"],
    "own": ["personal", "individual", "exclusive"],
    "other": ["alternative", "additional", "distinct"],
    "old": ["ancient", "established", "traditional", "historic"],
    "new": ["recent", "modern", "novel", "current"],
    "first": ["initial", "primary", "original", "foremost"],
    "last": ["final", "concluding", "ultimate", "closing"],
    "right": ["correct", "proper", "accurate", "valid"],
    "high": ["elevated", "significant", "considerable", "substantial"],
    "next": ["following", "subsequent", "succeeding"],
    "early": ["initial", "original", "preliminary"],
    "public": ["civic", "communal", "civil", "collective"],
    "final": ["concluding", "ultimate", "terminal", "closing"],
    "completed": ["finished", "concluded", "accomplished", "finalized"],
    "national": ["civic", "civil", "collective", "communal"],
    "local": ["regional", "community", "municipal"],
    "free": ["liberated", "unrestricted", "independent"],
    "full": ["complete", "entire", "total", "comprehensive"],
    "real": ["actual", "genuine", "true", "factual"],
    "important": ["significant", "critical", "vital", "essential"],
    "large": ["vast", "immense", "sizable", "considerable"],
    "strong": ["powerful", "forceful", "robust", "vigorous"],
    "certain": ["definite", "particular", "specific", "distinct"],
    "open": ["accessible", "available", "unrestricted", "receptive"],
    "major": ["significant", "principal", "primary", "central"],
    "simple": ["basic", "plain", "straightforward", "elementary"],
    "hard": ["difficult", "challenging", "demanding", "arduous"],
    # adverbs / function words
    "also": ["additionally", "likewise", "similarly", "moreover"],
    "just": ["merely", "simply", "only"],
    "now": ["currently", "presently", "today", "at present"],
    "still": ["yet", "nonetheless", "even now", "continuously"],
    "even": ["indeed", "truly", "actually"],
    "well": ["ably", "effectively", "properly", "soundly"],
    "only": ["merely", "solely", "exclusively", "just"],
    "then": ["subsequently", "afterward", "later", "next"],
    "here": ["at this location", "in this place", "locally"],
    "there": ["at that location", "in that place"],
    "very": ["quite", "remarkably", "notably", "truly"],
    "so": ["consequently", "therefore", "thus", "accordingly"],
    # prepositions / conjunctions
    "because": ["since", "as", "given that", "considering"],
    "since": ["because", "as", "given that"],
    "while": ["whereas", "although", "even though", "as"],
    "before": ["prior to", "ahead of", "preceding"],
    "after": ["following", "subsequent to", "in the wake of"],
    "through": ["via", "by means of", "across"],
    "about": ["regarding", "concerning", "relating to"],
    "against": ["contrary to", "in opposition to", "versus"],
    "between": ["amid", "among", "within"],
    "during": ["throughout", "in the course of", "amid"],
    "without": ["lacking", "absent", "devoid of"],
    "within": ["inside", "in", "amid"],
    "across": ["throughout", "over", "via"],
    "along": ["beside", "alongside", "together with"],
    "upon": ["on", "atop", "over"],
}


@runtime_checkable
class CandidateGenerator(Protocol):
    """Protocol for all candidate-sentence generators.

    Generators receive the *source* sentence, the *target* chronogram
    integer, and return an iterable of candidate strings.  The iterable may
    be lazy (a generator expression is fine).
    """

    def generate(self, source: str, target: int) -> list[str]:
        """Return a list of candidate rewrites for *source* targeting *target*."""
        ...


def _tokenize_preserve_case(text: str) -> list[str]:
    """Split *text* into tokens, preserving original spacing tokens."""
    return re.findall(r"\S+|\s+", text)


def _replace_word(tokens: list[str], index: int, replacement: str) -> str:
    """Return the sentence formed by replacing token at *index* with *replacement*."""
    result = tokens[:]
    # Preserve trailing punctuation if the original token ended with one
    original = tokens[index]
    punct = ""
    if original and original[-1] in ".,;:!?\"'":
        punct = original[-1]
        replacement = replacement.rstrip(".,;:!?\"'") + punct
    result[index] = replacement
    return "".join(result)


class DeterministicGenerator:
    """Fully local, zero-dependency candidate generator.

    Generates candidates by systematically substituting words in the source
    sentence with entries from the synonym bank.  Results are deterministic
    and reproducible — no randomness, no network calls.

    Candidates are sorted by how close they bring the chronogram score to the
    target, so callers can take a prefix of results with confidence.

    Parameters
    ----------
    synonym_bank:
        A mapping from lower-case word → list of replacements.  Defaults to
        the built-in ``_SYNONYM_BANK``.
    max_candidates:
        Cap on the total number of candidates returned per call.
    """

    def __init__(
        self,
        synonym_bank: dict[str, list[str]] | None = None,
        max_candidates: int = 200,
    ) -> None:
        self._bank = synonym_bank if synonym_bank is not None else _SYNONYM_BANK
        self._max = max_candidates

    def generate(self, source: str, target: int) -> list[str]:
        """Return candidate rewrites of *source* sorted by chronogram proximity to *target*."""
        tokens = _tokenize_preserve_case(source)
        word_indices = [i for i, t in enumerate(tokens) if t.strip() and not t.isspace()]

        candidates: list[str] = []
        seen: set[str] = {source}

        # Single-word substitutions
        for idx in word_indices:
            token = tokens[idx]
            # Strip punctuation to look up in bank
            key = re.sub(r"[^a-zA-Z]", "", token).lower()
            if not key:
                continue
            for replacement in self._bank.get(key, []):
                candidate = _replace_word(tokens, idx, replacement)
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
                    if len(candidates) >= self._max * 2:
                        break
            if len(candidates) >= self._max * 2:
                break

        # Two-word substitutions (to widen the search space)
        if len(candidates) < self._max:
            for i, j in itertools.combinations(word_indices, 2):
                token_i = tokens[i]
                token_j = tokens[j]
                key_i = re.sub(r"[^a-zA-Z]", "", token_i).lower()
                key_j = re.sub(r"[^a-zA-Z]", "", token_j).lower()
                replacements_i = self._bank.get(key_i, [])
                replacements_j = self._bank.get(key_j, [])
                for rep_i in replacements_i:
                    for rep_j in replacements_j:
                        partial = _replace_word(tokens, i, rep_i)
                        partial_tokens = _tokenize_preserve_case(partial)
                        candidate = _replace_word(partial_tokens, j, rep_j)
                        if candidate not in seen:
                            seen.add(candidate)
                            candidates.append(candidate)
                            if len(candidates) >= self._max * 3:
                                break
                    else:
                        continue
                    break
                if len(candidates) >= self._max * 3:
                    break

        # Sort by proximity to target
        candidates.sort(key=lambda c: abs(chronogram_score(c) - target))
        return candidates[: self._max]


class OllamaGenerator:
    """Candidate generator backed by a locally-running Ollama LLM.

    This generator calls the Ollama REST API (``http://localhost:11434`` by
    default).  It is **not** activated by default and requires:

    1. Ollama installed and running locally (https://ollama.ai).
    2. A pulled model, e.g. ``ollama pull mistral``.
    3. The ``requests`` library: ``pip install chronogrammer[llm]``.

    See the README section *"Plugging in a local LLM"* for a full walkthrough.

    Parameters
    ----------
    model:
        Name of the Ollama model to use (e.g. ``"mistral"``, ``"llama3"``).
    base_url:
        Base URL of the Ollama API.
    num_candidates:
        How many candidates to request per call.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        num_candidates: int = 20,
        timeout: int = 60,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._num_candidates = num_candidates
        self._timeout = timeout

    def _build_prompt(self, source: str, target: int) -> str:
        current = chronogram_score(source)
        delta = target - current
        direction = "increase" if delta > 0 else "decrease"
        return (
            f"You are revising a sentence to act as a chronogram.\n\n"
            f"Roman-letter values (additive, case-insensitive):\n"
            f"I=1, V=5, X=10, L=50, C=100, D=500, M=1000\n\n"
            f"Original sentence:\n\"{source}\"\n\n"
            f"Target Roman-letter total: {target}\n"
            f"Current total: {current}\n"
            f"We need to {direction} the total by {abs(delta)}.\n\n"
            f"Rewrite the sentence with minimal semantic drift.\n"
            f"Preserve all facts, entities, and tone.\n"
            f"Prefer the smallest possible lexical changes.\n"
            f"Return exactly {self._num_candidates} alternatives, one per line, "
            f"with no numbering or extra commentary."
        )

    def generate(self, source: str, target: int) -> list[str]:
        """Call Ollama to generate paraphrase candidates.

        Raises
        ------
        RuntimeError
            If the ``requests`` library is not installed or the Ollama API
            is unreachable.
        """
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'requests' package is required for OllamaGenerator. "
                "Install it with: pip install chronogrammer[llm]"
            ) from exc

        prompt = self._build_prompt(source, target)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama API request failed: {exc}\n"
                "Ensure Ollama is running locally and the model is pulled."
            ) from exc

        raw = response.json().get("response", "")
        candidates = [line.strip() for line in raw.splitlines() if line.strip()]
        return candidates[: self._num_candidates]

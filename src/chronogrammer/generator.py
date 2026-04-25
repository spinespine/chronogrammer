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
import sys
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


# Words that are not worth targeting as LLM replacement slots.
_SKIP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "nor", "yet", "so",
    "in", "on", "at", "to", "of", "for", "by", "up", "as",
    "its", "it", "is", "are", "was", "were", "has", "had",
    "not", "all", "any", "our", "his", "her",
})


class OllamaGenerator:
    """Candidate generator backed by a locally-running Ollama LLM.

    Uses full-context, forced-slot generation: each call selects one specific
    word to replace (rotated across calls to prevent loops) and asks the model
    to suggest alternatives with the **complete original sentence always in
    view** as a semantic anchor.  This exploits the LLM's next-token prediction
    strength — it is excellent at "what word fits naturally here given this
    surrounding context?" — while keeping the arithmetic scoring on our side.

    Parameters
    ----------
    model:
        Name of the Ollama model to use (e.g. ``"mistral"``, ``"llama3"``).
    base_url:
        Base URL of the Ollama API.
    num_candidates:
        How many replacement options to request per call.
    timeout:
        HTTP request timeout in seconds.
    original:
        The user's original sentence.  Used as a permanent semantic anchor so
        every LLM call can see what meaning must be preserved even as the beam
        evolves.  If ``None``, the first text passed to ``generate`` is used.
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        num_candidates: int = 20,
        timeout: int = 300,
        original: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._num_candidates = num_candidates
        self._timeout = timeout
        self._original: str | None = original
        # Monotonically incrementing counter used to rotate which word slot is
        # targeted each call, ensuring beam-item diversity.
        self._call_counter: int = 0
        # Tracks every replacement already suggested for each slot word so we
        # can tell the model what NOT to output, preventing it from burning the
        # entire candidate budget re-suggesting its top-probability tokens.
        # Key: lowercase slot word.  Value: set of lowercase replacements tried.
        self._tried: dict[str, set[str]] = {}

    def _select_slots(self, text: str, target: int) -> list[dict]:
        """Return substitutable word slots sorted by replacement feasibility.

        For each content word (≥ 3 letters), compute:
        - ``word_score``: its current Roman-numeral contribution
        - ``needed``: the Roman-numeral sum the replacement must have to reach
          *target* in one swap (``word_score + delta``)
        - ``feasible``: whether ``needed`` is in a range real English words
          can plausibly achieve (0–1500)

        Feasible slots come first; ties broken by highest ``word_score``
        (more Roman letters = more room to manoeuvre).
        """
        current = chronogram_score(text)
        delta = target - current
        tokens = _tokenize_preserve_case(text)
        slots = []
        for i, tok in enumerate(tokens):
            if not tok.strip():
                continue
            clean = re.sub(r"[^a-zA-Z]", "", tok)
            if len(clean) < 4 or clean.lower() in _SKIP_WORDS:
                continue
            ws = chronogram_score(clean)
            needed = ws + delta
            slots.append(
                {
                    "index": i,
                    "word": clean,
                    "word_score": ws,
                    "needed": needed,
                    "feasible": 0 <= needed <= 1500,
                }
            )
        # When delta < 0 (score too high), we want to replace high-value words
        # with lighter synonyms → sort by descending word_score.
        # When delta >= 0 (score too low), prefer feasible slots (those where
        # a natural synonym might plausibly land near the needed value) first.
        if delta < 0:
            slots.sort(key=lambda s: -s["word_score"])
        else:
            slots.sort(key=lambda s: (not s["feasible"], -s["word_score"]))
        return slots

    def _build_slot_prompt(
        self,
        original: str,
        current: str,
        word: str,
        n: int,
        exclude: set[str],
    ) -> str:
        """Build a full-context, forced-slot replacement prompt.

        The model is used purely for its linguistic ability: given the original
        sentence as a semantic anchor and the current candidate with one word
        bracketed, produce diverse alternatives that fit grammatically and
        preserve meaning.  Roman-numeral arithmetic is entirely our concern —
        the model has no knowledge of the letter-by-letter spelling of its own
        output tokens, so mentioning target values would only add noise.
        """
        marked = re.sub(
            rf"(?i)\b{re.escape(word)}\b",
            f"[{word}]",
            current,
            count=1,
        )
        exclusion_line = (
            f"  - Do NOT use any of these (already tried): "
            + ", ".join(sorted(exclude)[:30])  # cap list length in prompt
            + "\n"
            if exclude
            else ""
        )
        return (
            f"Replace the bracketed word in the sentence below with {n} different "
            f"alternatives that preserve the original meaning and fit grammatically.\n\n"
            f"Original sentence:\n"
            f'  "{original}"\n\n'
            f"Sentence to edit:\n"
            f'  "{marked}"\n\n'
            f"Rules:\n"
            f"  - Output ONLY the replacement word or short phrase (up to 3 words)\n"
            f"  - One option per line, no numbering, no extra commentary\n"
            f"  - Do not reproduce the full sentence\n"
            f"  - Each option must fit naturally where [{word}] appears\n"
            + exclusion_line
        )

    def _call_ollama(self, prompt: str, requests_mod) -> list[str]:
        """Stream one Ollama request and return all non-empty response lines."""
        import json as _json

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "think": False,  # disable chain-of-thought (qwen3 etc.) for speed
        }
        url = f"{self._base_url}/api/generate"
        try:
            resp = requests_mod.post(
                url, json=payload, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama API request failed: {exc}\n"
                "Ensure Ollama is running locally and the model is pulled."
            ) from exc

        parts: list[str] = []
        sys.stderr.write("    ")
        sys.stderr.flush()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = _json.loads(raw_line)
            token = chunk.get("response", "")
            parts.append(token)
            sys.stderr.write(token)
            sys.stderr.flush()
            if chunk.get("done"):
                break
        sys.stderr.write("\n")
        sys.stderr.flush()

        return [ln.strip() for ln in "".join(parts).splitlines() if ln.strip()]

    def generate(self, text: str, target: int) -> list[str]:
        """Generate candidates via full-context, forced-slot LLM substitution.

        Algorithm
        ---------
        1. Auto-capture the original sentence on first call (if not supplied at
           construction) so it is available as a semantic anchor forever after.
        2. Select all substitutable word slots ranked by replacement feasibility.
        3. Pick one slot per call using a round-robin counter — each invocation
           targets a *different* word, so consecutive beam-search expansions
           never get stuck swapping the same token.
        4. Ask the LLM (with full sentence context) for ``num_candidates``
           alternatives for that one word.  The model never does arithmetic;
           we score every reconstructed sentence ourselves.
        5. Return candidates sorted by proximity to *target*.

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

        # Capture original sentence on first call
        if self._original is None:
            self._original = text

        current_score = chronogram_score(text)
        if current_score == target:
            return []

        tokens = _tokenize_preserve_case(text)
        all_slots = self._select_slots(text, target)
        if not all_slots:
            return []

        # Round-robin across all slots.  _select_slots already orders them so
        # that words most likely to make per-step progress come first:
        #   - when delta < 0 (score too high): highest word_score first, so
        #     the LLM replaces heavy Roman words with lighter synonyms
        #   - when delta > 0 (score too low): feasible (low word_score) slots
        #     first, so the LLM replaces light words with heavier synonyms
        # The model never sees any of this arithmetic — it just produces
        # natural alternatives; we score every result ourselves.
        slot = all_slots[self._call_counter % len(all_slots)]
        self._call_counter += 1

        word = slot["word"]
        idx = slot["index"]
        word_key = word.lower()
        already_tried = self._tried.get(word_key, set())

        print(
            f"  [Ollama] Δ={target - current_score:+d} | targeting \"{word}\""
            + (f" (excl. {len(already_tried)})" if already_tried else ""),
            file=sys.stderr,
            flush=True,
        )

        prompt = self._build_slot_prompt(
            self._original, text, word, self._num_candidates, already_tried
        )
        replacements = self._call_ollama(prompt, requests)

        candidates: list[str] = []
        seen: set[str] = {text}
        seen_lower: set[str] = set()  # case-insensitive dedup within this response
        new_tried: set[str] = set()
        for repl in replacements:
            # Strip stray list markers the model may have added
            clean = re.sub(r"^[\d\.\)\-\*\s]+", "", repl).strip()
            # Remove any brackets that leaked through from the prompt
            clean = clean.strip("[]")
            # Reject empty, unchanged, or sentence-length outputs.
            if (
                not clean
                or clean.lower() == word.lower()
                or len(clean.split()) > 3
                or clean.endswith((".", "?", "!"))
            ):
                continue
            # Case-insensitive dedup within this batch
            if clean.lower() in seen_lower:
                continue
            seen_lower.add(clean.lower())
            new_tried.add(clean.lower())
            candidate = _replace_word(tokens, idx, clean)
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        # Record all suggestions (including ones filtered as duplicates) so
        # future calls for the same slot word avoid them.
        if word_key not in self._tried:
            self._tried[word_key] = set()
        self._tried[word_key].update(new_tried)

        candidates.sort(key=lambda c: abs(chronogram_score(c) - target))
        best_err = abs(chronogram_score(candidates[0]) - target) if candidates else "n/a"
        print(
            f"  [Ollama] {len(candidates)} candidates — best error = {best_err}",
            file=sys.stderr,
            flush=True,
        )
        return candidates[: self._num_candidates]

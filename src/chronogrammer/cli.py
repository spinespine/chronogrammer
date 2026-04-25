"""
cli.py – command-line interface for chronogrammer.

Usage
-----
Score a sentence::

    chronogrammer score "Viva the view!"

Rewrite a sentence to hit a target chronogram value::

    chronogrammer rewrite --target 1776 "The city honors its founders."

Run ``chronogrammer --help`` for full option documentation.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .generator import DeterministicGenerator, OllamaGenerator
from .scorer import ROMAN_VALUES, chronogram_score, delta_to_target, score_breakdown
from .search import ObjectiveWeights, SearchResult, beam_search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_report(text: str) -> str:
    """Return a human-readable score report for *text*."""
    total = chronogram_score(text)
    breakdown = score_breakdown(text)
    letter_parts = []
    for letter, value in ROMAN_VALUES.items():
        chars = breakdown[letter]
        if chars:
            letter_parts.append(f"{letter}={value}×{len(chars)}")
    detail = "  +  ".join(letter_parts) if letter_parts else "(none)"
    lines = [
        f'  Text   : "{text}"',
        f"  Score  : {total}",
        f"  Detail : {detail}",
    ]
    return "\n".join(lines)


def _result_report(result: SearchResult, source: str) -> str:
    """Return a human-readable rewrite result report."""
    exact = "✓ exact chronogram" if result.error == 0 else f"✗ off by {result.error}"
    lines = [
        f'  Source : "{source}"',
        f"  Target : {result.target}",
        "",
        f"  Best rewrite ({exact}):",
        f'    "{result.best}"',
        f"    Score : {result.chronogram_score}",
        "",
    ]
    if result.exact_matches:
        lines.append(f"  All exact matches found ({len(result.exact_matches)}):")
        for i, m in enumerate(result.exact_matches, 1):
            lines.append(f'    {i}. "{m}"')
        lines.append("")
    elif len(result.candidates) > 1:
        lines.append("  Top candidates:")
        for i, cand in enumerate(result.candidates[:5], 1):
            s = chronogram_score(cand)
            err = abs(s - result.target)
            marker = "✓" if err == 0 else f"Δ{err:+d}"
            lines.append(f"    {i}. [{marker}] score={s}  \"{cand}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    """Handle the ``score`` sub-command."""
    print("\nChronogram score\n" + "─" * 40)
    print(_score_report(args.text))
    if args.target is not None:
        delta = delta_to_target(args.text, args.target)
        direction = "above" if delta > 0 else "below" if delta < 0 else "exact"
        print(f"  Target : {args.target}")
        if direction == "exact":
            print("  Delta  : 0  ✓ exact chronogram!")
        else:
            print(f"  Delta  : {delta:+d}  ({abs(delta)} {direction} target)")
    print()
    return 0


def cmd_rewrite(args: argparse.Namespace) -> int:
    """Handle the ``rewrite`` sub-command."""
    # Build generator list
    generators = []
    if args.llm:
        print(f"  Using Ollama model: {args.model}", file=sys.stderr)
        generators.append(
            OllamaGenerator(
                model=args.model,
                base_url=args.ollama_url,
                num_candidates=args.llm_candidates,
                timeout=args.llm_timeout,
                original=args.text,
            )
        )
    # Always include the deterministic baseline
    generators.append(DeterministicGenerator(max_candidates=args.max_candidates))

    weights = ObjectiveWeights(
        chronogram_error=args.w_chron,
        semantic_drift=args.w_sem,
        length_penalty=args.w_len,
    )

    def _on_exact_match(partial: SearchResult) -> bool:
        """Print a match and ask the user whether to keep searching."""
        n = len(partial.exact_matches)
        print(f"\n  ✓ Exact match #{n} found (score={partial.chronogram_score}):", file=sys.stderr)
        print(f'    "{partial.best}"', file=sys.stderr)
        try:
            ans = input("  Continue searching for more matches? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return False
        return ans in ("y", "yes")

    print(
        f"\nSearching for chronogram of {args.target} …",
        file=sys.stderr,
    )
    result = beam_search(
        args.text,
        args.target,
        generators=generators,
        beam_width=args.beam_width,
        steps=args.steps,
        weights=weights,
        on_exact_match=_on_exact_match,
    )

    print("\nChronogram rewrite\n" + "─" * 40)
    print(_result_report(result, args.text))
    return 0 if result.error == 0 else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronogrammer",
        description=textwrap.dedent(
            """\
            A local-first chronogram rewriting system.

            A chronogram is a text whose Roman-numeral letters (I V X L C D M)
            sum to a target number (typically a commemorated year).
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── score ──────────────────────────────────────────────────────────────
    score_p = sub.add_parser(
        "score",
        help="Compute the chronogram score of a sentence.",
        description="Print the Roman-numeral total for TEXT.",
    )
    score_p.add_argument("text", metavar="TEXT", help="Sentence to score.")
    score_p.add_argument(
        "--target",
        type=int,
        default=None,
        metavar="N",
        help="Optional target value; prints the delta if supplied.",
    )
    score_p.set_defaults(func=cmd_score)

    # ── rewrite ────────────────────────────────────────────────────────────
    rewrite_p = sub.add_parser(
        "rewrite",
        help="Rewrite a sentence to hit a target chronogram value.",
        description="Use beam search to find rewrites of TEXT whose Roman-letter total equals TARGET.",
    )
    rewrite_p.add_argument("text", metavar="TEXT", help="Sentence to rewrite.")
    rewrite_p.add_argument(
        "--target",
        type=int,
        required=True,
        metavar="N",
        help="Desired chronogram total.",
    )
    rewrite_p.add_argument(
        "--beam-width",
        type=int,
        default=20,
        metavar="N",
        help="Beam width for search (default: 20).",
    )
    rewrite_p.add_argument(
        "--steps",
        type=int,
        default=8,
        metavar="N",
        help="Maximum search steps (default: 8).",
    )
    rewrite_p.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        metavar="N",
        help="Max candidates from the deterministic generator per step (default: 200).",
    )
    # Objective weights
    rewrite_p.add_argument("--w-chron", type=float, default=10.0, metavar="W",
                            help="Weight for chronogram error (default: 10.0).")
    rewrite_p.add_argument("--w-sem", type=float, default=3.0, metavar="W",
                            help="Weight for semantic drift (default: 3.0).")
    rewrite_p.add_argument("--w-len", type=float, default=0.2, metavar="W",
                            help="Weight for length change (default: 0.2).")
    # LLM options (optional)
    rewrite_p.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable the Ollama LLM generator (requires Ollama running locally).",
    )
    rewrite_p.add_argument(
        "--model",
        default="mistral",
        metavar="MODEL",
        help="Ollama model name (default: mistral).",
    )
    rewrite_p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    rewrite_p.add_argument(
        "--llm-candidates",
        type=int,
        default=20,
        metavar="N",
        help="Number of candidates to request from the LLM per step (default: 20).",
    )
    rewrite_p.add_argument(
        "--llm-timeout",
        type=int,
        default=300,
        metavar="SECS",
        help="Timeout in seconds for each Ollama API call (default: 300).",
    )
    rewrite_p.set_defaults(func=cmd_rewrite)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point.  Returns an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

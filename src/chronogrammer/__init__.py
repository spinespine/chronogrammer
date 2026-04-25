"""
chronogrammer — a local-first chronogram rewriting system.

A chronogram is a text whose Roman-numeral letters (I, V, X, L, C, D, M)
sum to a target number (usually a commemorated year).

This package exposes:
  - ``scorer``    – compute and compare Roman-numeral totals
  - ``semantic``  – lightweight semantic-drift utilities
  - ``generator`` – pluggable candidate-sentence generators
  - ``search``    – beam-search optimiser
  - ``cli``       – command-line interface
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chronogrammer")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]

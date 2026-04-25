# chronogrammer

A **local-first chronogram rewriting system** that rewrites sentences so that
the Roman-numeral letters (I, V, X, L, C, D, M) they contain sum to a
user-specified target value — while minimising semantic drift.

A *chronogram* is a text in which certain letters, interpreted as Roman
numerals, add up to a commemorated number.  For example:

> **V**i**v**a the **v**iew!  **V**ictors l**i**berate The Un**i**te**d**
> States of A**m**er**i****c**a

counts V+I+V+V+I+V+I+C+L+I+I+D+M+I+C = **1776** — the year of US
independence.

---

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [CLI usage](#cli-usage)
- [Running tests](#running-tests)
- [Architecture](#architecture)
- [Plugging in a local LLM](#plugging-in-a-local-llm)
- [Extension ideas](#extension-ideas)

---

## What it does

`chronogrammer` provides:

| Component | Description |
|-----------|-------------|
| **Scorer** | Computes the additive Roman-numeral total of any string (I=1, V=5, X=10, L=50, C=100, D=500, M=1000). |
| **Generator** | A pluggable interface for producing candidate rewrites.  Ships a fully-local deterministic synonym-substitution baseline and an Ollama stub. |
| **Search** | A beam-search optimiser that minimises a weighted objective combining chronogram error, semantic drift, and length change. |
| **CLI** | `chronogrammer score` and `chronogrammer rewrite` commands. |

---

## Installation

Requires **Python 3.9+**.  No third-party packages are needed for the core
(the optional LLM backend requires `requests`).

```bash
# Clone and install in editable mode (recommended for development)
git clone https://github.com/spinespine/chronogrammer.git
cd chronogrammer
pip install -e ".[dev]"

# Or install the core only (no test dependencies)
pip install -e .
```

Verify the installation:

```bash
chronogrammer --version
```

---

## CLI usage

### `score` — compute the chronogram total of a sentence

```bash
chronogrammer score "Viva the view!"
```

```
Chronogram score
────────────────────────────────────────
  Text   : "Viva the view!"
  Score  : 17
  Detail : I=1×2  +  V=5×3
```

Supply `--target` to see how far you are from a goal:

```bash
chronogrammer score --target 1776 \
  "Viva the view! Victors liberate The United States of America"
```

```
Chronogram score
────────────────────────────────────────
  Text   : "Viva the view! Victors liberate The United States of America"
  Score  : 1776
  Detail : I=1×6  +  V=5×4  +  L=50×1  +  C=100×2  +  D=500×1  +  M=1000×1
  Target : 1776
  Delta  : 0  ✓ exact chronogram!
```

### `rewrite` — search for a chronogram rewrite

```bash
chronogrammer rewrite --target 1776 \
  "The city honors its founders with a public ceremony."
```

```
Chronogram rewrite
────────────────────────────────────────
  Source : "The city honors its founders with a public ceremony."
  Target : 1776

  Best rewrite (✗ off by 9):
    "The civic center celebrates its establishers with a civil ceremony."
    Score : 1767

  Top candidates:
    1. [Δ+9] score=1767  "The civic center celebrates its establishers with a civil ceremony."
    2. [Δ+10] score=1766  "The civic center honors its establishers with a collective ceremony."
    ...
```

Exit code 0 means an exact chronogram was found; exit code 1 means the search
returned its best near-miss.

#### Rewrite options

| Flag | Default | Description |
|------|---------|-------------|
| `--target N` | *(required)* | Desired chronogram total. |
| `--beam-width N` | 20 | Candidates kept at each search step. |
| `--steps N` | 8 | Maximum search expansion rounds. |
| `--max-candidates N` | 200 | Max candidates from the generator per step. |
| `--w-chron W` | 10.0 | Weight for chronogram error. |
| `--w-sem W` | 3.0 | Weight for semantic drift. |
| `--w-len W` | 0.2 | Weight for length change. |
| `--llm` | off | Enable the Ollama LLM generator. |
| `--model MODEL` | mistral | Ollama model name. |
| `--ollama-url URL` | http://localhost:11434 | Ollama API base URL. |

---

## Running tests

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=chronogrammer --cov-report=term-missing

# Run a specific test module
pytest tests/test_scorer.py -v
```

All tests are deterministic and require no network access or external
services.

---

## Architecture

```
src/chronogrammer/
├── __init__.py      Package metadata / version
├── scorer.py        Roman-numeral scoring core
├── semantic.py      Semantic-drift utilities
├── generator.py     Pluggable candidate generators
├── search.py        Beam-search optimiser
└── cli.py           Command-line interface

tests/
├── test_scorer.py   Unit tests for scorer.py
├── test_search.py   Unit tests for search.py and semantic.py
├── test_generator.py Unit tests for generator.py
└── test_cli.py      Integration tests for the CLI
```

### Scoring (`scorer.py`)

Every character is scored independently and case-insensitively using the
purely **additive** convention used in classical chronograms — there is no
subtractive grouping (IV = 6, not 4).

```python
from chronogrammer.scorer import chronogram_score, delta_to_target

chronogram_score("Viva the view!")  # → 17
delta_to_target("Viva the view!", 20)  # → -3  (3 below target)
```

### Semantic similarity (`semantic.py`)

The MVP uses **Jaccard token-set overlap** (with stop-word removal) as a
zero-dependency proxy for semantic similarity.  The `SemanticSimilarity`
protocol makes it easy to drop in a richer backend:

```python
class SemanticSimilarity(Protocol):
    def __call__(self, source: str, candidate: str) -> float: ...
```

### Generators (`generator.py`)

All generators implement:

```python
class CandidateGenerator(Protocol):
    def generate(self, source: str, target: int) -> list[str]: ...
```

**`DeterministicGenerator`** (default) substitutes words from a built-in
synonym bank.  It is fully local and produces reproducible results without
any API calls.

**`OllamaGenerator`** (optional) calls a locally-running Ollama instance to
generate paraphrase candidates using an instruction-tuned LLM.

### Beam search (`search.py`)

The optimiser minimises a weighted objective:

```
score = w_chron × |chronogram(candidate) − target|
      + w_sem   × (1 − semantic_similarity(source, candidate))
      + w_len   × |word_count(candidate) − word_count(source)|
```

Default weights: `w_chron=10`, `w_sem=3`, `w_len=0.2`.  The chronogram
error term is dominant, so an exact match will always beat a near-miss
regardless of semantic similarity.

At each step every beam item is expanded by the active generators, the pool
is deduplicated and the best `beam_width` items survive.  Search stops early
if an exact chronogram is found.

---

## Plugging in a local LLM

The `OllamaGenerator` can be activated via `--llm`.  It requires:

### 1. Install Ollama

Follow the instructions at <https://ollama.ai> for your platform.

### 2. Pull a model

```bash
ollama pull mistral        # ~4 GB, good balance of speed and quality
# or
ollama pull llama3         # Meta Llama 3
# or
ollama pull qwen2.5        # Alibaba Qwen 2.5
```

### 3. Install the `requests` extra

```bash
pip install "chronogrammer[llm]"
```

### 4. Run with `--llm`

```bash
chronogrammer rewrite --target 2025 --llm --model mistral \
  "The scientists announced a major discovery."
```

### Custom backends

Implement `CandidateGenerator` and pass it directly from Python:

```python
from chronogrammer.generator import CandidateGenerator
from chronogrammer.search import beam_search

class MyLLMGenerator:
    def generate(self, source: str, target: int) -> list[str]:
        # Call your LLM here (llama.cpp, vLLM, Hugging Face, …)
        ...

result = beam_search(
    "The scientists announced a major discovery.",
    target=2025,
    generators=[MyLLMGenerator()],
)
print(result.best)
```

### Stronger semantic similarity

Replace `JaccardSimilarity` with an embedding-based backend:

```python
# Example using sentence-transformers (pip install sentence-transformers)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingSimilarity:
    def __call__(self, source: str, candidate: str) -> float:
        embeddings = model.encode([source, candidate])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

result = beam_search(
    "The city honors its founders.",
    target=1776,
    similarity=EmbeddingSimilarity(),
)
```

---

## Extension ideas

- **Stronger semantic similarity** — swap `JaccardSimilarity` for
  `sentence-transformers` embeddings or a local NLI model.
- **Protected spans** — lock named entities, dates, and domain keywords so
  the search only edits freely-substitutable words.
- **Delta-directed editing** — rank synonym substitutions by how close their
  Roman-numeral delta is to the remaining gap, dramatically speeding up
  convergence.
- **Phrase-level synonyms** — extend the synonym bank with multi-word
  equivalents (e.g. `"show" → "make visible"`) for richer coverage.
- **Fluency scoring** — add a perplexity penalty from a small local language
  model to discourage ungrammatical candidates.
- **Interactive mode** — a REPL that lets you accept/reject candidates and
  steer the search interactively.
- **Web UI** — a minimal FastAPI or Gradio front-end.

---

## Licence

MIT.  See `pyproject.toml` for details.

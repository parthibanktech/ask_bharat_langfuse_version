# 📊 Evaluation and Observability Tools

This directory contains offline evaluation tools for the RAG chatbot using TruLens.

## 🎯 Approach: Offline Evaluation

We use **offline evaluation** instead of integrating into the live chatbot because:

1. **Separation of Concerns**: Keep the chatbot simple and fast
2. **Batch Processing**: Evaluate multiple queries and variants at once
3. **Experimentation**: Try different metrics without affecting production
4. **Learning**: Easier to understand metrics and iterate
5. **Performance**: No overhead in the live chatbot
6. **Comparison**: Compare variants side-by-side easily

## 📦 Installation

Install evaluation dependencies using `uv`:

```bash
# Install evaluation optional dependencies (recommended)
uv sync --extra evaluation

# Or install directly
uv pip install trulens trulens-providers-openai
```

Note: The evaluation dependencies are defined as optional in `pyproject.toml`, so they won't be installed by default. Use `--extra evaluation` to include them.

## 🚀 Quick Start

### Evaluate a Single Variant

```bash
# Using uv to run (recommended)
uv run python evaluation_tools/truelens_evaluator.py --variant "Hybrid Retrieval"

# Or activate venv and run directly
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
python evaluation_tools/truelens_evaluator.py --variant "Hybrid Retrieval"
```

### Compare All Variants

```bash
uv run python evaluation_tools/truelens_evaluator.py --compare-all
```

### Use Custom Queries

```bash
uv run python evaluation_tools/truelens_evaluator.py \
  --variant "Fixed Chunking" \
  --queries "What is GST?" "How does MSME work?"
```

## 📈 What Gets Evaluated?

TruLens evaluates three key metrics (the "hallucination triad"):

1. **Groundedness**: Is the answer supported by the retrieved context?
2. **Answer Relevance**: How relevant is the answer to the question?
3. **Context Relevance**: How relevant are the retrieved documents to the question?

## 📁 Output

Results are saved to `evaluation_results/` directory:

- `results_{variant}_{timestamp}.json` - Individual variant results
- `comparison_{timestamp}.json` - Comparison across all variants

Each result file contains:
- Query and generated answer
- Retrieved documents and sources
- Evaluation scores (groundedness, relevance, etc.)
- Leaderboard with aggregated metrics

## 🔍 Viewing Results

### Using TruLens Dashboard

**Easy way (recommended):**
```bash
uv run python evaluation_tools/run_dashboard.py
```

**Or programmatically:**
```python
# Try newer import
try:
    from trulens.dashboard import run_dashboard
    from trulens.core import TruSession
    session = TruSession()
    run_dashboard(session)
except ImportError:
    # Fallback to legacy import
    from trulens_eval import Tru
    tru = Tru()
    tru.run_dashboard()
```

### Using Python

```python
from evaluation_tools.truelens_evaluator import evaluate_offline

# Evaluate a variant
results = evaluate_offline(
    variant_name="Hybrid Retrieval",
    test_queries=[
        "What is the contribution of MSME to GDP?",
        "Why was GST introduced?"
    ]
)

# Access metrics
print(results["leaderboard"])
```

## 🎓 Learning Resources

- [TruLens Documentation](https://www.truera.com/trulens/)
- [RAG Evaluation Guide](https://www.truera.com/trulens/get-started/rag-evaluation/)

## 🔧 Architecture

```
┌─────────────────────────────────────────┐
│   Live Chatbot (chatbot.py)            │
│   - Simple, fast, no evaluation overhead│
│   - Uses rag/pipeline.py                │
└─────────────────────────────────────────┘
                    │
                    │ (uses same pipeline)
                    ▼
┌─────────────────────────────────────────┐
│   Offline Evaluator                     │
│   (truelens_evaluator.py)               │
│   - Wraps pipeline with TruLens         │
│   - Runs batch evaluations              │
│   - Generates metrics & reports         │
└─────────────────────────────────────────┘
```

## 📝 Example: Programmatic Usage

```python
from evaluation_tools.truelens_evaluator import (
    evaluate_offline,
    compare_variants,
    DEFAULT_TEST_QUERIES
)

# Evaluate single variant
results = evaluate_offline(
    variant_name="Hybrid Retrieval",
    test_queries=DEFAULT_TEST_QUERIES
)

# Compare all variants
comparison = compare_variants(
    variant_names=["Fixed Chunking", "Hybrid Retrieval", "Cross-Encoder"],
    test_queries=DEFAULT_TEST_QUERIES
)
```

## ⚙️ Configuration

Set environment variables in `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

TruLens uses OpenAI for feedback functions (evaluating groundedness, relevance, etc.).

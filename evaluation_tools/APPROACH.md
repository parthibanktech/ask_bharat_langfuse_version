# 🎯 Evaluation Approach: Offline vs Live Integration

## Decision: **Offline Evaluation** ✅

After analyzing the requirements, we've chosen **offline evaluation** over live integration.

## Why Offline Evaluation?

### 1. **Separation of Concerns**
- **Live Chatbot**: Focuses on user experience, speed, and reliability
- **Evaluation**: Runs separately, doesn't affect production performance

### 2. **Learning Benefits**
- ✅ Run batch evaluations on multiple queries
- ✅ Compare variants side-by-side easily
- ✅ Experiment with different metrics without risk
- ✅ Understand evaluation metrics in isolation
- ✅ Iterate quickly on evaluation strategies

### 3. **Performance**
- No overhead in production chatbot
- Can run evaluations on schedule (e.g., nightly)
- Doesn't slow down user interactions

### 4. **Flexibility**
- Test with different query sets
- Compare multiple variants simultaneously
- Generate reports and visualizations
- Store results for historical analysis

## Architecture

```
┌─────────────────────────────────────┐
│   Production Chatbot               │
│   (chatbot.py)                      │
│   - Fast, simple, reliable          │
│   - Uses rag/pipeline.py            │
│   - No evaluation overhead          │
└─────────────────────────────────────┘
              │
              │ (shared pipeline)
              ▼
┌─────────────────────────────────────┐
│   RAG Pipeline                      │
│   (rag/pipeline.py)                 │
│   - Retrieval + Generation           │
│   - Used by both chatbot & eval     │
└─────────────────────────────────────┘
              │
              │ (wrapped with TruLens)
              ▼
┌─────────────────────────────────────┐
│   Offline Evaluator                 │
│   (evaluation_tools/truelens_...)   │
│   - Wraps pipeline with TruLens    │
│   - Runs batch evaluations          │
│   - Generates metrics & reports     │
└─────────────────────────────────────┘
```

## How It Works

1. **Instrumentation**: The `InstrumentedRAG` class wraps our existing pipeline with TruLens decorators
2. **Feedback Functions**: Three metrics are evaluated:
   - **Groundedness**: Is answer supported by context?
   - **Answer Relevance**: How relevant is answer to question?
   - **Context Relevance**: How relevant are retrieved docs?
3. **Batch Processing**: Run multiple queries through the pipeline
4. **Results Storage**: Save evaluation results to JSON files
5. **Analysis**: View results via TruLens dashboard or programmatically

## Usage Examples

### Command Line (using uv)
```bash
# Evaluate single variant
uv run python evaluation_tools/truelens_evaluator.py --variant "Hybrid Retrieval"

# Compare all variants
uv run python evaluation_tools/truelens_evaluator.py --compare-all
```

### Python API
```python
from evaluation_tools.truelens_evaluator import evaluate_offline

results = evaluate_offline(
    variant_name="Hybrid Retrieval",
    test_queries=["What is GST?", "How does MSME work?"]
)
```

## When to Use Live Integration?

Live integration would be useful for:
- Real-time monitoring in production
- A/B testing different variants
- Continuous evaluation as users interact

However, for **learning purposes**, offline evaluation is better because:
- You can control the test set
- You can compare variants fairly
- You can iterate quickly
- You can understand metrics deeply

## Next Steps

1. Install TruLens: `uv sync --extra evaluation` (or `uv pip install trulens trulens-providers-openai`)
2. Run evaluation: `uv run python evaluation_tools/truelens_evaluator.py --compare-all`
3. View results: Check `evaluation_results/` directory
4. Analyze: Use TruLens dashboard or JSON files


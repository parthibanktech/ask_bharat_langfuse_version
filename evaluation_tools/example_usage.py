"""
Example: Using TruLens Offline Evaluator
-----------------------------------------
Simple examples showing how to use the offline evaluation tool.

Run this from the project root:
    uv run python evaluation_tools/example_usage.py
    
Or with activated venv:
    python evaluation_tools/example_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_tools.truelens_evaluator import (
    evaluate_offline,
    compare_variants,
    DEFAULT_TEST_QUERIES
)

# Example 1: Evaluate a single variant
print("Example 1: Evaluating Hybrid Retrieval variant...")
results = evaluate_offline(
    variant_name="Hybrid Retrieval",
    test_queries=DEFAULT_TEST_QUERIES[:3]  # Use first 3 queries
)
print(f"✓ Evaluation complete. Results saved to evaluation_results/")

# Example 2: Compare multiple variants
print("\nExample 2: Comparing variants...")
comparison = compare_variants(
    variant_names=["Fixed Chunking", "Hybrid Retrieval"],
    test_queries=DEFAULT_TEST_QUERIES[:2]  # Use first 2 queries
)
print(f"✓ Comparison complete. Results saved to evaluation_results/")

# Example 3: Custom queries
print("\nExample 3: Using custom queries...")
custom_queries = [
    "What is the contribution of MSME to GDP?",
    "Explain GST benefits for small businesses"
]
results = evaluate_offline(
    variant_name="Cross-Encoder",
    test_queries=custom_queries
)
print(f"✓ Custom evaluation complete.")

print("\n" + "="*60)
print("All evaluations complete!")
print("Check evaluation_results/ directory for detailed results.")
print("="*60)


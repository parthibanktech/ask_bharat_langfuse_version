"""
📊 TruLens Offline RAG Evaluator
---------------------------------
Offline evaluation tool for RAG pipeline using TruLens.

This script:
1. Loads test queries
2. Runs them through the RAG pipeline with TruLens instrumentation
3. Evaluates using feedback functions (groundedness, relevance, etc.)
4. Saves results for analysis
5. Generates comparison reports

Usage:
    python evaluation_tools/truelens_evaluator.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import TruLens - fail gracefully if not installed
# Try both import styles (newer trulens package and older trulens_eval)
TRULENS_AVAILABLE = False
try:
    # Try newer package structure (trulens-eval on PyPI, imported as trulens)
    from trulens.core import TruSession, Feedback
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.providers.openai import OpenAI as TruLensOpenAI
    from trulens.apps.app import TruApp
    import numpy as np
    TRULENS_AVAILABLE = True
    TRULENS_IMPORT_STYLE = "new"
except ImportError:
    try:
        # Try older/alternative import style
        from trulens_eval import Tru
        import numpy as np
        TRULENS_AVAILABLE = True
        TRULENS_IMPORT_STYLE = "legacy"
        logger.warning("Using legacy TruLens import. Some features may not be available.")
    except ImportError as e:
        logger.error(f"TruLens not available: {e}")
        logger.error("Install with: uv sync --extra evaluation")
        logger.error("  or: uv pip install trulens-eval trulens-providers-openai")
        TRULENS_AVAILABLE = False
        TRULENS_IMPORT_STYLE = None

# Import project modules
from vector_db_manager import VARIANT_CONFIG
from rag.pipeline import run_rag

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class InstrumentedRAG:
    """
    Wrapper around the RAG pipeline with TruLens instrumentation.
    This allows TruLens to trace retrieval and generation steps.
    """
    
    def __init__(self, variant_name: str):
        self.variant_name = variant_name
    
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str, top_k: int = 5) -> tuple:
        """
        Instrumented retrieval step.
        TruLens will trace this to measure context relevance.
        """
        from vector_db_manager import get_vector_db
        from rag.retrieval import RETRIEVAL_FUNCTIONS
        
        collection = get_vector_db(self.variant_name, show_progress=False)
        retrieve_func = RETRIEVAL_FUNCTIONS[self.variant_name]
        documents, sources = retrieve_func(collection, query, top_k=top_k)
        
        # Return as tuple for TruLens to track
        return documents, sources
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, documents: List[str]) -> str:
        """
        Instrumented generation step.
        TruLens will trace this to measure answer quality.
        """
        from rag.generation import generate_answer
        return generate_answer(query, documents)
    
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query with instrumentation.
        This is the main entry point that TruLens will track.
        """
        documents, sources = self.retrieve(query, top_k=top_k)
        answer = self.generate_completion(query, documents)
        
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "sources": sources,
            "variant_name": self.variant_name
        }


def setup_feedback_functions():
    """
    Setup TruLens feedback functions for evaluation.
    
    Returns:
        List of feedback functions: groundedness, answer_relevance, context_relevance
    """
    if not TRULENS_AVAILABLE:
        return []
    
    if TRULENS_IMPORT_STYLE != "new":
        logger.warning("Legacy TruLens import detected. Feedback functions may not work correctly.")
        return []
    
    # Initialize OpenAI provider for feedback
    # Note: model_engine should match your OpenAI model
    provider = TruLensOpenAI(model_engine="gpt-4o-mini")
    
    # 1. Groundedness: Is the answer supported by the context?
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons_consider_answerability,
            name="Groundedness",
        )
        .on_context(collect_list=True)
        .on_output()
        .on_input()
    )
    
    # 2. Answer Relevance: How relevant is the answer to the question?
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()
        .on_output()
    )
    
    # 3. Context Relevance: How relevant is each context chunk to the question?
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, 
            name="Context Relevance"
        )
        .on_input()
        .on_context(collect_list=False)
        .aggregate(np.mean)  # Average across all context chunks
    )
    
    return [f_groundedness, f_answer_relevance, f_context_relevance]


def evaluate_offline(
    variant_name: str,
    test_queries: List[str],
    output_dir: str = "evaluation_results",
    app_version: str = None
) -> Dict[str, Any]:
    """
    Run offline evaluation for a RAG variant.
    
    Args:
        variant_name: Name of the RAG variant to evaluate
        test_queries: List of test queries to evaluate
        output_dir: Directory to save results
        app_version: Version identifier for this evaluation run
        
    Returns:
        Dictionary with evaluation results
    """
    if not TRULENS_AVAILABLE:
        raise ImportError("TruLens is not installed. Install with: pip install trulens trulens-providers-openai")
    
    logger.info(f"Starting offline evaluation for variant: {variant_name}")
    
    # Initialize TruLens session
    session = TruSession()
    session.reset_database()  # Start fresh for this evaluation
    
    # Create instrumented RAG
    rag = InstrumentedRAG(variant_name)
    
    # Setup feedback functions
    feedbacks = setup_feedback_functions()
    
    # Wrap with TruApp
    app_version = app_version or f"{variant_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tru_rag = TruApp(
        rag,
        app_name="AskBharat_RAG",
        app_version=app_version,
        feedbacks=feedbacks,
    )
    
    # Run evaluation
    results = []
    with tru_rag as recording:
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Evaluating query {i}/{len(test_queries)}: {query[:50]}...")
            try:
                result = rag.query(query, top_k=5)
                results.append({
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Get leaderboard (evaluation metrics)
    try:
        leaderboard = session.get_leaderboard()
        logger.info("Evaluation complete. Leaderboard:")
        logger.info(leaderboard)
    except Exception as e:
        logger.warning(f"Could not get leaderboard: {e}")
        leaderboard = None
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"results_{variant_name}_{timestamp}.json"
    
    evaluation_summary = {
        "variant_name": variant_name,
        "app_version": app_version,
        "timestamp": timestamp,
        "num_queries": len(test_queries),
        "results": results,
        "leaderboard": leaderboard.to_dict() if leaderboard is not None else None
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    return evaluation_summary


def compare_variants(
    variant_names: List[str],
    test_queries: List[str],
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Compare multiple RAG variants on the same test queries.
    
    Args:
        variant_names: List of variant names to compare
        test_queries: List of test queries
        output_dir: Directory to save comparison results
        
    Returns:
        Comparison summary
    """
    logger.info(f"Comparing variants: {variant_names}")
    
    all_results = {}
    for variant_name in variant_names:
        if variant_name not in VARIANT_CONFIG:
            logger.warning(f"Skipping unknown variant: {variant_name}")
            continue
        
        try:
            results = evaluate_offline(
                variant_name=variant_name,
                test_queries=test_queries,
                output_dir=output_dir,
                app_version=f"comparison_{variant_name}"
            )
            all_results[variant_name] = results
        except Exception as e:
            logger.error(f"Error evaluating {variant_name}: {e}")
            all_results[variant_name] = {"error": str(e)}
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_path / f"comparison_{timestamp}.json"
    
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Comparison saved to: {comparison_file}")
    
    return all_results


# Default test queries for AskBharat
DEFAULT_TEST_QUERIES = [
    "What is the contribution of MSME to GDP?",
    "Why was GST introduced?",
    "What are the benefits of GST?",
    "How does GST impact small businesses?",
    "What are the key features of MSME policies?",
]


if __name__ == "__main__":
    """
    Main entry point for offline evaluation.
    
    Examples (using uv):
        # Evaluate single variant
        uv run python evaluation_tools/truelens_evaluator.py --variant "Hybrid Retrieval"
        
        # Compare all variants
        uv run python evaluation_tools/truelens_evaluator.py --compare-all
        
        # Use custom queries
        uv run python evaluation_tools/truelens_evaluator.py --variant "Fixed Chunking" --queries "query1" "query2"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline RAG evaluation with TruLens")
    parser.add_argument(
        "--variant",
        type=str,
        help="RAG variant to evaluate (e.g., 'Hybrid Retrieval')",
        choices=list(VARIANT_CONFIG.keys())
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all available variants"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Custom test queries (default: uses predefined queries)",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Determine test queries
    test_queries = args.queries if args.queries else DEFAULT_TEST_QUERIES
    
    if args.compare_all:
        # Compare all variants
        variant_names = list(VARIANT_CONFIG.keys())
        compare_variants(
            variant_names=variant_names,
            test_queries=test_queries,
            output_dir=args.output_dir
        )
    elif args.variant:
        # Evaluate single variant
        evaluate_offline(
            variant_name=args.variant,
            test_queries=test_queries,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()
        logger.info("\nExample usage (using uv):")
        logger.info("  uv run python evaluation_tools/truelens_evaluator.py --variant 'Hybrid Retrieval'")
        logger.info("  uv run python evaluation_tools/truelens_evaluator.py --compare-all")
        logger.info("\nOr with activated venv:")
        logger.info("  python evaluation_tools/truelens_evaluator.py --variant 'Hybrid Retrieval'")
        logger.info("  python evaluation_tools/truelens_evaluator.py --compare-all")


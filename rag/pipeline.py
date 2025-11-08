"""
RAG Pipeline
------------
Simple function-based RAG pipeline that orchestrates retrieval and generation.
Integrated with Langfuse for observability and monitoring.
"""

import os
import logging
from vector_db_manager import get_vector_db, VARIANT_CONFIG
from rag.retrieval import RETRIEVAL_FUNCTIONS
from rag.generation import generate_answer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

logger = logging.getLogger(__name__)

# Import Langfuse - simple as per reference
try:
    from langfuse import observe, get_client
except ImportError:
    observe = None

# Apply @observe() decorator if Langfuse is available (matches reference pattern)
if observe is not None:
    @observe()  # decorator to automatically create trace and nest generations
    def run_rag(variant_name, query, top_k=5):
        """
        Run the complete RAG pipeline for a query.
        
        This is the main function that:
        1. Loads the vector database for the variant
        2. Retrieves relevant documents
        3. Generates an answer
        4. Tracks everything with Langfuse (if configured)
        
        Args:
            variant_name: Name of the RAG variant (e.g., "Hybrid Retrieval")
            query: User's question
            top_k: Number of documents to retrieve (default: 5)
            
        Returns:
            Dictionary with:
                - query: The original query
                - answer: Generated answer
                - documents: List of retrieved document texts
                - sources: List of source filenames
                - variant_name: Name of the variant used
                
        Raises:
            ValueError: If variant_name is not in VARIANT_CONFIG
            KeyError: If retrieval function not found for variant
        """
        # Validate variant name
        if variant_name not in VARIANT_CONFIG:
            raise ValueError(f"Unknown variant: {variant_name}. Available: {list(VARIANT_CONFIG.keys())}")
        
        # Get retrieval function for this variant
        if variant_name not in RETRIEVAL_FUNCTIONS:
            raise KeyError(f"No retrieval function found for variant: {variant_name}")
        
        retrieve_func = RETRIEVAL_FUNCTIONS[variant_name]
        
        # Step 1: Load vector database
        collection = get_vector_db(variant_name, show_progress=False)
        
        # Step 2: Retrieve documents
        documents, sources = retrieve_func(collection, query, top_k=top_k)
        
        # Step 3: Generate answer (automatically tracked by langfuse.openai)
        answer = generate_answer(query, documents)
        
        # Step 4: Return structured result
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "sources": sources,
            "variant_name": variant_name
        }
else:
    # If Langfuse not available, just run without decorator
    def run_rag(variant_name, query, top_k=5):
        """Run RAG pipeline without Langfuse tracing."""
        # Validate variant name
        if variant_name not in VARIANT_CONFIG:
            raise ValueError(f"Unknown variant: {variant_name}. Available: {list(VARIANT_CONFIG.keys())}")
        
        # Get retrieval function for this variant
        if variant_name not in RETRIEVAL_FUNCTIONS:
            raise KeyError(f"No retrieval function found for variant: {variant_name}")
        
        retrieve_func = RETRIEVAL_FUNCTIONS[variant_name]
        
        # Step 1: Load vector database
        collection = get_vector_db(variant_name, show_progress=False)
        
        # Step 2: Retrieve documents
        documents, sources = retrieve_func(collection, query, top_k=top_k)
        
        # Step 3: Generate answer
        answer = generate_answer(query, documents)
        
        # Step 4: Return structured result
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "sources": sources,
            "variant_name": variant_name
        }


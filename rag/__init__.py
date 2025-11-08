"""
RAG Pipeline Module
-------------------
Simple function-based RAG pipeline separated from UI.
"""

# Lazy import to avoid loading heavy dependencies at module level
__all__ = ['run_rag']

def __getattr__(name):
    if name == 'run_rag':
        from rag.pipeline import run_rag
        return run_rag
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


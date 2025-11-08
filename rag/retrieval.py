"""
Retrieval Functions
------------------
All retrieval strategies as simple functions.
One function per retrieval method.
"""

import os
import logging
from chromadb import Collection
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

logger = logging.getLogger(__name__)


def simple_retrieve(collection, query, top_k=5):
    """
    Basic semantic retrieval from vector database.
    
    Used by: Fixed Chunking, Sentence Overlap
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (documents, sources)
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    sources = [m.get("source", "") for m in metadatas]
    return documents, sources


def hybrid_retrieve(collection, query, top_k=5):
    """
    Hybrid retrieval using TF-IDF + semantic search.
    
    Used by: Hybrid Retrieval variant
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (documents, sources)
    """
    # Get all documents from collection
    all_results = collection.get()
    all_docs = all_results["documents"]
    all_metadatas = all_results["metadatas"]
    
    # TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_docs)
    qv = vectorizer.transform([query])
    lexical_scores = (tfidf @ qv.T).toarray().ravel()
    
    # Semantic scores
    semantic_results = collection.query(query_texts=[query], n_results=top_k)
    semantic_docs = semantic_results["documents"][0]
    semantic_distances = semantic_results["distances"][0]
    
    # Combine scores (alpha=0.5 for semantic weight)
    alpha = 0.5
    combined = []
    for doc, dist in zip(semantic_docs, semantic_distances):
        try:
            idx = all_docs.index(doc)
            lex_score = lexical_scores[idx]
            combined_score = alpha * (1 - dist) + (1 - alpha) * lex_score
            metadata = all_metadatas[idx]
            combined.append((doc, metadata, combined_score))
        except ValueError:
            pass
    
    # Sort and return top_k
    combined.sort(key=lambda x: x[2], reverse=True)
    top = combined[:top_k]
    
    documents = [doc for doc, _, _ in top]
    sources = [meta.get("source", "") for _, meta, _ in top]
    
    return documents, sources


def cross_encoder_retrieve(collection, query, top_k=5):
    """
    Retrieve with cross-encoder reranking (local model).
    
    Used by: Cross-Encoder variant
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (documents, sources)
    """
    # Lazy import to avoid loading heavy dependencies
    from utils.embeddings import CrossEncoderReranker
    
    # Get more candidates
    documents, sources = simple_retrieve(collection, query, top_k=top_k * 2)
    
    # Rerank with cross-encoder
    reranker = CrossEncoderReranker()
    ranked_indices = reranker.rerank(query, documents, top_k=top_k)
    
    # Return reranked
    reranked_docs = [documents[i] for i in ranked_indices]
    reranked_sources = [sources[i] for i in ranked_indices]
    
    return reranked_docs, reranked_sources


def llm_reranking_retrieve(collection, query, top_k=5):
    """
    Retrieve with LLM-based reranking (OpenAI).
    
    Used by: Reranking variant
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (documents, sources)
    """
    # Get more candidates using hybrid retrieval
    documents, sources = hybrid_retrieve(collection, query, top_k=top_k * 2)
    
    if not documents:
        return [], []
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to simple retrieval if no API key
        logger.warning("OPENAI_API_KEY not set. Returning unreranked results.")
        return documents[:top_k], sources[:top_k]
    
    # Create reranking prompt
    prompt = (
        f"Rank the following passages in order of relevance to the query:\n"
        f"Query: {query}\n\n"
        "Passages:\n" +
        "\n\n".join([f"{i+1}. {doc[:500]}" for i, doc in enumerate(documents)]) +
        "\n\nReturn the new order as numbers separated by commas."
    )
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        order_text = response.choices[0].message.content.strip()
        # Parse numeric order safely
        indices = [int(x)-1 for x in order_text.replace(",", " ").split() if x.isdigit()]
        indices = [i for i in indices if 0 <= i < len(documents)]
        
        if indices:
            reranked_docs = [documents[i] for i in indices[:top_k]]
            reranked_sources = [sources[i] for i in indices[:top_k]]
            return reranked_docs, reranked_sources
        else:
            # Fallback to original order
            logger.warning("Could not parse LLM reranking order. Returning original order.")
            return documents[:top_k], sources[:top_k]
    except Exception as e:
        logger.warning(f"LLM reranking failed: {e}. Returning original order.")
        return documents[:top_k], sources[:top_k]


# Mapping: variant name -> retrieval function
RETRIEVAL_FUNCTIONS = {
    "Fixed Chunking": simple_retrieve,
    "Sentence Overlap": simple_retrieve,
    "Hybrid Retrieval": hybrid_retrieve,
    "Reranking": llm_reranking_retrieve,
    "Cross-Encoder": cross_encoder_retrieve,
}


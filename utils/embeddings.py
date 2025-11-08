"""
🧩 Local Embeddings & Cross-Encoder Utilities
----------------------------------------------
Provides local embedding function for ChromaDB and cross-encoder reranking.
Uses HuggingFace sentence-transformers models to eliminate OpenAI API costs.

Models:
- Bi-Encoder (for retrieval): msmarco-MiniLM-L-6-v3
- Cross-Encoder (for reranking): cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import logging

logger = logging.getLogger(__name__)


class LocalEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using sentence-transformers.
    Uses MS MARCO trained MiniLM model for efficient retrieval.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/msmarco-MiniLM-L-6-v3'):
        """
        Initialize the embedding function.
        
        Args:
            model_name: HuggingFace model name (bi-encoder)
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"LocalEmbeddingFunction initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"Loading bi-encoder model: {self.model_name}")
            logger.info("(This may download ~80MB on first use)")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embeddings (as lists of floats)
        """
        try:
            # Process in batches for efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(input), batch_size):
                batch = input[i:i + batch_size]
                embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.extend(embeddings.tolist())
            
            logger.debug(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise


class CrossEncoderReranker:
    """
    Cross-encoder for reranking retrieved documents.
    More accurate than bi-encoders but slower (only use on top-k candidates).
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            logger.info("(This may download ~80MB on first use)")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model
    
    def rerank(self, query: str, documents: list[str], top_k: int = None) -> list[int]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of indices in ranked order (most relevant first)
        """
        try:
            if not documents:
                return []
            
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get relevance scores
            logger.debug(f"Reranking {len(documents)} documents...")
            scores = self.model.predict(pairs)
            
            # Create (index, score) pairs and sort by score (descending)
            scored_indices = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Extract indices in ranked order
            ranked_indices = [idx for idx, score in scored_indices]
            
            # Log top scores
            if scored_indices:
                top_score = scored_indices[0][1]
                logger.info(f"Reranking complete. Top score: {top_score:.4f}")
            
            # Return top_k if specified
            return ranked_indices[:top_k] if top_k else ranked_indices
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on failure
            return list(range(len(documents)))


# Convenience function for reranking
def rerank_with_cross_encoder(
    query: str,
    documents: list[str],
    top_k: int = None,
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) -> list[int]:
    """
    Convenience function to rerank documents using cross-encoder.
    
    Args:
        query: The search query
        documents: List of document texts to rerank
        top_k: Number of top results to return (None = all)
        model_name: HuggingFace cross-encoder model name
        
    Returns:
        List of indices in ranked order (most relevant first)
    """
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, documents, top_k)


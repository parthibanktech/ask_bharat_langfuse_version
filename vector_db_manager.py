"""
🗂️ Vector Database Manager
---------------------------
Simple manager to load or build vector databases for each RAG variant.

Keeps all variants in separate folders:
- vector_dbs/fixed_chunking/
- vector_dbs/sentence_overlap/
- vector_dbs/hybrid_retrieval/
- vector_dbs/reranking/
- vector_dbs/cross_encoder/

If a database exists, loads it. If not, builds it once.
"""

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from utils.embeddings import LocalEmbeddingFunction
from loaders.pdf_loader import load_all_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
import os
import logging

logger = logging.getLogger(__name__)

# Configuration for each variant
VARIANT_CONFIG = {
    "Fixed Chunking": {
        "folder": "fixed_chunking",
        "splitter_type": "recursive",
        "chunk_size": 2000,
        "overlap": 0,
        "embedding": "local"
    },
    "Sentence Overlap": {
        "folder": "sentence_overlap",
        "splitter_type": "nltk",
        "chunk_size": 200,
        "overlap": 50,
        "embedding": "local"
    },
    "Hybrid Retrieval": {
        "folder": "hybrid_retrieval",
        "splitter_type": "nltk",
        "chunk_size": 200,
        "overlap": 50,
        "embedding": "local"
    },
    "Reranking": {
        "folder": "reranking",
        "splitter_type": "nltk",
        "chunk_size": 200,
        "overlap": 50,
        "embedding": "local"
    },
    "Cross-Encoder": {
        "folder": "cross_encoder",
        "splitter_type": "nltk",
        "chunk_size": 200,
        "overlap": 50,
        "embedding": "local"
    }
}


def get_db_path(variant_name):
    """Get folder path for a variant's vector database."""
    config = VARIANT_CONFIG[variant_name]
    return f"vector_dbs/{config['folder']}"


def db_exists(variant_name):
    """Check if vector database already exists."""
    path = get_db_path(variant_name)
    return os.path.exists(path) and os.path.isdir(path)


def chunk_documents(texts, sources, config):
    """
    Chunk documents based on variant config.
    
    Args:
        texts: List of document texts
        sources: List of source filenames
        config: Configuration dict with splitter_type, chunk_size, overlap
        
    Returns:
        List of chunked documents
    """
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]
    
    # Choose splitter
    if config["splitter_type"] == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    else:  # nltk
        splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    
    # Create metadata
    metadatas = [{"source": s} for s in sources]
    
    # Chunk
    chunks = splitter.create_documents(texts, metadatas=metadatas)
    logger.info(f"Created {len(chunks)} chunks using {config['splitter_type']} splitter")
    
    return chunks


def build_vector_db(variant_name, show_progress=False):
    """
    Build vector database for a variant.
    
    Args:
        variant_name: Name of the variant (e.g., "Cross-Encoder")
        show_progress: Whether to show progress messages
        
    Returns:
        ChromaDB collection
    """
    config = VARIANT_CONFIG[variant_name]
    db_path = get_db_path(variant_name)
    
    if show_progress:
        logger.info(f"Building vector database for {variant_name}...")
        logger.info(f"Config: {config['splitter_type']} splitter, size={config['chunk_size']}, overlap={config['overlap']}")
    
    # Load PDFs
    texts, sources = load_all_pdfs("data/", return_filenames=True)
    
    # Chunk documents
    chunks = chunk_documents(texts, sources, config)
    
    # Create persistent client
    client = PersistentClient(path=db_path)
    
    # Choose embedding function
    if config["embedding"] == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
        if show_progress:
            logger.info("Using OpenAI embeddings")
    else:  # local
        ef = LocalEmbeddingFunction()
        if show_progress:
            logger.info("Using local embeddings (sentence-transformers)")
    
    # Create collection
    collection = client.get_or_create_collection("askbharat", embedding_function=ef)
    
    # Add documents
    if show_progress:
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
    
    collection.add(
        ids=[f"{config['folder']}-{i}" for i in range(len(chunks))],
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks]
    )
    
    if show_progress:
        logger.info(f"✓ Vector database built and saved to {db_path}")
    
    return collection


def load_vector_db(variant_name):
    """
    Load existing vector database for a variant.
    
    Args:
        variant_name: Name of the variant
        
    Returns:
        ChromaDB collection
    """
    config = VARIANT_CONFIG[variant_name]
    db_path = get_db_path(variant_name)
    
    # Create persistent client
    client = PersistentClient(path=db_path)
    
    # Choose embedding function (must match what was used to build)
    if config["embedding"] == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
    else:  # local
        ef = LocalEmbeddingFunction()
    
    # Get collection
    collection = client.get_or_create_collection("askbharat", embedding_function=ef)
    
    logger.info(f"Loaded vector database for {variant_name} from {db_path}")
    
    return collection


def get_vector_db(variant_name, show_progress=False):
    """
    Get vector database for a variant.
    Loads if exists, builds if not.
    
    Args:
        variant_name: Name of the variant (e.g., "Cross-Encoder")
        show_progress: Whether to show progress messages
        
    Returns:
        ChromaDB collection
    """
    if db_exists(variant_name):
        logger.info(f"Loading existing database for {variant_name}")
        return load_vector_db(variant_name)
    else:
        logger.info(f"Database not found for {variant_name}, building...")
        return build_vector_db(variant_name, show_progress=show_progress)


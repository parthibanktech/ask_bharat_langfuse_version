"""
Tests for retrieval functions
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag.retrieval import (
    simple_retrieve,
    hybrid_retrieve,
    cross_encoder_retrieve,
    llm_reranking_retrieve,
    RETRIEVAL_FUNCTIONS
)


class TestSimpleRetrieve:
    """Test simple_retrieve function"""
    
    def test_simple_retrieve_basic(self):
        """Test basic retrieval returns documents and sources"""
        # Mock collection
        collection = Mock()
        collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"source": "file1.pdf"}, {"source": "file2.pdf"}, {"source": "file3.pdf"}]]
        }
        
        documents, sources = simple_retrieve(collection, "test query", top_k=3)
        
        assert len(documents) == 3
        assert len(sources) == 3
        assert documents == ["doc1", "doc2", "doc3"]
        assert sources == ["file1.pdf", "file2.pdf", "file3.pdf"]
        collection.query.assert_called_once_with(query_texts=["test query"], n_results=3)
    
    def test_simple_retrieve_empty(self):
        """Test retrieval with empty results"""
        collection = Mock()
        collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]]
        }
        
        documents, sources = simple_retrieve(collection, "test query", top_k=5)
        
        assert documents == []
        assert sources == []


class TestHybridRetrieve:
    """Test hybrid_retrieve function"""
    
    @patch('rag.retrieval.TfidfVectorizer')
    def test_hybrid_retrieve_basic(self, mock_vectorizer_class):
        """Test hybrid retrieval combines TF-IDF and semantic search"""
        import numpy as np
        
        # Mock collection
        collection = Mock()
        collection.get.return_value = {
            "documents": ["doc1", "doc2", "doc3"],
            "metadatas": [{"source": "file1.pdf"}, {"source": "file2.pdf"}, {"source": "file3.pdf"}]
        }
        collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]]
        }
        
        # Mock TF-IDF
        mock_vectorizer = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        mock_tfidf = Mock()
        mock_qv = Mock()
        mock_vectorizer.fit_transform.return_value = mock_tfidf
        mock_vectorizer.transform.return_value = mock_qv
        
        # Create proper numpy array mock
        mock_result = Mock()
        mock_result.toarray.return_value = np.array([[0.5, 0.3, 0.2]])
        mock_tfidf.__matmul__ = Mock(return_value=mock_result)
        
        documents, sources = hybrid_retrieve(collection, "test query", top_k=2)
        
        assert len(documents) <= 2
        assert len(sources) <= 2
        collection.get.assert_called_once()
        collection.query.assert_called_once()


class TestCrossEncoderRetrieve:
    """Test cross_encoder_retrieve function"""
    
    @patch('rag.retrieval.simple_retrieve')
    @patch('utils.embeddings.CrossEncoderReranker')
    def test_cross_encoder_retrieve(self, mock_reranker_class, mock_simple_retrieve):
        """Test cross-encoder retrieval reranks results"""
        # Mock simple retrieve
        mock_simple_retrieve.return_value = (
            ["doc1", "doc2", "doc3", "doc4"],
            ["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"]
        )
        
        # Mock reranker
        mock_reranker = Mock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.rerank.return_value = [2, 0, 1]  # Reordered indices
        
        collection = Mock()
        documents, sources = cross_encoder_retrieve(collection, "test query", top_k=3)
        
        assert len(documents) == 3
        assert len(sources) == 3
        assert documents == ["doc3", "doc1", "doc2"]  # Reordered
        mock_simple_retrieve.assert_called_once_with(collection, "test query", top_k=6)
        mock_reranker.rerank.assert_called_once()


class TestLLMRerankingRetrieve:
    """Test llm_reranking_retrieve function"""
    
    @patch('rag.retrieval.hybrid_retrieve')
    @patch.dict('os.environ', {}, clear=True)
    def test_llm_reranking_no_api_key(self, mock_hybrid_retrieve):
        """Test LLM reranking falls back when no API key"""
        mock_hybrid_retrieve.return_value = (
            ["doc1", "doc2", "doc3", "doc4"],
            ["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"]
        )
        
        collection = Mock()
        documents, sources = llm_reranking_retrieve(collection, "test query", top_k=2)
        
        assert len(documents) == 2
        assert len(sources) == 2
        mock_hybrid_retrieve.assert_called_once_with(collection, "test query", top_k=4)
    
    @patch('rag.retrieval.hybrid_retrieve')
    @patch('rag.retrieval.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_llm_reranking_with_api_key(self, mock_openai_class, mock_hybrid_retrieve):
        """Test LLM reranking with API key"""
        mock_hybrid_retrieve.return_value = (
            ["doc1", "doc2", "doc3", "doc4"],
            ["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"]
        )
        
        # Mock OpenAI response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="2, 1, 3, 4"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        collection = Mock()
        documents, sources = llm_reranking_retrieve(collection, "test query", top_k=2)
        
        assert len(documents) == 2
        assert len(sources) == 2
        mock_hybrid_retrieve.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()


class TestRetrievalFunctionsMapping:
    """Test RETRIEVAL_FUNCTIONS mapping"""
    
    def test_all_variants_have_retrieval_functions(self):
        """Test that all variants have retrieval functions"""
        from vector_db_manager import VARIANT_CONFIG
        
        for variant_name in VARIANT_CONFIG.keys():
            assert variant_name in RETRIEVAL_FUNCTIONS, f"Missing retrieval function for {variant_name}"
            assert callable(RETRIEVAL_FUNCTIONS[variant_name]), f"Retrieval function for {variant_name} is not callable"


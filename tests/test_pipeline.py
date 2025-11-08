"""
Tests for RAG pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag.pipeline import run_rag
from vector_db_manager import VARIANT_CONFIG


class TestRunRag:
    """Test run_rag function"""
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_run_rag_simple_variant(self, mock_generate, mock_get_db):
        """Test run_rag with simple retrieval variant"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        # Mock vector DB
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        
        # Mock retrieval function
        mock_retrieve = Mock(return_value=(
            ["doc1", "doc2"],
            ["file1.pdf", "file2.pdf"]
        ))
        
        # Temporarily replace the function in the dict
        original_func = RETRIEVAL_FUNCTIONS["Fixed Chunking"]
        RETRIEVAL_FUNCTIONS["Fixed Chunking"] = mock_retrieve
        
        try:
            # Mock generation
            mock_generate.return_value = "This is the answer."
            
            result = run_rag("Fixed Chunking", "test query", top_k=2)
            
            assert result["query"] == "test query"
            assert result["answer"] == "This is the answer."
            assert result["documents"] == ["doc1", "doc2"]
            assert result["sources"] == ["file1.pdf", "file2.pdf"]
            assert result["variant_name"] == "Fixed Chunking"
            
            mock_get_db.assert_called_once_with("Fixed Chunking", show_progress=False)
            mock_retrieve.assert_called_once_with(mock_collection, "test query", top_k=2)
            mock_generate.assert_called_once_with("test query", ["doc1", "doc2"])
        finally:
            # Restore original function
            RETRIEVAL_FUNCTIONS["Fixed Chunking"] = original_func
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_run_rag_hybrid_variant(self, mock_generate, mock_get_db):
        """Test run_rag with hybrid retrieval variant"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        mock_retrieve = Mock(return_value=(["doc1"], ["file1.pdf"]))
        mock_generate.return_value = "Answer"
        
        original_func = RETRIEVAL_FUNCTIONS["Hybrid Retrieval"]
        RETRIEVAL_FUNCTIONS["Hybrid Retrieval"] = mock_retrieve
        
        try:
            result = run_rag("Hybrid Retrieval", "test query", top_k=5)
            assert result["variant_name"] == "Hybrid Retrieval"
            mock_retrieve.assert_called_once()
        finally:
            RETRIEVAL_FUNCTIONS["Hybrid Retrieval"] = original_func
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_run_rag_cross_encoder_variant(self, mock_generate, mock_get_db):
        """Test run_rag with cross-encoder variant"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        mock_retrieve = Mock(return_value=(["doc1"], ["file1.pdf"]))
        mock_generate.return_value = "Answer"
        
        original_func = RETRIEVAL_FUNCTIONS["Cross-Encoder"]
        RETRIEVAL_FUNCTIONS["Cross-Encoder"] = mock_retrieve
        
        try:
            result = run_rag("Cross-Encoder", "test query", top_k=5)
            assert result["variant_name"] == "Cross-Encoder"
            mock_retrieve.assert_called_once()
        finally:
            RETRIEVAL_FUNCTIONS["Cross-Encoder"] = original_func
    
    def test_run_rag_invalid_variant(self):
        """Test run_rag raises error for invalid variant"""
        with pytest.raises(ValueError, match="Unknown variant"):
            run_rag("Invalid Variant", "test query")
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_run_rag_empty_results(self, mock_generate, mock_get_db):
        """Test run_rag handles empty retrieval results"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        mock_retrieve = Mock(return_value=([], []))
        mock_generate.return_value = "No documents found."
        
        original_func = RETRIEVAL_FUNCTIONS["Fixed Chunking"]
        RETRIEVAL_FUNCTIONS["Fixed Chunking"] = mock_retrieve
        
        try:
            result = run_rag("Fixed Chunking", "test query", top_k=5)
            assert result["documents"] == []
            assert result["sources"] == []
            mock_generate.assert_called_once_with("test query", [])
        finally:
            RETRIEVAL_FUNCTIONS["Fixed Chunking"] = original_func
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_run_rag_all_variants(self, mock_generate, mock_get_db):
        """Test that run_rag works with all configured variants"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        mock_generate.return_value = "Answer"
        
        original_funcs = {}
        for variant_name in VARIANT_CONFIG.keys():
            original_funcs[variant_name] = RETRIEVAL_FUNCTIONS[variant_name]
            mock_retrieve = Mock(return_value=(["doc1"], ["file1.pdf"]))
            RETRIEVAL_FUNCTIONS[variant_name] = mock_retrieve
        
        try:
            for variant_name in VARIANT_CONFIG.keys():
                result = run_rag(variant_name, "test query", top_k=5)
                assert result["variant_name"] == variant_name
                assert "answer" in result
                assert "documents" in result
                assert "sources" in result
        finally:
            for variant_name, func in original_funcs.items():
                RETRIEVAL_FUNCTIONS[variant_name] = func


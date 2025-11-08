"""
Integration tests - test the full pipeline
"""

import pytest
from unittest.mock import patch, Mock
from rag.pipeline import run_rag


class TestIntegration:
    """Integration tests for full RAG pipeline"""
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_full_pipeline_flow(self, mock_generate, mock_get_db):
        """Test complete flow from query to answer"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        # Setup mocks
        mock_collection = Mock()
        mock_get_db.return_value = mock_collection
        
        mock_retrieve = Mock(return_value=(
            ["Document about GST", "Document about MSME"],
            ["gst.pdf", "msme.pdf"]
        ))
        
        original_func = RETRIEVAL_FUNCTIONS["Fixed Chunking"]
        RETRIEVAL_FUNCTIONS["Fixed Chunking"] = mock_retrieve
        
        try:
            mock_generate.return_value = "GST is a tax system introduced in India."
            
            # Run pipeline
            result = run_rag("Fixed Chunking", "What is GST?", top_k=2)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "documents" in result
            assert "sources" in result
            assert "variant_name" in result
            
            # Verify values
            assert result["query"] == "What is GST?"
            assert len(result["documents"]) == 2
            assert len(result["sources"]) == 2
            assert result["variant_name"] == "Fixed Chunking"
            
            # Verify function calls
            mock_get_db.assert_called_once()
            mock_retrieve.assert_called_once()
            mock_generate.assert_called_once()
        finally:
            RETRIEVAL_FUNCTIONS["Fixed Chunking"] = original_func
    
    @patch('rag.pipeline.get_vector_db')
    @patch('rag.pipeline.generate_answer')
    def test_pipeline_with_different_variants(self, mock_generate, mock_get_db):
        """Test pipeline works with different variants"""
        from rag.pipeline import RETRIEVAL_FUNCTIONS
        
        variants = ["Fixed Chunking", "Hybrid Retrieval", "Cross-Encoder"]
        original_funcs = {}
        
        for variant in variants:
            original_funcs[variant] = RETRIEVAL_FUNCTIONS[variant]
        
        try:
            for variant in variants:
                mock_collection = Mock()
                mock_get_db.return_value = mock_collection
                mock_generate.return_value = "Answer"
                
                # Mock the retrieval function
                mock_retrieve = Mock(return_value=(["doc1"], ["file1.pdf"]))
                RETRIEVAL_FUNCTIONS[variant] = mock_retrieve
                
                result = run_rag(variant, "test query", top_k=1)
                
                assert result["variant_name"] == variant
                mock_get_db.reset_mock()
                mock_generate.reset_mock()
        finally:
            for variant, func in original_funcs.items():
                RETRIEVAL_FUNCTIONS[variant] = func


"""
Tests for answer generation
"""

import pytest
from unittest.mock import Mock, patch
from rag.generation import generate_answer


class TestGenerateAnswer:
    """Test generate_answer function"""
    
    @patch.dict('os.environ', {}, clear=True)
    def test_generate_answer_no_api_key(self):
        """Test generation returns error message when no API key"""
        result = generate_answer("test query", ["doc1", "doc2"])
        
        assert "OPENAI_API_KEY" in result
        assert "not set" in result
    
    @patch('rag.generation.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_answer_success(self, mock_openai_class):
        """Test successful answer generation"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="This is a test answer."))]
        mock_client.chat.completions.create.return_value = mock_response
        
        result = generate_answer("test query", ["doc1", "doc2"])
        
        assert result == "This is a test answer."
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4o-mini"
        assert call_args[1]['temperature'] == 0.3
        assert call_args[1]['max_tokens'] == 500
    
    @patch('rag.generation.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_answer_error_handling(self, mock_openai_class):
        """Test error handling in answer generation"""
        # Mock OpenAI to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = generate_answer("test query", ["doc1", "doc2"])
        
        assert "Error generating answer" in result
        assert "API Error" in result
    
    @patch('rag.generation.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_generate_answer_uses_top_3_documents(self, mock_openai_class):
        """Test that generation only uses top 3 documents"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        documents = [f"doc{i}" for i in range(10)]
        generate_answer("test query", documents)
        
        # Check that prompt contains only first 3 documents
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]['messages'][0]['content']
        assert "doc0" in prompt
        assert "doc1" in prompt
        assert "doc2" in prompt
        assert "doc3" not in prompt


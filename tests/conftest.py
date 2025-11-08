"""
Pytest configuration and fixtures
"""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before any imports
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()
sys.modules['sentence_transformers.CrossEncoder'] = MagicMock()

# Mock langchain
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.text_splitter'] = MagicMock()
sys.modules['langchain.text_splitter.RecursiveCharacterTextSplitter'] = MagicMock
sys.modules['langchain.text_splitter.NLTKTextSplitter'] = MagicMock

# Mock the embedding classes
mock_local_embedding = MagicMock()
mock_cross_encoder = MagicMock()

# Create a mock module for utils.embeddings
mock_embeddings_module = MagicMock()
mock_embeddings_module.LocalEmbeddingFunction = MagicMock
mock_embeddings_module.CrossEncoderReranker = MagicMock
sys.modules['utils.embeddings'] = mock_embeddings_module


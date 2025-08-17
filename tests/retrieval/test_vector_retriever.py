"""
Unit tests for VectorRetriever class
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.retrieval.vector_retriever import VectorRetriever
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


class TestVectorRetriever:
    """Test suite for VectorRetriever class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return PipelineConfig(
            retrieval_k=3,
            embedding_model="models/embedding-001"
        )
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store"""
        mock = Mock()
        mock.as_retriever.return_value = Mock()
        mock._collection.count.return_value = 5
        return mock
    
    @pytest.fixture
    def mock_langchain_docs(self):
        """Create mock LangChain documents"""
        docs = []
        for i in range(3):
            doc = Mock()
            doc.page_content = f"Test content {i}"
            doc.metadata = {"source": f"test{i}", "doc_id": f"doc_{i}"}
            docs.append(doc)
        return docs
    
    def test_init_without_vectorstore(self, config):
        """Test VectorRetriever initialization without vector store"""
        retriever = VectorRetriever(config=config)
        
        assert retriever.vectorstore is None
        assert retriever.config == config
        assert retriever._retriever is None
        assert not retriever.is_ready()
    
    def test_init_with_vectorstore(self, config, mock_vectorstore):
        """Test VectorRetriever initialization with vector store"""
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.config == config
        assert retriever._retriever is not None
        assert retriever.is_ready()
        
        # Verify as_retriever was called with correct parameters
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": config.retrieval_k}
        )
    
    def test_init_with_default_config(self, mock_vectorstore):
        """Test initialization with default configuration"""
        retriever = VectorRetriever(vectorstore=mock_vectorstore)
        
        assert retriever.config is not None
        assert isinstance(retriever.config, PipelineConfig)
        assert retriever.is_ready()
    
    def test_set_vectorstore(self, config):
        """Test setting vector store after initialization"""
        retriever = VectorRetriever(config=config)
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        assert not retriever.is_ready()
        
        retriever.set_vectorstore(mock_vectorstore)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.is_ready()
        mock_vectorstore.as_retriever.assert_called_once()
    
    def test_retrieve_success(self, config, mock_vectorstore, mock_langchain_docs):
        """Test successful document retrieval"""
        # Setup mock retriever
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = mock_langchain_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve("test query", k=3)
        
        assert len(result) == 3
        for i, doc in enumerate(result):
            assert isinstance(doc, Document)
            assert doc.content == f"Test content {i}"
            assert doc.metadata == {"source": f"test{i}", "doc_id": f"doc_{i}"}
            assert doc.doc_id == f"doc_{i}"
        
        mock_retriever.get_relevant_documents.assert_called_once_with("test query")
    
    def test_retrieve_with_different_k(self, config, mock_vectorstore, mock_langchain_docs):
        """Test retrieval with different k value than config"""
        # Setup mock retriever
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = mock_langchain_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve("test query", k=5)  # Different from config.retrieval_k=3
        
        assert len(result) == 3
        # Should create a new retriever with k=5
        assert mock_vectorstore.as_retriever.call_count == 2  # Once in init, once for different k
    
    def test_retrieve_no_retriever(self, config):
        """Test retrieval when no retriever is available"""
        retriever = VectorRetriever(config=config)
        
        result = retriever.retrieve("test query")
        
        assert result == []
    
    def test_retrieve_error_handling(self, config, mock_vectorstore):
        """Test error handling during retrieval"""
        # Setup mock retriever to raise exception
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval error")
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve("test query")
        
        assert result == []
    
    def test_retrieve_with_scores_success(self, config, mock_vectorstore):
        """Test successful document retrieval with scores"""
        # Setup mock data
        mock_results = [
            (Mock(page_content="Content 1", metadata={"source": "test1", "doc_id": "doc_1"}), 0.9),
            (Mock(page_content="Content 2", metadata={"source": "test2", "doc_id": "doc_2"}), 0.8),
        ]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        mock_vectorstore.as_retriever.return_value = Mock()
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve_with_scores("test query", k=2)
        
        assert len(result) == 2
        for i, (doc, score) in enumerate(result):
            assert isinstance(doc, Document)
            assert doc.content == f"Content {i+1}"
            assert doc.doc_id == f"doc_{i+1}"
            assert score in [0.9, 0.8]
        
        mock_vectorstore.similarity_search_with_score.assert_called_once_with("test query", k=2)
    
    def test_retrieve_with_scores_no_vectorstore(self, config):
        """Test retrieval with scores when no vector store is available"""
        retriever = VectorRetriever(config=config)
        
        result = retriever.retrieve_with_scores("test query")
        
        assert result == []
    
    def test_retrieve_with_scores_error_handling(self, config, mock_vectorstore):
        """Test error handling in retrieve_with_scores"""
        mock_vectorstore.similarity_search_with_score.side_effect = Exception("Score retrieval error")
        mock_vectorstore.as_retriever.return_value = Mock()
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve_with_scores("test query")
        
        assert result == []
    
    def test_similarity_search_alias(self, config, mock_vectorstore, mock_langchain_docs):
        """Test similarity_search method as alias for retrieve"""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = mock_langchain_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.similarity_search("test query", k=2)
        
        assert len(result) == 3  # mock_langchain_docs has 3 items
        assert all(isinstance(doc, Document) for doc in result)
    
    def test_get_relevant_documents(self, config, mock_vectorstore, mock_langchain_docs):
        """Test get_relevant_documents using config k value"""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = mock_langchain_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.get_relevant_documents("test query")
        
        assert len(result) == 3
        mock_retriever.get_relevant_documents.assert_called_once_with("test query")
    
    def test_is_ready(self, config, mock_vectorstore):
        """Test is_ready method"""
        # Without vector store
        retriever = VectorRetriever(config=config)
        assert not retriever.is_ready()
        
        # With vector store
        mock_vectorstore.as_retriever.return_value = Mock()
        retriever.set_vectorstore(mock_vectorstore)
        assert retriever.is_ready()
    
    def test_get_document_count_success(self, config, mock_vectorstore):
        """Test getting document count"""
        mock_vectorstore.as_retriever.return_value = Mock()
        mock_vectorstore._collection.count.return_value = 10
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.get_document_count()
        
        assert result == 10
        mock_vectorstore._collection.count.assert_called_once()
    
    def test_get_document_count_no_vectorstore(self, config):
        """Test getting document count when no vector store"""
        retriever = VectorRetriever(config=config)
        
        result = retriever.get_document_count()
        
        assert result == 0
    
    def test_get_document_count_error_handling(self, config, mock_vectorstore):
        """Test error handling in get_document_count"""
        mock_vectorstore.as_retriever.return_value = Mock()
        mock_vectorstore._collection.count.side_effect = Exception("Count error")
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.get_document_count()
        
        assert result == 0
    
    def test_update_config(self, config, mock_vectorstore):
        """Test updating retriever configuration"""
        mock_vectorstore.as_retriever.return_value = Mock()
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        # Initial setup call
        initial_calls = mock_vectorstore.as_retriever.call_count
        
        # Update config
        new_config = PipelineConfig(retrieval_k=10)
        retriever.update_config(new_config)
        
        assert retriever.config == new_config
        # Should call as_retriever again to update with new config
        assert mock_vectorstore.as_retriever.call_count == initial_calls + 1
    
    def test_update_config_no_vectorstore(self, config):
        """Test updating config when no vector store is set"""
        retriever = VectorRetriever(config=config)
        
        new_config = PipelineConfig(retrieval_k=10)
        retriever.update_config(new_config)
        
        assert retriever.config == new_config
        # Should not raise any errors
    
    def test_retrieve_with_missing_doc_id(self, config, mock_vectorstore):
        """Test retrieval when documents don't have doc_id in metadata"""
        # Create mock documents without doc_id
        mock_docs = [
            Mock(page_content="Content 1", metadata={"source": "test1"}),
            Mock(page_content="Content 2", metadata={"source": "test2"}),
        ]
        
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = mock_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        retriever = VectorRetriever(vectorstore=mock_vectorstore, config=config)
        
        result = retriever.retrieve("test query")
        
        assert len(result) == 2
        # Should generate default doc_ids
        assert result[0].doc_id == "retrieved_doc_0"
        assert result[1].doc_id == "retrieved_doc_1"
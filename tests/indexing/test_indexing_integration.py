"""
Integration tests for all indexing strategies
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.indexing.indexing_manager import IndexingManager
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def config():
    """Create a test configuration"""
    return PipelineConfig(
        llm_model="gemini-pro",
        embedding_model="models/embedding-001",
        temperature=0.0,
        indexing_strategy="basic",
        chunk_size=1000,
        chunk_overlap=200,
        vector_store_config={"persist_directory": None}
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            content="This is a document about artificial intelligence and machine learning.",
            metadata={"title": "AI Overview", "author": "Test Author"},
            doc_id="doc1"
        ),
        Document(
            content="This document discusses the history of computer science and programming languages.",
            metadata={"title": "CS History", "author": "Another Author"},
            doc_id="doc2"
        )
    ]


class TestIndexingIntegration:
    """Integration tests for indexing strategies"""
    
    def test_indexing_manager_initialization(self, config):
        """Test IndexingManager initialization"""
        manager = IndexingManager(config)
        
        assert manager.get_active_strategy() == "basic"
        assert len(manager.list_strategies()) == 4
        assert "basic" in manager.list_strategies()
        assert "multi_representation" in manager.list_strategies()
        assert "colbert" in manager.list_strategies()
        assert "raptor" in manager.list_strategies()
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_basic_indexing_workflow(self, mock_basic_indexer, config, sample_documents):
        """Test basic indexing workflow"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = True
        mock_indexer_instance.get_document_count.return_value = 2
        mock_indexer_instance.clear_index.return_value = True
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        
        # Test indexing
        result = manager.index_documents(sample_documents)
        assert result is True
        
        # Test document count
        count = manager.get_document_count()
        assert count == 2
        
        # Test clearing
        result = manager.clear_index()
        assert result is True
    
    def test_strategy_switching(self, config):
        """Test switching between strategies"""
        manager = IndexingManager(config)
        
        # Initial strategy
        assert manager.get_active_strategy() == "basic"
        
        # Switch to multi-representation
        result = manager.set_active_strategy("multi_representation")
        assert result is True
        assert manager.get_active_strategy() == "multi_representation"
        
        # Try invalid strategy
        result = manager.set_active_strategy("invalid")
        assert result is False
        assert manager.get_active_strategy() == "multi_representation"  # Should remain unchanged
    
    def test_strategy_info(self, config):
        """Test getting strategy information"""
        manager = IndexingManager(config)
        
        # Get info for active strategy
        info = manager.get_strategy_info()
        assert info["strategy"] == "basic"
        assert info["is_active"] is True
        assert info["is_initialized"] is False
        assert info["document_count"] == 0
        
        # Get info for all strategies
        all_info = manager.get_all_strategies_info()
        assert len(all_info) == 4
        assert all_info["basic"]["is_active"] is True
        assert all_info["multi_representation"]["is_active"] is False
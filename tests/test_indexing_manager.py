"""
Unit tests for IndexingManager
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.indexing.indexing_manager import IndexingManager, IndexingStrategy
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


class TestIndexingStrategy:
    """Test cases for IndexingStrategy enum"""
    
    def test_indexing_strategy_values(self):
        """Test IndexingStrategy enum values"""
        assert IndexingStrategy.BASIC.value == "basic"
        assert IndexingStrategy.MULTI_REPRESENTATION.value == "multi_representation"
        assert IndexingStrategy.COLBERT.value == "colbert"
        assert IndexingStrategy.RAPTOR.value == "raptor"


class TestIndexingManager:
    """Test cases for IndexingManager"""
    
    def test_initialization(self, config):
        """Test IndexingManager initialization"""
        manager = IndexingManager(config)
        
        assert manager.config == config
        assert manager._active_strategy == "basic"
        assert len(manager._indexers) == 0
        assert len(manager._indexer_registry) == 4
        
        # Check that all strategies are registered
        expected_strategies = ["basic", "multi_representation", "colbert", "raptor"]
        assert set(manager._indexer_registry.keys()) == set(expected_strategies)
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_get_indexer_basic(self, mock_basic_indexer, config):
        """Test getting basic indexer"""
        mock_indexer_instance = Mock()
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        indexer = manager.get_indexer("basic")
        
        assert indexer == mock_indexer_instance
        assert "basic" in manager._indexers
        mock_basic_indexer.assert_called_once_with(config)
    
    @patch('src.rag_engine.indexing.indexing_manager.MultiRepresentationIndexer')
    def test_get_indexer_multi_representation(self, mock_multi_indexer, config):
        """Test getting multi-representation indexer"""
        mock_indexer_instance = Mock()
        mock_multi_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        indexer = manager.get_indexer("multi_representation")
        
        assert indexer == mock_indexer_instance
        assert "multi_representation" in manager._indexers
        mock_multi_indexer.assert_called_once_with(config)
    
    @patch('src.rag_engine.indexing.indexing_manager.ColBERTIndexer')
    def test_get_indexer_colbert(self, mock_colbert_indexer, config):
        """Test getting ColBERT indexer"""
        mock_indexer_instance = Mock()
        mock_colbert_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        indexer = manager.get_indexer("colbert")
        
        assert indexer == mock_indexer_instance
        assert "colbert" in manager._indexers
        mock_colbert_indexer.assert_called_once_with(config, index_name="colbert-colbert")
    
    @patch('src.rag_engine.indexing.indexing_manager.RAPTORIndexer')
    def test_get_indexer_raptor(self, mock_raptor_indexer, config):
        """Test getting RAPTOR indexer"""
        mock_indexer_instance = Mock()
        mock_raptor_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        indexer = manager.get_indexer("raptor")
        
        assert indexer == mock_indexer_instance
        assert "raptor" in manager._indexers
        mock_raptor_indexer.assert_called_once_with(config, max_levels=3, cluster_size_threshold=5)
    
    def test_get_indexer_invalid_strategy(self, config):
        """Test getting indexer with invalid strategy"""
        manager = IndexingManager(config)
        
        with pytest.raises(ValueError, match="Unsupported indexing strategy"):
            manager.get_indexer("invalid_strategy")
    
    def test_get_indexer_cached(self, config):
        """Test that indexers are cached after first creation"""
        with patch('src.rag_engine.indexing.indexing_manager.BasicIndexer') as mock_basic_indexer:
            mock_indexer_instance = Mock()
            mock_basic_indexer.return_value = mock_indexer_instance
            
            manager = IndexingManager(config)
            
            # First call should create indexer
            indexer1 = manager.get_indexer("basic")
            assert indexer1 == mock_indexer_instance
            mock_basic_indexer.assert_called_once()
            
            # Second call should return cached indexer
            indexer2 = manager.get_indexer("basic")
            assert indexer2 == mock_indexer_instance
            assert indexer1 is indexer2
            # Should not call constructor again
            mock_basic_indexer.assert_called_once()
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_index_documents_success(self, mock_basic_indexer, config, sample_documents):
        """Test successful document indexing"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = True
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        result = manager.index_documents(sample_documents)
        
        assert result is True
        mock_indexer_instance.index_documents.assert_called_once_with(sample_documents)
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_index_documents_with_strategy(self, mock_basic_indexer, config, sample_documents):
        """Test document indexing with specific strategy"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = True
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        result = manager.index_documents(sample_documents, strategy="basic")
        
        assert result is True
        mock_indexer_instance.index_documents.assert_called_once_with(sample_documents)
    
    def test_index_documents_empty_list(self, config):
        """Test indexing with empty document list"""
        manager = IndexingManager(config)
        result = manager.index_documents([])
        
        assert result is True
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_index_documents_failure(self, mock_basic_indexer, config, sample_documents):
        """Test document indexing failure"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = False
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        result = manager.index_documents(sample_documents)
        
        assert result is False
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_index_documents_exception(self, mock_basic_indexer, config, sample_documents):
        """Test document indexing with exception"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.side_effect = Exception("Indexing failed")
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        result = manager.index_documents(sample_documents)
        
        assert result is False
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_get_document_count(self, mock_basic_indexer, config):
        """Test getting document count"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.get_document_count.return_value = 5
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("basic")
        
        count = manager.get_document_count()
        
        assert count == 5
        mock_indexer_instance.get_document_count.assert_called_once()
    
    def test_get_document_count_no_indexer(self, config):
        """Test getting document count when no indexer exists"""
        manager = IndexingManager(config)
        count = manager.get_document_count()
        
        assert count == 0
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_clear_index_success(self, mock_basic_indexer, config):
        """Test successful index clearing"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.clear_index.return_value = True
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("basic")
        
        result = manager.clear_index()
        
        assert result is True
        mock_indexer_instance.clear_index.assert_called_once()
    
    def test_clear_index_no_indexer(self, config):
        """Test clearing index when no indexer exists"""
        manager = IndexingManager(config)
        result = manager.clear_index()
        
        assert result is True
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    @patch('src.rag_engine.indexing.indexing_manager.MultiRepresentationIndexer')
    def test_clear_all_indexes(self, mock_multi_indexer, mock_basic_indexer, config):
        """Test clearing all indexes"""
        mock_basic_instance = Mock()
        mock_basic_instance.clear_index.return_value = True
        mock_basic_indexer.return_value = mock_basic_instance
        
        mock_multi_instance = Mock()
        mock_multi_instance.clear_index.return_value = True
        mock_multi_indexer.return_value = mock_multi_instance
        
        manager = IndexingManager(config)
        # Initialize indexers
        manager.get_indexer("basic")
        manager.get_indexer("multi_representation")
        
        result = manager.clear_all_indexes()
        
        assert result is True
        mock_basic_instance.clear_index.assert_called_once()
        mock_multi_instance.clear_index.assert_called_once()
    
    def test_set_active_strategy_success(self, config):
        """Test setting active strategy successfully"""
        manager = IndexingManager(config)
        
        result = manager.set_active_strategy("multi_representation")
        
        assert result is True
        assert manager.get_active_strategy() == "multi_representation"
    
    def test_set_active_strategy_invalid(self, config):
        """Test setting invalid active strategy"""
        manager = IndexingManager(config)
        
        result = manager.set_active_strategy("invalid_strategy")
        
        assert result is False
        assert manager.get_active_strategy() == "basic"  # Should remain unchanged
    
    def test_get_active_strategy(self, config):
        """Test getting active strategy"""
        manager = IndexingManager(config)
        
        assert manager.get_active_strategy() == "basic"
    
    def test_list_strategies(self, config):
        """Test listing available strategies"""
        manager = IndexingManager(config)
        strategies = manager.list_strategies()
        
        expected_strategies = ["basic", "multi_representation", "colbert", "raptor"]
        assert set(strategies) == set(expected_strategies)
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_get_strategy_info_initialized(self, mock_basic_indexer, config):
        """Test getting strategy info for initialized indexer"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.get_document_count.return_value = 10
        mock_indexer_instance.get_chunk_count.return_value = 50
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("basic")
        
        info = manager.get_strategy_info("basic")
        
        assert info["strategy"] == "basic"
        assert info["is_active"] is True
        assert info["is_initialized"] is True
        assert info["document_count"] == 10
        assert info["chunk_count"] == 50
    
    def test_get_strategy_info_not_initialized(self, config):
        """Test getting strategy info for non-initialized indexer"""
        manager = IndexingManager(config)
        
        info = manager.get_strategy_info("multi_representation")
        
        assert info["strategy"] == "multi_representation"
        assert info["is_active"] is False
        assert info["is_initialized"] is False
        assert info["document_count"] == 0
    
    def test_get_all_strategies_info(self, config):
        """Test getting info for all strategies"""
        manager = IndexingManager(config)
        
        all_info = manager.get_all_strategies_info()
        
        assert len(all_info) == 4
        assert "basic" in all_info
        assert "multi_representation" in all_info
        assert "colbert" in all_info
        assert "raptor" in all_info
        
        # Check that basic is marked as active
        assert all_info["basic"]["is_active"] is True
        assert all_info["multi_representation"]["is_active"] is False
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_is_strategy_ready_with_is_ready_method(self, mock_basic_indexer, config):
        """Test strategy readiness when indexer has is_ready method"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.is_ready.return_value = True
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        
        result = manager.is_strategy_ready("basic")
        
        assert result is True
        mock_indexer_instance.is_ready.assert_called_once()
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_is_strategy_ready_without_is_ready_method(self, mock_basic_indexer, config):
        """Test strategy readiness when indexer doesn't have is_ready method"""
        mock_indexer_instance = Mock()
        # Remove is_ready method
        del mock_indexer_instance.is_ready
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        
        result = manager.is_strategy_ready("basic")
        
        assert result is True
    
    def test_is_strategy_ready_invalid_strategy(self, config):
        """Test strategy readiness for invalid strategy"""
        manager = IndexingManager(config)
        
        result = manager.is_strategy_ready("invalid_strategy")
        
        assert result is False
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_get_indexer_for_retrieval_basic(self, mock_basic_indexer, config):
        """Test getting retriever for basic indexer"""
        mock_indexer_instance = Mock()
        mock_vectorstore = Mock()
        mock_indexer_instance.get_vectorstore.return_value = mock_vectorstore
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("basic")
        
        retriever = manager.get_indexer_for_retrieval("basic")
        
        assert retriever == mock_vectorstore
        mock_indexer_instance.get_vectorstore.assert_called_once()
    
    @patch('src.rag_engine.indexing.indexing_manager.MultiRepresentationIndexer')
    def test_get_indexer_for_retrieval_multi_representation(self, mock_multi_indexer, config):
        """Test getting retriever for multi-representation indexer"""
        mock_indexer_instance = Mock()
        mock_retriever = Mock()
        mock_indexer_instance.get_retriever.return_value = mock_retriever
        mock_multi_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("multi_representation")
        
        retriever = manager.get_indexer_for_retrieval("multi_representation")
        
        assert retriever == mock_retriever
        mock_indexer_instance.get_retriever.assert_called_once()
    
    @patch('src.rag_engine.indexing.indexing_manager.ColBERTIndexer')
    def test_get_indexer_for_retrieval_colbert(self, mock_colbert_indexer, config):
        """Test getting retriever for ColBERT indexer"""
        mock_indexer_instance = Mock()
        mock_retriever = Mock()
        mock_indexer_instance.as_langchain_retriever.return_value = mock_retriever
        mock_colbert_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("colbert")
        
        retriever = manager.get_indexer_for_retrieval("colbert")
        
        assert retriever == mock_retriever
        mock_indexer_instance.as_langchain_retriever.assert_called_once()
    
    @patch('src.rag_engine.indexing.indexing_manager.RAPTORIndexer')
    def test_get_indexer_for_retrieval_raptor(self, mock_raptor_indexer, config):
        """Test getting retriever for RAPTOR indexer"""
        mock_indexer_instance = Mock()
        mock_raptor_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("raptor")
        
        retriever = manager.get_indexer_for_retrieval("raptor")
        
        # RAPTOR returns the indexer itself
        assert retriever == mock_indexer_instance
    
    def test_get_indexer_for_retrieval_no_indexer(self, config):
        """Test getting retriever when no indexer exists"""
        manager = IndexingManager(config)
        
        retriever = manager.get_indexer_for_retrieval("basic")
        
        assert retriever is None
    
    @patch('src.rag_engine.indexing.indexing_manager.BasicIndexer')
    def test_get_indexer_for_retrieval_exception(self, mock_basic_indexer, config):
        """Test getting retriever with exception"""
        mock_indexer_instance = Mock()
        mock_indexer_instance.get_vectorstore.side_effect = Exception("Retriever error")
        mock_basic_indexer.return_value = mock_indexer_instance
        
        manager = IndexingManager(config)
        # Initialize indexer
        manager.get_indexer("basic")
        
        retriever = manager.get_indexer_for_retrieval("basic")
        
        assert retriever is None
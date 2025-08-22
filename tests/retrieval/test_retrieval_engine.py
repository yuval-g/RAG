"""
Unit tests for RetrievalEngine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.retrieval.retrieval_engine import RetrievalEngine
from src.rag_engine.retrieval.vector_retriever import VectorRetriever
from src.rag_engine.retrieval.reranker import ReRanker
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.interfaces import BaseRetriever


class MockRetriever(BaseRetriever):
    """Mock retriever for testing"""
    
    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.documents[:k]
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        return [(doc, 0.8) for doc in self.documents[:k]]


class TestRetrievalEngine:
    """Test cases for RetrievalEngine"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            retrieval_k=5,
            use_reranking=False,
            reranker_top_k=3
        )
    
    @pytest.fixture
    def config_with_reranking(self):
        """Create test configuration with re-ranking enabled"""
        return PipelineConfig(
            retrieval_k=5,
            use_reranking=True,
            reranker_top_k=3
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Python programming tutorial",
                metadata={"source": "tutorial1"},
                doc_id="doc1"
            ),
            Document(
                content="JavaScript web development",
                metadata={"source": "tutorial2"},
                doc_id="doc2"
            ),
            Document(
                content="Machine learning basics",
                metadata={"source": "tutorial3"},
                doc_id="doc3"
            )
        ]
    
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_initialization_without_reranking(self, mock_vector_retriever, config):
        """Test RetrievalEngine initialization without re-ranking"""
        engine = RetrievalEngine(config)
        
        assert engine.config == config
        assert "vector" in engine.retrievers
        assert engine.default_retriever_name == "vector"
        assert engine.reranker is None
        mock_vector_retriever.assert_called_once_with(config=config)
    
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_initialization_with_reranking(self, mock_reranker, mock_vector_retriever, config_with_reranking):
        """Test RetrievalEngine initialization with re-ranking enabled"""
        # Mock vector retriever with _retriever attribute
        mock_retriever_instance = Mock()
        mock_retriever_instance._retriever = Mock()
        mock_vector_retriever.return_value = mock_retriever_instance
        
        engine = RetrievalEngine(config_with_reranking)
        
        assert engine.config == config_with_reranking
        assert engine.reranker is not None
        mock_reranker.assert_called_once()
    
    def test_add_retriever(self, config, sample_documents):
        """Test adding a retriever to the engine"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        
        engine.add_retriever("test_retriever", mock_retriever)
        
        assert "test_retriever" in engine.retrievers
        assert engine.retrievers["test_retriever"] == mock_retriever
    
    def test_set_default_retriever(self, config, sample_documents):
        """Test setting default retriever"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        
        engine.set_default_retriever("test_retriever")
        
        assert engine.default_retriever_name == "test_retriever"
    
    def test_set_default_retriever_not_found(self, config):
        """Test setting default retriever that doesn't exist"""
        engine = RetrievalEngine(config)
        
        with pytest.raises(ValueError, match="Retriever 'nonexistent' not found"):
            engine.set_default_retriever("nonexistent")
    
    def test_get_retriever_default(self, config, sample_documents):
        """Test getting default retriever"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        retriever = engine.get_retriever()
        
        assert retriever == mock_retriever
    
    def test_get_retriever_by_name(self, config, sample_documents):
        """Test getting retriever by name"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        
        retriever = engine.get_retriever("test_retriever")
        
        assert retriever == mock_retriever
    
    def test_get_retriever_not_found(self, config):
        """Test getting retriever that doesn't exist"""
        engine = RetrievalEngine(config)
        
        with pytest.raises(ValueError, match="Retriever 'nonexistent' not found"):
            engine.get_retriever("nonexistent")
    
    def test_retrieve(self, config, sample_documents):
        """Test basic document retrieval"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve("test query", k=2)
        
        assert len(results) == 2
        assert results == sample_documents[:2]
    
    def test_retrieve_with_retriever_name(self, config, sample_documents):
        """Test retrieval with specific retriever name"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        
        results = engine.retrieve("test query", k=2, retriever_name="test_retriever")
        
        assert len(results) == 2
        assert results == sample_documents[:2]
    
    def test_retrieve_with_scores(self, config, sample_documents):
        """Test retrieval with scores"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve_with_scores("test query", k=2)
        
        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(score == 0.8 for doc, score in results)
    
    def test_retrieve_with_scores_fallback(self, config, sample_documents):
        """Test retrieval with scores fallback for retrievers without score support"""
        engine = RetrievalEngine(config)
        
        # Create mock retriever without retrieve_with_scores method
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = sample_documents[:2]
        # Explicitly remove retrieve_with_scores method to test fallback
        if hasattr(mock_retriever, 'retrieve_with_scores'):
            delattr(mock_retriever, 'retrieve_with_scores')
        
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve_with_scores("test query", k=2)
        
        assert len(results) == 2
        assert all(score == 0.5 for doc, score in results)  # Default score
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_retrieve_with_rerank(self, mock_reranker_class, config, sample_documents):
        """Test retrieval with re-ranking"""
        # Setup mock re-ranker
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = sample_documents[:2]
        mock_reranker_class.return_value = mock_reranker
        
        engine = RetrievalEngine(config)
        engine.reranker = mock_reranker
        
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve_with_rerank("test query", k=5, top_k=2)
        
        assert len(results) == 2
        mock_reranker.rerank.assert_called_once_with("test query", sample_documents, 2)
    
    def test_retrieve_with_rerank_no_reranker(self, config, sample_documents):
        """Test retrieval with re-ranking when no re-ranker is available"""
        engine = RetrievalEngine(config)
        engine.reranker = None
        
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve_with_rerank("test query", k=5, top_k=2)
        
        assert len(results) == 2
        assert results == sample_documents[:2]  # Should return top_k from initial retrieval
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_retrieve_with_rerank_and_scores(self, mock_reranker_class, config, sample_documents):
        """Test retrieval with re-ranking and scores"""
        # Setup mock re-ranker
        mock_reranker = Mock()
        expected_results = [(sample_documents[0], 0.9), (sample_documents[1], 0.8)]
        mock_reranker.rerank_with_scores.return_value = expected_results
        mock_reranker_class.return_value = mock_reranker
        
        engine = RetrievalEngine(config)
        engine.reranker = mock_reranker
        
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        results = engine.retrieve_with_rerank_and_scores("test query", k=5, top_k=2)
        
        assert len(results) == 2
        assert results == expected_results
        mock_reranker.rerank_with_scores.assert_called_once_with("test query", sample_documents, 2)
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_enable_reranking_llm_strategy(self, mock_reranker_class, config):
        """Test enabling re-ranking with LLM strategy"""
        engine = RetrievalEngine(config)
        engine.enable_reranking("llm")
        
        assert engine.reranker is not None
        assert engine.config.use_reranking is True
        mock_reranker_class.assert_called_with(
            strategy="llm",
            base_retriever=None,
            config=config
        )
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_enable_reranking_contextual_strategy(self, mock_reranker_class, config, sample_documents):
        """Test enabling re-ranking with contextual strategy"""
        engine = RetrievalEngine(config)
        
        # Setup mock retriever with _retriever attribute
        mock_retriever = Mock()
        mock_retriever._retriever = Mock()
        engine.add_retriever("test_retriever", mock_retriever)
        engine.set_default_retriever("test_retriever")
        
        engine.enable_reranking("contextual")
        
        assert engine.reranker is not None
        assert engine.config.use_reranking is True
        mock_reranker_class.assert_called_with(
            strategy="contextual",
            base_retriever=mock_retriever._retriever,
            config=config
        )
    
    def test_disable_reranking(self, config):
        """Test disabling re-ranking"""
        engine = RetrievalEngine(config)
        engine.reranker = Mock()  # Set a mock re-ranker
        engine.config.use_reranking = True
        
        engine.disable_reranking()
        
        assert engine.reranker is None
        assert engine.config.use_reranking is False
    
    def test_is_reranking_enabled(self, config):
        """Test checking if re-ranking is enabled"""
        engine = RetrievalEngine(config)
        
        assert engine.is_reranking_enabled() is False
        
        engine.reranker = Mock()
        assert engine.is_reranking_enabled() is True
    
    def test_get_available_retrievers(self, config, sample_documents):
        """Test getting list of available retrievers"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        engine.add_retriever("test_retriever", mock_retriever)
        
        retrievers = engine.get_available_retrievers()
        
        assert "vector" in retrievers  # Default retriever
        assert "keyword" in retrievers  # Default keyword retriever
        assert "test_retriever" in retrievers
        assert len(retrievers) == 3
    
    def test_get_retrieval_stats(self, config, sample_documents):
        """Test getting retrieval engine statistics"""
        engine = RetrievalEngine(config)
        mock_retriever = MockRetriever(sample_documents)
        mock_retriever.get_document_count = Mock(return_value=10)
        mock_retriever.is_ready = Mock(return_value=True)
        
        engine.add_retriever("test_retriever", mock_retriever)
        
        stats = engine.get_retrieval_stats()
        
        assert "available_retrievers" in stats
        assert "default_retriever" in stats
        assert "reranking_enabled" in stats
        assert "config" in stats
        assert stats["default_retriever"] == "vector"
        assert stats["reranking_enabled"] is False
    
    def test_get_retrieval_stats_with_reranker(self, config):
        """Test getting retrieval statistics with re-ranker"""
        engine = RetrievalEngine(config)
        mock_reranker = Mock()
        mock_reranker.get_strategy.return_value = "llm"
        engine.reranker = mock_reranker
        
        stats = engine.get_retrieval_stats()
        
        assert stats["reranking_enabled"] is True
        assert stats["reranking_strategy"] == "llm"
    
    def test_update_config(self, config, sample_documents):
        """Test updating configuration"""
        engine = RetrievalEngine(config)
        
        # Add mock retriever with update_config method
        mock_retriever = Mock()
        mock_retriever.update_config = Mock()
        engine.add_retriever("test_retriever", mock_retriever)
        
        # Add mock re-ranker
        mock_reranker = Mock()
        mock_reranker.set_config = Mock()
        engine.reranker = mock_reranker
        
        new_config = PipelineConfig(retrieval_k=10)
        engine.update_config(new_config)
        
        assert engine.config == new_config
        mock_retriever.update_config.assert_called_once_with(new_config)
        mock_reranker.set_config.assert_called_once_with(new_config)
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_update_config_enable_reranking(self, mock_reranker_class, config):
        """Test updating configuration to enable re-ranking"""
        engine = RetrievalEngine(config)
        assert engine.reranker is None
        
        new_config = PipelineConfig(use_reranking=True)
        engine.update_config(new_config)
        
        assert engine.reranker is not None
        mock_reranker_class.assert_called()
    
    def test_update_config_disable_reranking(self, config_with_reranking):
        """Test updating configuration to disable re-ranking"""
        engine = RetrievalEngine(config_with_reranking)
        engine.reranker = Mock()  # Set a mock re-ranker
        
        new_config = PipelineConfig(use_reranking=False)
        engine.update_config(new_config)
        
        assert engine.reranker is None
        assert engine.config.use_reranking is False


class TestRetrievalEngineIntegration:
    """Integration tests for RetrievalEngine"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            retrieval_k=5,
            use_reranking=False
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Python is a versatile programming language used for web development, data science, and automation.",
                metadata={"source": "python_guide", "topic": "programming"},
                doc_id="python_doc"
            ),
            Document(
                content="JavaScript enables interactive web pages and is essential for modern web development.",
                metadata={"source": "js_guide", "topic": "programming"},
                doc_id="js_doc"
            ),
            Document(
                content="Machine learning algorithms can automatically improve through experience and data.",
                metadata={"source": "ml_intro", "topic": "ai"},
                doc_id="ml_doc"
            ),
            Document(
                content="The weather today is cloudy with a chance of rain in the afternoon.",
                metadata={"source": "weather_report", "topic": "weather"},
                doc_id="weather_doc"
            )
        ]
    
    def test_multiple_retrievers_workflow(self, config, sample_documents):
        """Test workflow with multiple retrievers"""
        engine = RetrievalEngine(config)
        
        # Add different retrievers
        retriever1 = MockRetriever(sample_documents[:2])  # Programming docs
        retriever2 = MockRetriever(sample_documents[2:])  # AI and weather docs
        
        engine.add_retriever("programming", retriever1)
        engine.add_retriever("general", retriever2)
        
        # Test retrieval from different retrievers
        prog_results = engine.retrieve("Python tutorial", k=2, retriever_name="programming")
        general_results = engine.retrieve("machine learning", k=2, retriever_name="general")
        
        assert len(prog_results) == 2
        assert len(general_results) == 2
        assert prog_results[0].doc_id == "python_doc"
        assert general_results[0].doc_id == "ml_doc"
    
    @patch('src.rag_engine.retrieval.retrieval_engine.ReRanker')
    def test_end_to_end_retrieval_with_reranking(self, mock_reranker_class, config, sample_documents):
        """Test end-to-end retrieval workflow with re-ranking"""
        # Setup mock re-ranker
        mock_reranker = Mock()
        reranked_docs = [sample_documents[0], sample_documents[2]]  # Python and ML docs
        mock_reranker.rerank.return_value = reranked_docs
        mock_reranker_class.return_value = mock_reranker
        
        engine = RetrievalEngine(config)
        engine.enable_reranking("llm")
        
        # Add retriever with all documents
        retriever = MockRetriever(sample_documents)
        engine.add_retriever("all_docs", retriever)
        engine.set_default_retriever("all_docs")
        
        # Test retrieval with re-ranking
        results = engine.retrieve_with_rerank("programming tutorial", k=4, top_k=2)
        
        assert len(results) == 2
        assert results[0].doc_id == "python_doc"
        assert results[1].doc_id == "ml_doc"
        mock_reranker.rerank.assert_called_once_with("programming tutorial", sample_documents, 2)
    
    def test_error_handling_retriever_failure(self, config):
        """Test error handling when retriever fails"""
        engine = RetrievalEngine(config)
        
        # Add mock retriever that raises exception
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Retrieval failed")
        engine.add_retriever("failing_retriever", mock_retriever)
        engine.set_default_retriever("failing_retriever")
        
        results = engine.retrieve("test query")
        
        assert results == []  # Should return empty list on failure
    
    def test_configuration_changes_propagation(self, config, sample_documents):
        """Test that configuration changes propagate to all components"""
        engine = RetrievalEngine(config)
        
        # Add mock retriever with update_config method
        mock_retriever = Mock()
        mock_retriever.update_config = Mock()
        engine.add_retriever("test_retriever", mock_retriever)
        
        # Update configuration
        new_config = PipelineConfig(
            retrieval_k=10,
            use_reranking=True,
            reranker_top_k=5
        )
        
        with patch('src.rag_engine.retrieval.retrieval_engine.ReRanker') as mock_reranker_class:
            engine.update_config(new_config)
        
        # Verify configuration was updated
        assert engine.config.retrieval_k == 10
        assert engine.config.use_reranking is True
        assert engine.config.reranker_top_k == 5
        
        # Verify retriever was updated
        mock_retriever.update_config.assert_called_once_with(new_config)
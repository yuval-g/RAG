"""
Integration tests for retrieval engine with self-correction
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.retrieval.retrieval_engine import RetrievalEngine
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.models import Document


class TestRetrievalSelfCorrectionIntegration:
    """Test integration between retrieval engine and self-correction"""
    
    @pytest.fixture
    def config_with_self_correction(self):
        """Create config with self-correction enabled"""
        return PipelineConfig(
            enable_self_correction=True,
            relevance_threshold=0.7,
            factuality_threshold=0.7,
            min_relevant_docs=2
        )
    
    @pytest.fixture
    def config_without_self_correction(self):
        """Create config without self-correction"""
        return PipelineConfig(
            enable_self_correction=False
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents"""
        return [
            Document(content="Machine learning content", metadata={"topic": "ml"}),
            Document(content="Weather content", metadata={"topic": "weather"}),
            Document(content="Deep learning content", metadata={"topic": "dl"})
        ]
    
    @patch('src.rag_engine.retrieval.retrieval_engine.SelfCorrectionEngine')
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_retrieval_engine_with_self_correction_enabled(self, mock_vector_retriever, mock_self_correction, config_with_self_correction):
        """Test retrieval engine initialization with self-correction enabled"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Mock the self-correction engine
        mock_correction_instance = Mock()
        mock_self_correction.return_value = mock_correction_instance
        
        # Initialize retrieval engine
        engine = RetrievalEngine(config_with_self_correction)
        
        # Verify self-correction engine was initialized
        mock_self_correction.assert_called_once_with(config_with_self_correction)
        assert engine.self_correction_engine == mock_correction_instance
        assert engine.is_self_correction_enabled() is True
    
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_retrieval_engine_without_self_correction(self, mock_vector_retriever, config_without_self_correction):
        """Test retrieval engine initialization without self-correction"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Initialize retrieval engine
        engine = RetrievalEngine(config_without_self_correction)
        
        # Verify self-correction engine was not initialized
        assert engine.self_correction_engine is None
        assert engine.is_self_correction_enabled() is False
    
    @patch('src.rag_engine.retrieval.retrieval_engine.SelfCorrectionEngine')
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_retrieve_with_correction(self, mock_vector_retriever, mock_self_correction, config_with_self_correction, sample_documents):
        """Test retrieve_with_correction method"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = sample_documents
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Mock the self-correction engine
        mock_correction_instance = Mock()
        corrected_docs = sample_documents[:2]  # Return first 2 documents
        correction_metadata = {
            "original_count": 3,
            "filtered_count": 2,
            "correction_applied": True,
            "fallback_triggered": False
        }
        mock_correction_instance.correct_retrieval.return_value = (corrected_docs, correction_metadata)
        mock_self_correction.return_value = mock_correction_instance
        
        # Initialize retrieval engine
        engine = RetrievalEngine(config_with_self_correction)
        
        # Test retrieve with correction
        query = "test query"
        result_docs, metadata = engine.retrieve_with_correction(query, k=10, top_k=5)
        
        # Verify the correction was applied
        assert len(result_docs) == 2
        assert result_docs == corrected_docs
        assert metadata["correction_applied"] is True
        assert "retrieval_time" in metadata
        
        # Verify the correction engine was called
        mock_correction_instance.correct_retrieval.assert_called_once_with(query, sample_documents)
    
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_retrieve_with_correction_disabled(self, mock_vector_retriever, config_without_self_correction, sample_documents):
        """Test retrieve_with_correction when self-correction is disabled"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = sample_documents
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Initialize retrieval engine without self-correction
        engine = RetrievalEngine(config_without_self_correction)
        
        # Test retrieve with correction (should fall back to normal retrieval)
        query = "test query"
        result_docs, metadata = engine.retrieve_with_correction(query, k=10, top_k=2)
        
        # Should return top_k documents without correction
        assert len(result_docs) == 2
        assert result_docs == sample_documents[:2]
        assert metadata["correction_applied"] is False
        assert metadata["reason"] == "not_enabled"
    
    @patch('src.rag_engine.retrieval.retrieval_engine.SelfCorrectionEngine')
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_enable_disable_self_correction(self, mock_vector_retriever, mock_self_correction, config_without_self_correction):
        """Test enabling and disabling self-correction"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Mock the self-correction engine
        mock_correction_instance = Mock()
        mock_self_correction.return_value = mock_correction_instance
        
        # Initialize retrieval engine without self-correction
        engine = RetrievalEngine(config_without_self_correction)
        assert engine.is_self_correction_enabled() is False
        
        # Enable self-correction
        engine.enable_self_correction()
        assert engine.is_self_correction_enabled() is True
        assert engine.config.enable_self_correction is True
        
        # Disable self-correction
        engine.disable_self_correction()
        assert engine.is_self_correction_enabled() is False
        assert engine.config.enable_self_correction is False
    
    @patch('src.rag_engine.retrieval.retrieval_engine.SelfCorrectionEngine')
    @patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever')
    def test_get_retrieval_stats_with_self_correction(self, mock_vector_retriever, mock_self_correction, config_with_self_correction):
        """Test retrieval stats include self-correction information"""
        # Mock the vector retriever
        mock_retriever_instance = Mock()
        mock_vector_retriever.return_value = mock_retriever_instance
        
        # Mock the self-correction engine
        mock_correction_instance = Mock()
        mock_correction_stats = {
            "relevance_threshold": 0.7,
            "factuality_threshold": 0.7,
            "min_relevant_docs": 2
        }
        mock_correction_instance.get_correction_stats.return_value = mock_correction_stats
        mock_self_correction.return_value = mock_correction_instance
        
        # Initialize retrieval engine
        engine = RetrievalEngine(config_with_self_correction)
        
        # Get stats
        stats = engine.get_retrieval_stats()
        
        # Verify self-correction stats are included
        assert stats["self_correction_enabled"] is True
        assert stats["config"]["enable_self_correction"] is True
        assert stats["config"]["relevance_threshold"] == 0.7
        assert stats["config"]["factuality_threshold"] == 0.7
        assert stats["config"]["min_relevant_docs"] == 2
        assert "self_correction_stats" in stats
        assert stats["self_correction_stats"] == mock_correction_stats


if __name__ == "__main__":
    pytest.main([__file__])
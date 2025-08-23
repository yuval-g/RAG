"""
Unit tests for document re-ranking functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.retrieval.reranker import (
    ReRanker, 
    LLMReRanker, 
    ContextualCompressionReRanker
)
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


class TestLLMReRanker:
    """Test cases for LLMReRanker"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_model="gemini-2.0-flash-lite",
            temperature=0.0
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Python is a programming language used for web development and data science.",
                metadata={"source": "doc1"},
                doc_id="doc1"
            ),
            Document(
                content="Machine learning algorithms can be implemented in various programming languages.",
                metadata={"source": "doc2"},
                doc_id="doc2"
            ),
            Document(
                content="The weather today is sunny with a temperature of 75 degrees.",
                metadata={"source": "doc3"},
                doc_id="doc3"
            )
        ]
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_llm_reranker_initialization(self, mock_get_llm, config):
        """Test LLMReRanker initialization"""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        reranker = LLMReRanker(config)
        
        assert reranker.config == config
        mock_get_llm.assert_called_once_with("gemini-2.0-flash-lite", 0.0)
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_score_document(self, mock_get_llm, config, sample_documents):
        """Test document scoring functionality"""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "0.85"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        reranker = LLMReRanker(config)
        score = reranker._score_document("Python programming", sample_documents[0])
        
        assert score == 0.85
        assert 0.0 <= score <= 1.0
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_score_document_invalid_response(self, mock_get_llm, config, sample_documents):
        """Test document scoring with invalid LLM response"""
        # Mock invalid LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "invalid_score"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        reranker = LLMReRanker(config)
        score = reranker._score_document("Python programming", sample_documents[0])
        
        assert score == 0.5  # Default score for invalid response
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_score_document_exception(self, mock_get_llm, config, sample_documents):
        """Test document scoring with exception"""
        # Mock LLM exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_get_llm.return_value = mock_llm
        
        reranker = LLMReRanker(config)
        score = reranker._score_document("Python programming", sample_documents[0])
        
        assert score == 0.5  # Default score for exception
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_rerank_with_scores(self, mock_get_llm, config, sample_documents):
        """Test re-ranking with scores"""
        # Mock LLM responses with different scores
        mock_llm = Mock()
        mock_responses = [
            Mock(content="0.9"),   # High relevance for Python doc
            Mock(content="0.7"),   # Medium relevance for ML doc
            Mock(content="0.1")    # Low relevance for weather doc
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_get_llm.return_value = mock_llm
        
        reranker = LLMReRanker(config)
        results = reranker.rerank_with_scores("Python programming", sample_documents, top_k=2)
        
        assert len(results) == 2
        assert results[0][1] == 0.9  # Highest score first
        assert results[1][1] == 0.7  # Second highest score
        assert results[0][0].doc_id == "doc1"  # Python document first
    
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_rerank(self, mock_llm, config, sample_documents):
        """Test re-ranking without scores"""
        # Mock LLM responses
        mock_responses = [
            Mock(content="0.9"),
            Mock(content="0.7"),
            Mock(content="0.1")
        ]
        mock_llm.return_value.invoke.side_effect = mock_responses
        
        reranker = LLMReRanker(config)
        results = reranker.rerank("Python programming", sample_documents, top_k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].doc_id == "doc1"  # Python document first
    
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_rerank_empty_documents(self, mock_llm, config):
        """Test re-ranking with empty document list"""
        reranker = LLMReRanker(config)
        results = reranker.rerank("test query", [], top_k=5)
        
        assert results == []
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_score_clamping(self, mock_get_llm, config, sample_documents):
        """Test that scores are clamped between 0 and 1"""
        # Mock responses with out-of-range scores
        mock_llm = Mock()
        mock_responses = [
            Mock(content="1.5"),   # Above 1.0
            Mock(content="-0.5"),  # Below 0.0
            Mock(content="0.5")    # Valid score
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_get_llm.return_value = mock_llm
        
        reranker = LLMReRanker(config)
        results = reranker.rerank_with_scores("test query", sample_documents, top_k=3)
        
        # Check that all scores are within valid range
        for doc, score in results:
            assert 0.0 <= score <= 1.0
        
        # Check that out-of-range scores were clamped
        scores = [score for doc, score in results]
        assert 1.0 in scores  # 1.5 clamped to 1.0
        assert 0.0 in scores  # -0.5 clamped to 0.0


class TestContextualCompressionReRanker:
    """Test cases for ContextualCompressionReRanker"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(llm_model="gemini-2.0-flash-lite")
    
    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever"""
        return Mock()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Python is a programming language.",
                metadata={"source": "doc1"},
                doc_id="doc1"
            ),
            Document(
                content="Java is also a programming language.",
                metadata={"source": "doc2"},
                doc_id="doc2"
            )
        ]
    
    @patch('src.rag_engine.retrieval.reranker.ContextualCompressionRetriever')
    @patch('src.rag_engine.retrieval.reranker.LLMChainExtractor')
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_initialization(self, mock_get_llm, mock_extractor, mock_compression_retriever,
                          mock_base_retriever, config):
        """Test ContextualCompressionReRanker initialization"""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        reranker = ContextualCompressionReRanker(mock_base_retriever, config)
        
        assert reranker.config == config
        assert reranker.base_retriever == mock_base_retriever
        mock_get_llm.assert_called_once()
        mock_extractor.from_llm.assert_called_once()
        mock_compression_retriever.assert_called_once()
    
    @patch('src.rag_engine.retrieval.reranker.ContextualCompressionRetriever')
    @patch('src.rag_engine.retrieval.reranker.LLMChainExtractor')
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_rerank_with_scores(self, mock_llm, mock_extractor, mock_compression_retriever,
                              mock_base_retriever, config, sample_documents):
        """Test contextual compression re-ranking with scores"""
        # Mock compressed documents
        from langchain.schema import Document as LangChainDocument
        compressed_docs = [
            LangChainDocument(
                page_content="Python is a programming language.",
                metadata={"source": "doc1", "doc_id": "doc1"}
            )
        ]
        
        mock_compressor = Mock()
        mock_compressor.compress_documents.return_value = compressed_docs
        mock_extractor.from_llm.return_value = mock_compressor
        
        reranker = ContextualCompressionReRanker(mock_base_retriever, config)
        reranker.compressor = mock_compressor
        
        results = reranker.rerank_with_scores("Python programming", sample_documents, top_k=1)
        
        assert len(results) == 1
        assert results[0][1] == 1.0  # First document gets highest score
        assert results[0][0].content == "Python is a programming language."
    
    @patch('src.rag_engine.retrieval.reranker.ContextualCompressionRetriever')
    @patch('src.rag_engine.retrieval.reranker.LLMChainExtractor')
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_rerank_with_exception(self, mock_llm, mock_extractor, mock_compression_retriever,
                                 mock_base_retriever, config, sample_documents):
        """Test contextual compression with exception handling"""
        mock_compressor = Mock()
        mock_compressor.compress_documents.side_effect = Exception("Compression error")
        mock_extractor.from_llm.return_value = mock_compressor
        
        reranker = ContextualCompressionReRanker(mock_base_retriever, config)
        reranker.compressor = mock_compressor
        
        results = reranker.rerank_with_scores("test query", sample_documents, top_k=2)
        
        # Should fallback to original documents with default scores
        assert len(results) == 2
        assert all(score == 0.5 for doc, score in results)
    
    @patch('src.rag_engine.retrieval.reranker.ContextualCompressionRetriever')
    @patch('src.rag_engine.retrieval.reranker.LLMChainExtractor')
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_rerank_empty_documents(self, mock_llm, mock_extractor, mock_compression_retriever,
                                  mock_base_retriever, config):
        """Test contextual compression with empty document list"""
        reranker = ContextualCompressionReRanker(mock_base_retriever, config)
        results = reranker.rerank_with_scores("test query", [], top_k=5)
        
        assert results == []


class TestReRanker:
    """Test cases for main ReRanker class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig()
    
    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever"""
        return Mock()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Test document 1",
                metadata={"source": "test1"},
                doc_id="test1"
            ),
            Document(
                content="Test document 2",
                metadata={"source": "test2"},
                doc_id="test2"
            )
        ]
    
    @patch('src.rag_engine.retrieval.reranker.LLMReRanker')
    def test_initialization_llm_strategy(self, mock_llm_reranker, config):
        """Test ReRanker initialization with LLM strategy"""
        reranker = ReRanker(strategy="llm", config=config)
        
        assert reranker.strategy == "llm"
        mock_llm_reranker.assert_called_once_with(config)
    
    @patch('src.rag_engine.retrieval.reranker.ContextualCompressionReRanker')
    def test_initialization_contextual_strategy(self, mock_contextual_reranker, 
                                              mock_base_retriever, config):
        """Test ReRanker initialization with contextual strategy"""
        reranker = ReRanker(strategy="contextual", base_retriever=mock_base_retriever, config=config)
        
        assert reranker.strategy == "contextual"
        mock_contextual_reranker.assert_called_once_with(mock_base_retriever, config)
    
    def test_initialization_contextual_without_retriever(self, config):
        """Test ReRanker initialization with contextual strategy but no base retriever"""
        with pytest.raises(ValueError, match="base_retriever is required"):
            ReRanker(strategy="contextual", config=config)
    
    def test_initialization_invalid_strategy(self, config):
        """Test ReRanker initialization with invalid strategy"""
        with pytest.raises(ValueError, match="Unknown re-ranking strategy"):
            ReRanker(strategy="invalid_strategy", config=config)
    
    @patch('src.rag_engine.retrieval.reranker.LLMReRanker')
    def test_rerank_delegation(self, mock_llm_reranker, config, sample_documents):
        """Test that rerank method delegates to underlying reranker"""
        mock_reranker_instance = Mock()
        mock_reranker_instance.rerank.return_value = sample_documents[:1]
        mock_llm_reranker.return_value = mock_reranker_instance
        
        reranker = ReRanker(strategy="llm", config=config)
        results = reranker.rerank("test query", sample_documents, top_k=1)
        
        mock_reranker_instance.rerank.assert_called_once_with("test query", sample_documents, 1)
        assert results == sample_documents[:1]
    
    @patch('src.rag_engine.retrieval.reranker.LLMReRanker')
    def test_rerank_with_scores_delegation(self, mock_llm_reranker, config, sample_documents):
        """Test that rerank_with_scores method delegates to underlying reranker"""
        expected_results = [(sample_documents[0], 0.9)]
        mock_reranker_instance = Mock()
        mock_reranker_instance.rerank_with_scores.return_value = expected_results
        mock_llm_reranker.return_value = mock_reranker_instance
        
        reranker = ReRanker(strategy="llm", config=config)
        results = reranker.rerank_with_scores("test query", sample_documents, top_k=1)
        
        mock_reranker_instance.rerank_with_scores.assert_called_once_with("test query", sample_documents, 1)
        assert results == expected_results
    
    @patch('src.rag_engine.retrieval.reranker.LLMReRanker')
    def test_get_strategy(self, mock_llm_reranker, config):
        """Test get_strategy method"""
        reranker = ReRanker(strategy="llm", config=config)
        assert reranker.get_strategy() == "llm"
    
    @patch('src.rag_engine.retrieval.reranker.LLMReRanker')
    def test_set_config(self, mock_llm_reranker, config):
        """Test set_config method"""
        mock_reranker_instance = Mock()
        mock_reranker_instance.config = config
        mock_llm_reranker.return_value = mock_reranker_instance
        
        reranker = ReRanker(strategy="llm", config=config)
        new_config = PipelineConfig(temperature=0.5)
        reranker.set_config(new_config)
        
        assert reranker.config == new_config
        assert mock_reranker_instance.config == new_config


class TestReRankerIntegration:
    """Integration tests for re-ranking functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_model="gemini-2.0-flash-lite",
            temperature=0.0
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with varying relevance"""
        return [
            Document(
                content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and machine learning.",
                metadata={"source": "python_guide", "topic": "programming"},
                doc_id="python_doc"
            ),
            Document(
                content="JavaScript is a programming language primarily used for web development. It runs in browsers and enables interactive web pages.",
                metadata={"source": "js_guide", "topic": "programming"},
                doc_id="js_doc"
            ),
            Document(
                content="The weather forecast shows sunny skies with temperatures reaching 80 degrees Fahrenheit today.",
                metadata={"source": "weather_report", "topic": "weather"},
                doc_id="weather_doc"
            ),
            Document(
                content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                metadata={"source": "ml_intro", "topic": "ai"},
                doc_id="ml_doc"
            )
        ]
    
    @patch('src.rag_engine.retrieval.reranker.get_llm')
    def test_programming_query_reranking(self, mock_get_llm, config, sample_documents):
        """Test re-ranking for a programming-related query"""
        # Mock LLM responses with realistic scores
        mock_llm = Mock()
        mock_responses = [
            Mock(content="0.95"),  # Python doc - highly relevant
            Mock(content="0.85"),  # JavaScript doc - relevant
            Mock(content="0.05"),  # Weather doc - not relevant
            Mock(content="0.30")   # ML doc - somewhat relevant
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_get_llm.return_value = mock_llm
        
        reranker = ReRanker(strategy="llm", config=config)
        results = reranker.rerank_with_scores("Python programming tutorial", sample_documents, top_k=3)
        
        assert len(results) == 3
        
        # Check that Python document is ranked highest
        assert results[0][0].doc_id == "python_doc"
        assert results[0][1] == 0.95
    
    @patch('src.rag_engine.retrieval.reranker.ChatGoogleGenerativeAI')
    def test_top_k_limiting(self, mock_llm, config, sample_documents):
        """Test that top_k parameter correctly limits results"""
        # Mock all documents as equally relevant
        mock_responses = [Mock(content="0.8")] * len(sample_documents)
        mock_llm.return_value.invoke.side_effect = mock_responses
        
        reranker = ReRanker(strategy="llm", config=config)
        
        # Test different top_k values
        results_2 = reranker.rerank("test query", sample_documents, top_k=2)
        results_3 = reranker.rerank("test query", sample_documents, top_k=3)
        results_all = reranker.rerank("test query", sample_documents, top_k=10)
        
        assert len(results_2) == 2
        assert len(results_3) == 3
        assert len(results_all) == len(sample_documents)  # Should not exceed available documents sample_documents, top_k=10)
        
        assert len(results_2) == 2
        assert len(results_3) == 3
        assert len(results_all) == len(sample_documents)  # Should not exceed available documentssert len(results_all) == len(sample_documents)  # Should not exceed available documents
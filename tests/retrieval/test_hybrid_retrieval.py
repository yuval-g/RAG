"""
Unit tests for hybrid retrieval system
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.retrieval.retrieval_engine import RetrievalEngine
from src.rag_engine.retrieval.keyword_retriever import KeywordRetriever


class TestKeywordRetriever:
    """Test cases for KeywordRetriever"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = PipelineConfig()
        self.retriever = KeywordRetriever(self.config)
        
        # Sample documents for testing
        self.sample_docs = [
            Document(
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                metadata={"source": "doc1"},
                doc_id="doc1"
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers to process data.",
                metadata={"source": "doc2"},
                doc_id="doc2"
            ),
            Document(
                content="Natural language processing helps computers understand human language.",
                metadata={"source": "doc3"},
                doc_id="doc3"
            ),
            Document(
                content="Computer vision enables machines to interpret visual information from images.",
                metadata={"source": "doc4"},
                doc_id="doc4"
            )
        ]
    
    def test_initialization(self):
        """Test KeywordRetriever initialization"""
        assert self.retriever.config == self.config
        assert len(self.retriever.documents) == 0
        assert len(self.retriever.vocabulary) == 0
        assert self.retriever.k1 == 1.5
        assert self.retriever.b == 0.75
    
    def test_tokenization(self):
        """Test text tokenization"""
        text = "Machine learning is a subset of artificial intelligence!"
        tokens = self.retriever._tokenize(text)
        
        expected_tokens = ["machine", "learning", "subset", "artificial", "intelligence"]
        assert tokens == expected_tokens
    
    def test_add_documents(self):
        """Test adding documents to the index"""
        result = self.retriever.add_documents(self.sample_docs)
        
        assert result is True
        assert len(self.retriever.documents) == 4
        assert len(self.retriever.vocabulary) > 0
        assert self.retriever.average_document_length > 0
        
        # Check that document frequencies are calculated
        assert "machine" in self.retriever.document_frequencies
        assert "learning" in self.retriever.document_frequencies
    
    def test_bm25_score_calculation(self):
        """Test BM25 score calculation"""
        self.retriever.add_documents(self.sample_docs)
        
        query_terms = ["machine", "learning"]
        score = self.retriever._calculate_bm25_score(query_terms, "doc1")
        
        assert isinstance(score, float)
        assert score > 0  # Should have positive score for relevant document
    
    def test_retrieve_basic(self):
        """Test basic retrieval functionality"""
        self.retriever.add_documents(self.sample_docs)
        
        query = "machine learning algorithms"
        results = self.retriever.retrieve(query, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # First result should be most relevant (contains "machine learning")
        if results:
            assert "machine" in results[0].content.lower() or "learning" in results[0].content.lower()
    
    def test_retrieve_with_scores(self):
        """Test retrieval with BM25 scores"""
        self.retriever.add_documents(self.sample_docs)
        
        query = "neural networks deep learning"
        results = self.retriever.retrieve_with_scores(query, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(doc, Document) and isinstance(score, float) for doc, score in results)
        
        # Scores should be in descending order
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_empty_query(self):
        """Test handling of empty query"""
        self.retriever.add_documents(self.sample_docs)
        
        results = self.retriever.retrieve("", k=5)
        assert len(results) == 0
        
        results_with_scores = self.retriever.retrieve_with_scores("", k=5)
        assert len(results_with_scores) == 0
    
    def test_no_documents(self):
        """Test retrieval with no documents indexed"""
        results = self.retriever.retrieve("machine learning", k=5)
        assert len(results) == 0
        
        results_with_scores = self.retriever.retrieve_with_scores("machine learning", k=5)
        assert len(results_with_scores) == 0
    
    def test_get_statistics(self):
        """Test retrieval of index statistics"""
        self.retriever.add_documents(self.sample_docs)
        
        # Test overall statistics
        stats = self.retriever.get_index_statistics()
        assert stats["total_documents"] == 4
        assert stats["vocabulary_size"] > 0
        assert stats["average_document_length"] > 0
        
        # Test term statistics
        term_stats = self.retriever.get_term_statistics("machine")
        assert term_stats["in_vocabulary"] is True
        assert term_stats["document_frequency"] > 0
        
        # Test non-existent term
        term_stats = self.retriever.get_term_statistics("nonexistent")
        assert term_stats["in_vocabulary"] is False
    
    def test_clear_index(self):
        """Test clearing the index"""
        self.retriever.add_documents(self.sample_docs)
        assert len(self.retriever.documents) > 0
        
        result = self.retriever.clear_index()
        assert result is True
        assert len(self.retriever.documents) == 0
        assert len(self.retriever.vocabulary) == 0
        assert self.retriever.average_document_length == 0.0
    
    def test_bm25_parameters(self):
        """Test setting BM25 parameters"""
        self.retriever.set_bm25_parameters(k1=2.0, b=0.5)
        assert self.retriever.k1 == 2.0
        assert self.retriever.b == 0.5


class TestHybridRetrievalEngine:
    """Test cases for hybrid retrieval functionality in RetrievalEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = PipelineConfig()
        
        # Mock the vector retriever to avoid external dependencies
        with patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever') as mock_vector:
            mock_vector_instance = Mock()
            mock_vector_instance.retrieve.return_value = []
            mock_vector_instance.retrieve_with_scores.return_value = []
            mock_vector.return_value = mock_vector_instance
            
            self.engine = RetrievalEngine(self.config)
        
        # Sample documents for testing
        self.sample_docs = [
            Document(
                content="Machine learning algorithms are used in artificial intelligence applications.",
                metadata={"source": "doc1"},
                doc_id="doc1"
            ),
            Document(
                content="Deep learning neural networks process complex data patterns effectively.",
                metadata={"source": "doc2"},
                doc_id="doc2"
            ),
            Document(
                content="Natural language processing enables human-computer communication.",
                metadata={"source": "doc3"},
                doc_id="doc3"
            )
        ]
    
    def test_engine_initialization(self):
        """Test RetrievalEngine initialization with hybrid capabilities"""
        assert "vector" in self.engine.get_available_retrievers()
        assert "keyword" in self.engine.get_available_retrievers()
        assert self.engine.default_retriever_name == "vector"
    
    def test_add_documents_to_all_retrievers(self):
        """Test adding documents to all retrievers"""
        # Mock the add_documents method for both retrievers
        for retriever in self.engine.retrievers.values():
            retriever.add_documents = Mock(return_value=True)
        
        result = self.engine.add_documents(self.sample_docs)
        assert result is True
        
        # Verify all retrievers received the documents
        for retriever in self.engine.retrievers.values():
            retriever.add_documents.assert_called_once_with(self.sample_docs)
    
    def test_hybrid_retrieve_basic(self):
        """Test basic hybrid retrieval functionality"""
        # Mock retrieval results from different retrievers
        vector_results = [
            (self.sample_docs[0], 0.9),
            (self.sample_docs[1], 0.7)
        ]
        keyword_results = [
            (self.sample_docs[1], 0.8),
            (self.sample_docs[2], 0.6)
        ]
        
        self.engine.retrievers["vector"].retrieve_with_scores = Mock(return_value=vector_results)
        self.engine.retrievers["keyword"].retrieve_with_scores = Mock(return_value=keyword_results)
        
        results = self.engine.hybrid_retrieve("machine learning", k=3)
        
        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_hybrid_retrieve_with_weights(self):
        """Test hybrid retrieval with custom weights"""
        # Mock retrieval results
        vector_results = [(self.sample_docs[0], 0.9)]
        keyword_results = [(self.sample_docs[1], 0.8)]
        
        self.engine.retrievers["vector"].retrieve_with_scores = Mock(return_value=vector_results)
        self.engine.retrievers["keyword"].retrieve_with_scores = Mock(return_value=keyword_results)
        
        results = self.engine.hybrid_retrieve(
            "test query", 
            k=2, 
            vector_weight=0.8, 
            keyword_weight=0.2
        )
        
        assert len(results) <= 2
    
    def test_hybrid_retrieve_insufficient_retrievers(self):
        """Test hybrid retrieval with insufficient retrievers"""
        # Remove one retriever to test fallback
        del self.engine.retrievers["keyword"]
        
        self.engine.retrievers["vector"].retrieve = Mock(return_value=[self.sample_docs[0]])
        
        results = self.engine.hybrid_retrieve("test query", k=2)
        
        # Should fall back to single retriever
        assert len(results) <= 2
        self.engine.retrievers["vector"].retrieve.assert_called_once()
    
    def test_long_context_retrieve(self):
        """Test long-context retrieval functionality"""
        # Mock initial retrieval
        self.engine.retrievers["vector"].retrieve = Mock(return_value=self.sample_docs)
        
        results, metadata = self.engine.long_context_retrieve(
            "test query", 
            k=10, 
            context_window_size=1000
        )
        
        assert isinstance(results, list)
        assert isinstance(metadata, dict)
        assert "strategy" in metadata
        assert metadata["strategy"] == "long_context"
        assert "documents_processed" in metadata
        assert "estimated_tokens" in metadata
    
    def test_filter_docs_for_context_window(self):
        """Test document filtering for context window"""
        # Create documents with known content lengths
        long_docs = [
            Document(content="a" * 1000, doc_id="long1"),  # ~250 tokens
            Document(content="b" * 2000, doc_id="long2"),  # ~500 tokens
            Document(content="c" * 1000, doc_id="long3"),  # ~250 tokens
        ]
        
        filtered_docs, total_tokens = self.engine._filter_docs_for_context_window(
            long_docs, max_tokens=600
        )
        
        assert len(filtered_docs) <= len(long_docs)
        assert total_tokens <= 600
    
    def test_adaptive_retrieve_auto_strategy(self):
        """Test adaptive retrieval with automatic strategy selection"""
        # Mock different retrieval methods
        self.engine.hybrid_retrieve = Mock(return_value=[self.sample_docs[0]])
        self.engine.long_context_retrieve = Mock(return_value=([self.sample_docs[0]], {}))
        self.engine.retrieve_with_rerank = Mock(return_value=[self.sample_docs[0]])
        self.engine.retrieve = Mock(return_value=[self.sample_docs[0]])
        
        # Test different query types
        queries = [
            ("find specific machine learning algorithms", "hybrid"),  # Should trigger hybrid
            ("compare and analyze different approaches to deep learning neural networks", "rerank"),  # Should trigger rerank
            ("explain the detailed methodology and comprehensive analysis of various machine learning techniques", "long_context"),  # Should trigger long_context
            ("what is AI", "vector")  # Should trigger vector
        ]
        
        for query, expected_strategy in queries:
            results, metadata = self.engine.adaptive_retrieve(query, k=3, strategy="auto")
            
            assert isinstance(results, list)
            assert isinstance(metadata, dict)
            assert "selected_strategy" in metadata
    
    def test_select_optimal_strategy(self):
        """Test optimal strategy selection logic"""
        # Test long query
        long_query = " ".join(["word"] * 25)
        strategy = self.engine._select_optimal_strategy(long_query)
        assert strategy == "long_context"
        
        # Test keyword query
        keyword_query = "find specific machine learning algorithms"
        strategy = self.engine._select_optimal_strategy(keyword_query)
        assert strategy == "hybrid"
        
        # Test complex query
        complex_query = "compare and analyze different approaches"
        strategy = self.engine._select_optimal_strategy(complex_query)
        assert strategy == "rerank"
        
        # Test simple query
        simple_query = "what is AI"
        strategy = self.engine._select_optimal_strategy(simple_query)
        assert strategy == "vector"
    
    def test_combine_retrieval_results(self):
        """Test combining results from multiple retrievers"""
        retrieval_results = {
            "vector": [
                (self.sample_docs[0], 0.9),
                (self.sample_docs[1], 0.7)
            ],
            "keyword": [
                (self.sample_docs[1], 0.8),  # Same document with different score
                (self.sample_docs[2], 0.6)
            ]
        }
        
        combined = self.engine._combine_retrieval_results(
            retrieval_results,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=3
        )
        
        assert len(combined) <= 3
        assert all(isinstance(doc, Document) for doc in combined)
    
    def test_retrieve_with_scores_safe(self):
        """Test safe retrieval with scores"""
        # Mock retriever with retrieve_with_scores method
        mock_retriever = Mock()
        mock_retriever.retrieve_with_scores.return_value = [(self.sample_docs[0], 0.8)]
        self.engine.retrievers["test"] = mock_retriever
        
        results = self.engine._retrieve_with_scores_safe("test query", 5, "test")
        
        assert len(results) == 1
        assert results[0][0] == self.sample_docs[0]
        assert results[0][1] == 0.8
    
    def test_retrieve_with_scores_safe_fallback(self):
        """Test safe retrieval fallback when retrieve_with_scores not available"""
        # Mock retriever without retrieve_with_scores method
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [self.sample_docs[0]]
        del mock_retriever.retrieve_with_scores  # Remove the method
        self.engine.retrievers["test"] = mock_retriever
        
        results = self.engine._retrieve_with_scores_safe("test query", 5, "test")
        
        assert len(results) == 1
        assert results[0][0] == self.sample_docs[0]
        assert results[0][1] == 0.5  # Default score
    
    def test_error_handling(self):
        """Test error handling in hybrid retrieval"""
        # Mock retriever that raises an exception
        mock_retriever = Mock()
        mock_retriever.retrieve_with_scores.side_effect = Exception("Test error")
        self.engine.retrievers["error_retriever"] = mock_retriever
        
        results = self.engine._retrieve_with_scores_safe("test query", 5, "error_retriever")
        assert results == []
        
        # Test hybrid retrieval with error
        results = self.engine.hybrid_retrieve("test query", k=3, retriever_names=["error_retriever"])
        assert results == []


class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval system"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.config = PipelineConfig()
        
        # Create a real keyword retriever for integration testing
        self.keyword_retriever = KeywordRetriever(self.config)
        
        # Sample documents
        self.sample_docs = [
            Document(
                content="Machine learning is a powerful tool for data analysis and pattern recognition.",
                metadata={"topic": "ML", "difficulty": "beginner"},
                doc_id="ml_intro"
            ),
            Document(
                content="Deep learning neural networks use multiple layers to process complex data.",
                metadata={"topic": "DL", "difficulty": "advanced"},
                doc_id="dl_networks"
            ),
            Document(
                content="Natural language processing helps computers understand human language.",
                metadata={"topic": "NLP", "difficulty": "intermediate"},
                doc_id="nlp_basics"
            ),
            Document(
                content="Computer vision algorithms enable machines to interpret visual information.",
                metadata={"topic": "CV", "difficulty": "intermediate"},
                doc_id="cv_algorithms"
            )
        ]
    
    def test_keyword_retriever_end_to_end(self):
        """Test complete keyword retrieval workflow"""
        # Add documents
        result = self.keyword_retriever.add_documents(self.sample_docs)
        assert result is True
        
        # Test retrieval
        results = self.keyword_retriever.retrieve("machine learning data analysis", k=2)
        assert len(results) > 0
        assert any("machine learning" in doc.content.lower() for doc in results)
        
        # Test retrieval with scores
        results_with_scores = self.keyword_retriever.retrieve_with_scores("neural networks", k=3)
        assert len(results_with_scores) > 0
        assert all(score > 0 for _, score in results_with_scores)
    
    def test_hybrid_system_integration(self):
        """Test integration of hybrid retrieval system"""
        # Mock vector retriever for integration test
        with patch('src.rag_engine.retrieval.retrieval_engine.VectorRetriever') as mock_vector:
            mock_vector_instance = Mock()
            mock_vector_instance.add_documents.return_value = True
            mock_vector_instance.retrieve_with_scores.return_value = [
                (self.sample_docs[0], 0.85),
                (self.sample_docs[2], 0.75)
            ]
            mock_vector.return_value = mock_vector_instance
            
            # Create engine and add documents
            engine = RetrievalEngine(self.config)
            engine.add_documents(self.sample_docs)
            
            # Test hybrid retrieval
            results = engine.hybrid_retrieve("machine learning neural networks", k=3)
            assert len(results) > 0
            
            # Test adaptive retrieval
            results, metadata = engine.adaptive_retrieve("find machine learning algorithms", k=2)
            assert len(results) > 0
            assert "selected_strategy" in metadata


if __name__ == "__main__":
    pytest.main([__file__])
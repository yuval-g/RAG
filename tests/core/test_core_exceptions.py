"""
Unit tests for core exception classes
"""

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.core.exceptions import (
    RAGEngineError, ConfigurationError, IndexingError, RetrievalError,
    GenerationError, EvaluationError, RoutingError, QueryProcessingError,
    VectorStoreError, LLMProviderError, EmbeddingProviderError
)


class TestRAGEngineError:
    """Test cases for base RAGEngineError"""
    
    def test_rag_engine_error_creation(self):
        """Test creating base RAG engine error"""
        error = RAGEngineError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_rag_engine_error_inheritance(self):
        """Test that RAGEngineError inherits from Exception"""
        error = RAGEngineError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, RAGEngineError)
    
    def test_rag_engine_error_empty_message(self):
        """Test creating error with empty message"""
        error = RAGEngineError("")
        
        assert str(error) == ""
    
    def test_rag_engine_error_no_message(self):
        """Test creating error with no message"""
        error = RAGEngineError()
        
        assert str(error) == ""


class TestSpecificErrors:
    """Test cases for specific error types"""
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Invalid configuration")
        
        assert str(error) == "Invalid configuration"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, ConfigurationError)
    
    def test_indexing_error(self):
        """Test IndexingError"""
        error = IndexingError("Indexing failed")
        
        assert str(error) == "Indexing failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, IndexingError)
    
    def test_retrieval_error(self):
        """Test RetrievalError"""
        error = RetrievalError("Retrieval failed")
        
        assert str(error) == "Retrieval failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, RetrievalError)
    
    def test_generation_error(self):
        """Test GenerationError"""
        error = GenerationError("Generation failed")
        
        assert str(error) == "Generation failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, GenerationError)
    
    def test_evaluation_error(self):
        """Test EvaluationError"""
        error = EvaluationError("Evaluation failed")
        
        assert str(error) == "Evaluation failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, EvaluationError)
    
    def test_routing_error(self):
        """Test RoutingError"""
        error = RoutingError("Routing failed")
        
        assert str(error) == "Routing failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, RoutingError)
    
    def test_query_processing_error(self):
        """Test QueryProcessingError"""
        error = QueryProcessingError("Query processing failed")
        
        assert str(error) == "Query processing failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, QueryProcessingError)
    
    def test_vector_store_error(self):
        """Test VectorStoreError"""
        error = VectorStoreError("Vector store operation failed")
        
        assert str(error) == "Vector store operation failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, VectorStoreError)
    
    def test_llm_provider_error(self):
        """Test LLMProviderError"""
        error = LLMProviderError("LLM provider failed")
        
        assert str(error) == "LLM provider failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, LLMProviderError)
    
    def test_embedding_provider_error(self):
        """Test EmbeddingProviderError"""
        error = EmbeddingProviderError("Embedding provider failed")
        
        assert str(error) == "Embedding provider failed"
        assert isinstance(error, RAGEngineError)
        assert isinstance(error, EmbeddingProviderError)


class TestErrorInheritanceHierarchy:
    """Test cases for error inheritance hierarchy"""
    
    def test_all_errors_inherit_from_base(self):
        """Test that all specific errors inherit from RAGEngineError"""
        error_classes = [
            ConfigurationError, IndexingError, RetrievalError, GenerationError,
            EvaluationError, RoutingError, QueryProcessingError, VectorStoreError,
            LLMProviderError, EmbeddingProviderError
        ]
        
        for error_class in error_classes:
            error = error_class("Test message")
            assert isinstance(error, RAGEngineError)
            assert isinstance(error, Exception)
    
    def test_error_class_names(self):
        """Test that error classes have expected names"""
        expected_names = [
            "RAGEngineError", "ConfigurationError", "IndexingError", "RetrievalError",
            "GenerationError", "EvaluationError", "RoutingError", "QueryProcessingError",
            "VectorStoreError", "LLMProviderError", "EmbeddingProviderError"
        ]
        
        error_classes = [
            RAGEngineError, ConfigurationError, IndexingError, RetrievalError,
            GenerationError, EvaluationError, RoutingError, QueryProcessingError,
            VectorStoreError, LLMProviderError, EmbeddingProviderError
        ]
        
        for error_class, expected_name in zip(error_classes, expected_names):
            assert error_class.__name__ == expected_name


class TestErrorUsageScenarios:
    """Test cases for realistic error usage scenarios"""
    
    def test_configuration_error_scenario(self):
        """Test realistic configuration error scenario"""
        try:
            # Simulate configuration validation
            config_value = None
            if config_value is None:
                raise ConfigurationError("Missing required configuration: llm_provider")
        except ConfigurationError as e:
            assert "Missing required configuration" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_indexing_error_scenario(self):
        """Test realistic indexing error scenario"""
        try:
            # Simulate indexing failure
            documents = []
            if not documents:
                raise IndexingError("Cannot index empty document list")
        except IndexingError as e:
            assert "Cannot index empty document list" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_retrieval_error_scenario(self):
        """Test realistic retrieval error scenario"""
        try:
            # Simulate retrieval failure
            vector_store_available = False
            if not vector_store_available:
                raise RetrievalError("Vector store is not available for retrieval")
        except RetrievalError as e:
            assert "Vector store is not available" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_generation_error_scenario(self):
        """Test realistic generation error scenario"""
        try:
            # Simulate generation failure
            llm_response = None
            if llm_response is None:
                raise GenerationError("LLM failed to generate response")
        except GenerationError as e:
            assert "LLM failed to generate response" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_evaluation_error_scenario(self):
        """Test realistic evaluation error scenario"""
        try:
            # Simulate evaluation failure
            test_cases = []
            responses = ["response1"]
            if len(test_cases) != len(responses):
                raise EvaluationError("Mismatch between test cases and responses")
        except EvaluationError as e:
            assert "Mismatch between test cases and responses" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_routing_error_scenario(self):
        """Test realistic routing error scenario"""
        try:
            # Simulate routing failure
            available_routes = []
            if not available_routes:
                raise RoutingError("No routes available for query routing")
        except RoutingError as e:
            assert "No routes available" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_query_processing_error_scenario(self):
        """Test realistic query processing error scenario"""
        try:
            # Simulate query processing failure
            query = ""
            if not query.strip():
                raise QueryProcessingError("Cannot process empty query")
        except QueryProcessingError as e:
            assert "Cannot process empty query" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_vector_store_error_scenario(self):
        """Test realistic vector store error scenario"""
        try:
            # Simulate vector store failure
            connection_status = False
            if not connection_status:
                raise VectorStoreError("Failed to connect to vector store")
        except VectorStoreError as e:
            assert "Failed to connect to vector store" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_llm_provider_error_scenario(self):
        """Test realistic LLM provider error scenario"""
        try:
            # Simulate LLM provider failure
            api_key = None
            if not api_key:
                raise LLMProviderError("LLM provider API key not configured")
        except LLMProviderError as e:
            assert "API key not configured" in str(e)
            assert isinstance(e, RAGEngineError)
    
    def test_embedding_provider_error_scenario(self):
        """Test realistic embedding provider error scenario"""
        try:
            # Simulate embedding provider failure
            model_loaded = False
            if not model_loaded:
                raise EmbeddingProviderError("Embedding model failed to load")
        except EmbeddingProviderError as e:
            assert "Embedding model failed to load" in str(e)
            assert isinstance(e, RAGEngineError)


class TestErrorChaining:
    """Test cases for error chaining and context"""
    
    def test_error_chaining_with_cause(self):
        """Test error chaining with original cause"""
        try:
            try:
                # Simulate original error
                raise ValueError("Original error")
            except ValueError as original_error:
                raise ConfigurationError("Configuration failed") from original_error
        except ConfigurationError as e:
            assert str(e) == "Configuration failed"
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"
    
    def test_error_with_additional_context(self):
        """Test error with additional context information"""
        context = {"operation": "document_indexing", "document_count": 100}
        error_msg = f"Indexing failed for operation: {context['operation']}"
        
        error = IndexingError(error_msg)
        
        assert "document_indexing" in str(error)
        assert isinstance(error, RAGEngineError)


class TestErrorMessages:
    """Test cases for error message formatting"""
    
    def test_error_with_formatted_message(self):
        """Test error with formatted message"""
        doc_id = "doc_123"
        error = IndexingError(f"Failed to index document: {doc_id}")
        
        assert "doc_123" in str(error)
        assert "Failed to index document" in str(error)
    
    def test_error_with_multiline_message(self):
        """Test error with multiline message"""
        message = """Configuration error occurred:
        - Missing API key
        - Invalid model name
        - Timeout value too low"""
        
        error = ConfigurationError(message)
        
        assert "Missing API key" in str(error)
        assert "Invalid model name" in str(error)
        assert "Timeout value too low" in str(error)
    
    def test_error_message_preservation(self):
        """Test that error messages are preserved exactly"""
        original_message = "Exact error message with special chars: @#$%"
        error = RAGEngineError(original_message)
        
        assert str(error) == original_message
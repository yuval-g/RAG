"""
Simple unit tests for core exception classes
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


def test_rag_engine_error_creation():
    """Test creating base RAG engine error"""
    error = RAGEngineError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_configuration_error():
    """Test ConfigurationError"""
    error = ConfigurationError("Invalid configuration")
    assert str(error) == "Invalid configuration"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, ConfigurationError)


def test_indexing_error():
    """Test IndexingError"""
    error = IndexingError("Indexing failed")
    assert str(error) == "Indexing failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, IndexingError)


def test_retrieval_error():
    """Test RetrievalError"""
    error = RetrievalError("Retrieval failed")
    assert str(error) == "Retrieval failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, RetrievalError)


def test_generation_error():
    """Test GenerationError"""
    error = GenerationError("Generation failed")
    assert str(error) == "Generation failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, GenerationError)


def test_evaluation_error():
    """Test EvaluationError"""
    error = EvaluationError("Evaluation failed")
    assert str(error) == "Evaluation failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, EvaluationError)


def test_routing_error():
    """Test RoutingError"""
    error = RoutingError("Routing failed")
    assert str(error) == "Routing failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, RoutingError)


def test_query_processing_error():
    """Test QueryProcessingError"""
    error = QueryProcessingError("Query processing failed")
    assert str(error) == "Query processing failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, QueryProcessingError)


def test_vector_store_error():
    """Test VectorStoreError"""
    error = VectorStoreError("Vector store operation failed")
    assert str(error) == "Vector store operation failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, VectorStoreError)


def test_llm_provider_error():
    """Test LLMProviderError"""
    error = LLMProviderError("LLM provider failed")
    assert str(error) == "LLM provider failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, LLMProviderError)


def test_embedding_provider_error():
    """Test EmbeddingProviderError"""
    error = EmbeddingProviderError("Embedding provider failed")
    assert str(error) == "Embedding provider failed"
    assert isinstance(error, RAGEngineError)
    assert isinstance(error, EmbeddingProviderError)


def test_all_errors_inherit_from_base():
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


def test_error_chaining():
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
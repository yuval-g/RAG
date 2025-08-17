"""
Custom exception classes for the RAG system
"""


class RAGEngineError(Exception):
    """Base exception for RAG engine errors"""
    pass


class ConfigurationError(RAGEngineError):
    """Configuration-related errors"""
    pass


class IndexingError(RAGEngineError):
    """Indexing operation errors"""
    pass


class RetrievalError(RAGEngineError):
    """Retrieval operation errors"""
    pass


class GenerationError(RAGEngineError):
    """Generation operation errors"""
    pass


class EvaluationError(RAGEngineError):
    """Evaluation operation errors"""
    pass


class RoutingError(RAGEngineError):
    """Routing operation errors"""
    pass


class QueryProcessingError(RAGEngineError):
    """Query processing errors"""
    pass


class VectorStoreError(RAGEngineError):
    """Vector store operation errors"""
    pass


class LLMProviderError(RAGEngineError):
    """LLM provider errors"""
    pass


class EmbeddingProviderError(RAGEngineError):
    """Embedding provider errors"""
    pass


class ExternalServiceError(RAGEngineError):
    """External service errors (for circuit breaker)"""
    pass


class ResilienceError(RAGEngineError):
    """Resilience pattern errors"""
    pass


class MonitoringError(RAGEngineError):
    """Monitoring system errors"""
    pass
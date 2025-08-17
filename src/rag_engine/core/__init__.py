"""
Core RAG engine components and orchestration
"""

from .engine import RAGEngine
from .config import PipelineConfig, ConfigurationManager
from .models import (
    Document, ProcessedQuery, RAGResponse, EvaluationResult, 
    TestCase, RoutingDecision, RouteConfig
)
from .interfaces import (
    BaseIndexer, BaseRetriever, BaseQueryProcessor, BaseRouter,
    BaseLLMProvider, BaseEmbeddingProvider, BaseVectorStore,
    BaseEvaluator, BaseReRanker, BaseRAGEngine
)
from .exceptions import (
    RAGEngineError, ConfigurationError, IndexingError, RetrievalError,
    GenerationError, EvaluationError, RoutingError, QueryProcessingError,
    VectorStoreError, LLMProviderError, EmbeddingProviderError
)
from .embedding_providers import (
    OpenAIEmbeddingProvider, HuggingFaceEmbeddingProvider,
    EmbeddingProviderFactory, create_embedding_provider
)

__all__ = [
    # Main classes
    'RAGEngine',
    'PipelineConfig',
    'ConfigurationManager',
    
    # Data models
    'Document',
    'ProcessedQuery', 
    'RAGResponse',
    'EvaluationResult',
    'TestCase',
    'RoutingDecision',
    'RouteConfig',
    
    # Interfaces
    'BaseIndexer',
    'BaseRetriever',
    'BaseQueryProcessor',
    'BaseRouter',
    'BaseLLMProvider',
    'BaseEmbeddingProvider',
    'BaseVectorStore',
    'BaseEvaluator',
    'BaseReRanker',
    'BaseRAGEngine',
    
    # Exceptions
    'RAGEngineError',
    'ConfigurationError',
    'IndexingError',
    'RetrievalError',
    'GenerationError',
    'EvaluationError',
    'RoutingError',
    'QueryProcessingError',
    'VectorStoreError',
    'LLMProviderError',
    'EmbeddingProviderError',
    
    # Embedding providers
    'OpenAIEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    'EmbeddingProviderFactory',
    'create_embedding_provider',
]
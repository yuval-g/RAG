"""
Vector store provider implementations for multi-backend support
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
from .models import Document
from .config import PipelineConfig
from .interfaces import BaseVectorStoreProvider
from .exceptions import VectorStoreError, ExternalServiceError
from .resilience import resilient_call, RetryConfig, CircuitBreakerConfig, get_resilience_manager

logger = logging.getLogger(__name__)


class VectorStoreProvider(BaseVectorStoreProvider):
    """Abstract base class for vector store providers"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._store = None
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents with embeddings to the store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores"""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        pass
    
    @abstractmethod
    def clear_store(self) -> bool:
        """Clear all documents from the store"""
        pass
    
    @abstractmethod
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        pass
    
    def get_native_store(self):
        """Get the native vector store instance for framework integration"""
        return self._store


class VectorStoreManager:
    """Manager for switching between different vector store providers"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._providers: Dict[str, type] = {}
        self._current_provider: Optional[VectorStoreProvider] = None
        
        # Register default providers
        self._register_default_providers()
    
    def _register_default_providers(self) -> None:
        """Register the default vector store providers"""
        try:
            from .providers import ChromaProvider, PineconeProvider, WeaviateProvider
            
            self._providers['chroma'] = ChromaProvider
            self._providers['pinecone'] = PineconeProvider
            self._providers['weaviate'] = WeaviateProvider
            
            logger.info("Registered default vector store providers: chroma, pinecone, weaviate")
        except ImportError as e:
            logger.warning(f"Failed to register some providers: {str(e)}")
    
    def register_provider(self, name: str, provider_class: type) -> None:
        """Register a vector store provider"""
        if not issubclass(provider_class, VectorStoreProvider):
            raise ValueError(f"Provider must inherit from VectorStoreProvider")
        
        self._providers[name] = provider_class
        logger.info(f"Registered vector store provider: {name}")
    
    def get_provider(self, name: Optional[str] = None) -> VectorStoreProvider:
        """Get a vector store provider by name with resilience configuration"""
        provider_name = name or self.config.vector_store
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown vector store provider: {provider_name}. Available: {list(self._providers.keys())}")
        
        # Configure resilience patterns for the provider
        self._configure_resilience_for_provider(provider_name)
        
        # Create provider instance if not already created or if switching providers
        current_provider_name = None
        if self._current_provider:
            current_provider_name = self._current_provider.__class__.__name__.lower().replace('provider', '')
        
        if self._current_provider is None or provider_name != current_provider_name:
            provider_class = self._providers[provider_name]
            self._current_provider = provider_class(self.config)
            self._current_provider.initialize(**self.config.vector_store_config)
        
        return self._current_provider
    
    def _configure_resilience_for_provider(self, provider_name: str) -> None:
        """Configure resilience patterns for a vector store provider"""
        manager = get_resilience_manager()
        
        # Configure circuit breakers for different operations
        operations = ['search', 'add', 'delete', 'clear']
        for operation in operations:
            cb_name = f"{provider_name}_vectorstore_{operation}"
            if not manager.get_circuit_breaker(cb_name):
                cb_config = CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60.0,
                    expected_exception=ExternalServiceError,
                    name=cb_name
                )
                manager.create_circuit_breaker(cb_name, cb_config)
        
        # Configure retry handlers
        retry_name = f"{provider_name}_vectorstore_retry"
        if not manager.get_retry_handler(retry_name):
            retry_config = RetryConfig(
                max_attempts=3,
                base_delay=0.5,
                max_delay=5.0,
                exponential_base=2.0,
                jitter=True,
                retryable_exceptions=[ExternalServiceError, ConnectionError, TimeoutError]
            )
            manager.create_retry_handler(retry_name, retry_config)
        
        logger.info(f"Configured resilience patterns for {provider_name} vector store provider")
    
    def list_providers(self) -> List[str]:
        """List all registered providers"""
        return list(self._providers.keys())
    
    def switch_provider(self, name: str, **kwargs) -> VectorStoreProvider:
        """Switch to a different vector store provider"""
        if name not in self._providers:
            raise ValueError(f"Unknown vector store provider: {name}. Available: {list(self._providers.keys())}")
        
        provider_class = self._providers[name]
        self._current_provider = provider_class(self.config)
        self._current_provider.initialize(**kwargs)
        
        logger.info(f"Switched to vector store provider: {name}")
        return self._current_provider


def create_vector_store_manager(config: PipelineConfig) -> VectorStoreManager:
    """Factory function to create a vector store manager with all providers registered"""
    return VectorStoreManager(config)
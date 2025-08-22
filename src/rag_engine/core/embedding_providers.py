"""
Embedding provider implementations for the RAG system
"""

from typing import List, Dict, Any, Optional
import logging
import os
from abc import ABC, abstractmethod

from .interfaces import BaseEmbeddingProvider
from .exceptions import RAGEngineError, ExternalServiceError
from .resilience import resilient_call, RetryConfig, CircuitBreakerConfig, get_resilience_manager


logger = logging.getLogger(__name__)


class EmbeddingProviderError(RAGEngineError):
    """Embedding provider specific errors"""
    pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI embedding provider
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI embedding model name
            dimensions: Embedding dimensions (for newer models)
            **kwargs: Additional parameters for the embedding model
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.kwargs = kwargs
        self._client = None
        self._embedding_dimension = None
        
        # Initialize the client
        self._initialize_client()
    
    import os
from openai import OpenAI # Moved to top level for easier mocking


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI embedding provider
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI embedding model name
            dimensions: Embedding dimensions (for newer models)
            **kwargs: Additional parameters for the embedding model
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.kwargs = kwargs
        self._client = None
        self._embedding_dimension = None
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client"""
        try:
            
            # Use provided API key or get from environment
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EmbeddingProviderError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            
            self._client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI embedding provider with model: {self.model}")
            
        except ImportError:
            raise EmbeddingProviderError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents with resilience patterns"""
        if not texts:
            return []
        
        def _embed_internal():
            try:
                # Prepare parameters
                params = {
                    "input": texts,
                    "model": self.model,
                    **self.kwargs
                }
                
                # Add dimensions parameter for newer models
                if self.dimensions:
                    params["dimensions"] = self.dimensions
                
                # Call OpenAI API
                response = self._client.embeddings.create(**params)
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                # Cache embedding dimension
                if embeddings and self._embedding_dimension is None:
                    self._embedding_dimension = len(embeddings[0])
                
                logger.debug(f"Generated embeddings for {len(texts)} documents")
                return embeddings
                
            except Exception as e:
                logger.error(f"Failed to generate document embeddings: {str(e)}")
                raise ExternalServiceError(f"Failed to generate document embeddings: {str(e)}")
        
        def _fallback_embed():
            logger.warning("Embedding generation failed, returning zero vectors")
            dimension = self.get_embedding_dimension()
            return [[0.0] * dimension for _ in texts]
        
        try:
            return resilient_call(
                _embed_internal,
                circuit_breaker_name="openai_embeddings",
                retry_name="openai_embeddings_retry",
                fallback_operation="openai_embed_documents",
                fallback_func=_fallback_embed
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate document embeddings after all resilience attempts: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        if not text:
            raise EmbeddingProviderError("Query text cannot be empty")
        
        try:
            embeddings = self.embed_documents([text])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate query embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self._embedding_dimension is None:
            # Generate a test embedding to determine dimension
            try:
                test_embedding = self.embed_query("test")
                self._embedding_dimension = len(test_embedding)
            except Exception as e:
                # Default dimensions for known models
                model_dimensions = {
                    "text-embedding-ada-002": 1536,
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                }
                self._embedding_dimension = model_dimensions.get(self.model, 1536)
                logger.warning(f"Could not determine embedding dimension, using default: {self._embedding_dimension}")
        
        return self._embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "provider": "openai",
            "model": self.model,
            "dimensions": self.get_embedding_dimension(),
            "max_input_tokens": 8191,  # Default for most OpenAI embedding models
        }


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider implementation"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize HuggingFace embedding provider
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run the model on ('cpu', 'cuda', etc.)
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional parameters for the model
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.kwargs = kwargs
        self._model = None
        self._embedding_dimension = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the HuggingFace model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load the model
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                **self.kwargs
            )
            
            # Get embedding dimension
            self._embedding_dimension = self._model.get_sentence_embedding_dimension()
            
            logger.info(f"Initialized HuggingFace embedding provider with model: {self.model_name} on {self.device}")
            
        except ImportError:
            raise EmbeddingProviderError(
                "sentence-transformers package not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize HuggingFace model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        if not texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert to list of lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.debug(f"Generated embeddings for {len(texts)} documents")
            return embeddings_list
            
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate document embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        if not text:
            raise EmbeddingProviderError("Query text cannot be empty")
        
        try:
            embeddings = self.embed_documents([text])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate query embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self._embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "provider": "huggingface",
            "model": self.model_name,
            "dimensions": self.get_embedding_dimension(),
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }


class EmbeddingProviderFactory:
    """Factory class for creating embedding providers"""
    
    _providers = {
        "openai": OpenAIEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
    }
    
    @classmethod
    def create_provider(
        self,
        provider_type: str,
        **kwargs
    ) -> BaseEmbeddingProvider:
        """
        Create an embedding provider instance
        
        Args:
            provider_type: Type of provider ('openai', 'huggingface')
            **kwargs: Provider-specific configuration
            
        Returns:
            BaseEmbeddingProvider: Configured embedding provider
        """
        if provider_type not in self._providers:
            raise EmbeddingProviderError(
                f"Unknown embedding provider: {provider_type}. "
                f"Available providers: {list(self._providers.keys())}"
            )
        
        provider_class = self._providers[provider_type]
        return provider_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available embedding providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type
    ) -> None:
        """
        Register a new embedding provider
        
        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise EmbeddingProviderError(
                f"Provider class must inherit from BaseEmbeddingProvider"
            )
        
        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")


# Convenience function for creating providers
def create_embedding_provider(
    provider_type: str,
    **kwargs
) -> BaseEmbeddingProvider:
    """
    Convenience function to create an embedding provider
    
    Args:
        provider_type: Type of provider ('openai', 'huggingface')
        **kwargs: Provider-specific configuration
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider
    """
    return EmbeddingProviderFactory.create_provider(provider_type, **kwargs)
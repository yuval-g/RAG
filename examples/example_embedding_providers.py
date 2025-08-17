"""
Example usage of embedding providers in the RAG system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine.core.embedding_providers import (
    create_embedding_provider,
    EmbeddingProviderFactory,
    EmbeddingProviderError
)
from src.rag_engine.core.config import PipelineConfig, ConfigurationManager


def example_openai_provider():
    """Example using OpenAI embedding provider"""
    print("=== OpenAI Embedding Provider Example ===")
    
    try:
        # Create OpenAI provider
        provider = create_embedding_provider(
            "openai",
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            model="text-embedding-ada-002"
        )
        
        # Example documents
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science."
        ]
        
        print(f"Provider info: {provider.get_model_info()}")
        print(f"Embedding dimension: {provider.get_embedding_dimension()}")
        
        # Generate embeddings for documents
        print("\nGenerating document embeddings...")
        doc_embeddings = provider.embed_documents(documents)
        print(f"Generated {len(doc_embeddings)} document embeddings")
        print(f"First embedding shape: {len(doc_embeddings[0])}")
        
        # Generate embedding for a query
        query = "What is machine learning?"
        print(f"\nGenerating query embedding for: '{query}'")
        query_embedding = provider.embed_query(query)
        print(f"Query embedding shape: {len(query_embedding)}")
        
        # Calculate similarity (simple dot product)
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = sum(a * b for a, b in zip(query_embedding, doc_emb))
            similarities.append((i, similarity))
            print(f"Similarity with document {i}: {similarity:.4f}")
        
        # Find most similar document
        most_similar = max(similarities, key=lambda x: x[1])
        print(f"\nMost similar document: {most_similar[0]} (similarity: {most_similar[1]:.4f})")
        print(f"Document text: {documents[most_similar[0]]}")
        
    except EmbeddingProviderError as e:
        print(f"Error with OpenAI provider: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_huggingface_provider():
    """Example using HuggingFace embedding provider"""
    print("\n=== HuggingFace Embedding Provider Example ===")
    
    try:
        # Create HuggingFace provider
        provider = create_embedding_provider(
            "huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",  # Use "cuda" if GPU is available
            normalize_embeddings=True
        )
        
        # Example documents
        documents = [
            "The capital of France is Paris.",
            "Python is used for web development and data science.",
            "The weather today is sunny and warm."
        ]
        
        print(f"Provider info: {provider.get_model_info()}")
        print(f"Embedding dimension: {provider.get_embedding_dimension()}")
        
        # Generate embeddings for documents
        print("\nGenerating document embeddings...")
        doc_embeddings = provider.embed_documents(documents)
        print(f"Generated {len(doc_embeddings)} document embeddings")
        print(f"First embedding shape: {len(doc_embeddings[0])}")
        
        # Generate embedding for a query
        query = "What programming language is good for data science?"
        print(f"\nGenerating query embedding for: '{query}'")
        query_embedding = provider.embed_query(query)
        print(f"Query embedding shape: {len(query_embedding)}")
        
        # Calculate cosine similarity (since embeddings are normalized)
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = sum(a * b for a, b in zip(query_embedding, doc_emb))
            similarities.append((i, similarity))
            print(f"Similarity with document {i}: {similarity:.4f}")
        
        # Find most similar document
        most_similar = max(similarities, key=lambda x: x[1])
        print(f"\nMost similar document: {most_similar[0]} (similarity: {most_similar[1]:.4f})")
        print(f"Document text: {documents[most_similar[0]]}")
        
    except EmbeddingProviderError as e:
        print(f"Error with HuggingFace provider: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_provider_factory():
    """Example using the provider factory"""
    print("\n=== Provider Factory Example ===")
    
    # List available providers
    providers = EmbeddingProviderFactory.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Create providers using factory
    for provider_name in providers:
        try:
            print(f"\nTesting {provider_name} provider...")
            
            if provider_name == "openai":
                provider = EmbeddingProviderFactory.create_provider(
                    provider_name,
                    api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                    model="text-embedding-ada-002"
                )
            elif provider_name == "huggingface":
                provider = EmbeddingProviderFactory.create_provider(
                    provider_name,
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            print(f"Created {provider_name} provider successfully")
            print(f"Model info: {provider.get_model_info()}")
            
        except EmbeddingProviderError as e:
            print(f"Could not create {provider_name} provider: {e}")
        except Exception as e:
            print(f"Unexpected error with {provider_name}: {e}")


def example_configuration_integration():
    """Example integrating with configuration system"""
    print("\n=== Configuration Integration Example ===")
    
    # Create configuration
    config = PipelineConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        embedding_dimensions=None
    )
    
    print(f"Configuration: {config.embedding_provider} - {config.embedding_model}")
    
    try:
        # Create provider based on configuration
        provider = create_embedding_provider(
            config.embedding_provider,
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model=config.embedding_model,
            dimensions=config.embedding_dimensions
        )
        
        print(f"Created provider from config: {provider.get_model_info()}")
        
    except EmbeddingProviderError as e:
        print(f"Error creating provider from config: {e}")


def example_custom_provider():
    """Example creating and registering a custom provider"""
    print("\n=== Custom Provider Example ===")
    
    from src.rag_engine.core.interfaces import BaseEmbeddingProvider
    import random
    
    class MockEmbeddingProvider(BaseEmbeddingProvider):
        """Mock embedding provider for testing"""
        
        def __init__(self, dimension=384):
            self.dimension = dimension
        
        def embed_documents(self, texts):
            """Generate random embeddings for documents"""
            return [[random.random() for _ in range(self.dimension)] for _ in texts]
        
        def embed_query(self, text):
            """Generate random embedding for query"""
            return [random.random() for _ in range(self.dimension)]
        
        def get_embedding_dimension(self):
            """Get embedding dimension"""
            return self.dimension
        
        def get_model_info(self):
            """Get model information"""
            return {
                "provider": "mock",
                "model": "random-embeddings",
                "dimensions": self.dimension
            }
    
    # Register custom provider
    EmbeddingProviderFactory.register_provider("mock", MockEmbeddingProvider)
    
    # Use custom provider
    provider = create_embedding_provider("mock", dimension=256)
    
    print(f"Custom provider info: {provider.get_model_info()}")
    
    # Test custom provider
    documents = ["Test document 1", "Test document 2"]
    embeddings = provider.embed_documents(documents)
    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    
    query_embedding = provider.embed_query("Test query")
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Clean up
    del EmbeddingProviderFactory._providers["mock"]


if __name__ == "__main__":
    print("Embedding Providers Example")
    print("=" * 50)
    
    # Run examples
    example_provider_factory()
    example_configuration_integration()
    example_custom_provider()
    
    # Only run provider examples if dependencies are available
    try:
        import openai
        example_openai_provider()
    except ImportError:
        print("\nSkipping OpenAI example (openai package not installed)")
    
    try:
        import sentence_transformers
        example_huggingface_provider()
    except ImportError:
        print("\nSkipping HuggingFace example (sentence-transformers package not installed)")
    
    print("\n" + "=" * 50)
    print("Example completed!")
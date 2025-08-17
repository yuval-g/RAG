"""
Example demonstrating multi-backend vector store support
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.core.config import PipelineConfig
from rag_engine.core.vector_store_providers import create_vector_store_manager
from rag_engine.core.models import Document


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing"""
    return [
        Document(
            content="The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.",
            metadata={"source": "pangram.txt", "type": "text", "category": "language"},
            doc_id="doc1"
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"source": "ml_intro.txt", "type": "text", "category": "technology"},
            doc_id="doc2"
        ),
        Document(
            content="Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
            metadata={"source": "vector_db.txt", "type": "text", "category": "database"},
            doc_id="doc3"
        ),
        Document(
            content="Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
            metadata={"source": "nlp_overview.txt", "type": "text", "category": "technology"},
            doc_id="doc4"
        )
    ]


def demonstrate_chroma_provider():
    """Demonstrate Chroma vector store provider"""
    print("\n=== Chroma Provider Demo ===")
    
    # Configure for Chroma
    config = PipelineConfig(
        vector_store="chroma",
        embedding_provider="google",
        google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
        vector_store_config={
            "collection_name": "demo_collection",
            "persist_directory": "/tmp/chroma_demo"
        }
    )
    
    # Create manager and get Chroma provider
    manager = create_vector_store_manager(config)
    
    try:
        provider = manager.get_provider("chroma")
        print(f"✓ Initialized Chroma provider")
        
        # Get store info
        info = provider.get_store_info()
        print(f"✓ Store info: {info['provider']} with {info['document_count']} documents")
        
        # Create sample documents
        documents = create_sample_documents()
        
        # Note: In a real scenario, you would add documents like this:
        # result = provider.add_documents(documents)
        # print(f"✓ Added {len(documents)} documents: {result}")
        
        print("✓ Chroma provider demo completed (mocked operations)")
        
    except Exception as e:
        print(f"✗ Chroma demo failed: {str(e)}")


def demonstrate_pinecone_provider():
    """Demonstrate Pinecone vector store provider"""
    print("\n=== Pinecone Provider Demo ===")
    
    # Configure for Pinecone
    config = PipelineConfig(
        vector_store="pinecone",
        embedding_provider="google",
        google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
        vector_store_config={
            "api_key": os.getenv("PINECONE_API_KEY", "demo-key"),
            "index_name": "demo-index",
            "environment": "us-east-1-aws",
            "dimension": 768
        }
    )
    
    # Create manager and get Pinecone provider
    manager = create_vector_store_manager(config)
    
    try:
        provider = manager.get_provider("pinecone")
        print(f"✓ Initialized Pinecone provider")
        
        # Get store info
        info = provider.get_store_info()
        print(f"✓ Store info: {info['provider']} with index '{info['index_name']}'")
        
        print("✓ Pinecone provider demo completed (mocked operations)")
        
    except Exception as e:
        print(f"✗ Pinecone demo failed: {str(e)}")


def demonstrate_weaviate_provider():
    """Demonstrate Weaviate vector store provider"""
    print("\n=== Weaviate Provider Demo ===")
    
    # Configure for Weaviate
    config = PipelineConfig(
        vector_store="weaviate",
        embedding_provider="google",
        google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
        vector_store_config={
            "url": "http://localhost:8080",
            "class_name": "DemoDocument"
        }
    )
    
    # Create manager and get Weaviate provider
    manager = create_vector_store_manager(config)
    
    try:
        provider = manager.get_provider("weaviate")
        print(f"✓ Initialized Weaviate provider")
        
        # Get store info
        info = provider.get_store_info()
        print(f"✓ Store info: {info['provider']} with class '{info['class_name']}'")
        
        print("✓ Weaviate provider demo completed (mocked operations)")
        
    except Exception as e:
        print(f"✗ Weaviate demo failed: {str(e)}")


def demonstrate_provider_switching():
    """Demonstrate switching between different vector store providers"""
    print("\n=== Provider Switching Demo ===")
    
    # Base configuration
    config = PipelineConfig(
        vector_store="chroma",
        embedding_provider="google",
        google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key")
    )
    
    manager = create_vector_store_manager(config)
    
    # List available providers
    providers = manager.list_providers()
    print(f"✓ Available providers: {providers}")
    
    # Demonstrate switching between providers
    for provider_name in ["chroma", "pinecone", "weaviate"]:
        try:
            if provider_name == "pinecone":
                provider = manager.switch_provider(provider_name, 
                                                 api_key="demo-key", 
                                                 index_name="demo-index")
            elif provider_name == "weaviate":
                provider = manager.switch_provider(provider_name, 
                                                 url="http://localhost:8080")
            else:
                provider = manager.switch_provider(provider_name, 
                                                 collection_name="demo")
            
            info = provider.get_store_info()
            print(f"✓ Switched to {provider_name}: {info['provider']}")
            
        except Exception as e:
            print(f"✗ Failed to switch to {provider_name}: {str(e)}")


def demonstrate_configuration_management():
    """Demonstrate different configuration approaches"""
    print("\n=== Configuration Management Demo ===")
    
    # Method 1: Direct configuration
    config1 = PipelineConfig(
        vector_store="chroma",
        vector_store_config={"collection_name": "direct_config"}
    )
    print("✓ Created direct configuration")
    
    # Method 2: Environment-based configuration
    os.environ["RAG_VECTOR_STORE"] = "pinecone"
    os.environ["PINECONE_API_KEY"] = "env-key"
    
    from rag_engine.core.config import ConfigurationManager
    config_manager = ConfigurationManager()
    config2 = config_manager.load_config()
    print(f"✓ Loaded environment configuration: vector_store={config2.vector_store}")
    
    # Method 3: Provider-specific configurations
    configs = {
        "chroma": {
            "collection_name": "chroma_collection",
            "persist_directory": "/tmp/chroma"
        },
        "pinecone": {
            "api_key": "pinecone-key",
            "index_name": "pinecone-index",
            "environment": "us-east-1-aws"
        },
        "weaviate": {
            "url": "http://localhost:8080",
            "class_name": "WeaviateDocument",
            "api_key": "weaviate-key"
        }
    }
    
    for provider_name, provider_config in configs.items():
        config = PipelineConfig(
            vector_store=provider_name,
            vector_store_config=provider_config
        )
        print(f"✓ Created {provider_name} configuration with {len(provider_config)} settings")


def main():
    """Run all demonstrations"""
    print("Vector Store Providers Demo")
    print("=" * 50)
    
    # Note: These demos use mocked operations since they require actual API keys
    # and running services. In a real environment, you would:
    # 1. Set up proper API keys in environment variables
    # 2. Ensure vector store services are running (for Weaviate)
    # 3. Have valid Pinecone projects set up
    
    demonstrate_chroma_provider()
    demonstrate_pinecone_provider()
    demonstrate_weaviate_provider()
    demonstrate_provider_switching()
    demonstrate_configuration_management()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the implementations for full functionality.")
    print("\nTo use in production:")
    print("1. Set GOOGLE_API_KEY environment variable")
    print("2. Set PINECONE_API_KEY for Pinecone usage")
    print("3. Start Weaviate service for Weaviate usage")
    print("4. Configure vector_store_config in PipelineConfig")


if __name__ == "__main__":
    main()
"""
Unit tests for vector store providers
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.rag_engine.core.vector_store_providers import (
    VectorStoreProvider, 
    VectorStoreManager,
    create_vector_store_manager
)
from src.rag_engine.core.providers import (
    ChromaProvider,
    PineconeProvider, 
    WeaviateProvider
)
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig
from src.rag_engine.core.exceptions import VectorStoreError


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing"""
    return PipelineConfig(
        vector_store="chroma",
        embedding_provider="google",
        google_api_key="test-key",
        vector_store_config={
            "collection_name": "test_collection",
            "persist_directory": "/tmp/test_chroma"
        }
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            content="This is the first test document",
            metadata={"source": "test1.txt", "type": "text"},
            doc_id="doc1"
        ),
        Document(
            content="This is the second test document",
            metadata={"source": "test2.txt", "type": "text"},
            doc_id="doc2"
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing"""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ]


class TestVectorStoreManager:
    """Test cases for VectorStoreManager"""
    
    def test_manager_initialization(self, sample_config):
        """Test manager initialization with default providers"""
        manager = VectorStoreManager(sample_config)
        
        assert manager.config == sample_config
        assert len(manager.list_providers()) >= 3  # At least chroma, pinecone, weaviate
        assert "chroma" in manager.list_providers()
        assert "pinecone" in manager.list_providers()
        assert "weaviate" in manager.list_providers()
    
    def test_register_custom_provider(self, sample_config):
        """Test registering a custom provider"""
        manager = VectorStoreManager(sample_config)
        
        class CustomProvider(VectorStoreProvider):
            def initialize(self, **kwargs): pass
            def add_documents(self, documents, embeddings): return True
            def similarity_search(self, query_embedding, k=5): return []
            def similarity_search_with_scores(self, query_embedding, k=5): return []
            def delete_documents(self, doc_ids): return True
            def get_document_count(self): return 0
            def clear_store(self): return True
            def get_store_info(self): return {}
        
        manager.register_provider("custom", CustomProvider)
        assert "custom" in manager.list_providers()
    
    def test_register_invalid_provider(self, sample_config):
        """Test registering an invalid provider raises error"""
        manager = VectorStoreManager(sample_config)
        
        class InvalidProvider:
            pass
        
        with pytest.raises(ValueError, match="Provider must inherit from VectorStoreProvider"):
            manager.register_provider("invalid", InvalidProvider)
    
    def test_get_unknown_provider(self, sample_config):
        """Test getting an unknown provider raises error"""
        manager = VectorStoreManager(sample_config)
        
        with pytest.raises(ValueError, match="Unknown vector store provider: unknown"):
            manager.get_provider("unknown")
    
    def test_factory_function(self, sample_config):
        """Test the factory function creates a manager correctly"""
        manager = create_vector_store_manager(sample_config)
        
        assert isinstance(manager, VectorStoreManager)
        assert manager.config == sample_config


class TestChromaProvider:
    """Test cases for ChromaProvider"""
    
    @patch('src.rag_engine.core.providers.chroma_provider.Chroma')
    @patch('src.rag_engine.core.providers.chroma_provider.GoogleGenerativeAIEmbeddings')
    def test_chroma_initialization(self, mock_embeddings, mock_chroma, sample_config):
        """Test Chroma provider initialization"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        provider = ChromaProvider(sample_config)
        provider.initialize(collection_name="test", persist_directory="/tmp/test")
        
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once_with(
            collection_name="test",
            embedding_function=mock_embeddings_instance,
            persist_directory="/tmp/test"
        )
        assert provider._store == mock_chroma_instance
    
    @patch('src.rag_engine.core.providers.chroma_provider.Chroma')
    @patch('src.rag_engine.core.providers.chroma_provider.GoogleGenerativeAIEmbeddings')
    def test_chroma_add_documents(self, mock_embeddings, mock_chroma, sample_config, sample_documents):
        """Test adding documents to Chroma"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        provider = ChromaProvider(sample_config)
        provider.initialize()
        
        result = provider.add_documents(sample_documents)
        
        assert result is True
        mock_chroma_instance.add_documents.assert_called_once()
        
        # Check that documents were converted correctly
        call_args = mock_chroma_instance.add_documents.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].page_content == "This is the first test document"
        assert call_args[0].metadata['doc_id'] == "doc1"
    
    @patch('src.rag_engine.core.providers.chroma_provider.Chroma')
    @patch('src.rag_engine.core.providers.chroma_provider.GoogleGenerativeAIEmbeddings')
    def test_chroma_similarity_search(self, mock_embeddings, mock_chroma, sample_config):
        """Test similarity search in Chroma"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock search results
        mock_result = Mock()
        mock_result.page_content = "Test content"
        mock_result.metadata = {"doc_id": "test1", "source": "test.txt"}
        
        mock_chroma_instance = Mock()
        mock_chroma_instance.similarity_search_by_vector.return_value = [mock_result]
        mock_chroma.return_value = mock_chroma_instance
        
        provider = ChromaProvider(sample_config)
        provider.initialize()
        
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = provider.similarity_search(query_embedding, k=5)
        
        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].doc_id == "test1"
        mock_chroma_instance.similarity_search_by_vector.assert_called_once_with(query_embedding, k=5)
    
    @patch('src.rag_engine.core.providers.chroma_provider.Chroma')
    @patch('src.rag_engine.core.providers.chroma_provider.GoogleGenerativeAIEmbeddings')
    def test_chroma_get_document_count(self, mock_embeddings, mock_chroma, sample_config):
        """Test getting document count from Chroma"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        
        mock_chroma_instance = Mock()
        mock_chroma_instance._collection = mock_collection
        mock_chroma.return_value = mock_chroma_instance
        
        provider = ChromaProvider(sample_config)
        provider.initialize()
        
        count = provider.get_document_count()
        assert count == 42
    
    def test_chroma_uninitialized_error(self, sample_config, sample_documents):
        """Test that operations on uninitialized provider raise errors"""
        provider = ChromaProvider(sample_config)
        
        with pytest.raises(VectorStoreError, match="Vector store not initialized"):
            provider.add_documents(sample_documents)
        
        with pytest.raises(VectorStoreError, match="Vector store not initialized"):
            provider.similarity_search([0.1, 0.2, 0.3])


class TestPineconeProvider:
    """Test cases for PineconeProvider"""
    
    @patch('src.rag_engine.core.providers.pinecone_provider.Pinecone')
    @patch('src.rag_engine.core.providers.pinecone_provider.PineconeVectorStore')
    @patch('src.rag_engine.core.providers.pinecone_provider.GoogleGenerativeAIEmbeddings')
    def test_pinecone_initialization(self, mock_embeddings, mock_vector_store, mock_pinecone, sample_config):
        """Test Pinecone provider initialization"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_pinecone_client = Mock()
        mock_pinecone_client.list_indexes.return_value = [Mock(name="existing_index")]
        mock_pinecone.return_value = mock_pinecone_client
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        provider = PineconeProvider(sample_config)
        provider.initialize(api_key="test-key", index_name="test-index")
        
        mock_pinecone.assert_called_once_with(api_key="test-key")
        mock_vector_store.assert_called_once()
        assert provider._store == mock_vector_store_instance
    
    @patch('src.rag_engine.core.providers.pinecone_provider.Pinecone')
    @patch('src.rag_engine.core.providers.pinecone_provider.PineconeVectorStore')
    @patch('src.rag_engine.core.providers.pinecone_provider.GoogleGenerativeAIEmbeddings')
    def test_pinecone_create_index(self, mock_embeddings, mock_vector_store, mock_pinecone, sample_config):
        """Test Pinecone index creation when it doesn't exist"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_pinecone_client = Mock()
        mock_pinecone_client.list_indexes.return_value = []  # No existing indexes
        mock_pinecone.return_value = mock_pinecone_client
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        provider = PineconeProvider(sample_config)
        provider.initialize(api_key="test-key", index_name="new-index")
        
        mock_pinecone_client.create_index.assert_called_once()
        create_call = mock_pinecone_client.create_index.call_args
        assert create_call[1]['name'] == "new-index"
        assert create_call[1]['dimension'] == 768  # Google embeddings dimension
    
    def test_pinecone_missing_api_key(self, sample_config):
        """Test that missing API key raises error"""
        provider = PineconeProvider(sample_config)
        
        with pytest.raises(VectorStoreError, match="Pinecone API key is required"):
            provider.initialize()


class TestWeaviateProvider:
    """Test cases for WeaviateProvider"""
    
    @patch('src.rag_engine.core.providers.weaviate_provider.weaviate.connect_to_local')
    @patch('src.rag_engine.core.providers.weaviate_provider.WeaviateVectorStore')
    @patch('src.rag_engine.core.providers.weaviate_provider.GoogleGenerativeAIEmbeddings')
    def test_weaviate_initialization(self, mock_embeddings, mock_vector_store, mock_connect, sample_config):
        """Test Weaviate provider initialization"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_weaviate_client = Mock()
        mock_weaviate_client.collections.exists.return_value = True
        mock_connect.return_value = mock_weaviate_client
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        provider = WeaviateProvider(sample_config)
        provider.initialize(url="http://localhost:8080", class_name="TestClass")
        
        mock_connect.assert_called_once_with(host="localhost", port=8080)
        mock_vector_store.assert_called_once()
        assert provider._store == mock_vector_store_instance
    
    @patch('src.rag_engine.core.providers.weaviate_provider.weaviate.connect_to_local')
    @patch('src.rag_engine.core.providers.weaviate_provider.WeaviateVectorStore')
    @patch('src.rag_engine.core.providers.weaviate_provider.GoogleGenerativeAIEmbeddings')
    def test_weaviate_create_class(self, mock_embeddings, mock_vector_store, mock_connect, sample_config):
        """Test Weaviate collection creation when it doesn't exist"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_weaviate_client = Mock()
        mock_weaviate_client.collections.exists.return_value = False  # Collection doesn't exist
        mock_connect.return_value = mock_weaviate_client
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        provider = WeaviateProvider(sample_config)
        provider.initialize(class_name="NewClass")
        
        mock_weaviate_client.collections.create.assert_called_once()
        create_call = mock_weaviate_client.collections.create.call_args
        assert create_call[1]['name'] == "NewClass"
    
    @patch('src.rag_engine.core.providers.weaviate_provider.weaviate.connect_to_local')
    @patch('src.rag_engine.core.providers.weaviate_provider.WeaviateVectorStore')
    @patch('src.rag_engine.core.providers.weaviate_provider.GoogleGenerativeAIEmbeddings')
    def test_weaviate_get_document_count(self, mock_embeddings, mock_vector_store, mock_connect, sample_config):
        """Test getting document count from Weaviate"""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock collection and aggregate result
        mock_collection = Mock()
        mock_aggregate_result = Mock()
        mock_aggregate_result.total_count = 25
        mock_collection.aggregate.over_all.return_value = mock_aggregate_result
        
        mock_weaviate_client = Mock()
        mock_weaviate_client.collections.exists.return_value = True
        mock_weaviate_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_weaviate_client
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        provider = WeaviateProvider(sample_config)
        provider.initialize()
        
        count = provider.get_document_count()
        assert count == 25


class TestVectorStoreProviderIntegration:
    """Integration tests for vector store providers"""
    
    def test_provider_switching(self, sample_config):
        """Test switching between different providers"""
        manager = VectorStoreManager(sample_config)
        
        # Mock the providers to avoid actual initialization
        with patch.object(ChromaProvider, 'initialize'), \
             patch.object(PineconeProvider, 'initialize'), \
             patch.object(WeaviateProvider, 'initialize'):
            
            # Test getting chroma provider
            chroma_provider = manager.get_provider("chroma")
            assert isinstance(chroma_provider, ChromaProvider)
            
            # Test switching to pinecone
            pinecone_provider = manager.switch_provider("pinecone", api_key="test")
            assert isinstance(pinecone_provider, PineconeProvider)
            
            # Test switching to weaviate
            weaviate_provider = manager.switch_provider("weaviate", url="http://localhost:8080")
            assert isinstance(weaviate_provider, WeaviateProvider)
    
    def test_provider_store_info(self, sample_config):
        """Test getting store information from providers"""
        with patch.object(ChromaProvider, 'initialize'):
            provider = ChromaProvider(sample_config)
            provider.initialize()
            
            info = provider.get_store_info()
            assert info['provider'] == 'chroma'
            assert 'embedding_model' in info
            assert 'embedding_provider' in info
    
    def test_native_store_access(self, sample_config):
        """Test accessing native store instances"""
        with patch.object(ChromaProvider, 'initialize'):
            provider = ChromaProvider(sample_config)
            provider.initialize()
            
            # Mock the native store
            mock_store = Mock()
            provider._store = mock_store
            
            native_store = provider.get_native_store()
            assert native_store == mock_store


if __name__ == "__main__":
    pytest.main([__file__])
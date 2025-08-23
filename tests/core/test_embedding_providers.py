"""
Unit tests for embedding providers
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.rag_engine.core.embedding_providers import (
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    EmbeddingProviderFactory,
    EmbeddingProviderError,
    create_embedding_provider
)


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAI embedding provider"""
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_initialization_with_api_key(self, mock_openai):
        """Test initialization with provided API key"""
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-ada-002")
        
        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-ada-002"
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_initialization_with_env_key(self, mock_openai):
        """Test initialization with environment variable API key"""
        provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
        
        mock_openai.assert_called_once_with(api_key="env-key")
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_initialization_no_api_key(self, mock_openai):
        """Test initialization fails without API key"""
        with pytest.raises(EmbeddingProviderError, match="OpenAI API key not provided"):
            OpenAIEmbeddingProvider()
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_embed_documents_success(self, mock_openai):
        """Test successful document embedding"""
        # Mock OpenAI client response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        texts = ["Hello world", "Test document"]
        
        embeddings = provider.embed_documents(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Verify API call
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-ada-002"
        )
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_embed_documents_with_dimensions(self, mock_openai):
        """Test document embedding with dimensions parameter"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=2)
        embeddings = provider.embed_documents(["test"])
        
        mock_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="text-embedding-ada-002",
            dimensions=2
        )
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_embed_documents_empty_list(self, mock_openai):
        """Test embedding empty document list"""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        embeddings = provider.embed_documents([])
        
        assert embeddings == []
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_embed_query_success(self, mock_openai):
        """Test successful query embedding"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        embedding = provider.embed_query("test query")
        
        assert embedding == [0.1, 0.2, 0.3]
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_embed_query_empty_text(self, mock_openai):
        """Test embedding empty query text"""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        with pytest.raises(EmbeddingProviderError, match="Query text cannot be empty"):
            provider.embed_query("")
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_get_embedding_dimension_cached(self, mock_openai):
        """Test getting embedding dimension when cached"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider.embed_documents(["test"])  # This caches the dimension
        
        dimension = provider.get_embedding_dimension()
        assert dimension == 3
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_get_embedding_dimension_default(self, mock_openai):
        """Test getting embedding dimension with default value"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-ada-002")
        dimension = provider.get_embedding_dimension()
        
        assert dimension == 1536  # Default for ada-002
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_get_model_info(self, mock_openai):
        """Test getting model information"""
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-ada-002")
        provider._embedding_dimension = 1536
        
        info = provider.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "text-embedding-ada-002"
        assert info["dimensions"] == 1536
        assert info["max_input_tokens"] == 8191
    
    @pytest.mark.skip(reason="WIP - OpenAI embedding provider resilience patterns need refinement")
    @pytest.mark.wip
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    @patch('src.rag_engine.core.resilience.resilient_call')
    def test_embed_documents_api_error(self, mock_resilient_call, mock_openai):
        """Test handling API errors during embedding"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Configure resilient_call to re-raise the exception after retries
        mock_resilient_call.side_effect = EmbeddingProviderError("Failed to generate document embeddings after all resilience attempts: API Error")
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        with pytest.raises(EmbeddingProviderError, match="Failed to generate document embeddings"):
            provider.embed_documents(["test"])


class TestHuggingFaceEmbeddingProvider:
    """Test cases for HuggingFace embedding provider"""
    
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_initialization_default(self, mock_cuda_available, mock_sentence_transformer):
        """Test initialization with default parameters"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.device == "cpu"
        assert provider.normalize_embeddings is True
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_initialization_cuda_available(self, mock_cuda_available, mock_sentence_transformer):
        """Test initialization when CUDA is available"""
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        
        assert provider.device == "cuda"
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"
        )
    
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_initialization_custom_model(self, mock_cuda_available, mock_sentence_transformer):
        """Test initialization with custom model"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider(
            model_name="custom-model",
            device="cpu",
            normalize_embeddings=False
        )
        
        assert provider.model_name == "custom-model"
        assert provider.device == "cpu"
        assert provider.normalize_embeddings is False
        mock_sentence_transformer.assert_called_once_with(
            "custom-model",
            device="cpu"
        )
    
    def test_initialization_missing_dependency(self):
        """Test initialization fails when sentence-transformers is not installed"""
        with patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer', side_effect=ImportError):
            with pytest.raises(EmbeddingProviderError, match="sentence-transformers package not installed"):
                HuggingFaceEmbeddingProvider()
    
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_embed_documents_success(self, mock_cuda_available, mock_sentence_transformer):
        """Test successful document embedding"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Mock embeddings as numpy arrays
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        texts = ["Hello world", "Test document"]
        
        embeddings = provider.embed_documents(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        mock_model.encode.assert_called_once_with(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_embed_documents_empty_list(self, mock_cuda_available, mock_sentence_transformer):
        """Test embedding empty document list"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        embeddings = provider.embed_documents([])
        
        assert embeddings == []
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_embed_query_success(self, mock_cuda_available, mock_sentence_transformer):
        """Test successful query embedding"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        embedding = provider.embed_query("test query")
        
        assert embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_embed_query_empty_text(self, mock_cuda_available, mock_sentence_transformer):
        """Test embedding empty query text"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        
        with pytest.raises(EmbeddingProviderError, match="Query text cannot be empty"):
            provider.embed_query("")
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_get_embedding_dimension(self, mock_cuda_available, mock_sentence_transformer):
        """Test getting embedding dimension"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        dimension = provider.get_embedding_dimension()
        
        assert dimension == 384
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_get_model_info(self, mock_cuda_available, mock_sentence_transformer):
        """Test getting model information"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider(
            model_name="custom-model",
            device="cpu",
            normalize_embeddings=False
        )
        
        info = provider.get_model_info()
        
        assert info["provider"] == "huggingface"
        assert info["model"] == "custom-model"
        assert info["dimensions"] == 384
        assert info["device"] == "cpu"
        assert info["normalize_embeddings"] is False
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.HuggingFaceEmbeddingProvider.torch.cuda.is_available')
    def test_embed_documents_model_error(self, mock_cuda_available, mock_sentence_transformer):
        """Test handling model errors during embedding"""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Model Error")
        mock_sentence_transformer.return_value = mock_model
        
        provider = HuggingFaceEmbeddingProvider()
        
        with pytest.raises(EmbeddingProviderError, match="Failed to generate document embeddings"):
            provider.embed_documents(["test"])


class TestEmbeddingProviderFactory:
    """Test cases for embedding provider factory"""
    
    @pytest.mark.skip(reason="WIP - OpenAI embedding provider factory needs refinement")
    @pytest.mark.wip
    @patch('openai.OpenAI')
    def test_create_openai_provider(self, mock_provider):
        """Test creating OpenAI provider"""
        EmbeddingProviderFactory.create_provider("openai", api_key="test-key")
        mock_provider.assert_called_once_with(api_key="test-key")
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider factory needs refinement")
    @pytest.mark.wip
    @patch('sentence_transformers.SentenceTransformer')
    @patch('src.rag_engine.core.embedding_providers.torch.cuda.is_available', return_value=False)
    def test_create_huggingface_provider(self, mock_cuda_available, mock_provider):
        """Test creating HuggingFace provider"""
        EmbeddingProviderFactory.create_provider("huggingface", model_name="test-model", device="cpu")
        mock_provider.assert_called_once_with(model_name="test-model", device="cpu")
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error"""
        with pytest.raises(EmbeddingProviderError, match="Unknown embedding provider: unknown"):
            EmbeddingProviderFactory.create_provider("unknown")
    
    def test_get_available_providers(self):
        """Test getting available providers"""
        providers = EmbeddingProviderFactory.get_available_providers()
        assert "openai" in providers
        assert "huggingface" in providers
    
    def test_register_provider(self):
        """Test registering new provider"""
        class CustomProvider:
            pass
        
        # Should fail for non-BaseEmbeddingProvider class
        with pytest.raises(EmbeddingProviderError, match="Provider class must inherit from BaseEmbeddingProvider"):
            EmbeddingProviderFactory.register_provider("custom", CustomProvider)
    
    @patch('src.rag_engine.core.embedding_providers.EmbeddingProviderFactory.create_provider')
    def test_convenience_function(self, mock_create):
        """Test convenience function for creating providers"""
        create_embedding_provider("openai", api_key="test-key")
        mock_create.assert_called_once_with("openai", api_key="test-key")


class TestIntegration:
    """Integration tests for embedding providers"""
    
    @patch('src.rag_engine.core.embedding_providers.OpenAI')
    def test_openai_provider_workflow(self, mock_openai):
        """Test complete OpenAI provider workflow"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create provider and test workflow
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        # Test document embedding
        docs = ["Document 1", "Document 2"]
        doc_embeddings = provider.embed_documents(docs)
        assert len(doc_embeddings) == 2
        
        # Test query embedding
        query_embedding = provider.embed_query("Test query")
        assert len(query_embedding) == 3
        
        # Test dimension
        dimension = provider.get_embedding_dimension()
        assert dimension == 3
        
        # Test model info
        info = provider.get_model_info()
        assert info["provider"] == "openai"
    
    @pytest.mark.skip(reason="WIP - HuggingFace embedding provider not ready")
    @patch('sentence_transformers.SentenceTransformer')
    @patch('torch.cuda.is_available')
    def test_huggingface_provider_workflow(self, mock_cuda_available, mock_sentence_transformer):
        """Test complete HuggingFace provider workflow"""
        # Mock dependencies
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_sentence_transformer.return_value = mock_model
        
        # Create provider and test workflow
        provider = HuggingFaceEmbeddingProvider()
        
        # Test document embedding
        docs = ["Document 1", "Document 2"]
        doc_embeddings = provider.embed_documents(docs)
        assert len(doc_embeddings) == 2
        
        # Test query embedding (mock single embedding)
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        query_embedding = provider.embed_query("Test query")
        assert len(query_embedding) == 3
        
        # Test dimension
        dimension = provider.get_embedding_dimension()
        assert dimension == 384
        
        # Test model info
        info = provider.get_model_info()
        assert info["provider"] == "huggingface"
"""
Unit tests for BasicIndexer class
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.indexing.basic_indexer import BasicIndexer
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


class TestBasicIndexer:
    """Test suite for BasicIndexer class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return PipelineConfig(
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="models/embedding-001",
            vector_store_config={"persist_directory": None}
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="This is the first test document. It contains some sample text for testing the indexing functionality.",
                metadata={"source": "test1", "type": "sample"},
                doc_id="doc1"
            ),
            Document(
                content="This is the second test document. It has different content to test chunking and embedding processes.",
                metadata={"source": "test2", "type": "sample"},
                doc_id="doc2"
            )
        ]
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings"""
        mock = Mock()
        mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock.embed_query.return_value = [0.1, 0.2, 0.3]
        return mock
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store"""
        mock = Mock()
        mock._collection.count.return_value = 2
        mock.add_documents.return_value = None
        return mock
    
    def test_init(self, config):
        """Test BasicIndexer initialization"""
        with patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings'):
            indexer = BasicIndexer(config)
            
            assert indexer.config == config
            assert indexer.text_splitter._chunk_size == config.chunk_size
            assert indexer.text_splitter._chunk_overlap == config.chunk_overlap
            assert indexer.vectorstore is None
            assert indexer._document_count == 0
    
    def test_init_with_default_embedding_model(self):
        """Test initialization with default OpenAI embedding model gets converted to Google"""
        config = PipelineConfig(embedding_model="text-embedding-ada-002")
        
        with patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings') as mock_embeddings:
            indexer = BasicIndexer(config)
            
            # Should use Google's default embedding model instead of OpenAI's
            mock_embeddings.assert_called_once_with(model="models/embedding-001")
    
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_index_documents_success(self, mock_embeddings_class, mock_chroma_class, 
                                   config, sample_documents, mock_vectorstore):
        """Test successful document indexing"""
        # Setup mocks
        mock_embeddings_class.return_value = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore
        
        indexer = BasicIndexer(config)
        
        # Test indexing
        result = indexer.index_documents(sample_documents)
        
        assert result is True
        assert indexer._document_count == 2
        assert indexer.vectorstore == mock_vectorstore
        mock_chroma_class.from_documents.assert_called_once()
    
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_index_documents_add_to_existing(self, mock_embeddings_class, mock_chroma_class,
                                           config, sample_documents, mock_vectorstore):
        """Test adding documents to existing vector store"""
        # Setup mocks
        mock_embeddings_class.return_value = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore
        
        indexer = BasicIndexer(config)
        
        # First indexing
        indexer.index_documents(sample_documents)
        
        # Second indexing should add to existing store
        result = indexer.index_documents(sample_documents)
        
        assert result is True
        assert indexer._document_count == 4  # 2 + 2
        mock_vectorstore.add_documents.assert_called_once()
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_index_documents_empty_list(self, mock_embeddings_class, config):
        """Test indexing empty document list"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        result = indexer.index_documents([])
        
        assert result is True
        assert indexer._document_count == 0
        assert indexer.vectorstore is None
    
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_index_documents_error_handling(self, mock_embeddings_class, mock_chroma_class,
                                          config, sample_documents):
        """Test error handling during indexing"""
        # Setup mocks to raise exception
        mock_embeddings_class.return_value = Mock()
        mock_chroma_class.from_documents.side_effect = Exception("Test error")
        
        indexer = BasicIndexer(config)
        
        result = indexer.index_documents(sample_documents)
        
        assert result is False
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_get_document_count(self, mock_embeddings_class, config):
        """Test getting document count"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        assert indexer.get_document_count() == 0
        
        indexer._document_count = 5
        assert indexer.get_document_count() == 5
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_clear_index_no_vectorstore(self, mock_embeddings_class, config):
        """Test clearing index when no vector store exists"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        result = indexer.clear_index()
        
        assert result is True
        assert indexer.vectorstore is None
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_clear_index_with_vectorstore(self, mock_embeddings_class, config, mock_vectorstore):
        """Test clearing index with existing vector store"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        indexer.vectorstore = mock_vectorstore
        indexer._document_count = 5
        
        result = indexer.clear_index()
        
        assert result is True
        assert indexer.vectorstore is None
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.basic_indexer.WebBaseLoader')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_load_web_documents_success(self, mock_embeddings_class, mock_loader_class, config):
        """Test successful web document loading"""
        # Setup mocks
        mock_embeddings_class.return_value = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test web content"
        mock_doc.metadata = {"url": "https://example.com"}
        
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        indexer = BasicIndexer(config)
        urls = ["https://example.com"]
        
        result = indexer.load_web_documents(urls)
        
        assert len(result) == 1
        assert result[0].content == "Test web content"
        assert result[0].metadata == {"url": "https://example.com"}
        assert result[0].doc_id == "web_doc_0"
        
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
    
    @patch('src.rag_engine.indexing.basic_indexer.WebBaseLoader')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_load_web_documents_with_custom_bs_kwargs(self, mock_embeddings_class, 
                                                     mock_loader_class, config):
        """Test web document loading with custom BeautifulSoup arguments"""
        mock_embeddings_class.return_value = Mock()
        mock_loader_class.return_value = Mock()
        mock_loader_class.return_value.load.return_value = []
        
        indexer = BasicIndexer(config)
        urls = ["https://example.com"]
        custom_bs_kwargs = {"parse_only": "custom_parser"}
        
        indexer.load_web_documents(urls, bs_kwargs=custom_bs_kwargs)
        
        # Verify WebBaseLoader was called with custom bs_kwargs
        call_args = mock_loader_class.call_args
        assert call_args[1]["bs_kwargs"] == custom_bs_kwargs
    
    @patch('src.rag_engine.indexing.basic_indexer.WebBaseLoader')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_load_web_documents_error_handling(self, mock_embeddings_class, 
                                             mock_loader_class, config):
        """Test error handling in web document loading"""
        mock_embeddings_class.return_value = Mock()
        mock_loader_class.side_effect = Exception("Network error")
        
        indexer = BasicIndexer(config)
        urls = ["https://example.com"]
        
        result = indexer.load_web_documents(urls)
        
        assert result == []
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_get_vectorstore(self, mock_embeddings_class, config, mock_vectorstore):
        """Test getting vector store"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        # Initially None
        assert indexer.get_vectorstore() is None
        
        # After setting
        indexer.vectorstore = mock_vectorstore
        assert indexer.get_vectorstore() == mock_vectorstore
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_get_chunk_count_no_vectorstore(self, mock_embeddings_class, config):
        """Test getting chunk count when no vector store exists"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        result = indexer.get_chunk_count()
        
        assert result == 0
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_get_chunk_count_with_vectorstore(self, mock_embeddings_class, config, mock_vectorstore):
        """Test getting chunk count with vector store"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        indexer.vectorstore = mock_vectorstore
        
        result = indexer.get_chunk_count()
        
        assert result == 2  # mock_vectorstore._collection.count.return_value = 2
    
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_get_chunk_count_error_handling(self, mock_embeddings_class, config):
        """Test error handling in get_chunk_count"""
        mock_embeddings_class.return_value = Mock()
        indexer = BasicIndexer(config)
        
        # Create a mock vectorstore that raises an exception
        mock_vectorstore = Mock()
        mock_vectorstore._collection.count.side_effect = Exception("Collection error")
        indexer.vectorstore = mock_vectorstore
        
        result = indexer.get_chunk_count()
        
        assert result == 0
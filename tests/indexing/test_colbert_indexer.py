"""
Unit tests for ColBERTIndexer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.indexing.colbert_indexer import ColBERTIndexer
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def config():
    """Create a test configuration"""
    return PipelineConfig(
        llm_model="gemini-pro",
        embedding_model="models/embedding-001",
        temperature=0.0,
        chunk_size=1000,
        chunk_overlap=200
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            content="This is a document about artificial intelligence and machine learning. "
                   "It covers various topics including neural networks, deep learning, and natural language processing.",
            metadata={"title": "AI Overview", "author": "Test Author"},
            doc_id="doc1"
        ),
        Document(
            content="This document discusses the history of computer science and programming languages. "
                   "It includes information about early computers, programming paradigms, and software development.",
            metadata={"title": "CS History", "author": "Another Author"},
            doc_id="doc2"
        )
    ]


class TestColBERTIndexer:
    """Test cases for ColBERTIndexer"""
    
    def test_initialization(self, config):
        """Test ColBERTIndexer initialization"""
        indexer = ColBERTIndexer(config, index_name="test-index")
        
        # Check that components are initialized
        assert indexer.config == config
        assert indexer.index_name == "test-index"
        assert indexer._document_count == 0
        assert indexer._is_indexed is False
        assert indexer.max_document_length == 180
        assert indexer.split_documents is True
        assert indexer._rag_model is None
    
    def test_initialization_default_index_name(self, config):
        """Test ColBERTIndexer initialization with default index name"""
        indexer = ColBERTIndexer(config)
        
        assert indexer.index_name == "colbert-index"
    
    def test_rag_model_lazy_loading(self, config):
        """Test lazy loading of RAGatouille model"""
        mock_model = Mock()
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        
        # Model should not be loaded initially
        assert indexer._rag_model is None
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            # Accessing rag_model property should trigger loading
            model = indexer.rag_model
            
            assert model == mock_model
            assert indexer._rag_model == mock_model
            mock_rag_model_class.from_pretrained.assert_called_once_with("colbert-ir/colbertv2.0")
    
    def test_rag_model_import_error(self, config):
        """Test handling of missing RAGatouille dependency"""
        indexer = ColBERTIndexer(config)
        
        # Mock the import to raise ImportError
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    raise ImportError("No module named 'ragatouille'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            with pytest.raises(ImportError, match="RAGatouille is required"):
                _ = indexer.rag_model
    
    def test_index_documents_success(self, config, sample_documents):
        """Test successful document indexing"""
        mock_model = Mock()
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config, index_name="test-index")
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            result = indexer.index_documents(sample_documents)
            
            # Assertions
            assert result is True
            assert indexer._document_count == 2
            assert indexer._is_indexed is True
            
            # Check that model.index was called with correct parameters
            mock_model.index.assert_called_once()
            call_args = mock_model.index.call_args
            
            assert call_args[1]['index_name'] == "test-index"
            assert call_args[1]['max_document_length'] == 180
            assert call_args[1]['split_documents'] is True
            assert len(call_args[1]['collection']) == 2
            
            # Check that metadata was included in the text
            collection = call_args[1]['collection']
            assert "Title: AI Overview" in collection[0]
            assert "Author: Test Author" in collection[0]
            assert "Title: CS History" in collection[1]
            assert "Author: Another Author" in collection[1]
    
    def test_index_empty_documents(self, config):
        """Test indexing with empty document list"""
        indexer = ColBERTIndexer(config)
        
        result = indexer.index_documents([])
        
        assert result is True
        assert indexer._document_count == 0
        assert indexer._is_indexed is False
    
    def test_index_documents_error(self, config, sample_documents):
        """Test error handling during document indexing"""
        mock_model = Mock()
        mock_model.index.side_effect = Exception("Indexing failed")
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            result = indexer.index_documents(sample_documents)
            
            assert result is False
            assert indexer._document_count == 0
            assert indexer._is_indexed is False
    
    def test_get_document_count(self, config):
        """Test getting document count"""
        indexer = ColBERTIndexer(config)
        
        assert indexer.get_document_count() == 0
        
        indexer._document_count = 5
        assert indexer.get_document_count() == 5
    
    def test_clear_index(self, config):
        """Test clearing the index"""
        indexer = ColBERTIndexer(config)
        
        # Set some state
        indexer._document_count = 5
        indexer._is_indexed = True
        
        result = indexer.clear_index()
        
        assert result is True
        assert indexer._document_count == 0
        assert indexer._is_indexed is False
    
    def test_search_success(self, config):
        """Test successful search"""
        mock_model = Mock()
        mock_results = [
            {"content": "Result 1", "score": 0.9, "rank": 1},
            {"content": "Result 2", "score": 0.8, "rank": 2}
        ]
        mock_model.search.return_value = mock_results
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        indexer._is_indexed = True
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            results = indexer.search("test query", k=2)
            
            assert results == mock_results
            mock_model.search.assert_called_once_with(query="test query", k=2)
    
    def test_search_not_indexed(self, config):
        """Test search when no documents are indexed"""
        indexer = ColBERTIndexer(config)
        
        results = indexer.search("test query")
        
        assert results == []
    
    def test_search_error(self, config):
        """Test error handling during search"""
        mock_model = Mock()
        mock_model.search.side_effect = Exception("Search failed")
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        indexer._is_indexed = True
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            results = indexer.search("test query")
            
            assert results == []
    
    def test_as_langchain_retriever_success(self, config):
        """Test creating LangChain retriever"""
        mock_model = Mock()
        mock_retriever = Mock()
        mock_model.as_langchain_retriever.return_value = mock_retriever
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        indexer._is_indexed = True
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            retriever = indexer.as_langchain_retriever(k=5)
            
            assert retriever == mock_retriever
            mock_model.as_langchain_retriever.assert_called_once_with(k=5)
    
    def test_as_langchain_retriever_not_indexed(self, config):
        """Test creating LangChain retriever when not indexed"""
        indexer = ColBERTIndexer(config)
        
        retriever = indexer.as_langchain_retriever()
        
        assert retriever is None
    
    def test_as_langchain_retriever_error(self, config):
        """Test error handling when creating LangChain retriever"""
        mock_model = Mock()
        mock_model.as_langchain_retriever.side_effect = Exception("Retriever creation failed")
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        indexer._is_indexed = True
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            retriever = indexer.as_langchain_retriever()
            
            assert retriever is None
    
    def test_get_index_info(self, config):
        """Test getting index information"""
        indexer = ColBERTIndexer(config, index_name="test-index")
        indexer._document_count = 5
        indexer._is_indexed = True
        
        info = indexer.get_index_info()
        
        expected_info = {
            "index_name": "test-index",
            "document_count": 5,
            "is_indexed": True,
            "max_document_length": 180,
            "split_documents": True,
            "model_name": "colbert-ir/colbertv2.0"
        }
        
        assert info == expected_info
    
    def test_set_max_document_length(self, config):
        """Test setting maximum document length"""
        indexer = ColBERTIndexer(config)
        
        indexer.set_max_document_length(256)
        
        assert indexer.max_document_length == 256
    
    def test_set_split_documents(self, config):
        """Test setting split documents flag"""
        indexer = ColBERTIndexer(config)
        
        indexer.set_split_documents(False)
        
        assert indexer.split_documents is False
    
    def test_is_ready_success(self, config):
        """Test is_ready when model loads successfully"""
        mock_model = Mock()
        mock_rag_model_class = Mock()
        mock_rag_model_class.from_pretrained.return_value = mock_model
        
        indexer = ColBERTIndexer(config)
        
        # Mock the import inside the property
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    mock_module = Mock()
                    mock_module.RAGPretrainedModel = mock_rag_model_class
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            assert indexer.is_ready() is True
    
    def test_is_ready_failure(self, config):
        """Test is_ready when model fails to load"""
        indexer = ColBERTIndexer(config)
        
        # Mock the import to raise ImportError
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ragatouille':
                    raise ImportError("No module named 'ragatouille'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            assert indexer.is_ready() is False
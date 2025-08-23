"""
Unit tests for MultiRepresentationIndexer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.indexing.multi_representation_indexer import MultiRepresentationIndexer
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def config():
    """Create a test configuration"""
    return PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        embedding_model="models/embedding-001",
        temperature=0.0,
        chunk_size=1000,
        chunk_overlap=200,
        vector_store_config={"persist_directory": None}
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


class TestMultiRepresentationIndexer:
    """Test cases for MultiRepresentationIndexer"""
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.multi_representation_indexer.Chroma')
    def test_initialization(self, mock_chroma, mock_llm, mock_embeddings, config):
        """Test MultiRepresentationIndexer initialization"""
        indexer = MultiRepresentationIndexer(config)
        
        # Check that components are initialized
        assert indexer.config == config
        assert indexer._document_count == 0
        assert indexer.id_key == "doc_id"
        assert indexer.vectorstore is None
        assert indexer.retriever is None
        
        # Check that Google services are initialized
        mock_embeddings.assert_called_once()
        mock_llm.assert_called_once()
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.multi_representation_indexer.Chroma')
    @patch('src.rag_engine.indexing.multi_representation_indexer.MultiVectorRetriever')
    def test_index_documents_success(self, mock_retriever_class, mock_chroma, mock_llm, mock_embeddings, 
                                   config, sample_documents):
        """Test successful document indexing"""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        indexer = MultiRepresentationIndexer(config)
        
        # Mock the summary chain batch method
        mock_summaries = [
            "Summary of AI document covering neural networks and deep learning.",
            "Summary of CS history document about early computers and programming."
        ]
        indexer.summary_chain = Mock()
        indexer.summary_chain.batch.return_value = mock_summaries
        
        # Test indexing
        result = indexer.index_documents(sample_documents)
        
        # Assertions
        assert result is True
        assert indexer._document_count == 2
        assert indexer.vectorstore is not None
        assert indexer.retriever is not None
        
        # Check that summary chain was called
        indexer.summary_chain.batch.assert_called_once()
        
        # Check that retriever methods were called
        mock_retriever.vectorstore.add_documents.assert_called_once()
        mock_retriever.docstore.mset.assert_called_once()
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_index_empty_documents(self, mock_llm, mock_embeddings, config):
        """Test indexing with empty document list"""
        indexer = MultiRepresentationIndexer(config)
        
        result = indexer.index_documents([])
        
        assert result is True
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_index_documents_error(self, mock_llm, mock_embeddings, config, sample_documents):
        """Test error handling during document indexing"""
        indexer = MultiRepresentationIndexer(config)
        
        # Mock the summary chain to raise an exception
        indexer.summary_chain = Mock()
        indexer.summary_chain.batch.side_effect = Exception("Summarization failed")
        
        result = indexer.index_documents(sample_documents)
        
        assert result is False
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_document_count(self, mock_llm, mock_embeddings, config):
        """Test getting document count"""
        indexer = MultiRepresentationIndexer(config)
        
        assert indexer.get_document_count() == 0
        
        indexer._document_count = 5
        assert indexer.get_document_count() == 5
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_clear_index(self, mock_llm, mock_embeddings, config):
        """Test clearing the index"""
        indexer = MultiRepresentationIndexer(config)
        
        # Set some state
        indexer.vectorstore = Mock()
        indexer.retriever = Mock()
        indexer._document_count = 5
        
        result = indexer.clear_index()
        
        assert result is True
        assert indexer.vectorstore is None
        assert indexer.retriever is None
        assert indexer._document_count == 0
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_retriever(self, mock_llm, mock_embeddings, config):
        """Test getting the retriever"""
        indexer = MultiRepresentationIndexer(config)
        
        assert indexer.get_retriever() is None
        
        mock_retriever = Mock()
        indexer.retriever = mock_retriever
        
        assert indexer.get_retriever() == mock_retriever
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_search_summaries(self, mock_llm, mock_embeddings, config):
        """Test searching summaries"""
        indexer = MultiRepresentationIndexer(config)
        
        # Test with no vectorstore
        result = indexer.search_summaries("test query")
        assert result == []
        
        # Test with vectorstore
        mock_vectorstore = Mock()
        mock_docs = [Mock(), Mock()]
        mock_vectorstore.similarity_search.return_value = mock_docs
        indexer.vectorstore = mock_vectorstore
        
        result = indexer.search_summaries("test query", k=2)
        
        assert result == mock_docs
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=2)
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_full_documents(self, mock_llm, mock_embeddings, config):
        """Test getting full documents via retriever"""
        indexer = MultiRepresentationIndexer(config)
        
        # Test with no retriever
        result = indexer.get_full_documents("test query")
        assert result == []
        
        # Test with retriever
        mock_retriever = Mock()
        mock_docs = [Mock(), Mock()]
        mock_retriever.get_relevant_documents.return_value = mock_docs
        indexer.retriever = mock_retriever
        
        result = indexer.get_full_documents("test query", k=2)
        
        assert result == mock_docs
        mock_retriever.get_relevant_documents.assert_called_once_with("test query", n_results=2)
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_summary_count(self, mock_llm, mock_embeddings, config):
        """Test getting summary count"""
        indexer = MultiRepresentationIndexer(config)
        
        # Test with no vectorstore
        assert indexer.get_summary_count() == 0
        
        # Test with vectorstore
        mock_vectorstore = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_vectorstore._collection = mock_collection
        indexer.vectorstore = mock_vectorstore
        
        assert indexer.get_summary_count() == 5
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_docstore_count(self, mock_llm, mock_embeddings, config):
        """Test getting docstore count"""
        indexer = MultiRepresentationIndexer(config)
        
        assert indexer.get_docstore_count() == 0
        
        indexer._document_count = 3
        assert indexer.get_docstore_count() == 3
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_search_summaries_error_handling(self, mock_llm, mock_embeddings, config):
        """Test error handling in search_summaries"""
        indexer = MultiRepresentationIndexer(config)
        
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.side_effect = Exception("Search failed")
        indexer.vectorstore = mock_vectorstore
        
        result = indexer.search_summaries("test query")
        
        assert result == []
    
    @patch('src.rag_engine.indexing.multi_representation_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.multi_representation_indexer.ChatGoogleGenerativeAI')
    def test_get_full_documents_error_handling(self, mock_llm, mock_embeddings, config):
        """Test error handling in get_full_documents"""
        indexer = MultiRepresentationIndexer(config)
        
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval failed")
        indexer.retriever = mock_retriever
        
        result = indexer.get_full_documents("test query")
        
        assert result == []
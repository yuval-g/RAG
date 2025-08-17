"""
Integration tests for RAGEngine end-to-end functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.core.engine import RAGEngine
from src.rag_engine.core.models import Document, TestCase
from src.rag_engine.core.config import PipelineConfig


class TestRAGEngineIntegration:
    """Integration test suite for RAGEngine"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return PipelineConfig(
            chunk_size=500,
            chunk_overlap=50,
            retrieval_k=3,
            embedding_model="models/embedding-001",
            llm_model="gpt-3.5-turbo",
            temperature=0.0
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
                metadata={"source": "ai_intro", "topic": "artificial_intelligence"},
                doc_id="doc1"
            ),
            Document(
                content="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
                metadata={"source": "ml_intro", "topic": "machine_learning"},
                doc_id="doc2"
            ),
            Document(
                content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP enables machines to understand, interpret, and generate human language in a valuable way.",
                metadata={"source": "nlp_intro", "topic": "natural_language_processing"},
                doc_id="doc3"
            )
        ]
    
    @pytest.fixture
    def test_cases(self):
        """Create test cases for evaluation"""
        return [
            TestCase(
                question="What is Artificial Intelligence?",
                expected_answer="AI is a branch of computer science that aims to create intelligent machines.",
                metadata={"category": "definition"}
            ),
            TestCase(
                question="How does Machine Learning work?",
                expected_answer="Machine Learning uses algorithms to analyze data, identify patterns, and make predictions.",
                metadata={"category": "process"}
            )
        ]
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_rag_engine_initialization(self, mock_chroma, mock_embeddings, mock_llm, config):
        """Test RAGEngine initialization with all components"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        assert engine.config == config
        assert engine._indexer is not None
        assert engine._retriever is not None
        assert engine._generator is not None
        assert not engine.is_ready()  # No documents indexed yet
    
    def test_rag_engine_default_config(self):
        """Test RAGEngine initialization with default configuration"""
        with patch('src.rag_engine.core.config.ConfigurationManager') as mock_config_manager:
            mock_config = PipelineConfig()
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            with patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
                with patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings'):
                    engine = RAGEngine()
                    
                    assert engine.config == mock_config
                    assert engine._indexer is not None
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_add_documents_success(self, mock_chroma, mock_embeddings, mock_llm, 
                                  config, sample_documents):
        """Test successful document addition and indexing"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        result = engine.add_documents(sample_documents)
        
        assert result is True
        assert engine.get_document_count() == 3
        assert engine.is_ready()  # Should be ready after adding documents
        
        # Verify indexer was called
        mock_chroma.from_documents.assert_called_once()
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_add_documents_empty_list(self, mock_embeddings, mock_llm, config):
        """Test adding empty document list"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        engine = RAGEngine(config)
        
        result = engine.add_documents([])
        
        assert result is False
        assert engine.get_document_count() == 0
        assert not engine.is_ready()
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_add_documents_invalid_content(self, mock_embeddings, mock_llm, config):
        """Test adding documents with invalid content"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        invalid_docs = [
            Document(content="", metadata={}, doc_id="empty"),
            Document(content="Valid content", metadata={}, doc_id="valid")
        ]
        
        engine = RAGEngine(config)
        
        result = engine.add_documents(invalid_docs)
        
        assert result is False
        assert engine.get_document_count() == 0
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_query_no_documents(self, mock_chroma, mock_embeddings, mock_llm, config):
        """Test querying when no documents are indexed"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        engine = RAGEngine(config)
        
        response = engine.query("What is AI?")
        
        assert "No documents have been indexed" in response.answer
        assert response.confidence_score == 0.0
        assert len(response.source_documents) == 0
        assert response.metadata["status"] == "no_documents"
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_query_with_documents_success(self, mock_chroma, mock_embeddings, mock_llm,
                                         config, sample_documents):
        """Test successful query processing with documents"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "AI is a branch of computer science that creates intelligent machines."
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        # Mock vector store and retrieval
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        
        # Create mock retrieved documents
        mock_retrieved_docs = [
            Mock(page_content=sample_documents[0].content, metadata=sample_documents[0].metadata)
        ]
        mock_retriever.get_relevant_documents.return_value = mock_retrieved_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        # Add documents first
        engine.add_documents(sample_documents)
        
        # Query the system
        response = engine.query("What is AI?")
        
        assert "AI is a branch of computer science" in response.answer
        assert response.confidence_score > 0.0
        assert len(response.source_documents) == 1
        assert response.processing_time > 0
        assert "retrieved_count" in response.metadata
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_query_no_relevant_documents(self, mock_chroma, mock_embeddings, mock_llm,
                                        config, sample_documents):
        """Test query when no relevant documents are found"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        # Mock vector store that returns no documents
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = []
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        # Add documents first
        engine.add_documents(sample_documents)
        
        # Query the system
        response = engine.query("What is quantum computing?")
        
        assert "couldn't find any relevant information" in response.answer
        assert response.confidence_score == 0.0
        assert len(response.source_documents) == 0
        assert response.metadata["status"] == "no_relevant_docs"
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_query_error_handling(self, mock_chroma, mock_embeddings, mock_llm,
                                 config, sample_documents):
        """Test query error handling"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        # Mock vector store that raises an error during retrieval
        # The VectorRetriever catches the error and returns empty list
        # which is then handled as "no relevant documents"
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval error")
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        # Add documents first
        engine.add_documents(sample_documents)
        
        # Query the system
        response = engine.query("What is AI?")
        
        # The retrieval error is caught and handled as "no relevant documents"
        assert "couldn't find any relevant information" in response.answer
        assert response.confidence_score == 0.0
        assert response.metadata["status"] == "no_relevant_docs"
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_evaluate_basic(self, mock_embeddings, mock_llm, config, test_cases):
        """Test basic evaluation functionality"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        engine = RAGEngine(config)
        
        result = engine.evaluate(test_cases)
        
        assert result.overall_score == 0.0  # Basic implementation returns 0
        assert len(result.recommendations) > 0
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_get_system_info(self, mock_chroma, mock_embeddings, mock_llm,
                            config, sample_documents):
        """Test getting system information"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_vectorstore._collection.count.return_value = 5
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        # Add documents
        engine.add_documents(sample_documents)
        
        info = engine.get_system_info()
        
        assert info["version"] == "0.1.0"
        assert info["config"]["llm_provider"] == config.llm_provider
        assert info["components"]["indexer"] is True
        assert info["components"]["retriever"] is True
        assert info["components"]["generator"] is True
        assert info["stats"]["indexed_documents"] == 3
        assert info["stats"]["retriever_ready"] is True
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_clear_documents(self, mock_chroma, mock_embeddings, mock_llm,
                            config, sample_documents):
        """Test clearing all documents"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        engine = RAGEngine(config)
        
        # Add documents first
        engine.add_documents(sample_documents)
        assert engine.get_document_count() == 3
        assert engine.is_ready()
        
        # Clear documents
        result = engine.clear_documents()
        
        assert result is True
        assert engine.get_document_count() == 0
        assert not engine.is_ready()
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.WebBaseLoader')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_load_web_documents(self, mock_chroma, mock_loader_class, mock_embeddings, 
                               mock_llm, config):
        """Test loading documents from web URLs"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Mock web loader
        mock_doc = Mock()
        mock_doc.page_content = "Web content about AI"
        mock_doc.metadata = {"url": "https://example.com"}
        
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        engine = RAGEngine(config)
        
        urls = ["https://example.com/ai-article"]
        result = engine.load_web_documents(urls)
        
        assert result is True
        assert engine.get_document_count() == 1
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    def test_query_with_custom_k(self, mock_embeddings, mock_llm, config):
        """Test querying with custom k parameter"""
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        engine = RAGEngine(config)
        
        # Query with no documents (should handle gracefully)
        response = engine.query("What is AI?", k=10)
        
        assert "No documents have been indexed" in response.answer
        assert response.metadata["query"] == "What is AI?"
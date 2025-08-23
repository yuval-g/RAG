"""
Unit tests for HyDEProcessor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.hyde import HyDEProcessor
from src.rag_engine.core.models import Document, ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestHyDEProcessor:
    """Test cases for HyDEProcessor"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(content="Task decomposition involves breaking down complex tasks into smaller subtasks", metadata={"id": "1"}),
            Document(content="LLM agents use planning mechanisms to organize their approach to problems", metadata={"id": "2"}),
            Document(content="Hierarchical task planning enables systematic problem solving in AI systems", metadata={"id": "3"}),
            Document(content="Decomposition strategies help manage complexity in large-scale systems", metadata={"id": "4"})
        ]
    
    @pytest.fixture
    def processor(self):
        """Create a HyDEProcessor instance for testing"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            return HyDEProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0,
                document_style="scientific_paper"
            )
    
    def test_init_default(self):
        """Test HyDEProcessor initialization with defaults"""
        with patch('src.rag_engine.query.hyde.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            processor = HyDEProcessor()
            
            assert processor.llm_model == "gemini-2.0-flash-lite"
            assert processor.temperature == 0.0
            assert processor.document_style == "scientific_paper"
            mock_get_llm.assert_called_once()
    
    def test_init_custom(self):
        """Test HyDEProcessor initialization with custom parameters"""
        with patch('src.rag_engine.query.hyde.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            processor = HyDEProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.3,
                document_style="technical_documentation"
            )
            
            assert processor.llm_model == "gemini-2.0-flash-lite"
            assert processor.temperature == 0.3
            assert processor.document_style == "technical_documentation"
            mock_get_llm.assert_called_once()
    
    def test_create_hyde_prompt_scientific_paper(self, processor):
        """Test HyDE prompt creation for scientific paper style"""
        processor.document_style = "scientific_paper"
        template = processor._create_hyde_prompt()
        
        template_str = str(template)
        assert "scientific paper passage" in template_str
        assert "question" in template_str
        assert "Passage:" in template_str
    
    def test_create_hyde_prompt_technical_documentation(self):
        """Test HyDE prompt creation for technical documentation style"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            processor = HyDEProcessor(document_style="technical_documentation")
            template = processor._create_hyde_prompt()
            
            template_str = str(template)
            assert "technical documentation section" in template_str
            assert "Documentation:" in template_str
    
    def test_create_hyde_prompt_tutorial(self):
        """Test HyDE prompt creation for tutorial style"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            processor = HyDEProcessor(document_style="tutorial")
            template = processor._create_hyde_prompt()
            
            template_str = str(template)
            assert "tutorial section" in template_str
            assert "Tutorial:" in template_str
    
    def test_create_hyde_prompt_encyclopedia(self):
        """Test HyDE prompt creation for encyclopedia style"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            processor = HyDEProcessor(document_style="encyclopedia")
            template = processor._create_hyde_prompt()
            
            template_str = str(template)
            assert "encyclopedia entry" in template_str
            assert "Entry:" in template_str
    
    def test_create_hyde_prompt_unknown_style(self):
        """Test HyDE prompt creation for unknown style (should default)"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            processor = HyDEProcessor(document_style="unknown_style")
            template = processor._create_hyde_prompt()
            
            template_str = str(template)
            assert "scientific paper passage" in template_str  # Should default
    
    def test_is_relevant_document_relevant(self, processor):
        """Test relevance detection for relevant document"""
        question = "What is task decomposition for LLM agents?"
        document = "Task decomposition is a fundamental approach in LLM agents that involves breaking down complex problems into smaller, manageable subtasks. This method enables agents to systematically approach complex challenges."
        
        assert processor._is_relevant_document(document, question) == True
    
    def test_is_relevant_document_irrelevant(self, processor):
        """Test relevance detection for irrelevant document"""
        question = "What is task decomposition for LLM agents?"
        document = "The weather today is sunny and warm. Birds are singing in the trees. It's a beautiful day for a picnic in the park."
        
        assert processor._is_relevant_document(document, question) == False
    
    def test_is_relevant_document_empty_question(self, processor):
        """Test relevance detection with empty question"""
        question = ""
        document = "Some document content here."
        
        assert processor._is_relevant_document(document, question) == True  # Should default to True
    
    def test_create_fallback_document_scientific_paper(self, processor):
        """Test fallback document creation for scientific paper style"""
        processor.document_style = "scientific_paper"
        question = "What is task decomposition for LLM agents?"
        
        fallback = processor._create_fallback_document(question)
        
        assert isinstance(fallback, str)
        assert len(fallback) > 50
        assert "paper" in fallback.lower() or "research" in fallback.lower()
        assert "task" in fallback.lower() or "decomposition" in fallback.lower()
    
    def test_create_fallback_document_technical_documentation(self):
        """Test fallback document creation for technical documentation style"""
        with patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI'):
            processor = HyDEProcessor(document_style="technical_documentation")
            question = "What is task decomposition for LLM agents?"
            
            fallback = processor._create_fallback_document(question)
            
            assert isinstance(fallback, str)
            assert len(fallback) > 50
            assert "section" in fallback.lower() or "implementation" in fallback.lower()
    
    def test_create_fallback_document_minimal_question(self, processor):
        """Test fallback document creation with minimal question"""
        question = "What?"
        
        fallback = processor._create_fallback_document(question)
        
        assert isinstance(fallback, str)
        assert len(fallback) > 50
        assert "comprehensive information" in fallback.lower()
    
    def test_validate_hypothetical_document_valid(self, processor):
        """Test validation of valid hypothetical document"""
        question = "What is task decomposition?"
        document = "Task decomposition is a fundamental approach that involves breaking down complex problems into smaller, manageable subtasks. This method enables systematic problem-solving and is widely used in AI systems."
        
        validated = processor._validate_hypothetical_document(document, question)
        
        assert validated == document  # Should remain unchanged
    
    def test_validate_hypothetical_document_empty(self, processor):
        """Test validation with empty document"""
        question = "What is task decomposition?"
        document = ""
        
        validated = processor._validate_hypothetical_document(document, question)
        
        assert isinstance(validated, str)
        assert len(validated) > 20
        assert validated != document
    
    def test_validate_hypothetical_document_too_long(self, processor):
        """Test validation with overly long document"""
        question = "What is task decomposition?"
        document = "A" * 3000  # Very long document
        
        validated = processor._validate_hypothetical_document(document, question)
        
        assert len(validated) < len(document)
        assert len(validated) <= 2000
    
    def test_validate_hypothetical_document_irrelevant(self, processor):
        """Test validation with irrelevant document"""
        question = "What is task decomposition?"
        document = "The weather is nice today. Birds are singing."
        
        validated = processor._validate_hypothetical_document(document, question)
        
        # Should be replaced with fallback
        assert validated != document
        assert len(validated) > len(document)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_generate_hypothetical_document_success(self, mock_llm_class):
        """Test successful hypothetical document generation"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        
        # Mock the chain invoke
        processor.generation_chain = Mock()
        processor.generation_chain.invoke.return_value = "Task decomposition in large language model (LLM) agents refers to the process of breaking down a complex, high-level task into a series of smaller, more manageable sub-tasks. This hierarchical approach is crucial for enabling agents to handle sophisticated goals."
        
        question = "What is task decomposition for LLM agents?"
        result = processor.generate_hypothetical_document(question)
        
        assert isinstance(result, str)
        assert len(result) > 20
        assert "task decomposition" in result.lower()
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_generate_hypothetical_document_failure(self, mock_llm_class):
        """Test hypothetical document generation failure handling"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        processor.generation_chain = Mock()
        processor.generation_chain.invoke.side_effect = Exception("LLM error")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.generate_hypothetical_document(question)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_process_with_retriever_success(self, mock_llm_class, sample_documents):
        """Test successful processing with retriever"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        
        # Mock generate_hypothetical_document
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.return_value = "Hypothetical document about task decomposition and its applications in AI systems."
        
        # Mock retriever function
        def mock_retriever(query):
            return sample_documents[:3]  # Return first 3 documents
        
        question = "What is task decomposition for LLM agents?"
        results = processor.process_with_retriever(question, mock_retriever, top_k=2)
        
        assert len(results) == 2  # Should respect top_k
        assert all(isinstance(doc, Document) for doc in results)
        processor.generate_hypothetical_document.assert_called_once_with(question)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_process_with_retriever_failure(self, mock_llm_class):
        """Test processing with retriever failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.side_effect = Exception("Generation failed")
        
        def mock_retriever(query):
            return []
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_with_retriever(question, mock_retriever)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_process_with_embedding_retriever_success(self, mock_llm_class, sample_documents):
        """Test successful processing with embedding retriever"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        
        # Mock generate_hypothetical_document
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.return_value = "Hypothetical document content"
        
        # Mock embedding and vector search functions
        def mock_embedding_func(text):
            return [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
        
        def mock_vector_search_func(embedding, top_k):
            return sample_documents[:top_k]
        
        question = "What is task decomposition?"
        results = processor.process_with_embedding_retriever(
            question, mock_embedding_func, mock_vector_search_func, top_k=3
        )
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_generate_multiple_hypothetical_documents_success(self, mock_llm_class):
        """Test successful generation of multiple hypothetical documents"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        
        # Mock generate_hypothetical_document to return different documents
        call_count = 0
        def mock_generate(question):
            nonlocal call_count
            call_count += 1
            return f"Hypothetical document {call_count} about {question}"
        
        processor.generate_hypothetical_document = Mock(side_effect=mock_generate)
        
        question = "What is task decomposition?"
        results = processor.generate_multiple_hypothetical_documents(question, num_documents=3)
        
        assert len(results) == 3
        assert all(isinstance(doc, str) for doc in results)
        assert all("Hypothetical document" in doc for doc in results)
        assert processor.generate_hypothetical_document.call_count == 3
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_generate_multiple_hypothetical_documents_failure(self, mock_llm_class):
        """Test multiple document generation failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.side_effect = Exception("Generation failed")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.generate_multiple_hypothetical_documents(question)
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_process_query_success(self, mock_llm_class):
        """Test successful query processing"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        
        # Mock generate_hypothetical_document
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.return_value = "Hypothetical document about task decomposition"
        
        question = "What is task decomposition for LLM agents?"
        result = processor.process_query(question)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == question
        assert len(result.transformed_queries) == 1
        assert result.transformed_queries[0] == "Hypothetical document about task decomposition"
        assert result.strategy_used == "hyde"
        assert "hypothetical_document" in result.metadata
        assert result.metadata["document_style"] == "scientific_paper"
    
    @patch('src.rag_engine.query.hyde.ChatGoogleGenerativeAI')
    def test_process_query_failure(self, mock_llm_class):
        """Test query processing failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = HyDEProcessor()
        processor.generate_hypothetical_document = Mock()
        processor.generate_hypothetical_document.side_effect = Exception("Generation failed")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_query(question)
    
    def test_get_config(self, processor):
        """Test configuration retrieval"""
        config = processor.get_config()
        
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        assert config["temperature"] == 0.0
        assert config["document_style"] == "scientific_paper"
        assert config["strategy"] == "hyde"
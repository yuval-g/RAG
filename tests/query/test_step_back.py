"""
Unit tests for StepBackProcessor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.step_back import StepBackProcessor
from src.rag_engine.core.models import Document, ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestStepBackProcessor:
    """Test cases for StepBackProcessor"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(content="Task decomposition involves breaking down complex tasks", metadata={"id": "1"}),
            Document(content="General AI principles include planning and reasoning", metadata={"id": "2"}),
            Document(content="LLM agents use various approaches for problem solving", metadata={"id": "3"}),
            Document(content="Software engineering practices help in system design", metadata={"id": "4"})
        ]
    
    @pytest.fixture
    def processor(self):
        """Create a StepBackProcessor instance for testing"""
        with patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI'):
            return StepBackProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0
            )
    
    def test_init(self):
        """Test StepBackProcessor initialization"""
        with patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI') as mock_llm:
            processor = StepBackProcessor(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.2
            )
            
            assert processor.llm_model == "gemini-2.0-flash-lite"
            assert processor.temperature == 0.2
            mock_llm.assert_called_once()
    
    def test_create_fallback_step_back(self, processor):
        """Test fallback step-back question creation"""
        original = "What is the specific implementation of task decomposition in GPT-4?"
        fallback = processor._create_fallback_step_back(original)
        
        assert isinstance(fallback, str)
        assert len(fallback) > 0
        assert fallback.endswith('?')
        assert "general principles" in fallback.lower() or "approaches" in fallback.lower()
    
    def test_create_fallback_step_back_minimal_input(self, processor):
        """Test fallback creation with minimal input"""
        original = "What is this?"
        fallback = processor._create_fallback_step_back(original)
        
        assert isinstance(fallback, str)
        assert len(fallback) > 0
        assert fallback.endswith('?')
        assert "fundamental concepts" in fallback.lower() or "general principles" in fallback.lower()
    
    def test_is_too_similar(self, processor):
        """Test similarity detection"""
        question1 = "What is task decomposition in LLM agents?"
        question2 = "What is task decomposition in LLM agents?"  # Identical
        question3 = "What are general AI principles?"  # Different
        
        assert processor._is_too_similar(question1, question2) == True
        assert processor._is_too_similar(question1, question3) == False
    
    def test_validate_step_back_question_valid(self, processor):
        """Test validation of valid step-back question"""
        original = "What is the specific implementation of task decomposition in GPT-4?"
        step_back = "what are the general approaches to task decomposition in AI systems"
        
        validated = processor._validate_step_back_question(step_back, original)
        
        assert validated.endswith('?')
        assert len(validated) > 5
        assert validated != original
    
    def test_validate_step_back_question_empty(self, processor):
        """Test validation with empty step-back question"""
        original = "What is task decomposition?"
        step_back = ""
        
        validated = processor._validate_step_back_question(step_back, original)
        
        assert isinstance(validated, str)
        assert len(validated) > 0
        assert validated.endswith('?')
    
    def test_validate_step_back_question_too_similar(self, processor):
        """Test validation when step-back is too similar to original"""
        original = "What is task decomposition?"
        step_back = "What is task decomposition?"  # Identical
        
        validated = processor._validate_step_back_question(step_back, original)
        
        assert isinstance(validated, str)
        assert len(validated) > 0
        assert validated.endswith('?')
        assert validated != original
    
    def test_format_documents(self, processor, sample_documents):
        """Test document formatting"""
        formatted = processor._format_documents(sample_documents[:2])
        
        assert "Document 1:" in formatted
        assert "Document 2:" in formatted
        assert sample_documents[0].content in formatted
        assert sample_documents[1].content in formatted
    
    def test_format_documents_empty(self, processor):
        """Test document formatting with empty list"""
        formatted = processor._format_documents([])
        
        assert formatted == "No relevant context found."
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_generate_step_back_question_success(self, mock_llm_class):
        """Test successful step-back question generation"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        
        # Mock the chain invoke
        processor.step_back_chain = Mock()
        processor.step_back_chain.invoke.return_value = "what are the general approaches to task decomposition in AI systems"
        
        question = "What is the specific implementation of task decomposition in GPT-4?"
        step_back = processor.generate_step_back_question(question)
        
        assert isinstance(step_back, str)
        assert len(step_back) > 0
        assert step_back.endswith('?')
        assert step_back != question
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_generate_step_back_question_failure(self, mock_llm_class):
        """Test step-back question generation failure handling"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        processor.step_back_chain = Mock()
        processor.step_back_chain.invoke.side_effect = Exception("LLM error")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.generate_step_back_question(question)
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_process_with_retriever_success(self, mock_llm_class, sample_documents):
        """Test successful processing with retriever"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        
        # Mock generate_step_back_question
        processor.generate_step_back_question = Mock()
        processor.generate_step_back_question.return_value = "What are general AI approaches?"
        
        # Mock retriever function
        def mock_retriever(query):
            if "specific" in query or "GPT-4" in query:
                return sample_documents[:2]  # Specific context
            else:
                return sample_documents[2:]  # General context
        
        question = "What is the specific implementation of task decomposition in GPT-4?"
        normal_docs, step_back_docs, step_back_q = processor.process_with_retriever(question, mock_retriever, top_k=3)
        
        assert len(normal_docs) <= 3
        assert len(step_back_docs) <= 3
        assert step_back_q == "What are general AI approaches?"
        assert all(isinstance(doc, Document) for doc in normal_docs)
        assert all(isinstance(doc, Document) for doc in step_back_docs)
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_process_with_retriever_failure(self, mock_llm_class):
        """Test processing with retriever failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        processor.generate_step_back_question = Mock()
        processor.generate_step_back_question.side_effect = Exception("Generation failed")
        
        def mock_retriever(query):
            return []
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_with_retriever(question, mock_retriever)
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_generate_response_with_step_back_success(self, mock_llm_class, sample_documents):
        """Test successful response generation with step-back context"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        
        # Mock the response chain
        processor.response_chain = Mock()
        processor.response_chain.invoke.return_value = "Comprehensive answer using both contexts"
        
        question = "What is task decomposition in GPT-4?"
        normal_context = sample_documents[:2]
        step_back_context = sample_documents[2:]
        
        response = processor.generate_response_with_step_back(question, normal_context, step_back_context)
        
        assert response == "Comprehensive answer using both contexts"
        processor.response_chain.invoke.assert_called_once()
        
        # Check that the invoke was called with proper context
        call_args = processor.response_chain.invoke.call_args[0][0]
        assert call_args["question"] == question
        assert "normal_context" in call_args
        assert "step_back_context" in call_args
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_generate_response_with_step_back_failure(self, mock_llm_class, sample_documents):
        """Test response generation failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        processor.response_chain = Mock()
        processor.response_chain.invoke.side_effect = Exception("Response generation failed")
        
        question = "What is task decomposition?"
        normal_context = sample_documents[:1]
        step_back_context = sample_documents[1:]
        
        with pytest.raises(QueryProcessingError):
            processor.generate_response_with_step_back(question, normal_context, step_back_context)
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_process_query_with_step_back_success(self, mock_llm_class, sample_documents):
        """Test full step-back workflow"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        
        # Mock the methods
        processor.process_with_retriever = Mock()
        processor.process_with_retriever.return_value = (
            sample_documents[:2],  # normal_docs
            sample_documents[2:],  # step_back_docs
            "What are general AI approaches?"  # step_back_question
        )
        
        processor.generate_response_with_step_back = Mock()
        processor.generate_response_with_step_back.return_value = "Final comprehensive answer"
        
        # Mock retriever function
        def mock_retriever(query):
            return sample_documents[:2]
        
        question = "What is the specific implementation of task decomposition in GPT-4?"
        result = processor.process_query_with_step_back(question, mock_retriever, top_k=3)
        
        assert result == "Final comprehensive answer"
        processor.process_with_retriever.assert_called_once_with(question, mock_retriever, 3)
        processor.generate_response_with_step_back.assert_called_once()
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_process_query_success(self, mock_llm_class):
        """Test successful query processing"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        
        # Mock generate_step_back_question
        processor.generate_step_back_question = Mock()
        processor.generate_step_back_question.return_value = "What are general AI approaches?"
        
        question = "What is the specific implementation of task decomposition in GPT-4?"
        result = processor.process_query(question)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == question
        assert len(result.transformed_queries) == 2
        assert result.transformed_queries[0] == question  # Original
        assert result.transformed_queries[1] == "What are general AI approaches?"  # Step-back
        assert result.strategy_used == "step_back"
        assert "step_back_question" in result.metadata
        assert result.metadata["step_back_question"] == "What are general AI approaches?"
    
    @patch('src.rag_engine.query.step_back.ChatGoogleGenerativeAI')
    def test_process_query_failure(self, mock_llm_class):
        """Test query processing failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        processor = StepBackProcessor()
        processor.generate_step_back_question = Mock()
        processor.generate_step_back_question.side_effect = Exception("Generation failed")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            processor.process_query(question)
    
    def test_get_config(self, processor):
        """Test configuration retrieval"""
        config = processor.get_config()
        
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        assert config["temperature"] == 0.0
        assert config["strategy"] == "step_back"
    
    def test_create_step_back_prompt(self, processor):
        """Test step-back prompt creation"""
        template = processor._create_step_back_prompt()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "expert at world knowledge" in template_str
        assert "step back" in template_str
        assert "generic" in template_str
        assert "question" in template_str
        
        # Check that few-shot examples are included
        assert "Police" in template_str  # From examples
        assert "Jan Sindel" in template_str  # From examples
    
    def test_create_response_prompt(self, processor):
        """Test response prompt creation"""
        template = processor._create_response_prompt()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "expert of world knowledge" in template_str
        assert "Normal Context" in template_str
        assert "Step-Back Context" in template_str
        assert "Original Question" in template_str
        assert "comprehensive" in template_str
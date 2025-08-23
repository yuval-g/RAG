"""
Unit tests for QueryDecomposer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.decomposition import QueryDecomposer
from src.rag_engine.core.models import Document, ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestQueryDecomposer:
    """Test cases for QueryDecomposer"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(content="Task decomposition involves breaking down complex tasks", metadata={"id": "1"}),
            Document(content="LLM agents use planning for task management", metadata={"id": "2"}),
            Document(content="Memory systems help agents retain information", metadata={"id": "3"}),
            Document(content="Tool use enables agents to interact with external systems", metadata={"id": "4"})
        ]
    
    @pytest.fixture
    def decomposer(self):
        """Create a QueryDecomposer instance for testing"""
        with patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI'):
            return QueryDecomposer(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0,
                num_sub_questions=3
            )
    
    def test_init(self):
        """Test QueryDecomposer initialization"""
        with patch('src.rag_engine.query.decomposition.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            decomposer = QueryDecomposer(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.2,
                num_sub_questions=4
            )
            
            assert decomposer.llm_model == "gemini-2.0-flash-lite"
            assert decomposer.temperature == 0.2
            assert decomposer.num_sub_questions == 4
            mock_get_llm.assert_called_once()
    
    def test_parse_sub_questions(self, decomposer):
        """Test sub-question parsing from LLM output"""
        output = """1. What are the core components of an LLM-powered autonomous agent system?
2. How is memory implemented in LLM-powered autonomous agents?
3. What role does planning and task decomposition play in autonomous agent systems?"""
        
        parsed = decomposer._parse_sub_questions(output)
        
        expected = [
            "What are the core components of an LLM-powered autonomous agent system?",
            "How is memory implemented in LLM-powered autonomous agents?",
            "What role does planning and task decomposition play in autonomous agent systems?"
        ]
        
        assert parsed == expected
    
    def test_parse_sub_questions_limit(self, decomposer):
        """Test sub-question parsing with limit"""
        output = """1. Question one
2. Question two
3. Question three
4. Question four
5. Question five"""
        
        parsed = decomposer._parse_sub_questions(output)
        
        # Should limit to num_sub_questions (3)
        assert len(parsed) == 3
        assert parsed[0] == "Question one"
        assert parsed[2] == "Question three"
    
    def test_parse_sub_questions_without_numbering(self, decomposer):
        """Test sub-question parsing without numbering"""
        output = """What are the main components?
How does the system work?
What are the benefits?"""
        
        parsed = decomposer._parse_sub_questions(output)
        
        expected = [
            "What are the main components?",
            "How does the system work?",
            "What are the benefits?"
        ]
        
        assert parsed == expected
    
    def test_is_question(self, decomposer):
        """Test question detection"""
        assert decomposer._is_question("What is task decomposition?") == True
        assert decomposer._is_question("How do agents work?") == True
        assert decomposer._is_question("Why is this important?") == True
        assert decomposer._is_question("Can agents learn?") == True
        assert decomposer._is_question("Task decomposition is important") == False
        assert decomposer._is_question("") == False
    
    def test_is_too_similar(self, decomposer):
        """Test similarity detection"""
        question1 = "What are the main components of an LLM agent system?"
        question2 = "What are the main components of an LLM agent system?"  # Identical
        question3 = "How do memory systems work in agents?"  # Different
        
        assert decomposer._is_too_similar(question1, question2) == True
        assert decomposer._is_too_similar(question1, question3) == False
    
    def test_create_fallback_questions(self, decomposer):
        """Test fallback question creation"""
        original = "What are the main components of an LLM-powered autonomous agent system?"
        fallback = decomposer._create_fallback_questions(original)
        
        assert len(fallback) >= 1
        assert all(isinstance(q, str) and len(q) > 0 for q in fallback)
        assert any("component" in q.lower() for q in fallback)
    
    def test_create_fallback_questions_empty_input(self, decomposer):
        """Test fallback question creation with minimal input"""
        original = "What is this?"
        fallback = decomposer._create_fallback_questions(original)
        
        assert len(fallback) >= 1
        assert all(isinstance(q, str) and len(q) > 0 for q in fallback)
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_decompose_query_success(self, mock_llm_class):
        """Test successful query decomposition"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock the chain invoke
        decomposer.decomposition_chain = Mock()
        decomposer.decomposition_chain.invoke.return_value = [
            "What are the core components of LLM agents?",
            "How do agents implement memory systems?",
            "What role does planning play in agent systems?"
        ]
        
        question = "What are the main components of an LLM-powered autonomous agent system?"
        sub_questions = decomposer.decompose_query(question)
        
        assert len(sub_questions) == 3
        assert "What are the core components of LLM agents?" in sub_questions
        assert "How do agents implement memory systems?" in sub_questions
        assert "What role does planning play in agent systems?" in sub_questions
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_decompose_query_failure(self, mock_llm_class):
        """Test query decomposition failure handling"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        decomposer.decomposition_chain = Mock()
        decomposer.decomposition_chain.invoke.side_effect = Exception("LLM error")
        
        question = "What are the main components?"
        
        with pytest.raises(QueryProcessingError):
            decomposer.decompose_query(question)
    
    def test_validate_sub_questions(self, decomposer):
        """Test sub-question validation"""
        original = "What are the main components of LLM agents?"
        sub_questions = [
            "What are the core components of LLM agents?",
            "What are the main components of LLM agents?",  # Too similar
            "",  # Empty
            "A" * 600,  # Too long
            "How do memory systems work?",
            "Task decomposition is important"  # Not a question
        ]
        
        validated = decomposer._validate_sub_questions(sub_questions, original)
        
        # Should filter out similar, empty, too long, and convert non-questions
        assert len(validated) <= 3  # num_sub_questions
        assert "What are the core components of LLM agents?" in validated
        assert "How do memory systems work?" in validated
        assert "What are the main components of LLM agents?" not in validated
        assert "" not in validated
        assert "A" * 600 not in validated
    
    def test_validate_sub_questions_with_fallback(self, decomposer):
        """Test validation with fallback when no valid questions"""
        original = "What are the main components?"
        sub_questions = ["", "   ", "A" * 600]  # All invalid
        
        validated = decomposer._validate_sub_questions(sub_questions, original)
        
        # Should create fallback questions
        assert len(validated) >= 1
        assert all(isinstance(q, str) and len(q) > 0 for q in validated)
    
    def test_format_qa_pairs(self, decomposer):
        """Test Q&A pair formatting"""
        questions = [
            "What are the main components?",
            "How does it work?",
            "What are the benefits?"
        ]
        answers = [
            "The main components are A, B, and C.",
            "It works by processing data through multiple stages.",
            "The benefits include improved efficiency and accuracy."
        ]
        
        formatted = decomposer._format_qa_pairs(questions, answers)
        
        assert "Question 1:" in formatted
        assert "Answer 1:" in formatted
        assert "Question 2:" in formatted
        assert "Answer 2:" in formatted
        assert "Question 3:" in formatted
        assert "Answer 3:" in formatted
        assert "The main components are A, B, and C." in formatted
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_answer_sub_questions_success(self, mock_llm_class, sample_documents):
        """Test successful sub-question answering"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock retriever and answerer functions
        def mock_retriever(question):
            return sample_documents[:2]  # Return first 2 documents
        
        def mock_answerer(question, documents):
            return f"Answer to: {question}"
        
        sub_questions = [
            "What are the main components?",
            "How does it work?",
            "What are the benefits?"
        ]
        
        answers = decomposer.answer_sub_questions(sub_questions, mock_retriever, mock_answerer)
        
        assert len(answers) == 3
        assert all("Answer to:" in answer for answer in answers)
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_answer_sub_questions_failure(self, mock_llm_class):
        """Test sub-question answering failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock functions that raise exceptions
        def mock_retriever(question):
            raise Exception("Retrieval failed")
        
        def mock_answerer(question, documents):
            return "Answer"
        
        sub_questions = ["What are the main components?"]
        
        with pytest.raises(QueryProcessingError):
            decomposer.answer_sub_questions(sub_questions, mock_retriever, mock_answerer)
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_synthesize_answer_success(self, mock_llm_class):
        """Test successful answer synthesis"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock the synthesis chain
        decomposer.synthesis_chain = Mock()
        decomposer.synthesis_chain.invoke.return_value = "Synthesized final answer"
        
        original_question = "What are the main components of LLM agents?"
        sub_questions = ["What are components?", "How do they work?"]
        sub_answers = ["Components are A, B, C", "They work by processing data"]
        
        result = decomposer.synthesize_answer(original_question, sub_questions, sub_answers)
        
        assert result == "Synthesized final answer"
        decomposer.synthesis_chain.invoke.assert_called_once()
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_synthesize_answer_failure(self, mock_llm_class):
        """Test answer synthesis failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        decomposer.synthesis_chain = Mock()
        decomposer.synthesis_chain.invoke.side_effect = Exception("Synthesis failed")
        
        original_question = "What are the main components?"
        sub_questions = ["What are components?"]
        sub_answers = ["Components are A, B, C"]
        
        with pytest.raises(QueryProcessingError):
            decomposer.synthesize_answer(original_question, sub_questions, sub_answers)
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_process_query_with_decomposition_success(self, mock_llm_class, sample_documents):
        """Test full decomposition workflow"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock all the methods
        decomposer.decompose_query = Mock()
        decomposer.decompose_query.return_value = ["Sub-question 1", "Sub-question 2"]
        
        decomposer.answer_sub_questions = Mock()
        decomposer.answer_sub_questions.return_value = ["Answer 1", "Answer 2"]
        
        decomposer.synthesize_answer = Mock()
        decomposer.synthesize_answer.return_value = "Final synthesized answer"
        
        # Mock functions
        def mock_retriever(question):
            return sample_documents[:2]
        
        def mock_answerer(question, documents):
            return f"Answer to {question}"
        
        question = "What are the main components of LLM agents?"
        result = decomposer.process_query_with_decomposition(question, mock_retriever, mock_answerer)
        
        assert result == "Final synthesized answer"
        decomposer.decompose_query.assert_called_once_with(question)
        decomposer.answer_sub_questions.assert_called_once()
        decomposer.synthesize_answer.assert_called_once()
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_process_query_success(self, mock_llm_class):
        """Test successful query processing"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        
        # Mock decompose_query
        decomposer.decompose_query = Mock()
        decomposer.decompose_query.return_value = [
            "What are the core components?",
            "How do memory systems work?",
            "What role does planning play?"
        ]
        
        question = "What are the main components of LLM agents?"
        result = decomposer.process_query(question)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == question
        assert len(result.transformed_queries) == 3
        assert result.strategy_used == "decomposition"
        assert "num_sub_questions" in result.metadata
        assert result.metadata["num_sub_questions"] == 3
    
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_process_query_failure(self, mock_llm_class):
        """Test query processing failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        decomposer = QueryDecomposer()
        decomposer.decompose_query = Mock()
        decomposer.decompose_query.side_effect = Exception("Decomposition failed")
        
        question = "What are the main components?"
        
        with pytest.raises(QueryProcessingError):
            decomposer.process_query(question)
    
    def test_get_config(self, decomposer):
        """Test configuration retrieval"""
        config = decomposer.get_config()
        
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        assert config["temperature"] == 0.0
        assert config["num_sub_questions"] == 3
        assert config["strategy"] == "decomposition"
    
    def test_create_decomposition_prompt(self, decomposer):
        """Test decomposition prompt creation"""
        template = decomposer._create_decomposition_prompt()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "helpful assistant" in template_str
        assert "sub-questions" in template_str
        assert "3 queries" in template_str  # num_sub_questions
        assert "question" in template_str
    
    def test_create_synthesis_prompt(self, decomposer):
        """Test synthesis prompt creation"""
        template = decomposer._create_synthesis_prompt()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "Q+A pairs" in template_str
        assert "synthesize" in template_str
        assert "context" in template_str
        assert "question" in template_str
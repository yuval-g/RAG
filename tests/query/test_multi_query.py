"""
Unit tests for MultiQueryGenerator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.query.multi_query import MultiQueryGenerator
from src.rag_engine.core.models import ProcessedQuery
from src.rag_engine.core.exceptions import QueryProcessingError


class TestMultiQueryGenerator:
    """Test cases for MultiQueryGenerator"""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response with sample queries"""
        return """1. How can LLM agents break down complex tasks?
2. What is the process of task decomposition in the context of large language model agents?
3. What are the methods for decomposing tasks for LLM-powered agents?
4. Explain the concept of task decomposition as it applies to AI agents using LLMs.
5. In what ways do LLM agents handle task decomposition?"""
    
    @pytest.fixture
    def generator(self):
        """Create a MultiQueryGenerator instance for testing"""
        with patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI'):
            return MultiQueryGenerator(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.0,
                num_queries=5
            )
    
    def test_init(self):
        """Test MultiQueryGenerator initialization"""
        with patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI') as mock_llm:
            generator = MultiQueryGenerator(
                llm_model="gemini-2.0-flash-lite",
                temperature=0.2,
                num_queries=3
            )
            
            assert generator.llm_model == "gemini-2.0-flash-lite"
            assert generator.temperature == 0.2
            assert generator.num_queries == 3
            mock_llm.assert_called_once()
    
    def test_parse_queries(self, generator):
        """Test query parsing from LLM output"""
        output = """1. How can LLM agents break down complex tasks?
2. What is the process of task decomposition?
3. What are the methods for decomposing tasks?"""
        
        parsed = generator._parse_queries(output)
        
        expected = [
            "How can LLM agents break down complex tasks?",
            "What is the process of task decomposition?",
            "What are the methods for decomposing tasks?"
        ]
        
        assert parsed == expected
    
    def test_parse_queries_with_empty_lines(self, generator):
        """Test query parsing with empty lines and whitespace"""
        output = """1. First query
        
2. Second query

3. Third query
        """
        
        parsed = generator._parse_queries(output)
        
        expected = [
            "First query",
            "Second query", 
            "Third query"
        ]
        
        assert parsed == expected
    
    def test_parse_queries_without_numbering(self, generator):
        """Test query parsing without numbering"""
        output = """How can LLM agents break down complex tasks?
What is the process of task decomposition?
What are the methods for decomposing tasks?"""
        
        parsed = generator._parse_queries(output)
        
        expected = [
            "How can LLM agents break down complex tasks?",
            "What is the process of task decomposition?",
            "What are the methods for decomposing tasks?"
        ]
        
        assert parsed == expected
    
    def test_validate_queries(self, generator):
        """Test query validation"""
        original = "What is task decomposition for LLM agents?"
        queries = [
            "How can LLM agents break down complex tasks?",
            "What is task decomposition for LLM agents?",  # Too similar
            "",  # Empty
            "A" * 600,  # Too long
            "What are the methods for decomposing tasks?"
        ]
        
        validated = generator._validate_queries(queries, original)
        
        # Should filter out similar, empty, and too long queries
        assert len(validated) <= 4  # num_queries - 1
        assert "How can LLM agents break down complex tasks?" in validated
        assert "What are the methods for decomposing tasks?" in validated
        assert "What is task decomposition for LLM agents?" not in validated
        assert "" not in validated
        assert "A" * 600 not in validated
    
    def test_is_too_similar(self, generator):
        """Test similarity detection"""
        query1 = "What is task decomposition for LLM agents?"
        query2 = "What is task decomposition for LLM agents?"  # Identical
        query3 = "How do LLM agents handle task decomposition?"  # Different
        
        assert generator._is_too_similar(query1, query2) == True
        assert generator._is_too_similar(query1, query3) == False
    
    @patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI')
    def test_generate_queries_success(self, mock_llm_class, mock_llm_response):
        """Test successful query generation"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Create generator
        generator = MultiQueryGenerator()
        
        # Mock the chain invoke
        generator.generation_chain = Mock()
        generator.generation_chain.invoke.return_value = [
            "How can LLM agents break down complex tasks?",
            "What are the methods for decomposing tasks?",
            "Explain task decomposition in AI agents"
        ]
        
        question = "What is task decomposition for LLM agents?"
        queries = generator.generate_queries(question)
        
        # Should include original + alternatives
        assert len(queries) >= 1
        assert question in queries  # Original should be included
        assert "How can LLM agents break down complex tasks?" in queries
    
    @patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI')
    def test_generate_queries_failure(self, mock_llm_class):
        """Test query generation failure handling"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        generator = MultiQueryGenerator()
        generator.generation_chain = Mock()
        generator.generation_chain.invoke.side_effect = Exception("LLM error")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            generator.generate_queries(question)
    
    @patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI')
    def test_process_query_success(self, mock_llm_class):
        """Test successful query processing"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        generator = MultiQueryGenerator()
        
        # Mock generate_queries
        generator.generate_queries = Mock()
        generator.generate_queries.return_value = [
            "What is task decomposition for LLM agents?",
            "How can LLM agents break down complex tasks?",
            "What are the methods for decomposing tasks?"
        ]
        
        question = "What is task decomposition for LLM agents?"
        result = generator.process_query(question)
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == question
        assert len(result.transformed_queries) == 3
        assert result.strategy_used == "multi_query"
        assert "num_generated" in result.metadata
        assert result.metadata["num_generated"] == 2  # Exclude original
    
    @patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI')
    def test_process_query_failure(self, mock_llm_class):
        """Test query processing failure handling"""
        # Setup mock
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        generator = MultiQueryGenerator()
        generator.generate_queries = Mock()
        generator.generate_queries.side_effect = Exception("Generation failed")
        
        question = "What is task decomposition?"
        
        with pytest.raises(QueryProcessingError):
            generator.process_query(question)
    
    def test_get_config(self, generator):
        """Test configuration retrieval"""
        config = generator.get_config()
        
        assert config["llm_model"] == "gemini-2.0-flash-lite"
        assert config["temperature"] == 0.0
        assert config["num_queries"] == 5
        assert config["strategy"] == "multi_query"
    
    def test_create_prompt_template(self, generator):
        """Test prompt template creation"""
        template = generator._create_prompt_template()
        
        # Check that template contains expected elements
        template_str = str(template)
        assert "AI language model assistant" in template_str
        assert "different versions" in template_str
        assert "vector" in template_str
        assert "question" in template_str
    
    def test_validate_queries_empty_input(self, generator):
        """Test validation with empty query list"""
        original = "What is task decomposition?"
        queries = []
        
        validated = generator._validate_queries(queries, original)
        
        # Should create a fallback query
        assert len(validated) == 1
        assert "what is" in validated[0].lower()
    
    def test_validate_queries_all_invalid(self, generator):
        """Test validation when all queries are invalid"""
        original = "What is task decomposition?"
        queries = ["", "   ", "A" * 600]  # All invalid
        
        validated = generator._validate_queries(queries, original)
        
        # Should create a fallback query
        assert len(validated) == 1
        assert isinstance(validated[0], str)
        assert len(validated[0]) > 0
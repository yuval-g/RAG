"""
Unit tests for QueryStructurer
"""

import pytest
import datetime
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from src.rag_engine.routing.query_structurer import (
    QueryStructurer, TutorialSearch, DocumentSearch, query_analyzer
)


class TestTutorialSearch:
    """Test cases for TutorialSearch model"""
    
    def test_tutorial_search_basic(self):
        """Test basic TutorialSearch creation"""
        search = TutorialSearch(
            content_search="rag from scratch",
            title_search="rag from scratch"
        )
        assert search.content_search == "rag from scratch"
        assert search.title_search == "rag from scratch"
        assert search.min_view_count is None
        assert search.max_view_count is None
    
    def test_tutorial_search_with_filters(self):
        """Test TutorialSearch with metadata filters"""
        search = TutorialSearch(
            content_search="chat langchain",
            title_search="chat langchain 2023",
            earliest_publish_date=datetime.date(2023, 1, 1),
            latest_publish_date=datetime.date(2024, 1, 1),
            max_length_sec=300
        )
        assert search.content_search == "chat langchain"
        assert search.title_search == "chat langchain 2023"
        assert search.earliest_publish_date == datetime.date(2023, 1, 1)
        assert search.latest_publish_date == datetime.date(2024, 1, 1)
        assert search.max_length_sec == 300
    
    def test_tutorial_search_to_dict(self):
        """Test TutorialSearch to_dict method"""
        search = TutorialSearch(
            content_search="test",
            title_search="test",
            min_view_count=100
        )
        result = search.to_dict()
        expected = {
            "content_search": "test",
            "title_search": "test", 
            "min_view_count": 100
        }
        assert result == expected
    
    def test_tutorial_search_pretty_print(self, capsys):
        """Test TutorialSearch pretty_print method"""
        search = TutorialSearch(
            content_search="test",
            title_search="test",
            min_view_count=100
        )
        search.pretty_print()
        captured = capsys.readouterr()
        assert "content_search: test" in captured.out
        assert "title_search: test" in captured.out
        assert "min_view_count: 100" in captured.out


class TestDocumentSearch:
    """Test cases for DocumentSearch model"""
    
    def test_document_search_basic(self):
        """Test basic DocumentSearch creation"""
        search = DocumentSearch(
            content_search="machine learning",
            title_search="machine learning"
        )
        assert search.content_search == "machine learning"
        assert search.title_search == "machine learning"
        assert search.author is None
        assert search.category is None
    
    def test_document_search_with_filters(self):
        """Test DocumentSearch with metadata filters"""
        search = DocumentSearch(
            content_search="AI research",
            title_search="AI research",
            author="John Doe",
            category="research",
            tags=["AI", "ML", "research"],
            min_word_count=1000
        )
        assert search.content_search == "AI research"
        assert search.author == "John Doe"
        assert search.category == "research"
        assert search.tags == ["AI", "ML", "research"]
        assert search.min_word_count == 1000
    
    def test_document_search_to_dict(self):
        """Test DocumentSearch to_dict method"""
        search = DocumentSearch(
            content_search="test",
            title_search="test",
            author="Test Author"
        )
        result = search.to_dict()
        expected = {
            "content_search": "test",
            "title_search": "test",
            "author": "Test Author"
        }
        assert result == expected


class TestQueryStructurer:
    """Test cases for QueryStructurer"""
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structurer_initialization(self, mock_llm_class):
        """Test structurer initialization"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        assert structurer.model_name == "gemini-2.0-flash-lite"
        assert structurer.temperature == 0.0
        assert hasattr(structurer, 'tutorial_analyzer')
        assert hasattr(structurer, 'document_analyzer')
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_tutorial_query_success(self, mock_llm_class):
        """Test successful tutorial query structuring"""
        # Mock LLM and structured output
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        # Create expected result
        expected_result = TutorialSearch(
            content_search="rag from scratch",
            title_search="rag from scratch"
        )
        mock_structured_llm.invoke.return_value = expected_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        # Manually set the analyzer to use our mock
        structurer.tutorial_analyzer = mock_structured_llm
        
        result = structurer.structure_tutorial_query("rag from scratch")
        
        assert isinstance(result, TutorialSearch)
        assert result.content_search == "rag from scratch"
        assert result.title_search == "rag from scratch"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_tutorial_query_with_filters(self, mock_llm_class):
        """Test tutorial query structuring with metadata filters"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        # Create expected result with filters
        expected_result = TutorialSearch(
            content_search="chat langchain",
            title_search="chat langchain 2023",
            earliest_publish_date=datetime.date(2023, 1, 1),
            latest_publish_date=datetime.date(2024, 1, 1)
        )
        mock_structured_llm.invoke.return_value = expected_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.tutorial_analyzer = mock_structured_llm
        
        result = structurer.structure_tutorial_query("videos on chat langchain published in 2023")
        
        assert isinstance(result, TutorialSearch)
        assert result.content_search == "chat langchain"
        assert result.earliest_publish_date == datetime.date(2023, 1, 1)
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_tutorial_query_error_fallback(self, mock_llm_class):
        """Test tutorial query structuring with error fallback"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.tutorial_analyzer = mock_structured_llm
        
        result = structurer.structure_tutorial_query("some query")
        
        assert isinstance(result, TutorialSearch)
        assert result.content_search == "some query"
        assert result.title_search == "some query"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_document_query_success(self, mock_llm_class):
        """Test successful document query structuring"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        expected_result = DocumentSearch(
            content_search="machine learning",
            title_search="machine learning",
            author="John Doe"
        )
        mock_structured_llm.invoke.return_value = expected_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.document_analyzer = mock_structured_llm
        
        result = structurer.structure_document_query("machine learning by John Doe")
        
        assert isinstance(result, DocumentSearch)
        assert result.content_search == "machine learning"
        assert result.author == "John Doe"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_document_query_error_fallback(self, mock_llm_class):
        """Test document query structuring with error fallback"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.document_analyzer = mock_structured_llm
        
        result = structurer.structure_document_query("some query")
        
        assert isinstance(result, DocumentSearch)
        assert result.content_search == "some query"
        assert result.title_search == "some query"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_query_tutorial_type(self, mock_llm_class):
        """Test structure_query with tutorial type"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        expected_result = TutorialSearch(
            content_search="test",
            title_search="test"
        )
        mock_structured_llm.invoke.return_value = expected_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.tutorial_analyzer = mock_structured_llm
        
        result = structurer.structure_query("test query", "tutorial")
        
        assert isinstance(result, dict)
        assert result["content_search"] == "test"
        assert result["title_search"] == "test"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_query_document_type(self, mock_llm_class):
        """Test structure_query with document type"""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        expected_result = DocumentSearch(
            content_search="test",
            title_search="test"
        )
        mock_structured_llm.invoke.return_value = expected_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        structurer = QueryStructurer()
        structurer.document_analyzer = mock_structured_llm
        
        result = structurer.structure_query("test query", "document")
        
        assert isinstance(result, dict)
        assert result["content_search"] == "test"
        assert result["title_search"] == "test"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_structure_query_invalid_type(self, mock_llm_class):
        """Test structure_query with invalid type"""
        mock_llm_class.return_value = Mock()
        structurer = QueryStructurer()
        
        with pytest.raises(ValueError, match="Unsupported search type"):
            structurer.structure_query("test query", "invalid")
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_analyze_query_complexity(self, mock_llm_class):
        """Test query complexity analysis"""
        mock_llm_class.return_value = Mock()
        structurer = QueryStructurer()
        
        # Test query with date filter
        analysis = structurer.analyze_query_complexity("videos published in 2023")
        assert analysis["has_date_filter"] is True
        assert analysis["complexity_score"] >= 1
        assert analysis["recommended_search_type"] == "tutorial"
        
        # Test query with length filter
        analysis = structurer.analyze_query_complexity("short videos under 5 minutes")
        assert analysis["has_length_filter"] is True
        assert analysis["recommended_search_type"] == "tutorial"
        
        # Test document query
        analysis = structurer.analyze_query_complexity("articles about machine learning")
        assert analysis["recommended_search_type"] == "document"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    def test_get_supported_filters(self, mock_llm_class):
        """Test getting supported filters"""
        mock_llm_class.return_value = Mock()
        structurer = QueryStructurer()
        
        tutorial_filters = structurer.get_supported_filters("tutorial")
        assert "content_search" in tutorial_filters
        assert "title_search" in tutorial_filters
        assert "min_view_count" in tutorial_filters
        assert "max_length_sec" in tutorial_filters
        
        document_filters = structurer.get_supported_filters("document")
        assert "content_search" in document_filters
        assert "title_search" in document_filters
        assert "author" in document_filters
        assert "category" in document_filters
        
        invalid_filters = structurer.get_supported_filters("invalid")
        assert invalid_filters == []


class TestLegacyFunctions:
    """Test cases for legacy compatibility functions"""
    
    @patch('src.rag_engine.routing.query_structurer.QueryStructurer')
    def test_query_analyzer_function(self, mock_structurer_class):
        """Test the legacy query_analyzer function"""
        # Mock the structurer
        mock_structurer = Mock()
        mock_result = TutorialSearch(
            content_search="test",
            title_search="test"
        )
        mock_structurer.structure_tutorial_query.return_value = mock_result
        mock_structurer_class.return_value = mock_structurer
        
        result = query_analyzer("test query")
        
        assert isinstance(result, TutorialSearch)
        assert result.content_search == "test"
        assert result.title_search == "test"
        mock_structurer.structure_tutorial_query.assert_called_once_with("test query")
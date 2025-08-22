"""
Unit tests for LogicalRouter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.routing.logical_router import LogicalRouter, RouteQuery
from src.rag_engine.core.models import RouteConfig, RoutingDecision


class TestLogicalRouter:
    """Test cases for LogicalRouter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_routes = [
            RouteConfig(
                name="python_docs",
                description="Python programming documentation and tutorials",
                keywords=["python", "django", "flask", "pandas", "numpy"]
            ),
            RouteConfig(
                name="js_docs", 
                description="JavaScript documentation and web development guides",
                keywords=["javascript", "react", "node", "vue", "angular"]
            ),
            RouteConfig(
                name="golang_docs",
                description="Go programming language documentation",
                keywords=["go", "golang", "goroutine", "gin", "fiber"]
            )
        ]
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_router_initialization(self, mock_llm_class):
        """Test router initialization"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter()
        assert router.model_name == "gemini-2.0-flash-lite"
        assert router.temperature == 0.0
        assert len(router.routes) == 0
        assert router.get_available_routes() == []
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_router_initialization_with_routes(self, mock_llm_class):
        """Test router initialization with predefined routes"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        assert "python_docs" in router.routes
        assert "js_docs" in router.routes
        assert "golang_docs" in router.routes
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_add_route(self, mock_llm_class):
        """Test adding a route"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter()
        route = RouteConfig(
            name="test_route",
            description="Test route description",
            keywords=["test", "example"]
        )
        
        result = router.add_route(route)
        assert result is True
        assert "test_route" in router.routes
        assert router.routes["test_route"] == route
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_get_available_routes(self, mock_llm_class):
        """Test getting available routes"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter(routes=self.sample_routes)
        routes = router.get_available_routes()
        assert len(routes) == 3
        assert "python_docs" in routes
        assert "js_docs" in routes
        assert "golang_docs" in routes
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_get_route_config(self, mock_llm_class):
        """Test getting route configuration"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter(routes=self.sample_routes)
        config = router.get_route_config("python_docs")
        assert config is not None
        assert config.name == "python_docs"
        assert "python" in config.keywords
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_remove_route(self, mock_llm_class):
        """Test removing a route"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        
        result = router.remove_route("python_docs")
        assert result is True
        assert len(router.routes) == 2
        assert "python_docs" not in router.routes
        
        # Test removing non-existent route
        result = router.remove_route("non_existent")
        assert result is False
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_route_query_success(self, mock_llm_class):
        """Test successful query routing"""
        # Mock the LLM response
        mock_llm = Mock()
        mock_structured_llm = Mock()
        
        # Create a proper mock result object
        class MockResult:
            def __init__(self):
                self.datasource = "python_docs"
                self.confidence = 0.9
                self.reasoning = "Question is about Python programming"
        
        mock_result = MockResult()
        mock_structured_llm.invoke.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        router = LogicalRouter(routes=self.sample_routes)
        # Manually set the router chain to ensure it uses our mock
        router.router_chain = mock_structured_llm
        
        decision = router.route("How do I use pandas for data analysis?")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route == "python_docs"
        assert decision.confidence == 0.9
        assert decision.reasoning == "Question is about Python programming"
        assert decision.metadata["router_type"] == "logical"
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_route_query_invalid_datasource_fallback(self, mock_llm_class):
        """Test routing with invalid datasource falls back correctly"""
        # Mock the LLM response with invalid datasource
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_result = Mock()
        mock_result.datasource = "invalid_route"
        mock_result.confidence = 0.8
        mock_result.reasoning = "Invalid reasoning"
        
        mock_structured_llm.invoke.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        router = LogicalRouter(routes=self.sample_routes)
        decision = router.route("Some query")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route in router.get_available_routes()
        assert decision.confidence == 0.5
        assert "Fallback" in decision.reasoning
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_route_query_no_routes_configured(self, mock_llm_class):
        """Test routing when no routes are configured"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter()
        
        with pytest.raises(ValueError, match="No routes configured"):
            router.route("Some query")
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_route_query_llm_error_fallback(self, mock_llm_class):
        """Test routing with LLM error falls back correctly"""
        # Mock the LLM to raise an exception
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_class.return_value = mock_llm
        
        router = LogicalRouter(routes=self.sample_routes)
        
        # Manually set the router chain to ensure it exists
        router.router_chain = mock_structured_llm
        
        decision = router.route("Some query")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route in router.get_available_routes()
        assert decision.confidence == 0.1
        assert "Error in routing" in decision.reasoning
        assert decision.metadata.get("fallback") is True
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_build_router_chain_no_routes(self, mock_llm_class):
        """Test building router chain with no routes"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter()
        router._build_router_chain()
        assert router.router_chain is None
    
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_build_router_chain_with_routes(self, mock_llm_class):
        """Test building router chain with routes"""
        mock_llm_class.return_value = Mock()
        router = LogicalRouter(routes=self.sample_routes)
        assert router.router_chain is not None
    
    def test_route_query_model(self):
        """Test RouteQuery model validation"""
        # Test valid model
        route_query = RouteQuery(
            datasource="python_docs",
            confidence=0.8,
            reasoning="Python-related question"
        )
        assert route_query.datasource == "python_docs"
        assert route_query.confidence == 0.8
        assert route_query.reasoning == "Python-related question"
        
        # Test confidence bounds
        with pytest.raises(ValueError):
            RouteQuery(datasource="test", confidence=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            RouteQuery(datasource="test", confidence=-0.1)  # < 0.0
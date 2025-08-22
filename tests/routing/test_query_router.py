"""
Integration tests for QueryRouter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.routing.query_router import QueryRouter, RoutingStrategy
from src.rag_engine.core.models import RouteConfig, RoutingDecision


class TestQueryRouter:
    """Test cases for QueryRouter"""
    
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
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_router_initialization(self, mock_logical, mock_semantic, mock_structurer):
        """Test router initialization"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        assert router.default_strategy == RoutingStrategy.AUTO
        assert router.llm_model == "gemini-2.0-flash-lite"
        assert router.embedding_model == "models/embedding-001"
        assert len(router.routes) == 0
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_router_initialization_with_routes(self, mock_logical, mock_semantic, mock_structurer):
        """Test router initialization with predefined routes"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        assert "python_docs" in router.routes
        assert "js_docs" in router.routes
        assert "golang_docs" in router.routes
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_add_route(self, mock_logical, mock_semantic, mock_structurer):
        """Test adding a route"""
        mock_logical_instance = Mock()
        mock_logical_instance.add_route.return_value = True
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.add_route.return_value = True
        mock_semantic.return_value = mock_semantic_instance
        
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        route = RouteConfig(
            name="test_route",
            description="Test route description",
            keywords=["test", "example"]
        )
        
        result = router.add_route(route)
        assert result is True
        assert "test_route" in router.routes
        assert router.routes["test_route"] == route
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_add_route_failure(self, mock_logical, mock_semantic, mock_structurer):
        """Test adding a route when one router fails"""
        mock_logical_instance = Mock()
        mock_logical_instance.add_route.return_value = True
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.add_route.return_value = False  # Fails
        mock_semantic.return_value = mock_semantic_instance
        
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        route = RouteConfig(name="test_route", description="Test")
        
        result = router.add_route(route)
        assert result is False
        assert "test_route" not in router.routes
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_get_available_routes(self, mock_logical, mock_semantic, mock_structurer):
        """Test getting available routes"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        routes = router.get_available_routes()
        assert len(routes) == 3
        assert "python_docs" in routes
        assert "js_docs" in routes
        assert "golang_docs" in routes
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_remove_route(self, mock_logical, mock_semantic, mock_structurer):
        """Test removing a route"""
        mock_logical_instance = Mock()
        mock_logical_instance.remove_route.return_value = True
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.remove_route.return_value = True
        mock_semantic.return_value = mock_semantic_instance
        
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        
        result = router.remove_route("python_docs")
        assert result is True
        assert len(router.routes) == 2
        assert "python_docs" not in router.routes
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_logical_strategy(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing with logical strategy"""
        # Mock logical router decision
        mock_decision = RoutingDecision(
            selected_route="python_docs",
            confidence=0.9,
            reasoning="Logical routing decision",
            metadata={"router_type": "logical"}
        )
        
        mock_logical_instance = Mock()
        mock_logical_instance.route.return_value = mock_decision
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        decision = router.route("Python question", RoutingStrategy.LOGICAL)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route == "python_docs"
        assert decision.confidence == 0.9
        assert decision.metadata["unified_strategy"] == "logical"
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_semantic_strategy(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing with semantic strategy"""
        # Mock semantic router decision
        mock_decision = RoutingDecision(
            selected_route="js_docs",
            confidence=0.8,
            reasoning="Semantic routing decision",
            metadata={"router_type": "semantic"}
        )
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.route.return_value = mock_decision
        mock_semantic.return_value = mock_semantic_instance
        
        mock_logical.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        decision = router.route("JavaScript question", RoutingStrategy.SEMANTIC)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route == "js_docs"
        assert decision.confidence == 0.8
        assert decision.metadata["unified_strategy"] == "semantic"
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_hybrid_strategy(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing with hybrid strategy"""
        # Mock logical router decision
        logical_decision = RoutingDecision(
            selected_route="python_docs",
            confidence=0.7,
            reasoning="Logical decision",
            metadata={"router_type": "logical"}
        )
        
        # Mock semantic router decision
        semantic_decision = RoutingDecision(
            selected_route="js_docs",
            confidence=0.9,
            reasoning="Semantic decision",
            metadata={"router_type": "semantic"}
        )
        
        mock_logical_instance = Mock()
        mock_logical_instance.route.return_value = logical_decision
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.route.return_value = semantic_decision
        mock_semantic.return_value = mock_semantic_instance
        
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        decision = router.route("Some question", RoutingStrategy.HYBRID)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.metadata["unified_strategy"] == "hybrid"
        assert "logical_decision" in decision.metadata
        assert "semantic_decision" in decision.metadata
        assert "route_scores" in decision.metadata
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_auto_strategy(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing with auto strategy"""
        # Mock query structurer analysis
        mock_analysis = {
            "complexity_score": 1,
            "has_date_filter": False,
            "recommended_search_type": "tutorial"
        }
        
        mock_structurer_instance = Mock()
        mock_structurer_instance.analyze_query_complexity.return_value = mock_analysis
        mock_structurer.return_value = mock_structurer_instance
        
        # Mock semantic router decision (for short query)
        semantic_decision = RoutingDecision(
            selected_route="js_docs",
            confidence=0.8,
            reasoning="Semantic decision",
            metadata={"router_type": "semantic"}
        )
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.route.return_value = semantic_decision
        mock_semantic.return_value = mock_semantic_instance
        
        mock_logical.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        decision = router.route("JS", RoutingStrategy.AUTO)  # Short query
        
        assert isinstance(decision, RoutingDecision)
        assert decision.metadata["unified_strategy"] == "auto"
        assert "auto_selected_strategy" in decision.metadata
        assert "query_analysis" in decision.metadata
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_no_routes_configured(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing when no routes are configured"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        
        with pytest.raises(ValueError, match="No routes configured"):
            router.route("Some query")
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_route_error_fallback(self, mock_logical, mock_semantic, mock_structurer):
        """Test routing with error fallback"""
        mock_logical_instance = Mock()
        mock_logical_instance.route.side_effect = Exception("Routing error")
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        decision = router.route("Some query", RoutingStrategy.LOGICAL)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.confidence == 0.1
        assert "Error in routing" in decision.reasoning
        assert decision.metadata.get("fallback") is True
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_get_routing_strategies(self, mock_logical, mock_semantic, mock_structurer):
        """Test getting available routing strategies"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        strategies = router.get_routing_strategies()
        
        assert "logical" in strategies
        assert "semantic" in strategies
        assert "hybrid" in strategies
        assert "auto" in strategies
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_strategy_config(self, mock_logical, mock_semantic, mock_structurer):
        """Test strategy configuration management"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter()
        
        # Test getting config
        config = router.get_strategy_config()
        assert "logical_threshold" in config
        assert "semantic_threshold" in config
        
        # Test setting config
        new_config = {"logical_threshold": 0.8}
        router.set_strategy_config(new_config)
        updated_config = router.get_strategy_config()
        assert updated_config["logical_threshold"] == 0.8
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_get_route_config(self, mock_logical, mock_semantic, mock_structurer):
        """Test getting route configuration"""
        mock_logical.return_value = Mock()
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        config = router.get_route_config("python_docs")
        
        assert config is not None
        assert config.name == "python_docs"
        assert "python" in config.keywords
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_analyze_query(self, mock_logical, mock_semantic, mock_structurer):
        """Test query analysis"""
        # Mock query structurer analysis
        mock_analysis = {
            "complexity_score": 2,
            "has_date_filter": True
        }
        
        mock_structurer_instance = Mock()
        mock_structurer_instance.analyze_query_complexity.return_value = mock_analysis
        mock_structurer.return_value = mock_structurer_instance
        
        # Mock router decisions
        logical_decision = RoutingDecision(
            selected_route="python_docs",
            confidence=0.8,
            reasoning="Logical decision"
        )
        
        semantic_decision = RoutingDecision(
            selected_route="js_docs",
            confidence=0.6,
            reasoning="Semantic decision"
        )
        
        mock_logical_instance = Mock()
        mock_logical_instance.route.return_value = logical_decision
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic_instance = Mock()
        mock_semantic_instance.route.return_value = semantic_decision
        mock_semantic_instance.get_similarity_scores.return_value = {"python_docs": 0.8}
        mock_semantic.return_value = mock_semantic_instance
        
        router = QueryRouter(routes=self.sample_routes)
        analysis = router.analyze_query("Complex query with date filter")
        
        assert "query" in analysis
        assert "complexity_analysis" in analysis
        assert "routing_recommendations" in analysis
        assert "similarity_scores" in analysis
        assert "recommended_strategy" in analysis
    
    @patch('src.rag_engine.routing.query_router.QueryStructurer')
    @patch('src.rag_engine.routing.query_router.SemanticRouter')
    @patch('src.rag_engine.routing.query_router.LogicalRouter')
    def test_bulk_route(self, mock_logical, mock_semantic, mock_structurer):
        """Test bulk routing"""
        # Mock decision
        mock_decision = RoutingDecision(
            selected_route="python_docs",
            confidence=0.8,
            reasoning="Test decision"
        )
        
        mock_logical_instance = Mock()
        mock_logical_instance.route.return_value = mock_decision
        mock_logical.return_value = mock_logical_instance
        
        mock_semantic.return_value = Mock()
        mock_structurer.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        queries = ["Query 1", "Query 2", "Query 3"]
        decisions = router.bulk_route(queries, RoutingStrategy.LOGICAL)
        
        assert len(decisions) == 3
        assert all(isinstance(d, RoutingDecision) for d in decisions)
        assert all(d.selected_route == "python_docs" for d in decisions)
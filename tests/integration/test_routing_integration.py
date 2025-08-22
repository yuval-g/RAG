"""
Integration tests for the complete routing system
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.routing import (
    QueryRouter, LogicalRouter, SemanticRouter, QueryStructurer,
    RoutingStrategy, RouteQuery, TutorialSearch
)
from src.rag_engine.core.models import RouteConfig, RoutingDecision


class TestRoutingIntegration:
    """Integration tests for the routing system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_routes = [
            RouteConfig(
                name="python_expert",
                description="Python programming expert for coding questions",
                keywords=["python", "django", "flask", "pandas", "numpy"],
                metadata={
                    "prompt_template": "You are a Python expert. Help with: {query}"
                }
            ),
            RouteConfig(
                name="javascript_expert", 
                description="JavaScript expert for web development questions",
                keywords=["javascript", "react", "node", "vue", "angular"],
                metadata={
                    "prompt_template": "You are a JavaScript expert. Help with: {query}"
                }
            )
        ]
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_complete_routing_workflow(self, mock_logical_llm, mock_embeddings, mock_structurer_llm):
        """Test the complete routing workflow with all components"""
        
        # Mock embeddings for semantic router
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock LLMs
        mock_logical_llm.return_value = Mock()
        mock_structurer_llm.return_value = Mock()
        
        # Create unified router
        router = QueryRouter(
            default_strategy=RoutingStrategy.AUTO,
            routes=self.sample_routes
        )
        
        # Verify all components are initialized
        assert isinstance(router.logical_router, LogicalRouter)
        assert isinstance(router.semantic_router, SemanticRouter)
        assert isinstance(router.query_structurer, QueryStructurer)
        
        # Verify routes are added to all sub-routers
        assert len(router.get_available_routes()) == 2
        assert "python_expert" in router.get_available_routes()
        assert "javascript_expert" in router.get_available_routes()
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_routing_strategy_switching(self, mock_logical_llm, mock_embeddings, mock_structurer_llm):
        """Test switching between different routing strategies"""
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock LLMs
        mock_logical_llm.return_value = Mock()
        mock_structurer_llm.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        
        # Mock the individual router responses
        logical_decision = RoutingDecision(
            selected_route="python_expert",
            confidence=0.9,
            reasoning="Logical routing",
            metadata={"router_type": "logical"}
        )
        router.logical_router.route = Mock(return_value=logical_decision)
        
        semantic_decision = RoutingDecision(
            selected_route="javascript_expert",
            confidence=0.8,
            reasoning="Semantic routing",
            metadata={"router_type": "semantic"}
        )
        router.semantic_router.route = Mock(return_value=semantic_decision)
        
        # Test logical strategy
        decision = router.route("Python question", RoutingStrategy.LOGICAL)
        assert decision.selected_route == "python_expert"
        assert decision.metadata["unified_strategy"] == "logical"
        
        # Test semantic strategy
        decision = router.route("JavaScript question", RoutingStrategy.SEMANTIC)
        assert decision.selected_route == "javascript_expert"
        assert decision.metadata["unified_strategy"] == "semantic"
    
    @patch('src.rag_engine.routing.query_structurer.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    def test_query_analysis_integration(self, mock_logical_llm, mock_embeddings, mock_structurer_llm):
        """Test query analysis with all components"""
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_instance.get_similarity_scores.return_value = {"python_expert": 0.8}
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock LLMs
        mock_logical_llm.return_value = Mock()
        mock_structurer_llm.return_value = Mock()
        
        router = QueryRouter(routes=self.sample_routes)
        
        # Mock individual components
        router.query_structurer.analyze_query_complexity = Mock(return_value={
            "complexity_score": 2,
            "has_date_filter": True,
            "recommended_search_type": "tutorial"
        })
        
        router.logical_router.route = Mock(return_value=RoutingDecision(
            selected_route="python_expert",
            confidence=0.9,
            reasoning="Logical decision"
        ))
        
        router.semantic_router.route = Mock(return_value=RoutingDecision(
            selected_route="javascript_expert", 
            confidence=0.7,
            reasoning="Semantic decision"
        ))
        
        router.semantic_router.get_similarity_scores = Mock(return_value={
            "python_expert": 0.8,
            "javascript_expert": 0.6
        })
        
        # Analyze query
        analysis = router.analyze_query("Complex Python question with date filter")
        
        # Verify analysis structure
        assert "query" in analysis
        assert "complexity_analysis" in analysis
        assert "routing_recommendations" in analysis
        assert "similarity_scores" in analysis
        assert "recommended_strategy" in analysis
        
        # Verify routing recommendations
        recommendations = analysis["routing_recommendations"]
        assert "logical" in recommendations
        assert "semantic" in recommendations
        assert recommendations["logical"]["route"] == "python_expert"
        assert recommendations["semantic"]["route"] == "javascript_expert"
    
    def test_route_config_model_validation(self):
        """Test RouteConfig model validation"""
        # Valid route config
        route = RouteConfig(
            name="test_expert",
            description="Test expert for testing",
            keywords=["test", "example"],
            metadata={"custom_field": "value"}
        )
        
        assert route.name == "test_expert"
        assert route.description == "Test expert for testing"
        assert "test" in route.keywords
        assert route.metadata["custom_field"] == "value"
    
    def test_tutorial_search_model_validation(self):
        """Test TutorialSearch model validation and functionality"""
        import datetime
        
        # Create tutorial search with filters
        search = TutorialSearch(
            content_search="machine learning tutorials",
            title_search="ML tutorials",
            min_view_count=1000,
            max_length_sec=600,
            earliest_publish_date=datetime.date(2023, 1, 1)
        )
        
        assert search.content_search == "machine learning tutorials"
        assert search.title_search == "ML tutorials"
        assert search.min_view_count == 1000
        assert search.max_length_sec == 600
        assert search.earliest_publish_date == datetime.date(2023, 1, 1)
        
        # Test to_dict method
        search_dict = search.to_dict()
        assert "content_search" in search_dict
        assert "min_view_count" in search_dict
        assert "latest_publish_date" not in search_dict  # Should exclude None values
    
    def test_routing_decision_model(self):
        """Test RoutingDecision model functionality"""
        decision = RoutingDecision(
            selected_route="python_expert",
            confidence=0.85,
            reasoning="High confidence Python classification",
            metadata={
                "router_type": "logical",
                "model_used": "gemini-2.0-flash-lite",
                "processing_time": 0.123
            }
        )
        
        assert decision.selected_route == "python_expert"
        assert decision.confidence == 0.85
        assert "Python classification" in decision.reasoning
        assert decision.metadata["router_type"] == "logical"
        assert decision.metadata["processing_time"] == 0.123
    
    def test_routing_strategy_enum(self):
        """Test RoutingStrategy enum functionality"""
        # Test enum values
        assert RoutingStrategy.LOGICAL.value == "logical"
        assert RoutingStrategy.SEMANTIC.value == "semantic"
        assert RoutingStrategy.HYBRID.value == "hybrid"
        assert RoutingStrategy.AUTO.value == "auto"
        
        # Test enum comparison
        assert RoutingStrategy.LOGICAL != RoutingStrategy.SEMANTIC
        assert RoutingStrategy.AUTO == RoutingStrategy.AUTO
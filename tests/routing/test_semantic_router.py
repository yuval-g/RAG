"""
Unit tests for SemanticRouter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from src.rag_engine.routing.semantic_router import SemanticRouter, prompt_router
from src.rag_engine.core.models import RouteConfig, RoutingDecision


class TestSemanticRouter:
    """Test cases for SemanticRouter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_routes = [
            RouteConfig(
                name="physics_expert",
                description="Physics professor expert at answering physics questions in a concise manner",
                keywords=["physics", "quantum", "mechanics", "thermodynamics", "relativity"],
                metadata={
                    "prompt_template": "You are a physics professor. Answer this physics question: {query}"
                }
            ),
            RouteConfig(
                name="math_expert", 
                description="Mathematics expert good at breaking down complex math problems step by step",
                keywords=["math", "algebra", "calculus", "geometry", "statistics"],
                metadata={
                    "prompt_template": "You are a math expert. Solve this step by step: {query}"
                }
            ),
            RouteConfig(
                name="programming_expert",
                description="Programming expert specializing in software development and coding",
                keywords=["programming", "code", "software", "python", "javascript"],
                metadata={
                    "prompt_template": "You are a programming expert. Help with this coding question: {query}"
                }
            )
        ]
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_router_initialization(self, mock_embeddings_class):
        """Test router initialization"""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter()
        assert router.embedding_model == "models/embedding-001"
        assert len(router.routes) == 0
        assert len(router.route_embeddings) == 0
        assert router.get_available_routes() == []
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_router_initialization_with_routes(self, mock_embeddings_class):
        """Test router initialization with predefined routes"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        assert len(router.route_embeddings) == 3
        assert "physics_expert" in router.routes
        assert "math_expert" in router.routes
        assert "programming_expert" in router.routes
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_add_route(self, mock_embeddings_class):
        """Test adding a route"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter()
        route = RouteConfig(
            name="test_expert",
            description="Test expert description",
            keywords=["test", "example"]
        )
        
        result = router.add_route(route)
        assert result is True
        assert "test_expert" in router.routes
        assert "test_expert" in router.route_embeddings
        assert router.routes["test_expert"] == route
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_get_available_routes(self, mock_embeddings_class):
        """Test getting available routes"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        routes = router.get_available_routes()
        assert len(routes) == 3
        assert "physics_expert" in routes
        assert "math_expert" in routes
        assert "programming_expert" in routes
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_get_route_config(self, mock_embeddings_class):
        """Test getting route configuration"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        config = router.get_route_config("physics_expert")
        assert config is not None
        assert config.name == "physics_expert"
        assert "physics" in config.keywords
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_remove_route(self, mock_embeddings_class):
        """Test removing a route"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        assert len(router.routes) == 3
        assert len(router.route_embeddings) == 3
        
        result = router.remove_route("physics_expert")
        assert result is True
        assert len(router.routes) == 2
        assert len(router.route_embeddings) == 2
        assert "physics_expert" not in router.routes
        assert "physics_expert" not in router.route_embeddings
        
        # Test removing non-existent route
        result = router.remove_route("non_existent")
        assert result is False
    
    @patch('src.rag_engine.routing.semantic_router.cosine_similarity')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_route_query_success(self, mock_embeddings_class, mock_cosine_similarity):
        """Test successful query routing"""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [0.1, 0.2, 0.3],  # physics_expert embedding
            [0.4, 0.5, 0.6],  # math_expert embedding  
            [0.7, 0.8, 0.9],  # programming_expert embedding
            [0.15, 0.25, 0.35]  # query embedding
        ]
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock cosine similarity to return physics as most similar
        mock_cosine_similarity.return_value = [[0.95, 0.3, 0.2]]  # physics wins
        
        router = SemanticRouter(routes=self.sample_routes)
        decision = router.route("What is quantum mechanics?")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route == "physics_expert"
        assert decision.confidence == 0.95
        assert "semantic similarity" in decision.reasoning
        assert decision.metadata["router_type"] == "semantic"
        assert "similarity_scores" in decision.metadata
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_route_query_no_routes_configured(self, mock_embeddings_class):
        """Test routing when no routes are configured"""
        mock_embeddings_class.return_value = Mock()
        router = SemanticRouter()
        
        with pytest.raises(ValueError, match="No routes configured"):
            router.route("Some query")
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_route_query_embedding_error_fallback(self, mock_embeddings_class):
        """Test routing with embedding error falls back correctly"""
        # Mock embeddings to raise an exception
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [0.1, 0.2, 0.3],  # First call succeeds (for route setup)
            [0.4, 0.5, 0.6],  # Second call succeeds
            [0.7, 0.8, 0.9],  # Third call succeeds
            Exception("Embedding error")  # Query embedding fails
        ]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        decision = router.route("Some query")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_route in router.get_available_routes()
        assert decision.confidence == 0.1
        assert "Error in semantic routing" in decision.reasoning
        assert decision.metadata.get("fallback") is True
    
    @patch('src.rag_engine.routing.semantic_router.cosine_similarity')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_get_similarity_scores(self, mock_embeddings_class, mock_cosine_similarity):
        """Test getting similarity scores for all routes"""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [0.1, 0.2, 0.3],  # physics_expert embedding
            [0.4, 0.5, 0.6],  # math_expert embedding  
            [0.7, 0.8, 0.9],  # programming_expert embedding
            [0.15, 0.25, 0.35]  # query embedding
        ]
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock cosine similarity
        mock_cosine_similarity.return_value = [[0.95, 0.3, 0.2]]
        
        router = SemanticRouter(routes=self.sample_routes)
        scores = router.get_similarity_scores("What is quantum mechanics?")
        
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert "physics_expert" in scores
        assert "math_expert" in scores
        assert "programming_expert" in scores
        assert scores["physics_expert"] == 0.95
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_get_similarity_scores_no_routes(self, mock_embeddings_class):
        """Test getting similarity scores with no routes"""
        mock_embeddings_class.return_value = Mock()
        router = SemanticRouter()
        scores = router.get_similarity_scores("Some query")
        assert scores == {}
    
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_get_similarity_scores_error(self, mock_embeddings_class):
        """Test getting similarity scores with error"""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [0.1, 0.2, 0.3],  # First call succeeds
            [0.4, 0.5, 0.6],  # Second call succeeds
            [0.7, 0.8, 0.9],  # Third call succeeds
            Exception("Error")  # Query embedding fails
        ]
        mock_embeddings_class.return_value = mock_embeddings
        
        router = SemanticRouter(routes=self.sample_routes)
        scores = router.get_similarity_scores("Some query")
        assert scores == {}
    
    @patch('src.rag_engine.routing.semantic_router.SemanticRouter')
    def test_prompt_router_function(self, mock_router_class):
        """Test the legacy prompt_router function"""
        # Mock the router
        mock_router = Mock()
        mock_decision = Mock()
        mock_decision.selected_route = "physics_expert"
        mock_router.route.return_value = mock_decision
        mock_router.get_route_config.return_value = self.sample_routes[0]
        mock_router_class.return_value = mock_router
        
        input_dict = {"query": "What is quantum mechanics?"}
        result = prompt_router(input_dict, self.sample_routes)
        
        # Should return a PromptTemplate
        from langchain_core.prompts import PromptTemplate
        assert isinstance(result, PromptTemplate)
    
    @patch('src.rag_engine.routing.semantic_router.SemanticRouter')
    def test_prompt_router_function_fallback(self, mock_router_class):
        """Test the legacy prompt_router function with fallback"""
        # Mock the router to return a route without prompt_template
        mock_router = Mock()
        mock_decision = Mock()
        mock_decision.selected_route = "test_route"
        mock_router.route.return_value = mock_decision
        
        # Return a route config without prompt_template in metadata
        fallback_route = RouteConfig(name="test_route", description="test")
        mock_router.get_route_config.return_value = fallback_route
        mock_router_class.return_value = mock_router
        
        input_dict = {"query": "Some query"}
        result = prompt_router(input_dict, self.sample_routes)
        
        # Should return a fallback PromptTemplate
        from langchain_core.prompts import PromptTemplate
        assert isinstance(result, PromptTemplate)
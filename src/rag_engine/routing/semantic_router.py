"""
Semantic routing implementation using embedding-based similarity
"""

from typing import Dict, Any, List, Optional
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity

from ..core.interfaces import BaseRouter
from ..core.models import RoutingDecision, RouteConfig


class SemanticRouter(BaseRouter):
    """
    Semantic router that uses embedding-based similarity to route queries
    to the most semantically similar expert prompt/route.
    """
    
    def __init__(
        self,
        embedding_model: str = "models/embedding-001",
        routes: List[RouteConfig] = None
    ):
        """
        Initialize the semantic router.
        
        Args:
            embedding_model: Name of the Google embedding model to use
            routes: List of available routes/expert prompts
        """
        self.embedding_model = embedding_model
        self.routes: Dict[str, RouteConfig] = {}
        self.route_embeddings: Dict[str, List[float]] = {}
        
        # Initialize embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model
        )
        
        # Add default routes if provided
        if routes:
            for route in routes:
                self.add_route(route)
    
    def add_route(self, route_config: RouteConfig) -> bool:
        """Add a new routing destination with its embedding"""
        try:
            self.routes[route_config.name] = route_config
            
            # Generate embedding for the route description/template
            # Use description as the text to embed for semantic similarity
            route_text = route_config.description
            if route_config.keywords:
                # Include keywords in the embedding text for better matching
                route_text += " " + " ".join(route_config.keywords)
            
            embedding = self.embeddings.embed_query(route_text)
            self.route_embeddings[route_config.name] = embedding
            
            return True
        except Exception:
            return False
    
    def get_available_routes(self) -> List[str]:
        """Get list of available routes"""
        return list(self.routes.keys())
    
    def route(self, query: str) -> RoutingDecision:
        """Route a query to the most semantically similar destination"""
        if not self.routes:
            raise ValueError("No routes configured. Add routes before routing queries.")
        
        try:
            # Embed the incoming query
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate cosine similarity with all route embeddings
            similarities = {}
            route_embeddings_list = []
            route_names = []
            
            for route_name, route_embedding in self.route_embeddings.items():
                route_embeddings_list.append(route_embedding)
                route_names.append(route_name)
            
            # Calculate similarities using cosine similarity
            similarity_scores = cosine_similarity([query_embedding], route_embeddings_list)[0]
            
            # Find the most similar route
            max_similarity_idx = np.argmax(similarity_scores)
            best_route = route_names[max_similarity_idx]
            confidence = float(similarity_scores[max_similarity_idx])
            
            # Generate reasoning
            reasoning = f"Selected based on semantic similarity (score: {confidence:.3f})"
            
            return RoutingDecision(
                selected_route=best_route,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "router_type": "semantic",
                    "embedding_model": self.embedding_model,
                    "similarity_scores": {
                        route_names[i]: float(similarity_scores[i]) 
                        for i in range(len(route_names))
                    },
                    "available_routes": list(self.routes.keys())
                }
            )
            
        except Exception as e:
            # Return a default routing decision in case of error
            available_routes = list(self.routes.keys())
            if available_routes:
                return RoutingDecision(
                    selected_route=available_routes[0],
                    confidence=0.1,
                    reasoning=f"Error in semantic routing: {str(e)}. Using fallback route.",
                    metadata={
                        "router_type": "semantic",
                        "error": str(e),
                        "fallback": True
                    }
                )
            else:
                raise ValueError("No routes configured and routing failed")
    
    def get_route_config(self, route_name: str) -> RouteConfig:
        """Get configuration for a specific route"""
        return self.routes.get(route_name)
    
    def remove_route(self, route_name: str) -> bool:
        """Remove a route from the router"""
        if route_name in self.routes:
            del self.routes[route_name]
            if route_name in self.route_embeddings:
                del self.route_embeddings[route_name]
            return True
        return False
    
    def get_similarity_scores(self, query: str) -> Dict[str, float]:
        """Get similarity scores for all routes for debugging/analysis"""
        if not self.routes:
            return {}
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            similarities = {}
            route_embeddings_list = []
            route_names = []
            
            for route_name, route_embedding in self.route_embeddings.items():
                route_embeddings_list.append(route_embedding)
                route_names.append(route_name)
            
            similarity_scores = cosine_similarity([query_embedding], route_embeddings_list)[0]
            
            return {
                route_names[i]: float(similarity_scores[i]) 
                for i in range(len(route_names))
            }
        except Exception:
            return {}


def prompt_router(input_dict: Dict[str, Any], routes: List[RouteConfig]) -> PromptTemplate:
    """
    Legacy function to maintain compatibility with workplan examples.
    Routes the input query to the most similar prompt template.
    
    Args:
        input_dict: Dictionary containing the query
        routes: List of route configurations with prompt templates
        
    Returns:
        PromptTemplate: The most similar prompt template
    """
    # Create a semantic router
    router = SemanticRouter()
    
    # Add routes to the router
    for route in routes:
        router.add_route(route)
    
    # Route the query
    query = input_dict.get("query", "")
    decision = router.route(query)
    
    # Find the selected route and return its prompt template
    selected_route = router.get_route_config(decision.selected_route)
    if selected_route and "prompt_template" in selected_route.metadata:
        return PromptTemplate.from_template(selected_route.metadata["prompt_template"])
    
    # Fallback to a basic template
    return PromptTemplate.from_template("Answer the following question: {query}")
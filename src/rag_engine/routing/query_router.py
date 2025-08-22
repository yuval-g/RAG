"""
Unified query router that coordinates all routing strategies
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..core.interfaces import BaseRouter
from ..core.models import RoutingDecision, RouteConfig
from .logical_router import LogicalRouter
from .semantic_router import SemanticRouter
from .query_structurer import QueryStructurer


class RoutingStrategy(Enum):
    """Available routing strategies"""
    LOGICAL = "logical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    AUTO = "auto"


class QueryRouter(BaseRouter):
    """
    Unified query router that coordinates all routing strategies and provides
    a single interface for query routing with multiple backends.
    """
    
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.AUTO,
        llm_model: str = "gemini-2.0-flash-lite",
        embedding_model: str = "models/embedding-001",
        routes: List[RouteConfig] = None
    ):
        """
        Initialize the unified query router.
        
        Args:
            default_strategy: Default routing strategy to use
            llm_model: LLM model for logical routing
            embedding_model: Embedding model for semantic routing
            routes: Initial routes to configure
        """
        self.default_strategy = default_strategy
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Initialize individual routers
        self.logical_router = LogicalRouter(
            model_name=llm_model,
            routes=routes if routes else []
        )
        self.semantic_router = SemanticRouter(
            embedding_model=embedding_model,
            routes=routes if routes else []
        )
        
        # Initialize query structurer
        self.query_structurer = QueryStructurer(
            model_name=llm_model
        )
        
        # Route configuration
        self.routes: Dict[str, RouteConfig] = {}
        if routes:
            for route in routes:
                self.routes[route.name] = route
        
        # Strategy configuration
        self.strategy_config = {
            "logical_threshold": 0.7,  # Minimum confidence for logical routing
            "semantic_threshold": 0.6,  # Minimum confidence for semantic routing
            "hybrid_weight_logical": 0.6,  # Weight for logical in hybrid mode
            "hybrid_weight_semantic": 0.4,  # Weight for semantic in hybrid mode
        }
    
    def add_route(self, route_config: RouteConfig) -> bool:
        """Add a new routing destination to all routers"""
        try:
            # Add to all individual routers
            logical_success = self.logical_router.add_route(route_config)
            semantic_success = self.semantic_router.add_route(route_config)
            
            if logical_success and semantic_success:
                self.routes[route_config.name] = route_config
                return True
            return False
        except Exception:
            return False
    
    def get_available_routes(self) -> List[str]:
        """Get list of available routes"""
        return list(self.routes.keys())
    
    def remove_route(self, route_name: str) -> bool:
        """Remove a route from all routers"""
        try:
            logical_success = self.logical_router.remove_route(route_name)
            semantic_success = self.semantic_router.remove_route(route_name)
            
            if route_name in self.routes:
                del self.routes[route_name]
            
            return logical_success or semantic_success
        except Exception:
            return False
    
    def route(self, query: str, strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """
        Route a query using the specified or default strategy.
        
        Args:
            query: Query to route
            strategy: Routing strategy to use (overrides default)
            
        Returns:
            RoutingDecision: Routing decision with metadata
        """
        if not self.routes:
            raise ValueError("No routes configured. Add routes before routing queries.")
        
        # Use provided strategy or default
        routing_strategy = strategy or self.default_strategy
        
        try:
            if routing_strategy == RoutingStrategy.LOGICAL:
                return self._route_logical(query)
            elif routing_strategy == RoutingStrategy.SEMANTIC:
                return self._route_semantic(query)
            elif routing_strategy == RoutingStrategy.HYBRID:
                return self._route_hybrid(query)
            elif routing_strategy == RoutingStrategy.AUTO:
                return self._route_auto(query)
            else:
                raise ValueError(f"Unknown routing strategy: {routing_strategy}")
                
        except Exception as e:
            # Fallback to first available route
            available_routes = list(self.routes.keys())
            if available_routes:
                return RoutingDecision(
                    selected_route=available_routes[0],
                    confidence=0.1,
                    reasoning=f"Error in routing: {str(e)}. Using fallback route.",
                    metadata={
                        "router_type": "unified",
                        "strategy": str(routing_strategy),
                        "error": str(e),
                        "fallback": True
                    }
                )
            else:
                raise ValueError("No routes configured and routing failed")
    
    def _route_logical(self, query: str) -> RoutingDecision:
        """Route using logical router"""
        decision = self.logical_router.route(query)
        decision.metadata["unified_strategy"] = "logical"
        return decision
    
    def _route_semantic(self, query: str) -> RoutingDecision:
        """Route using semantic router"""
        decision = self.semantic_router.route(query)
        decision.metadata["unified_strategy"] = "semantic"
        return decision
    
    def _route_hybrid(self, query: str) -> RoutingDecision:
        """Route using hybrid approach combining logical and semantic"""
        try:
            # Get decisions from both routers
            logical_decision = self.logical_router.route(query)
            semantic_decision = self.semantic_router.route(query)
            
            # Calculate weighted scores
            logical_weight = self.strategy_config["hybrid_weight_logical"]
            semantic_weight = self.strategy_config["hybrid_weight_semantic"]
            
            # Get all unique routes and their scores
            route_scores = {}
            
            # Add logical scores
            route_scores[logical_decision.selected_route] = (
                logical_decision.confidence * logical_weight
            )
            
            # Add semantic scores
            if semantic_decision.selected_route in route_scores:
                route_scores[semantic_decision.selected_route] += (
                    semantic_decision.confidence * semantic_weight
                )
            else:
                route_scores[semantic_decision.selected_route] = (
                    semantic_decision.confidence * semantic_weight
                )
            
            # Find best route
            best_route = max(route_scores.keys(), key=lambda k: route_scores[k])
            best_score = route_scores[best_route]
            
            # Create reasoning
            reasoning = (
                f"Hybrid routing: logical={logical_decision.selected_route}"
                f"({logical_decision.confidence:.3f}), "
                f"semantic={semantic_decision.selected_route}"
                f"({semantic_decision.confidence:.3f}), "
                f"final={best_route}({best_score:.3f})"
            )
            
            return RoutingDecision(
                selected_route=best_route,
                confidence=best_score,
                reasoning=reasoning,
                metadata={
                    "router_type": "unified",
                    "unified_strategy": "hybrid",
                    "logical_decision": {
                        "route": logical_decision.selected_route,
                        "confidence": logical_decision.confidence
                    },
                    "semantic_decision": {
                        "route": semantic_decision.selected_route,
                        "confidence": semantic_decision.confidence
                    },
                    "route_scores": route_scores,
                    "weights": {
                        "logical": logical_weight,
                        "semantic": semantic_weight
                    }
                }
            )
            
        except Exception as e:
            # Fallback to logical routing
            return self._route_logical(query)
    
    def _route_auto(self, query: str) -> RoutingDecision:
        """Route using automatic strategy selection based on query characteristics"""
        try:
            # Analyze query complexity and characteristics
            analysis = self.query_structurer.analyze_query_complexity(query)
            
            # Decision logic for strategy selection
            if analysis["complexity_score"] >= 2:
                # Complex queries with multiple filters -> use logical routing
                strategy = RoutingStrategy.LOGICAL
                reasoning_prefix = "Auto-selected logical routing due to query complexity"
            elif len(query.split()) <= 3:
                # Short queries -> use semantic routing
                strategy = RoutingStrategy.SEMANTIC
                reasoning_prefix = "Auto-selected semantic routing for short query"
            elif any(keyword in query.lower() for keyword in [
                "similar", "like", "related", "about", "regarding"
            ]):
                # Semantic similarity queries -> use semantic routing
                strategy = RoutingStrategy.SEMANTIC
                reasoning_prefix = "Auto-selected semantic routing for similarity query"
            else:
                # Default to hybrid for balanced approach
                strategy = RoutingStrategy.HYBRID
                reasoning_prefix = "Auto-selected hybrid routing for balanced approach"
            
            # Route using selected strategy
            if strategy == RoutingStrategy.LOGICAL:
                decision = self._route_logical(query)
            elif strategy == RoutingStrategy.SEMANTIC:
                decision = self._route_semantic(query)
            else:  # HYBRID
                decision = self._route_hybrid(query)
            
            # Update metadata and reasoning
            decision.reasoning = f"{reasoning_prefix}. {decision.reasoning}"
            decision.metadata["unified_strategy"] = "auto"
            decision.metadata["auto_selected_strategy"] = str(strategy)
            decision.metadata["query_analysis"] = analysis
            
            return decision
            
        except Exception as e:
            # Fallback to logical routing
            return self._route_logical(query)
    
    def get_routing_strategies(self) -> List[str]:
        """Get list of available routing strategies"""
        return [strategy.value for strategy in RoutingStrategy]
    
    def set_strategy_config(self, config: Dict[str, Any]) -> None:
        """Update strategy configuration"""
        self.strategy_config.update(config)
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration"""
        return self.strategy_config.copy()
    
    def get_route_config(self, route_name: str) -> Optional[RouteConfig]:
        """Get configuration for a specific route"""
        return self.routes.get(route_name)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and provide routing recommendations.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with routing recommendations
        """
        try:
            # Get query complexity analysis
            complexity_analysis = self.query_structurer.analyze_query_complexity(query)
            
            # Get routing decisions from all strategies
            logical_decision = self.logical_router.route(query)
            semantic_decision = self.semantic_router.route(query)
            
            # Get semantic similarity scores
            similarity_scores = self.semantic_router.get_similarity_scores(query)
            
            return {
                "query": query,
                "complexity_analysis": complexity_analysis,
                "routing_recommendations": {
                    "logical": {
                        "route": logical_decision.selected_route,
                        "confidence": logical_decision.confidence,
                        "reasoning": logical_decision.reasoning
                    },
                    "semantic": {
                        "route": semantic_decision.selected_route,
                        "confidence": semantic_decision.confidence,
                        "reasoning": semantic_decision.reasoning
                    }
                },
                "similarity_scores": similarity_scores,
                "recommended_strategy": self._recommend_strategy(
                    complexity_analysis, logical_decision, semantic_decision
                )
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "recommended_strategy": "logical"
            }
    
    def _recommend_strategy(
        self, 
        complexity_analysis: Dict[str, Any],
        logical_decision: RoutingDecision,
        semantic_decision: RoutingDecision
    ) -> str:
        """Recommend the best strategy based on analysis"""
        
        # High confidence logical routing
        if logical_decision.confidence >= self.strategy_config["logical_threshold"]:
            return "logical"
        
        # High confidence semantic routing
        if semantic_decision.confidence >= self.strategy_config["semantic_threshold"]:
            return "semantic"
        
        # Complex queries favor logical
        if complexity_analysis["complexity_score"] >= 2:
            return "logical"
        
        # Similar confidence scores favor hybrid
        confidence_diff = abs(logical_decision.confidence - semantic_decision.confidence)
        if confidence_diff < 0.2:
            return "hybrid"
        
        # Default to auto for adaptive behavior
        return "auto"
    
    def bulk_route(
        self, 
        queries: List[str], 
        strategy: Optional[RoutingStrategy] = None
    ) -> List[RoutingDecision]:
        """
        Route multiple queries efficiently.
        
        Args:
            queries: List of queries to route
            strategy: Routing strategy to use for all queries
            
        Returns:
            List[RoutingDecision]: List of routing decisions
        """
        return [self.route(query, strategy) for query in queries]
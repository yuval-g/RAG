"""
Logical routing implementation using structured LLM output
"""

from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from ..core.interfaces import BaseRouter
from ..core.models import RoutingDecision, RouteConfig


class RouteQuery(BaseModel):
    """A data model to route a user query to the most relevant datasource."""
    
    datasource: str = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question.",
    )
    confidence: float = Field(
        default=0.8,
        description="Confidence score for the routing decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this datasource was chosen"
    )


class LogicalRouter(BaseRouter):
    """
    Logical router that uses LLM with structured output to classify queries
    into predefined categories/datasources.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        routes: List[RouteConfig] = None
    ):
        """
        Initialize the logical router.
        
        Args:
            model_name: Name of the Google Gemini model to use
            temperature: Temperature for LLM generation
            routes: List of available routes/datasources
        """
        self.model_name = model_name
        self.temperature = temperature
        self.routes: Dict[str, RouteConfig] = {}
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        # Add default routes if provided
        if routes:
            for route in routes:
                self.add_route(route)
        
        self._build_router_chain()
    
    def add_route(self, route_config: RouteConfig) -> bool:
        """Add a new routing destination"""
        try:
            self.routes[route_config.name] = route_config
            self._build_router_chain()  # Rebuild chain with new routes
            return True
        except Exception:
            return False
    
    def get_available_routes(self) -> List[str]:
        """Get list of available routes"""
        return list(self.routes.keys())
    
    def _build_router_chain(self):
        """Build the routing chain with current routes"""
        if not self.routes:
            # No routes configured yet
            self.router_chain = None
            return
        
        # Create dynamic RouteQuery model with current routes
        route_names = list(self.routes.keys())
        
        # Create a new Pydantic model with Literal type for current routes
        class DynamicRouteQuery(BaseModel):
            datasource: str = Field(
                ...,
                description=f"Choose from: {', '.join(route_names)}. Select the most relevant datasource for the user question."
            )
            confidence: float = Field(
                default=0.8,
                description="Confidence score for the routing decision (0.0 to 1.0)",
                ge=0.0,
                le=1.0
            )
            reasoning: str = Field(
                default="",
                description="Brief explanation of why this datasource was chosen"
            )
        
        # Create structured LLM
        structured_llm = self.llm.with_structured_output(DynamicRouteQuery)
        
        # Build route descriptions for the prompt
        route_descriptions = []
        for name, config in self.routes.items():
            desc = f"- {name}: {config.description}"
            if config.keywords:
                desc += f" (Keywords: {', '.join(config.keywords)})"
            route_descriptions.append(desc)
        
        # Create system prompt
        system_prompt = f"""You are an expert at routing user questions to the appropriate data source.

Available data sources:
{chr(10).join(route_descriptions)}

Based on the content and context of the question, route it to the most relevant data source.
Provide a confidence score and brief reasoning for your decision."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        # Build the complete router chain
        self.router_chain = prompt | structured_llm
    
    def route(self, query: str) -> RoutingDecision:
        """Route a query to appropriate destination"""
        if not self.router_chain:
            raise ValueError("No routes configured. Add routes before routing queries.")
        
        try:
            # Get routing decision from LLM
            result = self.router_chain.invoke({"question": query})
            
            # Validate that the selected datasource exists
            if result.datasource not in self.routes:
                # Fallback to first available route if LLM returns invalid route
                available_routes = list(self.routes.keys())
                if available_routes:
                    return RoutingDecision(
                        selected_route=available_routes[0],
                        confidence=0.5,
                        reasoning=f"Fallback to {available_routes[0]} due to invalid route selection",
                        metadata={
                            "router_type": "logical",
                            "model_used": self.model_name,
                            "available_routes": available_routes,
                            "original_selection": result.datasource
                        }
                    )
                else:
                    raise ValueError("No valid routes available")
            
            return RoutingDecision(
                selected_route=result.datasource,
                confidence=result.confidence,
                reasoning=result.reasoning,
                metadata={
                    "router_type": "logical",
                    "model_used": self.model_name,
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
                    reasoning=f"Error in routing: {str(e)}. Using fallback route.",
                    metadata={
                        "router_type": "logical",
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
            self._build_router_chain()
            return True
        return False
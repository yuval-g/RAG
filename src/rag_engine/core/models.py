"""
Core data models for the RAG system
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any


class Document(BaseModel):
    """Core document model for the RAG system"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None


class ProcessedQuery(BaseModel):
    """Processed query with transformations and metadata"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    original_query: str
    transformed_queries: List[str] = Field(default_factory=list)
    strategy_used: str = "basic"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Response from the RAG system with metadata"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    answer: str
    source_documents: List[Document] = Field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Results from evaluation framework"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    overall_score: float
    metric_scores: Dict[str, float] = Field(default_factory=dict)
    test_case_results: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EvaluationTestCase(BaseModel):
    """Test case for evaluation"""
    __test__ = False
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    question: str
    expected_answer: str
    context: List[Document] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Keep TestCase as an alias for backward compatibility
TestCase = EvaluationTestCase


class RoutingDecision(BaseModel):
    """Decision from routing system"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    selected_route: str
    confidence: float
    reasoning: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouteConfig(BaseModel):
    """Configuration for a routing destination"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
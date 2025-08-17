"""
API request/response models for the RAG engine
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from ..core.models import Document, RAGResponse, EvaluationResult, EvaluationTestCase


class QueryRequest(BaseModel):
    """Request model for query processing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    question: str = Field(..., description="The question to ask the RAG system", min_length=1)
    k: Optional[int] = Field(default=None, description="Number of documents to retrieve", ge=1, le=100)
    include_sources: Optional[bool] = Field(default=True, description="Whether to include source documents in response")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the query")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    answer: str = Field(..., description="The generated answer")
    source_documents: List[Document] = Field(default_factory=list, description="Source documents used for the answer")
    confidence_score: float = Field(..., description="Confidence score for the answer", ge=0.0, le=1.0)
    processing_time: float = Field(..., description="Time taken to process the query in seconds", ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")


class DocumentInput(BaseModel):
    """Input model for a single document"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str = Field(..., description="The document content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID")


class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    documents: List[DocumentInput] = Field(..., description="List of documents to ingest", min_length=1)
    clear_existing: Optional[bool] = Field(default=False, description="Whether to clear existing documents before ingestion")


class WebDocumentIngestionRequest(BaseModel):
    """Request model for web document ingestion"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    urls: List[str] = Field(..., description="List of URLs to load documents from", min_length=1)
    clear_existing: Optional[bool] = Field(default=False, description="Whether to clear existing documents before ingestion")
    max_depth: Optional[int] = Field(default=1, description="Maximum crawling depth", ge=1, le=3)
    include_links: Optional[bool] = Field(default=False, description="Whether to include links in the content")


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool = Field(..., description="Whether the ingestion was successful")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(..., description="Number of documents processed", ge=0)
    chunks_created: int = Field(..., description="Number of chunks created", ge=0)
    processing_time: float = Field(..., description="Time taken to process documents in seconds", ge=0.0)


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    test_cases: List[EvaluationTestCase] = Field(..., description="List of test cases for evaluation", min_length=1)
    frameworks: Optional[List[str]] = Field(default=["custom"], description="Evaluation frameworks to use")
    metrics: Optional[List[str]] = Field(default_factory=list, description="Specific metrics to evaluate")


class EvaluationResponse(BaseModel):
    """Response model for evaluation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool = Field(..., description="Whether the evaluation was successful")
    result: Optional[EvaluationResult] = Field(default=None, description="Evaluation results")
    message: str = Field(..., description="Status message")
    processing_time: float = Field(..., description="Time taken for evaluation in seconds", ge=0.0)


class SystemInfoResponse(BaseModel):
    """Response model for system information"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    version: str = Field(..., description="System version")
    config: Dict[str, Any] = Field(..., description="Current configuration")
    components: Dict[str, bool] = Field(..., description="Component status")
    stats: Dict[str, Any] = Field(..., description="System statistics")


class HealthResponse(BaseModel):
    """Response model for health check"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    version: str = Field(..., description="System version")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime: float = Field(..., description="System uptime in seconds", ge=0.0)


class ErrorResponse(BaseModel):
    """Response model for errors"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
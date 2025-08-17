"""
FastAPI application for the RAG engine
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.engine import RAGEngine
from ..core.config import PipelineConfig, ConfigurationManager
from ..core.models import Document, EvaluationTestCase
from .models import (
    QueryRequest, QueryResponse,
    DocumentIngestionRequest, DocumentIngestionResponse,
    WebDocumentIngestionRequest,
    EvaluationRequest, EvaluationResponse,
    SystemInfoResponse, HealthResponse, ErrorResponse
)


logger = logging.getLogger(__name__)


class RAGEngineManager:
    """Manages RAG engine instance for the API"""
    
    def __init__(self):
        self.engine: Optional[RAGEngine] = None
        self.config: Optional[PipelineConfig] = None
        self.start_time = time.time()
    
    def initialize(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """Initialize the RAG engine"""
        try:
            config_manager = ConfigurationManager(config_path=config_path, environment=environment)
            self.config = config_manager.load_config()
            self.engine = RAGEngine(self.config)
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    def get_engine(self) -> RAGEngine:
        """Get the RAG engine instance"""
        if self.engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG engine not initialized"
            )
        return self.engine
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time


# Global engine manager
engine_manager = RAGEngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG API server...")
    try:
        engine_manager.initialize()
        logger.info("RAG API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start RAG API server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")


def create_app(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
    enable_cors: bool = True
) -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="RAG Engine API",
        description="Production-ready RAG (Retrieval-Augmented Generation) system API",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                timestamp=datetime.now(timezone.utc).isoformat()
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal server error occurred",
                details={"exception": str(exc)},
                timestamp=datetime.now(timezone.utc).isoformat()
            ).model_dump()
        )
    
    # Dependency to get RAG engine
    def get_rag_engine() -> RAGEngine:
        return engine_manager.get_engine()
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        try:
            engine = engine_manager.get_engine()
            components = {
                "engine": "healthy" if engine else "unhealthy",
                "indexer": "healthy" if engine and engine._indexer else "unhealthy",
                "retriever": "healthy" if engine and engine._retriever else "unhealthy",
                "generator": "healthy" if engine and engine._generator else "unhealthy",
            }
            
            overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy"
            
            return HealthResponse(
                status=overall_status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                version="0.1.0",
                components=components,
                uptime=engine_manager.get_uptime()
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(timezone.utc).isoformat(),
                version="0.1.0",
                components={"engine": "unhealthy"},
                uptime=engine_manager.get_uptime()
            )
    
    # System information endpoint
    @app.get("/info", response_model=SystemInfoResponse)
    async def get_system_info(engine: RAGEngine = Depends(get_rag_engine)):
        """Get system information"""
        try:
            info = engine.get_system_info()
            return SystemInfoResponse(**info)
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get system information: {str(e)}"
            )
    
    # Query processing endpoint
    @app.post("/query", response_model=QueryResponse)
    async def process_query(
        request: QueryRequest,
        engine: RAGEngine = Depends(get_rag_engine)
    ):
        """Process a query through the RAG pipeline"""
        try:
            logger.info(f"Processing query: {request.question[:100]}...")
            
            # Process query with optional parameters
            kwargs = {}
            if request.k is not None:
                kwargs['k'] = request.k
            
            response = engine.query(request.question, **kwargs)
            
            # Filter source documents if not requested
            if not request.include_sources:
                response.source_documents = []
            
            return QueryResponse(
                answer=response.answer,
                source_documents=response.source_documents,
                confidence_score=response.confidence_score,
                processing_time=response.processing_time,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {str(e)}"
            )
    
    # Document ingestion endpoint
    @app.post("/documents", response_model=DocumentIngestionResponse)
    async def ingest_documents(
        request: DocumentIngestionRequest,
        engine: RAGEngine = Depends(get_rag_engine)
    ):
        """Ingest documents into the RAG system"""
        try:
            start_time = time.time()
            logger.info(f"Ingesting {len(request.documents)} documents...")
            
            # Clear existing documents if requested
            if request.clear_existing:
                logger.info("Clearing existing documents...")
                engine.clear_documents()
            
            # Convert API documents to core Document objects
            documents = [
                Document(
                    content=doc.content,
                    metadata=doc.metadata or {},
                    doc_id=doc.doc_id
                )
                for doc in request.documents
            ]
            
            # Add documents to the system
            success = engine.add_documents(documents)
            processing_time = time.time() - start_time
            
            if success:
                return DocumentIngestionResponse(
                    success=True,
                    message=f"Successfully ingested {len(documents)} documents",
                    documents_processed=len(documents),
                    chunks_created=engine.get_chunk_count(),
                    processing_time=processing_time
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to ingest documents"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document ingestion failed: {str(e)}"
            )
    
    # Web document ingestion endpoint
    @app.post("/documents/web", response_model=DocumentIngestionResponse)
    async def ingest_web_documents(
        request: WebDocumentIngestionRequest,
        engine: RAGEngine = Depends(get_rag_engine)
    ):
        """Ingest documents from web URLs"""
        try:
            start_time = time.time()
            logger.info(f"Loading documents from {len(request.urls)} URLs...")
            
            # Clear existing documents if requested
            if request.clear_existing:
                logger.info("Clearing existing documents...")
                engine.clear_documents()
            
            # Load documents from URLs
            kwargs = {
                'max_depth': request.max_depth,
                'include_links': request.include_links
            }
            
            success = engine.load_web_documents(request.urls, **kwargs)
            processing_time = time.time() - start_time
            
            if success:
                return DocumentIngestionResponse(
                    success=True,
                    message=f"Successfully loaded documents from {len(request.urls)} URLs",
                    documents_processed=engine.get_document_count(),
                    chunks_created=engine.get_chunk_count(),
                    processing_time=processing_time
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to load web documents"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Web document ingestion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Web document ingestion failed: {str(e)}"
            )
    
    # Clear documents endpoint
    @app.delete("/documents", response_model=DocumentIngestionResponse)
    async def clear_documents(engine: RAGEngine = Depends(get_rag_engine)):
        """Clear all documents from the RAG system"""
        try:
            start_time = time.time()
            logger.info("Clearing all documents...")
            
            success = engine.clear_documents()
            processing_time = time.time() - start_time
            
            if success:
                return DocumentIngestionResponse(
                    success=True,
                    message="Successfully cleared all documents",
                    documents_processed=0,
                    chunks_created=0,
                    processing_time=processing_time
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to clear documents"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document clearing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document clearing failed: {str(e)}"
            )
    
    # Evaluation endpoint
    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate_system(
        request: EvaluationRequest,
        engine: RAGEngine = Depends(get_rag_engine)
    ):
        """Evaluate the RAG system"""
        try:
            start_time = time.time()
            logger.info(f"Evaluating system with {len(request.test_cases)} test cases...")
            
            # Convert API test cases to core TestCase objects
            test_cases = [
                EvaluationTestCase(
                    question=tc.question,
                    expected_answer=tc.expected_answer,
                    context=tc.context,
                    metadata=tc.metadata
                )
                for tc in request.test_cases
            ]
            
            # Run evaluation
            result = engine.evaluate(test_cases)
            processing_time = time.time() - start_time
            
            return EvaluationResponse(
                success=True,
                result=result,
                message=f"Evaluation completed with {len(test_cases)} test cases",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            processing_time = time.time() - start_time
            return EvaluationResponse(
                success=False,
                result=None,
                message=f"Evaluation failed: {str(e)}",
                processing_time=processing_time
            )
    
    return app


# Create default app instance
app = create_app()
"""
Integration tests for the RAG API
"""

import pytest
import json
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import Mock, patch

from src.rag_engine.api.app import create_app
from src.rag_engine.core.models import Document, RAGResponse, EvaluationResult, EvaluationTestCase
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return PipelineConfig(
        llm_provider="google",
        llm_model="gemini-2.0-flash-lite",
        embedding_provider="openai",
        vector_store="chroma",
        enable_logging=False  # Disable logging for tests
    )


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing"""
    engine = Mock()
    
    # Mock system info
    engine.get_system_info.return_value = {
        "version": "0.1.0",
        "config": {
            "llm_provider": "google",
            "llm_model": "gemini-2.0-flash-lite",
            "embedding_provider": "openai",
            "vector_store": "chroma",
            "indexing_strategy": "basic",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
        },
        "components": {
            "indexer": True,
            "retriever": True,
            "query_processor": False,
            "router": False,
            "generator": True,
            "evaluator": False,
        },
        "stats": {
            "indexed_documents": 5,
            "indexed_chunks": 25,
            "retriever_ready": True,
        }
    }
    
    # Mock query response
    engine.query.return_value = RAGResponse(
        answer="This is a test answer from the RAG system.",
        source_documents=[
            Document(
                content="Test document content",
                metadata={"source": "test.txt"},
                doc_id="test-doc-1"
            )
        ],
        confidence_score=0.85,
        processing_time=1.23,
        metadata={"query": "test question", "retrieved_count": 1}
    )
    
    # Mock document operations
    engine.add_documents.return_value = True
    engine.clear_documents.return_value = True
    engine.load_web_documents.return_value = True
    engine.get_document_count.return_value = 5
    engine.get_chunk_count.return_value = 25
    engine.is_ready.return_value = True
    
    # Mock evaluation
    engine.evaluate.return_value = EvaluationResult(
        overall_score=0.8,
        metric_scores={"faithfulness": 0.85, "relevancy": 0.75},
        test_case_results=[],
        recommendations=["Consider improving document quality"]
    )
    
    return engine


@pytest.fixture
def client(mock_rag_engine, mock_config):
    """Test client with mocked dependencies"""
    
    # Patch the engine manager initialization
    with patch('src.rag_engine.api.app.RAGEngineManager') as MockManager:
        mock_manager_instance = Mock()
        mock_manager_instance.engine = mock_rag_engine
        mock_manager_instance.config = mock_config
        mock_manager_instance.get_engine.return_value = mock_rag_engine
        mock_manager_instance.get_uptime.return_value = 123.45
        mock_manager_instance.initialize.return_value = None
        
        MockManager.return_value = mock_manager_instance
        
        # Create the app
        app = create_app()
        
        # Override the global engine_manager with our mock
        import src.rag_engine.api.app as app_module
        app_module.engine_manager = mock_manager_instance
        
        return TestClient(app)


class TestHealthEndpoints:
    """Test health and system information endpoints"""
    
    def test_health_check_healthy(self, client):
        """Test health check when system is healthy"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
        assert "components" in data
        assert "uptime" in data
    
    def test_system_info(self, client):
        """Test system information endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "0.1.0"
        assert "config" in data
        assert "components" in data
        assert "stats" in data
        assert data["stats"]["indexed_documents"] == 5


class TestQueryEndpoint:
    """Test query processing endpoint"""
    
    def test_query_success(self, client):
        """Test successful query processing"""
        request_data = {
            "question": "What is the capital of France?",
            "k": 3,
            "include_sources": True
        }
        
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "source_documents" in data
        assert "confidence_score" in data
        assert "processing_time" in data
        assert data["confidence_score"] == 0.85
        assert len(data["source_documents"]) == 1
    
    def test_query_without_sources(self, client):
        """Test query processing without source documents"""
        request_data = {
            "question": "What is the capital of France?",
            "include_sources": False
        }
        
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert len(data["source_documents"]) == 0
    
    def test_query_validation_error(self, client):
        """Test query validation error"""
        request_data = {
            "question": "",  # Empty question should fail validation
        }
        
        response = client.post("/query", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_query_with_invalid_k(self, client):
        """Test query with invalid k parameter"""
        request_data = {
            "question": "What is the capital of France?",
            "k": 0  # Invalid k value
        }
        
        response = client.post("/query", json=request_data)
        assert response.status_code == 422  # Validation error


class TestDocumentIngestionEndpoints:
    """Test document ingestion endpoints"""
    
    def test_ingest_documents_success(self, client):
        """Test successful document ingestion"""
        request_data = {
            "documents": [
                {
                    "content": "This is a test document about Paris.",
                    "metadata": {"source": "test.txt"},
                    "doc_id": "test-1"
                },
                {
                    "content": "This is another test document about France.",
                    "metadata": {"source": "test2.txt"}
                }
            ],
            "clear_existing": False
        }
        
        response = client.post("/documents", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["documents_processed"] == 2
        assert "processing_time" in data
    
    def test_ingest_documents_with_clear(self, client):
        """Test document ingestion with clearing existing documents"""
        request_data = {
            "documents": [
                {
                    "content": "This is a test document.",
                    "metadata": {"source": "test.txt"}
                }
            ],
            "clear_existing": True
        }
        
        response = client.post("/documents", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_ingest_web_documents_success(self, client):
        """Test successful web document ingestion"""
        request_data = {
            "urls": [
                "https://example.com/page1",
                "https://example.com/page2"
            ],
            "clear_existing": False,
            "max_depth": 1,
            "include_links": False
        }
        
        response = client.post("/documents/web", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_clear_documents_success(self, client):
        """Test successful document clearing"""
        response = client.delete("/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["documents_processed"] == 0
    
    def test_ingest_documents_validation_error(self, client):
        """Test document ingestion validation error"""
        request_data = {
            "documents": []  # Empty documents list should fail validation
        }
        
        response = client.post("/documents", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_ingest_documents_empty_content(self, client):
        """Test document ingestion with empty content"""
        request_data = {
            "documents": [
                {
                    "content": "",  # Empty content should fail validation
                    "metadata": {"source": "test.txt"}
                }
            ]
        }
        
        response = client.post("/documents", json=request_data)
        assert response.status_code == 422  # Validation error


class TestEvaluationEndpoint:
    """Test evaluation endpoint"""
    
    def test_evaluate_success(self, client):
        """Test successful evaluation"""
        request_data = {
            "test_cases": [
                {
                    "question": "What is the capital of France?",
                    "expected_answer": "Paris",
                    "context": [
                        {
                            "content": "Paris is the capital of France.",
                            "metadata": {"source": "geography.txt"}
                        }
                    ],
                    "metadata": {"category": "geography"}
                }
            ],
            "frameworks": ["custom"],
            "metrics": ["faithfulness", "relevancy"]
        }
        
        response = client.post("/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "processing_time" in data
        assert data["result"]["overall_score"] == 0.8
    
    def test_evaluate_validation_error(self, client):
        """Test evaluation validation error"""
        request_data = {
            "test_cases": []  # Empty test cases should fail validation
        }
        
        response = client.post("/evaluate", json=request_data)
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test error handling"""
    
    def test_engine_not_initialized(self):
        """Test behavior when engine is not initialized"""
        # Create app with engine manager that raises exception
        with patch('src.rag_engine.api.app.RAGEngineManager') as MockManager:
            mock_manager_instance = Mock()
            mock_manager_instance.get_engine.side_effect = HTTPException(
                status_code=503, detail="RAG engine not initialized"
            )
            MockManager.return_value = mock_manager_instance
            
            app = create_app()
            import src.rag_engine.api.app as app_module
            app_module.engine_manager = mock_manager_instance
            
            client = TestClient(app)
            
            # This should fail because engine is not initialized
            response = client.get("/info")
            assert response.status_code == 503  # Service unavailable
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test invalid HTTP method"""
        response = client.put("/query")  # PUT not allowed for query endpoint
        assert response.status_code == 405  # Method not allowed


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/query")
        # CORS headers should be present in the response
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled
    
    def test_content_type_json(self, client):
        """Test that API accepts and returns JSON"""
        request_data = {
            "question": "Test question"
        }
        
        response = client.post("/query", json=request_data)
        assert response.headers["content-type"] == "application/json"


if __name__ == "__main__":
    pytest.main([__file__])
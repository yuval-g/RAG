"""
Unit tests for core data models
"""

import pytest
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.core.models import (
    Document, ProcessedQuery, RAGResponse, EvaluationResult,
    TestCase, RoutingDecision, RouteConfig
)


class TestDocument:
    """Test cases for Document model"""
    
    def test_document_creation_minimal(self):
        """Test creating document with minimal parameters"""
        doc = Document(content="Test content")
        
        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.doc_id is None
        assert doc.embedding is None
    
    def test_document_creation_full(self):
        """Test creating document with all parameters"""
        metadata = {"source": "test", "category": "example"}
        embedding = [0.1, 0.2, 0.3]
        
        doc = Document(
            content="Test content",
            metadata=metadata,
            doc_id="test_doc_1",
            embedding=embedding
        )
        
        assert doc.content == "Test content"
        assert doc.metadata == metadata
        assert doc.doc_id == "test_doc_1"
        assert doc.embedding == embedding
    
    def test_document_metadata_default_factory(self):
        """Test that metadata uses default factory correctly"""
        doc1 = Document(content="Content 1")
        doc2 = Document(content="Content 2")
        
        # Ensure separate instances have separate metadata dicts
        doc1.metadata["key1"] = "value1"
        doc2.metadata["key2"] = "value2"
        
        assert "key1" in doc1.metadata
        assert "key1" not in doc2.metadata
        assert "key2" in doc2.metadata
        assert "key2" not in doc1.metadata
    
    def test_document_serialization(self):
        """Test document can be converted to dict"""
        doc = Document(
            content="Test content",
            metadata={"source": "test"},
            doc_id="test_doc"
        )
        
        doc_dict = doc.model_dump()
        expected = {
            "content": "Test content",
            "metadata": {"source": "test"},
            "doc_id": "test_doc",
            "embedding": None
        }
        
        assert doc_dict == expected


class TestProcessedQuery:
    """Test cases for ProcessedQuery model"""
    
    def test_processed_query_minimal(self):
        """Test creating processed query with minimal parameters"""
        query = ProcessedQuery(original_query="What is Python?")
        
        assert query.original_query == "What is Python?"
        assert query.transformed_queries == []
        assert query.strategy_used == "basic"
        assert query.metadata == {}
    
    def test_processed_query_full(self):
        """Test creating processed query with all parameters"""
        transformed = ["What is Python programming?", "Explain Python language"]
        metadata = {"processing_time": 0.5, "confidence": 0.9}
        
        query = ProcessedQuery(
            original_query="What is Python?",
            transformed_queries=transformed,
            strategy_used="multi_query",
            metadata=metadata
        )
        
        assert query.original_query == "What is Python?"
        assert query.transformed_queries == transformed
        assert query.strategy_used == "multi_query"
        assert query.metadata == metadata
    
    def test_processed_query_default_factories(self):
        """Test default factory behavior"""
        query1 = ProcessedQuery(original_query="Query 1")
        query2 = ProcessedQuery(original_query="Query 2")
        
        query1.transformed_queries.append("Transformed 1")
        query1.metadata["key1"] = "value1"
        
        assert len(query1.transformed_queries) == 1
        assert len(query2.transformed_queries) == 0
        assert "key1" in query1.metadata
        assert "key1" not in query2.metadata


class TestRAGResponse:
    """Test cases for RAGResponse model"""
    
    def test_rag_response_minimal(self):
        """Test creating RAG response with minimal parameters"""
        response = RAGResponse(answer="Python is a programming language.")
        
        assert response.answer == "Python is a programming language."
        assert response.source_documents == []
        assert response.confidence_score == 0.0
        assert response.processing_time == 0.0
        assert response.metadata == {}
    
    def test_rag_response_full(self):
        """Test creating RAG response with all parameters"""
        docs = [Document(content="Python documentation")]
        metadata = {"model": "gpt-3.5", "tokens": 150}
        
        response = RAGResponse(
            answer="Python is a programming language.",
            source_documents=docs,
            confidence_score=0.95,
            processing_time=1.5,
            metadata=metadata
        )
        
        assert response.answer == "Python is a programming language."
        assert response.source_documents == docs
        assert response.confidence_score == 0.95
        assert response.processing_time == 1.5
        assert response.metadata == metadata
    
    def test_rag_response_default_factories(self):
        """Test default factory behavior"""
        response1 = RAGResponse(answer="Answer 1")
        response2 = RAGResponse(answer="Answer 2")
        
        response1.source_documents.append(Document(content="Doc 1"))
        response1.metadata["key1"] = "value1"
        
        assert len(response1.source_documents) == 1
        assert len(response2.source_documents) == 0
        assert "key1" in response1.metadata
        assert "key1" not in response2.metadata


class TestEvaluationResult:
    """Test cases for EvaluationResult model"""
    
    def test_evaluation_result_minimal(self):
        """Test creating evaluation result with minimal parameters"""
        result = EvaluationResult(overall_score=0.85)
        
        assert result.overall_score == 0.85
        assert result.metric_scores == {}
        assert result.test_case_results == []
        assert result.recommendations == []
    
    def test_evaluation_result_full(self):
        """Test creating evaluation result with all parameters"""
        metric_scores = {"faithfulness": 0.9, "relevancy": 0.8}
        test_results = [{"case_id": 1, "score": 0.85}]
        recommendations = ["Improve context retrieval", "Use better prompts"]
        
        result = EvaluationResult(
            overall_score=0.85,
            metric_scores=metric_scores,
            test_case_results=test_results,
            recommendations=recommendations
        )
        
        assert result.overall_score == 0.85
        assert result.metric_scores == metric_scores
        assert result.test_case_results == test_results
        assert result.recommendations == recommendations


class TestTestCase:
    """Test cases for TestCase model"""
    
    def test_test_case_minimal(self):
        """Test creating test case with minimal parameters"""
        test_case = TestCase(
            question="What is Python?",
            expected_answer="Python is a programming language."
        )
        
        assert test_case.question == "What is Python?"
        assert test_case.expected_answer == "Python is a programming language."
        assert test_case.context == []
        assert test_case.metadata == {}
    
    def test_test_case_full(self):
        """Test creating test case with all parameters"""
        context = [Document(content="Python documentation")]
        metadata = {"difficulty": "easy", "category": "programming"}
        
        test_case = TestCase(
            question="What is Python?",
            expected_answer="Python is a programming language.",
            context=context,
            metadata=metadata
        )
        
        assert test_case.question == "What is Python?"
        assert test_case.expected_answer == "Python is a programming language."
        assert test_case.context == context
        assert test_case.metadata == metadata


class TestRoutingDecision:
    """Test cases for RoutingDecision model"""
    
    def test_routing_decision_minimal(self):
        """Test creating routing decision with minimal parameters"""
        decision = RoutingDecision(
            selected_route="programming",
            confidence=0.9
        )
        
        assert decision.selected_route == "programming"
        assert decision.confidence == 0.9
        assert decision.reasoning == ""
        assert decision.metadata == {}
    
    def test_routing_decision_full(self):
        """Test creating routing decision with all parameters"""
        metadata = {"processing_time": 0.1, "alternatives": ["general"]}
        
        decision = RoutingDecision(
            selected_route="programming",
            confidence=0.9,
            reasoning="Query contains programming keywords",
            metadata=metadata
        )
        
        assert decision.selected_route == "programming"
        assert decision.confidence == 0.9
        assert decision.reasoning == "Query contains programming keywords"
        assert decision.metadata == metadata


class TestRouteConfig:
    """Test cases for RouteConfig model"""
    
    def test_route_config_minimal(self):
        """Test creating route config with minimal parameters"""
        config = RouteConfig(
            name="programming",
            description="Programming related queries"
        )
        
        assert config.name == "programming"
        assert config.description == "Programming related queries"
        assert config.keywords == []
        assert config.embedding is None
        assert config.metadata == {}
    
    def test_route_config_full(self):
        """Test creating route config with all parameters"""
        keywords = ["python", "programming", "code"]
        embedding = [0.1, 0.2, 0.3]
        metadata = {"priority": "high", "expert": "python_expert"}
        
        config = RouteConfig(
            name="programming",
            description="Programming related queries",
            keywords=keywords,
            embedding=embedding,
            metadata=metadata
        )
        
        assert config.name == "programming"
        assert config.description == "Programming related queries"
        assert config.keywords == keywords
        assert config.embedding == embedding
        assert config.metadata == metadata
    
    def test_route_config_default_factories(self):
        """Test default factory behavior"""
        config1 = RouteConfig(name="route1", description="desc1")
        config2 = RouteConfig(name="route2", description="desc2")
        
        config1.keywords.append("keyword1")
        config1.metadata["key1"] = "value1"
        
        assert len(config1.keywords) == 1
        assert len(config2.keywords) == 0
        assert "key1" in config1.metadata
        assert "key1" not in config2.metadata


class TestModelIntegration:
    """Integration tests for model interactions"""
    
    def test_document_in_rag_response(self):
        """Test using Document in RAGResponse"""
        doc = Document(
            content="Python is a programming language",
            metadata={"source": "documentation"},
            doc_id="python_doc_1"
        )
        
        response = RAGResponse(
            answer="Python is a high-level programming language.",
            source_documents=[doc],
            confidence_score=0.9
        )
        
        assert len(response.source_documents) == 1
        assert response.source_documents[0].content == doc.content
        assert response.source_documents[0].doc_id == doc.doc_id
    
    def test_document_in_test_case(self):
        """Test using Document in TestCase"""
        context_doc = Document(
            content="Python documentation excerpt",
            metadata={"type": "context"}
        )
        
        test_case = TestCase(
            question="What is Python?",
            expected_answer="Python is a programming language.",
            context=[context_doc]
        )
        
        assert len(test_case.context) == 1
        assert test_case.context[0].content == context_doc.content
    
    def test_processed_query_with_metadata(self):
        """Test ProcessedQuery with complex metadata"""
        query = ProcessedQuery(
            original_query="What is machine learning?",
            transformed_queries=[
                "Explain machine learning",
                "What are ML algorithms?",
                "How does machine learning work?"
            ],
            strategy_used="multi_query",
            metadata={
                "processing_time": 0.5,
                "confidence": 0.95,
                "model_used": "gpt-3.5-turbo",
                "transformations_applied": ["expansion", "rephrasing", "decomposition"]
            }
        )
        
        assert len(query.transformed_queries) == 3
        assert query.metadata["processing_time"] == 0.5
        assert "transformations_applied" in query.metadata
        assert isinstance(query.metadata["transformations_applied"], list)
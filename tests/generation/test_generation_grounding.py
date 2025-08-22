"""
Unit tests for grounded generation features in GenerationEngine.
Tests requirement 5.4 implementation for grounded responses, citations, and validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.generation.generation_engine import GenerationEngine
from src.rag_engine.generation.llm_providers import GoogleLLMProvider
from src.rag_engine.core.models import Document, RAGResponse
from src.rag_engine.core.config import PipelineConfig


class TestGroundedGeneration:
    """Test grounded generation functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_provider="google",
            llm_model="gemini-pro",
            temperature=0.0,
            max_tokens=1000
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Python is a high-level programming language created by Guido van Rossum.",
                metadata={"source": "python_docs", "title": "Python Introduction", "doc_id": "doc1"},
                doc_id="doc1"
            ),
            Document(
                content="Python was first released in 1991 and is known for its simplicity.",
                metadata={"source": "python_history", "title": "Python History", "doc_id": "doc2"},
                doc_id="doc2"
            ),
            Document(
                content="Machine learning libraries like TensorFlow and PyTorch are popular in Python.",
                metadata={"source": "ml_guide", "title": "ML with Python", "doc_id": "doc3"},
                doc_id="doc3"
            )
        ]
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing"""
        mock_response = Mock()
        mock_response.content = "Python is a programming language created by Guido van Rossum in 1991."
        return mock_response
    
    @pytest.fixture
    def generation_engine(self, config):
        """Create GenerationEngine instance with mocked LLM"""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
            engine = GenerationEngine(config)
            engine.llm_provider.llm = mock_llm.return_value
            return engine
    
    def test_generate_grounded_with_context(self, generation_engine, sample_documents, mock_llm_response):
        """Test grounded generation with valid context"""
        # Mock the LLM response
        generation_engine.llm_provider.llm.invoke.return_value = mock_llm_response
        
        query = "What is Python?"
        result = generation_engine.generate_grounded(query, sample_documents)
        
        assert result == "Python is a programming language created by Guido van Rossum in 1991."
        generation_engine.llm_provider.llm.invoke.assert_called_once()
        
        # Verify the prompt contains grounding instructions
        call_args = generation_engine.llm_provider.llm.invoke.call_args[0][0]
        assert "Use ONLY the following pieces of retrieved context" in call_args
        assert "Base your answer ONLY on the information provided" in call_args
    
    def test_generate_grounded_without_context(self, generation_engine):
        """Test grounded generation without context"""
        query = "What is Python?"
        result = generation_engine.generate_grounded(query, [])
        
        assert result == "I don't have enough information in the provided context to answer this question."
        generation_engine.llm_provider.llm.invoke.assert_not_called()
    
    def test_generate_grounded_with_error(self, generation_engine, sample_documents):
        """Test grounded generation with LLM error"""
        generation_engine.llm_provider.llm.invoke.side_effect = Exception("LLM error")
        
        query = "What is Python?"
        result = generation_engine.generate_grounded(query, sample_documents)
        
        assert "Generation temporarily unavailable" in result
    
    def test_generate_with_citations(self, generation_engine, sample_documents):
        """Test citation generation"""
        # Mock LLM response with citations
        mock_response = Mock()
        mock_response.content = "Python is a programming language [1] created by Guido van Rossum [2]."
        generation_engine.llm_provider.llm.invoke.return_value = mock_response
        
        query = "What is Python?"
        response, source_metadata = generation_engine.generate_with_citations(query, sample_documents)
        
        assert "[1]" in response and "[2]" in response
        assert len(source_metadata) == 3
        assert source_metadata[0]["citation_number"] == 1
        assert source_metadata[0]["doc_id"] == "doc1"
        assert source_metadata[1]["source"] == "python_history"
        
        # Verify the prompt contains citation instructions
        call_args = generation_engine.llm_provider.llm.invoke.call_args[0][0]
        assert "Include citations as [1], [2], etc." in call_args
        assert "[1] Python is a high-level programming language" in call_args
    
    def test_generate_with_citations_no_context(self, generation_engine):
        """Test citation generation without context"""
        query = "What is Python?"
        response, source_metadata = generation_engine.generate_with_citations(query, [])
        
        assert response == "I don't have enough information to answer this question."
        assert source_metadata == []
    
    def test_create_numbered_context(self, generation_engine, sample_documents):
        """Test numbered context creation for citations"""
        numbered_context, source_metadata = generation_engine._create_numbered_context(sample_documents)
        
        assert "[1] Python is a high-level programming language" in numbered_context
        assert "[2] Python was first released in 1991" in numbered_context
        assert "[3] Machine learning libraries like TensorFlow" in numbered_context
        
        assert len(source_metadata) == 3
        assert source_metadata[0]["citation_number"] == 1
        assert source_metadata[1]["title"] == "Python History"
        assert source_metadata[2]["source"] == "ml_guide"
    
    def test_validate_response_grounding_valid(self, generation_engine, sample_documents):
        """Test response validation for grounded response"""
        # Mock validation response
        mock_response = Mock()
        mock_response.content = """GROUNDED: YES
CONFIDENCE: 0.9
ISSUES: None
EXPLANATION: The answer is fully supported by the provided context."""
        generation_engine.llm_provider.llm.invoke.return_value = mock_response
        
        query = "What is Python?"
        response = "Python is a programming language created by Guido van Rossum."
        validation_results = generation_engine.validate_response_grounding(query, sample_documents, response)
        
        assert validation_results["grounded"] == "YES"
        assert validation_results["confidence"] == 0.9
        assert validation_results["issues"] == []
        assert "fully supported" in validation_results["explanation"]
    
    def test_validate_response_grounding_invalid(self, generation_engine, sample_documents):
        """Test response validation for ungrounded response"""
        # Mock validation response
        mock_response = Mock()
        mock_response.content = """GROUNDED: NO
CONFIDENCE: 0.8
ISSUES: Contains information not in context
EXPLANATION: The answer includes claims not supported by the provided sources."""
        generation_engine.llm_provider.llm.invoke.return_value = mock_response
        
        query = "What is Python?"
        response = "Python is a programming language that was invented in 2000."
        validation_results = generation_engine.validate_response_grounding(query, sample_documents, response)
        
        assert validation_results["grounded"] == "NO"
        assert validation_results["confidence"] == 0.8
        assert "Contains information not in context" in validation_results["issues"]
    
    def test_validate_response_grounding_empty_response(self, generation_engine, sample_documents):
        """Test validation with empty response"""
        validation_results = generation_engine.validate_response_grounding("What is Python?", sample_documents, "")
        
        assert validation_results["grounded"] == "NO"
        assert validation_results["confidence"] == 0.0
        assert "Empty response" in validation_results["issues"]
    
    def test_validate_response_grounding_no_context(self, generation_engine):
        """Test validation without context"""
        validation_results = generation_engine.validate_response_grounding(
            "What is Python?", [], "Python is a programming language."
        )
        
        assert validation_results["grounded"] == "NO"
        assert validation_results["confidence"] == 0.0
        assert "No context provided" in validation_results["issues"]
    
    def test_parse_validation_response(self, generation_engine):
        """Test parsing of validation response"""
        validation_text = """GROUNDED: PARTIALLY
CONFIDENCE: 0.7
ISSUES: Some unsupported claims, Missing citations
EXPLANATION: Most information is correct but some details are not verified."""
        
        results = generation_engine._parse_validation_response(validation_text)
        
        assert results["grounded"] == "PARTIALLY"
        assert results["confidence"] == 0.7
        assert len(results["issues"]) == 2
        assert "Some unsupported claims" in results["issues"]
        assert "Missing citations" in results["issues"]
        assert "Most information is correct" in results["explanation"]
    
    def test_generate_with_full_grounding_success(self, generation_engine, sample_documents):
        """Test full grounding generation with all features"""
        # Mock citation response
        citation_response = Mock()
        citation_response.content = "Python is a programming language [1] created by Guido van Rossum [2]."
        
        # Mock validation response
        validation_response = Mock()
        validation_response.content = """GROUNDED: YES
CONFIDENCE: 0.9
ISSUES: None
EXPLANATION: Fully grounded response."""
        
        generation_engine.llm_provider.llm.invoke.side_effect = [citation_response, validation_response]
        
        query = "What is Python?"
        result = generation_engine.generate_with_full_grounding(
            query, sample_documents, include_citations=True, validate_grounding=True
        )
        
        assert isinstance(result, RAGResponse)
        assert "[1]" in result.answer and "[2]" in result.answer
        assert result.confidence_score >= 0.8
        assert result.metadata["grounding_applied"] is True
        assert result.metadata["citations_included"] is True
        assert result.metadata["validation_applied"] is True
        assert result.metadata["source_count"] == 3
        assert "source_metadata" in result.metadata
        assert "validation_results" in result.metadata
    
    def test_generate_with_full_grounding_no_context(self, generation_engine):
        """Test full grounding generation without context"""
        result = generation_engine.generate_with_full_grounding("What is Python?", [])
        
        assert isinstance(result, RAGResponse)
        assert "I don't have enough information" in result.answer
        assert result.confidence_score == 0.0
        assert result.metadata["grounding_applied"] is True
        assert result.metadata["citations_included"] is False
        assert result.metadata["validation_applied"] is False
        assert "error" in result.metadata
    
    def test_generate_with_full_grounding_partial_validation(self, generation_engine, sample_documents):
        """Test full grounding with partially grounded response"""
        # Mock citation response
        citation_response = Mock()
        citation_response.content = "Python is a programming language [1] with some additional claims."
        
        # Mock validation response indicating partial grounding
        validation_response = Mock()
        validation_response.content = """GROUNDED: PARTIALLY
CONFIDENCE: 0.6
ISSUES: Some unsupported claims
EXPLANATION: Mostly grounded but contains some unverified information."""
        
        generation_engine.llm_provider.llm.invoke.side_effect = [citation_response, validation_response]
        
        query = "What is Python?"
        result = generation_engine.generate_with_full_grounding(query, sample_documents)
        
        # Confidence should be reduced for partially grounded responses
        assert result.confidence_score < 0.7
        assert result.metadata["validation_results"]["grounded"] == "PARTIALLY"
    
    def test_generate_with_full_grounding_error_handling(self, generation_engine, sample_documents):
        """Test error handling in full grounding generation"""
        generation_engine.llm_provider.llm.invoke.side_effect = Exception("LLM error")
        
        query = "What is Python?"
        result = generation_engine.generate_with_full_grounding(query, sample_documents)
        
        assert isinstance(result, RAGResponse)
        # The error is handled by the citation generation method, not the main handler
        assert "Generation temporarily unavailable" in result.answer
        assert result.confidence_score == 0.0  # No confidence due to error
    
    def test_extract_citations_from_response(self, generation_engine):
        """Test citation extraction from response text"""
        response_text = "Python [1] is a language [2] used for AI [3] and web development [1]."
        citations = generation_engine.extract_citations_from_response(response_text)
        
        assert citations == [1, 2, 3, 1]
    
    def test_extract_citations_no_citations(self, generation_engine):
        """Test citation extraction from response without citations"""
        response_text = "Python is a programming language used for many applications."
        citations = generation_engine.extract_citations_from_response(response_text)
        
        assert citations == []


class TestGoogleLLMProvider:
    """Test Google LLM provider functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_provider="google",
            llm_model="gemini-2.0-flash-lite",  # Should be mapped to gemini-2.0-flash-lite
            temperature=0.5,
            max_tokens=500
        )
    
    def test_google_llm_provider_initialization(self, config):
        """Test Google LLM provider initialization"""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
            provider = GoogleLLMProvider(config)
            
            # Verify model mapping
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["model"] == "gemini-2.0-flash-lite"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500
            assert provider.config == config
    
    def test_google_llm_provider_generate(self, config):
        """Test text generation with Google LLM provider"""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
            mock_response = Mock()
            mock_response.content = "Generated response"
            mock_llm.return_value.invoke.return_value = mock_response
            
            provider = GoogleLLMProvider(config)
            result = provider.generate("Test prompt")
            
            assert result == "Generated response"
            mock_llm.return_value.invoke.assert_called_once_with("Test prompt")
    
    def test_google_llm_provider_generate_error(self, config):
        """Test error handling in Google LLM provider"""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
            mock_llm.return_value.invoke.side_effect = Exception("API error")
            
            provider = GoogleLLMProvider(config)
            
            # Should return fallback response when generation fails
            result = provider.generate("Test prompt")
            assert "Generation temporarily unavailable" in result
    
    def test_google_llm_provider_model_info(self, config):
        """Test getting model information"""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
            mock_llm.return_value.model = "gemini-2.0-flash-lite"
            
            provider = GoogleLLMProvider(config)
            info = provider.get_model_info()
            
            assert info["provider"] == "google"
            assert info["model"] == "gemini-2.0-flash-lite"
            assert info["temperature"] == 0.5
            assert info["max_tokens"] == 500


class TestPromptTemplates:
    """Test prompt template functionality"""
    
    @pytest.fixture
    def generation_engine(self):
        """Create GenerationEngine instance for testing prompts"""
        config = PipelineConfig()
        with patch('langchain_google_genai.ChatGoogleGenerativeAI'):
            return GenerationEngine(config)
    
    def test_grounded_prompt_template(self, generation_engine):
        """Test grounded prompt template"""
        prompt = generation_engine.grounded_prompt_template
        formatted = prompt.format(
            question="What is Python?",
            context="Python is a programming language."
        )
        
        assert "Use ONLY the following pieces of retrieved context" in formatted
        assert "Base your answer ONLY on the information provided" in formatted
        assert "What is Python?" in formatted
        assert "Python is a programming language." in formatted
    
    def test_citation_prompt_template(self, generation_engine):
        """Test citation prompt template"""
        prompt = generation_engine.citation_prompt_template
        formatted = prompt.format(
            question="What is Python?",
            numbered_context="[1] Python is a programming language.\n[2] Created by Guido van Rossum."
        )
        
        assert "Include citations as [1], [2], etc." in formatted
        assert "[1] Python is a programming language" in formatted
        assert "[2] Created by Guido van Rossum" in formatted
    
    def test_validation_prompt_template(self, generation_engine):
        """Test validation prompt template"""
        prompt = generation_engine.validation_prompt_template
        formatted = prompt.format(
            question="What is Python?",
            context="Python is a programming language.",
            answer="Python is a language for coding."
        )
        
        assert "You are a validator checking if an answer is properly grounded" in formatted
        assert "GROUNDED: [YES/NO/PARTIALLY]" in formatted
        assert "CONFIDENCE: [0.0-1.0]" in formatted
        assert "What is Python?" in formatted
        assert "Python is a programming language." in formatted
        assert "Python is a language for coding." in formatted


if __name__ == "__main__":
    pytest.main([__file__])
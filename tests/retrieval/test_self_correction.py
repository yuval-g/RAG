"""
Unit tests for self-correction mechanisms (CRAG and Self-RAG)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.retrieval.self_correction import (
    CRAGRelevanceChecker,
    SelfRAGValidator,
    SelfCorrectionEngine,
    RelevanceGrade,
    FactualityGrade,
    RelevanceAssessment,
    FactualityAssessment
)
from src.rag_engine.core.models import Document
from src.rag_engine.core.config import PipelineConfig


class TestCRAGRelevanceChecker:
    """Test CRAG relevance checking functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_model="gemini-pro",
            temperature=0.0,
            enable_self_correction=True
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for relevance assessment"""
        return """Grade: relevant
Confidence: 0.85
Reasoning: The document contains keywords related to machine learning and neural networks."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                content="Machine learning is a subset of artificial intelligence that focuses on neural networks.",
                metadata={"source": "ml_guide.pdf"}
            ),
            Document(
                content="The weather today is sunny with a temperature of 75 degrees.",
                metadata={"source": "weather_report.txt"}
            ),
            Document(
                content="Deep learning uses multiple layers in neural networks to learn complex patterns.",
                metadata={"source": "deep_learning.pdf"}
            )
        ]
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_init(self, mock_llm_class, config):
        """Test CRAGRelevanceChecker initialization"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        checker = CRAGRelevanceChecker(config)
        
        assert checker.config == config
        assert checker.llm == mock_llm
        mock_llm_class.assert_called_once_with(
            model="gemini-pro",
            temperature=0.0
        )
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_assess_relevance_relevant(self, mock_llm_class, config, mock_llm_response, sample_documents):
        """Test relevance assessment for relevant document"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock the chain invoke
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_llm_response
        
        checker = CRAGRelevanceChecker(config)
        checker.grading_chain = mock_chain
        
        query = "What is machine learning?"
        document = sample_documents[0]
        
        assessment = checker.assess_relevance(query, document)
        
        assert assessment.grade == RelevanceGrade.RELEVANT
        assert assessment.confidence == 0.85
        assert "machine learning" in assessment.reasoning.lower()
        
        mock_chain.invoke.assert_called_once_with({
            "question": query,
            "document": document.content
        })
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_assess_relevance_irrelevant(self, mock_llm_class, config, sample_documents):
        """Test relevance assessment for irrelevant document"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        irrelevant_response = """Grade: irrelevant
Confidence: 0.95
Reasoning: The document is about weather, not machine learning."""
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = irrelevant_response
        
        checker = CRAGRelevanceChecker(config)
        checker.grading_chain = mock_chain
        
        query = "What is machine learning?"
        document = sample_documents[1]  # Weather document
        
        assessment = checker.assess_relevance(query, document)
        
        assert assessment.grade == RelevanceGrade.IRRELEVANT
        assert assessment.confidence == 0.95
        assert "weather" in assessment.reasoning.lower()
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_parse_relevance_response_malformed(self, mock_llm_class, config):
        """Test parsing of malformed LLM response"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        checker = CRAGRelevanceChecker(config)
        
        # Test with malformed response
        malformed_response = "This is not a properly formatted response"
        assessment = checker._parse_relevance_response(malformed_response)
        
        # Should return default values
        assert assessment.grade == RelevanceGrade.AMBIGUOUS
        assert assessment.confidence == 0.5
        assert assessment.reasoning == ""
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_filter_relevant_documents(self, mock_llm_class, config, sample_documents):
        """Test filtering of documents based on relevance"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock assessments for each document
        assessments = [
            "Grade: relevant\nConfidence: 0.9\nReasoning: About ML",
            "Grade: irrelevant\nConfidence: 0.8\nReasoning: About weather",
            "Grade: relevant\nConfidence: 0.85\nReasoning: About deep learning"
        ]
        
        mock_chain = Mock()
        mock_chain.invoke.side_effect = assessments
        
        checker = CRAGRelevanceChecker(config)
        checker.grading_chain = mock_chain
        
        query = "What is machine learning?"
        relevant_docs, all_assessments = checker.filter_relevant_documents(
            query, sample_documents, min_confidence=0.7
        )
        
        # Should return 2 relevant documents (ML and deep learning)
        assert len(relevant_docs) == 2
        assert len(all_assessments) == 3
        
        # Check that the correct documents were kept
        assert relevant_docs[0].content == sample_documents[0].content
        assert relevant_docs[1].content == sample_documents[2].content
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_assess_relevance_error_handling(self, mock_llm_class, config, sample_documents):
        """Test error handling in relevance assessment"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock chain that raises an exception
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        
        checker = CRAGRelevanceChecker(config)
        checker.grading_chain = mock_chain
        
        query = "What is machine learning?"
        document = sample_documents[0]
        
        assessment = checker.assess_relevance(query, document)
        
        # Should return default assessment on error
        assert assessment.grade == RelevanceGrade.AMBIGUOUS
        assert assessment.confidence == 0.5
        assert "Error occurred" in assessment.reasoning


class TestSelfRAGValidator:
    """Test Self-RAG validation functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_model="gemini-pro",
            temperature=0.0,
            enable_self_correction=True
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context documents"""
        return [
            Document(
                content="Python is a high-level programming language known for its simplicity and readability.",
                metadata={"source": "python_guide.pdf"}
            ),
            Document(
                content="Machine learning libraries in Python include scikit-learn, TensorFlow, and PyTorch.",
                metadata={"source": "ml_libraries.pdf"}
            )
        ]
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_init(self, mock_llm_class, config):
        """Test SelfRAGValidator initialization"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        validator = SelfRAGValidator(config)
        
        assert validator.config == config
        assert validator.llm == mock_llm
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_validate_response_grounded(self, mock_llm_class, config, sample_context):
        """Test validation of grounded response"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        grounded_response = """Grade: grounded
Confidence: 0.9
Citations: true
Reasoning: The answer is fully supported by the provided context about Python."""
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = grounded_response
        
        validator = SelfRAGValidator(config)
        validator.validation_chain = mock_chain
        
        query = "What is Python?"
        response = "Python is a high-level programming language known for its simplicity."
        
        assessment = validator.validate_response(query, sample_context, response)
        
        assert assessment.grade == FactualityGrade.GROUNDED
        assert assessment.confidence == 0.9
        assert assessment.citations_found is True
        assert "supported" in assessment.reasoning.lower()
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_validate_response_not_grounded(self, mock_llm_class, config, sample_context):
        """Test validation of not grounded response"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        not_grounded_response = """Grade: not_grounded
Confidence: 0.85
Citations: false
Reasoning: The answer contains claims not supported by the context."""
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = not_grounded_response
        
        validator = SelfRAGValidator(config)
        validator.validation_chain = mock_chain
        
        query = "What is Python?"
        response = "Python was invented by aliens and is used for time travel."
        
        assessment = validator.validate_response(query, sample_context, response)
        
        assert assessment.grade == FactualityGrade.NOT_GROUNDED
        assert assessment.confidence == 0.85
        assert assessment.citations_found is False
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_parse_factuality_response_malformed(self, mock_llm_class, config):
        """Test parsing of malformed validation response"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        validator = SelfRAGValidator(config)
        
        malformed_response = "This is not properly formatted"
        assessment = validator._parse_factuality_response(malformed_response)
        
        # Should return default values
        assert assessment.grade == FactualityGrade.PARTIALLY_GROUNDED
        assert assessment.confidence == 0.5
        assert assessment.citations_found is False
        assert assessment.reasoning == ""
    
    @patch('src.rag_engine.retrieval.self_correction.ChatGoogleGenerativeAI')
    def test_validate_response_error_handling(self, mock_llm_class, config, sample_context):
        """Test error handling in response validation"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Validation error")
        
        validator = SelfRAGValidator(config)
        validator.validation_chain = mock_chain
        
        query = "What is Python?"
        response = "Python is a programming language."
        
        assessment = validator.validate_response(query, sample_context, response)
        
        # Should return default assessment on error
        assert assessment.grade == FactualityGrade.PARTIALLY_GROUNDED
        assert assessment.confidence == 0.5
        assert "Error occurred" in assessment.reasoning


class TestSelfCorrectionEngine:
    """Test the main self-correction engine"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(
            llm_model="gemini-pro",
            temperature=0.0,
            enable_self_correction=True,
            relevance_threshold=0.7,
            factuality_threshold=0.7,
            min_relevant_docs=2
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents"""
        return [
            Document(
                content="Machine learning is a subset of AI.",
                metadata={"source": "ml.pdf"}
            ),
            Document(
                content="The weather is sunny today.",
                metadata={"source": "weather.txt"}
            ),
            Document(
                content="Deep learning uses neural networks.",
                metadata={"source": "dl.pdf"}
            )
        ]
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_init(self, mock_crag_class, mock_selfrag_class, config):
        """Test SelfCorrectionEngine initialization"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        engine = SelfCorrectionEngine(config)
        
        assert engine.config == config
        assert engine.relevance_checker == mock_crag
        assert engine.response_validator == mock_selfrag
        assert engine.relevance_threshold == 0.7
        assert engine.factuality_threshold == 0.7
        assert engine.min_relevant_docs == 2
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_correct_retrieval_success(self, mock_crag_class, mock_selfrag_class, config, sample_documents):
        """Test successful retrieval correction"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        # Mock relevance checker to return 2 relevant documents
        relevant_docs = [sample_documents[0], sample_documents[2]]
        mock_assessments = [
            RelevanceAssessment(grade=RelevanceGrade.RELEVANT, confidence=0.9, reasoning="ML related"),
            RelevanceAssessment(grade=RelevanceGrade.IRRELEVANT, confidence=0.8, reasoning="Weather related"),
            RelevanceAssessment(grade=RelevanceGrade.RELEVANT, confidence=0.85, reasoning="DL related")
        ]
        
        mock_crag.filter_relevant_documents.return_value = (relevant_docs, mock_assessments)
        
        engine = SelfCorrectionEngine(config)
        
        query = "What is machine learning?"
        corrected_docs, metadata = engine.correct_retrieval(query, sample_documents)
        
        assert len(corrected_docs) == 2
        assert corrected_docs == relevant_docs
        assert metadata["original_count"] == 3
        assert metadata["filtered_count"] == 2
        assert metadata["correction_applied"] is True
        assert metadata["fallback_triggered"] is False
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_correct_retrieval_insufficient_docs(self, mock_crag_class, mock_selfrag_class, config, sample_documents):
        """Test retrieval correction with insufficient relevant documents"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        # Mock relevance checker to return only 1 relevant document (below threshold)
        relevant_docs = [sample_documents[0]]
        mock_assessments = [
            RelevanceAssessment(grade=RelevanceGrade.RELEVANT, confidence=0.9, reasoning="ML related"),
            RelevanceAssessment(grade=RelevanceGrade.IRRELEVANT, confidence=0.8, reasoning="Weather related"),
            RelevanceAssessment(grade=RelevanceGrade.IRRELEVANT, confidence=0.7, reasoning="Not relevant")
        ]
        
        mock_crag.filter_relevant_documents.return_value = (relevant_docs, mock_assessments)
        
        engine = SelfCorrectionEngine(config)
        
        query = "What is machine learning?"
        corrected_docs, metadata = engine.correct_retrieval(query, sample_documents)
        
        # Should return original documents due to fallback
        assert len(corrected_docs) == 3
        assert corrected_docs == sample_documents
        assert metadata["fallback_triggered"] is True
        assert metadata["fallback_reason"] == "insufficient_relevant_docs"
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_validate_generation_success(self, mock_crag_class, mock_selfrag_class, config, sample_documents):
        """Test successful generation validation"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        # Mock validator to return grounded assessment
        mock_assessment = FactualityAssessment(
            grade=FactualityGrade.GROUNDED,
            confidence=0.9,
            reasoning="Response is well grounded",
            citations_found=True
        )
        
        mock_selfrag.validate_response.return_value = mock_assessment
        
        engine = SelfCorrectionEngine(config)
        
        query = "What is machine learning?"
        response = "Machine learning is a subset of AI that uses neural networks."
        
        validated_response, metadata = engine.validate_generation(query, sample_documents, response)
        
        assert validated_response == response  # No correction needed
        assert metadata["grade"] == "grounded"
        assert metadata["confidence"] == 0.9
        assert metadata["correction_needed"] is False
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_validate_generation_needs_correction(self, mock_crag_class, mock_selfrag_class, config, sample_documents):
        """Test generation validation that needs correction"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        # Mock validator to return not grounded assessment
        mock_assessment = FactualityAssessment(
            grade=FactualityGrade.NOT_GROUNDED,
            confidence=0.8,
            reasoning="Response contains unsupported claims",
            citations_found=False
        )
        
        mock_selfrag.validate_response.return_value = mock_assessment
        
        engine = SelfCorrectionEngine(config)
        
        query = "What is machine learning?"
        response = "Machine learning was invented by aliens."
        
        validated_response, metadata = engine.validate_generation(query, sample_documents, response)
        
        assert validated_response != response  # Should be corrected
        assert "cannot fully verify" in validated_response.lower()
        assert metadata["correction_needed"] is True
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_process_rag_pipeline_full(self, mock_crag_class, mock_selfrag_class, config, sample_documents):
        """Test full RAG pipeline with self-correction"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        # Mock retrieval correction
        relevant_docs = [sample_documents[0], sample_documents[2]]
        mock_assessments = [Mock(), Mock(), Mock()]
        mock_crag.filter_relevant_documents.return_value = (relevant_docs, mock_assessments)
        
        # Mock response validation
        mock_assessment = FactualityAssessment(
            grade=FactualityGrade.GROUNDED,
            confidence=0.9,
            reasoning="Well grounded",
            citations_found=True
        )
        mock_selfrag.validate_response.return_value = mock_assessment
        
        engine = SelfCorrectionEngine(config)
        
        query = "What is machine learning?"
        response = "Machine learning is a subset of AI."
        
        corrected_docs, validated_response, metadata = engine.process_rag_pipeline(
            query, sample_documents, response
        )
        
        assert len(corrected_docs) == 2
        assert validated_response == response
        assert metadata["self_correction_applied"] is True
        assert "retrieval_correction" in metadata
        assert "response_validation" in metadata
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_get_correction_stats(self, mock_crag_class, mock_selfrag_class, config):
        """Test getting correction statistics"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        engine = SelfCorrectionEngine(config)
        
        stats = engine.get_correction_stats()
        
        assert stats["relevance_threshold"] == 0.7
        assert stats["factuality_threshold"] == 0.7
        assert stats["min_relevant_docs"] == 2
        assert "components" in stats
    
    @patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator')
    @patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker')
    def test_update_thresholds(self, mock_crag_class, mock_selfrag_class, config):
        """Test updating correction thresholds"""
        mock_crag = Mock()
        mock_selfrag = Mock()
        mock_crag_class.return_value = mock_crag
        mock_selfrag_class.return_value = mock_selfrag
        
        engine = SelfCorrectionEngine(config)
        
        # Update thresholds
        engine.update_thresholds(
            relevance_threshold=0.8,
            factuality_threshold=0.75,
            min_relevant_docs=3
        )
        
        assert engine.relevance_threshold == 0.8
        assert engine.factuality_threshold == 0.75
        assert engine.min_relevant_docs == 3
    
    def test_empty_documents_handling(self, config):
        """Test handling of empty document list"""
        with patch('src.rag_engine.retrieval.self_correction.CRAGRelevanceChecker'), \
             patch('src.rag_engine.retrieval.self_correction.SelfRAGValidator'):
            
            engine = SelfCorrectionEngine(config)
            
            query = "What is machine learning?"
            corrected_docs, metadata = engine.correct_retrieval(query, [])
            
            assert len(corrected_docs) == 0
            assert metadata["correction_applied"] is False
            assert metadata["reason"] == "no_documents"


if __name__ == "__main__":
    pytest.main([__file__])